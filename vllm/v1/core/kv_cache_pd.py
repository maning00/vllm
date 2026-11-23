# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request-scoped PD storage retention, independent of attention and hit rules."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from vllm.exceptions import VLLMValidationError
from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_coordinator import replay_boundaries
from vllm.v1.core.single_type_kv_cache_manager import (
    FullAttentionManager,
    SlidingWindowManager,
)

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.request import Request


@dataclass(frozen=True)
class ReplayAllocation:
    generation: int
    ranges: tuple[tuple[int, int], ...]
    local_tokens: int


class PDReplayRetention:
    def __init__(self, manager: "KVCacheManager", lookahead: int, producer: bool):
        self.manager, self.lookahead, self.producer = manager, lookahead, producer
        self.managers = manager.coordinator.single_type_managers
        self.alignment = (
            manager.kv_cache_config.cache_hit_alignment_tokens
            or manager.coordinator.scheduler_block_size
        )
        self.allocations: dict[str, ReplayAllocation] = {}
        self._completed: set[str] = set()
        self._generation = 0
        self._steps: dict[int, tuple[object, dict[str, int]]] = {}
        for group in self.managers:
            if type(group) not in (FullAttentionManager, SlidingWindowManager):
                raise ValueError(
                    "PD replay retention supports only full attention and "
                    "plain SlidingWindowManager groups"
                )
            if isinstance(group, SlidingWindowManager) and group.enable_caching:
                params = group.manager_config.params
                assert isinstance(params, SlidingWindowManager.Params)
                required = (
                    1
                    if params.bounded_replay
                    else max(1, lookahead) + self.alignment - 1
                )
                if params.pd_replay_retention_tokens < required:
                    raise ValueError(
                        "PD replay retention requires a matching "
                        "storage planning margin"
                    )

    def signature(self) -> list[Any]:
        policies = []
        for group in self.managers:
            params = group.manager_config.params
            if isinstance(group, SlidingWindowManager):
                assert isinstance(params, SlidingWindowManager.Params)
                policies.append(
                    [
                        group.block_size,
                        group.num_retained_tokens,
                        params.sliding_window,
                        params.extra_retained_tokens,
                        params.bounded_replay,
                    ]
                )
            else:
                policies.append([group.block_size])
        return [self.lookahead, self.alignment, policies]

    def ranges(
        self, prompt: int, computed: int, *, local_tokens: int = 0
    ) -> tuple[tuple[int, int], ...]:
        result = []
        boundaries = replay_boundaries(prompt, self.lookahead)
        for group in self.managers:
            size = group.block_size
            end = cdiv(computed, size)
            start = group.get_num_skipped_tokens(max(0, computed - 1)) // size
            if isinstance(group, SlidingWindowManager) and group.prefix_cacheable(
                group.manager_config
            ):
                params = group.manager_config.params
                assert isinstance(params, SlidingWindowManager.Params)
                first_boundary = min(boundaries) // self.alignment * self.alignment
                first = max(
                    0, first_boundary // size - cdiv(params.sliding_window - 1, size)
                )
                mask = group.reachable_block_mask(
                    first,
                    end,
                    self.alignment,
                    size,
                    group.manager_config,
                    retention_interval=0,
                    reachable_boundaries=boundaries,
                )
                if mask is None:
                    start = min(start, first)
                else:
                    start = min(
                        start,
                        next((first + i for i, keep in enumerate(mask) if keep), start),
                    )
                # A local hit only backs its attention tail, not re-prefill storage.
                start = max(
                    start,
                    local_tokens // size - cdiv(params.sliding_window - 1, size),
                )
            result.append((min(start, end), end))
        return tuple(result)

    def source(self, request: "Request") -> dict[str, Any]:
        params = request.kv_transfer_params or {}
        source = params.get("pd_replay")
        if not isinstance(source, dict) or source.get("version") != 1:
            raise VLLMValidationError("Enable pd_replay_retention on both PD instances")
        if source.get("signature") != self.signature():
            raise VLLMValidationError(
                "PD replay geometry or speculative settings differ between instances"
            )
        prompt, computed = source.get("prompt_tokens"), source.get("computed_tokens")
        if (
            not isinstance(prompt, int)
            or not isinstance(computed, int)
            or not 0 <= computed <= prompt == request.num_prompt_tokens
        ):
            raise VLLMValidationError(
                "Invalid PD replay prompt or computed-token range"
            )
        local = source.get("local_tokens", 0)
        max_hit = max(0, prompt - max(1, self.lookahead))
        max_hit = max_hit // self.alignment * self.alignment
        if not isinstance(local, int) or not 0 <= local <= min(computed, max_hit):
            raise VLLMValidationError("Invalid PD replay producer local-hit length")
        ranges = source.get("ranges")
        if ranges != [
            list(r) for r in self.ranges(prompt, computed, local_tokens=local)
        ]:
            raise VLLMValidationError("Invalid PD replay absolute block ranges")
        blocks = params.get("remote_block_ids")
        if (
            not isinstance(blocks, (list, tuple))
            or len(blocks) != len(self.managers)
            or any(
                not isinstance(ids, (list, tuple))
                or len(ids) != end - start
                or any(not isinstance(b, int) or b <= 0 for b in ids)
                for ids, (start, end) in zip(blocks, ranges, strict=True)
            )
        ):
            raise VLLMValidationError(
                "PD replay source contains null or missing blocks"
            )
        return source

    def validate_request(self, request: "Request") -> None:
        params = request.kv_transfer_params or {}
        if self.producer and params.get("do_remote_decode"):
            if request.max_tokens != 1:
                raise VLLMValidationError(
                    "PD replay producer requests require max_tokens=1"
                )
        elif not self.producer and params.get("do_remote_prefill"):
            self.source(request)

    def prepare(
        self, request: "Request", external_tokens: int, *, local_tokens: int = 0
    ) -> ReplayAllocation | None:
        params = request.kv_transfer_params or {}
        if request.request_id in self._completed:
            return None
        active = (
            params.get("do_remote_decode")
            if self.producer
            else (params.get("do_remote_prefill") and external_tokens > 0)
        )
        if not active:
            return self.allocations.get(request.request_id)
        self.validate_request(request)
        if request.request_id in self.allocations:
            return self.allocations[request.request_id]
        if self.producer:
            ranges = self.ranges(
                request.num_prompt_tokens,
                request.num_prompt_tokens,
                local_tokens=local_tokens,
            )
        else:
            source = self.source(request)
            ranges = tuple((r[0], r[1]) for r in source["ranges"])
            backed = self.ranges(
                request.num_prompt_tokens,
                source["computed_tokens"],
                local_tokens=local_tokens,
            )
            ranges = tuple(
                (max(start, local_start), end)
                for (start, end), (local_start, _) in zip(ranges, backed, strict=True)
            )
        self._generation += 1
        allocation = ReplayAllocation(self._generation, ranges, local_tokens)
        self.allocations[request.request_id] = allocation
        for group, (start, _) in zip(self.managers, ranges, strict=True):
            group.pd_replay_starts[request.request_id] = start
        return allocation

    def release(self, request_id: str, generation: int | None = None) -> None:
        current = self.allocations.get(request_id)
        if current is None or (
            generation is not None and generation != current.generation
        ):
            return
        self.allocations.pop(request_id)
        self._completed.add(request_id)
        for group in self.managers:
            group.pd_replay_starts.pop(request_id, None)

    def forget(self, request_id: str) -> None:
        """Discard an allocation lifetime on free or preemption."""
        self.release(request_id)
        self._completed.discard(request_id)

    def track_step(self, step: object, request_ids: Sequence[str]) -> None:
        generations = {
            rid: self.allocations[rid].generation
            for rid in request_ids
            if rid in self.allocations
        }
        if generations:
            self._steps[id(step)] = (step, generations)

    def complete_step(self, step: object) -> None:
        """Lift retention after core has reconciled this step's load errors."""
        entry = self._steps.pop(id(step), None)
        if entry is not None:
            assert entry[0] is step
            for rid, generation in entry[1].items():
                # Failure only lifts storage retention; core owns invalidation/free.
                self.release(rid, generation)

    def export(
        self, request: "Request", blocks: Sequence[Sequence[int]]
    ) -> tuple[list[list[int]], dict[str, Any]]:
        computed = min(request.num_computed_tokens, request.num_prompt_tokens)
        local = self.allocations[request.request_id].local_tokens
        ranges = self.ranges(request.num_prompt_tokens, computed, local_tokens=local)
        selected = []
        for group, (start, end) in zip(blocks, ranges, strict=True):
            chosen = list(group[start:end])
            if len(chosen) != end - start or any(b <= 0 for b in chosen):
                raise VLLMValidationError(
                    "PD replay export requires real blocks throughout each range"
                )
            selected.append(chosen)
        return selected, dict(
            version=1,
            prompt_tokens=request.num_prompt_tokens,
            computed_tokens=computed,
            local_tokens=local,
            signature=self.signature(),
            ranges=[list(r) for r in ranges],
        )

    def pair(
        self, request: "Request", local: Sequence[Sequence[int]], external: int
    ) -> tuple[list[list[int]], list[list[int]]]:
        source = self.source(request)
        remote = (request.kv_transfer_params or {})["remote_block_ids"]
        local_tokens = self.allocations[request.request_id].local_tokens
        load_end = local_tokens + external
        if load_end > source["computed_tokens"]:
            raise ValueError("PD replay read exceeds the producer's computed tokens")
        # Local cache hits are measured before num_computed_tokens is updated.
        dst, src = [], []
        for group, local_ids, remote_ids, (start, end), (backed_start, _) in zip(
            self.managers,
            local,
            remote,
            source["ranges"],
            self.allocations[request.request_id].ranges,
            strict=True,
        ):
            read_end = min(end, cdiv(load_end, group.block_size))
            read_start = max(backed_start, local_tokens // group.block_size)
            if read_end > len(local_ids) or any(
                b <= 0 for b in local_ids[backed_start:read_end]
            ):
                raise RuntimeError(
                    "PD replay destination contains null or missing blocks"
                )
            dst.append(list(local_ids[read_start:read_end]))
            src.append(list(remote_ids[read_start - start : read_end - start]))
        return dst, src
