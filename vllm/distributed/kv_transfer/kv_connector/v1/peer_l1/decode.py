# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Release-batch ordering and bounded retirement of redundant decode rows."""

from collections.abc import Callable
from dataclasses import dataclass
from itertools import islice
from time import perf_counter_ns

import numpy as np

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import KVCacheBlock

from .session import peer_key
from .stats import PeerL1Stats


@dataclass(slots=True)
class _Idle:
    key: bytes
    generation: int
    redundant: bool
    hint: bytes


class DecodeCacheOrder:
    def __init__(
        self,
        pool: BlockPool,
        query: Callable[[list[bytes]], np.ndarray | None],
        stats: PeerL1Stats,
        scan_blocks: int,
    ):
        self.pool, self.query, self.stats = pool, query, stats
        self.scan_blocks = scan_blocks
        self._idle: dict[int, _Idle] = {}
        self._cursor: tuple[KVCacheBlock, bytes | None, int] | None = None
        self._known_no_copy = 0
        self._calls = self._rows = self._ns = self._max_ns = 0

    def forget(self, block: KVCacheBlock) -> None:
        old = self._idle.pop(block.block_id, None)
        if old is not None and not old.redundant:
            self._known_no_copy -= 1

    def classify(self, blocks: list[KVCacheBlock]) -> list[bool]:
        started = perf_counter_ns()
        keys, hint_keys = [], []
        for block in blocks:
            assert block.block_hash is not None
            keys.append(block.block_hash)
            previous = self._idle.get(block.block_id)
            hint_keys.append(
                previous.hint
                if previous is not None and previous.key == block.block_hash
                else peer_key(block.block_hash)
            )
        hints = self.query(hint_keys)
        if hints is None:
            self.stats.inc("decode_classify_contended")
            flags = [False] * len(blocks)
        else:
            flags = [bool(value) for value in hints]
        for block, key, redundant, hint in zip(
            blocks, keys, flags, hint_keys, strict=True
        ):
            assert key is not None
            self.forget(block)
            self._idle[block.block_id] = _Idle(
                key, block.free_queue_generation, redundant, hint
            )
            self._known_no_copy += not redundant
        elapsed = perf_counter_ns() - started
        self._calls += 1
        self._rows += len(blocks)
        self._ns += elapsed
        self._max_ns = max(self._max_ns, elapsed)
        self.stats.inc("decode_classify_calls")
        self.stats.inc("decode_classify_rows", len(blocks))
        self.stats.inc("decode_order_front_rows", sum(flags))
        return flags

    def _stamp(self, block: KVCacheBlock):
        return block, block.block_hash, block.free_queue_generation

    def scan(self) -> None:
        queue = self.pool.free_block_queue
        cursor = self._cursor
        if cursor is not None:
            block, _, _ = cursor
            if (
                block.ref_cnt
                or block.next_free_block is None
                or self._stamp(block) != cursor
            ):
                cursor = None
        previous = cursor[0] if cursor is not None else None
        batch = list(
            islice(
                queue.iter_blocks_after(previous, cached_only=True), self.scan_blocks
            )
        )
        if not batch and previous is not None:
            previous = None
            batch = list(
                islice(
                    queue.iter_blocks_after(None, cached_only=True), self.scan_blocks
                )
            )
        candidates = [
            b
            for b in batch
            if (entry := self._idle.get(b.block_id)) is not None
            and entry.redundant
            and entry.key == b.block_hash
            and entry.generation == b.free_queue_generation
        ]
        hints = (
            self.query([self._idle[b.block_id].hint for b in candidates])
            if candidates
            else []
        )
        self.stats.inc("decode_order_scanned_rows", len(batch))
        if hints is None:
            self.stats.gauge("decode_no_prefill_free_blocks", self._known_no_copy)
            return
        missing = [not v for v in hints]
        retired = {
            b.block_id for b, absent in zip(candidates, missing, strict=True) if absent
        }
        stable = previous
        for block in batch:
            entry = self._idle.get(block.block_id)
            if block.ref_cnt or block.next_free_block is None:
                self.forget(block)
                continue
            if block.block_hash is None or (
                entry is not None
                and (
                    entry.key != block.block_hash
                    or entry.generation != block.free_queue_generation
                )
            ):
                self.forget(block)
                stable = block
                continue
            if (
                block.block_id in retired
                and entry is not None
                and entry.redundant
                and entry.key == block.block_hash
            ):
                queue.remove(block)
                queue.append(block)
                entry.redundant = False
                entry.generation = block.free_queue_generation
                self._known_no_copy += 1
                self.stats.inc("decode_order_retired_rows")
            else:
                stable = block
        self._cursor = None if stable is None else self._stamp(stable)
        self.stats.gauge("decode_no_prefill_free_blocks", self._known_no_copy)

    def finish_step(self) -> None:
        self.scan()
        self.stats.observe("decode_classify_calls_per_step", self._calls)
        self.stats.observe("decode_classify_rows_per_step", self._rows)
        self.stats.observe("decode_classify_seconds_per_step", self._ns / 1e9)
        self.stats.observe("decode_classify_max_seconds_per_step", self._max_ns / 1e9)
        self._calls = self._rows = self._ns = self._max_ns = 0

    def reset(self) -> None:
        self._idle.clear()
        self._cursor = None
        self._known_no_copy = 0
