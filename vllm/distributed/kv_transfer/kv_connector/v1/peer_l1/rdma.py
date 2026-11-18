# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Transport-independent registered reads with explicit DMA ownership."""

import queue
import threading
from collections.abc import Callable, Mapping
from contextlib import suppress
from time import monotonic, perf_counter
from typing import Any

import msgspec
import torch

from vllm.logger import init_logger

from .async_load import AsyncRead
from .data import PeerMemory
from .native import description
from .registry import Domain
from .stats import PeerL1Stats
from .transport import ReadBatch, ReadState, RemoteSession, get_transport

logger = init_logger(__name__)
# Remote readers and unquiesced batches may outlive connector shutdown.
_PROCESS_LIFETIME: list[object] = []


class RdmaReadSession:
    def __init__(
        self,
        session: RemoteSession,
        local_rows: int,
        local_segments: tuple[tuple[tuple[int, int, int], ...], ...],
        remote: PeerMemory,
        credential: bytes,
        *,
        max_ranges_per_post: int,
        max_range_bytes: int,
    ):
        self.session = session
        self.local_rows, self.local_segments = local_rows, local_segments
        self.remote, self.credential = remote, credential
        self.max_ranges_per_post = max_ranges_per_post
        self.max_range_bytes = max_range_bytes
        self._first_post = True

    def _segments(self, read: AsyncRead) -> list[tuple[int, int, int]]:
        segments: list[tuple[int, int, int]] = []
        for group, source, target in read.job.blocks:
            if not (
                0 < source < self.remote.num_blocks and 0 < target < self.local_rows
            ):
                raise ValueError("RDMA peer row outside registered geometry")
            for src, dst in zip(
                self.remote.segments[group], self.local_segments[group], strict=True
            ):
                offset, stride, size = src
                local_offset, local_stride, local_size = dst
                if size != local_size:
                    raise ValueError("RDMA peer pages differ")
                copied = 0
                while copied < size:
                    width = min(size - copied, self.max_range_bytes)
                    entry = (
                        local_offset + target * local_stride + copied,
                        offset + source * stride + copied,
                        width,
                    )
                    if (
                        segments
                        and segments[-1][2] + width <= self.max_range_bytes
                        and segments[-1][0] + segments[-1][2] == entry[0]
                        and segments[-1][1] + segments[-1][2] == entry[1]
                    ):
                        local, remote, previous = segments.pop()
                        segments.append((local, remote, previous + width))
                    else:
                        segments.append(entry)
                    copied += width
        if not segments:
            raise ValueError("RDMA receive has no byte regions")
        return segments

    def run(
        self, read: AsyncRead, ready: Callable[[], bool], stop: threading.Event
    ) -> None:
        batches: list[ReadBatch] = []
        unknown = False
        success = False
        reason = "exception"
        try:
            segments = self._segments(read)
            # A grant can outlive its owner while queued here. Reject a restarted
            # owner before posting against a cached MR/session and needing quarantine.
            try:
                hello = description(
                    self.remote.endpoint, self.remote.epoch, credential=self.credential
                )
            except ConnectionError:
                reason = "owner_changed"
                raise
            if msgspec.json.decode(hello, type=PeerMemory) != self.remote:
                reason = "owner_changed"
                raise ValueError("RDMA owner epoch or description changed")
            while not ready():
                if stop.is_set() or monotonic() >= read.deadline:
                    reason = "cancelled" if stop.is_set() else "deadline_before_post"
                    return
                stop.wait(0.001)
            for begin in range(0, len(segments), self.max_ranges_per_post):
                ranges = segments[begin : begin + self.max_ranges_per_post]
                batch = self.session.prepare(ranges)
                read.resources.append((self, batch))
                if stop.is_set():
                    reason = "cancelled"
                    return
                if not read.begin(sum(size for _, _, size in ranges)):
                    return
                batches.append(batch)
                # An exception may follow partial posting without returning a status.
                unknown = True
                started = perf_counter()
                self.session.post(batch)
                if self._first_post:
                    logger.info(
                        "Peer L1 RDMA first post owner=%s epoch=%d seconds=%.6f",
                        self.remote.endpoint,
                        self.remote.epoch,
                        perf_counter() - started,
                    )
                    self._first_post = False
                unknown = False
                while batch.state() is not ReadState.SUCCEEDED:
                    if batch.state() is ReadState.FAILED:
                        reason = "status_failed"
                        return
                    if stop.is_set() or monotonic() >= read.deadline:
                        reason = "cancelled" if stop.is_set() else "deadline_after_post"
                        return
                    stop.wait(0.001)
            success = monotonic() < read.deadline
        except Exception:
            logger.debug(
                "Peer RDMA receive failed request=%s",
                read.job.request_id,
                exc_info=True,
            )
        finally:
            query = (
                None if unknown else lambda: all(batch.quiesced() for batch in batches)
            )
            drained = not read.submitted or (query is not None and query())
            read.complete(success, drained=drained, drain_query=query, reason=reason)


class RdmaReadMover:
    def __init__(
        self,
        domain: Domain,
        storage: torch.UntypedStorage,
        rows: int,
        segments: tuple[tuple[tuple[int, int, int], ...], ...],
        stats: PeerL1Stats,
        options: Mapping[str, Any],
    ):
        self.domain, self.rows, self.segments = domain, rows, segments
        self.storage, self.stats = storage, stats
        transport = get_transport(options.get("rdma_transport"))
        self.transport = transport(storage, domain.host, options)
        # Keep published registrations even if initialization or shutdown fails.
        _PROCESS_LIFETIME.append(self)
        self.advertisement = self.transport.advertisement()
        self._sessions: dict[tuple[str, int], RdmaReadSession] = {}
        self._reads: dict[str, AsyncRead] = {}
        self._stop = threading.Event()
        self._work: queue.Queue[tuple[AsyncRead, Callable[[], bool]] | None] = (
            queue.Queue(maxsize=128)
        )
        self._thread = threading.Thread(target=self._run, daemon=True, name="peer-rdma")
        self._thread.start()

    def _session(self, remote: PeerMemory) -> RdmaReadSession:
        key = remote.endpoint, remote.epoch
        if key not in self._sessions:
            started = perf_counter()
            if remote.rdma_transport != self.transport.name:
                raise ValueError("RDMA transport differs from the peer advertisement")
            self._sessions[key] = RdmaReadSession(
                self.transport.open(remote.rdma),
                self.rows,
                self.segments,
                remote,
                self.domain.credential,
                max_ranges_per_post=self.transport.max_ranges_per_post,
                max_range_bytes=self.transport.max_range_bytes,
            )
            logger.info(
                "Peer L1 RDMA session owner=%s epoch=%d seconds=%.6f",
                remote.endpoint,
                remote.epoch,
                perf_counter() - started,
            )
        return self._sessions[key]

    def submit(self, read: AsyncRead, ready: Callable[[], bool]) -> None:
        self._reads[read.identity] = read
        try:
            self._work.put_nowait((read, ready))
        except queue.Full:
            read.cancel_unsubmitted("queue_full")

    def retire(self, identity: str) -> None:
        self._reads.pop(identity, None)

    def _run(self) -> None:
        torch.accelerator.set_device_index(self.storage.device)
        while True:
            if self._stop.is_set() and self._work.empty():
                return
            item = self._work.get()
            if item is None:
                return
            read, ready = item
            if self._stop.is_set():
                read.cancel_unsubmitted()
                continue
            if read.snapshot()[0]:
                continue
            try:
                if read.job.memory is None:
                    raise ValueError("RDMA receive lacks its owner descriptor")
                session = self._session(read.job.memory)
                session.run(read, ready, self._stop)
            except Exception:
                read.complete(False, drained=not read.submitted)
                logger.debug(
                    "Peer RDMA setup failed request=%s",
                    read.job.request_id,
                    exc_info=True,
                )

    def close(self) -> None:
        self._stop.set()
        with suppress(queue.Full):
            self._work.put_nowait(None)
        # A provider may never finish an ERR-QP post. Retain ownership, not a join.
        self._thread.join(timeout=0.1)
        self.transport.close()
