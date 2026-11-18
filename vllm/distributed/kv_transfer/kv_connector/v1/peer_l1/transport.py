# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Byte-range reads with explicit completion and DMA quiescence."""

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from enum import Enum
from importlib import import_module
from typing import Any, ClassVar, Protocol

import torch


class ReadState(Enum):
    PENDING = "pending"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class ReadBatch(Protocol):
    def state(self) -> ReadState:
        """Return immediately; SUCCEEDED also proves quiescence."""
        ...

    def quiesced(self) -> bool:
        """Return immediately, including during concurrent posts or cleanup."""
        ...


class RemoteSession(Protocol):
    def prepare(self, ranges: Sequence[tuple[int, int, int]]) -> ReadBatch:
        """Prepare (local offset, remote offset, bytes) without posting."""
        ...

    def post(self, batch: ReadBatch) -> None:
        """Post a prepared batch; an exception leaves DMA permanently unknown."""
        ...


class PeerReadTransport(ABC):
    name: ClassVar[str]
    max_ranges_per_post: ClassVar[int]
    max_range_bytes: ClassVar[int]

    @classmethod
    @abstractmethod
    def available(cls) -> bool: ...

    @abstractmethod
    def __init__(
        self, storage: torch.UntypedStorage, host: str, options: Mapping[str, Any]
    ) -> None:
        """Register the whole backing allocation or raise synchronously."""
        ...

    @abstractmethod
    def advertisement(self) -> bytes: ...

    @abstractmethod
    def open(self, advertisement: bytes) -> RemoteSession: ...

    @abstractmethod
    def close(self) -> None:
        """Return promptly; never revoke published or possibly active memory."""
        ...


TRANSPORTS = {
    "mori": (
        "vllm.distributed.kv_transfer.kv_connector.v1.peer_l1.transports.mori:"
        "MoRITransport"
    )
}


def _load(path: str) -> type[PeerReadTransport]:
    module, separator, name = path.partition(":")
    if not separator:
        raise ValueError(
            f"Unknown RDMA transport {path!r}; choose from {list(TRANSPORTS)}"
        )
    cls = getattr(import_module(module), name)
    if not isinstance(cls, type) or not issubclass(cls, PeerReadTransport):
        raise TypeError(f"RDMA transport {path!r} must implement PeerReadTransport")
    if not cls.name or cls.max_ranges_per_post <= 0 or cls.max_range_bytes <= 0:
        raise ValueError(f"RDMA transport {path!r} has invalid name or limits")
    return cls


def get_transport(name: str | None = None) -> type[PeerReadTransport]:
    if name is not None:
        cls = _load(TRANSPORTS.get(name, name))
        if not cls.available():
            raise RuntimeError(
                f"RDMA transport {name!r} is unavailable; install its dependencies"
            )
        return cls
    available = []
    for path in dict.fromkeys(TRANSPORTS.values()):
        cls = _load(path)
        if cls.available():
            available.append(cls)
    if len(available) != 1:
        raise RuntimeError(
            "Set rdma_transport to one available implementation; "
            f"available={[cls.name for cls in available]}, "
            f"registered={list(TRANSPORTS)}"
        )
    return available[0]
