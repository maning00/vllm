# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Transport plugins preserve ownership independently of failure reporting."""

import os
import sys
import threading
from dataclasses import replace
from types import ModuleType

import pytest
import torch

from tests.v1.kv_connector.unit.test_peer_l1 import (
    clock as clock,
)
from tests.v1.kv_connector.unit.test_peer_l1 import (
    tcp_members as tcp_members,
)
from tests.v1.kv_connector.unit.test_peer_l1 import (
    wait_until,
)
from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1 import (
    async_load,
    rdma,
    transport,
)
from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1.data import CopyJob
from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1.registry import (
    Domain,
    compatible,
)
from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1.session import (
    NativeOwnerChannel,
    peer_key,
)
from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1.stats import PeerL1Stats
from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1.transport import (
    PeerReadTransport,
    ReadState,
)

pytestmark = pytest.mark.cpu_test
PLUGIN = __name__ + ":FakeTransport"


class Batch:
    def __init__(self):
        self.result = ReadState.PENDING
        self.drained = threading.Event()
        self.lock = threading.Lock()

    def state(self):
        return self.result

    def quiesced(self):
        if not self.lock.acquire(blocking=False):
            return False
        try:
            return self.drained.is_set()
        finally:
            self.lock.release()


class FakeTransport(PeerReadTransport):
    name = "fake"
    max_ranges_per_post = 1
    max_range_bytes = 1
    enabled = True

    @classmethod
    def available(cls):
        return cls.enabled

    def __init__(self, storage, host, options):
        self.storage = storage
        self.options = options
        self.batch = Batch()
        self.posted = threading.Event()
        self.closed = threading.Event()

    def advertisement(self):
        return b"fake registered memory"

    def open(self, advertisement):
        return self

    def prepare(self, ranges):
        assert ranges == [(17, 1, 1)]
        return self.batch

    def post(self, batch):
        self.posted.set()
        if self.options.get("raise_post"):
            raise RuntimeError("partial post")
        if self.options.get("success"):
            batch.result = ReadState.SUCCEEDED
            batch.drained.set()
        elif self.options.get("failed"):
            batch.result = ReadState.FAILED

    def close(self):
        self.closed.set()


def test_transport_selection_and_wire_identity(monkeypatch, tcp_members):
    monkeypatch.setattr(transport, "TRANSPORTS", {"alias": PLUGIN})
    assert transport.get_transport() is FakeTransport
    assert transport.get_transport("alias") is transport.get_transport(PLUGIN)
    _, member = tcp_members[0]()
    local = replace(member, rdma_transport=transport.get_transport("alias").name)
    remote = replace(member, rdma_transport=transport.get_transport(PLUGIN).name)
    assert compatible(local, remote)
    assert not compatible(local, replace(remote, rdma_transport="different"))
    monkeypatch.setattr(FakeTransport, "enabled", False)
    with pytest.raises(RuntimeError, match="unavailable"):
        transport.get_transport("alias")
    with pytest.raises(RuntimeError, match="available=\\[\\]"):
        transport.get_transport()
    monkeypatch.setattr(FakeTransport, "enabled", True)
    monkeypatch.setitem(transport.TRANSPORTS, "another", __name__ + ":OtherTransport")
    with pytest.raises(RuntimeError, match="Set rdma_transport"):
        transport.get_transport()
    with pytest.raises(ValueError, match="Unknown RDMA transport"):
        transport.get_transport("missing")


def test_mori_transport_registers_whole_storage_and_reuses_owned_reads(monkeypatch):
    import msgspec

    from vllm.distributed.kv_transfer.kv_connector.v1.moriio import moriio_engine
    from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1.transports.mori import (
        Advertisement,
        MoRISession,
        MoRITransport,
    )

    events = []
    status = Batch()
    module = ModuleType("mori.io")
    module.__dict__["BackendType"] = type("BackendType", (), {"RDMA": "rdma"})
    module.__dict__["IOEngine"] = lambda *args: object()
    module.__dict__["IOEngineConfig"] = lambda **kwargs: object()
    package = ModuleType("mori")
    package.__dict__["io"] = module
    monkeypatch.setitem(sys.modules, "mori", package)
    monkeypatch.setitem(sys.modules, "mori.io", module)

    class Status:
        def Succeeded(self):
            return status.result is ReadState.SUCCEEDED

        def Failed(self):
            return status.result is ReadState.FAILED

    class Wrapper:
        def __init__(self, engine):
            self.local_memory_metadata = b"local"

        def set_backend_type(self, backend):
            assert backend == module.BackendType.RDMA

        def register_local_tensor(self, tensor):
            events.append(("register", tensor.numel()))
            return b"memory"

        def get_agent_metadata(self):
            return b"engine"

        def register_remote_engine(self, metadata):
            assert metadata == b"engine"

        def get_unpack_memory_metadata(self, metadata):
            assert metadata == b"memory"
            return b"remote"

        def build_session(self, local, remote):
            assert (local, remote) == (b"local", b"remote")
            return object()

        def prepare_owned_read(self, sizes, local, remote, memory):
            assert memory == b"remote"
            return moriio_engine.MoRIIOOwnedRead(Status(), (sizes, local, remote))

        def submit_owned_read(self, submission):
            events.append(("post", submission.arguments))

    monkeypatch.setattr(moriio_engine, "MoRIIOWrapper", Wrapper)
    monkeypatch.delenv("MORI_DISABLE_AUTO_XGMI", raising=False)
    backend = MoRITransport(torch.UntypedStorage(128), "127.0.0.1", {})
    assert events == [("register", 128)]
    assert os.environ["MORI_DISABLE_AUTO_XGMI"] == "1"
    session = backend.open(backend.advertisement())
    assert isinstance(session, MoRISession)
    assert session.advertisement == Advertisement(1, b"engine", b"memory")
    batch = session.prepare([(4, 8, 16)])
    assert len(events) == 1 and batch.state() is ReadState.PENDING
    session.post(batch)
    assert events[-1] == ("post", ([16], [4], [8]))
    status.result = ReadState.FAILED
    assert batch.state() is ReadState.FAILED and not batch.quiesced()
    status.result = ReadState.SUCCEEDED
    assert batch.state() is ReadState.SUCCEEDED and batch.quiesced()
    backend.close()
    assert backend.tensor.numel() == 128
    with pytest.raises(ValueError, match="version"):
        backend.open(msgspec.json.encode(Advertisement(2, b"engine", b"memory")))
    monkeypatch.setenv("MORI_DISABLE_AUTO_XGMI", "0")
    with pytest.raises(ValueError, match="MORI_DISABLE_AUTO_XGMI=1"):
        MoRITransport(torch.UntypedStorage(128), "127.0.0.1", {})


class OtherTransport(FakeTransport):
    name = "other"


@pytest.fixture
def receive(tcp_members, monkeypatch):
    owner, member = tcp_members[0](lease_ms=30000)
    channel = NativeOwnerChannel(
        member.endpoint, member.epoch, 88, tcp_members[2], lease_s=30
    )
    key = peer_key(b"transport")
    owner.publish([key], [1])
    grant = channel.acquire_exact([key])
    now = [grant.acquired_before]
    monkeypatch.setattr(async_load, "monotonic", lambda: now[0])
    monkeypatch.setattr(rdma, "monotonic", lambda: now[0])
    receives = async_load.AsyncReceives(channel.release)
    read = receives.add(CopyJob("source", grant, ((0, 1, 17),), "read", member), 1)
    return owner, member, tcp_members[3], receives, read, now


@pytest.mark.parametrize("mode", ["success", "failed", "late", "raise_post", "before"])
def test_generic_executor_keeps_failure_separate_from_quiescence(receive, mode):
    owner, member, credential, receives, read, now = receive
    backend = FakeTransport(torch.UntypedStorage(128), "host", {mode: True})
    session = rdma.RdmaReadSession(
        backend,
        128,
        member.segments,
        member,
        credential,
        max_ranges_per_post=1,
        max_range_bytes=1,
    )
    if mode == "before":
        now[0] += 16
    thread = threading.Thread(
        target=session.run, args=(read, lambda: True, threading.Event())
    )
    thread.start()
    if mode == "late":
        assert backend.posted.wait(5)
        now[0] += 2
    thread.join(timeout=5)
    assert not thread.is_alive()
    output = receives.poll()
    assert output.finished_recving == {"read"}
    assert output.failed_recving == (set() if mode == "success" else {"read"})
    unsafe = mode in ("failed", "late", "raise_post")
    assert bool(receives.metadata(0).quarantined) == unsafe
    if unsafe:
        assert owner.refs == 1
        backend.batch.result = ReadState.SUCCEEDED
        backend.batch.drained.set()
        output = receives.poll()
        assert not output.finished_recving
        assert bool(receives.metadata(0).drained) == (mode != "raise_post")
    if mode != "raise_post":
        wait_until(lambda: owner.refs == 0)
    else:
        assert owner.refs == 1


def test_worker_drain_query_does_not_wait_for_a_transport_lock(receive):
    _, _, _, receives, read, _ = receive
    batch = Batch()
    assert read.begin(1)
    read.complete(False, drained=False, drain_query=batch.quiesced)
    held, release, queried = threading.Event(), threading.Event(), threading.Event()

    def post():
        with batch.lock:
            held.set()
            assert release.wait(10)
            batch.drained.set()

    def poll():
        receives.poll()
        queried.set()

    io = threading.Thread(target=post)
    io.start()
    assert held.wait(5)
    worker = threading.Thread(target=poll)
    worker.start()
    try:
        assert queried.wait(5)
        assert receives.metadata(0).quarantined == {read.identity: (17,)}
    finally:
        release.set()
        io.join(timeout=5)
        worker.join(timeout=5)
    receives.poll()
    assert receives.metadata(0).drained == {read.identity}


def test_close_retains_published_storage_and_unknown_batches(receive, monkeypatch):
    _, member, _, receives, read, _ = receive
    monkeypatch.setattr(transport, "TRANSPORTS", {"fake": PLUGIN})
    monkeypatch.setattr(torch.accelerator, "set_device_index", lambda device: None)
    domain = replace(
        Domain.from_options(
            {
                "instance_id": "local",
                "kv_compatibility_id": "weights",
                "control_host": "127.0.0.1",
                "control_port": 29000,
            }
        ),
        credential=receive[2],
    )
    storage = torch.UntypedStorage(128)
    mover = rdma.RdmaReadMover(
        domain,
        storage,
        128,
        member.segments,
        PeerL1Stats(),
        {"rdma_transport": "fake", "raise_post": True},
    )
    # HELLO must use the same credential and descriptor as the live owner.
    read.job = replace(read.job, memory=replace(member, rdma_transport="fake"))
    # Keep HELLO exact while changing only the transport advertisement name.
    monkeypatch.setattr(
        rdma,
        "description",
        lambda *a, **kw: __import__("msgspec").json.encode(read.job.memory),
    )
    mover.submit(read, lambda: True)
    assert mover.transport.posted.wait(5)
    wait_until(lambda: read.snapshot()[0])
    receives.poll()
    mover.close()
    assert mover in rdma._PROCESS_LIFETIME
    assert mover.storage is storage
    assert mover.transport.closed.is_set()
    assert read.resources and not read.snapshot()[2]
    # A published registration stays retained even with no reader-side jobs.
    empty = rdma.RdmaReadMover(
        domain, storage, 128, member.segments, PeerL1Stats(), {"rdma_transport": "fake"}
    )
    empty.close()
    assert empty in rdma._PROCESS_LIFETIME and empty.storage is storage
