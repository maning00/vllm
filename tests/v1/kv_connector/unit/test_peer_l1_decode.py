# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decode owners lend within a shared budget and preserve eviction order."""

import threading
from contextlib import ExitStack, nullcontext
from copy import deepcopy
from dataclasses import replace
from uuid import uuid4

import numpy as np
import pytest

from tests.v1.core.test_prefix_caching import make_request
from tests.v1.kv_connector.unit.test_peer_l1 import (
    _publish_blocks,
    wait_until,
)
from tests.v1.kv_connector.unit.test_peer_l1 import (
    adaptive_peer as adaptive_peer,
)
from tests.v1.kv_connector.unit.test_peer_l1 import (
    async_scheduler as async_scheduler,
)
from tests.v1.kv_connector.unit.test_peer_l1 import (
    clock as clock,
)
from tests.v1.kv_connector.unit.test_peer_l1 import (
    scheduler_resources as scheduler_resources,
)
from tests.v1.kv_connector.unit.test_peer_l1 import (
    tcp_members as tcp_members,
)
from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1.async_load import (
    AsyncReceives,
)
from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1.data import (
    PeerHandshake,
)
from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1.decode import DecodeCacheOrder
from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1.native import (
    CleanupQueue,
    NativeOwner,
)
from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1.registry import compatible
from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1.scheduler import PeerScheduler
from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1.session import (
    NativeOwnerChannel,
    OwnerDeferred,
    OwnerGated,
    peer_key,
)
from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1.stats import PeerL1Stats
from vllm.utils.hashing import sha256
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import init_none_hash, make_block_hash_with_group_id
from vllm.v1.request import RequestStatus

pytestmark = pytest.mark.cpu_test


class HintSource:
    def __init__(self):
        self.owners: list[set[bytes]] = [set(), set()]
        self.lock = threading.Lock()
        self.queries: list[list[bytes]] = []

    def query(self, keys):
        self.queries.append(list(keys))
        if not self.lock.acquire(blocking=False):
            return None
        try:
            return np.asarray(
                [any(key in owner for owner in self.owners) for key in keys],
                dtype=np.uint8,
            )
        finally:
            self.lock.release()


class OrderGuard:
    def __init__(self, order):
        self.order = order

    def on_cached(self, block, key, *, num_tokens):
        self.order.forget(block)

    def on_uncached(self, block, *, popped):
        self.order.forget(block)
        return False

    def classify_freed(self, blocks):
        return self.order.classify(blocks)

    def admission_reserve(self):
        return 0

    def reset_scope(self):
        return nullcontext(True)


def order_pool(scan=2):
    init_none_hash(sha256)
    pool = BlockPool(25, enable_caching=True, hash_block_size=4)
    blocks = pool.get_new_blocks(8)
    request = make_request("order", list(range(32)), 4, sha256)
    pool.cache_full_blocks(request, blocks, 0, 8, 4, 0)
    source, stats = HintSource(), PeerL1Stats()
    source.owners[0].update(peer_key(b.block_hash) for b in blocks)
    order = DecodeCacheOrder(pool, source.query, stats, scan)
    pool.lease_guard = OrderGuard(order)
    return pool, blocks, source, order, stats


def cached(pool):
    return list(pool.free_block_queue.iter_blocks_after(None, cached_only=True))


def test_decode_order_keeps_new_batches_first_and_reclassifies_local_hits():
    pool, blocks, source, order, _ = order_pool()
    pool.free_blocks(blocks[3::-1])
    pool.free_blocks(blocks[7:3:-1])
    expected = blocks[::-1]
    assert cached(pool) == expected
    assert list(map(len, source.queries)) == [4, 4]
    pool.touch([blocks[1]])
    pool.free_blocks([blocks[1]])
    expected.remove(blocks[1])
    expected.insert(0, blocks[1])
    assert cached(pool) == expected
    for _ in range(12):
        order.finish_step()
        assert cached(pool) == expected


def test_retirement_aggregates_owners_and_never_reorders_unchanged_no_copy_rows():
    pool, blocks, source, order, stats = order_pool()
    source.owners[1].update(source.owners[0])
    pool.free_blocks(blocks[::-1])
    expected = cached(pool)
    key = peer_key(blocks[-1].block_hash)
    source.owners[0].remove(key)
    order.finish_step()
    assert cached(pool) == expected
    source.owners[1].remove(key)
    for _ in range(4):
        before = stats.data.get("counters", {}).get("decode_order_scanned_rows", 0)
        order.finish_step()
        after = stats.data["counters"]["decode_order_scanned_rows"]
        assert 0 < after - before <= 2
    expected.remove(blocks[-1])
    expected.append(blocks[-1])
    assert cached(pool) == expected
    for _ in range(16):
        order.finish_step()
        assert cached(pool) == expected


def test_retirement_cursor_survives_reuse_keep_last_and_invalidation():
    pool, blocks, source, order, _ = order_pool(scan=1)
    pool.free_blocks(blocks[::-1])
    order.finish_step()
    # Reinserted same-key rows must not reuse an earlier free lifetime.
    pool.touch([blocks[-1]])
    pool.unpin_blocks([blocks[-1]], lambda block: None)
    source.owners[0].clear()
    for _ in range(24):
        order.finish_step()
    assert cached(pool) == blocks[-1:] + blocks[-2::-1]
    unchanged = cached(pool)
    for _ in range(16):
        order.finish_step()
    assert cached(pool) == unchanged
    pool.evict_blocks({blocks[2].block_id})
    assert pool.free_block_queue.get_all_free_blocks()[0] is blocks[2]
    assert pool.get_new_blocks(1) == [blocks[2]]
    for _ in range(12):
        order.finish_step()
    assert blocks[2] not in cached(pool)


def test_classification_contention_falls_back_without_waiting():
    pool, blocks, source, order, stats = order_pool()
    held, release, finished = threading.Event(), threading.Event(), threading.Event()

    def hold_snapshot():
        with source.lock:
            held.set()
            assert release.wait(10)

    def free():
        pool.free_blocks(blocks[::-1])
        finished.set()

    updater = threading.Thread(target=hold_snapshot)
    updater.start()
    assert held.wait(5)
    worker = threading.Thread(target=free)
    worker.start()
    try:
        assert finished.wait(5)
        assert cached(pool) == blocks[::-1]
    finally:
        release.set()
        updater.join(timeout=5)
        worker.join(timeout=5)
    assert stats.data["counters"]["decode_classify_contended"] > 0
    order.finish_step()
    assert "decode_classify_max_seconds_per_step" in stats.data["histograms"]


@pytest.mark.parametrize("cap", [0, 1, 5, 10, 25, 100])
def test_native_deferred_budget_includes_reserve_and_release_ack(cap):
    with ExitStack() as stack:
        owner = NativeOwner(
            f"vllm-peer-l1/budget/{uuid4().hex}",
            7,
            1024,
            window=16,
            pin_cap=256,
            protected_cap=cap,
        )
        stack.callback(owner.close)
        cleanup = CleanupQueue()
        stack.callback(cleanup.close)
        channel = NativeOwnerChannel(owner.endpoint, 7, 91, cleanup)
        keys = [peer_key(str(i).encode()) for i in range(201)]
        owner.publish(keys, list(range(1, 202)))
        demand = max(1, min(cap, 20))
        with pytest.raises(OwnerDeferred) as queued:
            channel.acquire_exact(keys[:demand], defer=True)
        owner.deferred_rows()
        touched, _, _ = owner.absorb(free_blocks=1000)
        grant = channel.acquire_exact(
            keys[:demand],
            defer=True,
            sequence=queued.value.sequence,
            acquired_before=queued.value.acquired_before,
        )
        assert owner.absorb_stats()["owner_protected_blocks"] <= cap
        if cap == 0:
            assert grant is None and not touched
            return
        assert grant.rows == tuple(range(1, demand + 1))
        assert len(touched) == demand
        owner.complete_absorb(1000 - demand)
        channel.cancel_unsubmitted(grant.sequence)
        _, freed, _ = owner.absorb(free_blocks=1000 - demand)
        assert len(freed) == demand
        # Pool references still exist until the scheduler acknowledges its actions.
        assert not owner.freeze()
        owner.thaw()
        assert owner.absorb_stats()["owner_protected_blocks"] >= demand
        owner.complete_absorb(1000)
        assert owner.freeze()
        owner.thaw()
        assert owner.absorb_stats()["owner_protected_high_water"] <= cap
        with pytest.raises(OwnerDeferred) as too_large:
            channel.acquire_exact(keys[: cap + 1], defer=True)
        owner.deferred_rows()
        owner.absorb(free_blocks=1000)
        assert (
            channel.acquire_exact(
                keys[: cap + 1],
                defer=True,
                sequence=too_large.value.sequence,
                acquired_before=too_large.value.acquired_before,
            )
            is None
        )
        assert owner.absorb_stats()["owner_deferred_reject_budget"] > 0


@pytest.mark.parametrize(
    "free,window,cap", [(90, 10, 20), (100, 10, 20), (200, 10, 25)]
)
@pytest.mark.parametrize("absorb_before_alloc", [False, True])
def test_decode_admission_loss_stays_bounded_across_window_recomputation(
    adaptive_peer, free, window, cap, absorb_before_alloc
):
    peer, pool, channel = adaptive_peer(
        free, window, role="decode", protected_cap_blocks=cap
    )
    blocks, keys = _publish_blocks(peer, pool, free - window)
    pool.free_blocks(blocks)
    peer.build()
    grant = channel.acquire_exact(keys[:window])
    assert grant is not None
    if absorb_before_alloc:
        peer.build()
    assert (
        free - pool.free_block_queue.num_free_blocks + peer.admission_reserve() <= cap
    )
    available = pool.get_num_free_blocks()
    allocated = pool.get_new_blocks(available)
    assert pool.get_num_free_blocks() == 0
    channel.cancel_unsubmitted(grant.sequence)
    peer.build()
    assert (free - available) - pool.get_num_free_blocks() <= cap
    pool.free_blocks(allocated)
    peer.build()
    assert free - pool.get_num_free_blocks() <= cap
    assert peer.get_stats().data["gauges"]["owner_protected_high_water"] <= cap


def test_decode_request_gates_block_new_grants_but_keep_existing_ones(adaptive_peer):
    peer, pool, channel = adaptive_peer(
        100, 10, role="decode", protected_cap_blocks=20, decode_max_running_requests=1
    )
    blocks, keys = _publish_blocks(peer, pool, 1)
    pool.free_blocks(blocks)
    request = make_request("gate", list(range(17)), 16, sha256)
    peer.new_request(request)
    with pytest.raises(OwnerGated):
        channel.acquire_exact(keys)
    request.status = RequestStatus.RUNNING
    peer.build()
    grant = channel.acquire_exact(keys)
    assert grant is not None
    another = make_request("gate-2", list(range(17)), 16, sha256)
    another.status = RequestStatus.RUNNING
    peer.new_request(another)
    with pytest.raises(OwnerGated):
        channel.acquire_exact(keys)
    assert channel.acquire_exact(keys, sequence=grant.sequence).rows == grant.rows
    channel.cancel_unsubmitted(grant.sequence)
    assert peer.match(request, 0) == (0, False)
    stats = peer.get_stats().data["counters"]
    assert stats["decode_gate_closures"] > 0


def test_finished_decode_wakes_one_gate_refresh_after_pool_release(adaptive_peer):
    peer, pool, _ = adaptive_peer(100, 10, role="decode", protected_cap_blocks=20)
    request = make_request("idle-gate", list(range(17)), 16, sha256)
    peer.new_request(request)
    peer.build()
    assert peer.get_stats().data["gauges"]["decode_gate_open"] == 0
    peer.finished(request.request_id)
    assert peer.pending()
    peer.build()
    assert not peer.pending()
    assert peer.get_stats().data["gauges"]["decode_gate_open"] == 1


def test_shared_rows_charge_once_and_gated_deferred_plans_are_rejected():
    with ExitStack() as stack:
        owner = NativeOwner(
            f"vllm-peer-l1/shared-budget/{uuid4().hex}",
            7,
            128,
            window=10,
            protected_cap=10,
        )
        stack.callback(owner.close)
        cleanup = CleanupQueue()
        stack.callback(cleanup.close)
        channel = NativeOwnerChannel(owner.endpoint, 7, 91, cleanup)
        keys = [peer_key(str(i).encode()) for i in range(12)]
        owner.publish(keys, [1, 1, *range(2, 12)])
        first = channel.acquire_exact(keys[:2])
        second = channel.acquire_exact(keys[1:2])
        assert first.rows == (1, 1) and second.rows == (1,)
        touched, _, _ = owner.absorb(free_blocks=100)
        assert touched == [1]
        owner.complete_absorb(99)
        assert (
            owner.absorb_stats()["owner_protected_blocks"]
            == 1 + owner.admission_reserve()
        )
        channel.cancel_unsubmitted(first.sequence)
        channel.cancel_unsubmitted(second.sequence)
        owner.absorb(free_blocks=99)
        assert not owner.freeze()
        owner.thaw()
        owner.complete_absorb(100)
        with pytest.raises(OwnerDeferred) as queued:
            channel.acquire_exact(keys[2:10], defer=True)
        owner.set_gate(False)
        owner.deferred_rows()
        owner.absorb(free_blocks=100)
        assert owner.absorb_stats()["owner_deferred_reject_gate"] > 0
        assert (
            channel.acquire_exact(
                keys[2:10],
                defer=True,
                sequence=queued.value.sequence,
                acquired_before=queued.value.acquired_before,
            )
            is None
        )
        assert owner.refs == 0


@pytest.mark.parametrize("scheduler_resources", [{"mtp": True}], indirect=True)
@pytest.mark.parametrize("ipc_tokens", [0, 1024])
def test_prefill_accepts_decode_owners_and_preserves_ipc_preference(
    async_scheduler, ipc_tokens, tcp_members
):
    scheduler, request, scheduled, owner, ipc = async_scheduler(
        ipc_tokens=ipc_tokens, remote_role="decode"
    )
    job = scheduled.kv_connector_metadata.jobs[0]
    if ipc is not None:
        assert job.grant.lease_s is None and job.memory is None
    else:
        assert job.memory.role == "decode"
        stats = PeerL1Stats()
        channel = NativeOwnerChannel(
            job.grant.endpoint,
            job.grant.epoch,
            job.grant.reader,
            tcp_members[2],
            lease_s=30,
        )
        receives = AsyncReceives(channel.release, stats)
        read = receives.add(job, 1.0)
        assert read.begin(4096)
        read.complete(False, drained=True)
        receives.poll()
        wait_until(lambda: owner.refs == 0)
        assert stats.data["counters"]["worker_decode_rdma_bytes"] > 0
        assert stats.data["counters"]["worker_decode_rdma_failures"] > 0
        assert (
            scheduler.connector.get_kv_connector_stats().data["counters"][
                "reader_decode_tokens"
            ]
            > 0
        )


def test_decode_role_only_subscribes_prefill_members(scheduler_resources, tcp_members):
    base, _, _ = scheduler_resources
    create, publish, _, _, replace_member = tcp_members
    source, member = create(window=512)
    reserved, local = create(epoch=991, window=512)
    reserved.close()
    local = replace_member(
        local, instance_id="decode", restart_group="decode", role="decode"
    )
    config = deepcopy(base.config)
    config.kv_transfer_config.kv_connector_extra_config.update(
        role="decode",
        instance_id="decode",
        kv_compatibility_id="weights",
        control_host="127.0.0.1",
        control_port=int(local.endpoint.rsplit(":", 1)[1]),
        peer_seeds=[member.endpoint],
        shared_secret="deployment secret",
        protected_cap_blocks=10,
    )
    manager = KVCacheManager(
        base.cache, max_model_len=256, scheduler_block_size=16, hash_block_size=4
    )
    peer = PeerScheduler(config, base.cache, (0, 1))
    try:
        peer.bind(manager)
        peer.handshake(PeerHandshake([local]))
        request = make_request("zero-match", list(range(64)), 4, sha256)
        assert manager.allocate_slots(request, 64) is not None
        blocks = [
            b for group in manager.get_blocks(request.request_id).blocks for b in group
        ]
        keys = [peer_key(b.block_hash) for b in blocks]
        source.publish(keys, list(range(1, len(keys) + 1)))

        def ready():
            peer.build()
            result = peer._prefill_copies(keys)
            return result is not None and result.all()

        wait_until(ready)
        assert peer.match(request, 0) == (0, False)
        assert source.refs == 0
        source.absorb()
        assert source.absorb_stats()["owner_grants"] == 0
        assert compatible(local, member)
        assert not compatible(local, replace(member, role="decode"))
        assert compatible(member, local)
    finally:
        peer.close()


@pytest.mark.parametrize(
    "options",
    [
        {"role": "unknown"},
        {"role": "decode", "protected_cap_blocks": -1},
        {"role": "decode", "order_scan_blocks": 0},
        {"role": "decode", "decode_min_free_blocks": 0},
        {"role": "decode", "decode_max_running_requests": -1},
        {"role": "decode", "decode_max_waiting_requests": -1},
    ],
)
def test_decode_options_reject_invalid_budgets_and_thresholds(adaptive_peer, options):
    with pytest.raises(ValueError, match="role must be|limits must be"):
        adaptive_peer(100, 10, **options)


def test_members_reject_mixed_roles_without_device_registration(tcp_members):
    from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1.worker import (
        validated_members,
    )

    _, first = tcp_members[0](rank=0)
    _, second = tcp_members[0](rank=1)
    with pytest.raises(ValueError, match="roles differ"):
        validated_members(
            [first, replace(second, role="decode")], rank=0, implicit=False
        )


def test_retire_scan_preserves_the_batch_and_cursor_while_hints_are_unknown():
    pool, blocks, source, order, _ = order_pool(scan=2)
    pool.free_blocks(blocks[::-1])
    order.finish_step()
    before = cached(pool)
    source.owners[0].clear()
    with source.lock:
        order.finish_step()
        assert cached(pool) == before
    order.finish_step()
    assert cached(pool) == before[:2] + before[4:] + before[2:4]


def test_explicit_evict_first_rows_are_not_moved_after_prefill_retires():
    pool, blocks, source, order, _ = order_pool(scan=8)
    pool.free_blocks(blocks[:4])
    pool.free_blocks(blocks[4:], evict_first=True)
    source.owners[0].clear()
    order.finish_step()
    assert cached(pool) == blocks[4:] + blocks[:4]
    assert list(map(len, source.queries)) == [4, 4]


def test_prefill_placement_is_unknown_until_every_member_has_a_snapshot():
    from vllm.distributed.kv_transfer.kv_connector.v1.peer_l1.native import (
        PlacementReplica,
    )

    endpoint = f"vllm-peer-l1/no-snapshot/{uuid4().hex}"
    key = peer_key(b"not-yet-connected")
    with ExitStack() as stack:
        replica = PlacementReplica([(endpoint, 7)])
        stack.callback(replica.close)
        assert replica.try_query_any([key]) is None
        owner = NativeOwner(endpoint, 7, 32, window=10)
        stack.callback(owner.close)
        wait_until(lambda: replica.try_query_any([key]) is not None)
        assert not replica.try_query_any([key]).any()


def test_decode_pd_waiters_do_not_close_the_gate(adaptive_peer):
    peer, pool, channel = adaptive_peer(100, 10, role="decode", protected_cap_blocks=20)
    blocks, keys = _publish_blocks(peer, pool, 1)
    pool.free_blocks(blocks)
    request = make_request("pd-load", list(range(17)), 16, sha256)
    request.kv_transfer_params = {"do_remote_prefill": True}
    peer.new_request(request)
    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
    peer.build()
    grant = channel.acquire_exact(keys)
    assert grant is not None
    channel.cancel_unsubmitted(grant.sequence)


@pytest.mark.parametrize("scheduler_resources", [{"mtp": True}], indirect=True)
def test_decode_capacity_shortens_a_remote_hit_instead_of_recomputing(async_scheduler):
    scheduler, request, step, owner, _ = async_scheduler(
        remote_role="decode", remote_cap=24, remote_tokens=512
    )
    assert request.num_computed_tokens == 512
    assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert len(step.kv_connector_metadata.jobs[0].blocks) <= 24
    assert 0 < owner.refs <= 24


@pytest.mark.parametrize("scheduler_resources", [{"mtp": True}], indirect=True)
def test_gated_decode_is_skipped_without_deferring_a_reader(async_scheduler):
    scheduler, request, _, owner, _ = async_scheduler(remote_role="decode")
    owner.set_gate(False)
    policy = scheduler.connector._scheduler
    wait_until(lambda: policy._resolver.snapshot().replica.capacity(0) == 0)
    another = make_request(
        "gate-reader",
        list(request.prompt_token_ids),
        4,
        sha256,
        num_hash_lookahead_tokens=2,
    )
    assert scheduler.connector.get_num_new_matched_tokens(another, 0) == (0, False)
    assert policy.get_stats().reduce().get("reader_busy_replies", 0) == 0


def test_default_decode_capacity_covers_a_long_multigroup_prefix(scheduler_resources):
    from tests.v1.core.test_kv_cache_pd import new_manager, request_of

    base, _, _ = scheduler_resources
    config = deepcopy(base.config)
    config.model_config.max_model_len = 131072
    config.kv_transfer_config.kv_connector_extra_config["role"] = "decode"
    cache = replace(new_manager().kv_cache_config, num_blocks=44000)
    peer = PeerScheduler(config, cache, tuple(range(5)))
    try:
        request = request_of("long-prefix", 100003)
        length, hashes = peer._lookup.find_longest_cache_hit(
            request.block_hashes,
            99840,
            lambda h, g: True,
            num_request_tokens=request.num_tokens,
        )
        keys = [
            peer_key(make_block_hash_with_group_id(h, g))
            for g, group in enumerate(hashes)
            for h in group
            if h is not None
        ]
        assert length == 99840 and len(keys) > 400
        assert peer.protected_cap >= 2 * len(keys)
        with ExitStack() as stack:
            owner = NativeOwner(
                f"vllm-peer-l1/default-cap/{uuid4().hex}",
                7,
                44000,
                window=16384,
                protected_cap=peer.protected_cap,
            )
            stack.callback(owner.close)
            cleanup = CleanupQueue()
            stack.callback(cleanup.close)
            channel = NativeOwnerChannel(owner.endpoint, 7, 91, cleanup)
            owner.publish(keys, list(range(1, len(keys) + 1)))
            grant = channel.acquire_exact(keys)
            assert grant is not None and len(grant.rows) == len(keys)
            channel.cancel_unsubmitted(grant.sequence)
    finally:
        peer.close()
