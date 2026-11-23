# Peer L1 ownership

The connector's directory contains full-block prefix hashes and physical KV rows.
Placement replicas are hints; the owner revalidates complete keys and its epoch
before granting a read. KV compatibility includes geometry and hash settings.

See [decode owner invariants](peer_l1_decode_lending.md) for decode cache lending
and the PD replay-storage lifecycle.

## Publication

A cached block is initially pending. The worker records completion after all
step writers, including speculative draft writes, and reports consecutive
completed fences. The scheduler publishes only keys covered by that watermark.
Invalidation removes a pending key or retires a published row before reuse.
Only finalized full blocks may be shared.

## Immediate grants and admission

The free queue keeps uncached rows before cached rows and explicit keep-last
rows. `evict_first=True` inserts before other cached rows without displacing
uncached capacity. Invalidation and successful reset maintain that boundary.
An optional lease-guard `classify_freed(blocks) -> list[bool]` classifies each
batch of newly free cached rows once. True rows go to the cached front in their
original order; other rows retain normal LRU placement. Explicit `evict_first`
overrides that hint. Each pool invokes its own guard.

The native owner can grant rows between scheduler steps. A retired pinned free
row is withheld rather than reused. If `W` is the interval's initial reserve and
`h` rows have been withheld, the remaining reserve is `W - h`. Allocating `a`
rows and withholding `d` changes admissible capacity by exactly `a`:

```text
(F - a - d) - (W - h - d) = F - (W - h) - a
```

The new-pin window bounds all rows that could still become withheld before
absorb. Absorbed pins hold pool references. Retiring a pinned in-use row adopts
its pool reference before the local reference can be freed.

At absorb, the window is bounded by `pin_window` and free headroom. It need not
reserve the configured maximum in a busy pool. Owner pins never block eviction
or force the owner to wait for a reader.

## Deferred grants

An oversized request queues keys without pinning rows. The scheduler snapshots
candidate rows and their refcounts, then commits eligible grants under the
native mutex. Entries arriving after the snapshot wait for another absorb.
Free-row touches are limited to half the remaining free headroom after immediate
touches. No allocation may interleave between snapshot and applying these pool
references.

Pins become absorbed before a committed result is visible. A concurrent release
dirties the row for a later absorb, preserving the reference even if release
precedes application of the returned touch. Reset refuses pending entries as
well as active pins. Queue capacity is bounded by total key positions.

An idle owner receives an engine wakeup after enqueue. The callback requests a
step only; all pool operations stay on the scheduler thread. Uncommitted entries
survive until an eligible absorb or reader cancellation. A late reader consumes
a decided result before applying its busy deadline to an undecided result.

## Completion and failures

IPC batches share a copy stream and one completion callback releases all grants
for the step. A possible failed load retains ownership under fail-stop; there is
no timer that can release a possibly active IPC read.

RDMA grants retain their original lease clock. Queueing, setup and the GPU fence
must fit within half that lease. The first possible post starts the transfer
budget, capped by the same bound; subsequent segments do not renew it.

An unsubmitted failure is cancelled and released. A possible post releases only
when drained. Otherwise the worker reports both failed and finished, and the
scheduler takes an extra reference on each target row before recomputation,
abort or normal finish can free the request's references. Quarantine belongs to
the transfer identity, independently of the request, until drain or process exit.
An elapsed lease or failed status alone is not drain proof.

After failure, that live admission excludes RDMA sources; source cooldown and the
quarantine circuit breaker apply independently. Source cleanup uses epoch and
sequence, so a restarted owner cannot lose a newer grant through stale cleanup.

Published registrations and potentially referenced argument/status objects live until
process exit when revocation or drain cannot be proven. Discovery and replica
refresh run off the request ingress path; HELLO revalidation before a post rejects
an already observable owner restart while the job was queued.

## Read transport

`PeerReadTransport(storage, host, options)` registers the complete backing
allocation synchronously. Its class `name` is the wire identity: discovery
requires equal transport names as well as KV and hash compatibility. A lazy
registry resolves `rdma_transport` names or `module:Class` implementations;
without a selection, exactly one implementation must be available.

The interface separates preparation, posting, completion and drain:

- `available()` checks dependencies; `advertisement() -> bytes` exports opaque
  registration metadata; `open(advertisement) -> RemoteSession` creates a session.
- `prepare(ranges) -> ReadBatch` prepares `(local_offset, remote_offset, bytes)`
  ranges without posting. `post(batch) -> None` can partially submit before raising.
- `ReadBatch.state() -> ReadState` returns `PENDING`, `SUCCEEDED` or `FAILED`.
  Success proves quiescence; failure alone does not.
- `ReadBatch.quiesced() -> bool` never waits, including during concurrent posts,
  status checks or cleanup. A serialized backend must return `False` if its lock
  is held; handle release must use that same lock.
- `close() -> None` returns promptly without revoking published registrations
  or freeing anything an unquiesced batch can use.

The executor splits and batches ranges using `max_range_bytes` and
`max_ranges_per_post`. It owns queueing, step fences, deadlines, HELLO checks and
process-lifetime retention, independently of the transport implementation.
An exception from `post()` leaves that read permanently unknown, even if a
returned status later indicates success. A normal post followed by a deadline
still permits drain queries and removal of quarantine once quiesced; the failed
request never becomes successful afterward.

Published registrations and their backing storage remain retained until process
exit even when no local read is outstanding: remote unilateral reads cannot be
enumerated or revoked. Unquiesced reads additionally retain batches, sessions,
descriptors and submission arguments. The MoRI-IO implementation uses the
existing engine wrapper and its owned-read helpers.
