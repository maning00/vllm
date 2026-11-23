# Decode owner invariants

The decode role has no request read path. It retains the regular owner,
publication watermark, leases and replica resolver. Membership marks an
instance as `full` or `decode`; decode resolvers accept only `full` members.
All rank-local members must agree on role.

## Free-row ordering

`BlockPool.free_blocks` invokes an optional classification callable once for
the owning pool's newly unreferenced cached rows. It preserves core release
order. Redundant rows are prepended after the queue's cached boundary; others
are appended. Explicit `evict_first` bypasses classification and retirement. Full-role guards
leave this callable unset.

Each dequeue changes the row's free-queue generation. Classification records
contain the key and generation, so local hits, re-free, invalidation and
keep-last reinsertion cannot reuse stale ordering state. Retirement checks
queue membership, key, generation and zero refcount. It moves only redundant
to no-copy transitions, preserving a stable predecessor as cursor when scanned
rows move. Each maintenance pass visits at most `order_scan_blocks` real nodes.
There is no scan of the uncached prefix.

The replica's aggregate query uses try-locks for the snapshot and shards. Lock
contention returns no result; the whole batch is placed conservatively. It does
not truncate a release batch or defer part of it into a newer batch. Retirement
keeps its cursor and skips the batch on contention or a missing member snapshot.

## Shared admission budget

Let `P` be the number of unique budgeted rows and `R` the remaining admission
reserve. Native owner state maintains `P + R <= protected_cap_blocks` under its
mutex. A row is charged from its first grant until it has neither pins nor an
absorbed/withheld pool reference. Locally used rows are conservatively charged.
Aliases and overlapping grants do not charge the same row twice.

Immediate grants check the window, reference cap, gate and shared budget before
adding pins. Deferred decisions check the same budget at absorb, alongside the
existing limit on free-row touches. Absorb resets the old rate window atomically
and computes a new window no greater than half the remaining protected budget
or the usual free-headroom limit. This leaves room for the fast path while
conservatively charging both pending pins and reserve.

Rows returned as release actions remain charged until `complete_absorb`, after
the scheduler applies touch/free/return actions. A concurrent re-grant keeps the
existing charge. Window recomputation uses the latest budget and never revokes
existing protection. No allocation interleaves with applying an absorb.
Withheld rows remain covered by reserve and are not reused.

Gates read the pool's already-reserved free count and live request statuses.
They affect only new grants. Finishing a request requests one coalesced
maintenance step so an idle engine can refresh its gate after physical free.
Pool operations stay on the scheduler thread; the decode request path performs
no ACQUIRE/CANCEL RPC.

## PD replay storage

MoRIIO's explicit `pd_replay_retention` option enables request-scoped absolute
block ranges. A range is the minimal contiguous suffix covering the normal PD
window and each manager's required tails at `replay_boundaries`. Dense segment
tails are disabled for this calculation. Storage, attention reach and prefix-hit
requirements remain separate.

The producer holds storage through chunked prefill and completion-ACK delayed
free. It exports only computed prompt blocks and validates each range has no
nulls. The consumer validates geometry/hash/speculation settings, allocates the
same range, and pairs by absolute position, excluding fully local blocks and
whole trailing recompute/speculative blocks. Copying a partial valid block
retains the existing block-granular transfer semantics.

The consumer records allocation generations against the actual local scheduler
output object. Core reconciles load errors before lifting the extra constraint;
free/pop discard old generations. Worker metadata and the default PD wire
format are unchanged when disabled. Confirmed failures only lift the extra
constraint: core retains ownership of error recovery and physical references.
Existing synchronous HMA READ failure recovery remains fail-stop.

Capacity planning includes `max(1, H) + A - 1` tokens plus normal re-prefill
retention and block rounding. Bounded-replay groups need only the normal READ's
recomputed-last-token margin. This storage-only parameter is excluded from the
KV compatibility representation and never expands attention or hit masks.

Subscription frames advertise remaining protected capacity (zero when gated).
Readers recompute coordinator-aligned plans within that bound. The budget check
also queues deferred acquires that exceed the fast-path reserve budget. A closed
gate has a separate reply; it does not cause reader busy waiting. Remote-KV,
grammar and streaming waiters do not count as decode compute pressure.
