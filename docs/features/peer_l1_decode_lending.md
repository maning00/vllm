# Decode instance cache lending

`PeerL1Connector` with `role: "decode"` lets prefill instances reuse cached KV
from a decode instance over the configured RDMA transport. It publishes local
cache blocks and adjusts eviction order; it never acquires KV for a request.
PD remains responsible for all decode request loads. No background backfill or
separate lending pool is created.

Use the [peer L1 requirements and security settings](peer_l1_share.md), including
MoE, prefix caching, compatible weights/geometry and the required ROCm copy
priority environment variables. Configure both PD legs for MoRIIO READ and
[PD replay-tail retention](moriio_connector_usage.md#retaining-replay-tails).
`DFlashSlidingWindowManager` groups and WRITE mode are unsupported for replay-tail
retention. DSpark drafters using ordinary sliding-window managers are supported.

## Enable a decode owner

On decode, combine the PD consumer with an owner-only peer child:

```json
{
  "kv_connector": "MultiConnector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "connectors": [
      {
        "kv_connector": "MoRIIOConnector",
        "kv_role": "kv_consumer",
        "kv_connector_extra_config": {
          "read_mode": true,
          "pd_replay_retention": true
        }
      },
      {
        "kv_connector": "PeerL1Connector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {
          "role": "decode",
          "instance_id": "decode-b",
          "kv_compatibility_id": "weights-and-config-v1",
          "control_host": "10.0.0.2",
          "control_port": 28600,
          "peer_seeds": ["tcp://10.0.0.1:28500"],
          "shared_secret_env": "PEER_L1_SECRET"
        }
      }
    ]
  }
}
```

Pass this object to `--kv-transfer-config`. Prefill keeps the default peer
`full` role and adds the decode endpoint to its seeds. Use the same compatibility
ID and secret on both instances. The secret named by `shared_secret_env` must be
inherited by every worker; direct `shared_secret` is plain text in the config.
Restrict control and transport endpoints to the trusted deployment.

Decode subscribes only to prefill placement hints. Prefill reads decode owners
as ordinary cross-instance sources. Equal-length same-instance IPC hits remain
preferred, and the remote minimum-gain rule still applies.

## Limits and capacity

These keys belong to the peer child's `kv_connector_extra_config`:

| Key | Default | Meaning |
| --- | --- | --- |
| `role` | `full` | `decode` publishes and orders cache rows without loading requests. |
| `protected_cap_blocks` | Twice the longest model-context prefix row bound, capped by `pin_cap` | Upper bound on protected rows plus admission reserve. Zero disables new grants. |
| `order_scan_blocks` | `256` | Maximum free-queue nodes visited per maintenance step. Positive. |
| `decode_min_free_blocks` | `min(256, num_blocks // 100)`, minimum 1 | Close the gate below this already-reserved admissible free count. Positive. |
| `decode_max_running_requests` | Scheduler `max_num_seqs` | Close the gate above this running-request count. Nonnegative. |
| `decode_max_waiting_requests` | `0` | Close the gate above this compute-waiting request count; PD loads, grammar and streaming waits are excluded. Nonnegative. |

The budget conservatively charges every unique externally protected row,
including locally used rows and references awaiting pool-side release, plus
the admission reserve. A new immediate window uses at most half the remaining
budget. Deferred grants can use the remaining budget at absorb and retain the
normal free-headroom limit. Existing grants are never revoked to satisfy a cap
or gate. `pin_window` and `pin_cap` remain upper bounds. A gated decode owner is
skipped immediately rather than treated as a busy window.

The default budget sums full-attention rows at `max_model_len` and up to two
replay-boundary tails per sliding-window group. Doubling leaves room for both
one such read and its reserve. This can reserve substantially more capacity
than a percentage default on models with a long context; set an explicit cap
to choose a smaller admission-loss bound. Readers shorten a hit at coordinator
alignment boundaries to the decode owner's advertised remaining budget and pin capacity rather
than repeatedly requesting an oversized grant.

Relative to ordinary PD, admission loses at most this protected cap plus the
separately planned temporary replay-tail storage. This does not guarantee zero
ITL impact: remote reads also consume HBM and network bandwidth. Increase the
cap only after choosing an acceptable admission-loss budget.

Newly free rows still present on any prefill owner go before other cached rows,
newest release batch first, preserving order within a batch. A bounded cursor
moves formerly redundant rows to the tail after their prefill copies disappear.
Contended hint queries conservatively classify a batch as having no prefill
copy on release. Retirement skips unknown or contended batches without advancing
the cursor. Explicit `evict_first` rows are never classified for retirement.
Hints can lag and never authorize a read. Re-publication affects placement
when a row is freed again.

## Metrics

All names use the `vllm:peer_l1_` prefix and per-engine labels:

- `owner_protected_blocks`, `owner_protected_high_water`, `owner_busy_protected_cap_total`.
- `decode_gate_open`, `decode_gate_closures_total`, `owner_busy_gate_total` and
  `owner_deferred_rejections_total{reason="gate"}`.
- `decode_order_front_rows_total`, `decode_order_retired_rows_total`,
  `decode_order_scanned_rows_total` and `decode_no_prefill_free_blocks`.
- Classification call/row counters, contention counter, and
  `decode_classify_{calls,rows,seconds,max_seconds}_per_step` histograms.
- `reader_truncations_total{reason="protected_cap"}` and the corresponding
  `reader_truncation_requested_positions` / `reader_truncation_granted_positions` histograms.
- `reader_decode_tokens_total`, `worker_decode_rdma_bytes_total` and
  `worker_decode_rdma_failures_total` on prefill readers.

The no-copy gauge describes the last release/scan classifications, not a global
placement census. It can temporarily include a locally reused row until the
bounded cursor visits it. Use the native protected-cap metrics for admission
safety. No-copy to redundant transitions are refreshed on the next release.
