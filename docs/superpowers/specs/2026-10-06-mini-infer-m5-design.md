# mini-infer M5: shared paged KV storage

## Goal and approved approach

Replace the scheduler's private contiguous request caches with optional fixed-size pages from a shared pool. Preserve public GPT-2 token outputs, request sampling streams, FIFO admission, phase recovery, and the existing contiguous implementation as a reference. Measure actual capacity and fragmentation without assuming paging improves throughput or memory use.

The user approved a shared pool with 16-token pages, per-request page tables, full-budget reservation on admission, FIFO waiting when pages are unavailable, rejection of requests larger than the entire pool, and PyTorch gathering into existing attention workspaces. Growing allocations during decoding is deferred: full reservation prevents already-running requests from becoming unable to finish because of another admission.

Use the standalone `harshith49/mini-infer` repository only. M5 starts from the verified M4 head `f16da70ee54b444c2af1d449830abe3a5bbdc01e`. Preserve M4's draft PR and earlier measurements. Publish milestone work on `codex/mini-infer-m5`; the eventual draft PR targets M4 while that dependency is unmerged. Retain native implementation and one fresh final independent review.

## Storage and cache interfaces

Extend `engine/kv_cache.py` with `PagePool` and `PagedKVCache`; no new dependency, allocator framework, abstract base class, CUDA kernel, or model architecture.

`PagePool(config, *, num_pages, page_size=16, device, dtype)` allocates two tensors shaped `[layers, pages, heads, page_size, head_dim]`. Every page ID refers to corresponding key/value blocks for every layer. Validate positive integer counts, floating dtype, and model dimensions before allocation. Page size is configurable for small boundary tests, with 16 as the public default. Pool allocation is fixed for its lifetime; unused/free pages remain physically resident.

A deterministic free-ID collection supplies any available pages, even when those IDs are not consecutive. Reserving a page list is all-or-nothing. Insufficient configured capacity raises a clear `MemoryError` before modifying free IDs. Returning IDs validates uniqueness, range and current allocation before mutation. A cache owns its page IDs; callers must not free or edit another cache's reservation. Pool ownership is synchronous, matching M4's single-owner model.

`PagedKVCache(pool, *, capacity)` reserves `ceil(capacity / page_size)` pages immediately. Capacity is a positive logical token limit within model context, and the cache has batch size one. It exposes `length`, `requires_attention_mask`, `validate`, `write`, byte accounting, a read-only page-ID sequence, and idempotent `close()`. Use after close rejects clearly. There is no finalizer-based correctness: explicit lifecycle release is mandatory.

`write(layer_idx, key, value)` accepts the same single-request K/V chunk shape as the contiguous cache. Translate logical token offsets into page-table entries and in-page offsets, including chunks crossing several boundaries. Writes after `length` are tentative; `length` advances once only after model logits succeed. Gather the written layer's prefix into contiguous `[1, heads, tokens, head_dim]` tensors for the existing attention math. The unused tail of a reserved page must never enter attention.

Provide a per-layer prefix read and chunk store operation for scheduler packing/scattering; both cache types use the same logical lengths. Reuse existing contiguous tensor slices, with paged storage translating offsets. This is a concrete two-backend extension, not a general storage plugin system. Model type annotations include both caches; model arithmetic, mask/position rules, and the default contiguous behavior stay unchanged.

Direct single-request paged model forwarding tests exercise page gathering at the attention boundary. Scheduler batching continues to use M4's temporary contiguous batch cache; it gathers real paged prefixes one layer/request at a time into that workspace and scatters only the new K/V column back. Gather directly by logical-token page IDs and in-page offsets, producing only the real prefix rather than materializing whole reserved page tables. Release each gathered K/V pair before reading the next layer/request or forwarding the batch. This still copies prefixes every step and does not provide a paged GPU attention kernel.

## Scheduler admission and exhaustion

Add an optional `page_pool` to `Scheduler`; omission retains the existing contiguous mode and behavior. Validate matching model config/device/dtype. A scheduler exclusively owns its supplied pool during execution; external allocations, multiple schedulers sharing the pool, or manual table mutation are outside the supported lifecycle. The pool must be entirely free when attached.

For each positive-budget request, required pages are based on `prompt_length + max_new_tokens`, matching M4's full capacity reservation. Reject an individually impossible request during `submit()` before changing queue, result, RNG, or pool state. Zero-output requests require no pages. Do not allocate pages for waiting submissions.

At each step, snapshot the old running set and select new requests subject to both free active slots and free pages. Complete FIFO zero-output requests without consuming either resource. Stop admission at the first positive-budget waiter that cannot fit; later waiters do not jump ahead. Existing running requests continue decoding. Because their full budgets are reserved and every admissible waiter fits an empty pool, finite requests eventually release enough space; this avoids a growth-induced deadlock.

Reserve all selected positive requests' page tables before the admission forward. Keep these reservations staged until that phase succeeds. A prefill forward failure closes only the staged caches, restores all their page IDs, and leaves queue, private prefixes, outputs and generators unchanged. Successful prefill scatters real prompt K/V into those tables. First-token completion closes its reservation immediately, including early stop or one-token budget.

Old-request decoding does not allocate more pool pages. A decode forward failure changes only the temporary workspace, preserving private prefixes and reservations. Successful decode scatters the new column, then commits each private length and selected token. Completion explicitly closes its cache before clearing ownership. Previously committed admission events survive a later failed decode, as in M4.

A step whose admission is page-blocked still decodes running requests; do not spin an empty loop while retaining reservations owned by an external caller. The exclusive-pool ownership contract is required for this admission policy. Configured pool exhaustion is handled by validation/queueing; host RAM or CUDA allocator failure when initially creating the pool is a separate allocation error and is not a promise of successful resource provisioning.

## Memory accounting

Keep physical allocation, live reservation and useful prefix storage separate:

- Pool resident bytes: actual key/value tensor storage; constant while the pool lives.
- Free/owned pages: allocator state, with `free + owned == num_pages` after every successful transition.
- Request reserved bytes: owned pages multiplied by actual bytes per page, including rounded tail slots.
- Used bytes: committed real-token K/V slots; excludes tentative writes and rounding.
- Tail rounding waste: owned page slots minus each cache's logical capacity.
- Unused logical reservation: logical capacity minus committed prefix length.

For paged scheduler mode, current request reservation sums owned pages, while peak physical K/V includes the entire resident pool plus any simultaneous batch/gather workspace, counting shared storage once. Report resident pool bytes even after all requests complete: all pages must be free and request reservation zero, but the pool tensors still exist. Contiguous-mode counters keep their M4 meaning. Include explicitly allocated prefix-gather pairs while they coexist with the pool/workspace; record their actual tensor bytes before releasing them. The cache-buffer peak excludes transformer activations, token/mask/index tensors and model weights, consistently with the existing cache-only scope. Direct paged attention reports pool residency and per-layer gather size separately; total CUDA tensor peak requires hardware measurement.

CPU process peak memory stays unmeasured. CUDA allocated tensor peak, if actually run, includes model and pool storage and must be labelled accordingly. Never report the sum of page capacity as total process memory.

## Runnable demonstration

Keep the current scheduler CLI default contiguous. Add `--cache-backend contiguous|paged`, `--num-pages` (default 32) and `--page-size` (default 16). Allocate the pool only in paged mode; contiguous remains the default. Validate settings before loading weights where possible. Use the same repeated prompts, budgets, stops, sampling flags, JSON output order and empty-prompt handling. A deliberately undersized pool must reject an impossible request with a clear CLI error; a smaller pool that fits requests individually must demonstrate queued admission and eventual completion.

No server, cancellation, swapping, eviction, prefix sharing, growing page allocations, memory-budget autotuning, or custom kernel is added.

## Benchmarks and fragmentation comparison

Add focused M5 measurements, preserving `results/kv_cache.csv` and `results/continuous_batching.csv` unchanged. The principal comparison is maximum concurrently admitted reservations under an explicit fixed KV byte budget.

In the clean-capacity case, compare actual contiguous request allocations with actual page-pool reservations for the same capacities, dtype and model dimensions. Convert the requested byte budget to a whole-page pool, report any unused budget remainder, and use the same resulting physical budget for both stages. Count whole requests that fit, reserved/used bytes and rounding waste. Paging can fit fewer requests here because of page rounding; publish that outcome.

For external fragmentation, use a separate deterministic allocation-only trace: allocate several reservations, free alternating ones, then attempt a larger reservation. Paged allocation uses real pool IDs; the comparator uses first-fit contiguous intervals in an equal-sized arena represented by standard-library metadata. Include a trace where total free slots suffice but no contiguous interval is large enough. Label this explicitly as an arena-allocation experiment, not measured PyTorch/CUDA allocator fragmentation. It demonstrates reuse of nonconsecutive page IDs; it does not prove the independent M4 tensor allocator suffers the same fragmentation.

Save stage, experiment, device/hardware, model/dtype, page size, effective budget, request capacities, completed allocation trace, admitted count, resident/reserved/used bytes, rounding/unused reservation, and free-space/largest-hole statistics as applicable. Blank fields mean unmeasured or inapplicable, not zero. Exact defaults and trace values will be fixed in the implementation plan.

Run a real mixed-request paged-versus-contiguous inference correctness gate alongside capacity measurements. If adding end-to-end timings, use identical requests and useful-token counts, at least three repetitions after warmup, include pool allocation in its documented scope, and publish overhead rather than enforce a hardware-dependent speedup. A fixed resident pool and gather workspace may increase memory over independent caches; full-budget reservation does not reduce unused output-budget commitments.

## Acceptance

Retain all original M1–M4 tests and their tolerances. Preserve the documented M4 additional public suffix-check scope; M5 must not narrow it further.

Test pool initialization/validation, deterministic nonconsecutive allocation, atomic exhaustion, invalid/double frees, idempotent cache release, closed-cache rejection, boundary-crossing chunk writes, gathering without unused page tails, exact context limits, and reserved/used/rounding byte formulas. Inject a late model failure and prove old committed page contents/length/mask metadata remain reusable; retries match the contiguous reference.

Compare direct paged cached logits with contiguous cached logits on identical prefixes at unchanged tolerances, including page-boundary one-token and multi-token appends. Check real public GPT-2 50-token greedy outputs against contiguous generation and existing engine/HF acceptance on short, long and Unicode prompts. Seeded same-device scheduler output must match the contiguous scheduler on the established sampling settings; CUDA checks remain conditional.

Scheduler tests must prove page-pressure FIFO admission, no skipping a blocked head, rejection before mutation, zero-budget behavior, first-token/EOS release, full completion with no page leaks, staged-allocation rollback, admission-success/decode-failure event retention, and truthful resident-versus-owned memory counters. Reused pages may contain stale data; only newly written valid prefixes can affect logits.

Benchmark checks validate clean capacity, a concrete external-fragmentation trace, rounding costs, equal effective budgets, metadata and CSV parsing. No timing assertion requires paging to win. Run the full suite, real paged CLI comparison, focused benchmarks, compile/whitespace checks, and one independent final review before publishing implementation.

## Spec self-review and limits

The approved reservation policy protects running requests but can leave budget space unused; growing pages and preemption are deferred. Page tables eliminate the need for one contiguous interval in the trace, but physical pool residency and temporary gathers remain real costs. FIFO admission can leave usable pages idle behind a larger waiter; this is deliberate fairness, not best-fit packing. The fixed pool, single-owner lifecycle, direct gather path, scheduler packing path and byte scopes have distinct contracts. No throughput, CPU process-memory, or CUDA result is claimed before measurement.

Written-spec approval precedes the implementation plan and implementation. No M5 product code or dependencies are introduced by this design document.
