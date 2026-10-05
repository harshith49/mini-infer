# mini-infer: Milestone 4 — continuous batching

## Intent and scope

Implement a readable synchronous scheduler that replaces completed requests between decoding steps. Preserve each request's independent output while supporting different output budgets, stop-token IDs, and seeded sampling settings. Measure the actual static-versus-continuous result rather than assuming a CPU speedup. This implements the user's approved FIFO queue, bounded running set, separate admission prefill, and batched decode approach.

The project is the standalone `harshith49/mini-infer` repository, rooted in the current project directory. Create `codex/mini-infer-m4` from the reviewed M3 head; keep the existing M3 draft PR separate. Publish milestone commits to this remote only; no automatic merge or force push. A later M4 PR will target the M3 branch while that dependency remains unmerged. No model downloads, environments, secrets, or unrelated enclosing-project files enter commits. Retain native execution with one fresh independent final reviewer.

## Approach and alternatives

Use a standard-library FIFO deque and an ordered running set bounded by `max_batch_size`. Each request owns a contiguous cache holding only its real tokens. Temporarily pack cache prefixes into a left-padded batch for one-token decoding, then copy only new K/V columns back to request caches. Reuse M3's masks and logical position handling; introduce no permanent cache pool, new dependency, threading framework, or server.

Forwarding every request individually would be simpler but would lose batched decoding. Keeping a permanent shared cache arena would reduce copying but introduces allocation and lifetime machinery that belongs to paged caching. The reference packing approach exposes the scheduling idea while making its copy/allocation cost measurable.

## Request and sampling contracts

Add a frozen `SamplingParams` in `engine/sampler.py` with `temperature=0.0`, `top_k=0`, `top_p=1.0`, and `seed=0`. Temperature zero selects existing greedy argmax without consuming randomness. Positive temperature divides logits before filtering; apply top-k before nucleus filtering, retain the token that crosses the cumulative top-p threshold, and always retain at least one candidate. Sample with `torch.multinomial` and a request-owned `torch.Generator` on the model device.

Require finite nonnegative temperature, integer top-k in `[0, vocab_size]` (zero disables it), finite top-p in `(0, 1]`, and an integer seed in `[0, 2**63 - 1]`. Validate settings even in greedy mode. Preserve `greedy()` and its lowest-ID tie behavior. Sampling accepts a single rank-1 floating logit vector; reject malformed/nonfinite raw logits rather than construct an invalid probability distribution. Filtered negative-infinity entries are internal and permitted.

Each submission supplies a unique nonempty string request ID, a nonempty rank-1 `torch.long` prompt on the model device, its own nonnegative integer `max_new_tokens`, optional stop-token IDs (default none), and sampling settings. Validate all values and `prompt_length + max_new_tokens <= model context` before changing scheduler state. Copy the accepted prompt so later caller mutation cannot change the request. Request IDs cannot be reused during one scheduler instance's lifetime.

Stop only on newly generated IDs; prompt tokens matching a stop ID do not finish the request. Include the first selected stop ID in the result. The finish reason is `stop` when that token matches, otherwise `length` at the output budget. Zero-output requests complete with their original prompt and no cache allocation or model forward. Padding ID is a validated model vocabulary ID and may also be a real or stopping token.

Seeded sampling must be independent of batch row, arrival time, other requests' seeds, and other requests finishing. Compare a sampled request alone versus amid other arrivals on the same device. Do not claim stochastic bitwise parity across CPU and CUDA, or token equality with HF's separate random-number consumption. Greedy public GPT-2 comparisons remain token-exact.

## Scheduler API and lifecycle

Create `engine/scheduler.py` exposing `Scheduler(model, *, max_batch_size, pad_token_id)`, `submit(request_id, prompt, max_new_tokens, *, stop_token_ids=(), sampling=None)`, `step() -> list[TokenEvent]`, `result(request_id) -> torch.Tensor`, and an `idle` property. `TokenEvent` carries `request_id`, `token_id: int | None`, and `finish_reason: str | None`; zero-output completion has no token. Results contain the original unpadded prompt plus actual new tokens. Unknown or unfinished result lookups fail clearly; completed results remain available for this scheduler's lifetime.

`max_batch_size` is a positive integer. The API is synchronous and single-owner: callers submit between steps; later server work will own concurrency. A step on an idle scheduler returns an empty list. At each step:

1. Snapshot existing running requests eligible to decode.
2. Admit waiting requests in FIFO order into free slots. Complete zero-output requests without consuming a running slot. Prefill newly admitted nonzero requests together using M3 left padding, sample their first token, and retain only real-token K/V in private caches.
3. Decode the existing running snapshot in one packed batch, producing one token per request. Newly prefilling requests do not receive an additional decode token in this same step.
4. Remove requests finishing by stop or budget, release their private caches, and retain output IDs. Their free slots become available at the next step. Deliver token/completion events in a documented deterministic phase order: admission events, then existing decode events.

A full active set does not wait for its longest request to finish before allowing replacements. No already-admitted request is evicted or reordered ahead of an older waiter. With finite request budgets and completed model forwards, every queued request is eventually admitted; this is FIFO admission fairness, not a wall-clock latency guarantee. Bound newly admitted prefills by free slots so ongoing decodes are not displaced by an unbounded admission pass.

Model-forward exceptions do not commit the failing phase's private cache prefixes, output IDs, or sampling state. Keep its requests available to retry. If a preceding phase already succeeded, its progress stays committed and its undelivered events remain available on the next successful call; do not silently lose or duplicate those events. This is phase-level recovery, not a transactional rollback of the entire step. Do not build a general retry framework or catch/retry indefinitely.

## Cache packing and memory

Allocate each private cache with capacity `prompt_length + max_new_tokens`, batch size one, model dtype/device, and no persistent padding. Prefill uses a temporary batch cache of capacity equal to the longest admitted prompt; copy each continuing row's real prompt K/V into its private cache before releasing the workspace. Requests finishing on their first token need no persistent cache. Sampling the first token does not yet commit that token's K/V.

For existing decodes, let `p` be the largest committed private prefix length. Allocate a temporary cache of capacity `p + 1`, left-pad each shorter prefix with initialized finite zeros, set temporary committed length to `p`, and construct the full key mask for those copied prefixes plus the valid next token. The current input is each request's previous sampled token. Derive learned positions from valid prefix lengths; the physical padding is only temporary. After a successful forward, copy only each row's new column into its private cache and advance that private length once. Never copy temporary padding into persistent request history.

Finished requests leave before the next decode batch. Never forward a final selected token when no further prediction is needed. Release vocabulary logits before any subsequent forward. Temporary K/V workspaces are initialized at masked prefix slots: multiplying zero attention weights by uninitialized NaN values would still propagate NaNs. Preserve original M1–M3 operations and APIs as independent correctness references.

Report current reserved private-cache bytes from actual owned tensors, and peak K/V tensor bytes including simultaneously live private and temporary caches. After all requests finish, scheduler-owned cache bytes must be zero; completed token-ID results are separate. These counters are K/V allocations, not total process memory. Packing copies and transient double storage are explicit reference limitations.

## Benchmark and runnable demonstration

Provide a small scheduler CLI accepting repeated prompts, per-request output budgets, `--max-batch-size`, device selection, and optional common sampling/seed flags. A single budget may apply to all prompts; otherwise require one per prompt. Seed empty text with GPT-2 EOS/BOS, preserve original prompt text, decode only new IDs, and print a JSON list in submission order despite differing completion times.

Add a focused standard-library benchmark saving `results/continuous_batching.csv`. Use deterministic synthetic token IDs, varied prompt/output lengths, more requests than active slots, greedy decoding, and no stop IDs. Both stages receive the exact same requests, device, thread count, and active limit. Static execution uses fixed FIFO cohorts with existing cached `generate_batch()`, the cohort's largest output budget, then trims each result to its requested budget; retained rows perform excess work until their cohort completes. Name this comparator precisely rather than imply M3 supports individual budgets. Choose workloads whose padded cohort budgets fit context.

Continuous execution drains the new scheduler. Check every useful requested output against independent cached generation before timing. Count only useful requested new tokens for throughput; report static excess token computation separately. Use one warmup and at least three repetitions, median end-to-end elapsed time, total useful tokens per second, and p50/p95 request completion latency measured from common workload submission. Static results become available at their cohort's return time; continuous results become available when their completion event is delivered. Include allocation, packing, prefill, sampling, and decode; exclude loading and tokenization. Do not compare whole-request completion latency with M2's per-token decode percentiles.

CSV rows identify stage, device/hardware, PyTorch version, thread count, request count, active limit, explicit prompt/output length lists, repetitions, total useful tokens, extra static token computation, median seconds, throughput, completion p50/p95, peak K/V tensor bytes, peak process-memory value/scope, and synthetic workload/seed description. Leave CPU process peak memory blank and GPU values unmeasured unless run on hardware. If measured continuous throughput is lower, publish it and explain observed packing/CPU overhead; do not enforce a hardware-dependent speedup in correctness tests or fabricate a winning workload result. Keep the existing M2 CSV unchanged.

## Acceptance and documentation

- Rerun all existing tests without changing their meaning or tolerances. Baseline, cached, and static-batch public GPT-2 checks remain mandatory.
- Add tiny-model sampler tests for greedy/no-RNG use, temperature/filtering boundaries, retained nucleus threshold token, reproducible per-request streams, and rejection of invalid settings/logits. Verify sampled IDs belong to explicitly known allowed candidates.
- Test FIFO waiting/admission, bounded active rows, replacements while a long request continues, dynamic submissions between steps, per-request budgets/stops, zero-output behavior, input/result identity rules, and cache release. Observe real batched forwards rather than mock the scheduling algorithm.
- Compare packed prefill/decode logits with independent private-cache forwards at `atol=1e-4, rtol=1e-4`, including different prompt/prefix lengths, changes in row order/occupancy, exact context limits, and initialized masked slots.
- For actual public GPT-2, compare at least three requests' 50-plus-token greedy outputs against independent engine and HF generation, with staggered admission, short/long/Unicode prompts, and active limits one and greater than one.
- Verify seeded stochastic results alone versus interleaved requests on the same device, multiple stop IDs, prompt mutation isolation, previous-logits release, model fault/retry with preserved events, no starvation in a finite queue, and peak/private K/V accounting. CUDA checks remain conditional and explicitly unverified on this host.
- Run the full suite, real CPU scheduler demonstration, mixed-length benchmark, compile/whitespace checks, and one fresh final read-only review. Fix important findings, update lessons with actual bugs/results, and push the verified milestone to `mini-infer`.

Update README and architecture Mermaid with queue/admission/prefill/decode/completion flow, sampling determinism, temporary packing costs, measured workload/results, and remaining limits. No page pool, cancellation protocol, asynchronous worker, HTTP server, memory-budget admission, chunked prefill, sampling penalties, or GPU speedup claim in this milestone. Those remain later work. Written-spec approval precedes the implementation plan; native execution is retained.
