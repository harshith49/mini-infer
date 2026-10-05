# mini-infer: Milestone 2 — KV cache

## Intent and scope

Continue the user's milestone sequence with a per-request contiguous KV cache. Process the prompt once (prefill), then process one newly generated token at a time (decode). Preserve the Milestone 1 uncached path and its real GPT-2 correctness tests. Measure actual CPU performance before claiming a speedup; CUDA measurements remain conditional on hardware.

The user authorized pushing the project to GitHub and proceeding step by step. Milestone 1 is published to the configured repository `harshith49/mini-infer`, branch `codex/mini-infer-m1`, standalone project root. Keep commits small, push each verified milestone, and never include weights, local environments, or credentials. Do not merge feature branches into main automatically.

## Approach

Use a fixed-capacity contiguous cache allocated once for each request. This avoids repeated concatenation and provides an honest contiguous-allocation baseline for later paging comparisons. A concatenating cache would be shorter initially but reallocates and copies on every decode; a shared allocator would introduce scheduling machinery before it is needed. No new dependency or cache abstraction framework is required.

`SimpleKVCache` stores keys and values shaped `[layers, batch, heads, capacity, head_dim]`. It records the committed sequence length. Allocate on the model's device in the model's dtype, with capacity equal to prompt length plus requested output budget. Report allocated bytes from the actual tensor sizes, distinguishing reserved capacity from used token slots.

## Model integration

Extend `GPT2Model.forward(input_ids, *, cache=None)` without changing uncached behavior. With a cache, position IDs start at its committed length. Every attention layer writes new keys/values into its own slot and reads the prefix plus new positions. Advance the cache length once after all layers succeed, never once per layer.

For an existing prefix of length `past` and a new chunk of length `length`, use causal-mask rows `past:past+length` and key columns `:past+length`. This is crucial for both one-token decode and multi-token chunked prefill: a query's position is absolute, not its index in the new chunk.

Validate model/cache dimensions, dtype/device, positive capacity, input batch shape, and remaining capacity before any cache writes. Reject context or capacity overflow clearly. Keys/values outside the committed prefix are never read; a failed forward must not advance the committed length. Independent cache instances isolate requests. Cache tensors are request state, not model parameters or persistent checkpoint buffers.

## Generation and CLI

Keep `generate(..., use_cache=False)` available as the baseline. Add cached generation selected by `use_cache=True` and a CLI `--use-cache` flag. Both return the prompt plus generated token IDs, preserve EOS semantics, validate the full output budget, and handle zero new tokens without unnecessary cache allocation or a forward.

For a positive output budget, prefill the prompt to select the first new token. For each subsequent token, forward only the previously selected token through the cache. The final selected token need not be forwarded because generation is ending. Do not feed the prompt's last token twice, and do not process the first generated token before returning it.

## Correctness criteria

- Rerun every existing test unchanged in meaning, including FP32 public GPT-2 logits at `atol=1e-4, rtol=1e-4` and three exact 50-new-token HF comparisons.
- For all three existing public prompts, cached greedy generation must match both the uncached engine and HF for at least 50 new tokens.
- Compare cached one-token and multi-token-chunk logits with corresponding suffix logits from uncached forward. Cover different chunk boundaries, a single-token prompt, and exact context capacity.
- Check cache length advances once per model call; request caches remain independent; reserved/used byte accounting is exact; capacity overflow and incompatible cache requests are rejected without advancing state.
- Recheck zero-token requests, EOS termination, empty CLI prompt seeding, and original prompt preservation in both generation modes.
- Use tiny models for allocation and error-path tests, but actual public GPT-2 weights remain the acceptance oracle. Diagnose failures without relaxing tolerances.

## Benchmark

Add `benchmarks/bench_stages.py` with a runnable naive-versus-cached experiment for this milestone, writing `results/kv_cache.csv`. Generate exactly the same number of tokens in both modes with EOS stopping disabled. Verify token equality before timing. Use fixed seed, documented prompt-token lengths (16, 64, 128, 256 by default), 32 generated tokens, one warmup, and at least three measured repetitions. Expose device, lengths, output count, repetitions, and CPU thread count as CLI arguments.

Use `time.perf_counter()` and CUDA synchronization around timed work when CUDA is available. Record end-to-end generation seconds, generated tokens per second, time to first token, per-decode-step p50/p95 latency, cache reserved bytes, device/hardware, thread count, prompt/output lengths, and repetition count. Keep the naive/cached token-selection and timing overhead comparable. Include prefill in total generation time; separate it from decode latency. With fewer than two generated tokens, decode latency is unavailable and must be marked explicitly rather than invented.

Do not label cache bytes as process peak memory. CUDA peak allocated memory may be reported using PyTorch's device counters; CPU peak process memory remains unmeasured unless independently measured. Save observed CPU numbers with their full environment and workload context, even if caching is slower on a short workload. No mandatory speedup assertion belongs in correctness tests.

## Files and release

Create `engine/kv_cache.py`, `tests/test_kv_cache.py`, and `benchmarks/bench_stages.py`; modify model/generation code and real-reference tests. Update README and architecture diagrams to show prefill versus decode, add benchmark commands, and append actual cache/masking lessons and observed results. Keep other optimizations, stochastic sampling, and serving out of this milestone.

After written-spec review, create the implementation plan and retain native execution with one final independent review. Complete the full suite, cached CPU CLI demo, and measured CPU benchmark before committing/pushing the verified milestone. Publish only the engine project changes to the configured GitHub branch; no automatic merge or force push.
