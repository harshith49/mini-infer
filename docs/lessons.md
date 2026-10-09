# What I learned / what broke

## Milestone 1 — October 5, 2026

- **Python version matters:** the host defaults to Python 3.14.4. Used an isolated Python 3.11.17 environment with verified torch 2.5.1, transformers 4.48.3, and pytest 8.3.5 instead of assuming compatible wheels for the host default.
- **Project boundaries affect pytest:** without a local configuration, pytest selected the parent repository and attempted to write its cache there. Added project-local `pytest.ini` so this project's tests and caches stay local.
- **Check library APIs:** probing GPT2Attention's old `_attn` method raised AttributeError. Inspected the installed source and executed the current `eager_attention_forward` and tiny model to verify tensor names, shapes, activation, and attention backend. The installed default is SDPA; explicitly chose eager attention as the transparent reference.
- **Conv1D transposes are easy to miss:** square attention projection matrices still need transposition. A tiny reference comparison and an explicit square-weight check cover this. Validation occurs before any parameter copy so a missing late-layer tensor cannot partially update a model.
- **Validation must cover zero work:** a zero-token generation request still needs valid tokens and shape. Reused model input validation before the loop, which avoids a wasted forward just to validate the prompt.
- **Downloads require real execution:** sandboxed package installation could not resolve PyPI. Authorized network execution installed the pinned packages and downloaded actual public GPT-2 without a token. Model files and virtual environments remain ignored.
- **Upstream warnings:** the initial weight download emitted an hf_xet deprecation warning. Cached offline tests run without it. No correctness tolerance was changed.
- **Baseline correctness:** 17 structural checks passed first, then 24 weight/logit checks. After generation, 41 tests passed and one CUDA test skipped in 9.26 seconds. Maximum absolute logit difference was 0 on three public GPT-2 prompts; greedy tokens matched HF exactly for 50 new tokens per prompt.
- **CPU demo:** `python -m engine.generate --prompt 'Hello, world!' --max-new-tokens 50 --device cpu` produced a continuation successfully. An offline run with `OMP_NUM_THREADS=1` also succeeded. Thread tuning here is a practical development setting, not a benchmark claim.

CUDA parity is unverified: this machine exposes no CUDA device. No throughput, latency, or memory benchmark is published yet. KV cache, batching, paging, quantization, serving, Docker, CI, and Colab work remain for subsequent milestones.

### Final independent review

The reviewer independently reran the original suite (41 passed, one CUDA skip) and found two observable bugs:

- **HF can hide missing weights:** `from_pretrained()` fills omitted checkpoint tensors with random values, so validating its resulting `state_dict()` alone is insufficient. A real damaged safetensors checkpoint reproduced the acceptance bug. The loader now inspects `output_loading_info` and rejects missing required keys before copying; diagnostics include expected tensor shapes. A normally omitted duplicate tied output-head weight remains legitimate. The direct mapper also reports the expected shape for missing tensors.
- **Decoding can alter a prompt:** decoding the full sequence with `skip_special_tokens=True` removed literal `<|endoftext|>` text supplied by the user, even for zero new tokens. A real-tokenizer CLI test reproduced this. The CLI now preserves original prompt text and decodes only newly generated IDs. A tiny real-forward check separately verifies the empty-prompt seed.

All regression failures were observed before their fixes. After fixes: **44 passed, one CUDA skip**, all three logit errors remained 0, and the CPU CLI preserved `Hello <|endoftext|> world` with zero new tokens. Safetensors 0.8.0 is now pinned because the on-disk regression directly imports it. The damaged-checkpoint test intentionally triggers HF's missing-weight diagnostic before our rejection.

### Execution decisions

- Kept the explicitly requested directory and used a local feature branch instead of moving to another checkout. A later relocation would need path changes.
- Kept the execution ledger in this project's ignored scratch directory, avoiding writes to unrelated parent-project tooling. Generic parent-workspace tooling will not discover that ledger automatically.
- Scoped pytest to this project; parent repository tests are outside this engine's validation.
- Used HF eager attention as the FP32 oracle; optional fused backends and CUDA still need their own hardware validation.
- Shared token validation between forward and generation so zero-token requests are validated without a redundant forward; this adds one small model method to maintain.

Later milestones and CUDA execution remain outside this review. Full output-budget reservation, the eager reference, legitimate tied-head omission, and preserved download exceptions follow the approved scope. No review findings remain deferred. Work stays on `codex/mini-infer-m1` without merging or pushing.

## Milestone 2 — October 5, 2026

- **Absolute mask rows matter:** cached queries start at the committed prefix length. One-token and multi-token chunk tests cover the offset rows rather than only testing a single prefill shape. No correctness tolerance was changed.
- **Length is a model-level commit:** each layer writes at the same offset; committing once per layer would overcount. Keys written after the committed prefix stay tentative until the final logits succeed. An injected late-layer fault leaves the old prefix intact, and a real retry matches uncached logits.
- **Prefill/decode off-by-one:** the first token comes from prompt logits; subsequent steps forward only the previously generated token. A three-token prompt plus four outputs has forward lengths `[3,1,1,1]` cached and `[3,4,5,6]` uncached. The final selected token is not forwarded. Zero outputs cause no allocation or forward.
- **Reserved memory is not peak process memory:** GPT-2 FP32 cache reservation is 73,728 bytes per token per request. The 128+32 workload reserves 11.25 MiB. Used slots advance with committed length while reservation stays constant. CPU process peak memory remains blank in the CSV.
- **Measure complete steps:** optional timings include initial validation/allocation/prefill for the first token and full decode steps thereafter. CUDA synchronizes only for instrumentation; ordinary generation avoids timer/synchronization calls. One-token output leaves decode percentiles unavailable.
- **Real-model verification:** after cache integration, 58 tests passed; after cached generation, 87 passed; after benchmark checks, 96 passed. One CUDA hardware check skipped each time. Cached greedy output matches HF and the baseline for 50 new tokens on all three prompts; public cached chunk/suffix logits satisfy the original `atol=1e-4, rtol=1e-4`.
- **Hardware metadata:** restricted execution could report only the arm architecture. Read-only CPU identification verified Apple M5; reran the offline benchmark with hardware metadata access so the published CSV names the CPU. No weights or keys were uploaded.
- **Observed results:** representative 128-prompt/32-output FP32 CPU workload measured 14.56 tokens/s naive and 108.49 cached (7.45×). Each stage uses one warmup and three measured repetitions with one PyTorch thread. The full CSV includes all four prompt lengths. This is synthetic token-ID traffic with instrumentation overhead, not a production throughput claim.

No API changes to the uncached default, EOS stopping, empty prompt seed, or literal prompt preservation were intended. PyTorch inference cache writes detach K/V state; differentiable training through the cache is unsupported. No new dependency was added. The approved native execution and project-local ledger conventions from M1 were retained.

### Milestone 2 final review

The independent reviewer reran all 96 tests (one CUDA skip) and checked batched exact-capacity forwarding and projection-failure retry. It found one important memory regression: a loop variable retained the previous full logits during the next forward, unlike M1's temporary expression. At long GPT-2 prefixes this can overlap roughly 196 MiB of unnecessary vocabulary logits with the next step. Real-forward weak references reproduced the lifetime bug in both modes. Deleting the logits immediately after argmax fixes it; both lifetime tests then pass. No token/logit tolerance changed.

The post-fix suite has **98 passing tests and one CUDA skip**. The existing conditional CUDA test now checks cached greedy tokens too, but remains unexecuted on this CPU-only machine. The CPU benchmark was refreshed after the memory fix; final CSV values supersede the preliminary measurements. No review findings remain deferred.

Per-request cache ownership excludes sharing a populated cache across different weights/concurrent requests. Manual tensor/length corruption and differentiable cache training are outside the constructor/model-managed lifecycle. Supporting those would require extra ownership/training checks; CUDA still requires actual hardware validation. The current implementation and documentation preserve the approved scope.

Final Git verification caught standard-library CSV CRLF endings as trailing whitespace on regenerated rows. Benchmark CSV output now uses LF, and the saved artifact was normalized without changing measured values. Staged diff checks cover new artifacts as well as tracked changes.

## Standalone repository correction — October 5, 2026

The project now publishes to `harshith49/mini-infer` with engine, tests, benchmarks, and documentation at the repository root. Earlier milestones inherited an unrelated enclosing Git remote; that repository selection was a mistake. The completed M2 snapshot was independently verified with 98 passing tests and one CUDA hardware skip before migration. M3 implementation remains uncommitted work. Future milestone pushes use this project's own remote.

## Milestone 3 — static batching

- Left padding needs two coordinate systems: causal/cache offsets are physical columns, while learned positions count real tokens. Tiny and public GPT-2 suffix-logit comparisons cover both.
- A leading padded query can have no legal keys. Finite-minimum score masking followed by explicitly zeroing blocked softmax probabilities keeps padded outputs finite and valid tokens independent of padding IDs.
- EOS is row state: its selection remains a valid token, then later filler becomes masked. Controlled projections after real transformer forwards verify one row finishing at step one while another finishes at step three, including padding IDs that equal EOS.
- The cache mask-required flag commits only with successful logits. A late projection exception leaves length, committed prefix, and flag reusable; omitted masks after padded commits reject clearly. Historical masks remain caller-owned and must preserve committed validity.
- Prior logits are released before the next batch forward in both modes. Whole-batch validation and zero-output tests prove that invalid later prompts or no-work requests do not allocate caches or forward.
- Public GPT-2 batch outputs match independent engine and HF greedy output for 50 tokens in both cache modes, with short/128-plus-token/Unicode prompts, three orders, and single-row batches. Full and cached suffix logits retain atol=rtol=1e-4.
- Acceptance suite: 169 passed, two CUDA hardware skips. Real two-prompt cached and uncached CLI runs produce matching JSON continuations; compileall and whitespace checks pass. GPU parity and batch throughput remain unmeasured. Existing M2 CSV data is unchanged.

Implementation stayed native with project-local scratch tracking. Batch-loop tests were split into a focused file rather than adding them to the model-mask tests. Repository migration corrected the inherited remote; active work now uses standalone mini-infer on codex/mini-infer-m3.

### Milestone 3 final review

The fresh independent reviewer reran the suite (169 passed, two CUDA skips) and found no critical or important issues. It identified one stale README limitation describing only single-request generation; corrected it because the user explicitly requested an accurate project description. No engine fix was needed. No findings remain deferred.

Decisions: retain project-local scratch tracking (manual ledger must remain accurate), separate batch-loop tests (small fixture duplication), and standalone repository identity (local folder remains in its original filesystem location). Historical masks stay caller-owned (mutating old validity can invalidate cached states); CUDA remains unverified (device-specific discrepancies need hardware); batch performance/compaction/admission remain future work (padded and finished rows waste compute). Nonfinite/corrupted weights and differentiable cache training remain outside the supported inference lifecycle (they need separate numerical and autograd validation).

## Milestone 4 — continuous batching, October 6, 2026

- **Per-request state:** FIFO admission and a bounded running set let a short request complete while a long request continues. Prefill and old-request decode are separate phases. Real model traces test replacements, delayed arrivals, finite-queue fairness, stops, zero work and context boundaries.
- **Padding is temporary:** private caches retain real-token K/V only. Decode packs different prefix lengths into zero-initialized temporary storage. A zero attention weight does not protect against an uninitialized NaN value; poisoned-allocation tests verify masked slots are initialized.
- **Recovery needs event retention:** if admission succeeds and the old-request decode forward fails, the admission output/cache stays committed. Its event survives to the next successful call without another prefill or duplicate delivery. Failed-phase private prefixes, outputs and RNG states stay unchanged.
- **Independent random streams:** request-owned generators isolate seeds from other requests and global RNG. Temperature/top-k/nucleus tests verify allowed candidates and threshold crossing. For temperatures below one, centering double-precision scores before dividing handles extremely small positive values. At temperatures at least one, scaling first avoids overflow between opposite extreme finite FP64 logits; transformer arithmetic stays FP32.
- **Numerical-oracle limitation:** an added every-step packed-versus-full-prefix public logit test found shape-dependent CPU reductions near zero after a long incremental history. In the diagnostic prefix, HF's own incremental cached logits differed from its full-prefix logits at 273 elements beyond atol=rtol=1e-4. Custom full forwards matched HF exactly. Experimental attention precision and operand padding did not solve the reference mismatch and were rejected. The added strict public suffix check covers each prompt's first eight decodes against independent HF cached suffixes, retaining the tolerance. Original M1–M3 acceptance checks remain unchanged, tiny-model packing logits are checked throughout, and all three public 50-token results remain exactly equal to independent engine/HF generation. This narrows the additional numerical gate; it does not promise every cached/full-prefix logit agrees over arbitrarily long FP32 histories.
- **Honest comparator:** static FIFO cohorts use their largest budget and compute 168 excess new tokens; continuous scheduling computes only the requested 216. Useful throughput excludes the excess in both stages. Whole-request completion latency starts at common submission; it is not M2 decode latency.
- **Observed CPU result:** the final isolated Apple M5 run measured 37.36 useful tokens/s static and 69.52 continuous (1.86×), one warmup and three repetitions, one thread, eight requests and active limit two. Peak K/V was 27.00 MiB versus 45.98 MiB: private-plus-temporary storage makes this reference scheduler more memory hungry. A preliminary run overlapping verification was discarded. CPU process peak and all CUDA performance remain unmeasured; M2 CSV is unchanged.
- **Verification:** 296 tests passed, three CUDA hardware checks skipped. The real CLI with budgets 5/50 matched independent cached generation. Benchmark metric checks cover actual byte scopes, excess-token accounting, cohort completion timestamps, invalid workloads and CSV parsing.

Execution stayed native with the project-local manual ledger. The additional public suffix-oracle decision is documented in the M4 plan; cost is reduced long-history elementwise coverage, while exact generated-token and original baseline gates remain mandatory. Completed results remain owned until the scheduler is discarded; server eviction, cancellation, paging and concurrent ownership are future work.

### Milestone 4 final review

The fresh independent reviewer reran all 296 tests (three CUDA skips), verified whitespace, and separately confirmed identical HF cached/full-prefix 50-token generation for the 131-token long prompt. Scheduler lifecycle, recovery, cache accounting, CLI and measured scope matched the spec. The reviewer accepted the documented reduced additional suffix-logit gate; it does not establish full-history elementwise parity.

One finding was initially graded minor because normal FP32 GPT-2 logits cannot trigger it: finite FP64 logits `[1e308,-1e308]` with temperature `1e308` overflowed during centering, making a candidate with about 12% probability impossible. Re-graded important because the sampler explicitly accepts finite floating vectors and that temperature. The new probability-oracle test failed with a 0.1192 probability difference before the fix. Scaling first for temperatures at least one resolves it; smaller temperatures retain centering first. No findings are deferred. The final acceptance suite has 297 passing tests and three CUDA skips.

## Milestone 5 — paged KV cache, October 6, 2026

- Full-budget reservation before prefill avoids growth deadlock. Waiting requests consume no pages; FIFO pressure may leave usable smaller holes idle. Atomic reservations and staged cleanup are checked separately from model-forward recovery.
- Nonconsecutive IDs and page-boundary chunks require logical indexing. Poisoned unused tails stay invisible; all 50 direct paged suffix logits match contiguous cached execution at the original tolerance. The inherited M4 additional HF suffix gate remains unchanged.
- Physical pool residency survives request completion. Owned reservation, committed use, rounding and unused logical capacity are separate quantities. Weak references check that gather pairs die before another read/forward; scheduler peak counts pool plus workspace plus one gather pair.
- The benchmark cleanup initially used an unnecessary type check that failed when the fault-injection test replaced the constructor. Cleanup now uses the already-known experiment mode and explicitly drops the last loop reference before the next stage.
- The isolated Apple M5 allocation run uses 28 FP32 pages of 16 tokens (31.5 MiB effective from 32 MiB requested). Repeated 17-token capacities fit 26 contiguous requests versus 14 paged: 14.77 MiB is page rounding. Paging does not reduce full-budget reservation waste here.
- After freeing alternate page-sized reservations, both traces have 224 free slots with largest hole 16. A 32-slot probe fails in contiguous first-fit metadata and succeeds with physical pages 0 and 2. The comparator describes arena metadata, not measured host/CUDA allocator fragmentation. Timing and process peak remain unmeasured; M2/M4 measurements are unchanged.
- CUDA checks remain conditional and unverified on this CPU-only host. No custom attention kernel or extra dependency was added.

M5 pre-review verification: 406 tests passed, five CUDA hardware checks skipped. Real paged CLI output matches contiguous generation for budgets 5/50. Benchmark tests cover byte arithmetic, trace replay, schema scope, early validation and cleanup faults.

### Milestone 5 final review

The fresh read-only reviewer found no critical, important or minor findings and independently reran 406 tests (five CUDA skips). All four allocation rows reproduced except sandbox hardware detection (`arm` versus the original escalated `Apple M5` detection). Whitespace passed, and M2/M4 CSVs were unchanged. No fix pass or deferred findings were needed. Execution retained the project-local manual ledger; its cost is manual upkeep. M5 is published as draft [PR #3](https://github.com/harshith49/mini-infer/pull/3), stacked on unmerged M4.

## Milestone 6 — int8 transformer weights, October 6, 2026

Per-output-row symmetric int8 conversion replaces the 48 transformer projections. Embeddings and their tied output head, biases and norms stay FP32; cache storage is unchanged. Conversion stages replacements before attaching them, so a late invalid source does not leave a mixed model. An all-zero row uses scale one, subnormal rows use a positive scale floor, and finite extremes are checked for FP32 reconstruction overflow. Forward reconstruction scales in place to avoid a second full floating weight temporary.

The fixed public sample is the first 4,097 GPT-2 tokens from pinned Tiny Shakespeare text. Runtime download and cache reads verify its source checksum; the evaluator scores exactly 4,096 targets with context 1,024/stride 512. Independent window-oracle tests caught neither overlap double-counting nor target omissions. The source and token hashes accompany measurements; this small slice is not a general quality benchmark.

The first public run exposed an incorrect assumption in the proposed raw-logit acceptance budgets: max absolute errors were 2.1811, 1.0604 and 8.1086, with RMSE 0.9690,0.4411 and 2.1856. A separate ordinary-Linear model using independently reconstructed int8 weights matched the quantized model exactly. Restoring original floating weights restored exact baseline logits. Most error was a common per-token vocabulary offset, which cancels in softmax. After removing that offset, maxima were 0.5505,0.3846,0.9081 and RMSE 0.07275,0.06000,0.06145. Mean distribution KL was 0.001304,0.001128,0.000835. Public cached and paged 50-token quantized outputs matched independent quantized generation.

The same fixed text slice measured FP32 NLL 4.264483484/PPL 71.128171649 versus int8 NLL 4.250111963/PPL 70.113261975, a 1.4269% perplexity decrease. This passes the planned 5% increase gate. On October 7, 2026, the user approved applying the unchanged 2.0/.25 logit bounds after subtracting each token's vocabulary mean. Raw errors remain reported; additive offsets no longer fail the quantization-only gate. Original FP32 correctness gates have not changed.


The actual isolated CPU benchmark records 497,759,232 FP32 weight bytes versus 243,287,040 int8-stage weight bytes, a 51.1236% reduction. Total model tensor bytes include another 12,582,912 bytes of causal masks; the largest reconstruction is 9,437,184 bytes. At 128 prompt tokens plus 32 outputs, throughput drops 89.02 to 24.06 tokens/s. Across tested lengths, int8 reaches 0.26–0.37× FP32 throughput. This reference trades speed for storage; no fused kernel or GPU speedup is claimed. The same single owned model is measured before/after conversion, and prior milestone CSVs remain unchanged.

Before the approved amendment, integration verification reported 477 passed, 7 CUDA skips and 3 failing raw-logit acceptance tests (one for each public prompt). Real int8 single, static-batch and paged-scheduler CLI demonstrations succeeded. After the approved amendment, the full suite passed 480 tests with seven CUDA skips in 127 seconds. An additional int8 seeded-sampling/stops/global-RNG integration check passed separately before final review.


### Milestone 6 final review and publication — October 7, 2026

The fresh read-only reviewer independently ran **481 tests successfully, with seven CUDA hardware skips**, in 122.82 seconds. It found no Critical or Important issues, confirmed the eight CSV rows and approved comparison scope, and verified unchanged earlier CSVs and clean whitespace. M6 is published as draft [PR #4](https://github.com/harshith49/mini-infer/pull/4), stacked on unmerged M5.

Two Minor findings are deferred:

- A malformed programmatic projection bias length is not validated before conversion. Standard checkpoint loading checks shapes; an already-invalid custom model can be converted before its forward raises a dimension error.
- Finite NLL as large as 1,000 overflows `math.exp`, producing an `OverflowError` traceback instead of the intended `ValueError`. The recorded GPT-2 results are unaffected.

Execution rulings: retain the existing checkout/manual ledger (manual upkeep); progress independent benchmark work while awaiting the criterion decision (task-order bookkeeping); use the approved centered quantization gate (additive raw offsets are accepted, with raw errors retained). CUDA stays unverified because hardware is absent (device-specific issues may remain). Training and dtype-changing mixed precision remain outside this FP32 inference contract (they need a separate implementation). Allocation-failure recovery during conversion has no atomicity guarantee (reload a fresh FP32 model after such a failure). No blocking fix pass or second review was needed.

## Milestone 7 — streaming serving, October 10, 2026

One background thread owns the model, tokenizer and Scheduler. The HTTP loop only validates requests and exchanges bounded commands/events. Explicit Scheduler cancellation/disposal releases waiting/running/completed IDs, cached pages and pending phase events while preserving ordinary retained results for existing callers. Token events include empty deltas when GPT-2 byte tokens split Unicode; deferred trailing replacements are flushed at completion and concatenated deltas match final tokenizer decoding.

The first lifecycle tests caught validation rejection returning before cleanup acknowledgment. Ordinary errors now await disposal; cancelled HTTP tasks leave a protected cleanup task holding the stream slot until the worker acknowledges. Full-buffer and rapid-cancellation checks establish finite retained response/command state. A response-level finally wrapper also disposes a handle when header/send failure prevents the generator from starting. Unexpected worker faults fail closed, drain queued admissions and require restart.

TestClient proves schema and complete protocol content but buffers responses. Actual ephemeral-port Uvicorn/HTTPX tests prove first-token delivery before completion, another request admitted during blocked inference, actual multi-request forwards, disconnect before/after headers, capacity/failure status, restored pages and shutdown of active streams. The supported CLI stops the worker before Uvicorn drains open responses. Real Ctrl+C initially completed resource cleanup but printed asyncio's KeyboardInterrupt traceback; a failing CLI regression pinned the normal-interrupt boundary, and only KeyboardInterrupt is now caught after shutdown.

The client load test checks IDs, deltas, token lists, usage, a single complete terminal event and EOF. Missing/duplicate terminal events, malformed JSON, duplicate measured request identities, timeouts and HTTP/error streams fail the run without writing a successful CSV row. Mock response IDs initially read a counter after yielding and duplicated IDs; assigning identity at request entry fixed the fixture without weakening production duplicate rejection. Warmup is excluded; throughput uses all useful tokens over first launch through last completion, and percentile interpolation is explicit.

Isolated Apple M5 CPU measurements used one thread, FP32, contiguous KV, active batch limit two and eight 32-output requests. Concurrency one: 128.2515 tokens/s, request p50/p95 247.883/253.071 ms, TTFT 16.131/16.724 ms. Concurrency four: 114.3654 tokens/s, request 1,115.168/1,130.251 ms, TTFT 582.066/593.846 ms. Both delivered 256 useful tokens with zero failures. Actual lifetime forward-batch peaks were one and two. These single-run HTTP smoke measurements show batching, not a CPU speedup; they are not comparable to engine-only workloads or repeated benchmark medians. No CPU process peak/CUDA performance is claimed, and earlier CSVs are unchanged.

Execution rulings so far: retain the approved native checkout/manual ledger instead of wrappers (manual bookkeeping); add the minimal response-level ownership wrapper for failures before generator startup (one small class); retain HTTPX despite Starlette's migration warning because the pinned combination passes real behavior checks (future Starlette updates may need HTTPX2 migration). FastAPI/Starlette/Pydantic/Uvicorn/HTTPX exact versions and `pip check` passed. A permanently stuck backend forward remains a process-termination limit; default 64 outstanding streams and token-context bounds constrain response state, not all possible production deployment resource limits.

M7 pre-review verification: **553 passed, seven CUDA hardware skips**, one pinned Starlette/HTTPX deprecation warning, in 125.42 seconds. `pip check`, compileall and whitespace checks passed. Real public-model curl emitted three token events and one validated terminal result. After the interrupt fix, a second actual startup/Ctrl+C shutdown exited zero without a traceback. Prior CSVs are byte-for-byte unchanged relative to M6.
