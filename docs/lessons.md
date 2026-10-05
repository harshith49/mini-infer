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
