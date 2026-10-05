# mini-infer Milestone 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans for the retained native execution method. Steps use checkbox syntax for tracking.

**Goal:** Add a correct contiguous KV cache and publish measured naive-versus-cached CPU results.

**Architecture:** Each request owns fixed-capacity key/value tensors; the model appends one chunk at a time and commits cache length once per successful forward. The uncached path stays available. Generation exposes optional step-duration collection solely for the benchmark; no scheduler or cache registry is added.

**Tech Stack:** Existing Python 3.11, PyTorch 2.5.1, transformers 4.48.3, pytest 8.3.5, safetensors 0.8.0; standard-library CSV, timing, statistics, and hardware metadata. No new dependencies.

**Spec:** [Approved Milestone 2 design](../specs/2026-10-05-mini-infer-m2-design.md).

## Global Constraints

- Work in `.` on the existing feature branch; preserve unrelated files.
- Rerun every existing test unchanged in meaning, including FP32 public GPT-2 logits at `atol=1e-4, rtol=1e-4` and three exact 50-new-token HF comparisons.
- Fixed-capacity contiguous request cache; no new dependency or cache abstraction framework.
- Allocate on the model's device in the model's dtype, with capacity equal to prompt length plus requested output budget.
- Cache tensors are request state, not model parameters or persistent checkpoint buffers.
- A failed forward must not advance the committed length; never read uncommitted tensor slots.
- Keep `generate(..., use_cache=False)` available as the baseline; cached execution is opt-in.
- Full context/output-budget validation, zero-token identity, EOS inclusion, and original CLI prompt preservation remain required.
- Keep commits small, push each verified milestone, and never include weights, local environments, or credentials.
- No automatic merge or force push; later milestones remain out of scope.

## Review Focus

- Cache/model shape or dtype mismatch must reject before writes or length changes (Task 1).
- A failure in a late transformer layer must leave the committed prefix reusable (Task 1).
- Nonzero cache offsets and multi-token decode chunks must obey absolute-position causal masking (Task 1).
- Zero/one output token must avoid unnecessary forwards and report absent decode latency honestly (Tasks 2 and 3).
- CPU timing and cache bytes must never be presented as GPU speed or process peak memory (Task 3).

## Task 1: Cache storage and model integration

**Files:** Create `engine/kv_cache.py`, `tests/test_kv_cache.py`; modify `engine/model.py`.

**Interfaces:**
- `SimpleKVCache(config: ModelConfig, *, batch_size: int, capacity: int, device: torch.device, dtype: torch.dtype)`: `keys` and `values` have shape `[layers, batch, heads, capacity, head_dim]`; `length: int` starts at zero. Read-only `allocated_bytes` and `used_bytes` derive from actual tensor sizes and committed length. Require positive batch/capacity and capacity at most model context length.
- `validate(config: ModelConfig, input_ids: torch.Tensor, *, device: torch.device, dtype: torch.dtype) -> None`: check dimensions, batch, device/dtype, and available slots before writes.
- `write(layer_idx: int, key: torch.Tensor, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]`: write the chunk at the current committed offset, return layer views ending at offset plus chunk length. Do not advance length here.
- `GPT2Model.forward(input_ids: torch.Tensor, *, cache: SimpleKVCache | None = None) -> torch.Tensor`; transformer blocks and attention receive the layer's index, offset, and optional cache. Retain existing return shape and checkpoint parameter names.

- [x] Write failing tiny-model tests: allocate FP32 `[2,1,4,16,6]` key/value tensors; assert reserved bytes `2*2*1*4*16*6*4` and zero used bytes. Prefill three tokens, assert `length == 3`, exact used bytes, and unchanged allocation. Compare concatenated cached chunk logits to uncached logits at `atol=1e-4, rtol=1e-4` for chunk partitions `[1,1,1,1]`, `[2,2]`, and `[1,3]`.
- [x] Add independent-request tests, exact-context-capacity checks, capacity/shape/batch/dtype rejection, and an injected late-layer exception. Assert rejected/failed calls leave `length` and committed tensor prefixes unchanged. Retry after the injected failure and compare logits to uncached forward. Use real tiny forwards; mock only the injected fault.
- [x] Run `.venv/bin/python -m pytest tests/test_kv_cache.py -q`; expect failure from missing cache/interface.
- [x] Implement contiguous allocation and validated writes. Offset embeddings by `cache.length`; use causal rows `past:past+new_length` and key columns through `past+new_length`. Compute logits before committing length, so a failed projection also leaves it unchanged. Validate before entering blocks. Keep uncached tensor operations unchanged.
- [x] Run `HF_HUB_OFFLINE=1 .venv/bin/python -m pytest -q`; require the new cache checks and every existing CPU test to pass (CUDA hardware skip remains conditional).
- [x] Commit with `feat: add contiguous KV cache and offset-aware attention`.

## Task 2: Cached generation, CLI, and real GPT-2 parity

**Files:** Modify `engine/generate.py`, `tests/test_generation.py`, `tests/test_correctness_vs_hf.py`; extend `tests/test_kv_cache.py` if needed.

**Interfaces:**
- Extend `generate(model, input_ids, max_new_tokens, *, eos_token_id=None, use_cache: bool=False, step_times: list[float] | None=None) -> torch.Tensor`. Existing positional calls remain compatible.
- `step_times`, when supplied, is empty at entry and receives one elapsed duration per produced token. Its first entry includes request validation, cache allocation, prefill, sampling, and append; later entries measure complete decode steps. Use `time.perf_counter()` and synchronize the model's CUDA device before/after timed intervals only when collecting metrics. No timing/synchronization overhead on the ordinary path.
- Add CLI `--use-cache` flag; retain all previous arguments and output semantics.

- [x] Parameterize generation boundary checks over cached and uncached modes. Add observed-forward-length test: prompt length 3, output count 4 gives cached forward lengths `[3,1,1,1]`; uncached gives `[3,4,5,6]`. Zero output produces no forwards and no cache allocation; EOS ends after the first selected terminal token.
- [x] Add public GPT-2 tests for all three existing prompts: cached output for 50 new tokens equals uncached output and HF's output exactly. Compare public-model cached token and multi-token suffix logits against full-prefix logits at `atol=1e-4, rtol=1e-4`. Keep existing baseline tests meaningful and unchanged in tolerance.
- [x] Add timing tests on a tiny real model: zero output leaves the metrics list empty; one output yields one nonnegative duration; multiple outputs yield the requested number. Recheck cached CLI empty-prompt seeding and literal-special-token preservation using existing CLI tests parameterized with `--use-cache`.
- [x] Run targeted generation/cache/reference tests; expect failures from missing `use_cache`/metrics behavior before implementation.
- [x] Implement one prompt prefill followed by one-token decode. Forward the previously generated token only if another output token is needed; never feed the final sampled token unnecessarily. Allocate from model dimensions/device/dtype; collect timings only when requested. Leave the default generation path uncached.
- [x] Run the full suite, then `HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 .venv/bin/python -m engine.generate --prompt 'Hello, world!' --max-new-tokens 50 --device cpu --use-cache`; require public-model parity and a successful real CPU continuation.
- [x] Commit with `feat: add cached prefill and single-token decoding`.

## Task 3: Honest measurements and milestone release

**Files:** Create `benchmarks/bench_stages.py`, `tests/test_benchmarks.py`, `results/kv_cache.csv`; update `README.md`, `docs/architecture.md`, `docs/lessons.md`, and this plan's completed checkboxes.

**Interfaces:**
- `run_workload(model: GPT2Model, input_ids: torch.Tensor, *, max_new_tokens: int, repetitions: int, use_cache: bool) -> dict[str, object]`: one warmup, at least three repetitions, metrics from Task 2, and fixed-length output with EOS stopping disabled. Summarize end-to-end time using the median; throughput is output count divided by median time. Compute p50/p95 over measured decode steps excluding each repetition's first token. Mark absent decode samples as an empty CSV cell.
- Benchmark CLI: `--device` (auto), `--prompt-lengths` (16 64 128 256), `--max-new-tokens` (32), `--repetitions` (3), `--threads` (1), `--output` (`results/kv_cache.csv`). Seed 0; build a deterministic synthetic GPT-2 token-ID workload and record this fact rather than label it natural-language traffic.
- CSV columns: `stage`, `device`, `hardware`, `torch_version`, `threads`, `prompt_tokens`, `generated_tokens`, `repetitions`, `median_seconds`, `tokens_per_second`, `ttft_seconds`, `decode_p50_ms`, `decode_p95_ms`, `cache_allocated_bytes`, `peak_memory_bytes`, `peak_memory_kind`, `workload`. CPU peak memory stays empty with kind `unmeasured`; CUDA may use synchronized/reset PyTorch allocated-memory counters and must name that scope.

- [x] Write failing tiny-model benchmark tests that reject invalid repetitions/output lengths, verify generated counts and stage labels, leave one-token decode percentiles absent, leave CPU peak memory unmeasured, and report literal expected contiguous-cache bytes. Verify CSV roundtrip preserves these distinctions. Do not assert wall-clock speedups or exact timing values.
- [x] Run `.venv/bin/python -m pytest tests/test_benchmarks.py -q`; expect missing benchmark implementation.
- [x] Implement benchmark and CSV writing with standard-library tools. Before timing both stages, assert token equality on the same input/output workload. Reject prompt/output lengths beyond context capacity, nonpositive output/thread counts, repetitions below three, or missing prompt lengths. Record actual device/hardware and CPU thread count; keep allocation included in total request timing.
- [x] Run `HF_HUB_OFFLINE=1 .venv/bin/python -m benchmarks.bench_stages --device cpu`; inspect all eight CSV rows (two stages by four lengths) and retain measured results regardless of which stage wins. Check output parity before accepting any row.
- [x] Update README with measured CPU results and commands; show prefill/decode in Mermaid; document reserved versus used cache memory, actual bugs, and limitations. Clearly retain unmeasured GPU placeholders. Do not imply synthetic workload results predict production traffic.
- [x] Run the full suite, cached CLI demo, `python -m compileall -q engine benchmarks`, and `git diff --check`. Request one independent read-only final review under the retained native workflow. Fix important findings with reproducing tests and a green full suite; record any deferrals honestly.
- [x] Commit with `bench: measure naive versus cached GPT-2 on CPU`, push the verified milestone to `origin/codex/mini-infer-m1`, and confirm local/remote HEAD match. No merge or force push.

## Plan self-review

Cache validation/transactional length and offset masking belong to Task 1; generation semantics, public-model parity, and instrumentation belong to Task 2; benchmark honesty and release belong to Task 3. Interface names and tensor layouts match the approved spec. All five review conditions have owning tests, default uncached behavior remains preserved, and no future subsystem is introduced.
