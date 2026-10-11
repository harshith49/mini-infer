# mini-infer M8 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Native inline execution is the preserved user preference; one fresh whole-branch reviewer runs at the end.

**Goal:** Complete six-stage comparable measurements, reference comparisons, charts and reproducible project packaging.

**Architecture:** Reuse the existing generation/scheduler/cache/quantization implementations. A small shared benchmark module owns workload validation, stage execution, uniform measurements and complete CSV publication; standalone comparison and chart scripts consume it. Packaging exposes the existing engine/server without changing inference behavior.

**Tech Stack:** Python 3.11, existing pinned PyTorch/Transformers/FastAPI toolchain, stdlib CSV/JSON/statistics/tempfile, verified pinned Matplotlib. vLLM remains optional in a separate environment.

**Spec:** `docs/superpowers/specs/2026-10-11-mini-infer-m8-design.md` (approved October 11, 2026).

## Global Constraints

- Work only in the standalone `harshith49/mini-infer` repository on `codex/mini-infer-m8`, starting from M7 `27383e7`; retain the approved checkout and native execution.
- Preserve engine APIs, public GPT-2 FP32 `atol=1e-4` gates, exact 50-token greedy tests and all pre-M8 CSVs. HF execution is permitted only in comparison/test code.
- Default prompt lengths `[16,64,32,128,16,64,32,128]`, budgets `[4,32,8,64,4,32,8,64]`, seed 0, batch limit 2, one warmup, three measured repetitions, one CPU thread, paged pool 32 × 16 tokens; 216 useful outputs.
- Fixed-budget greedy generation disables EOS. Stage order: `naive`, `kv_cache`, `static_batching`, `continuous_batching`, `paged_kv`, `int8`; final stage uses paged KV plus converted weights.
- Report whole-request completion latency from common submission, linear-interpolated p50/p95 and useful tokens divided by median workload elapsed time. Exclude loading/tokenization/conversion/quality/warmup/post-run comparisons.
- CPU process peak is blank/`unmeasured`; CUDA peak counts allocated tensors including the active model, excluding allocator reservations. KV/model storage has separate explicit scopes.
- Int8 retains the existing checksum-pinned Tiny Shakespeare scoring and maximum 5% perplexity-increase gate. No quantized token-parity or speedup promise.
- No secrets, hosted APIs, committed weights, new inference architecture, new serving behavior, production features, PyPI release, merge or release tag.
- Matplotlib version and installed API behavior must be verified before pinning. Keep missing-GPU/Docker/vLLM evidence separate from executed CPU verification.

## Review Focus

1. Failed late-stage runs and interrupted file writes must preserve the previous complete result; Task 1 tests atomic publication and cleanup.
2. Padding and differing output budgets must not inflate useful throughput or alter trimmed tokens; Task 1 tests deliberately unequal cohorts.
3. Warmup/reference outputs and unused models must not contaminate CUDA resident-memory scope; Tasks 1–2 keep parity tensors on CPU and test model lifetimes, with conditional CUDA checks.
4. Optional vLLM runtime/API faults must not be hidden as dependency skips; Task 2 distinguishes absence from failures.
5. Missing memory and incompatible/invalid CSVs must not yield misleading charts; Task 3 rejects malformed rows and annotates unmeasured values.

## Files and interfaces

- Create `benchmarks/stage_suite.py`: validated `Workload` dataclass, `RunResult`, common timing/CSV helpers and six-stage runner. Modify `benchmarks/bench_stages.py` only to route `--suite all`; retain `run_workload` and legacy mode.
- Create `benchmarks/bench_vs_hf.py`, `benchmarks/bench_vs_vllm.py`: explicit reference boundaries, same workload and metric schema.
- Create `benchmarks/make_charts.py`; modify `requirements.txt` with verified Matplotlib pin.
- Create `Dockerfile`, `.dockerignore`, `.env.example`, `.github/workflows/tests.yml`, `notebooks/colab_benchmark.ipynb`.
- Create focused tests `tests/test_stage_suite.py`, `tests/test_reference_benchmarks.py`, `tests/test_charts.py`; retain existing tests. Modify README, architecture and lessons; create `results/stages.csv`, `results/vs_hf.csv`.

Shared names/signatures (later tasks import these rather than invent equivalents):

- `Workload(prompt_lengths: tuple[int, ...], output_budgets: tuple[int, ...], max_batch_size: int = 2, repetitions: int = 3, seed: int = 0, page_size: int = 16, num_pages: int = 32)`; frozen dataclass with positive/nonempty/length validation.
- `RunResult(outputs: list[torch.Tensor], completions: list[float], elapsed: float, peak_kv_bytes: int | None, peak_memory_bytes: int | None)`; output tensors returned on the execution device, CPU parity copies made after timing.
- `make_prompts(config: ModelConfig, workload: Workload, device: torch.device) -> list[torch.Tensor]`; validate individual/static-cohort context and paged feasibility, use a CPU seed-zero generator.
- `run_stage_once(model: GPT2Model, prompts: list[torch.Tensor], workload: Workload, stage: str) -> RunResult`.
- `measure(run_once: Callable[[], RunResult], expected: list[torch.Tensor], workload: Workload) -> dict[str, object]`; one warmup and repeated deterministic/count checks, with expected tensors held on CPU.
- `metadata(model: torch.nn.Module, prompts: list[torch.Tensor], workload: Workload, stage: str, device: torch.device) -> dict[str, object]`; uniform hardware/version/workload/dtype/cache/storage fields.
- `write_rows(path: Path, rows: list[dict[str, object]]) -> None`; validate compatible row keys, serialize to a sibling temporary file and atomically replace only on success.
- `run_suite(model: GPT2Model, tokenizer, workload: Workload, *, cache_dir: str = 'model_cache') -> list[dict[str, object]]`; fresh FP32 eval model, validate/quality-score outside timing, convert only after five FP32 stages, return six complete rows.

## Task 1: Six-stage runner and preserved legacy entry point

**Files:** `benchmarks/stage_suite.py`, `benchmarks/bench_stages.py`, `tests/test_stage_suite.py`, `docs/lessons.md`.

**Consumes:** `generate`, `generate_batch`, `Scheduler`, `PagePool`, `quantize_model`, `model_storage_bytes`; quality loader/scorer/constants from `bench_quantization`.
**Produces:** all shared interfaces above, `python -m benchmarks.bench_stages --suite all` and six-row schema. Legacy invocation continues writing `results/kv_cache.csv`; all-suite default is `results/stages.csv`. An explicit output overrides either default.

- [x] Write RED tests using tiny real GPT-2: five FP32 stages equal independent generation; budgets `[1,3,2]` with batch 2 produce six useful tokens and two excess static tokens; scheduler/page state closes on success and injected failure; int8 equals its own warmup. Test invalid booleans/nonpositive integers/list mismatch/context overflow/impossible pool and reused already-quantized model. Mock only public quality text/scoring for tiny-model tests, retaining a real independent public quality acceptance later.
- [x] Add exact metric assertions with controlled observations: median elapsed `[1,2,3]` is 2, six useful tokens give 3 tokens/s, completions `[1,2,3,4]` have p50 2.5 and p95 3.85. CPU memory is `None`/`unmeasured`, never weight/KV bytes. Inject a late mismatch and serialization failure; existing output file stays byte-identical and temporary files disappear. Conditional CUDA checks verify synchronization/reset and no retained unused reference/warmup outputs.
- [x] Run `.venv/bin/python -m pytest tests/test_stage_suite.py -q`; expect missing-module/API failures. Record RED before implementation.
- [x] Implement the shared interfaces with explicit six-stage branches. Sequential timing observes request returns; static timing observes cohort returns and trims budgets; scheduler timing observes terminal events and closes in `finally`. Hooks inspect actual contiguous cache bytes; scheduler counters cover workspace/pool scope. Reset/read CUDA allocated peaks within each run. Keep expected/warmup IDs on CPU between repetitions and release GPU result tensors before the next repetition. Use unique actual model storages for metadata, not parameter-count estimates.
- [x] Implement the suite's quality gate using 4,096 scored targets, context 1,024, stride 512 and existing source/token checksums. Score FP32 before conversion and int8 after; include FP32/int8 NLL/perplexity/delta/source metadata in rows with consistent keys. Fail visibly on corrupt/unavailable text or a >5% increase. Import shared code lazily from all-suite dispatch to avoid circular imports through legacy `bench_stages.run_workload`.
- [x] Add all-suite CLI flags with separate defaults while preserving legacy `--prompt-lengths`, `--max-new-tokens`, output and `run_workload` behavior. Validate before loading where possible, and publish only after all six rows succeed.
- [x] Run targeted tests plus `tests/test_benchmarks.py`, scheduler/paged/quantization benchmark tests; expect GREEN. Verify both CLI help paths and existing CSVs untouched. Commit `feat: benchmark six engine stages on one workload`.

## Task 2: HF pairing and optional vLLM boundary

**Files:** `benchmarks/bench_vs_hf.py`, `benchmarks/bench_vs_vllm.py`, `tests/test_reference_benchmarks.py`, lessons.

**Consumes:** Task 1 workload, prompts, `RunResult`, `measure`, metadata and atomic writer.
**Produces:** `run_hf_once(reference: GPT2LMHeadModel, prompts: list[torch.Tensor], workload: Workload) -> RunResult`; paired HF CLI output with stages `mini_infer_cached`/`hf_cached`; optional-vLLM CLI with explicit skip status and no CSV on skip.

- [x] Write RED tests against a tiny real HF/custom checkpoint: cached HF output equals custom output for differing budgets even when generation config contains EOS IDs. Assert exact output lengths, identical sequential completion semantics and same workload metadata. Verify parity failures preserve old output and each implementation is unloaded before the other measured implementation loads (CPU expected IDs may remain). Test vLLM absence/no-CUDA skip without file writes, but injected installed-engine API/runtime failures propagate.
- [x] Run `.venv/bin/python -m pytest tests/test_reference_benchmarks.py -q`; expect missing APIs. Inspect installed Transformers generation behavior with a runnable probe and consult official reference APIs as needed.
- [x] Implement HF eager FP32 eval loading/generation with `do_sample=False`, `use_cache=True`, explicit no-EOS generation configuration and all-ones mask. Compute paired CPU-held expected IDs outside timers, then unload each model before timing its counterpart; never retain both on CUDA. Use Task 1 measurement definitions and two-row atomic publication to `results/vs_hf.csv`.
- [x] Verify official vLLM token-prompt/sampling APIs for the documented optional version. Validate CLI inputs before skip handling. Detect dependency absence and unavailable CUDA only; propagate import-time broken-install errors and runtime/API faults. Submit all token prompts with per-request fixed budgets, collect outputs and whole-request completion observations with supported installed APIs; label dtype, eager/runtime configuration and observation scope. Do not invent unavailable per-request latency or memory: unsupported measurements are blank with explicit scope. No vLLM dependency in main requirements.
- [x] Run reference tests and actual optional script on this host; expected skip/no file if unavailable. Commit `bench: add HF and optional vLLM comparisons`.

## Task 3: Validated headless charts

**Files:** `benchmarks/make_charts.py`, `requirements.txt`, `tests/test_charts.py`.

**Consumes:** the six-row Task 1 schema. **Produces:** `make_charts(csv_path: Path, output_dir: Path) -> list[Path]` and a CLI defaulting to `results/stages.csv`/`results/`, generating throughput, latency and memory PNGs.

- [x] Write RED tests with six valid tiny rows and invalid variants: duplicate/missing stage, nonfinite/negative metric, mismatched hardware/workload/repetitions and malformed memory kind/value. Assert expected files are nonempty PNGs, p95 is never below p50, blank memory is explicitly annotated as unmeasured, and invalid input produces no charts. Dtype/cache backend are stage-specific and may differ as specified; shared experiment fields must match.
- [x] Run `.venv/bin/python -m pytest tests/test_charts.py -q`; expect missing module. Check official Matplotlib/PyPI metadata for a Python 3.11-compatible exact pin, install it, run `pip check` and a headless rendering probe before recording the pin.
- [x] Implement validation before rendering, use Matplotlib Agg, readable units and captions carrying hardware/device/workload/repetitions and int8 dtype scope. Annotate missing memory rather than zero-valued bars; close figures. Existing PNG ignore remains.
- [x] Run tests and visually inspect rendered charts, correcting clipped labels/captions. Commit `bench: render comparable stage measurements`.

## Task 4: Docker, CI, notebook and runnable documentation

**Files:** Dockerfile, `.dockerignore`, `.env.example`, `.github/workflows/tests.yml`, `notebooks/colab_benchmark.ipynb`, README, architecture, lessons.

**Consumes:** working commands from Tasks 1–3 and existing `python -m server.app`/generation CLIs. **Produces:** reproducible CPU recipe and GPU notebook instructions, no new engine interfaces.

- [x] Inspect available Docker/runtime tools, existing model cache and installed dependencies. Create minimal Python 3.11 image with cached dependency-install layer and server module entry command binding `0.0.0.0:8000`, CPU device and one thread; allow generation command override. Exclude Git, venv, caches, secrets, weights and scratch from context. Explain persistent `model_cache` mount; no build-time download. `.env.example` has only commented optional `HF_TOKEN`, unnecessary for GPT-2.
- [x] Create CI workflow for push/PR with read-only repository permissions, Python 3.11, pinned requirements and full CPU pytest. Preserve public weights tests and loopback tests, no secrets. If caching model downloads, include dependency hash in key. Verify action versions using official sources before selecting them.
- [x] Create notebook with editable `REVISION='codex/mini-infer-m8'`, clone URL `https://github.com/harshith49/mini-infer.git`, dependency install, actual GPU inspection/guard, full tests then stage/HF/charts, viewing/downloading results. Use checked subprocess execution so failure prevents later benchmarks. Keep outputs/execution counts empty; document Colab Python/runtime compatibility and separate optional vLLM environment. Provide explicitly labeled CPU equivalent.
- [x] Parse notebook with stdlib JSON and compile code cells; check source paths/commands against live CLIs and verify no outputs/secrets. If Docker daemon is available, build/run CPU generation and a server/curl smoke with finite timeout; otherwise record unverified container execution. Local tests do not imply hosted CI success or T4 execution.
- [x] Update README with required exact opening sentence, three-command CPU quick start, all commands, optimization costs, existing measured tables with scope, new result placeholders pending Task 5 and interview concepts. Check existing Mermaid lifecycle/scheduler/page diagrams and add only missing coverage. Commit `build: add CPU container CI and Colab workflow`.

## Task 5: Isolated measurements, acceptance review and publication

**Files:** `results/stages.csv`, `results/vs_hf.csv`, README, architecture, lessons, this plan; ignored `.superpowers/m8` ledger/review package.

- [x] Verify clean M7 baseline before executing Task 1: full offline suite with cached public GPT-2, current known baseline 559 passes/seven CUDA skips. Record each task's RED/GREEN evidence and rulings/costs in the retained native manual ledger. Use systematic debugging for failures; never loosen gates. This is an execution preflight even though acceptance is listed here.
- [x] After Tasks 1–4, run the full inherited/new suite and required dependency/compile/whitespace checks. Fix failures before measurements. Then ensure no test/reviewer/other benchmark process is running; run `HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false .venv/bin/python -m benchmarks.bench_stages --suite all --device cpu` and the matching HF command. Require six/two complete rows, exact FP32 parity, deterministic outputs and passing fresh quality gate; record actual hardware, useful counts, repetitions and timings.
- [x] Render charts from fresh CSVs and visually inspect them. Replace README placeholders with actual CPU throughput, whole-request latency, explicit memory blanks and GPU-unmeasured cells. Record fresh quality results and slower stages without positive-speed assumptions. Compare all pre-M8 CSVs byte-for-byte to M7.
- [x] Run full offline pytest after final product changes, compileall for engine/server/benchmarks, `pip check`, `git diff --check`, a cached CPU 50-token CLI demo and notebook CPU-equivalent commands. Verify Docker/Colab/vLLM/hosted-CI limits are accurately disclosed. Commit measured rows/docs as `bench: publish final CPU comparisons and project guide`.
- [x] Request one fresh independent read-only whole-M8 reviewer against M7/spec/plan using the required review skill. Provide exact base/head, test/measurement evidence and all five Review Focus entries. No per-task or repeated independent review. Reproduce Critical/Important findings with RED tests, perform one executor fix pass and rerun affected/full checks; record declined-to-judge decisions and deferred Minor costs in lessons/ledger.
- [x] Verify origin `harshith49/mini-infer`, M7 PR state and base branch. Push authorized M8 commits, create/attach stacked draft PR, update repository description only if needed. Record publication in plan/lessons, commit/push final docs, verify local/remote/PR head equality and clean checkout. Collect all rulings/deferred costs before removing only M8 scratch. No merge or release tag.
- [x] Deliver PR link, verified results/limitations, exact CPU/server/test/chart/Docker/Colab commands and interview concepts. Distinguish implemented packaging from unexecuted external environments.

## Plan self-review

All spec sections have owners: Task 1 uniform stage/quality/memory metrics and safe output, Task 2 reference/optional boundary, Task 3 charts/dependency, Task 4 packaging/notebook/docs, Task 5 isolated measurements/acceptance/publication. Shared signatures are defined once and reused. Review Focus failures have targeted checks; external CUDA/T4/Docker/vLLM/hosted-CI claims require actual evidence or explicit unverified status. Legacy mode avoids breaking earlier imports/commands, and CPU-held parity tensors avoid contaminating CUDA peaks. No product code or dependency changes occur before plan approval.

Acceptance record: 614 passed/eight CUDA skips/one inherited warning; review findings fixed. Published stacked draft PR #6 against M7, attached to this chat; no merge. Publication and remote-head checks use the authorized native workflow.
