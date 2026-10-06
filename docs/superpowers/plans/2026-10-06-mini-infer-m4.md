# mini-infer Milestone 4 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans for the retained native execution method. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Replace completed requests between batched decode steps while preserving independent per-request outputs and publishing honest mixed-length measurements.

**Architecture:** A FIFO waiting deque feeds a bounded ordered running set. Admission prefills separately; existing decodes temporarily pack real-token private caches into a masked batch and copy only new K/V columns back. Request-owned generators isolate sampling from other arrivals; the existing single-request/static loops remain independent references.

**Tech Stack:** Existing Python 3.11, PyTorch 2.5.1, transformers 4.48.3, pytest 8.3.5, safetensors 0.8.0; standard-library dataclasses, deque, argparse, JSON, CSV, statistics, and timing. No new dependencies.

**Spec:** [Approved Milestone 4 design](../specs/2026-10-06-mini-infer-m4-design.md).

## Global Constraints

- Work in standalone `harshith49/mini-infer` on `codex/mini-infer-m4`, based on reviewed M3; keep the M3 draft PR separate.
- Publish milestone commits to this remote only; no automatic merge or force push. A later M4 PR targets the M3 branch while that dependency remains unmerged.
- No model downloads, environments, secrets, or unrelated enclosing-project files enter commits.
- Retain native execution with one fresh independent final reviewer.
- Rerun all existing tests without changing their meaning or tolerances; public GPT-2 greedy comparisons remain token-exact and logits use `atol=1e-4, rtol=1e-4`.
- Each request owns a contiguous cache holding only its real tokens; temporary padding never becomes persistent request history.
- Validate submissions before changing scheduler state. Zero outputs require no cache allocation or model forward; newly generated stop IDs are included.
- Seeded sampling is independent of other requests on the same device; do not claim stochastic parity across CPU/CUDA or with HF's random-number consumption.
- A failing model-forward phase does not commit private prefixes, outputs, or sampling state. Earlier successful phase events remain available for delivery without duplication.
- Report actual private plus temporary K/V allocations separately from process peak memory. After all requests finish, scheduler-owned cache bytes must be zero.
- Measure the actual static-versus-continuous result rather than assuming a CPU speedup. Keep the existing M2 CSV unchanged.
- No page pool, cancellation protocol, asynchronous worker, HTTP server, memory-budget admission, chunked prefill, or sampling penalties in this milestone.

## Review Focus

- A very small positive temperature or a tightly filtered distribution must retain a valid candidate and reproducible sampling rather than produce NaNs (Task 1).
- A successful admission followed by failed decode must preserve and later deliver admission events exactly once while allowing retry (Task 2).
- Uninitialized temporary padding must never propagate NaNs through zero-weight attention, and private prefixes must retain only real positions (Task 2).
- Caller prompt mutation and duplicate/completed request IDs must not silently change accepted work; zero-output waiters must not consume a running slot (Task 2).
- Static excess work and cohort-return latency must not be reported as useful generated tokens or per-token decode latency (Task 4).

## Task 1: Validated, request-owned sampling

**Files:** Modify `engine/sampler.py`; create `tests/test_sampler.py`.

**Interfaces:**
- Frozen `SamplingParams(temperature: float=0.0, top_k: int=0, top_p: float=1.0, seed: int=0)` with `validate(vocab_size: int) -> None`: finite nonnegative temperature, integer top-k in `[0, vocab_size]`, finite top-p in `(0, 1]`, and integer seed in `[0, 2**63-1]`. Reject booleans for integer settings to avoid treating flags as counts/seeds.
- `sample(logits: torch.Tensor, params: SamplingParams, *, generator: torch.Generator) -> torch.Tensor`: one rank-1 nonempty floating vector, one scalar token tensor on that device; reject nonfinite raw logits. Preserve `greedy(logits)` unchanged. Temperature zero returns greedy without an RNG draw.
- Positive temperature: center and scale scores in float64 for stability at very small temperatures, apply top-k then top-p, retain the threshold-crossing candidate, and use `torch.multinomial` with the supplied generator. No global seed resets.

- [x] Write `test_greedy_sampling_preserves_generator_state`: tied largest logits choose the lowest token ID; generator state before/after is identical. Add `test_seeded_sampling_matches_probability_oracle` against direct multinomial draws from independently calculated probabilities for an unfiltered three-token vector.

  ```python
  generator = torch.Generator().manual_seed(7)
  before = generator.get_state().clone()
  token = sample(torch.tensor([4., 4., 1.]), SamplingParams(seed=7), generator=generator)
  assert token.item() == 0
  assert torch.equal(generator.get_state(), before)
  ```

- [x] Write filtering tests using probabilities `[0.6,0.3,0.1]`: top-p `0.7` permits IDs `{0,1}`, retaining the threshold-crossing ID; top-k one permits only ID zero; top-p one with top-k zero permits all candidates. Check top-k followed by top-p using known normalized candidate probabilities, and compare repeated draws from equal seed states.
- [x] Write tiny-positive-temperature and extreme-finite-logit tests: argmax remains a valid selectable candidate and all draws succeed. Parameterize invalid temperatures, top-k/top-p/seed bounds/types, empty/rank-2/integer logits, and NaN/infinity raw logits; each raises `ValueError` before drawing randomness.
- [x] Run `HF_HUB_OFFLINE=1 .venv/bin/python -m pytest tests/test_sampler.py -q`; expect missing sampling API failures before implementation.
- [x] Implement the interfaces with tensor operations and one dataclass; validate even in greedy mode. Verify actual CPU generator/multinomial behavior on the pinned runtime.
- [x] Run the sampler tests and full suite; require all existing CPU checks and new sampler assertions to pass, with CUDA skips determined only by hardware.
- [x] Commit with `feat: add seedable temperature top-k and top-p sampling`.

## Task 2: Scheduler, cache packing, and recovery

**Files:** Create `engine/scheduler.py`, `tests/test_scheduler.py`; reuse `engine/model.py` and `engine/kv_cache.py` without changing their contracts.

**Interfaces:**
- Frozen `TokenEvent(request_id: str, token_id: int | None, finish_reason: str | None)`; reasons are `stop`, `length`, or `None`.
- `Scheduler(model: GPT2Model, *, max_batch_size: int, pad_token_id: int)`; `submit(request_id: str, prompt: torch.Tensor, max_new_tokens: int, *, stop_token_ids: tuple[int, ...]=(), sampling: SamplingParams | None=None) -> None`; `step() -> list[TokenEvent]`; `result(request_id: str) -> torch.Tensor`; properties `idle: bool`, `cache_allocated_bytes: int`, and `peak_kv_bytes: int`.
- Unknown result IDs raise `KeyError`; unfinished results raise `ValueError`; completed results return defensive copies. Reject duplicate IDs throughout the scheduler lifetime. Validate/clone prompt and stop settings before enqueueing; reject invalid/Boolean counts and stop IDs.
- Keep private state simple: `_requests` maps IDs to a private dataclass with prompt/output, budget/stops/settings/generator, optional cache, and finish reason; `_waiting` is a deque of IDs; `_running` is an ordered list of IDs; `_pending_events` retains committed events across a later phase exception.
- Private `_prefill(requests) -> None` and `_decode(requests) -> None` commit their successful phase and append events. `step` snapshots old running requests, admits a bounded cohort, then decodes only that old snapshot; it returns and clears pending events on success. Release finished caches before the next step. No internal indefinite retry.

- [ ] Write lifecycle tests with real tiny models: active limit two, requests A/B/C with budgets four/one/two. First step admits A/B and finishes B; second admits C while A continues. Assert one token per eligible request per step, admission-before-decode event order, FIFO replacement, no active batch beyond two, results equal independent greedy generation, and idle steps return `[]`.

  ```python
  scheduler = Scheduler(model, max_batch_size=2, pad_token_id=0)
  for name, prompt, budget in [('A', [1], 4), ('B', [2, 3, 4], 1), ('C', [5], 2)]:
      scheduler.submit(name, torch.tensor(prompt), budget)
  assert [event.request_id for event in scheduler.step()] == ['A', 'B']
  assert scheduler.result('B').shape == (4,)
  assert [event.request_id for event in scheduler.step()] == ['C', 'A']
  ```

- [ ] Add dynamic submissions, active limits one/two, multiple stop IDs, prompt stop IDs that do not finish early, stop-versus-budget reason precedence, one-token requests, and zero-output waiters ahead of normal requests. A zero-only workload observes no cache construction or forward and reaches idle; a finite queue with varied budgets admits every ID in order.
- [ ] Add upfront validation tests for IDs, later prompt dtype/device/rank/tokens, counts/context, pad/stop IDs, sampling settings, and duplicate IDs before/after completion. Assert no queue/result mutation on rejection. Mutate a caller's prompt after submit and a returned completed result; accepted/stored IDs remain unchanged.
- [ ] Add packed-logit oracle checks against a separate identical tiny model doing full independent forwards: masked admission valid positions and each packed decode suffix satisfy `atol=1e-4, rtol=1e-4`, including new arrivals with different prefix lengths and exact context budgets. Poison newly allocated unused K/V with NaNs to prove the packer initializes masked prefix slots and copies only real-token private prefixes.
- [ ] Add literal byte-accounting test with two-layer/24-hidden FP32 model, prompts of lengths one/three and budgets four/three: private reservations after first step are `(5+6)*384 = 4224` bytes; peak private-plus-workspace is `(11+10)*384 = 8064` bytes; all private caches are released on completion. Check first-token completion has no persistent cache and temporary workspaces do not survive into the next forward.
- [ ] Add projection-fault tests for admission and decoding: failing phase leaves accepted queue/private prefix/output/generator state reusable; retry produces the independent outputs. For successful admission C followed by failed old-request A decode, retain C's first event, then deliver it once with the retry's events; no repeated C prefill or skipped A token. Observe real forwards and inject only the fault.
- [ ] Run `HF_HUB_OFFLINE=1 .venv/bin/python -m pytest tests/test_scheduler.py -q`; expect missing scheduler API failures.
- [ ] Implement bounded FIFO lifecycle, per-request generator ownership, and request result/event storage. Reject inputs before mutation; clone accepted prompts and completed results.
- [ ] Implement padded admission workspace of longest-prompt capacity and packed decode workspace of largest-private-prefix-plus-one capacity. Initialize masked prefix storage, use full key masks/logical positions, and scatter only real prefill/new decode K/V after successful logits. Record peak bytes including staged private caches while the workspace still lives. Drop logits/views before the next forward.
- [ ] Implement phase recovery using pending committed events. Model-forward failure occurs before private cache/output/RNG commits; keep failed work available and release the local temporary workspace. Earlier successful work remains committed.
- [ ] Run scheduler/sampler tests and the full suite; require literal traces, byte counts, retry behavior, and all original acceptance checks to pass.
- [ ] Commit with `feat: schedule continuous batches with per-request cache state`.

## Task 3: Public-model parity, seeded isolation, and CLI

**Files:** Extend `tests/test_correctness_vs_hf.py`, `tests/test_scheduler.py`; add `main()` to `engine/scheduler.py` and CLI tests in `tests/test_scheduler.py`.

**Interfaces:**
- Consume Task 2's public scheduler API and Task 1's settings. Reuse session `public_models` and the existing M3 public batch prompt/expected-output fixture in its current module.
- CLI: repeated required `--prompt`; `--max-new-tokens` integer list default `[50]` (one budget broadcasts, otherwise exactly one per prompt); `--max-batch-size` default two; `--device` auto/cpu/cuda; `--temperature` default zero, `--top-k` zero, `--top-p` one, `--seed` zero. CLI stop IDs use tokenizer EOS. Output JSON uses `ensure_ascii=False`, original text plus only decoded new IDs, submission order, and EOS/BOS seeding for empty text.

- [ ] Add same-device seeded tests against an independent full-prefix tiny-model sampling loop using Task 1's sampler: a request alone, in different rows, after delayed admission, and amid differently seeded/early-finishing requests gets identical IDs. Different request generators do not affect global RNG state. Test both temperature-only and combined top-k/top-p settings.
- [ ] Add public GPT-2 scheduler acceptance with three short/128-plus-token/Unicode prompts and 50 new tokens each, active limits one/two, and one submission arriving after decoding starts. Compare every result exactly with existing independent engine/HF expected IDs; compare observed packed suffix logits to independent full forward at `atol=1e-4, rtol=1e-4`.
- [ ] Add conditional CUDA greedy parity and same-device sampled-alone-versus-interleaved checks using a separate model copy; keep CPU fixture weights unchanged. Do not assert cross-device stochastic equality. Add real-forward weak-reference checks that prior logits/workspaces are released before subsequent prefill or decode.
- [ ] Write CLI tests for budget broadcast/list mismatch, negative/invalid settings before model load when vocabulary-independent, distinct budgets and output order, empty prompts, Unicode/newlines, literal special-token text, and zero outputs. Parse JSON and compare newly decoded suffixes with programmatic scheduler results.
- [ ] Run targeted CLI tests; expect missing `main` behavior before implementing it. Existing scheduler tests may already pass as acceptance extensions to Task 2.
- [ ] Implement `main()` with the existing loader/device behavior and stated argument validation. Drain scheduler events synchronously and obtain results in original submission order.
- [ ] Run full suite and real CPU demo: `HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 .venv/bin/python -m engine.scheduler --prompt 'Hello' --prompt 'The quick brown fox' --max-new-tokens 5 50 --max-batch-size 2 --device cpu`; require valid ordered JSON and compare each greedy continuation with independent generation.
- [ ] Commit with `test: verify scheduler parity and request-owned sampling`.

## Task 4: Mixed-length benchmark, documentation, and release

**Files:** Create `benchmarks/bench_scheduler.py`, `tests/test_scheduler_benchmarks.py`, `results/continuous_batching.csv`; update `README.md`, `docs/architecture.md`, `docs/lessons.md`, and this plan's checkboxes. Do not edit `results/kv_cache.csv`.

**Interfaces:**
- `run_scheduler_workload(model: GPT2Model, prompts: list[torch.Tensor], budgets: list[int], *, pad_token_id: int, max_batch_size: int, repetitions: int, continuous: bool) -> dict[str, object]`. Greedy, no stops, positive budgets, at least three repetitions, one warmup; check all outputs against independent cached generation before timing.
- Static uses existing cached `generate_batch` on FIFO cohorts with each cohort's maximum budget, trims useful outputs, and exposes results only on cohort return. Continuous submits all at time zero and observes completion events. Total timers include validation/allocation/packing/prefill/decoding; no loading/tokenization. CUDA synchronizes timed boundaries and completion timestamps when measured.
- CLI defaults: prompt lengths `[16,64,32,128,16,64,32,128]`, budgets `[4,32,8,64,4,32,8,64]`, active limit two, repetitions three, threads one, device auto, seed zero, output `results/continuous_batching.csv`. Reject mismatched lists, invalid counts/threads, empty workload, and over-context individual or padded static-cohort budgets before timing.
- CSV: `stage` (`static_cohorts`/`continuous`), `device`, `hardware`, `torch_version`, `threads`, `request_count`, `max_batch_size`, `prompt_lengths`, `output_budgets`, `repetitions`, `useful_generated_tokens`, `extra_static_tokens`, `median_seconds`, `tokens_per_second`, `completion_p50_ms`, `completion_p95_ms`, `peak_kv_bytes`, `peak_memory_bytes`, `peak_memory_kind`, `workload`. Serialize lists as JSON; use LF endings. Reuse existing `hardware_name`.

- [ ] Write tiny real-model metric tests for prompt lengths one/three/one and budgets one/three/one, active limit two: useful tokens five, static extra tokens two, continuous extra tokens zero, equal useful outputs, valid completion p50/p95, CPU process peak `None`/`unmeasured`, and correctly scoped K/V bytes. Static first-cohort completions share one cohort-return timestamp. CSV roundtrip preserves blank process-memory cells and JSON workload lists.

  ```python
  prompts = [torch.tensor([1]), torch.tensor([2, 3, 4]), torch.tensor([5])]
  row = run_scheduler_workload(model, prompts, [1, 3, 1], pad_token_id=0,
      max_batch_size=2, repetitions=3, continuous=False)
  assert row['useful_generated_tokens'] == 5 and row['extra_static_tokens'] == 2
  assert row['peak_memory_bytes'] is None and row['peak_memory_kind'] == 'unmeasured'
  assert row['peak_kv_bytes'] == 4608
  ```

- [ ] Add invalid-workload/repetition/context tests and one-request/one-token workload checks; no test requires an elapsed-time speedup. Verify static excess tokens are excluded from the throughput numerator and latency columns describe whole-request completion, not M2 decode steps.
- [ ] Run `HF_HUB_OFFLINE=1 .venv/bin/python -m pytest tests/test_scheduler_benchmarks.py -q`; expect missing benchmark implementation.
- [ ] Implement timing/CSV using standard-library tools. Observe actual static cache allocations through real-forward cache arguments; use scheduler counters for continuous private-plus-temporary K/V. Count useful/excess tokens explicitly; use median end-to-end time and inclusive completion-latency quantiles. Mark actual CUDA allocated tensor peak scope if run there; leave CPU process peak blank.
- [ ] Run `HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 .venv/bin/python -m benchmarks.bench_scheduler --device cpu`; inspect both CSV rows and retain measured results whichever stage wins. If restricted hardware lookup only reports architecture, obtain read-only CPU metadata and rerun the exact workload with that access, as in M2.
- [ ] Update README/status/results and queue/prefill/decode/completion Mermaid. Explain FIFO fairness, per-request seeds/stops/budgets, fixed-cohort comparator, completion-latency scope, temporary-copy memory/cost, result retention, synchronous ownership, and observed benchmark results. Record actual bugs and unresolved hardware limits in lessons; preserve project identity and earlier results.
- [ ] Run full suite, CPU demo, `.venv/bin/python -m compileall -q engine benchmarks`, tracked and staged whitespace checks, and verify M2 CSV is unchanged. Request one fresh independent read-only review of M4 against the approved spec and this plan; fix important findings with reproducing checks and a green full suite.
- [ ] Commit with `bench: compare static cohorts and continuous batching on CPU`, push to `origin/codex/mini-infer-m4`, and confirm local/remote HEAD match. Create/attach a draft M4 PR against `codex/mini-infer-m3` while M3 is unmerged (otherwise the verified main dependency), using a body that matches the final implementation. Keep branches and checkout; remove only M4 scratch tracking after completion.

## Plan self-review

Task 1 owns sampling validation/filtering and RNG behavior. Task 2 owns FIFO lifecycle, private/temporary cache correctness, bytes, and phase recovery. Task 3 owns public-model acceptance, same-device sampling isolation, and the CLI. Task 4 owns comparable measurements, docs, final review, and publication. Every review condition has explicit owning tests; consumed signatures agree. No permanent cache arena, serving dependency, hardware-dependent speedup gate, or change to earlier acceptance tolerances is introduced.
