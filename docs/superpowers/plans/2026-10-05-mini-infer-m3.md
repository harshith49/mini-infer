# mini-infer Milestone 3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans for the retained native execution method. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Generate different-length prompts in a fixed batch with exactly the same greedy token IDs as independent requests.

**Architecture:** Left-pad prompts and carry a full binary key mask, deriving learned positions from real-token counts. Reuse the existing model and contiguous cache; finished rows retain their slots with masked filler. Keep single-request generation as an independent reference.

**Tech Stack:** Existing Python 3.11, PyTorch 2.5.1, transformers 4.48.3, pytest 8.3.5, safetensors 0.8.0; standard-library argparse and JSON. No new dependencies.

**Spec:** [Approved Milestone 3 design](../specs/2026-10-05-mini-infer-m3-design.md).

## Global Constraints

- Continue in the standalone `mini-infer` project on `codex/mini-infer-m3`, publishing verified milestone commits to `harshith49/mini-infer`.
- Keep unrelated files, model downloads, environments, and credentials out of commits. No automatic merge or force push.
- Retain all Milestone 1/2 correctness checks, and keep CPU mandatory with CUDA conditional on hardware.
- Valid-token logits must match individual unpadded forwards using `atol=1e-4, rtol=1e-4`; actual public GPT-2 greedy comparisons require at least 50 new tokens and exact token equality.
- The batch has one common `max_new_tokens` budget and optional common EOS stop ID. It remains fixed during execution.
- Token values never determine whether a slot is padding. Historical masks must preserve previously committed key validity; the caller owns that consistency.
- Cache length counts physical columns, including left padding; learned position IDs count real tokens separately.
- Validate all requests before allocation or forwarding. Invalid model inputs must not change committed cache state.
- Include each row's first generated EOS and discard subsequent filler. Preserve input order and original CLI prompt text.
- Keep batching performance unmeasured, preserve existing CPU KV results and unmeasured GPU labels, and add no new dependency.
- Do not add waiting queues, per-request sampling settings, batch compaction, or continuous admission in this milestone.

## Review Focus

- Padding ID equal to EOS or a real prompt token must not hide real tokens or stop an unfinished row (Task 2).
- Leading padded queries with no legal keys must produce finite outputs without exposing padded keys (Task 1).
- Omitted masks after padded cache commits and failed late projections must preserve cache metadata and committed prefixes (Task 1).
- A bad later prompt must fail before the first model forward or cache allocation; zero-token requests still validate (Task 2).
- Finished rows must retain valid EOS history while later filler stays masked, and previous logits must die before the next forward (Task 2).

## Task 1: Masked attention, logical positions, and cache commits

**Files:** Modify `engine/model.py`, `engine/kv_cache.py`, `tests/test_kv_cache.py`; create `tests/test_batching.py` for tiny-model checks.

**Interfaces:**
- Extend `GPT2Model.forward(input_ids: torch.Tensor, *, cache: SimpleKVCache | None=None, attention_mask: torch.Tensor | None=None, position_ids: torch.Tensor | None=None) -> torch.Tensor`.
- Thread optional `attention_mask` through `TransformerBlock.forward` and `CausalAttention.forward`, retaining existing `cache` and `layer_idx` names and checkpoint parameters.
- Add `SimpleKVCache.requires_attention_mask: bool`, initially false. Set it only after successful logits if a supplied mask contains masked keys; once true it stays true. Reject omitted masks while true.
- Masks are same-device rank-2 Boolean or integer 0/1 tensors shaped `[batch, cache.length + new_length]`, with at least one real key in each row. Explicit positions are same-device `torch.long` shaped `[batch, new_length]`, within `[0, max_positions)`.
- Derive implicit logical positions from full-mask cumulative counts minus one, zero masked positions, and take the new chunk suffix. Retain physical offset positions when neither mask nor positions is supplied.

- [x] Write tiny-model tests with vocabulary 37, context 16, hidden size 24, two layers, four heads, and intermediate size 96. For IDs `[[0,0,1],[2,3,4]]`, mask `[[0,0,1],[1,1,1]]`, compare valid logits to individual forwards and explicit positions `[[0,0,0],[0,1,2]]` at `atol=1e-4, rtol=1e-4`. Assert every output is finite and changing masked token IDs leaves valid logits unchanged.
- [x] Add cache tests: padded prefill of width three followed by a two-token chunk matches each independent five/three-token logical sequence's suffix logits; cache length becomes five and byte counts include physical padding. Exercise exact capacity and explicit position IDs without a mask.
- [x] Add parameterized rejection tests for wrong mask rank/length/batch/device, floating masks, nonbinary integers, all-zero rows, bad position rank/shape/dtype/device/range, and exhausted context/capacity. Snapshot committed prefixes, length, and flag and assert no change. Use a meta-device tensor for same-device validation without requiring CUDA.
- [x] Add missing-mask and late-projection-failure tests. After successful padded prefill, omitted mask raises without changes. Starting with an unmasked committed prefix, inject a projection exception during a masked extension; flag and prefix stay unchanged. Retry with the full valid mask, compare suffix logits to a full forward, and assert the flag commits only on success.
- [x] Run `HF_HUB_OFFLINE=1 .venv/bin/python -m pytest tests/test_batching.py tests/test_kv_cache.py -q`; require failures from missing new arguments or metadata before implementation.
- [x] Implement validation before transformer writes and combine causal visibility with key validity. On the masked path, softmax finite-minimum masked scores and explicitly zero blocked probabilities, including all-blocked padded queries. Preserve the no-mask operations and commit cache length/flag only after logits succeed.
- [x] Run the full suite with `HF_HUB_OFFLINE=1 .venv/bin/python -m pytest -q`; require all CPU checks to pass and hardware-dependent CUDA checks to skip only when unavailable.
- [x] Commit with `feat: add padding masks and logical positions to GPT-2`.

## Task 2: Fixed-batch generation and CLI

**Files:** Create `engine/batching.py`; extend `tests/test_batching.py` and `tests/test_generation.py` for CLI boundaries.

**Interfaces:**
- `generate_batch(model: GPT2Model, prompts: list[torch.Tensor], max_new_tokens: int, *, pad_token_id: int, eos_token_id: int | None=None, use_cache: bool=False) -> list[torch.Tensor]`, under inference mode. Prompts and results are rank-1 tensors; outputs contain original prompt plus actual new tokens, in input order.
- Consume Task 1's masked forward and cache flag; reuse `engine.sampler.greedy`, `load_model(EngineConfig(...))`, and `SimpleKVCache`. Leave `engine.generate.generate` independent.
- CLI `main() -> None`: repeated required `--prompt`, `--max-new-tokens` default 50, `--device` choices auto/cpu/cuda, and `--use-cache`. Seed empty text with tokenizer EOS/BOS; print a JSON list with `ensure_ascii=False`, raw original prompt plus only decoded new IDs.

- [x] Add real tiny-model tests in both cache modes: unequal prompts equal independent `generate` results; single-row batch works; output order survives reordered prompts; pad ID may equal a genuine prompt token or EOS. Assert exact output count with EOS disabled.
- [x] Observe real forward input lengths: width-three prompts and four outputs give cached `[3,1,1,1]` and uncached `[3,4,5,6]`; capture masks to assert leading padding remains blocked. Use a weak reference to the prior logits and assert it is dead at the next forward.
- [x] Add independent EOS tests using a controlled final-logit projection after real transformer forwards: row zero selects EOS at step one, row one at step three. Assert result lengths are prompt lengths plus one/three, both include EOS, exactly three batch forwards occur, terminal EOS keys stay valid, and later filler for row zero is masked. Add all-rows-EOS-first-step stopping.
- [x] Add rejection tests for empty list, empty/wrong-rank/wrong-dtype/wrong-device/invalid-ID later prompts, negative or noninteger output budget, invalid pad/EOS IDs, and longest-prompt-plus-budget overflow. A forward observer and allocation spy must see zero calls on rejected or zero-output requests; zero outputs return unchanged valid prompts.
- [x] Add CLI tests for repeated prompts, empty prompt seeding, Unicode/newlines, zero outputs, literal `<|endoftext|>` preservation, and negative-budget argparse errors. Decode only the generated suffix and parse output with `json.loads` to check order and exact original prefixes.
- [x] Run targeted tests; require failures from the missing batch API/CLI before implementation.
- [x] Implement upfront validation, left padding, full mask history, and one forward per iteration. Allocate cache capacity `longest_prompt_length + max_new_tokens`. Track active rows and actual output lengths; append EOS as valid on its selection step, then mask filler on later steps. Release logits immediately after selection and never forward the final sampled tokens unnecessarily.
- [x] Implement the CLI using the existing loader/device behavior and JSON output contract.
- [x] Run the full suite and commit with `feat: generate fixed batches of different-length prompts`.

## Task 3: Public-model parity, documentation, and release

**Files:** Extend `tests/test_correctness_vs_hf.py`; update `README.md`, `docs/architecture.md`, `docs/lessons.md`, and this plan's checkboxes.

**Interfaces:**
- Reuse the existing session-scoped `public_models` fixture in its current test module, Task 2's `generate_batch`, and independent `generate`/HF references. No fixture relocation or new benchmark API.

- [ ] Add public GPT-2 valid-logit checks for masked full forwards and cached two-token suffixes against individual engine and HF forwards at `atol=1e-4, rtol=1e-4`. Supply explicit logical positions to HF when using padded reference inputs.
- [ ] Add 50-new-token exact greedy comparisons in both cache modes for batches containing `Hello, world!`, `Café — hello!\n  Spaces matter.`, and a repeated natural-language prompt tokenized to at least 128 tokens. Check original, reversed, and rotated order, plus a single-row batch. Compute each independent HF/engine expected continuation once per prompt/mode and reuse it across orders; disable EOS stopping and assert all output lengths.
- [ ] Add conditional CUDA batch parity checks, keeping CPU fixture weights unchanged (use a separate device copy), with exact generated IDs and the same logit tolerance. Assert only hardware availability determines the skip.
- [ ] Run the full suite; if any mismatch occurs, diagnose it and add a reproducer before changing implementation. Keep all existing tests and tolerances meaningful.
- [ ] Update README and architecture Mermaid with left padding, real-token positions, fixed slots, masked filler, result order, and cache reservations including padding. Document common budgets, caller-owned historical masks, and unmeasured batching performance; preserve M2 CSV and GPU labels. Record actual findings in lessons.
- [ ] Run `HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 .venv/bin/python -m engine.batching --prompt 'Hello' --prompt 'The quick brown fox' --max-new-tokens 50 --device cpu --use-cache` and the uncached equivalent; parse the JSON and confirm matching continuations.
- [ ] Run the full suite, `.venv/bin/python -m compileall -q engine benchmarks`, and `git diff --check` (also staged diff when staging new files). Request one fresh independent read-only reviewer under native execution. Fix important findings with reproducing tests and a green full suite.
- [ ] Commit with `test: verify public GPT-2 static batch parity`, push the verified milestone to `origin/codex/mini-infer-m3`, and confirm local/remote HEAD match. Remove milestone scratch tracking after completion; no merge or force push.

## Plan self-review

Task 1 owns model masks, positions, validation, and transactional cache metadata; Task 2 owns batch generation and CLI semantics; Task 3 owns public-model acceptance, docs, review, and publication. Every review condition has explicit tests. Full padded prefill precedes cached chunk tests, so no row is asked to prefill an all-padding prefix. Interface names match the approved spec, and no later scheduling or performance subsystem is introduced.
