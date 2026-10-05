# mini-infer Milestone 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Ship a readable custom GPT-2 forward pass and uncached generation with verified public GPT-2 parity on CPU.

**Architecture:** Explicit PyTorch transformer blocks, one Hugging Face weight-loading boundary, and a full-sequence greedy loop. Keep the baseline independently runnable as later optimizations arrive. Work inside the existing project's directory and repository.

**Tech Stack:** Python 3.11+, PyTorch, transformers for loading/reference, pytest for correctness. Pin dependency versions after checking actual installation and execution.

**Spec:** [Approved design](../specs/2026-10-05-mini-infer-m1-design.md).

## Global Constraints

- Project root: `.`; standalone Git repository; no unrelated edits.
- Python 3.11+, CPU execution, and optional CUDA are required.
- Public GPT-2 weights require no API key or hosted inference service.
- No Hugging Face forward or `generate()` call in engine execution.
- Model loading and execution default to FP32.
- Logit comparisons use `atol=1e-4` and `rtol=1e-4`; generated greedy token IDs must be exactly equal.
- Cache downloads in gitignored `model_cache/`; never stage downloaded weights or credentials.
- Do not relax tolerance or remove correctness checks to get green.
- Commit this project's changes in small milestone commits; do not push.
- Milestone 1 only; keep other milestones on the documented roadmap.

## Review Focus

- Wrong or incomplete checkpoint tensors: reject with the offending key and expected shape (Task 2).
- UTF-8 and whitespace prompts: preserve tokenizer input and compare real GPT-2 outputs (Task 3).
- Explicit CUDA request without CUDA: fail clearly rather than silently choosing CPU (Task 3).
- Context boundary: allow exactly the supported length, reject one token over before model execution (Tasks 1 and 3).
- Offline first download: expose a useful loading error; cached execution must not require credentials (Task 2).

## Task 1: Custom GPT-2 transformer

**Files:** Create `engine/__init__.py`, `engine/config.py`, `engine/model.py`, `tests/test_model.py`, `requirements.txt`, `.gitignore`.

**Interfaces:**
- `ModelConfig`: frozen dataclass with `vocab_size`, `max_positions`, `hidden_size`, `num_layers`, `num_heads`, `intermediate_size`, `layer_norm_epsilon=1e-5`, and `activation_function="gelu_new"`. Reject nonpositive dimensions and a hidden dimension not divisible by head count.
- `EngineConfig`: frozen dataclass with `device="auto"`, `model_name="gpt2"`, and `cache_dir="model_cache"`.
- `GPT2Model(config: ModelConfig)`; `forward(input_ids: torch.Tensor) -> torch.Tensor` returns `[batch, sequence, vocabulary]` logits. `lm_head.weight` and `token_embedding.weight` share storage.

- [x] Inspect Python and installed packages, create a local `.venv` if needed, install compatible dependencies, and write exact direct-version pins. Confirm imports and versions by execution. Do not alter parent dependencies.
- [x] Write failing structural tests on a seeded tiny model with vocabulary 37, context 16, hidden size 24, two layers, and four heads. Assert output shape, FP32 output, tied storage, future-token independence, and rejection of invalid IDs, rank/dtype, empty token sequences, and context length 17. Assert length 16 succeeds.
- [x] Run `python -m pytest tests/test_model.py -q`; verify failure reflects the missing implementation.
- [x] Implement validated config and model: learned embeddings, pre-norm residual blocks, explicit scaled attention, registered causal mask, GPT-2 GELU, final normalization, tied head. Use deterministic evaluation behavior and no architecture registry. Comments explain causal masking and attention layout.
- [x] Run `python -m pytest tests/test_model.py -q`; require all structural checks to pass.
- [x] Commit only Task 1 files with `feat: implement readable GPT-2 transformer`.

## Task 2: Public weights and logit parity

**Files:** Create `engine/weights.py`, `tests/test_correctness_vs_hf.py`; extend `tests/test_model.py` only if needed for shared structural behavior.

**Interfaces:**
- Consumes Task 1's `ModelConfig` and `GPT2Model`.
- `resolve_device(device: str) -> torch.device`: accept `auto`, `cpu`, and `cuda`; reject unavailable explicit CUDA and unsupported values.
- `copy_hf_weights(model: GPT2Model, state_dict: Mapping[str, torch.Tensor]) -> None`: validate all required keys/shapes before modifying model weights; map embeddings/norms directly and transpose Conv1D projection matrices.
- `load_model(config: EngineConfig) -> tuple[GPT2Model, PreTrainedTokenizerBase]`: load public configuration/tokenizer/checkpoint through transformers, construct the custom FP32 model, copy weights, move to resolved device, return evaluation model/tokenizer, release the reference model.

- [x] Write failing tests that load real `gpt2` into a session-scoped FP32 CPU reference and compare full logits for `"Hello, world!"`, `"The quick brown fox jumps over the lazy dog."`, and `"Café — hello!\n  Spaces matter."`. Assert close with `atol=1e-4, rtol=1e-4`, recording maximum absolute error in diagnostic messages. Add a small checkpoint-mapping test that checks every parameter against a tiny HF model, plus malformed missing-key/shape cases and a simulated offline loading exception.
- [x] Run `python -m pytest tests/test_correctness_vs_hf.py -q`; confirm missing loader failure. If weights cannot download, record the blocker rather than skip public-model acceptance.
- [x] Implement weight mapping for every transformer layer and the tied head. Read actual HF configuration and state-dict names; verify Conv1D orientation from tensor shapes. Include errors naming the mismatched key/shape. Preserve download error context. Do not keep a reference model on the engine instance.
- [x] Run `python -m pytest -q`; require real GPT-2 logit parity and structural tests to pass. Diagnose root causes such as transpose, normalization, masking, or activation before editing tolerances.
- [x] Commit Task 2 files with `feat: load GPT-2 weights and verify logit parity`.

## Task 3: Uncached generation and CPU CLI

**Files:** Create `engine/sampler.py`, `engine/generate.py`, `tests/test_generation.py`; extend `tests/test_correctness_vs_hf.py`.

**Interfaces:**
- `greedy(logits: torch.Tensor) -> torch.Tensor`: argmax over the vocabulary dimension.
- `generate(model: GPT2Model, input_ids: torch.Tensor, max_new_tokens: int, *, eos_token_id: int | None = None) -> torch.Tensor`: accept one unpadded request shaped `[1, sequence]`, return prompt plus generated IDs, recompute full sequence each iteration under inference mode. `eos_token_id=None` disables stopping; otherwise include the terminal EOS before returning.
- `main() -> None`: parse `--prompt`, `--max-new-tokens` (default 50), and `--device` (default auto), load model/tokenizer, seed an empty prompt with the tokenizer's EOS token, print decoded text. Keep tokenization out of the tensor generation loop.

- [x] Write failing tests for zero-token identity, negative length, exact/overflow context budgets, EOS inclusion and termination using a controlled tiny model, and invalid batch shape. Test unavailable CUDA via a mocked availability check. Verify an empty CLI prompt receives one EOS seed using a tiny local model/tokenizer double; use this double only for CLI boundaries.
- [x] Extend real GPT-2 tests: for each Task 2 prompt, compare at least 50 new greedy IDs against the HF reference with `do_sample=False` and EOS stopping disabled consistently. Test exact token equality; reference forward or generation stays in test code. Add conditional CUDA parity tests without replacing CPU coverage.
- [x] Run `python -m pytest tests/test_generation.py tests/test_correctness_vs_hf.py -q`; verify failures come from missing generation behavior.
- [x] Implement the simple loop, greedy selection, validation, and CLI. Validate prompt-plus-output budget before decoding. Keep EOS behavior explicit and reproducible.
- [x] Run `python -m pytest -q`, then `python -m engine.generate --prompt 'Hello, world!' --max-new-tokens 50 --device cpu`. Require passing public GPT-2 tests and real decoded demo output.
- [x] Commit Task 3 files with `feat: add uncached greedy generation and CLI`.

## Task 4: Usable milestone release and final verification

**Files:** Create `README.md`, `docs/architecture.md`, `docs/lessons.md`, `LICENSE`; update requirements only if actual execution exposed necessary pin changes.

**Interfaces:** Document Task 3's runnable CLI and the existing test suite; introduce no new product APIs.

- [x] Write README with the exact requested first line: “LLM serving wastes GPU time and memory. mini-infer is a from-scratch engine that shows exactly how to reclaim both, with every optimization measured.” Follow immediately with current status: only the baseline is implemented and no speedup is measured yet. Include three-command CPU quick start, test command, baseline limitations, and remaining milestone roadmap. Do not list unimplemented commands as usable.
- [x] Document the baseline transformer and full-sequence generation flow in Mermaid, tied embeddings, reference boundary, and deferred prefill/decode separation. Record only bugs and checks actually observed in `docs/lessons.md`. Add MIT license without inventing a copyright holder.
- [x] Run `python -m pytest -q`, the CPU CLI demo, `python -m compileall -q engine`, and `git diff --check`. Check tracked paths exclude model binaries, cache files, credentials, and `.venv`. Record test counts, maximum observed logit error, and CUDA availability accurately.
- [x] Request an independent final review under the selected execution workflow. Resolve actionable findings and repeat affected checks; do not spawn additional implementation agents under native execution.
- [x] Commit milestone documentation with `docs: document verified mini-infer baseline` and report commands, results, known limitations, and the next milestone. Stop after Milestone 1 acceptance; continue later milestones through their required designs.

## Plan self-review

The four tasks cover the approved spec: structural correctness, real weight/reference parity, generation/CLI boundaries, and honest release documentation. Interfaces match across tasks; all five review conditions have checks in their owning tasks. Real CPU validation is mandatory, CUDA conditional, and no subsequent milestone or speculative abstraction is included.
