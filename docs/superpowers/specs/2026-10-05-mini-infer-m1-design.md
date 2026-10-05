# mini-infer: Milestone 1 design

## Purpose and scope

Build a readable, from-scratch PyTorch GPT-2 inference engine for learning and interview discussion. Correctness against Hugging Face is the first release criterion; performance improvements follow in later milestones. Python 3.11+, CPU execution, and optional CUDA are required. Public GPT-2 weights require no API key or hosted inference service.

This spec covers Milestone 1: model configuration, weight loading, a GPT-2 forward pass, uncached greedy generation, a CLI, and correctness tests. Later milestones remain the requested roadmap: KV caching, static batching, continuous batching, paged KV caching, int8 weights, streaming serving, and final benchmarks and packaging. They receive their own designs when their prerequisites exist.

## Workspace and boundaries

Place project files directly in `.`. This project has its own Git repository and remote. Keep unrelated projects out of commits. Commit this project's changes in small milestone commits; do not push.

Use the requested `engine/`, `tests/`, and `docs/` layout. Add only files needed by this milestone, including `requirements.txt`, `.gitignore`, a usable initial README, and MIT license. Defer server, cache, batching, quantization, benchmark, Docker, CI, and notebook implementations until their milestones. No speculative architecture registry, cache interface, or second model architecture.

## Components

- `engine/config.py`: typed GPT-2 configuration containing vocabulary size, context length, layer count, head count, hidden dimension, LayerNorm epsilon, and other forward-pass settings required by the downloaded model. Engine settings cover device selection and model/cache location without duplicating model settings.
- `engine/model.py`: token and learned position embeddings, pre-norm transformer blocks, causal multi-head attention, residual connections, GPT-2 GELU MLPs, final LayerNorm, and an output projection tied to token embeddings. Use standard PyTorch modules and explicit attention math so masking and tensor shapes remain visible.
- `engine/weights.py`: load public `gpt2` configuration, tokenizer, and weights through Hugging Face. Map weights into the custom model, transposing Hugging Face Conv1D matrices into PyTorch linear weights. Validate required names and shapes. Do not retain a reference model inside the inference engine after loading.
- `engine/sampler.py`: greedy token selection for this milestone. Temperature, top-k, top-p, and seeded stochastic sampling are deferred until needed.
- `engine/generate.py`: an uncached generation function and runnable CLI. Each iteration forwards the full current sequence, selects the next token from the last position, and appends it. No Hugging Face forward or `generate()` call in engine execution.
- `tests/test_correctness_vs_hf.py`: real GPT-2 reference comparisons, reusable by later milestones. Small synthetic configurations may support faster structural tests but cannot replace public GPT-2 validation.
- `docs/lessons.md`: record observed bugs, fixes, validation results, and actual environment limitations. Do not invent lessons or measurements.

## Model and generation behavior

The baseline forward accepts unpadded token IDs shaped `[batch, sequence]` and returns logits shaped `[batch, sequence, vocabulary]`. Position IDs start at zero. Attention prevents each query from reading future keys; attention scaling, GPT-2 GELU behavior, and LayerNorm epsilon match the reference configuration. Evaluation mode disables dropout, and generation runs under inference mode.

Model loading and execution default to FP32. `--device auto` chooses CUDA when available and CPU otherwise; explicit `cpu` and `cuda` are supported. An unavailable requested CUDA device produces a clear error. CPU correctness is mandatory; CUDA checks run only when available.

The CLI supports at least `--prompt`, `--max-new-tokens`, and `--device`. Public GPT-2 is the default model. Empty prompts use an explicit EOS/BOS seed token so generation has a valid initial position. Reject negative token counts, invalid token IDs, and requests whose prompt plus requested output exceeds the model's context capacity. A zero-token request returns the prompt unchanged. Support EOS termination with a way for correctness tests to disable early stopping and verify at least 50 decoding steps.

No padded batching or cached positions are implemented here. Keep the baseline generation available in later milestones as the independent oracle for engine optimizations.

## Hugging Face boundary and dependencies

Hugging Face may download/load configuration, weights, and tokenizer. Only tests and later benchmark code may execute its model forward or generation methods. Pin direct dependencies to versions verified by installation and actual execution during implementation; this spec does not guess untested API versions.

Cache downloads in gitignored `model_cache/`. Ignore virtual environments, Python caches, environment files, model binaries, and safetensors. Never stage downloaded weights or credentials. No token is required for GPT-2.

Loading errors should retain useful context, such as an unavailable download or mismatched tensor shape. Never silently fall back to random weights or a different model.

## Correctness and acceptance

1. With both models in evaluation mode and FP32, compare full logits on several fixed prompts of different lengths, including punctuation and non-ASCII text. Use `torch.testing.assert_close` with `atol=1e-4` and `rtol=1e-4`; record maximum absolute error for diagnosis. Test the same token IDs and positions on both sides.
2. Compare custom uncached greedy output against the Hugging Face reference for at least 50 newly generated tokens on several fixed prompts. Disable EOS stopping consistently for this fixed-length check. Generated token IDs must be exactly equal.
3. Verify tied embedding weights, causal masking, context-limit rejection, empty-prompt handling, and zero-token generation with small meaningful checks. Do not create per-function test scaffolding that repeats implementation details.
4. Run the full Milestone 1 suite with actual GPT-2 weights on CPU, then run the documented CLI demo. CUDA validation is conditional on hardware availability and must be reported accurately.
5. Diagnose failures before changing code. Do not relax tolerance or remove correctness checks to get green. Any mathematically justified tolerance change requires documented evidence and explicit explanation.
6. Report download/runtime blockers honestly. A synthetic model passing is not proof of public GPT-2 parity.

The README provides working install, test, and CPU generation commands, states the supported architecture, and distinguishes implemented baseline behavior from the remaining roadmap. No speedup claim or benchmark number is published before measurement.

## Tradeoffs

Explicit attention and full-sequence recomputation cost time and memory, but provide a transparent correctness baseline. PyTorch owns tensor operations; this milestone writes no kernels. Llama support is deferred while module boundaries stay readable. Exact greedy parity applies to the unquantized engine; the later quantized stage needs separate quality criteria rather than a weakened baseline test.

## Next handoff

After the user reviews this written spec, write a Milestone 1 implementation plan with concrete files and checks, then obtain the execution-method selection required by the Superpowers workflow. Implementation begins after that handoff. Subsequent milestones rerun the baseline correctness suite and update the lessons log before advancing.
