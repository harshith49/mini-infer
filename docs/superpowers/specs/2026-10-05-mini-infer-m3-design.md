# mini-infer: Milestone 3 — static batching

## Intent and scope

Generate several different-length prompts in one fixed batch, preserving the same per-request greedy token IDs as running each prompt alone. Support the existing uncached and contiguous-cache paths, retain all Milestone 1/2 correctness checks, and keep CPU mandatory with CUDA conditional on hardware. This is the next step in the user's authorized milestone sequence.

Continue in the standalone `mini-infer` project on `codex/mini-infer-m3`, publishing verified milestone commits to `harshith49/mini-infer`. Keep unrelated files, model downloads, environments, and credentials out of commits. No automatic merge or force push.

## Approach and alternatives

Use one left-padded tensor, a binary attention mask, and per-row learned position IDs. Reuse `GPT2Model`, `SimpleKVCache`, and greedy selection. Calling the single-request loop once per prompt would preserve correctness but would not implement batching. Length bucketing can reduce padding but adds grouping policy before the fixed-batch baseline exists; defer it.

The batch has one common `max_new_tokens` budget and optional common EOS stop ID. It remains fixed during execution: finished rows keep their slots and receive masked filler tokens while other rows continue. Do not add waiting queues, per-request sampling settings, batch compaction, or continuous admission in this milestone.

## Model input contract

Extend `GPT2Model.forward(input_ids, *, cache=None, attention_mask=None, position_ids=None)` while preserving existing no-mask behavior and parameter names.

- `input_ids` remains a nonempty rank-2 `torch.long` tensor of valid vocabulary IDs.
- `attention_mask`, when supplied, is a same-device rank-2 Boolean or integer 0/1 tensor shaped `[batch, past_physical_length + new_chunk_length]`. It describes all keys, including cached history. Each row must contain at least one real token. Token values never determine whether a slot is padding.
- `position_ids`, when supplied, is a same-device `torch.long` tensor shaped `[batch, new_chunk_length]`, with every value inside the learned position-embedding range.
- Without explicit position IDs, derive them from `attention_mask.cumsum(-1) - 1`, set masked positions to zero, and select the new chunk's suffix. Without a mask, retain the existing absolute physical-offset position IDs.
- Validate masks, positions, batch shape, and context/cache capacity before writes. Invalid input must not change committed cache state.

Attention combines the existing absolute-offset causal mask with the per-row key-validity mask. Leading padded queries may have no legal keys; ensure finite attention outputs by assigning zero probability to all blocked positions, including fully blocked rows. Valid-token logits must match individual unpadded forwards. Padded-query logits are discarded, and padded keys remain invisible to later valid queries.

## Cache integration

Keep the current `[layers, batch, heads, capacity, head_dim]` storage. Its length counts physical columns, including left padding; learned position IDs count real tokens separately. Allocate capacity `longest_prompt_length + max_new_tokens` for the whole batch, using model dtype/device and the batch size. Reject a batch when that physical budget exceeds context capacity; do not silently truncate long requests.

Pass the full historical attention mask on every cached padded forward. Add only a small cache flag recording whether masked keys have ever been committed, so omitting the mask on a later call fails clearly rather than exposing padding. Commit that flag with length only after successful logits. No persistent mask tensor or automatic reconstruction from token IDs is needed: the batch loop owns the mask history.

Historical masks must preserve previously committed key validity. Changing a committed prefix's mask is unsupported because its cached layer states were computed under the original mask; the caller owns that consistency.

Cache byte accounting remains actual K/V tensor storage, including padded slots. Keep the prefix-preserving retry behavior and the logits-lifetime fix from Milestone 2. No change to single-request default behavior or the published Milestone 2 measurements is intended.

## Batch generation API and CLI

Create `engine/batching.py` with `generate_batch(model, prompts, max_new_tokens, *, pad_token_id, eos_token_id=None, use_cache=False) -> list[torch.Tensor]`. Each prompt is a nonempty rank-1 `torch.long` tensor on the model device. Return rank-1 tensors containing each original unpadded prompt plus its generated IDs, in input order, with no filler tokens. Keep `engine.generate.generate` as the independent single-request reference.

Validate every request before allocation or forwarding. Reject an empty prompt list, empty token tensors, invalid dtype/device/IDs, negative output budget, or invalid padding/EOS IDs. Zero new tokens returns the original prompts without allocating a cache or forwarding. Padding may use the same ID as a real token or EOS; the mask distinguishes those roles.

Left-pad to the longest prompt. Each generation step performs one model forward for the batch, selects final-position greedy tokens, and appends them with a validity mask. Record the first generated EOS for each row, include it in that row's result, and discard later filler. Stop when every row finishes or the common output budget is exhausted. With EOS disabled, every row receives exactly the requested number of new tokens.

Cached execution prefills the padded batch once and then forwards one physical column per step. Uncached execution forwards the full padded sequence each step. Finished rows may still perform dummy computation; retaining their slots is the static baseline, not a scheduling optimization.

Add a small runnable CLI: `python -m engine.batching --prompt 'Hello' --prompt 'The quick brown fox' --max-new-tokens 50 --device cpu --use-cache`. Repeated `--prompt` arguments define the batch. Match existing device selection and empty-text EOS/BOS seeding. Print a JSON list of continuations that preserves each original prompt verbatim and decodes only new IDs. No server or new dependency.

## Correctness and acceptance

1. Rerun all existing tests with unchanged meaning and tolerances; public GPT-2 baseline and cached 50-token comparisons remain mandatory.
2. Compare padded valid-position logits with independent unpadded logits and the HF reference using `atol=1e-4, rtol=1e-4`. Check both masked full forwards and cached chunk/decode suffixes.
3. For actual public GPT-2, compare each batch row against single-request engine and HF greedy output for at least 50 new tokens. Cover short versus very long prompts, Unicode/newlines, several prompt orders, and single-row batches in both cache modes. No relaxation of token equality.
4. Use tiny models for padding invariance (changing masked token IDs cannot change real-token logits), explicit/derived position equivalence, finite padded-query outputs, context bounds, and invalid mask/position rejection before cache commits.
5. Check independent EOS completion and result trimming, all-finished stopping, zero-token no-work behavior, literal-special-token prompt preservation, and no retention of previous logits into the next batch forward. Use real forwards or targeted fault injection rather than replacing the generation algorithm with a mock.
6. Test that a missing mask after cached padded prefill is rejected without advancing state. A failed masked forward must leave the old prefix and metadata reusable. CUDA batch parity checks run only when CUDA exists and remain explicitly unverified on this CPU-only host.
7. Run the full suite and real batched CPU CLI demo, obtain one independent final review under retained native execution, record actual bugs/results in `docs/lessons.md`, and push the verified milestone.

## Documentation and limits

Update README and architecture diagrams with left padding, real-token positions, fixed batch slots, output order, and padded cache reservations. Keep batching performance unmeasured until an actual batch benchmark is implemented; this milestone makes no new speedup claim. Preserve existing CPU KV results and unmeasured GPU labels.

Per-request output budgets/sampling, length bucketing, removal/admission between decode steps, paged cache allocation, quantization, and serving remain subsequent milestones. After written-spec review, write the implementation plan and retain native execution with one final independent reviewer.
