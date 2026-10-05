# Transformer and KV-cache architecture

The engine implements GPT-2 small using PyTorch tensors and modules. Hugging Face is the loading and correctness boundary, not the inference runtime.

```mermaid
flowchart TD
    IDs[Unpadded token IDs] --> Emb[Token + learned position embeddings]
    Emb --> LN1[LayerNorm]
    LN1 --> Attn[Causal multi-head self-attention]
    Attn --> R1[Residual addition]
    Emb --> R1
    R1 --> LN2[LayerNorm]
    LN2 --> MLP[Linear / GPT-2 GELU / Linear]
    MLP --> R2[Residual addition]
    R1 --> R2
    R2 --> Next[Repeat for 12 blocks]
    Next --> Final[Final LayerNorm]
    Final --> Head[Tied embedding projection]
    Head --> Logits[Logits for every position]
```

Each attention head computes `softmax(QKᵀ / sqrt(head_dim))V`. Queries and keys have shape `[batch, heads, tokens, head_dim]`. A lower-triangular mask removes future keys before softmax. No left padding or attention-mask API is supported yet. Position IDs are learned indices starting at zero.

GPT-2's Hugging Face Conv1D weights use `[input, output]`; PyTorch Linear uses `[output, input]`. `weights.py` first rejects HF loading diagnostics that indicate absent/randomly initialized parameters, then validates the entire mapping before copying, transposing all projection weights even when square. Embedding and normalization weights copy directly. The language-model head shares the token embedding parameter rather than storing a second vocabulary matrix.

```mermaid
sequenceDiagram
    participant Request
    participant Loop as Uncached generation
    participant Model as Custom GPT-2
    Request->>Loop: Prompt and output budget
    Loop->>Loop: Validate tokens and context capacity
    loop Until output limit or EOS
        Loop->>Model: Entire prompt and all generated tokens
        Model-->>Loop: Full sequence logits
        Loop->>Loop: Argmax final-position logits; append ID
    end
    Loop-->>Request: Prompt plus new token IDs
```

The first forward processes the prompt. Each later forward processes the growing sequence again. The uncached baseline intentionally retains this repeated work as an independent correctness oracle. Milestone 2 adds an optional cached path that separates **prefill**, which processes the prompt once, from **decode**, which processes one new token with saved keys and values.

`generate()` reserves prompt length plus the full requested output budget, even if EOS might arrive earlier. It accepts one request, returns terminal EOS when configured, and runs under inference mode. With no EOS stop ID, it produces exactly the requested count. The CLI supplies GPT-2 EOS as the stop ID and uses the same token as a seed for an empty prompt.

Loading temporarily holds an HF checkpoint and a custom model, copies weights, releases the HF object, and moves the custom model to CPU or CUDA in evaluation mode. No HF object is retained as a model child. Exceptions preserve download/checkpoint context rather than falling back to random weights.

`tests/test_correctness_vs_hf.py` verifies actual public weights. Its synthetic tiny checkpoint checks are fast diagnostics, never replacements for the public-model acceptance tests. The eager HF attention backend is the transparent FP32 comparison; CUDA backend numerical behavior must be checked on actual CUDA hardware.

## Contiguous request cache

```mermaid
sequenceDiagram
    participant Request
    participant Model as Custom GPT-2
    participant Cache as Request KV cache
    Request->>Cache: Allocate prompt + output capacity
    Request->>Model: Prefill whole prompt
    Model->>Cache: Write each layer's K/V, then commit length
    Model-->>Request: First new token from final prompt logits
    loop Remaining output tokens
        Request->>Model: Previous sampled token only
        Model->>Cache: Append layer K/V; read committed prefix + new token
        Model->>Cache: Commit length once after logits succeed
        Model-->>Request: Next token
    end
```

`SimpleKVCache` owns key and value tensors with shape `[layers, batch, heads, capacity, head_dim]`. It is separate from the model/checkpoint. GPT-2 FP32 cache storage costs `2 × 12 × 768 × 4 = 73,728` bytes per reserved token per request. Capacity is fixed for the request, so **allocated bytes** stay constant while **used bytes** grow with committed token count. Normal generation reserves prompt plus output budget; the last sampled token need not be forwarded, so it does not need a committed KV slot. EOS may leave more unused capacity.

Absolute position IDs start at `cache.length`. A chunk of length `n` after a prefix of length `p` uses causal-mask rows `p:p+n` and key columns `:p+n`. Using rows starting at zero would wrongly hide past keys and mis-mask a multi-token chunk. Both one-token decode and chunked prefill are tested.

Every layer writes tentatively at the same committed offset; the model advances the length only after the final logits succeed. A failed late layer leaves the earlier committed prefix intact. Retrying overwrites tentative slots. Dimensions, batch, dtype/device, and capacity are validated before the first write. This cache stores inference state and does not support differentiable training through cached keys/values.

Caching saves repeated projections and MLP computation for old tokens, but attention still reads all previous keys and values. Contiguous allocation reserves the full request budget; it does not solve padding, admission, or fragmentation. Those belong to later batching and paging milestones.

## Measurement scope

`benchmarks/bench_stages.py` runs the same deterministic synthetic token IDs and fixed greedy output count through both stages, checking output equality before timing. Total generation time includes request validation, cache allocation, prefill, sampling, and decode; model/tokenizer loading and text tokenization are outside it. One warmup precedes three or more measured repetitions.

The optional `step_times` collector synchronizes CUDA only when measuring. Its first duration is time to the first generated token; later durations supply decode p50/p95. A one-token output has no decode samples, so those fields stay blank. Throughput is output-token count divided by median total generation time. Instrumentation adds timer calls on CPU and per-step synchronization on CUDA; compare these as instrumented engine measurements, not production serving results.

CPU process peak memory is unmeasured and left blank. Cache reserved bytes are actual tensor storage, not total process memory. CUDA peak memory, when run there, is PyTorch allocated tensor memory including resident model weights, excluding tokenizer/process RAM and driver reservations. Hardware and thread counts travel with each CSV row.
