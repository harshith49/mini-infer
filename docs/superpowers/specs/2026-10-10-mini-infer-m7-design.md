# mini-infer M7: streaming HTTP generation

## Goal and approved approach

Expose the existing GPT-2 scheduler through FastAPI so concurrent clients receive generated tokens while continuous batching runs. Include request cleanup, real streaming checks and a reproducible concurrent load test. The user approved the conversational design on October 10, 2026. This written spec still requires review; implementation follows approval of the subsequent written plan.

Start from M6 commit `bbbf070366eaa7ff2760dfe26f58bbab01968642` in the standalone `harshith49/mini-infer` repository. Use `codex/mini-infer-m7`, retained native execution and one independent final review. Publish a stacked draft PR against M6 if it remains unmerged. Preserve prior milestones, measurements and correctness gates.

## Architecture and scope

Add `server/app.py` with a FastAPI application and CLI. One background thread owns the model, tokenizer and Scheduler, including submission, generation, cancellation and disposal. HTTP handlers exchange commands and responses with that thread through standard-library queues and event-loop notification. They never call model forward or mutate the scheduler. Load the model once during application lifespan; readiness occurs only after successful loading. Idle workers block on commands; active workers process pending commands between scheduler steps and then advance generation.

Use FastAPI's native StreamingResponse and ordinary SSE framing. Add only FastAPI, Uvicorn and HTTPX as direct dependencies, with compatible exact versions verified in the implementation plan; HTTPX serves both TestClient and the load test. No separate SSE library, task broker, distributed worker system or new inference implementation. Direct inference in HTTP handlers would block the event loop; multiple model workers would duplicate model/cache memory. The selected single-owner worker preserves the existing batching algorithm.

Server defaults: bind `127.0.0.1`, port 8000, one process, device auto, max batch size 2, maximum outstanding streams 64, contiguous KV and FP32 weights. CLI options expose host/port, device, batch limit, outstanding-stream limit, existing paged backend/page settings and int8. Paged defaults remain 32 pages of 16 tokens. Strictly validate positive configuration values. Multiple Uvicorn processes, training, authentication, persistent sessions, reconnect/resume, full OpenAI API compatibility and M8 packaging are outside M7.

## Request and admission contract

`POST /v1/generate` accepts a JSON object with:

| Field | Default | Validation |
| --- | --- | --- |
| `prompt` | Required | String, at most 65,536 characters |
| `max_new_tokens` | 50 | Strict integer, nonnegative, bounded by model context |
| `temperature` | 0 | Finite number, nonnegative; Boolean rejected |
| `top_k` | 0 | Strict integer from zero through vocabulary size |
| `top_p` | 1 | Finite number in `(0,1]`; Boolean rejected |
| `seed` | 0 | Strict integer from zero through `2**63-1` |
| `stop_token_ids` | Tokenizer EOS ID | List of strict vocabulary integers; explicit empty list disables stopping |

Reject unknown fields and coercions of integer/Boolean/string types. Existing SamplingParams and Scheduler validations remain the authoritative model-dependent checks. Tokenize without adding special tokens; an empty tokenized prompt uses one EOS seed token. Require prompt-token count plus output budget within the model context. Paged requests with a positive output budget must fit the entire pool individually, as today. Zero output budgets remain valid and perform no forward.

The server generates a UUID request ID. Reserve an outstanding-stream slot before sending a submit command. Admission acknowledges validation and scheduler submission before returning SSE headers: schema/model-budget failures return JSON HTTP 422, exhausted outstanding-stream capacity returns 429, unavailable/stopping worker returns 503. A slot covers queued validation, waiting/running generation and completed but undrained streams; release it exactly once after the HTTP stream closes and worker cleanup is acknowledged. Cancellation during admission must dispose a submission that subsequently succeeds. Request text is not logged or stored on disk.

## SSE protocol and Unicode

Encode each event as `event: <name>\ndata: <JSON>\n\n`, with JSON escaping for line breaks and Unicode. Response content type is `text/event-stream`, with `Cache-Control: no-cache`. No SSE replay IDs or heartbeat mechanism is required for these finite generations.

* `token`: `request_id`, integer `token_id`, and string `delta`. Emit one event for every selected token, including a stop token; special tokens can have an empty text delta.
* `done`: `request_id`, `finish_reason` (`length` or `stop`), generated-only `token_ids`, final `text`, `usage` with prompt/completion/total token counts, and `server_peak_forward_batch_size`.
* `error`: `request_id` and a stable error code/message. Emit once on runtime failure after headers, then end the stream without a `done` event. Do not expose tracebacks.

`done.text` is the original prompt string plus decoded generated tokens, with special tokens skipped and tokenizer cleanup disabled. The artificial EOS seed counts as one prompt token but does not appear in text. Generated stop tokens count in IDs and usage even when their decoded text is empty. Zero budget emits only `done` with an empty generated list and length finish reason.

GPT-2 byte tokens can split a Unicode character. Decode the accumulated generated IDs with `skip_special_tokens=True, clean_up_tokenization_spaces=False`; on intermediate events defer the trailing replacement-character suffix, then emit only the new suffix of the stable decoded prefix. On the finishing token flush the full final decode, including any trailing replacement characters. Concatenated token deltas must equal final decoded generated text. A legitimate trailing replacement character may therefore be delayed until another character or completion. Verify prefix stability for the supported GPT-2 tokenizer, including split multibyte text, malformed byte sequences, literal replacement characters, special tokens and whitespace. A violated prefix invariant fails the request explicitly instead of silently revising emitted text. Cumulative decoding is bounded by the 1,024-token context; mark its quadratic ceiling with a `ponytail:` comment and defer an incremental byte decoder until larger contexts justify it.

`server_peak_forward_batch_size` measures the largest actual prefill/decode forward batch since worker startup, not HTTP concurrency or a per-request peak. Add a small Scheduler counter updated at the two forward sites; controlled tests prove multi-request forwards. Reports label the counter's lifetime scope and do not attribute a historical peak to a particular load-test run.

## Ownership, bounded buffering and failure behavior

Add explicit Scheduler disposal methods: `cancel(request_id)` removes any waiting/running/completed request, releases private KV/pages, removes pending events for that ID and is harmless if already absent; `discard(request_id)` removes a completed result and rejects an unfinished request; `close()` cancels all requests and is repeatable. Existing result retention remains unchanged until these methods are called, preserving CLI and programmatic behavior.

The worker creates the terminal result and immediately discards completed scheduler state. HTTP streams retain only their bounded response data until consumed/closed. Each response queue reserves room for at most the request's token budget plus one terminal event, so generation never waits for a slow client's network writes. At most 64 streams are retained by default. HTTP disconnect or generator cancellation always sends disposal; keep its slot reserved until the worker acknowledges cleanup, even if the HTTP task has been cancelled. A small cancellation-protected cleanup task may finish that handshake. This also bounds pending submit/disposal commands under rapid disconnects; coalesce repeated disposal for the same ID. Cancellation is processed between steps; an already executing PyTorch forward is allowed to finish. Events scheduled for a cancelled stream must not resurrect state or enter another client's queue.

If a scheduler/model step raises, fail the worker closed: notify every affected admitted stream, dispose all owned state, reject later admissions with 503 and require restart. M7 does not introduce automatic inference retries; the existing synchronous Scheduler recovery tests remain unchanged. Unexpected admission faults also fail the worker closed, whereas ordinary validation rejection leaves it usable. Unexpected worker termination must wake admission waiters and streams instead of leaving them waiting forever.

Shutdown stops new admission, finishes any current forward, wakes admission waiters, terminates streams and disposes every request before joining the worker. No daemon thread may outlive the application. A permanently stuck backend forward cannot be forcibly interrupted safely and requires process termination; document this limit. Do not claim cancellation latency shorter than one forward. Paginated pool ownership must return to zero after completion, disconnect, error and shutdown, although the fixed pool storage itself remains allocated until model teardown.

## Load test and measurements

Add `benchmarks/load_test.py` using HTTPX async streaming. Accept server URL, positive concurrency and request count, prompt, token budget and finite timeout. Run a bounded number of client tasks, synchronize the first concurrent group, and time with a monotonic clock. Default workload is 8 requests, concurrency 4, prompt `The future of machine learning is`, 32 new tokens, seed 0 and explicit empty stop list. No model loading or warmup occurs inside measured client elapsed time; perform a separate single-request warmup first.

Parse SSE incrementally and tolerate network chunks splitting lines/JSON. Validate exactly one terminal result, matching request IDs, token-event count/IDs, usage and concatenated deltas; HTTP failures or error streams fail the run rather than disappearing from the denominator. Record actual useful completion tokens divided by elapsed time from the first timed launch through the last completion, whole-request p50/p95 latency, and TTFT p50/p95 from the first token event. Define percentiles with linear interpolation of sorted observations. Zero-token requests have no TTFT and are excluded only from that statistic, with sample count reported. Client metrics include transport and SSE parsing and are distinct from existing engine-only timings.

Write `results/serving.csv` with workload, URL, actual counts, elapsed time, concurrency, timing distributions, failures, server batch limit/backend/int8/device/hardware/thread settings supplied explicitly by the benchmark command, and reported lifetime batch peak. Configuration supplied by the operator is labeled as declared, not remotely verified. Make no speedup claim without an actual comparable baseline. Run real CPU measurements in isolation using public cached GPT-2, one PyTorch CPU thread, with matching concurrency-1 and concurrency-4 workloads; retain raw measurements even if batching is slower. Prior CSVs remain unchanged; CUDA is conditional and unclaimed without hardware.

## Verification and publication

Use a tiny real custom model and injected tokenizer/model ownership for fast tests; the production CLI still uses the shared loader. TestClient covers request schema, admission errors, complete SSE content, sampling/stop/zero-budget behavior and lifespan. TestClient can buffer SSE, so it is insufficient evidence of live delivery or disconnect cleanup.

Add actual loopback Uvicorn/HTTPX streaming checks on an ephemeral port. Synchronization events and controlled model forwards demonstrate a token reaching the client before completion, another request entering while the first is active, an actual multi-request forward, and disconnect cleanup without interrupting unrelated clients. Use finite timeouts rather than timing assertions or arbitrary sleeps. Cover cancellation while waiting/admitting/running, slow consumers, capacity rejection, completed-result disposal, stale events, model failure, shutdown and restored paged ownership. Verify concurrent request output against independent generation with the same model/sampling seed; test contiguous and paged paths and an int8 tiny-model smoke case. Keep real public-model CPU load measurements separate from controlled tests.

Run the full inherited suite unchanged, new API/lifecycle/network tests, a real CLI server demonstration, load-test measurements, compileall and git whitespace checks. Document curl streaming usage, all request/event fields, single-worker ownership, slot/buffer limits, cancellation/error semantics, Unicode delay, measured CPU performance and unavailable CUDA in README, architecture and lessons. Use one fresh read-only final review, one TDD fix pass for Critical/Important findings, and record deferred Minor costs and execution rulings. Push authorized milestone commits, create and attach the draft PR, verify repository identity and local/remote/PR heads before claiming publication. M8 remains subsequent work.

## Spec self-review

This spec preserves the existing engine and limits new work to HTTP serving, explicit request lifetime ownership and client measurement. Admission happens before headers; streaming failures have a separate terminal event. Completed scheduler state is released independently from slow network consumers, while stream slots and token budgets bound retained data. Token IDs, Unicode text, EOS/empty-prompt counts, lifetime batch instrumentation and client timing denominators are explicit. Real network tests address TestClient buffering. No implementation dependencies have been installed and no product code has changed during design.
