"""Validated client-side SSE throughput and latency; no silent failed requests."""
import argparse
import asyncio
import csv
import json
import math
from pathlib import Path
import time

import httpx


DEFAULT_PROMPT = 'The future of machine learning is'


def percentile(values: list[float], q: float) -> float:
    if not values or not 0 <= q <= 1:
        raise ValueError('Percentile requires samples and q in [0,1]')
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    low = math.floor(index)
    high = math.ceil(index)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


async def read_generation(response: httpx.Response, *, prompt: str,
                          max_new_tokens: int, started: float) -> dict[str, object]:
    if response.status_code != 200:
        raise ValueError(f'Generation returned HTTP {response.status_code}')
    if not response.headers.get('content-type','').startswith('text/event-stream'):
        raise ValueError('Generation did not return SSE')
    kind, data_lines = None, []
    request_id = None
    tokens, deltas = [], []
    terminal = None
    first = None
    async for line in response.aiter_lines():
        if line.startswith(':'):
            continue
        if line:
            field, _, value = line.partition(':')
            value = value.removeprefix(' ')
            if field == 'event':
                kind = value
            elif field == 'data':
                data_lines.append(value)
            continue
        if kind is None and not data_lines:
            continue
        if terminal is not None:
            raise ValueError('Event after terminal generation result')
        try:
            data = json.loads('\n'.join(data_lines))
        except (ValueError, TypeError) as error:
            raise ValueError('Invalid SSE JSON') from error
        if not isinstance(data,dict) or not isinstance(data.get('request_id'),str) or not data['request_id']:
            raise ValueError('Missing request identity')
        if request_id is None:
            request_id = data['request_id']
        if data['request_id'] != request_id:
            raise ValueError('Mismatched request identity')
        if kind == 'error':
            raise ValueError(f"Generation error: {data.get('code','unknown')}")
        if kind == 'token':
            if (type(data.get('token_id')) is not int or data['token_id'] < 0
                    or not isinstance(data.get('delta'),str) or len(tokens) >= max_new_tokens):
                raise ValueError('Invalid token event or output budget exceeded')
            if first is None:
                first = time.perf_counter() - started
            tokens.append(data['token_id'])
            deltas.append(data['delta'])
        elif kind == 'done':
            terminal = data
        else:
            raise ValueError('Unknown generation event')
        kind, data_lines = None, []
    if kind is not None or data_lines or terminal is None:
        raise ValueError('Truncated generation stream')
    usage = terminal.get('usage',{})
    reason = terminal.get('finish_reason')
    ids = terminal.get('token_ids')
    if (not isinstance(ids,list) or any(type(i) is not int for i in ids) or ids != tokens
            or terminal.get('text') != prompt + ''.join(deltas)
            or not isinstance(usage,dict)
            or any(type(usage.get(k)) is not int for k in ['prompt_tokens','completion_tokens','total_tokens'])
            or usage['prompt_tokens'] < 1 or usage['completion_tokens'] != len(tokens)
            or usage['total_tokens'] != usage['prompt_tokens'] + len(tokens)
            or reason not in ('length','stop')
            or (reason == 'length' and len(tokens) != max_new_tokens)
            or (reason == 'stop' and not tokens)
            or type(terminal.get('server_peak_forward_batch_size')) is not int
            or terminal['server_peak_forward_batch_size'] < 0):
        raise ValueError('Final result disagrees with streamed tokens/text/usage')
    return dict(request_id=request_id, completion_tokens=len(tokens),
        latency_seconds=time.perf_counter()-started, ttft_seconds=first,
        server_peak_forward_batch_size=terminal['server_peak_forward_batch_size'])


async def run_load(url: str, *, requests: int = 8, concurrency: int = 4,
                   prompt: str = DEFAULT_PROMPT, max_new_tokens: int = 32,
                   timeout: float = 120.) -> dict[str, object]:
    if (any(type(n) is not int or n <= 0 for n in [requests,concurrency])
            or type(max_new_tokens) is not int or not 0 <= max_new_tokens <= 1024
            or isinstance(timeout,bool) or not isinstance(timeout,(int,float))
            or not math.isfinite(timeout) or timeout <= 0 or not isinstance(prompt,str)):
        raise ValueError('Use positive request/concurrency/finite timeout and budget 0..1024')
    payload = dict(prompt=prompt,max_new_tokens=max_new_tokens,seed=0,stop_token_ids=[])
    launches, finishes, observations = [], [], []
    async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
        async def generation():
            async with asyncio.timeout(timeout):
                started = time.perf_counter()
                async with client.stream('POST',url.rstrip('/')+'/v1/generate',json=payload) as response:
                    result = await read_generation(response,prompt=prompt,max_new_tokens=max_new_tokens,started=started)
                return started,time.perf_counter(),result
        await generation()  # Warmup is not in measured observations/timers.
        count = min(requests, concurrency)
        ready, start = asyncio.Event(), asyncio.Event()
        waiting = 0
        jobs = iter(range(requests))
        async def worker():
            nonlocal waiting
            waiting += 1
            if waiting == count:
                ready.set()
            await start.wait()
            for _ in jobs:
                launched, finished, result = await generation()
                launches.append(launched)
                finishes.append(finished)
                observations.append(result)
        tasks = [asyncio.create_task(worker()) for _ in range(count)]
        try:
            await ready.wait()
            start.set()
            await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
    if len({o['request_id'] for o in observations}) != requests:
        raise ValueError('Duplicate request IDs in measured workload')
    elapsed = max(finishes) - min(launches)
    latencies = [o['latency_seconds'] for o in observations]
    ttft = [o['ttft_seconds'] for o in observations if o['ttft_seconds'] is not None]
    useful = sum(o['completion_tokens'] for o in observations)
    return dict(requests=requests,concurrency=count,prompt=prompt,max_new_tokens=max_new_tokens,
        completion_tokens=useful,elapsed_seconds=elapsed,tokens_per_second=useful/elapsed,
        latency_p50_seconds=percentile(latencies,.5),latency_p95_seconds=percentile(latencies,.95),
        ttft_p50_seconds=percentile(ttft,.5) if ttft else None,
        ttft_p95_seconds=percentile(ttft,.95) if ttft else None,ttft_samples=len(ttft),failures=0,
        server_peak_forward_batch_size=max(o['server_peak_forward_batch_size'] for o in observations),
        batch_peak_scope='server lifetime',configuration_scope='operator declared',seed=0,stop_token_ids='[]')


def main() -> None:
    parser = argparse.ArgumentParser(description='mini-infer: validated concurrent HTTP load test')
    parser.add_argument('--url',default='http://127.0.0.1:8000')
    parser.add_argument('--requests',type=int,default=8)
    parser.add_argument('--concurrency',type=int,default=4)
    parser.add_argument('--prompt',default=DEFAULT_PROMPT)
    parser.add_argument('--max-new-tokens',type=int,default=32)
    parser.add_argument('--timeout',type=float,default=120.)
    parser.add_argument('--output',type=Path,default=Path('results/serving.csv'))
    parser.add_argument('--server-device',required=True)
    parser.add_argument('--server-hardware',required=True)
    parser.add_argument('--server-threads',type=int,required=True)
    parser.add_argument('--server-max-batch-size',type=int,required=True)
    parser.add_argument('--server-cache-backend',choices=('contiguous','paged'),required=True)
    parser.add_argument('--server-int8',action='store_true')
    args=parser.parse_args()
    if min(args.server_threads,args.server_max_batch_size)<=0:
        parser.error('Declared server thread and batch counts must be positive')
    try:
        row=asyncio.run(run_load(args.url,requests=args.requests,concurrency=args.concurrency,
            prompt=args.prompt,max_new_tokens=args.max_new_tokens,timeout=args.timeout))
        row.update(url=args.url,server_device=args.server_device,server_hardware=args.server_hardware,
            server_threads=args.server_threads,server_max_batch_size=args.server_max_batch_size,
            server_cache_backend=args.server_cache_backend,server_int8=args.server_int8,
            client_httpx_version=httpx.__version__)
        fields=list(row)
        existing=args.output.exists() and args.output.stat().st_size>0
        if existing:
            with args.output.open(newline='') as source:
                if next(csv.reader(source),[])!=fields:
                    raise ValueError('Existing CSV header differs; use a new output file')
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('a',newline='') as output:
            writer=csv.DictWriter(output,fieldnames=fields,lineterminator='\n')
            if not existing:
                writer.writeheader()
            writer.writerow(row)
    except (ValueError,httpx.HTTPError,TimeoutError) as error:
        parser.exit(1,f'Load test failed (failed_runs=1; no result row written): {error}\n')
    print(json.dumps(row,ensure_ascii=False))


if __name__=='__main__':
    main()
