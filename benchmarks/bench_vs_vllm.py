"""Optional vLLM 0.8.5 offline comparison; use a separate CUDA environment."""
import importlib.util
import time

import torch
from engine.weights import GPT2Config  # Initialize the shared local HF cache boundary first.

from benchmarks.stage_suite import (RunResult, arguments, make_prompts, measure,
    write_rows, hardware_name)
from engine.config import ModelConfig


def run_vllm(workload):
    import vllm
    # Verified against https://docs.vllm.ai/en/v0.8.5/api/offline_inference/llm.html
    cfg = GPT2Config.from_pretrained('gpt2',cache_dir='model_cache')
    config = ModelConfig(vocab_size=cfg.vocab_size,max_positions=cfg.n_positions,
        hidden_size=cfg.n_embd,num_layers=cfg.n_layer,num_heads=cfg.n_head,
        intermediate_size=cfg.n_inner or 4*cfg.n_embd)
    prompts = make_prompts(config,workload,torch.device('cpu'))
    engine = vllm.LLM(model='gpt2',dtype='float32',seed=workload.seed,
        tensor_parallel_size=1,enforce_eager=True,skip_tokenizer_init=True,
        max_model_len=config.max_positions,max_num_seqs=workload.max_batch_size,
        gpu_memory_utilization=.5,download_dir='model_cache')
    inputs = [{'prompt_token_ids':p.tolist()} for p in prompts]
    sampling = [vllm.SamplingParams(temperature=0,max_tokens=b,ignore_eos=True,detokenize=False)
                for b in workload.output_budgets]
    def run():
        started = time.perf_counter()
        results = engine.generate(inputs,sampling_params=sampling,use_tqdm=False)
        elapsed = time.perf_counter()-started
        if len(results) != len(prompts):
            raise RuntimeError('vLLM returned wrong request count')
        outputs = []
        for result,prompt,budget in zip(results,prompts,workload.output_budgets):
            if (not result.finished or result.prompt_token_ids != prompt.tolist()
                    or len(result.outputs) != 1 or len(result.outputs[0].token_ids) != budget):
                raise RuntimeError('vLLM returned mismatched prompts or output budgets')
            outputs.append(torch.tensor(prompt.tolist()+list(result.outputs[0].token_ids)))
        # Blocking offline generate exposes whole-cohort return, not live per-request delivery.
        return RunResult(outputs,[elapsed]*len(prompts),elapsed,None,None)
    warmup = run()
    expected = warmup.outputs
    del warmup
    row = measure(run,expected,workload)
    import json
    row.update(stage='vllm',device='cuda',hardware=hardware_name(torch.device('cuda')),
        vllm_version=vllm.__version__,torch_version=str(torch.__version__),dtype='float32',
        prompt_lengths=json.dumps(workload.prompt_lengths),output_budgets=json.dumps(workload.output_budgets),
        seed=workload.seed,max_batch_size=workload.max_batch_size,repetitions=workload.repetitions,warmups=1,
        latency_scope='common_submission_to_blocking_cohort_return_seconds',
        workload='synthetic token IDs; fixed outputs; greedy; no EOS stop',
        engine_configuration='single GPU; eager; max_num_seqs=batch limit; memory utilization=0.5',
        memory_scope='unmeasured_vllm_worker_processes')
    return [row]


def main(argv=None):
    parser,args,workload = arguments(__doc__,'results/vs_vllm.csv',argv)
    if args.device == 'cpu':
        parser.error('The optional vLLM comparison requires CUDA')
    if importlib.util.find_spec('vllm') is None:
        print('SKIP: vLLM is not installed; no measurement file written')
        return
    if not torch.cuda.is_available():
        print('SKIP: CUDA is unavailable; no measurement file written')
        return
    torch.set_num_threads(args.threads)
    rows = run_vllm(workload)  # Broken installation/API/runtime errors are deliberately visible.
    write_rows(args.output,rows)
    print(f'Saved {len(rows)} rows to {args.output}')


if __name__ == '__main__':
    main()
