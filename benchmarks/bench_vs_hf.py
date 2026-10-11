"""Paired sequential cached generation: custom GPT-2 versus HF generate."""
import gc
import time

import torch
from engine.weights import GPT2LMHeadModel, load_model, resolve_device

from benchmarks.stage_suite import (RunResult, Workload, arguments, make_prompts,
    measure, metadata, run_stage_once, synchronize, write_rows)
from engine.config import EngineConfig


@torch.inference_mode()
def run_hf_once(reference: GPT2LMHeadModel, prompts: list[torch.Tensor], workload: Workload) -> RunResult:
    device = reference.device
    synchronize(device)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    outputs, completions = [], []
    for prompt,budget in zip(prompts,workload.output_budgets):
        # Explicit kwargs override model defaults; a separate config with None falls back to checkpoint EOS in 4.48.3.
        outputs.append(reference.generate(prompt[None],attention_mask=torch.ones_like(prompt[None]),
            max_new_tokens=budget,do_sample=False,use_cache=True,eos_token_id=None,pad_token_id=0)[0])
        synchronize(device)
        completions.append(time.perf_counter()-started)
    synchronize(device)
    elapsed = time.perf_counter()-started
    return RunResult(outputs,completions,elapsed,None,
        torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None)


def run_comparison(workload: Workload, *, device_name: str = 'auto', cache_dir: str = 'model_cache') -> list[dict[str, object]]:
    device = resolve_device(device_name)
    model, _ = load_model(EngineConfig(device=device_name,cache_dir=cache_dir))
    prompts = make_prompts(model.config,workload,device)
    result = run_stage_once(model,prompts,workload,'kv_cache')
    expected = [x.cpu() for x in result.outputs]
    del result
    own = metadata(model,prompts,workload,'mini_infer_cached',device)
    own.update(measure(lambda: run_stage_once(model,prompts,workload,'kv_cache'),expected,workload))
    del model
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    reference = GPT2LMHeadModel.from_pretrained('gpt2',cache_dir=cache_dir,
        torch_dtype=torch.float32,attn_implementation='eager').to(device).eval()
    try:
        hf = metadata(reference,prompts,workload,'hf_cached',device)
        hf['kv_memory_scope'] = 'unmeasured_hf_cache'
        hf.update(measure(lambda: run_hf_once(reference,prompts,workload),expected,workload))
    finally:
        del reference
        gc.collect()
    return [own,hf]


def main(argv=None):
    parser,args,workload = arguments(__doc__,'results/vs_hf.csv',argv)
    torch.set_num_threads(args.threads)
    try:
        rows = run_comparison(workload,device_name=args.device)
        write_rows(args.output,rows)
    except ValueError as error:
        parser.error(str(error))
    for row in rows:
        print(f"{row['stage']}: {row['tokens_per_second']:.2f} useful tokens/s")
    print(f'Saved {len(rows)} rows to {args.output}')


if __name__ == '__main__':
    main()
