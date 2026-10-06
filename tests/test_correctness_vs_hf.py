"""Public GPT-2 is the mandatory FP32 oracle, never a synthetic substitute."""
import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel, AutoTokenizer

from engine.config import EngineConfig, ModelConfig
from engine.model import GPT2Model
from engine.weights import copy_hf_weights, load_model

PROMPTS = ['Hello, world!', 'The quick brown fox jumps over the lazy dog.',
           'Café — hello!\n  Spaces matter.']


@pytest.fixture(scope='session')
def public_models():
    reference = GPT2LMHeadModel.from_pretrained('gpt2', cache_dir='model_cache',
                                               attn_implementation='eager').float().eval()
    model, tokenizer = load_model(EngineConfig(device='cpu'))
    return model, reference, tokenizer


@pytest.mark.parametrize('prompt', PROMPTS)
def test_public_gpt2_logits(public_models, prompt):
    model, reference, tokenizer = public_models
    ids = tokenizer(prompt, return_tensors='pt')['input_ids']
    with torch.inference_mode():
        actual = model(ids)
        expected = reference(ids, use_cache=False).logits
    error = (actual - expected).abs().max().item()
    print(f'max absolute logit error: {error:.9g}')
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4,
                               msg=lambda msg: f'max error {error}: {msg}')


def test_tiny_weight_mapping_and_logits():
    torch.manual_seed(11)
    reference = GPT2LMHeadModel(GPT2Config(vocab_size=37, n_positions=16, n_embd=24,
        n_layer=2, n_head=4, attn_implementation='eager')).eval()
    model = GPT2Model(ModelConfig(vocab_size=37, max_positions=16, hidden_size=24,
        num_layers=2, num_heads=4, intermediate_size=96)).eval()
    copy_hf_weights(model, reference.state_dict())
    ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    with torch.inference_mode():
        torch.testing.assert_close(model(ids), reference(ids).logits, atol=1e-6, rtol=1e-5)
    # Every parameter participates in reference behavior; square projections
    # especially must transpose even though their shape alone cannot reveal it.
    torch.testing.assert_close(model.blocks[0].attention.projection.weight,
                                reference.transformer.h[0].attn.c_proj.weight.T)
    assert model.lm_head.weight.data_ptr() == model.token_embedding.weight.data_ptr()


@pytest.mark.parametrize('failure', ['missing', 'shape'])
def test_bad_checkpoint_does_not_partially_modify_model(failure):
    cfg = ModelConfig(vocab_size=37, max_positions=16, hidden_size=24,
                      num_layers=2, num_heads=4, intermediate_size=96)
    model = GPT2Model(cfg)
    state = GPT2LMHeadModel(GPT2Config(vocab_size=37, n_positions=16, n_embd=24,
                           n_layer=2, n_head=4)).state_dict()
    key = 'transformer.h.1.mlp.c_proj.weight'
    if failure == 'missing':
        del state[key]
    else:
        state[key] = torch.zeros(1, 1)
    before = model.token_embedding.weight.detach().clone()
    with pytest.raises(ValueError, match=key.replace('.', r'\.')) as error:
        copy_hf_weights(model, state)
    assert 'expected' in str(error.value)
    torch.testing.assert_close(model.token_embedding.weight, before, atol=0, rtol=0)


def test_loading_failure_preserves_context(monkeypatch):
    from engine import weights

    def offline(*args, **kwargs):
        raise OSError('not cached and network unavailable')

    monkeypatch.setattr(weights.GPT2Config, 'from_pretrained', offline)
    with pytest.raises(OSError, match='network unavailable'):
        load_model(EngineConfig(device='cpu'))


@pytest.mark.parametrize('prompt', PROMPTS)
def test_public_gpt2_greedy_50_tokens(public_models, prompt):
    from engine.generate import generate

    model, reference, tokenizer = public_models
    ids = tokenizer(prompt, return_tensors='pt')['input_ids']
    actual = generate(model, ids, 50)
    with torch.inference_mode():
        expected = reference.generate(ids, attention_mask=torch.ones_like(ids),
            max_new_tokens=50, do_sample=False, use_cache=False, eos_token_id=None,
            pad_token_id=tokenizer.eos_token_id)
    assert expected.shape[1] == ids.shape[1] + 50
    assert torch.equal(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA hardware unavailable')
def test_public_gpt2_cuda_parity(public_models):
    from engine.generate import generate
    model, reference, tokenizer = public_models
    ids = tokenizer(PROMPTS[0], return_tensors='pt')['input_ids'].to('cuda')
    try:
        model.to('cuda')
        reference.to('cuda')
        with torch.inference_mode():
            torch.testing.assert_close(model(ids), reference(ids, use_cache=False).logits,
                                       atol=1e-4, rtol=1e-4)
            expected = reference.generate(ids, attention_mask=torch.ones_like(ids),
                max_new_tokens=50, do_sample=False, use_cache=False, eos_token_id=None,
                pad_token_id=tokenizer.eos_token_id)
        assert torch.equal(generate(model, ids, 50), expected)
        assert torch.equal(generate(model, ids, 50, use_cache=True), expected)
    finally:
        model.cpu()
        reference.cpu()


def test_loading_incomplete_checkpoint_rejects_random_fallback(tmp_path, public_models):
    from safetensors.torch import load_file, save_file

    _, _, tokenizer = public_models
    checkpoint = tmp_path / 'damaged-gpt2'
    reference = GPT2LMHeadModel(GPT2Config(vocab_size=37, n_positions=16, n_embd=24,
                                          n_layer=2, n_head=4))
    reference.save_pretrained(checkpoint)
    tokenizer.save_pretrained(checkpoint)
    path = checkpoint / 'model.safetensors'
    state = load_file(path)
    missing_key = 'transformer.h.1.mlp.c_proj.weight'
    del state[missing_key]
    save_file(state, path, metadata={'format': 'pt'})
    with pytest.raises(ValueError, match=missing_key.replace('.', r'\.')):
        load_model(EngineConfig(device='cpu', model_name=str(checkpoint)))


@pytest.mark.parametrize('prompt', PROMPTS)
def test_public_gpt2_cached_greedy_50_tokens(public_models, prompt):
    from engine.generate import generate

    model, reference, tokenizer = public_models
    ids = tokenizer(prompt, return_tensors='pt')['input_ids']
    cached = generate(model, ids, 50, use_cache=True)
    baseline = generate(model, ids, 50)
    with torch.inference_mode():
        expected = reference.generate(ids, attention_mask=torch.ones_like(ids),
            max_new_tokens=50, do_sample=False, use_cache=False, eos_token_id=None,
            pad_token_id=tokenizer.eos_token_id)
    assert torch.equal(cached, baseline)
    assert torch.equal(cached, expected)


@pytest.mark.parametrize('prompt', PROMPTS)
@torch.inference_mode()
def test_public_gpt2_cached_suffix_logits(public_models, prompt):
    from engine.kv_cache import SimpleKVCache

    model, _, tokenizer = public_models
    ids = tokenizer(prompt, return_tensors='pt')['input_ids']
    expected = model(ids)
    for chunks in [[1] * ids.shape[1], [1, ids.shape[1] - 1]]:
        cache = SimpleKVCache(model.config, batch_size=1, capacity=ids.shape[1],
                              device=torch.device('cpu'), dtype=torch.float32)
        offset = 0
        for size in chunks:
            actual = model(ids[:, offset:offset + size], cache=cache)
            error = (actual - expected[:, offset:offset + size]).abs().max().item()
            print(f'cached suffix max absolute error: {error:.9g}')
            torch.testing.assert_close(actual, expected[:, offset:offset + size],
                                       atol=1e-4, rtol=1e-4)
            offset += size


@pytest.fixture(scope='session')
def public_batch_cases(public_models):
    model, reference, tokenizer = public_models
    texts = [PROMPTS[0], PROMPTS[2], 'The quick brown fox jumps over the lazy dog. ' * 13]
    prompts = [tokenizer(text, return_tensors='pt')['input_ids'][0] for text in texts]
    assert len(prompts[2]) >= 128
    expected = []
    from engine.generate import generate
    with torch.inference_mode():
        for prompt in prompts:
            hf = reference.generate(prompt[None], attention_mask=torch.ones_like(prompt[None]),
                max_new_tokens=50, do_sample=False, use_cache=True, eos_token_id=None,
                pad_token_id=tokenizer.eos_token_id)[0]
            assert len(hf) == len(prompt) + 50
            for use_cache in [False, True]:
                assert torch.equal(generate(model, prompt[None], 50, use_cache=use_cache)[0], hf)
            expected.append(hf)
    return prompts, expected


@torch.inference_mode()
def test_public_gpt2_masked_full_and_cached_suffix_logits(public_models):
    from engine.kv_cache import SimpleKVCache
    model, reference, tokenizer = public_models
    prompts = [tokenizer(text, return_tensors='pt')['input_ids'][0] for text in PROMPTS]
    width = max(map(len, prompts))
    ids = torch.full((len(prompts), width), tokenizer.eos_token_id, dtype=torch.long)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    for row, prompt in enumerate(prompts):
        ids[row, -len(prompt):] = prompt
        mask[row, -len(prompt):] = True
    positions = (mask.long().cumsum(-1) - 1).masked_fill(~mask, 0)
    actual = model(ids, attention_mask=mask)
    padded_reference = reference(ids, attention_mask=mask, position_ids=positions, use_cache=False).logits
    cache = SimpleKVCache(model.config, batch_size=len(prompts), capacity=width+2,
        device=torch.device('cpu'), dtype=torch.float32)
    model(ids, cache=cache, attention_mask=mask)
    suffix = torch.tensor([[15496, 11]] * len(prompts))
    cached = model(suffix, cache=cache,
        attention_mask=torch.cat((mask, torch.ones_like(suffix, dtype=torch.bool)), dim=1))
    for row, prompt in enumerate(prompts):
        hf = reference(prompt[None], use_cache=False).logits[0]
        solo = model(prompt[None])[0]
        for expected in [hf, solo, padded_reference[row, -len(prompt):]]:
            torch.testing.assert_close(actual[row, -len(prompt):], expected, atol=1e-4, rtol=1e-4)
        complete = torch.cat((prompt, suffix[row]))[None]
        for expected in [model(complete)[0, -2:], reference(complete, use_cache=False).logits[0, -2:]]:
            torch.testing.assert_close(cached[row], expected, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize('use_cache', [False, True])
@pytest.mark.parametrize('order', [[0, 1, 2], [2, 1, 0], [1, 2, 0], [1]])
def test_public_gpt2_batch_greedy_50_tokens(public_models, public_batch_cases, use_cache, order):
    from engine.batching import generate_batch
    model, _, tokenizer = public_models
    prompts, expected = public_batch_cases
    actual = generate_batch(model, [prompts[i] for i in order], 50,
                            pad_token_id=tokenizer.eos_token_id, use_cache=use_cache)
    assert len(actual) == len(order)
    for row, index in zip(actual, order):
        assert len(row) == len(prompts[index]) + 50
        assert torch.equal(row, expected[index])


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA hardware unavailable')
@torch.inference_mode()
def test_public_gpt2_cuda_batch_parity(public_models):
    import copy
    from engine.batching import generate_batch
    model, _, tokenizer = public_models
    cuda_model = copy.deepcopy(model).to('cuda').eval()
    prompts = [tokenizer(text, return_tensors='pt')['input_ids'][0] for text in PROMPTS[:2]]
    cuda_prompts = [prompt.to('cuda') for prompt in prompts]
    for use_cache in [False, True]:
        cpu = generate_batch(model, prompts, 50, pad_token_id=tokenizer.eos_token_id, use_cache=use_cache)
        gpu = generate_batch(cuda_model, cuda_prompts, 50, pad_token_id=tokenizer.eos_token_id, use_cache=use_cache)
        assert all(torch.equal(a, b.cpu()) for a, b in zip(cpu, gpu))
    ids = torch.tensor([[0, 0, 1], [2, 3, 4]])
    mask = torch.tensor([[0, 0, 1], [1, 1, 1]], dtype=torch.bool)
    torch.testing.assert_close(model(ids, attention_mask=mask),
        cuda_model(ids.cuda(), attention_mask=mask.cuda()).cpu(), atol=1e-4, rtol=1e-4)
    assert model.token_embedding.weight.device.type == 'cpu'


@pytest.mark.parametrize('limit', [1, 2])
@torch.inference_mode()
def test_public_gpt2_continuous_greedy_50_tokens(public_models, public_batch_cases, limit, monkeypatch):
    from engine.scheduler import Scheduler
    model, reference, tokenizer = public_models
    prompts, expected = public_batch_cases
    scheduler = Scheduler(model, max_batch_size=limit, pad_token_id=tokenizer.eos_token_id)
    for i in [0, 1]:
        scheduler.submit(str(i), prompts[i], 50)
    # Check representative cached suffixes for every prompt/admission regime.
    # Over long incremental histories, HF itself differs from its full-prefix
    # reduction near zero logits; exact 50-token outputs remain mandatory.
    original_decode = scheduler._decode
    def checked_decode(requests):
        if not requests:
            return original_decode(requests)
        snapshots = [(r.output.clone(), len(r.output) - len(r.prompt)) for r in requests]
        def check(module, args, logits):
            for row, (ids, generated) in enumerate(snapshots):
                if generated > 8:
                    continue
                prefix = reference(ids[None, :-1], use_cache=True)
                want = reference(ids[None, -1:], past_key_values=prefix.past_key_values,
                                 use_cache=True).logits[0]
                torch.testing.assert_close(logits[row], want, atol=1e-4, rtol=1e-4)
        hook = model.register_forward_hook(check)
        try:
            return original_decode(requests)
        finally:
            hook.remove()
    monkeypatch.setattr(scheduler, '_decode', checked_decode)
    for _ in range(3):
        scheduler.step()
    scheduler.submit('2', prompts[2], 50)
    while not scheduler.idle:
        scheduler.step()
    for i in range(3):
        assert torch.equal(scheduler.result(str(i)), expected[i])
    assert scheduler.cache_allocated_bytes == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA hardware unavailable')
@torch.inference_mode()
def test_public_gpt2_cuda_scheduler_parity(public_models):
    import copy
    from engine.scheduler import Scheduler
    from engine.sampler import SamplingParams
    model, _, tokenizer = public_models
    cuda_model = copy.deepcopy(model).cuda().eval()
    prompts = [tokenizer(text, return_tensors='pt')['input_ids'][0] for text in PROMPTS[:2]]
    def run(target, prompts, settings):
        scheduler = Scheduler(target, max_batch_size=2, pad_token_id=tokenizer.eos_token_id)
        for i, prompt in enumerate(prompts):
            scheduler.submit(str(i), prompt, 50, sampling=settings)
        while not scheduler.idle:
            scheduler.step()
        return [scheduler.result(str(i)) for i in range(len(prompts))]
    cpu = run(model, prompts, SamplingParams())
    gpu_prompts = [prompt.cuda() for prompt in prompts]
    gpu = run(cuda_model, gpu_prompts, SamplingParams())
    assert all(torch.equal(a, b.cpu()) for a, b in zip(cpu, gpu))
    settings = SamplingParams(temperature=0.8, top_k=20, top_p=0.9, seed=7)
    mixed = run(cuda_model, gpu_prompts, settings)
    alone = run(cuda_model, gpu_prompts[:1], settings)
    assert torch.equal(mixed[0], alone[0])
    assert model.token_embedding.weight.device.type == 'cpu'


@pytest.mark.parametrize('index', [0, 1, 2])
@torch.inference_mode()
def test_public_gpt2_direct_paged_50_tokens(public_models, public_batch_cases, index):
    from engine.kv_cache import PagePool, PagedKVCache, SimpleKVCache
    model, _, _ = public_models
    prompts, expected = public_batch_cases
    prompt = prompts[index]
    capacity = len(prompt) + 50
    pool = PagePool(model.config, num_pages=(capacity + 15) // 16, page_size=16,
                    device=prompt.device, dtype=model.token_embedding.weight.dtype)
    paged = PagedKVCache(pool, capacity=capacity)
    contiguous = SimpleKVCache(model.config, batch_size=1, capacity=capacity,
                               device=prompt.device, dtype=model.token_embedding.weight.dtype)
    output = prompt[None].clone()
    try:
        for step in range(50):
            current = output if step == 0 else output[:, -1:]
            actual = model(current, cache=paged)
            want = model(current, cache=contiguous)
            torch.testing.assert_close(actual, want, atol=1e-4, rtol=1e-4)
            token = actual[:, -1].argmax(-1)
            del actual, want
            output = torch.cat((output, token[:, None]), dim=1)
        assert torch.equal(output[0], expected[index])
    finally:
        paged.close()
    assert pool.free_pages == pool.num_pages and pool.owned_pages == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA hardware unavailable')
@torch.inference_mode()
def test_public_gpt2_cuda_direct_paged_parity(public_models):
    import copy
    from engine.kv_cache import PagePool, PagedKVCache, SimpleKVCache
    model, _, tokenizer = public_models
    target = copy.deepcopy(model).cuda().eval()
    prompt = tokenizer(PROMPTS[0], return_tensors='pt')['input_ids'].cuda()
    capacity = prompt.shape[1] + 50
    pool = PagePool(target.config, num_pages=(capacity + 15) // 16,
                    device=prompt.device, dtype=target.token_embedding.weight.dtype)
    paged = PagedKVCache(pool, capacity=capacity)
    contiguous = SimpleKVCache(target.config, batch_size=1, capacity=capacity,
                               device=prompt.device, dtype=target.token_embedding.weight.dtype)
    output = prompt.clone()
    try:
        for step in range(50):
            current = output if step == 0 else output[:, -1:]
            actual = target(current, cache=paged)
            expected = target(current, cache=contiguous)
            torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
            assert torch.equal(actual[:, -1].argmax(-1), expected[:, -1].argmax(-1))
            output = torch.cat((output, actual[:, -1].argmax(-1)[:, None]), dim=1)
            del actual, expected
    finally:
        paged.close()
    assert pool.free_pages == pool.num_pages
    assert model.token_embedding.weight.device.type == 'cpu'


@pytest.mark.parametrize('limit', [1, 2])
@pytest.mark.parametrize('budgets', [[50, 50, 50], [5, 50, 50]])
@torch.inference_mode()
def test_public_gpt2_paged_scheduler_parity_and_pressure(public_models, public_batch_cases, limit, budgets, monkeypatch):
    from engine.kv_cache import PagePool
    from engine.scheduler import Scheduler
    model, reference, tokenizer = public_models
    prompts, expected = public_batch_cases
    pool = PagePool(model.config, num_pages=14, page_size=16,
                    device=prompts[0].device, dtype=model.token_embedding.weight.dtype)
    scheduler = Scheduler(model, max_batch_size=limit, pad_token_id=tokenizer.eos_token_id, page_pool=pool)
    for i in [0, 1]:
        scheduler.submit(str(i), prompts[i], budgets[i])
    original = scheduler._decode
    def checked_decode(requests):
        snapshots = [(r.output.clone(), len(r.output) - len(r.prompt)) for r in requests]
        def check(module, args, logits):
            for row, (ids, generated) in enumerate(snapshots):
                if generated > 8:
                    continue
                prefix = reference(ids[None, :-1], use_cache=True)
                want = reference(ids[None, -1:], past_key_values=prefix.past_key_values, use_cache=True).logits[0]
                torch.testing.assert_close(logits[row], want, atol=1e-4, rtol=1e-4)
        hook = model.register_forward_hook(check)
        try:
            return original(requests)
        finally:
            hook.remove()
    monkeypatch.setattr(scheduler, '_decode', checked_decode)
    for _ in range(3):
        scheduler.step()
    scheduler.submit('2', prompts[2], budgets[2])
    page_blocked = False
    while not scheduler.idle:
        if scheduler._waiting and len(scheduler._running) < limit:
            request = scheduler._requests[scheduler._waiting[0]]
            page_blocked |= scheduler._required_pages(len(request.prompt) + request.max_new_tokens) > pool.free_pages
        scheduler.step()
    for i in range(3):
        assert torch.equal(scheduler.result(str(i)), expected[i][:len(prompts[i]) + budgets[i]])
    if limit == 2 and budgets[0] == 5:
        assert page_blocked
    assert pool.free_pages == pool.num_pages and scheduler.cache_allocated_bytes == 0
    assert scheduler.pool_resident_bytes == pool.allocated_bytes


@torch.inference_mode()
def test_public_gpt2_paged_sampling_same_device(public_models):
    from engine.kv_cache import PagePool
    from engine.scheduler import Scheduler
    from engine.sampler import SamplingParams
    model, _, tokenizer = public_models
    prompts = [tokenizer(text, return_tensors='pt')['input_ids'][0] for text in PROMPTS[:2]]
    budgets = [5, 20]
    pages = max((len(p) + n + 15) // 16 for p, n in zip(prompts, budgets))
    outputs = []
    settings = SamplingParams(temperature=.8, top_k=20, top_p=.9, seed=7)
    for paged in [False, True]:
        pool = PagePool(model.config, num_pages=pages, device=prompts[0].device,
                        dtype=model.token_embedding.weight.dtype) if paged else None
        scheduler = Scheduler(model, max_batch_size=2, pad_token_id=tokenizer.eos_token_id, page_pool=pool)
        for i, (prompt, budget) in enumerate(zip(prompts, budgets)):
            scheduler.submit(str(i), prompt, budget, sampling=settings)
        while not scheduler.idle:
            scheduler.step()
        outputs.append([scheduler.result(str(i)) for i in range(2)])
        if pool is not None:
            assert pool.free_pages == pool.num_pages
    assert all(torch.equal(a, b) for a, b in zip(*outputs))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA hardware unavailable')
@torch.inference_mode()
def test_public_gpt2_cuda_paged_scheduler_parity(public_models):
    import copy
    from engine.kv_cache import PagePool
    from engine.scheduler import Scheduler
    from engine.sampler import SamplingParams
    model, _, tokenizer = public_models
    target = copy.deepcopy(model).cuda().eval()
    prompts = [tokenizer(text, return_tensors='pt')['input_ids'][0].cuda() for text in PROMPTS[:2]]
    def run(prompts, paged, settings):
        pool = PagePool(target.config, num_pages=64, device=prompts[0].device,
                        dtype=target.token_embedding.weight.dtype) if paged else None
        scheduler = Scheduler(target, max_batch_size=2, pad_token_id=tokenizer.eos_token_id, page_pool=pool)
        for i, prompt in enumerate(prompts):
            scheduler.submit(str(i), prompt, 50, sampling=settings)
        while not scheduler.idle:
            scheduler.step()
        if pool is not None:
            assert pool.free_pages == pool.num_pages
        return [scheduler.result(str(i)) for i in range(len(prompts))]
    greedy = SamplingParams()
    assert all(torch.equal(a, b) for a, b in zip(run(prompts, False, greedy), run(prompts, True, greedy)))
    sampled = SamplingParams(temperature=.8, top_k=20, top_p=.9, seed=7)
    assert torch.equal(run(prompts, True, sampled)[0], run(prompts[:1], True, sampled)[0])
    assert model.token_embedding.weight.device.type == 'cpu'


@pytest.fixture(scope='session')
def public_int8_model():
    model,_=load_model(EngineConfig(device='cpu',int8=True))
    return model


@pytest.mark.parametrize('prompt',PROMPTS)
def test_public_int8_logit_error(public_models,public_int8_model,prompt):
    fp32,_,tokenizer=public_models
    ids=tokenizer(prompt,return_tensors='pt')['input_ids']
    with torch.inference_mode():
        difference=(public_int8_model(ids)-fp32(ids)).double()
    maximum=difference.abs().max().item();rmse=difference.square().mean().sqrt().item()
    # A per-token vocabulary offset cancels in softmax. Keep raw errors visible.
    centered = difference - difference.mean(dim=-1, keepdim=True)
    centered_max = centered.abs().max().item()
    centered_rmse = centered.square().mean().sqrt().item()
    print(f'int8 raw max={maximum:.9g}, raw rmse={rmse:.9g}; '
          f'centered max={centered_max:.9g}, centered rmse={centered_rmse:.9g}')
    assert centered_max <= 2.0 and centered_rmse <= .25
    assert isinstance(fp32.blocks[0].attention.qkv,torch.nn.Linear)


@pytest.mark.parametrize('limit',[1,2])
@pytest.mark.parametrize('pressure',[False,True])
def test_public_int8_cached_and_paged_50_tokens(public_models,public_int8_model,limit,pressure):
    from engine.generate import generate
    from engine.scheduler import Scheduler
    from engine.kv_cache import PagePool
    _,_,tokenizer=public_models;model=public_int8_model
    prompts=[tokenizer(text,return_tensors='pt')['input_ids'][0] for text in PROMPTS]
    expected=[generate(model,p[None],50,use_cache=True)[0] for p in prompts]
    pages=[(len(p)+50+15)//16 for p in prompts]
    pool=PagePool(model.config,num_pages=max(pages) if pressure else sum(pages),page_size=16,
        device=torch.device('cpu'),dtype=torch.float32)
    for backend in [None,pool]:
        scheduler=Scheduler(model,max_batch_size=limit,pad_token_id=tokenizer.eos_token_id,page_pool=backend)
        for i,prompt in enumerate(prompts):scheduler.submit(str(i),prompt,50)
        while not scheduler.idle:scheduler.step()
        assert all(torch.equal(scheduler.result(str(i)),value) for i,value in enumerate(expected))
    assert pool.free_pages==pool.num_pages


def test_quantized_public_sample_perplexity_delta(public_models,public_int8_model):
    import hashlib
    from benchmarks.bench_quantization import load_quality_text,quality_input_ids,evaluate_perplexity
    fp32,_,tokenizer=public_models
    ids=quality_input_ids(tokenizer,load_quality_text())
    original=evaluate_perplexity(fp32,ids);quantized=evaluate_perplexity(public_int8_model,ids)
    delta=quantized['perplexity']/original['perplexity']-1
    print(f'quality ids sha256={hashlib.sha256(ids.numpy().tobytes()).hexdigest()} FP32={original} int8={quantized} relative_delta={delta:.9g}')
    assert original['scored_tokens']==quantized['scored_tokens']==4096
    assert delta<=.05


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA hardware unavailable')
def test_public_int8_cuda_error_and_cache(public_models):
    from copy import deepcopy
    from engine.quantize import quantize_model
    from engine.generate import generate
    from engine.scheduler import Scheduler
    from engine.kv_cache import PagePool
    fp32,_,tokenizer=public_models
    baseline=deepcopy(fp32).cuda();model=quantize_model(deepcopy(fp32)).cuda()
    prompt=tokenizer(PROMPTS[0],return_tensors='pt')['input_ids'].cuda()
    with torch.inference_mode():
        error=(model(prompt)-baseline(prompt)).double()
    print(f'CUDA int8 raw max={error.abs().max().item()}, raw rmse={error.square().mean().sqrt().item()}')
    centered = error - error.mean(dim=-1, keepdim=True)
    assert centered.abs().max().item() <= 2 and centered.square().mean().sqrt().item() <= .25
    expected=generate(model,prompt,50,use_cache=True)[0]
    pool=PagePool(model.config,num_pages=4,page_size=16,device=torch.device('cuda'),dtype=torch.float32)
    s=Scheduler(model,max_batch_size=1,pad_token_id=tokenizer.eos_token_id,page_pool=pool)
    s.submit('a',prompt[0],50)
    while not s.idle:s.step()
    assert torch.equal(s.result('a'),expected) and pool.free_pages==4
