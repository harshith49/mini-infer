"""Single owner, finite channels, cancellation and text correctness."""
import asyncio
import threading

import pytest
import torch

from engine.config import ModelConfig
from engine.model import GPT2Model
from engine.generate import generate
from engine.quantize import quantize_model
from engine.sampler import SamplingParams


class Tokenizer:
    eos_token_id = 0
    def __call__(self, text, **kwargs):
        return {'input_ids': torch.tensor([[1 + ord(c) % 36 for c in text]])}
    def decode(self, ids, **kwargs):
        return ''.join(chr(65 + i % 26) for i in ids if i != 0)


def tiny_model():
    torch.manual_seed(31)
    return GPT2Model(ModelConfig(vocab_size=37, max_positions=32, hidden_size=24,
        num_layers=2, num_heads=4, intermediate_size=96)).eval()


def payload(prompt='A', budget=5, **kwargs):
    return dict(prompt=prompt, max_new_tokens=budget, temperature=0., top_k=0,
                top_p=1., seed=0, stop_token_ids=[], **kwargs)


async def collect(handle):
    events = []
    while True:
        event = await asyncio.wait_for(handle.events.get(), 10)
        events.append(event)
        if event[0] in ('done', 'error'):
            return events


@pytest.mark.parametrize('backend,int8', [('contiguous', False), ('paged', False), ('paged', True)])
def test_worker_single_owner_and_independent_outputs(backend, int8):
    from server.worker import GenerationWorker
    model = tiny_model()
    if int8:
        quantize_model(model)
    tokenizer = Tokenizer()
    threads = []
    def load():
        threads.append(threading.get_ident())
        return model, tokenizer
    hook = model.register_forward_pre_hook(lambda *a: threads.append(threading.get_ident()))
    async def check():
        worker = GenerationWorker(load, cache_backend=backend, num_pages=16, page_size=4)
        await worker.start()
        a, b = await asyncio.gather(worker.submit(payload()), worker.submit(payload('BC', 3)))
        outputs = await asyncio.gather(collect(a), collect(b))
        for h, events, text, budget in zip([a,b], outputs, ['A','BC'], [5,3]):
            done = events[-1][1]
            assert events[-1][0] == 'done' and len(events) == budget + 1
            assert ''.join(e['delta'] for kind,e in events if kind == 'token') == done['text'][len(text):]
            await worker.release(h)
        assert not worker.scheduler._requests
        await worker.stop()
        assert not worker.thread.is_alive()
        if worker.scheduler.page_pool:
            assert worker.scheduler.page_pool.owned_pages == 0
        return outputs
    outputs = asyncio.run(check())
    hook.remove()
    assert len(set(threads)) == 1 and threads[0] != threading.get_ident()
    for events, text, budget in zip(outputs, ['A','BC'], [5,3]):
        expected = generate(model, tokenizer(text)['input_ids'], budget, use_cache=True)[0, len(text):].tolist()
        assert events[-1][1]['token_ids'] == expected


def test_worker_slots_remain_reserved_until_disposal_ack():
    from server.worker import GenerationWorker, AdmissionError
    model = tiny_model()
    entered, unblock = threading.Event(), threading.Event()
    def block(*args):
        entered.set()
        assert unblock.wait(10)
    hook = model.register_forward_pre_hook(block)
    async def check():
        worker = GenerationWorker(lambda: (model, Tokenizer()), max_outstanding=1, cache_backend='paged')
        await worker.start()
        h = await worker.submit(payload())
        assert await asyncio.to_thread(entered.wait, 5)
        releasing = asyncio.create_task(worker.release(h))
        await asyncio.sleep(0)
        with pytest.raises(AdmissionError) as exc:
            await worker.submit(payload())
        assert exc.value.status_code == 429
        unblock.set()
        await asyncio.wait_for(releasing, 5)
        await worker.release(h)
        assert not worker._handles and worker.scheduler.page_pool.owned_pages == 0
        await worker.stop()
    try:
        asyncio.run(check())
    finally:
        unblock.set()
        hook.remove()


def test_worker_cancel_during_admission_and_rapid_cycles():
    from server.worker import GenerationWorker
    model = tiny_model()
    entered, unblock = threading.Event(), threading.Event()
    def block(*args):
        entered.set()
        assert unblock.wait(10)
    hook = model.register_forward_pre_hook(block)
    async def check():
        worker = GenerationWorker(lambda: (model, Tokenizer()), max_outstanding=2)
        await worker.start()
        first = await worker.submit(payload())
        assert await asyncio.to_thread(entered.wait, 5)
        pending = asyncio.create_task(worker.submit(payload()))
        await asyncio.sleep(0)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert len(worker._handles) == 2
        unblock.set()
        await worker.release(first)
        # Event-driven barrier: stop joins worker and completes every cleanup.
        await worker.stop()
        assert not worker._handles and worker.scheduler.idle
        for _ in range(20):
            w = GenerationWorker(lambda: (model, Tokenizer()))
            await w.start()
            h = await w.submit(payload(budget=0))
            await w.release(h)
            await w.stop()
            assert not w._handles and w._commands.empty()
    try:
        asyncio.run(check())
    finally:
        unblock.set()
        hook.remove()


def test_worker_slow_consumer_is_bounded_and_does_not_block_other_jobs():
    from server.worker import GenerationWorker
    async def check():
        worker = GenerationWorker(lambda: (tiny_model(), Tokenizer()))
        await worker.start()
        slow = await worker.submit(payload(budget=12))
        peer = await worker.submit(payload(budget=3))
        assert (await collect(peer))[-1][0] == 'done'
        assert (await collect(slow))[-1][0] == 'done'
        assert slow.events.maxsize == 13 and len(worker._handles) == 2
        assert not worker.scheduler._requests
        await worker.release(slow)
        await worker.release(peer)
        await worker.stop()
        assert not worker._handles
    asyncio.run(check())


def test_worker_fatal_fault_wakes_all_waiters_and_requires_restart():
    from server.worker import GenerationWorker, AdmissionError
    model = tiny_model()
    entered, unblock = threading.Event(), threading.Event()
    def fail(*args):
        entered.set()
        assert unblock.wait(10)
        raise RuntimeError('private backend detail')
    hook = model.register_forward_pre_hook(fail)
    async def check():
        worker = GenerationWorker(lambda: (model, Tokenizer()), cache_backend='paged')
        await worker.start()
        first = await worker.submit(payload())
        assert await asyncio.to_thread(entered.wait, 5)
        pending = asyncio.create_task(worker.submit(payload()))
        await asyncio.sleep(0)
        unblock.set()
        with pytest.raises(AdmissionError) as exc:
            await asyncio.wait_for(pending, 5)
        assert exc.value.status_code == 503
        events = await collect(first)
        assert [kind for kind, _ in events] == ['error']
        assert 'private backend detail' not in str(events)
        with pytest.raises(AdmissionError):
            await worker.submit(payload())
        await worker.stop()
        assert not worker._handles and not worker.thread.is_alive()
        assert worker.scheduler.page_pool.owned_pages == 0
    try:
        asyncio.run(check())
    finally:
        unblock.set()
        hook.remove()


def test_worker_validation_startup_and_repeat_stop():
    from server.worker import GenerationWorker, AdmissionError
    async def check():
        worker = GenerationWorker(lambda: (tiny_model(), Tokenizer()))
        await worker.start()
        with pytest.raises(AdmissionError) as exc:
            await worker.submit(payload(budget=100))
        assert exc.value.status_code == 422 and not worker._handles
        h = await worker.submit(payload(budget=0))
        done = (await collect(h))[-1][1]
        assert done['token_ids'] == [] and done['usage']['completion_tokens'] == 0
        await worker.stop()
        await worker.stop()
        assert not worker._handles
        broken = GenerationWorker(lambda: (_ for _ in ()).throw(OSError('load failed')))
        with pytest.raises(OSError, match='load failed'):
            await broken.start()
        await broken.stop()
        assert not broken.thread.is_alive()
    asyncio.run(check())


def test_text_deltas_match_final_gpt2_decode():
    from server.worker import TextDeltas
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('gpt2', cache_dir='model_cache', local_files_only=True)
    sequences = [tokenizer.encode(text, add_special_tokens=False) for text in
                 ['Café 日本語 🦄\n  spaces', 'literal �', '�x�', 'abc']]
    sequences += [[172, 253], [172, 253, 222], [172, 0, 50256], [50256], [0, 172]]
    for ids in sequences:
        deltas = TextDeltas(tokenizer)
        text = ''.join(deltas.push(i, final=n == len(ids)-1) for n,i in enumerate(ids))
        assert text == tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    class Bad:
        def decode(self, ids, **kwargs):
            return 'a' if len(ids) == 1 else 'b'
    d = TextDeltas(Bad())
    assert d.push(1) == 'a'
    with pytest.raises(ValueError, match='prefix'):
        d.push(2)
