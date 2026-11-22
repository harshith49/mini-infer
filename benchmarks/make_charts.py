"""Headless charts for one complete, comparable six-stage experiment."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import textwrap

os.environ.setdefault('MPLCONFIGDIR',str(Path('model_cache/matplotlib').resolve()))
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from benchmarks.stage_suite import STAGES

SHARED = ('device','hardware','python_version','torch_version','threads','prompt_lengths','output_budgets','request_count',
          'max_batch_size','seed','warmups','repetitions','workload','latency_scope')
METRICS = ('tokens_per_second','completion_p50_seconds','completion_p95_seconds')


def read_rows(path: Path) -> list[dict[str, object]]:
    with path.open(newline='') as handle:
        reader = csv.DictReader(handle)
        required = {'stage','dtype','peak_memory_bytes','peak_memory_kind',
                    'cache_backend','page_size','num_pages',*SHARED,*METRICS}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError('Missing six-stage CSV columns')
        rows = list(reader)
    if len(rows) != len(STAGES) or sorted(row['stage'] for row in rows) != sorted(STAGES):
        raise ValueError('Expected exactly one row per engine stage')
    rows.sort(key=lambda row: STAGES.index(row['stage']))
    try:
        for row in rows:
            if any(row[key] != rows[0][key] for key in SHARED):
                raise ValueError('Rows describe different experiments')
            lengths,budgets = json.loads(row['prompt_lengths']),json.loads(row['output_budgets'])
            if (not isinstance(lengths,list) or not isinstance(budgets,list) or not lengths
                    or len(lengths)!=len(budgets) or any(type(n) is not int or n<=0 for n in lengths+budgets)
                    or int(row['request_count'])!=len(lengths)
                    or any(int(row[k])<=0 for k in ('threads','max_batch_size','warmups','repetitions'))
                    or int(row['repetitions'])<3 or not 0<=int(row['seed'])<2**63
                    or any(not row[k].strip() for k in ('hardware','python_version','torch_version','workload','latency_scope'))):
                raise ValueError('Invalid workload metadata')
            paged = row['stage'] in ('paged_kv','int8')
            backend = 'paged' if paged else 'none' if row['stage']=='naive' else 'contiguous'
            if row['cache_backend'] != backend:
                raise ValueError('Unexpected stage cache backend')
            if paged:
                if any(int(row[k])<=0 or row[k]!=rows[-1][k] for k in ('page_size','num_pages')):
                    raise ValueError('Invalid or incompatible page configuration')
            elif any(row[k]!='' for k in ('page_size','num_pages')):
                raise ValueError('Unexpected page configuration')
            expected_dtype = 'int8_weights_fp32_compute' if row['stage']=='int8' else 'fp32'
            if row['dtype']!=expected_dtype:
                raise ValueError('Unexpected stage dtype')
            for key in METRICS:
                row[key] = float(row[key])
                if not math.isfinite(row[key]) or row[key]<0:
                    raise ValueError('Invalid chart metric')
            if row['completion_p95_seconds']<row['completion_p50_seconds']:
                raise ValueError('p95 cannot be below p50')
            kind = row['peak_memory_kind']
            value = row['peak_memory_bytes']
            if kind == 'unmeasured' and value == '':
                row['peak_memory_bytes'] = None
            elif (kind == 'cuda_allocated_including_model' and row['device'].startswith('cuda')
                    and value != ''):
                row['peak_memory_bytes'] = int(value)
                if row['peak_memory_bytes']<0:
                    raise ValueError('Negative memory')
            else:
                raise ValueError('Memory value and scope disagree')
            if row['device']!='cpu' and not row['device'].startswith('cuda'):
                raise ValueError('Unexpected device')
    except (TypeError,KeyError,ValueError) as error:
        raise ValueError(f'Invalid stage CSV: {error}') from error
    return rows


def make_charts(csv_path: Path, output_dir: Path) -> list[Path]:
    rows = read_rows(csv_path)  # Validate everything before creating output files.
    output_dir.mkdir(parents=True,exist_ok=True)
    labels = ['Naive','KV cache','Static\nbatching','Continuous\nbatching','Paged KV','Paged +\nint8 weights']
    first = rows[0]
    caption = textwrap.fill(f"{first['hardware']} | {first['device']} | FP32; final stage int8 weights/FP32 compute | "
        f"threads={first['threads']}; batch limit={first['max_batch_size']}; warmups={first['warmups']}; repetitions={first['repetitions']}\n"
        f"prompt lengths={first['prompt_lengths']}; output budgets={first['output_budgets']}. "
        'Fixed-budget synthetic greedy workload. Request latency includes waiting.',110)
    files = []
    for name,title,unit in [('throughput','Useful generation throughput','Tokens / second'),
                            ('latency','Whole-request completion latency','Seconds'),
                            ('memory','Peak allocated device memory including model','MiB')]:
        fig,ax = plt.subplots(figsize=(11,6))
        try:
            if name=='throughput':
                ax.bar(labels,[r['tokens_per_second'] for r in rows],color='#3979b7')
            elif name=='latency':
                x=list(range(len(rows)))
                ax.bar([i-.18 for i in x],[r['completion_p50_seconds'] for r in rows],width=.36,label='p50')
                ax.bar([i+.18 for i in x],[r['completion_p95_seconds'] for r in rows],width=.36,label='p95')
                ax.set_xticks(x,labels);ax.legend()
            elif all(r['peak_memory_bytes'] is None for r in rows):
                ax.text(.5,.5,f"Unmeasured on {first['device']}\nNo peak-memory measurement",ha='center',va='center',transform=ax.transAxes,fontsize=16)
                ax.set_xticks(range(len(labels)),labels)
            else:
                for i,row in enumerate(rows):
                    if row['peak_memory_bytes'] is None:
                        ax.annotate('Unmeasured',(i,.05),xycoords=('data','axes fraction'),ha='center')
                    else:
                        ax.bar(i,row['peak_memory_bytes']/2**20,color='#3979b7')
                ax.set_xticks(range(len(labels)),labels)
            ax.set_title(title);ax.set_ylabel(unit);ax.set_ylim(bottom=0)
            ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
            fig.text(.04,.02,caption,fontsize=8,va='bottom')
            fig.tight_layout(rect=(0,.19,1,1))
            target = output_dir/f'stages_{name}.png'
            fig.savefig(target,dpi=150)
            files.append(target)
        finally:
            plt.close(fig)
    return files


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=Path('results/stages.csv'))
    parser.add_argument('--output-dir',type=Path,default=Path('results'))
    args=parser.parse_args()
    try:
        for path in make_charts(args.input,args.output_dir): print(path)
    except ValueError as error:
        parser.error(str(error))


if __name__=='__main__':
    main()
