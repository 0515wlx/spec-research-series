"""Compare sampling-matched cache, target-greedy cache, and real standard-SD loop."""
from __future__ import annotations
import json, os
D=[('qwen3.5-0.8b',0.8),('qwen3.5-2b',2.0),('qwen3.5-4b',4.0),('qwen3.5-9b',9.0),('qwen3.5-27b',27.0)]
for dom in ['wikitext_gen','fineweb_gen']:
    print('\n==',dom,'==')
    print('draft       sample q  greedy q  loop q  sample/greedy  loop/sample')
    for d,_ in D:
        ps=f'out/accept/sample_matched/{dom}_{d}.json'
        pg=f'out/accept/{dom}_{d}.json'
        pl=f'out/std_sd/gen_{dom}_{d}_g4.json'
        if not os.path.exists(ps): continue
        s=json.load(open(ps)); g=json.load(open(pg)) if os.path.exists(pg) else {}
        l=json.load(open(pl)) if os.path.exists(pl) else {}
        sq=s.get('q'); gq=g.get('q'); lq=l.get('q_greedy_loop')
        print(f'{d:16s} {sq:.3f}    {gq:.3f}    {lq if lq is None else round(lq,3)}    {sq/gq:.2f}           {lq/sq if lq else float("nan"):.2f}')
