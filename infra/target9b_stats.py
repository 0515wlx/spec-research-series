"""Summarise the total-parameter super-capacity control (9B target, drafts up to 27B)."""
from __future__ import annotations
import json, os
import numpy as np

DRAFTS=[('qwen3.5-0.8b',0.8),('qwen3.5-2b',2.0),('qwen3.5-4b',4.0),('qwen3.5-9b',9.0),('qwen3.5-27b',27.0)]
DOMS=[('math500','MATH-500'),('wikitext','WikiText raw')]

def main():
    out={}
    for dom,label in DOMS:
        rows=[]
        for d,D in DRAFTS:
            p=f'out/accept/target9b/{dom}_{d}.json'
            if not os.path.exists(p): continue
            j=json.load(open(p)); rows.append({'draft':d,'D':D,'q':j['q'],'E_TV':j['E_TV']})
        if len(rows)>=3:
            D=np.array([r['D'] for r in rows]); q=np.array([r['q'] for r in rows])
            alpha=float(-np.polyfit(np.log(D),np.log(q),1)[0])
            out[label]={'rows':rows,'alpha':alpha,'ratio':rows[0]['q']/rows[-1]['q']}
        print('\n',label)
        for r in rows: print(f"  {r['draft']:16s} q={r['q']:.4f} TV={r['E_TV']:.4f}")
        if rows: print('  alpha',round(out[label]['alpha'],3),'q0.8/q27',round(out[label]['ratio'],3))
    with open('out/accept/target9b/summary.json','w') as f: json.dump(out,f,indent=1)
if __name__=='__main__': main()
