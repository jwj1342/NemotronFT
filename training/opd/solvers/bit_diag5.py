"""Part 5: SELECTIVE override - only correct model bits that are (a) uniquely
determined, or (b) dominated by prior with margin>=THRESH. Measure net gain.
This tests whether a verifier-style selective fix beats the model."""
import json, glob, csv, sys, collections, random
csv.field_size_limit(10**7)
sys.path.insert(0,'/scratch/jwj/NemotronFT/training/opd/solvers')
from bit_solver import _parse_prompt, _GATES_2, _GATES_3
from bit_solver_v2 import FAMILY_PRIOR
from itertools import combinations, product

def load_all():
    out={}
    for f in sorted(glob.glob('/scratch/jwj/NemotronFT/output/eval_full_huikang-baseline-converted_shard*of4.jsonl')):
        for line in open(f):
            line=line.strip()
            if not line: continue
            d=json.loads(line)
            if d['category']=='bit_manipulation': out[d['id']]=d
    return out
recs=load_all()
prompts={}
with open('/scratch/jwj/NemotronFT/data/train.csv',newline='') as fh:
    r=csv.DictReader(fh)
    for row in r:
        if row['id'] in recs: prompts[row['id']]=row['prompt']
def enum(out_col,inp_bits,n,tgt_bits):
    cands=[]
    if all(x==0 for x in out_col): cands.append(("CONST0",0))
    if all(x==1 for x in out_col): cands.append(("CONST1",1))
    for j in range(8):
        col=[inp_bits[e][j] for e in range(n)]
        if col==out_col: cands.append(("IDENTITY",tgt_bits[j]))
        if [1-x for x in col]==out_col: cands.append(("NOT",1-tgt_bits[j]))
    for name,fn in _GATES_2:
        for j,k in combinations(range(8),2):
            if all(fn(inp_bits[e][j],inp_bits[e][k])==out_col[e] for e in range(n)):
                cands.append((name,fn(tgt_bits[j],tgt_bits[k])))
    for name,fn in _GATES_3:
        for j,k,l in product(range(8),repeat=3):
            if all(fn(inp_bits[e][j],inp_bits[e][k],inp_bits[e][l])==out_col[e] for e in range(n)):
                cands.append((name,fn(tgt_bits[j],tgt_bits[k],tgt_bits[l])))
    return cands
def pw(fam): return FAMILY_PRIOR.get(fam,0.0005)

# only override a model bit if column is UNIQUELY determined (all consistent gates agree)
def selective(inp_ex,out_ex,target,model_bits,only_unique=True,margin=None):
    n=len(inp_ex)
    inp_bits=[[(v>>(7-j))&1 for j in range(8)] for v in inp_ex]
    tgt_bits=[(target>>(7-j))&1 for j in range(8)]
    out=list(model_bits)
    for o in range(8):
        out_col=[(out_ex[e]>>(7-o))&1 for e in range(n)]
        cands=enum(out_col,inp_bits,n,tgt_bits)
        if not cands: continue
        tvals=set(c[1] for c in cands)
        if len(tvals)==1:
            out[o]=str(list(tvals)[0])
        elif (not only_unique) and margin is not None:
            votes={0:0.0,1:0.0}
            for fam,tv in cands: votes[tv]+=pw(fam)
            tot=votes[0]+votes[1]
            if tot>0:
                top=max(votes,key=votes.get)
                if votes[top]/tot>=margin:
                    out[o]=str(top)
    return "".join(out)

for mode in [("unique-only",dict(only_unique=True)),
             ("unique+margin0.95",dict(only_unique=False,margin=0.95)),
             ("unique+margin0.85",dict(only_unique=False,margin=0.85))]:
    name,kw=mode
    fix=0;brk=0
    for pid,rec in recs.items():
        p=prompts.get(pid)
        if not p: continue
        inp_ex,out_ex,target=_parse_prompt(p)
        if inp_ex is None: continue
        pred=rec['predicted']
        if not (isinstance(pred,str) and len(pred)==8 and set(pred)<=set('01')):
            # model bits unknown -> skip override sim (can't reconstruct)
            continue
        new=selective(inp_ex,out_ex,target,list(pred),**kw)
        was=rec['correct']
        now=(new==rec['answer'])
        # Note: 'was' uses grader; 'now' approximates via predicted-string reconstruction
        if (not was) and now: fix+=1
        elif was and (not now): brk+=1
    print(f"[{name}] FIX={fix} BREAK={brk} NET={fix-brk:+d}")
print("\n(Caveat: model bits reconstructed from 'predicted' string; on 178/1402 correct problems predicted!=answer, so BREAK is overstated for correct cases & FIX uses true grader label.)")
