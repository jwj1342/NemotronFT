"""Part 3: generator reverse-engineering + 4-input rescue of no-gate columns."""
import json, glob, csv, sys, collections, statistics
csv.field_size_limit(10**7)
sys.path.insert(0,'/scratch/jwj/NemotronFT/training/opd/solvers')
from bit_solver import _parse_prompt, _GATES_2, _GATES_3, _GATES_4
from bit_solver_v2 import FAMILY_PRIOR, PRIOR_4
from itertools import combinations, product

def load_all():
    out={}
    for f in sorted(glob.glob('/scratch/jwj/NemotronFT/output/eval_full_huikang-baseline-converted_shard*of4.jsonl')):
        for line in open(f):
            line=line.strip()
            if not line: continue
            d=json.loads(line)
            if d['category']=='bit_manipulation':
                out[d['id']]=d
    return out
recs=load_all()
fails={k:v for k,v in recs.items() if not v['correct']}
goods={k:v for k,v in recs.items() if v['correct']}
allids=set(recs)
prompts={}
with open('/scratch/jwj/NemotronFT/data/train.csv',newline='') as fh:
    r=csv.DictReader(fh)
    for row in r:
        if row['id'] in allids: prompts[row['id']]=row['prompt']

# ---- number of examples per problem; input value distribution ----
nex=collections.Counter()
input_vals=collections.Counter()
target_eq_example=0
nprob=0
for pid in allids:
    p=prompts.get(pid)
    if not p: continue
    inp_ex,out_ex,target=_parse_prompt(p)
    if inp_ex is None: continue
    nprob+=1
    nex[len(inp_ex)]+=1
    for v in inp_ex: input_vals[v]+=1
    if target in inp_ex: target_eq_example+=1
print(f"problems parsed: {nprob}")
print(f"examples-per-problem dist: {dict(sorted(nex.items()))}")
print(f"target == one of examples: {target_eq_example}/{nprob}")
vals=sorted(input_vals)
print(f"distinct input bytes seen: {len(vals)} (of 256); min={min(vals)} max={max(vals)}")
# bit-position balance of inputs (how often each MSB-position bit is 1)
posones=[0]*8; tot=sum(input_vals.values())
for v,c in input_vals.items():
    for j in range(8):
        if (v>>(7-j))&1: posones[j]+=c
print("P(input bit=1) per MSB position:", [round(posones[j]/tot,3) for j in range(8)])

# ---- TRUE gate family distribution: use UNIQUELY-determined columns across ALL (good+fail) ----
def enum_unique(out_col,inp_bits,n,tgt_bits,do4=False):
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
    if do4:
        for name,fn in _GATES_4:
            for j,k,l,m in product(range(8),repeat=4):
                if all(fn(inp_bits[e][j],inp_bits[e][k],inp_bits[e][l],inp_bits[e][m])==out_col[e] for e in range(n)):
                    cands.append(("G4_"+name,fn(tgt_bits[j],tgt_bits[k],tgt_bits[l],tgt_bits[m])))
    return cands

# Sample 250 GOOD problems for true-generator family freq (uniquely determined cols only)
import random; random.seed(1)
gids=list(goods); random.shuffle(gids)
fam_unique=collections.Counter()
fam_anycons=collections.Counter()   # all consistent families (over-counts ambiguous)
ncols_unique=0; ncols_total=0
nogate_cols_3in=0
for pid in gids[:250]:
    p=prompts.get(pid)
    if not p: continue
    inp_ex,out_ex,target=_parse_prompt(p)
    if inp_ex is None: continue
    n=len(inp_ex)
    inp_bits=[[(v>>(7-j))&1 for j in range(8)] for v in inp_ex]
    tgt_bits=[(target>>(7-j))&1 for j in range(8)]
    for o in range(8):
        out_col=[(out_ex[e]>>(7-o))&1 for e in range(n)]
        cands=enum_unique(out_col,inp_bits,n,tgt_bits,do4=False)
        ncols_total+=1
        fams=set(c[0] for c in cands)
        for fm in fams: fam_anycons[fm]+=1
        if not cands: nogate_cols_3in+=1; continue
        tvals=set(c[1] for c in cands)
        if len(tvals)==1 and len(fams)==1:
            ncols_unique+=1; fam_unique[list(fams)[0]]+=1
print(f"\n[GOOD sample n={min(250,len(gids))}] columns total={ncols_total} uniquely-single-family={ncols_unique} nogate(3in)={nogate_cols_3in}")
print("single-family uniquely-determined gate freq (cleanest generator signal):")
for fm,c in fam_unique.most_common(): print(f"  {fm}: {c} ({c/max(1,ncols_unique)*100:.1f}%)")

# ---- 4-input rescue of the no-gate(3in) columns among FAILURES ----
print("\n=== 4-INPUT RESCUE of no-gate columns (failures) ===")
rescued=0; still_no=0
for pid in fails:
    p=prompts.get(pid)
    if not p: continue
    inp_ex,out_ex,target=_parse_prompt(p)
    if inp_ex is None: continue
    n=len(inp_ex)
    inp_bits=[[(v>>(7-j))&1 for j in range(8)] for v in inp_ex]
    tgt_bits=[(target>>(7-j))&1 for j in range(8)]
    for o in range(8):
        out_col=[(out_ex[e]>>(7-o))&1 for e in range(n)]
        c3=enum_unique(out_col,inp_bits,n,tgt_bits,do4=False)
        if c3: continue
        c4=enum_unique(out_col,inp_bits,n,tgt_bits,do4=True)
        if c4:
            rescued+=1
            print(f"  {pid} o[{o}]: NO 3-in gate, 4-in matches: {sorted(set(g[0] for g in c4))}")
        else:
            still_no+=1
            print(f"  {pid} o[{o}]: NO gate even at 4-input -> >4-input function")
print(f"no-gate(3in) cols rescued by 4-input: {rescued}; still unresolved: {still_no}")
