"""Part 4: net-gain simulation of prior-vote post-processor across ALL bit problems,
and a cleaner generator-prior estimate via incremental example saturation."""
import json, glob, csv, sys, collections, statistics, random
csv.field_size_limit(10**7)
sys.path.insert(0,'/scratch/jwj/NemotronFT/training/opd/solvers')
from bit_solver import _parse_prompt, _GATES_2, _GATES_3
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

def prior_solve(inp_ex,out_ex,target):
    n=len(inp_ex)
    inp_bits=[[(v>>(7-j))&1 for j in range(8)] for v in inp_ex]
    tgt_bits=[(target>>(7-j))&1 for j in range(8)]
    bits=[]
    for o in range(8):
        out_col=[(out_ex[e]>>(7-o))&1 for e in range(n)]
        cands=enum(out_col,inp_bits,n,tgt_bits)
        if not cands: bits.append(None); continue
        votes={0:0.0,1:0.0}
        for fam,tv in cands: votes[tv]+=pw(fam)
        bits.append(0 if votes[0]>=votes[1] else 1)
    return bits

# Net-gain simulation: replace model answer with prior-solver answer everywhere.
# fix = was wrong, prior gets it right (exact 8/8). break = was right, prior gets it wrong.
fix=0; brk=0; both_right=0; both_wrong=0; prior_has_none=0
n_eval=0
for pid,rec in recs.items():
    p=prompts.get(pid)
    if not p: continue
    inp_ex,out_ex,target=_parse_prompt(p)
    if inp_ex is None: continue
    n_eval+=1
    true_ans=rec['answer']
    model_right=rec['correct']
    bits=prior_solve(inp_ex,out_ex,target)
    if None in bits:
        prior_has_none+=1
        prior_ans=None
    else:
        prior_ans="".join(str(b) for b in bits)
    prior_right=(prior_ans==true_ans)
    if model_right and prior_right: both_right+=1
    elif (not model_right) and prior_right: fix+=1
    elif model_right and (not prior_right): brk+=1
    else: both_wrong+=1
print(f"=== PRIOR-SOLVER vs MODEL net gain (n={n_eval}) ===")
print(f"both right: {both_right}")
print(f"FIX (model wrong -> prior right): {fix}")
print(f"BREAK (model right -> prior wrong): {brk}")
print(f"both wrong: {both_wrong}")
print(f"prior had unresolved(None) bit: {prior_has_none}")
print(f"NET delta if we replaced model with prior-solver: {fix-brk:+d}  ({(fix-brk)/n_eval*100:+.2f}pp of bit category)")
print(f"model acc={sum(1 for r in recs.values() if r['correct'])/len(recs)*100:.1f}%  prior-solver acc={(both_right+fix)/n_eval*100:.1f}%")

# Generator prior via SATURATION: among GOOD problems, look at columns that become
# uniquely determined; take ALL consistent families on those and report which family
# the prior would assign. Better: count consistent-family multiplicity to gauge true mix.
# Simpler robust estimate: for uniquely-determined columns, if multiple families all map
# to same single function, list them; report the SIMPLEST (lowest-arity) as canonical.
ARITY={"CONST0":0,"CONST1":0,"IDENTITY":1,"NOT":1}
for nm,_ in _GATES_2: ARITY[nm]=2
for nm,_ in _GATES_3: ARITY[nm]=3
random.seed(2)
gids=[k for k,v in recs.items() if v['correct']]; random.shuffle(gids)
canon=collections.Counter(); ncol=0; uniq=0
for pid in gids[:300]:
    p=prompts.get(pid)
    if not p: continue
    inp_ex,out_ex,target=_parse_prompt(p)
    if inp_ex is None: continue
    n=len(inp_ex)
    inp_bits=[[(v>>(7-j))&1 for j in range(8)] for v in inp_ex]
    tgt_bits=[(target>>(7-j))&1 for j in range(8)]
    for o in range(8):
        out_col=[(out_ex[e]>>(7-o))&1 for e in range(n)]
        cands=enum(out_col,inp_bits,n,tgt_bits)
        ncol+=1
        if not cands: continue
        if len(set(c[1] for c in cands))==1:
            uniq+=1
            # canonical = lowest arity family among consistent
            fam=min(set(c[0] for c in cands), key=lambda f: (ARITY.get(f,3), -pw(f)))
            canon[fam]+=1
print(f"\n=== generator prior (lowest-arity canonical, GOOD sample) cols={ncol} uniquely-valued={uniq} ===")
for fm,c in canon.most_common(): print(f"  {fm}: {c} ({c/max(1,uniq)*100:.1f}%)")
