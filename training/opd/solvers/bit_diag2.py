"""
Part 2: capacity-vs-info split focused on the MODEL's wrong bits, plus a
control on correctly-solved problems, plus 4-input rescue for no-gate columns.
"""
import json, glob, csv, sys, collections, pickle
csv.field_size_limit(10**7)
sys.path.insert(0, '/scratch/jwj/NemotronFT/training/opd/solvers')
from bit_solver import _parse_prompt, _GATES_2, _GATES_3, _GATES_4
from bit_solver_v2 import FAMILY_PRIOR, PRIOR_4
from itertools import combinations, product

def load(cat_correct):
    out={}
    for f in sorted(glob.glob('/scratch/jwj/NemotronFT/output/eval_full_huikang-baseline-converted_shard*of4.jsonl')):
        for line in open(f):
            line=line.strip()
            if not line: continue
            d=json.loads(line)
            if d['category']=='bit_manipulation' and d['correct']==cat_correct:
                out[d['id']]=d
    return out

fails=load(False)
goods=load(True)

allids=set(fails)|set(goods)
prompts={}
with open('/scratch/jwj/NemotronFT/data/train.csv', newline='') as fh:
    r=csv.DictReader(fh)
    for row in r:
        if row['id'] in allids:
            prompts[row['id']]=row['prompt']

def enum_gates(out_col, inp_bits, n, tgt_bits, do4=False):
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

def prior_w(fam):
    if fam.startswith("G4_"): return PRIOR_4
    return FAMILY_PRIOR.get(fam,0.0005)

def analyze(recs, label):
    inp_cache={}
    # per-problem: prior full-answer recovery rate
    full_recover=0
    total=0
    # focus on MODEL-WRONG bits
    wrong_bits_total=0
    wrong_bit_prior_picks_correct=0   # capacity-limited: prior would have fixed it
    wrong_bit_prior_picks_wrong=0     # info-limited: prior also wrong
    wrong_bit_correct_is_majority_consistent=0  # >=50% of consistent gates give true val
    wrong_bit_frac_consistent_for_true=[]
    for pid,rec in recs.items():
        prompt=prompts.get(pid)
        if prompt is None: continue
        inp_ex,out_ex,target=_parse_prompt(prompt)
        if inp_ex is None: continue
        total+=1
        n=len(inp_ex)
        inp_bits=[[(v>>(7-j))&1 for j in range(8)] for v in inp_ex]
        tgt_bits=[(target>>(7-j))&1 for j in range(8)]
        true_ans=rec['answer']
        pred=rec['predicted'] if isinstance(rec['predicted'],str) and len(rec['predicted'])==8 else None
        prior_bits=[]
        for o in range(8):
            out_col=[(out_ex[e]>>(7-o))&1 for e in range(n)]
            cands=enum_gates(out_col,inp_bits,n,tgt_bits,do4=False)
            true_bit=int(true_ans[o])
            if not cands:
                prior_bits.append('?'); continue
            votes={0:0.0,1:0.0}
            for fam,tv in cands: votes[tv]+=prior_w(fam)
            voted=0 if votes[0]>=votes[1] else 1
            prior_bits.append(str(voted))
            # model-wrong bit analysis (only for failures where pred available)
            if pred and pred[o]!=true_ans[o]:
                wrong_bits_total+=1
                if voted==true_bit: wrong_bit_prior_picks_correct+=1
                else: wrong_bit_prior_picks_wrong+=1
                ncons=len(cands)
                ntrue=sum(1 for f,tv in cands if tv==true_bit)
                wrong_bit_frac_consistent_for_true.append(ntrue/ncons)
                if ntrue/ncons>=0.5: wrong_bit_correct_is_majority_consistent+=1
        prior_ans="".join(prior_bits)
        if '?' not in prior_ans and prior_ans==true_ans: full_recover+=1
    print(f"\n=== {label} (n={total}) ===")
    print(f"prior-vote reproduces FULL true answer: {full_recover}/{total} ({full_recover/max(1,total)*100:.1f}%)")
    if wrong_bits_total:
        print(f"model-WRONG bits total: {wrong_bits_total}")
        print(f"  prior-vote would pick CORRECT (capacity-limited, salvageable): {wrong_bit_prior_picks_correct} ({wrong_bit_prior_picks_correct/wrong_bits_total*100:.1f}%)")
        print(f"  prior-vote also WRONG (info-limited): {wrong_bit_prior_picks_wrong} ({wrong_bit_prior_picks_wrong/wrong_bits_total*100:.1f}%)")
        print(f"  correct value is MAJORITY among consistent gates: {wrong_bit_correct_is_majority_consistent} ({wrong_bit_correct_is_majority_consistent/wrong_bits_total*100:.1f}%)")
        import statistics
        fr=wrong_bit_frac_consistent_for_true
        print(f"  mean frac of consistent gates giving TRUE value (on wrong bits): {statistics.mean(fr):.3f}  median: {statistics.median(fr):.3f}")
    return full_recover,total

fr_f,tot_f=analyze(fails,"FAILURES")
# control: sample goods to estimate baseline prior recovery (subset for speed)
import random
random.seed(0)
good_ids=list(goods)
random.shuffle(good_ids)
sub={k:goods[k] for k in good_ids[:200]}
fr_g,tot_g=analyze(sub,"CONTROL (correctly-solved sample)")
