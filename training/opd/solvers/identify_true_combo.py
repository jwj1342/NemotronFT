"""
Identify the GENERATOR's true (digit_order, op, fmt) combo + cipher for puzzles
where we know the answer (model-CORRECT puzzles, and also failed puzzles whose
answer we have from eval).

Method: include the QUESTION (input->answer) as an additional "example", then
find combos that are consistent across ALL examples INCLUDING the target. The
extra constraint from the answer drastically shrinks the consistent set and
usually pins the combo. Tally the combo frequency => reverse-engineered generator
distribution. Also measures: for the model-correct set, how concentrated is the
true combo (=> strength of the frequency prior the model exploits).

Pure stdlib, fast (answer constraint prunes hard).
"""
from __future__ import annotations
import sys, csv, json, re, random, collections, time
csv.field_size_limit(10**7)
from pathlib import Path
_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS))
from symbol_digit_solver import COMBOS, apply_combo
CIPHER_COMBOS = [c for c in COMBOS if "op_" not in c[2]]

def parse(p):
    raw=[];q=None
    for line in p.strip().splitlines():
        line=line.strip()
        if not line:continue
        if re.search(r'(?:determine|result for)',line,re.I):
            m=re.search(r':\s*(\S+)\s*$',line)
            if m:q=m.group(1)
            continue
        m=re.match(r'^(\S+)\s*=\s*(\S+)$',line)
        if m: raw.append((m.group(1),m.group(2)))
    if q is None or len(q)<5: return None,None,None
    op=q[2]
    valid=[(i,o) for i,o in raw if len(i)==5 and i[2]==op]
    return valid,q,op

def build_pm(inp,cout,nout,A,B,C,D):
    if len(cout)!=len(nout): return None
    m={};r={}
    for sym,dg in [(inp[0],A),(inp[1],B),(inp[3],C),(inp[4],D)]:
        if sym in m and m[sym]!=dg: return None
        if dg in r and r[dg]!=sym: return None
        m[sym]=dg;r[dg]=sym
    for cs,dc in zip(cout,nout):
        dg=int(dc)
        if cs in m and m[cs]!=dg: return None
        if dg in r and r[dg]!=cs: return None
        m[cs]=dg;r[dg]=cs
    return m,r

def merge(bm,br,nm,nr):
    m=dict(bm);r=dict(br)
    for s,d in nm.items():
        if s in m and m[s]!=d: return None
        if d in r and r[d]!=s: return None
        m[s]=d;r[d]=s
    return m,r

def consistent(combo, examples, cap=400):
    """fully example-consistent mappings (list), cap small."""
    ex1i,ex1o=examples[0]; l1=len(ex1o); op=ex1i[2]
    seed=[]
    for L in range(100):
        A,B=L//10,L%10
        for R in range(100):
            C,D=R//10,R%10
            no=apply_combo(combo,A,B,C,D,op)
            if no is None or not no.isdigit() or len(no)!=l1: continue
            pm=build_pm(ex1i,ex1o,no,A,B,C,D)
            if pm: seed.append(pm)
        if len(seed)>=cap: break
    cur=seed
    for (inp,out) in examples[1:]:
        op2=inp[2]; lo=len(out); nxt=[]
        for (m,r) in cur:
            rgs=[([m[s]] if s in m else range(10)) for s in (inp[0],inp[1],inp[3],inp[4])]
            for A in rgs[0]:
              for B in rgs[1]:
                for C in rgs[2]:
                  for D in rgs[3]:
                    no=apply_combo(combo,A,B,C,D,op2)
                    if no is None or not no.isdigit() or len(no)!=lo: continue
                    pm=build_pm(inp,out,no,A,B,C,D)
                    if pm is None: continue
                    mg=merge(m,r,*pm)
                    if mg: nxt.append(mg)
        seen=set();ded=[]
        for (mm,rr) in nxt:
            k=tuple(sorted(mm.items()))
            if k not in seen: seen.add(k);ded.append((mm,rr))
        cur=ded
        if not cur: return []
    return cur

def identify(examples, question, answer, budget=2.5):
    """Return set of combos consistent with examples+(question->answer).
    Put (question->answer) FIRST so it prunes the candidate explosion early."""
    aug = [(question, answer)] + examples
    combos_ok=[]
    t0=time.time()
    for combo in CIPHER_COMBOS:
        if time.time()-t0>budget:
            break
        if consistent(combo, aug):
            combos_ok.append(combo)
    return combos_ok

def load_set(correct=True, cat="cryptarithm_deduce", n=80, seed=0):
    items=[]
    for s in range(4):
        with open(f"/scratch/jwj/NemotronFT/output/eval_full_huikang-baseline-converted_shard{s}of4.jsonl") as f:
            for line in f:
                r=json.loads(line)
                if r["category"]==cat and (r["correct"]==correct):
                    items.append(r)
    random.seed(seed); random.shuffle(items)
    return items[:n]

def main():
    correct = sys.argv[1]=="correct" if len(sys.argv)>1 else True
    cat = sys.argv[2] if len(sys.argv)>2 else "cryptarithm_deduce"
    n = int(sys.argv[3]) if len(sys.argv)>3 else 80
    items=load_set(correct,cat,n)
    ids={r["id"] for r in items}
    P={}
    with open("/scratch/jwj/NemotronFT/data/train.csv") as f:
        for row in csv.DictReader(f):
            if row["id"] in ids: P[row["id"]]=row["prompt"]
    combo_freq=collections.Counter()
    rank_of_true=collections.Counter()   # frequency-rank of the (a) true combo
    n_pinned=0; n_ambig=0; n_none=0; n_proc=0
    t0=time.time()
    for r in items:
        if time.time()-t0>300: break
        ex,q,op=parse(P[r["id"]])
        if ex is None: continue
        n_proc+=1
        combos=identify(ex,q,r["answer"])
        if not combos:
            n_none+=1; continue
        if len(combos)==1:
            n_pinned+=1
        else:
            n_ambig+=1
        # tally the earliest (highest-freq) consistent combo as the "true" combo proxy
        ranks=[CIPHER_COMBOS.index(c) for c in combos]
        best_rank=min(ranks)
        rank_of_true[best_rank]+=1
        for c in combos:
            combo_freq[c]+=1
    print(json.dumps({
        "set":"correct" if correct else "failed",
        "cat":cat,"n_processed":n_proc,
        "pinned_unique_combo":n_pinned,"ambiguous(multi-combo)":n_ambig,
        "no_combo_consistent_with_answer":n_none,
        "top_combos_by_freq":[(str(c),f) for c,f in combo_freq.most_common(12)],
        "best_rank_histogram(0=top freq)":dict(sorted(rank_of_true.items())),
    }, indent=1))

if __name__=="__main__":
    main()
