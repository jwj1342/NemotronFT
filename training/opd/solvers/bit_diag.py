"""
Diagnostic: enumerate all gates consistent with examples for every failed
bit_manipulation problem. Splits salvageable (capacity-limited) vs unsolvable
(information-limited) by checking whether the frequency prior would pick the
correct answer for ambiguous bits.

Login-node safe: pure stdlib, 200 problems x 8 bits x (10+280+4608) gates.
"""
import json, glob, csv, sys, collections
csv.field_size_limit(10**7)
sys.path.insert(0, '/scratch/jwj/NemotronFT/training/opd/solvers')

from bit_solver import _parse_prompt, _GATES_2, _GATES_3, _GATES_4
from bit_solver_v2 import FAMILY_PRIOR, PRIOR_4
from itertools import combinations, product

# ---- load failures ----
fails = {}
for f in sorted(glob.glob('/scratch/jwj/NemotronFT/output/eval_full_huikang-baseline-converted_shard*of4.jsonl')):
    for line in open(f):
        line=line.strip()
        if not line: continue
        d=json.loads(line)
        if d['category']=='bit_manipulation' and not d['correct']:
            fails[d['id']]=d

prompts={}
with open('/scratch/jwj/NemotronFT/data/train.csv', newline='') as fh:
    r=csv.DictReader(fh)
    for row in r:
        if row['id'] in fails:
            prompts[row['id']]=row['prompt']

print(f"failed bit problems: {len(fails)}, prompts found: {len(prompts)}")

# ---- enumeration of all consistent gates for one output column ----
# returns list of (family, label, target_bit_value)
def enum_gates(out_col, inp_bits, n, tgt_bits, do4=False):
    cands = []
    if all(x == 0 for x in out_col):
        cands.append(("CONST0", "CONST_0", 0))
    if all(x == 1 for x in out_col):
        cands.append(("CONST1", "CONST_1", 1))
    for j in range(8):
        col = [inp_bits[e][j] for e in range(n)]
        if col == out_col:
            cands.append(("IDENTITY", f"ID(in[{j}])", tgt_bits[j]))
        if [1 - x for x in col] == out_col:
            cands.append(("NOT", f"NOT(in[{j}])", 1 - tgt_bits[j]))
    for name, fn in _GATES_2:
        for j, k in combinations(range(8), 2):
            if all(fn(inp_bits[e][j], inp_bits[e][k]) == out_col[e] for e in range(n)):
                cands.append((name, f"{name}({j},{k})", fn(tgt_bits[j], tgt_bits[k])))
    for name, fn in _GATES_3:
        for j, k, l in product(range(8), repeat=3):
            if all(fn(inp_bits[e][j], inp_bits[e][k], inp_bits[e][l]) == out_col[e] for e in range(n)):
                cands.append((name, f"{name}({j},{k},{l})", fn(tgt_bits[j], tgt_bits[k], tgt_bits[l])))
    if do4:
        for name, fn in _GATES_4:
            for j, k, l, m in product(range(8), repeat=4):
                if all(fn(inp_bits[e][j], inp_bits[e][k], inp_bits[e][l], inp_bits[e][m]) == out_col[e] for e in range(n)):
                    cands.append(("G4_"+name, f"{name}({j},{k},{l},{m})", fn(tgt_bits[j], tgt_bits[k], tgt_bits[l], tgt_bits[m])))
    return cands

def prior_w(family):
    if family.startswith("G4_"): return PRIOR_4
    return FAMILY_PRIOR.get(family, 0.0005)

# ---- main loop ----
fully_unique = 0           # all 8 bits uniquely determined
has_ambiguous = 0          # >=1 bit ambiguous (multiple gates disagree on target)
no_gate_3input = 0         # >=1 bit has NO consistent gate within 3-input family
per_bit_unique = 0
per_bit_total = 0
per_bit_ambig = 0
per_bit_nogate = 0

# capacity vs info split
# For each failed problem: compute prior-voted answer; compare to true answer.
prior_recover_problems = 0   # prior vote would get the FULL answer right
amb_bit_prior_correct = 0    # ambiguous bits where prior vote picks correct
amb_bit_prior_wrong = 0
amb_bit_count = 0

# error analysis vs model predicted
hamming_pred = collections.Counter()
# for the bits where model was WRONG, was correct gate prior-high?
wrong_bit_prior_rank = []   # rank of correct value's winning family among consistent gates
nogate_examples = []
family_freq = collections.Counter()   # unique-determined columns: which family
detail = []

for pid, rec in fails.items():
    prompt = prompts.get(pid)
    if prompt is None: continue
    inp_ex, out_ex, target = _parse_prompt(prompt)
    if inp_ex is None: continue
    n = len(inp_ex)
    inp_bits = [[(v >> (7 - j)) & 1 for j in range(8)] for v in inp_ex]
    tgt_bits = [(target >> (7 - j)) & 1 for j in range(8)]
    true_ans = rec['answer']
    pred = rec['predicted'] if isinstance(rec['predicted'], str) and len(rec['predicted'])==8 else None

    prob_all_unique = True
    prob_any_ambig = False
    prob_any_nogate = False
    prior_ans_bits = []
    prob_amb_prior_ok = True

    for o in range(8):
        out_col = [(out_ex[e] >> (7 - o)) & 1 for e in range(n)]
        cands = enum_gates(out_col, inp_bits, n, tgt_bits, do4=False)
        per_bit_total += 1
        true_bit = int(true_ans[o])
        if not cands:
            prob_any_nogate = True
            prob_all_unique = False
            per_bit_nogate += 1
            prior_ans_bits.append('?')
            continue
        tvals = set(c[2] for c in cands)
        if len(tvals) == 1:
            per_bit_unique += 1
            # record family for the (any) gate -- use highest prior family as canonical
            fam = max(cands, key=lambda c: prior_w(c[0]))[0]
            family_freq[fam]+=1
            prior_ans_bits.append(str(cands[0][2]))
        else:
            prob_any_ambig = True
            prob_all_unique = False
            per_bit_ambig += 1
            amb_bit_count += 1
            # prior weighted vote
            votes = {0:0.0,1:0.0}
            for fam,label,tv in cands:
                votes[tv]+=prior_w(fam)
            voted = 0 if votes[0]>=votes[1] else 1
            prior_ans_bits.append(str(voted))
            if voted == true_bit:
                amb_bit_prior_correct += 1
            else:
                amb_bit_prior_wrong += 1
                prob_amb_prior_ok = False
    if prob_all_unique: fully_unique += 1
    if prob_any_ambig: has_ambiguous += 1
    if prob_any_nogate: no_gate_3input += 1
    # prior full recovery
    prior_ans = "".join(prior_ans_bits)
    if '?' not in prior_ans and prior_ans == true_ans:
        prior_recover_problems += 1
    # hamming model pred vs answer
    if pred:
        h = sum(1 for a,b in zip(pred,true_ans) if a!=b)
        hamming_pred[h]+=1
    detail.append((pid, prob_all_unique, prob_any_ambig, prob_any_nogate, prior_ans, true_ans, pred))

print("\n=== PER-PROBLEM (200 failures) ===")
print(f"fully unique (all 8 bits determined by 3-input family): {fully_unique}")
print(f"has >=1 ambiguous bit: {has_ambiguous}")
print(f"has >=1 bit with NO consistent gate (3-input family): {no_gate_3input}")

print("\n=== PER-BIT (1600 bit-columns) ===")
print(f"unique: {per_bit_unique}  ambiguous: {per_bit_ambig}  no-gate(3in): {per_bit_nogate}  total: {per_bit_total}")

print("\n=== CAPACITY vs INFO (prior-vote recovery) ===")
print(f"problems where prior-vote (no-? bits) reproduces FULL true answer: {prior_recover_problems}")
print(f"ambiguous bits total: {amb_bit_count}")
print(f"  prior-vote picks CORRECT bit: {amb_bit_prior_correct} ({amb_bit_prior_correct/max(1,amb_bit_count)*100:.1f}%)")
print(f"  prior-vote picks WRONG bit:   {amb_bit_prior_wrong} ({amb_bit_prior_wrong/max(1,amb_bit_count)*100:.1f}%)")

print("\n=== MODEL HAMMING (predicted vs answer, only 8-char preds) ===")
for h in sorted(hamming_pred):
    print(f"  {h} bits off: {hamming_pred[h]}")

print("\n=== FAMILY FREQ (unique-determined columns in failures) ===")
for fam,c in family_freq.most_common():
    print(f"  {fam}: {c}")

# save detail
import pickle
with open('/scratch/jwj/NemotronFT/training/opd/solvers/bit_diag_detail.pkl','wb') as fh:
    pickle.dump(detail, fh)
print("\nsaved detail pkl")
