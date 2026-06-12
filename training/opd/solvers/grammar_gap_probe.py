"""
Grammar-gap probe for deduce cases that have >=2 examples sharing the question op
(i.e. genuinely constrainable) yet the standard solver finds NO consistent solution.

Question: is "no_solution" a real grammar gap (operation/format the solver lacks)
or just under-constraint that the wide grammar also can't fill?

We compare:
  - STANDARD grammar (the solver's COMBOS op/fmt set)
  - WIDE grammar (adds div/mod/and/or/xor/max/min/avg/sumsq/sq-diff + zpad3)
on the SAME no_solution-by-standard cases, restricted to n_op>=2.

A case that the WIDE grammar solves but STANDARD does not  => real grammar gap.
A case neither solves                                       => not a grammar issue
                                                                (likely the rule
                                                                 is outside both,
                                                                 or 2 ex insufficient).
"""
from __future__ import annotations
import sys, csv, json, re, random
csv.field_size_limit(10**7)
from pathlib import Path
_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS))
from crypt_salvage_analysis import parse_prompt, load_fails, load_prompts, _consistent_mappings_for_combo
from symbol_digit_solver import COMBOS

STD_COMBOS = [c for c in COMBOS if "op_" not in c[2]]

# Wide grammar built inline (no apply_combo dependency)
ORDERS = {"AB_CD": lambda A,B,C,D:(A*10+B,C*10+D),
          "BA_DC": lambda A,B,C,D:(B*10+A,D*10+C),
          "AB_DC": lambda A,B,C,D:(A*10+B,D*10+C),
          "BA_CD": lambda A,B,C,D:(B*10+A,C*10+D)}
def dsum(n): return sum(int(c) for c in str(abs(n)))
WIDE_OPS = {
 "add":lambda L,R:L+R,"mul":lambda L,R:L*R,"sub_lr":lambda L,R:L-R,
 "sub_rl":lambda L,R:R-L,"abs_sub":lambda L,R:abs(L-R),
 "cat_lr":lambda L,R:int(f"{L:02d}{R:02d}"),"cat_rl":lambda L,R:int(f"{R:02d}{L:02d}"),
 "addm1":lambda L,R:L+R-1,"add1":lambda L,R:L+R+1,
 "mulsub1":lambda L,R:L*R-1,"muladd1":lambda L,R:L*R+1,
 "dsum_add":lambda L,R:dsum(L)+dsum(R),"dsum_mul":lambda L,R:dsum(L)*dsum(R),
 # extras:
 "div_lr":lambda L,R:(L//R if R else None),"mod_lr":lambda L,R:(L%R if R else None),
 "and":lambda L,R:L&R,"or":lambda L,R:L|R,"xor":lambda L,R:L^R,
 "max":lambda L,R:max(L,R),"min":lambda L,R:min(L,R),"avg":lambda L,R:(L+R)//2,
 "sumsq":lambda L,R:L*L+R*R,"sqdiff":lambda L,R:(L-R)**2,
}
def fmtset(raw):
    if raw is None: return {}
    a=str(abs(raw))
    return {"raw":str(raw),"abs":a,"rev":a[::-1],"zpad2":f"{abs(raw):02d}",
            "dsum":str(dsum(raw)),"zpad3":f"{abs(raw):03d}"}

def wide_solves(examples):
    """Return True if ANY wide (order,op,fmt) yields a fully consistent cipher for all examples."""
    ex1i, ex1o = examples[0]; l1=len(ex1o)
    for oname, ofn in ORDERS.items():
        for opn, opfn in WIDE_OPS.items():
            for fmt in ("raw","abs","rev","zpad2","dsum","zpad3"):
                # ex1 candidate gen
                for L in range(100):
                    A,B=L//10,L%10
                    for R in range(100):
                        C,D=R//10,R%10
                        Lv,Rv=ofn(A,B,C,D)
                        rv=opfn(Lv,Rv)
                        ff=fmtset(rv).get(fmt)
                        if ff is None or not ff.isdigit() or len(ff)!=l1: continue
                        m={};r={};bad=False
                        for sym,dg in [(ex1i[0],A),(ex1i[1],B),(ex1i[3],C),(ex1i[4],D)]:
                            if sym in m and m[sym]!=dg: bad=True;break
                            if dg in r and r[dg]!=sym: bad=True;break
                            m[sym]=dg;r[dg]=sym
                        if bad: continue
                        for cs,dc in zip(ex1o,ff):
                            dg=int(dc)
                            if cs in m and m[cs]!=dg: bad=True;break
                            if dg in r and r[dg]!=cs: bad=True;break
                            m[cs]=dg;r[dg]=cs
                        if bad: continue
                        # verify remaining examples
                        allok=True
                        for (ii,oo) in examples[1:]:
                            rgs=[([m[s]] if s in m else range(10)) for s in (ii[0],ii[1],ii[3],ii[4])]
                            sub=False
                            for A2 in rgs[0]:
                              for B2 in rgs[1]:
                                for C2 in rgs[2]:
                                  for D2 in rgs[3]:
                                    Lv2,Rv2=ofn(A2,B2,C2,D2)
                                    rv2=opfn(Lv2,Rv2)
                                    ff2=fmtset(rv2).get(fmt)
                                    if ff2 is None or not ff2.isdigit() or len(ff2)!=len(oo): continue
                                    m2=dict(m);r2=dict(r);b2=False
                                    for sym,dg in [(ii[0],A2),(ii[1],B2),(ii[3],C2),(ii[4],D2)]:
                                        if sym in m2 and m2[sym]!=dg:b2=True;break
                                        if dg in r2 and r2[dg]!=sym:b2=True;break
                                        m2[sym]=dg;r2[dg]=sym
                                    if b2: continue
                                    for cs,dc in zip(oo,ff2):
                                        dg=int(dc)
                                        if cs in m2 and m2[cs]!=dg:b2=True;break
                                        if dg in r2 and r2[dg]!=cs:b2=True;break
                                        m2[cs]=dg;r2[dg]=cs
                                    if b2: continue
                                    sub=True;break
                                  if sub:break
                                if sub:break
                              if sub:break
                            if not sub: allok=False;break
                        if allok:
                            return True, (oname,opn,fmt)
    return False, None

def std_solves(examples):
    for combo in STD_COMBOS:
        maps=_consistent_mappings_for_combo(combo, examples)
        if maps: return True, combo
    return False, None

def main():
    seed=int(sys.argv[1]) if len(sys.argv)>1 else 0
    target=int(sys.argv[2]) if len(sys.argv)>2 else 40  # how many n_op>=2 no_solution cases to probe
    random.seed(seed)
    fails=load_fails()["cryptarithm_deduce"]
    random.shuffle(fails)
    ids={r["id"] for r in fails}
    P=load_prompts(ids)
    probed=0; gap_real=0; both_fail=0; std_ok=0
    rows=[]
    for r in fails:
        ex,q,op,nop = parse_prompt(P[r["id"]])
        if ex is None or nop<2: continue
        s_ok,_=std_solves(ex)
        if s_ok:
            std_ok+=1
            continue
        # std failed despite nop>=2 -> probe wide
        w_ok,wcombo=wide_solves(ex)
        probed+=1
        if w_ok: gap_real+=1
        else: both_fail+=1
        rows.append({"id":r["id"],"nop":nop,"wide_ok":w_ok,"wide_combo":wcombo,"true":r["answer"]})
        if probed>=target: break
    print(json.dumps({"std_ok_among_nop>=2_scanned":std_ok,
                      "probed_no_solution":probed,
                      "wide_grammar_solves(real gap)":gap_real,
                      "neither_solves":both_fail,
                      "rows":rows}, indent=1, default=str))

if __name__=="__main__":
    main()
