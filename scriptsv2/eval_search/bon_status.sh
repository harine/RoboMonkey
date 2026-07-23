#!/bin/bash
# One status snapshot: resume->50k progress + BoN eval cells done/running + a
# finetuned-vs-base success-rate comparison for the cells finished so far.
cd /mmfs1/home/harine/RoboMonkey
echo "===== STATUS $(date '+%Y-%m-%d %H:%M') ====="

echo "-- resume -> 50k --"
for j in 36393345:v1_ramp 36393346:v2_all 36393353:v3_dist; do
  jid="${j%%:*}"; nm="${j##*:}"
  st=$(squeue -j "$jid" -h -o '%T' 2>/dev/null); [[ -z "$st" ]] && st="$(sacct -j $jid -n -P -o State 2>/dev/null|head -1)"
  f=$(ls data/viz_q/slurm-ftR_${nm}-${jid}.out 2>/dev/null|head -1)
  line=$(timeout 15 tr '\r' '\n' < "$f" 2>/dev/null | grep -oE "[0-9]+/20001 \[[0-9:]+<[0-9:?]+, *[0-9.]+(s/it|it/s)" | tail -1)
  n=$(echo "$line"|grep -oE "^[0-9]+")
  echo "   $nm: $st  global ~$(( ${n:-0}+30000 ))/50000  (${line:-?})"
done

echo "-- BoN eval cells --"
echo "   running:$(squeue -u $USER -h -t RUNNING -o '%j' 2>/dev/null|grep -c '^ft_')  pending:$(squeue -u $USER -h -t PENDING -o '%j' 2>/dev/null|grep -c '^ft_')"

python3 - <<'PY'
import json
from pathlib import Path
NS=[1,2,4,8,16,32,64]
VT={"v2all30k":"v2_all@30k","v3dist30k":"v3_dist@30k"}
# policy -> (sub, finetuned-prefix-fmt, base-prefix)
SEARCH={
 "diff10k":("search_dqc_chunk","diff10k_{vt}_chunk8_predh8_roll4_softmax","diff10k_chunk8_predh8_roll4_softmax"),
 "diff33k":("search_dqc_chunk","diff33k_{vt}_chunk8_predh8_roll4_softmax","diffusion_chunk8_predh8_roll4_softmax"),
 "gauss20k":("search_dqc_chunk","gauss20k_{vt}_chunk8_predh8_roll4_softmax","gauss20k_chunk8_predh8_roll4_softmax"),
}
BC=("bc_dqc_chunk","bcH8_{vt}_chunk8_predh8_roll4_softmax_ddpm","bcH8_chunk8_predh8_roll4_softmax_ddpm")
def sr(sub,prefix,n):
    p=Path("data/eval")/sub/f"{prefix}_n{n}_seed1000_ep50"/"eval_log.json"
    try:
        d=json.load(open(p))
        if int(d.get("num_episodes",0))>=50: return int(round(100*d["success_rate"]))
    except Exception: pass
    return None
def row(label,sub,ftpfx,basepfx,vt):
    base=[sr(sub,basepfx,n) for n in NS]
    ft=[sr(sub,ftpfx.format(vt=vt),n) for n in NS]
    f=lambda v:"·" if v is None else f"{v:>3}"
    print(f"   {label:18s} base {' '.join(f(x) for x in base)}")
    print(f"   {label:18s}  ft  {' '.join(f(x) for x in ft)}")
print("   (success% over N=1 2 4 8 16 32 64; '·'=not done)")
for vt,vn in VT.items():
    print(f"  == verifier {vn} ==")
    for pol,(sub,ftp,bp) in SEARCH.items(): row(pol,sub,ftp,bp,vt)
    row("bc_h8",*BC,vt)
PY
echo "============================================="
