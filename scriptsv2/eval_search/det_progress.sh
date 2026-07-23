#!/bin/bash
# Per-run progress for the deterministic sub50 campaign: lists every SR + VIZ run
# (task x policy x steps) with how far it's gotten. Used by the 30-min tracker.
cd /mmfs1/home/harine/RoboMonkey
python3 - <<'PY'
import glob, os, re, json, collections
E="data/eval"; NALL=[1,2,4,8,16,32,64]
SRC=re.compile(r"^(eggplant|stack_cube|spoon_on_towel)_(?:sub50_)?([0-9]+k?|bc)_st([0-9]+)_n([0-9]+)_seed1000_ep50(_argmax)?$")
VIZ=re.compile(r"^(eggplant|stack_cube|spoon_on_towel)_([0-9]+|bc)_st([0-9]+)_n([0-9]+)$")
sr=collections.defaultdict(dict)     # (task,kind,sel,steps) -> {N: (sr, dist)}
for d in glob.glob(f"{E}/*/*_seed1000_ep50")+glob.glob(f"{E}/*/*_seed1000_ep50_argmax"):
    m=SRC.match(os.path.basename(d))
    if not m: continue
    try:
        j=json.load(open(f"{d}/eval_log.json"));
        if int(j.get("num_episodes",0))<50: continue
    except: continue
    sel='argmax' if m.group(5) else 'softmax'
    k=(m.group(1),'bc' if m.group(2)=='bc' else 'search',sel,int(m.group(3)))
    sr[k][int(m.group(4))]=(round(100*j['success_rate']), j.get('mean_obj_goal_dist_min'))
viz=collections.defaultdict(set)     # (task,kind,sel,steps) -> set(N with videos)
for cell in glob.glob(f"{E}/*/_viz/*_st*_n*")+glob.glob(f"{E}/*/_viz_argmax/*_st*_n*"):
    m=VIZ.match(os.path.basename(cell))
    if not m: continue
    sel='argmax' if '_viz_argmax' in cell else 'softmax'
    if glob.glob(f"{cell}/search_q/videos/*.mp4"):
        viz[(m.group(1),'bc' if m.group(2)=='bc' else 'search',sel,int(m.group(3)))].add(int(m.group(4)))

print("--- SR runs (N done / 7 ; SR% at N1..N64) ---")
for k in sorted(sr):
    r=sr[k]; row=" ".join((f"{r[n][0]}" if n in r else "·") for n in NALL)
    pk=max((v[0],n) for n,v in r.items()) if r else (0,0)
    print(f"  {k[0]:14s} {k[1]:6s} {k[2]:7s} st{k[3]:<3d}: {len(r)}/7  [{row}]  peak {pk[0]}%@N{pk[1]}")
print("--- VIZ runs (cells with videos / 7) ---")
allviz=[(t,kd,'softmax',st) for t in ("eggplant","stack_cube","spoon_on_towel") for kd in ("search","bc")
        for st in ( (60 if t!='eggplant' else 120,500) if kd=='search' else (120,) )]
for k in sorted(set(list(viz)+allviz)):
    print(f"  {k[0]:14s} {k[1]:6s} {k[2]:7s} st{k[3]:<3d}: {len(viz.get(k,set()))}/7 N with videos")
PY
