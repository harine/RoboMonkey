"""Generate MUSE (USE-3 multilingual) task instruction embeddings for the DQC
language-conditioned critic, in the SAME embedding space as task_emb_use3.npy.

Validates by re-encoding the eggplant instruction and comparing to the existing
task_emb_use3.npy (should be ~identical), then writes task_emb_<task>_use3.npy.
Run in an env with tensorflow + tensorflow_hub + tensorflow_text (dqc_jax)."""
import os, sys
import numpy as np

REPO = "/mmfs1/home/harine/RoboMonkey"
sys.path.insert(0, os.path.join(REPO, "dqc"))
os.environ.setdefault("TFHUB_CACHE_DIR", "/gscratch/robotics/harine/tfhub_cache")

from utils.text_processing import make_text_processor  # noqa: E402

OUT = os.path.join(REPO, "scriptsv2", "q_fn_eval")

# Canonical SIMPLER WidowX instructions (from ManiSkill2_real2sim put_on_in_scene.py
# get_language_instruction): the exact strings the envs emit.
TASKS = {
    "stack": "stack the green block on the yellow block",
    "spoon": "put the spoon on the towel",
}
# eggplant strings to identify which one produced the existing task_emb_use3.npy
EGG = ["put the eggplant in the basket", "put eggplant into yellow basket"]

tp = make_text_processor("muse_embedding")

# --- validation: does our encoder reproduce the stored eggplant embedding? ---
ref = np.load(os.path.join(OUT, "task_emb_use3.npy")).astype(np.float32).reshape(-1)
print(f"[ref] task_emb_use3.npy shape={ref.shape} norm={np.linalg.norm(ref):.4f}")
for s in EGG:
    e = np.asarray(tp.encode([s]), np.float32)[0]
    cos = float(e @ ref / (np.linalg.norm(e) * np.linalg.norm(ref)))
    print(f"[val] MUSE('{s}')  cos_to_ref={cos:.4f}  maxdiff={np.abs(e-ref).max():.4f}")

# --- generate the new task embeddings ---
for k, instr in TASKS.items():
    e = np.asarray(tp.encode([instr]), np.float32)[0]
    assert e.shape == (512,), e.shape
    p = os.path.join(OUT, f"task_emb_{k}_use3.npy")
    np.save(p, e)
    print(f"[out] {k}: '{instr}' -> {p}  shape={e.shape} norm={np.linalg.norm(e):.4f}")
