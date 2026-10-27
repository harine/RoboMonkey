# Source before eval_search.sh; task/checkpoint/camera/plan/split remain explicit.
# Training counterpart: robomonkey_veritas_h16, campaign kind=uniform.
export VERIFIER_TYPE=veritas
export OBS_CORRUPTION=clean
export EVAL_OBS_NOISE_LEVEL=
export FULL_FILTER_LOOKAHEAD=0
unset SIM_LOOKAHEAD_MODE SUBTASK_RECIPE
export BC_GRIPPER_AUG=0
export RAND_AUG_K=0
export EVAL_CHUNK_SELECT=0
export VERITAS_HORIZON=8
export N_ACTION_STEPS=8
export VERITAS_ACTION_POS_SCALE=null
export VERITAS_PLAN_DONE_MODE=zero
export BC_SAMPLER=ddim
export BC_VERIFIER_SCORE=full
export BC_BATCH_CANDIDATES=1
export USE_EMA=1
export N_SAMPLES=16
export MODE=argmax
# Set MODE=final after sourcing for the final-candidate comparison.
# Plan generation/cache settings intentionally remain caller-controlled.
