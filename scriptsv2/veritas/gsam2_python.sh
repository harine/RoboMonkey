#!/usr/bin/env bash
# Python for the Grounded-SAM-2 tracking-server subprocess (GROUNDED_SAM2_PYTHON): the
# gsam2 env (torch 2.12 cu130, sam2, transformers 4.47.1), without the eval's
# LD_LIBRARY_PATH (simpler_env/lib), which would shadow the cu130 torch libraries.
unset LD_LIBRARY_PATH
exec /gscratch/robotics/harine/miniconda3/envs/gsam2/bin/python "$@"
