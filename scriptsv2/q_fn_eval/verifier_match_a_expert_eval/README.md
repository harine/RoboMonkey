# verifier_match_a_expert_eval — ARCHIVED (do not use to judge the critic)

These scripts evaluate the DQC critic by how well it **matches the expert action a\***
— the advantage A = Q(a\*) − Q(a_pred), RMSE-to-a\*, rank-of-a\*, and top-1 match to the
lowest-RMSE candidate.

**This is the WRONG objective for a best-of-N critic.** The critic does not need to
reproduce the demo action; it only needs to select candidates that **succeed**. Judge the
critic by **BoN success rate** (the SimplerEnv sweep: `scriptsv2/eval_search/eval_search*.sh`
+ `scriptsv2/eval_search/build_bon_table.py`), not by match to a\*.

Kept here for reference only. Results archived under
`data/eval/_verifier_comparison/verifier_match_a_expert_eval/`.

- `plot_advantage_investigation.py` — A = Q(a\*) − Q(a_pred) distributions.
- `plot_offline_diagnostics.py` — q_vs_rmse / ranking_correctness / astar_rank_quality.
- `plot_ft_progression.py` — per-checkpoint advantage / RMSE / top-1 (+ a\*-Q AUC).
- `build_matrix.py` — the a\*-match matrix (advantage + ranking) §1 of the old doc.
- `ft_plots.sbatch` — launcher that drove the above on rescored candidates.
