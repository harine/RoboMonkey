# Design: advantage policy-gradient  ∇log π · (Q(s,a_expert) − Q(s,a_pred))

**Status: design only (not implemented).** This specifies how to add an advantage-weighted
policy-gradient term to the search policy training, using a (finetuned) DQC chunk Q as the critic.

## Idea
We observed the DQC verifier ranks candidates near chance vs the demo action a\* (the verifier scores its
own pick *above* a\*: advantage A = Q(a\*)−Q(a_pred) ≈ −3.3 train / −1.7 val). If a finetuned Q makes A a
meaningful signal, we can **push the policy toward the expert wherever the critic says the expert is better
than the policy's prediction**, via the gradient

```
∇θ L_pg = − E_s[ stop_grad(A(s)) · ∇θ log π_θ(a_expert | s) ] ,   A(s) = Q(s, a_expert) − Q(s, a_pred)
```

- A > 0 (critic prefers expert) → increase log π(a_expert): move the policy toward the demo.
- A ≤ 0 (critic prefers the policy's own action) → little/negative push.

This is advantage-weighted regression (AWR) with a **linear** weight; the exponential variant
`w = exp(β·A)` (clamped) is the alternative (see `AAWRImagePolicy`).

## Where it plugs in (gaussian policy only)
`diffusion_policy/diffusion_policy/policy/search_policy_robomonkey.py` → `SearchPolicyRoboMonkey`:
- `forward(...)` builds a `torch.distributions.Normal` (`dist`) over `(B, max_actions, horizon, action_dim)`
  — has `dist.log_prob` and `dist.mean` (≈ lines 206–273).
- `compute_loss(batch)` (≈ lines 389–424) already:
  - normalizes obs + `target_actions = a*` (expert);
  - runs `actions, values = _run_search(self, batch['obs'], self.max_actions-1)` — **the policy's sampled
    candidates `actions` and their verifier Q `values`** (the in-process DQC critic);
  - builds `dist`, and computes `log_prob = dist.log_prob(target_actions).sum(dim=(-1,-2))` → current NLL loss.

So everything needed is already in scope:
- `log π(a_expert)` = `dist.log_prob(target_actions)` (already computed).
- `a_pred` = the policy's prediction. Two clean choices: **(a)** `dist.mean` (the policy's mode), or **(b)**
  a candidate sample from `_run_search` (on-policy). Recommend (a) `dist.mean` for low variance, or the
  per-state mean over the sampled candidates.
- `Q(s, a_expert)` and `Q(s, a_pred)`: call `self.verifier.get_value(obs_dict, chunk)` (same verifier used in
  `_run_search`) on the expert chunk and on `a_pred`'s chunk. `Q(s,a_expert)` may need one extra
  `get_value` call (the search only scores sampled candidates, not a\*).

### Sketch (inside `compute_loss`, after `dist` is built)
```python
with torch.no_grad():
    q_expert = self.verifier.get_value(obs_dict, expert_chunk)        # (B,)
    a_pred   = dist.mean[:, 0]            # policy mode, one token       (B,horizon,Da)
    q_pred   = self.verifier.get_value(obs_dict, unnormalize(a_pred))  # (B,)
    adv = (q_expert - q_pred)                                          # (B,) ; optionally normalize
    w = adv                              # linear AWR ; or torch.exp(beta*adv).clamp(max=W_MAX)
logp_expert = dist.log_prob(target_actions).sum(dim=(-1,-2))[:, 0]     # (B,)
L_pg = -(w.detach() * logp_expert).mean()
total = nll_loss + self.pg_lambda * L_pg
```
Add hyperparameters: `pg_lambda` (weight), `beta` (if exp), `adv_norm` (z-score advantages per batch),
`W_MAX` (clamp). Use the **finetuned** DQC as `self.verifier`, **frozen** (`stop_grad` through Q via
`no_grad` — the gradient flows only through `log π`).

## Critic choice & sign sanity
- Use the **best finetuned Q** (WS1–WS3); the base Q ranks at chance, so its advantage is unreliable and
  L_pg would push the policy toward whatever the bad critic prefers. The whole update is only as good as Q.
- a_pred must be a *current-policy* action so A measures "expert better than what I'd do now"; recompute each
  step (on-policy). Normalizing A per batch (subtract mean / divide std) controls scale and variance.

## Diffusion / BC policies (no tractable log_prob)
`SearchPolicyRoboMonkeyDiffusion` and the BC `DiffusionUnetImagePolicy` have **no closed-form log π** for a
sampled action, so the above doesn't apply directly. Options (not committed):
1. **Advantage-reweighted denoising loss**: weight the per-sample diffusion MSE by `w = exp(β·A)` (AWR for
   diffusion / "AW-flow") — reuses the existing denoiser loss, no log_prob needed.
2. **Auxiliary gaussian readout head** trained alongside the diffusion policy, take the PG on that head.
3. **Score-function / reparam estimators** — heavier; likely unnecessary given (1).

## Risks
- Q quality dominates — gate on the WS3 progression (only enable PG once the finetuned Q's val-RMSE/AUC and
  advantage clearly beat base).
- Extra `get_value` calls per step add JAX↔torch overhead in the training loop (the search already calls the
  verifier, so marginal).
- High variance / collapse toward the expert (mode-seeking) — control with `pg_lambda` schedule + advantage
  normalization + keeping the NLL term.
