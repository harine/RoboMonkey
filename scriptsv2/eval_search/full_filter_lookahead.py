"""Simulator-only diagnostic for the full strict eggplant demo filter."""
import copy
import numpy as np


def runs(mask):
    out=[];n=0
    for x in mask:
        if x:n+=1
        elif n:out.append(n);n=0
    if n:out.append(n)
    return out


def evaluate_filter(history):
    d=np.array([x['deb'] for x in history]);g=np.array([x['dge'] for x in history])
    contact=np.array([x['contact'] for x in history],bool);opened=np.array([x['open'] for x in history],bool)
    finite=bool(np.isfinite(d).all() and np.isfinite(g).all())
    if not finite:return dict(passed=False,score=-1e6,finite=False)
    first=np.flatnonzero(contact);grasp=int(g.argmin());place=int(d.argmin())
    carry=float(g[grasp:max(grasp,place)+1].max());tol=max(.012,.06*d[0])
    backtrack=float((d-np.minimum.accumulate(d)).max());ratio=float(d[-1]/max(d[0],1e-6))
    success=any(x['success'] for x in history)
    if len(first):
        fc=int(first[0]);hi=max(fc,place)+1
        closes=int(((opened[:-1]&~opened[1:]) & (np.arange(1,len(d))<=fc+1)).sum())
        reopen=sum(n>=2 for n in runs(opened[fc:hi]))
        gap=max(runs(~contact[fc:hi]),default=0)
        rise=float(g[fc:hi].max()-g[fc])
    else:closes=reopen=gap=0;rise=0.
    checks=dict(success=bool(success),finite=finite,net_decrease=ratio<=.9,
        bounded_backtrack=backtrack<=tol,carry_proximity=carry<=.03,
        grasp_observed=bool(len(first)),first_try=closes<=1,no_reopen=reopen==0,
        contact_continuity=gap<=1,no_separation_rise=rise<=.015)
    checks={k:bool(v) for k,v in checks.items()}
    passed=all(checks.values())
    # Bounded dense terms support progress before a full successful prefix exists.
    violations=[not success,max(0,ratio-.9),max(0,backtrack/tol-1),max(0,carry/.03-1),
                not len(first),max(0,closes-1),reopen,max(0,gap-1),max(0,rise/.015-1)]
    penalty=float(np.clip(violations,0,1).mean())
    progress=float(np.clip(1-ratio,-1,1))
    score=(float(passed)+progress-penalty)/2
    return dict(passed=passed,score=score,checks=checks,final_distance=float(d[-1]),
        net_ratio=ratio,backtrack_m=backtrack,backtrack_tolerance_m=tol,carry_max_m=carry,
        first_contact=None if not len(first) else int(first[0]),closest_placement=place,
        closes_to_contact=closes,reopen_runs=reopen,max_contact_gap=gap,separation_rise_m=rise)


class FullFilterLookahead:
    def __init__(self,env,source,target,convert,seed,task='widowx_put_eggplant_in_basket'):
        import simpler_env
        self.real_env=env;self.real_source=source;self.real_target=target
        self.probe_wrapper=simpler_env.make(task)
        self.probe_wrapper.reset(seed=int(seed))
        self.env=self.probe_wrapper.unwrapped
        self.source=self.env.episode_source_obj;self.target=self.env.episode_target_obj;self.convert=convert
        self.history=[];self.validation_errors=[];self.score_variation=[]

    def observe(self,opened,success=False):
        tcp=np.asarray(self.real_env.tcp.pose.p);src=np.asarray(self.real_source.pose.p);tgt=np.asarray(self.real_target.pose.p)
        row=dict(dge=float(np.linalg.norm(tcp-src)),deb=float(np.linalg.norm(src-tgt)),
                 contact=bool(self.real_env.agent.check_grasp(self.real_source)),open=bool(opened),success=bool(success))
        self.history.append(row);return row

    def controllers(self,env=None):
        def walk(c):
            yield c
            for child in getattr(c,'controllers',{}).values():yield from walk(child)
        return list(walk((env or self.env).agent.controller))

    def snapshot(self):
        env=self.real_env
        # Env state excludes controller targets and task counters; capture explicitly.
        return dict(sim=env.get_state().copy(),
            drives=env.agent.robot.get_drive_target().copy(),drive_vel=env.agent.robot.get_drive_velocity_target().copy(),
            controller_runtime=[{k:copy.deepcopy(getattr(c,k)) for k in ['_step','_start_qpos','_target_qpos','_last_drive_qpos_targets','_interpolation_path','_interpolation_path_vel','_integral'] if hasattr(c,k)} for c in self.controllers(env)],agent=copy.deepcopy(env.agent.get_state()),
            counters={k:copy.deepcopy(getattr(env,k)) for k in ['_elapsed_steps','consecutive_grasp','episode_stats']},
            rng={k:getattr(env,k).get_state() for k in ['_episode_rng','_main_rng'] if hasattr(env,k) and hasattr(getattr(env,k),'get_state')})

    def restore(self,snap):
        self.env.set_state(snap['sim']);self.env.agent.set_state(copy.deepcopy(snap['agent']))
        for c,state in zip(self.controllers(),snap['controller_runtime']):
            for k,v in state.items():setattr(c,k,copy.deepcopy(v))
        self.env.agent.robot.set_drive_target(snap['drives'])
        self.env.agent.robot.set_drive_velocity_target(snap['drive_vel'])
        for k,v in snap['counters'].items():setattr(self.env,k,copy.deepcopy(v))
        for k,v in snap['rng'].items():getattr(self.env,k).set_state(v)

    def branch(self,actions):
        history=list(self.history)
        for action in actions:
            self.env.step_action(self.convert(action.copy()))
            self.env._elapsed_steps+=1
            info=self.env.evaluate()
            tcp=np.asarray(self.env.tcp.pose.p);src=np.asarray(self.source.pose.p);tgt=np.asarray(self.target.pose.p)
            history.append(dict(dge=float(np.linalg.norm(tcp-src)),deb=float(np.linalg.norm(src-tgt)),
                contact=bool(self.env.agent.check_grasp(self.source)),open=bool(action[6]>=.5),
                success=bool(info['success'])))
        result=evaluate_filter(history)
        result['eef_endpoint']=np.asarray(self.env.tcp.pose.p).tolist()
        result['object_endpoint']=np.asarray(self.source.pose.p).tolist()
        return result,self.env.get_state().copy()

    def score(self,candidates):
        snap=self.snapshot();results=[]
        try:
            for actions in candidates:
                self.restore(snap);result,state=self.branch(actions);results.append(result)
            # Repeat candidate zero after all branches to detect restoration drift/order effects.
            self.restore(snap);a,sa=self.branch(candidates[0])
            self.restore(snap);b,sb=self.branch(candidates[0])
            error=float(max(np.linalg.norm(np.array(a[k])-np.array(b[k])) for k in ['eef_endpoint','object_endpoint']))
            first_error=float(max(np.linalg.norm(np.array(a[k])-np.array(results[0][k])) for k in ['eef_endpoint','object_endpoint']))
            error=max(error,first_error);self.validation_errors.append(error)
            self.score_variation.append(float(abs(a['score']-results[0]['score'])))
            if not np.isfinite(error):raise RuntimeError('nonfinite simulator replay state')
        finally:self.restore(snap)
        return results

    def close(self):
        self.probe_wrapper.close()
