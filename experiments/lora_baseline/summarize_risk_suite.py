"""Read-only audit and aggregate disease results, including independent bootstrap.

Patient resampling is represented by multinomial counts. AUROC is computed from
grouped positive/negative masses, independently of the training metric loop.
Only aggregates and hashes are written to the requested report directory.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
from pathlib import Path
import statistics
import numpy as np
import torch
from checkpointing import sha256, load_epoch
from audit_record_resume import tree_equal


class Scores:
    def __init__(self, y, p, w):
        self.y, self.p, self.w = [np.asarray(x, dtype=np.float64) for x in (y,p,w)]
        self.order = np.argsort(self.p, kind='stable')
        self.starts = np.r_[0, np.flatnonzero(np.diff(self.p[self.order]) != 0)+1]

    def metrics(self, counts):
        weight = self.w * counts
        pos = np.add.reduceat((weight*self.y)[self.order], self.starts)
        neg = np.add.reduceat((weight*(1-self.y))[self.order], self.starts)
        denom = pos.sum()*neg.sum()
        auc = float(np.sum(pos*(np.cumsum(neg)-.5*neg))/denom) if denom else float('nan')
        brier = float(np.sum(weight*(self.p-self.y)**2)/weight.sum())
        return np.array([auc,brier])


def bootstrap(y, predictions, w, seed, resamples=2000):
    scorers = [Scores(y,p,w) for p in predictions]
    n = len(y); rng = np.random.default_rng(seed)
    point = np.array([s.metrics(np.ones(n)) for s in scorers])
    draws = np.empty((resamples,len(scorers),2))
    for i in range(resamples):
        counts = np.bincount(rng.integers(0,n,n),minlength=n)
        draws[i] = [s.metrics(counts) for s in scorers]
    if not np.isfinite(draws).all():
        raise ValueError('Bootstrap replicate lacks finite metrics')
    contrasts=[]
    for j in range(1,len(scorers)):
        ci = np.percentile(draws[:,0]-draws[:,j],[2.5,97.5],axis=0)
        contrasts.append({name:dict(point=float(point[0,k]-point[j,k]),
            ci_low=float(ci[0,k]),ci_high=float(ci[1,k]),n_patients=n,
            n_resamples=resamples,seed=seed) for k,name in enumerate(('auroc','brier'))})
    return point,contrasts


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ('runs','data','preflight','out'):
        p.add_argument('--'+flag,type=Path,required=True)
    p.add_argument('--wandb-entity',required=True)
    args=p.parse_args()
    status=json.loads((args.runs/'status.json').read_text())
    if status['status']!='completed' or len(status['completed'])!=18:
        raise ValueError('Expected all 18 completed experiments')
    assert all(sha256(k)==v for k,v in status['source_sha256'].items())
    reports={}; probabilities={}; testids=np.load(args.data/'test/patient_id.npy')
    def remote_audit(run):
        import wandb
        r=wandb.Api(timeout=30).run(args.wandb_entity+'/fermat-lora-baseline/'+run['tracking']['run_id'])
        assert r.state=='finished'
        for k,v in run['summary']['test'].items():
            assert np.isclose(r.summary['test/'+k],v,rtol=0,atol=1e-12)
        files=sorted(f.name for f in r.files())
        assert files==['config.yaml','wandb-summary.json']
        return dict(name=run['name'],run_id=run['tracking']['run_id'],url=r.url,state=r.state,files=files)
    with ThreadPoolExecutor(max_workers=4) as pool:
        remote=list(pool.map(remote_audit,status['completed']))
    for run in status['completed']:
        folder=args.runs/run['name']; log=json.loads((folder/'summary.json').read_text())
        manifest=json.loads((folder/'last_training.json').read_text())
        payload=torch.load(folder/manifest['checkpoint'],map_location='cpu',weights_only=True)
        payload,_=load_epoch(folder/'last_training.json',payload['contract'])
        state=torch.load(folder/'trainable_state.pt',map_location='cpu',weights_only=True)
        assert tree_equal(state,payload['best_state'])
        assert payload['optimizer_steps']==run['summary']['optimizer_steps']
        assert run['summary']['audit']['frozen_base_unchanged']
        assert min(log['epochs'],key=lambda r:r['validation_weighted_bce'])['epoch']==run['summary']['selected']['epoch']
        saved=torch.load(folder/'test_scores.pt',map_location='cpu',weights_only=True)
        assert np.array_equal(saved['patient_id'].numpy(),testids)
        values=saved['probabilities'].numpy(); assert np.isfinite(values).all() and np.all((values>=0)&(values<=1))
        if run['group']=='lora' and run['seed']==1:
            assert tree_equal(saved,torch.load(folder/'reload/test_scores.pt',map_location='cpu',weights_only=True))
        probabilities[run['name']]=values
        reports[run['name']]=dict(disease=run['disease'],group=run['group'],seed=run['seed'],
            selected_epoch=run['summary']['selected']['epoch'],completed_epochs=len(log['epochs']),
            test=run['summary']['test'],elapsed_seconds=run['summary']['elapsed_seconds'],
            epoch_seconds=[r['epoch_seconds'] for r in log['epochs']],
            checkpoint_sha256=manifest['sha256'],predictions_sha256=sha256(folder/'test_scores.pt'),
            independent_reload_exact=True if run['group']=='lora' and run['seed']==1 else None)
    diseases={}; max_error=0.
    for d in ('ckd','fatty_liver','diabetes'):
        y=np.load(args.data/'test'/f'outcome_{d}.npy'); w=np.load(args.data/'test'/f'weight_{d}.npy'); keep=w>0
        y,w=y[keep],w[keep]
        original=np.load(args.data/'test'/f'baseline_probability_{d}.npy')[keep]
        refit=torch.load(args.preflight/f'gate_{d}/test_scores.pt',map_location='cpu',weights_only=True)
        assert np.array_equal(refit['patient_id'].numpy(),testids)
        reference=Scores(y,original,w).metrics(np.ones(len(y)))
        for seed in (1,2,3):
            name=f'lora_{d}_seed{seed}'; run=reports[name]; control=f'C_{d}_seed{seed}'
            point,contrasts=bootstrap(y,[probabilities[name][keep],original,
                refit['probabilities'].numpy()[keep],probabilities[control][keep]],w,seed)
            old=json.loads((args.runs/name/'summary.json').read_text())['summary']['paired_minus_saved_v1']
            for metric in ('auroc','brier'):
                for key in ('point','ci_low','ci_high'):
                    error=abs(contrasts[0][metric][key]-old[metric][key]); max_error=max(max_error,error)
                    assert error<1e-12,(name,metric,key,error)
            for row, target in ((point[0],name),(point[3],control)):
                assert np.allclose(row,[reports[target]['test']['weighted_auroc'],reports[target]['test']['weighted_brier']],rtol=0,atol=1e-12)
            run['contrasts']=dict(zip(('minus_saved_v1','minus_refit_v1','minus_matched_C'),contrasts))
            print(json.dumps(dict(audited=name,max_bootstrap_error=max_error)),flush=True)
        groups={}
        for group in ('C','lora'):
            groups[group]={k:dict(mean=statistics.mean(reports[f'{group}_{d}_seed{s}']['test'][k] for s in (1,2,3)),
                sample_sd=statistics.stdev(reports[f'{group}_{d}_seed{s}']['test'][k] for s in (1,2,3)))
                for k in ('weighted_auroc','weighted_brier')}
        diseases[d]=dict(observed=len(y),cases=int(y.sum()),baseline=dict(weighted_auroc=float(reference[0]),weighted_brier=float(reference[1])),groups=groups)
    result=dict(status='COMPLETED_AND_AUDITED',started=status['started'],finished=status['finished'],
        wall_seconds=(datetime.fromisoformat(status['finished'])-datetime.fromisoformat(status['started'])).total_seconds(),
        training_commit=status['execution_commit'],runs=reports,diseases=diseases,wandb=remote,
        audit=dict(all_18_checkpoints_verified=True,all_18_test_metrics_verified=True,
            all_9_saved_v1_bootstraps_independently_verified=True,maximum_absolute_bootstrap_difference=max_error,
            method='Independent NumPy weighted positive/negative grouped masses with patient multiplicity counts',
            source_hashes_unchanged=True),
        interpretation=dict(primary_seed=1,sensitivity_seeds=[2,3],resamples=2000,interval='95% percentile',
            fixed_observation_weights=True,multiplicity_adjusted=False,refit_uncertainty_included=False,
            seeds_pooled_as_patients=False,C_and_refit_contrasts='secondary'))
    args.out.mkdir(parents=True,exist_ok=True)
    target=args.out/'summary.json'
    if target.exists():raise ValueError('Refusing to overwrite an existing final report')
    target.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(wall_hours=result['wall_seconds']/3600,diseases=diseases,audit=result['audit']),indent=2))


if __name__=='__main__':
    main()
