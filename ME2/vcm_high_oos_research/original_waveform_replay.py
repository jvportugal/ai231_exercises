"""Waveform/runtime replay of a joint-development time-warp policy.

TEST and historical HOLDOUT both selected this policy; neither is independent.
Uses existing ONNX models and frontend; no model changes or deployment.
"""
import json,sys,time
from pathlib import Path
import numpy as np,pandas as pd,torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT.parent/'oos_priority_20261028'));sys.path.insert(0,str(ROOT/'research_runtime'))
import vcm_infer as runtime
from numeral_runtime import NumeralResearchSession
from vcm_selective_scope import _Outputs
from vcm_long_window import LongWindowSession
from suite import read_baseline
from train_centered import decode_wav
from evaluate_long_window_holdout import load_frame
from experiment_weight_soup import labels as data_labels,load_oos
from eval_ens5c import load_split
import config as C
OUT=ROOT/'feature_timewarp_20261102';SOURCE=ROOT/'clean_numeral_expert_20261030';which=sys.argv[1];assert which in ('test','holdout');torch.set_num_threads(2);start=time.monotonic();policy_choice=json.loads((OUT/'eighteen_extended_robust_development.json').read_text())['stages'];bundle=ROOT/'research_runtime/vcm_benchmark_joint_recovery';labels=runtime.load_labels(str(bundle/'labels.json'));policy=runtime.load_policy(str(bundle/'policy.json'));session=NumeralResearchSession(bundle/'ensemble.json');
if which=='test':
 frame=pd.concat([load_split('test'),load_oos(str(Path(C.HF_DIR)/'synthetic_negatives/tune_oos-*.parquet'))],ignore_index=True);data,_,_,_,_=read_baseline()
else:
 hold,_=load_frame('data/holdout-*.parquet');extra,_=load_frame('synthetic_negatives/test-*.parquet');extra=extra[(~extra.neg_kind.fillna('').isin(['babble','noise_source']))&(~extra.source_files.fillna('').str.contains('noise/',regex=False))].reset_index(drop=True);frame=pd.concat([hold,extra],ignore_index=True);ya,yb=data_labels(frame);data={'ya':ya,'yb':yb,'real':frame.is_synthetic.fillna(1).astype(int).to_numpy()==0,'test':np.arange(len(frame))<len(hold)}
cache=np.load(ROOT/f'clean_micro_addon_20261102/research_{which}_cached.npz');expected=(cache['a'],cache['b']);assert len(frame)==len(expected[0]);intents=labels['intents'];values=labels['value_tokens'];quality=session.extra_policy['quality_thresholds_db'];counts={k:{'n':0,'before':0,'after':0} for k in ('real_command','synthetic_command','native_oos','tune_oos')};cached_diff=[];changes=[]
def numeric(result):
 ok,_=runtime.safe_decision(result[0],result[1],result[3])
 if not ok:return 19,18
 token=result[3].get('slot_token');return intents.index(result[0]),(values.index(token) if token in values else 18)
def warp(x,factor):
 a=torch.from_numpy(np.asarray(x))
 if factor>=200:
  shift=factor-200 if factor<210 else -(factor-210);y=a.clone();v=a[:,:1];n=v.shape[-2];v=F.pad(v,(0,0,max(0,shift),max(0,-shift)));y[:,:1]=v[:,:,:n] if shift>=0 else v[:,:,-n:];return y.numpy()
 n=a.shape[-1];length=int(round(n*factor/100));y=F.interpolate(a,size=(a.shape[-2],length),mode='bilinear',align_corners=False);y=F.pad(y,((n-length)//2,n-length-(n-length)//2)) if length<n else y[...,(length-n)//2:(length-n)//2+n]
 if y.shape[1]>1:y[:,1:]=(y[:,1:]>=.5).float()
 return y.numpy()
def softmax(x):
 x=x-np.max(x,axis=-1,keepdims=True);p=np.exp(x);return p/p.sum(axis=-1,keepdims=True)
for i,audio in enumerate(frame.audio):
 wave=decode_wav(audio['bytes'] if isinstance(audio,dict) else audio);ref=runtime.classify(session,labels,wave,policy=policy);oa,ob=numeric(ref);feed,info=session.prepare_feed(wave);ta,tb=oa,ob
 current_a,current_b=oa,ob;proposal_cache={}
 if info.get('span') is not None:
  original=LongWindowSession.run(session,None,feed);oldp=[softmax(x.reshape(-1,6,3)).reshape(x.shape) if j==2 else softmax(x) for j,x in enumerate(original[:3])];expert=session.expert.run(None,{'mel':feed['long']});extra=session.extra.run(None,{'mel':feed['long']})
  for stage in policy_choice:
   choice=stage['choices'][current_a]
   if choice['kind']=='none' or choice['threshold']>=1e8:continue
   key=(choice['factor'],choice['weight'])
   if key not in proposal_cache:
    warped=dict(feed,long=warp(feed['long'],choice['factor']),short=warp(feed['short'],choice['factor']));altered=LongWindowSession.run(session,None,warped);newp=[softmax(x.reshape(-1,6,3)).reshape(x.shape) if j==2 else softmax(x) for j,x in enumerate(altered[:3])];bp=[(1-choice['weight'])*x+choice['weight']*y for x,y in zip(oldp,newp)];mixed=[np.log(np.maximum(x,1e-12)).astype(np.float32) for x in bp]+original[3:];base_output=runtime.classify(_Outputs(mixed,info),labels,wave,policy=policy);prior=session._stage(labels,wave,info,policy,mixed,base_output,expert,session.scope_policy);proposal=session._stage(labels,wave,info,policy,mixed,prior,extra,session.extra_policy,float(session.extra_policy.get('scope_mix',0.)),expert[4]);pa,pb=numeric(prior);ca,cb=numeric(proposal);idx=pa if pa<19 else ca
    if idx<19:
     x=np.asarray(wave,dtype=np.float64);x=np.pad(x,(0,(-len(x))%320));rms=np.sqrt(np.mean(x.reshape(-1,320)**2,axis=1)+1e-12);low,high=np.percentile(rms,[20,95]);contrast=20*np.log10(high/low)
     if contrast<float(quality[idx]):ca,cb=pa,pb
    gain=float(bp[0][0,ca]-oldp[0][0,ca]);margin=float(np.log(max(float(bp[0][0,ca]),1e-10))-np.log(max(float(bp[0][0,oa]),1e-10)));proposal_cache[key]=(ca,cb,gain,margin)
   ca,cb,gain,margin=proposal_cache[key];value=margin if choice['kind']=='margin' else gain
   if value>=choice['threshold']:current_a,current_b=ca,cb
 ta,tb=current_a,current_b
 if (oa,ob)!=(expected[0][i],expected[1][i]):cached_diff.append(i)
 ya,yb=int(data['ya'][i]),int(data['yb'][i]);group='tune_oos' if not data['test'][i] else ('native_oos' if ya==19 else ('real_command' if data['real'][i] else 'synthetic_command'));truth=lambda aa,bb:aa==ya and (yb==18 or bb==yb);counts[group]['n']+=1;counts[group]['before']+=truth(oa,ob);counts[group]['after']+=truth(ta,tb)
 if (ta,tb)!=(oa,ob):changes.append({'index':i,'before':[oa,ob],'after':[ta,tb],'truth':[ya,yb]})
 if i%250==0:print('RUNTIME',which,i+1,'/',len(frame),'elapsed',time.monotonic()-start,flush=True)
report={'protocol':__doc__,'which':which,'policy':policy_choice,'counts':counts,'baseline_cache_differences':cached_diff,'changes':changes,'elapsed':time.monotonic()-start,'physical_pi_test':False,'deployed':False};(OUT/f'eighteen_extended_robust_actual_{which}.json').write_text(json.dumps(report,indent=2)+'\n');print('RESULT',json.dumps({'counts':counts,'cache_differences':len(cached_diff),'changed':len(changes)}),flush=True)
