"""Research-only stacked scope session; never modifies the shipped bundle."""
import json
from pathlib import Path
import numpy as np,onnxruntime as ort
import vcm_infer as runtime
from vcm_selective_scope import SelectiveScopeSession,_Outputs
from vcm_long_window import LongWindowSession

class NumeralResearchSession(SelectiveScopeSession):
 def __init__(self,manifest_path,extra_session=None,policy_file='candidate_quality_policy.json'):
  super().__init__(manifest_path)
  self.extra_policy=json.loads((Path(manifest_path).parent/policy_file).read_text())['policy']
  quality=self.extra_policy.get('quality_thresholds_db')
  if (self.extra_policy.get('verifier')!='scope' or not self.extra_policy.get('rescue_only') or
      len(self.extra_policy.get('thresholds',[]))!=19 or len(self.extra_policy.get('rescue_thresholds',[]))!=19 or
      (quality is not None and len(quality)!=19)):
   raise ValueError('unsupported research candidate policy')
  if extra_session is None:
   options=ort.SessionOptions();options.intra_op_num_threads=2
   extra_session=ort.InferenceSession(str(Path(manifest_path).parent/'candidate_expert.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
  self.extra=extra_session

 @staticmethod
 def _stage(labels,waveform,info,policy,baseline,original,expert,scope_policy,scope_mix=0.,old_scope=None):
  original_ok,_=runtime.safe_decision(original[0],original[1],original[3]);chosen=original
  if not original_ok:
   weight=float(scope_policy['weight']);temperature=float(scope_policy['temperature']);fused=[]
   for index in range(3):
    a=np.asarray(baseline[index],dtype=np.float64);b=np.asarray(expert[index],dtype=np.float64)/temperature;shape=a.shape
    if index==2:a,b=a.reshape(-1,6,3),b.reshape(-1,6,3)
    a=a-np.max(a,axis=-1,keepdims=True);b=b-np.max(b,axis=-1,keepdims=True);a=np.exp(a);a/=a.sum(axis=-1,keepdims=True);b=np.exp(b);b/=b.sum(axis=-1,keepdims=True)
    p=np.exp((1-weight)*np.log(np.maximum(a,1e-12))+weight*np.log(np.maximum(b,1e-12)));p/=p.sum(axis=-1,keepdims=True)
    fused.append(np.log(np.maximum(p,1e-12)).reshape(shape).astype(np.float32))
   chosen=runtime.classify(_Outputs(fused+baseline[3:],info),labels,waveform,policy=policy)
  accepted,_=runtime.safe_decision(chosen[0],chosen[1],chosen[3])
  if not accepted:return chosen
  index=labels['intents'].index(chosen[0]);thresholds=scope_policy['thresholds'] if original_ok else scope_policy['rescue_thresholds']
  scope=float(expert[4][0,index])
  if scope_mix:scope+=scope_mix*float(old_scope[0,index])
  if scope>=thresholds[index]:return chosen
  rejected_info=dict(chosen[3],rejected=True,slot=None,scope_verifier_rejected=True)
  return 'REJECT',float(chosen[2][-1]),chosen[2],rejected_info

 def classify_audio(self,labels,waveform,verbose=False,policy=None):
  feed,info=self.prepare_feed(waveform,verbose=verbose);baseline=LongWindowSession.run(self,None,feed)
  original=runtime.classify(_Outputs(baseline,info),labels,waveform,policy=policy)
  if info.get('span') is None:self.last_original=original;return original
  expert=self.expert.run(None,{'mel':feed['long']})
  prior=self._stage(labels,waveform,info,policy,baseline,original,expert,self.scope_policy);self.last_original=prior
  extra=self.extra.run(None,{'mel':feed['long']})
  candidate=self._stage(labels,waveform,info,policy,baseline,prior,extra,self.extra_policy,float(self.extra_policy.get('scope_mix',0.)),expert[4])
  quality=self.extra_policy.get('quality_thresholds_db')
  if quality is not None:
   prior_ok,_=runtime.safe_decision(prior[0],prior[1],prior[3]);candidate_ok,_=runtime.safe_decision(candidate[0],candidate[1],candidate[3])
   if prior_ok or candidate_ok:
    intent=prior[0] if prior_ok else candidate[0];index=labels['intents'].index(intent)
    x=np.asarray(waveform,dtype=np.float64);x=np.pad(x,(0,(-len(x))%320))
    rms=np.sqrt(np.mean(x.reshape(-1,320)**2,axis=1)+1e-12);low,high=np.percentile(rms,[20,95]);contrast=20*np.log10(high/low)
    if contrast<float(quality[index]):return prior
  return candidate
