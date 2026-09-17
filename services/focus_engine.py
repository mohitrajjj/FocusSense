from collections import deque
import numpy as np

class FocusEngine:
    """Personalized, smoothed focus scoring from interpretable signals."""
    def __init__(self, window=12, threshold=55):
        self.window=deque(maxlen=window); self.threshold=threshold; self.baseline=None; self.state='Unknown'
        self.last_score=0.0
    def calibrate(self, samples):
        arr=np.asarray(samples,dtype=float)
        if len(arr): self.baseline=np.median(arr,axis=0)
    def _normalize(self,x):
        x=np.asarray(x,dtype=float); return np.clip(x,0,1)
    def update(self, signals):
        vals={k:float(signals.get(k,0)) for k in ('gaze','head_pose','eye_state','face_presence')}
        raw=.35*vals['gaze']+.25*vals['head_pose']+.20*vals['eye_state']+.20*vals['face_presence']
        self.window.append(raw); smooth=float(np.mean(self.window)); self.last_score=round(smooth*100,1)
        if vals['face_presence'] < .35: self.state='Away'
        elif vals['eye_state'] < .25: self.state='Drowsy'
        elif smooth >= self.threshold/100: self.state='Focused'
        else: self.state='Distracted'
        confidence=round(min(1.0, .55 + .45*abs(smooth-.5)*2),2)
        return self.last_score,self.state,confidence
