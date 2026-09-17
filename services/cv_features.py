import math, numpy as np

# MediaPipe FaceMesh landmark indices
LEFT_EYE=[33,160,158,133,153,144]
RIGHT_EYE=[362,385,387,263,373,380]
LEFT_IRIS=[474,475,476,477]
RIGHT_IRIS=[469,470,471,472]

def _p(lm,i): return np.array([lm[i].x,lm[i].y,lm[i].z],dtype=float)
def _dist(a,b): return float(np.linalg.norm(a-b))
def _ear(lm,ids):
    a,b,c,d,e,f=[_p(lm,i) for i in ids]
    return (_dist(b,f)+_dist(c,e))/(2*_dist(a,d)+1e-8)
def _iris_ratio(lm, eye, iris):
    a,b=[_p(lm,i) for i in eye]; center=np.mean([_p(lm,i) for i in iris],axis=0)
    width=max(_dist(a,b),1e-6); return float((center[0]-min(a[0],b[0]))/width)

def extract_signals(face_landmarks):
    if face_landmarks is None: return {'gaze':0,'head_pose':0,'eye_state':0,'face_presence':0}
    lm=face_landmarks.landmark
    ear=float((_ear(lm,LEFT_EYE)+_ear(lm,RIGHT_EYE))/2)
    eye_state=float(np.clip((ear-.12)/.18,0,1))
    # Iris position near the center is treated as screen-facing gaze.
    l=_iris_ratio(lm,[33,133],LEFT_IRIS); r=_iris_ratio(lm,[362,263],RIGHT_IRIS)
    gaze=float(np.clip(1-abs(((l+r)/2)-.5)*2.4,0,1))
    # Nose relative to eye midpoint: stable, camera-relative head orientation proxy.
    le=np.mean([_p(lm,i) for i in [33,133]],axis=0); re=np.mean([_p(lm,i) for i in [362,263]],axis=0); nose=_p(lm,1)
    mid=(le+re)/2; span=max(_dist(le,re),1e-6); dx=abs(nose[0]-mid[0])/span; dy=abs(nose[1]-mid[1])/span
    head=float(np.clip(1-(abs(dx-.0)*1.7 + abs(dy-.55)*.9),0,1))
    return {'gaze':gaze,'head_pose':head,'eye_state':eye_state,'face_presence':1.0,'ear':ear,'gaze_left':l,'gaze_right':r}
