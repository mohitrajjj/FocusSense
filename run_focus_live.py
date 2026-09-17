import argparse,time,requests,cv2,mediapipe as mp,numpy as np
from services.cv_features import extract_signals
from services.focus_engine import FocusEngine
from services.model import OptionalModel
from feature_extractor import extract_features

parser=argparse.ArgumentParser(); parser.add_argument('--cam',type=int,default=0); parser.add_argument('--student-id',default='S1'); parser.add_argument('--name',default='Student 1'); parser.add_argument('--server',default='http://127.0.0.1:5000'); args=parser.parse_args()
session_id=None
try: session_id=requests.post(args.server+'/api/session/start',json={'student_id':args.student_id,'name':args.name},timeout=1).json().get('session_id')
except Exception as e: print('Dashboard unavailable; running camera locally:',e)
engine=FocusEngine(); model=OptionalModel(); cap=cv2.VideoCapture(args.cam); cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280); cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
mp_face=mp.solutions.face_mesh; last_send=0; last_state='Unknown'; event_start=None
with mp_face.FaceMesh(max_num_faces=1,refine_landmarks=True,min_detection_confidence=.6,min_tracking_confidence=.6) as mesh:
    while cap.isOpened():
        ok,frame=cap.read()
        if not ok: break
        frame=cv2.flip(frame,1); rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB); res=mesh.process(rgb)
        face=res.multi_face_landmarks[0] if res.multi_face_landmarks else None
        signals=extract_signals(face)
        score,state,conf=engine.update(signals)
        ml_prob=None
        if face: ml_prob=model.update(extract_features(face.landmark))
        if ml_prob is not None:
            score=round(.65*score+.35*ml_prob*100,1)
            if state not in ('Away','Drowsy'): state='Focused' if score>=55 else 'Distracted'
        if state!=last_state and state!='Unknown':
            if event_start:
                try: requests.post(args.server+'/api/event',json={'student_id':args.student_id,'session_id':session_id,'event_type':last_state,'duration':time.time()-event_start},timeout=.3)
                except Exception: pass
            event_start=time.time(); last_state=state
        if time.time()-last_send>=1:
            try: requests.post(args.server+'/api/update',json={'student_id':args.student_id,'name':args.name,'session_id':session_id,'score':score,'state':state,'signals':signals},timeout=.3)
            except Exception: pass
            last_send=time.time()
        cv2.rectangle(frame,(20,20),(470,170),(10,15,25),-1)
        cv2.putText(frame,f'FocusSense   {score:.0f}%',(35,58),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
        cv2.putText(frame,state,(35,95),cv2.FONT_HERSHEY_SIMPLEX,.8,(80,220,130) if state=='Focused' else (60,180,255),2)
        cv2.putText(frame,f'Confidence {conf:.0%}   ML: {"ON" if ml_prob is not None else "fallback"}',(35,128),cv2.FONT_HERSHEY_SIMPLEX,.52,(200,200,200),1)
        cv2.putText(frame,'Q / ESC to stop',(35,153),cv2.FONT_HERSHEY_SIMPLEX,.45,(130,145,160),1)
        cv2.imshow('FocusSense — Live',frame)
        if cv2.waitKey(1)&0xff in (27,ord('q')): break
cap.release(); cv2.destroyAllWindows()
if session_id:
    try: requests.post(args.server+'/api/session/end',json={'session_id':session_id},timeout=1)
    except Exception: pass
