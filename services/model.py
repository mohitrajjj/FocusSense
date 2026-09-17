import os, numpy as np
from collections import deque

class OptionalModel:
    def __init__(self,path='focussense_model.h5',seq_len=30,feature_dim=9):
        self.model=None; self.buffer=deque(maxlen=seq_len)
        if os.path.exists(path):
            try:
                from tensorflow.keras.models import load_model
                self.model=load_model(path,compile=False)
                print('Optional FocusSense model loaded:',path)
            except Exception as e: print('Optional model disabled:',e)
    def update(self,features):
        if self.model is None or features is None: return None
        f=np.asarray(features,dtype=np.float32).reshape(-1)
        if len(f)!=9:return None
        self.buffer.append(f)
        if len(self.buffer)<self.buffer.maxlen:return None
        try:return float(self.model.predict(np.expand_dims(np.asarray(self.buffer),0),verbose=0)[0][0])
        except Exception:return None
