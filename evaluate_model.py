"""Evaluate the saved FocusSense classifier on the .npy dataset.
Prints measured metrics; it never invents results."""
import os,numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_recall_fscore_support,confusion_matrix
from services.model import OptionalModel
from tensorflow.keras.models import load_model
X=[];y=[]
for f in os.listdir('data'):
    if f.endswith('.npy'):
        a=np.load(os.path.join('data',f))
        if a.shape==(1,30,9):X.append(a[0]);y.append(1 if 'focused' in f.lower() else 0)
X=np.asarray(X);y=np.asarray(y)
if len(X)<4: raise SystemExit('Not enough labeled sequences for evaluation.')
_,Xtest,_,ytest=train_test_split(X,y,test_size=.25,random_state=42,stratify=y)
model=load_model('focussense_model.h5',compile=False); p=(model.predict(Xtest,verbose=0).ravel()>=.6).astype(int)
pr,re,f1,_=precision_recall_fscore_support(ytest,p,average='binary',zero_division=0)
print(f'Samples: {len(ytest)}'); print(f'Accuracy: {accuracy_score(ytest,p):.4f}'); print(f'Precision: {pr:.4f}'); print(f'Recall: {re:.4f}'); print(f'F1: {f1:.4f}'); print('Confusion matrix:'); print(confusion_matrix(ytest,p))
