from __future__ import annotations
import numpy as np
import pennylane as qml
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import torch
from torch import nn


def _preprocess(train_x, other_x, n_components, seed):
 scaler=StandardScaler().fit(train_x)
 xt=scaler.transform(train_x); xo=scaler.transform(other_x)
 n_components=min(int(n_components),xt.shape[1],max(1,xt.shape[0]-1))
 pca=PCA(n_components=n_components,svd_solver='full').fit(xt)
 zt=pca.transform(xt); zo=pca.transform(xo)
 zscaler=StandardScaler().fit(zt)
 zt=zscaler.transform(zt); zo=zscaler.transform(zo)
 return zt,zo,{'input_scaler':'training-only StandardScaler','pca_components':n_components,'pca_explained_variance_ratio_sum':float(pca.explained_variance_ratio_.sum()),'post_pca_scaler':'training-only StandardScaler'}

def _angles(z):
 return np.pi*np.tanh(np.asarray(z,dtype=float)/3.0)

def _state_function(nq):
 dev=qml.device('default.qubit',wires=nq,shots=None)
 @qml.qnode(dev,interface=None,diff_method=None)
 def circuit(x):
  for w in range(nq): qml.Hadamard(wires=w)
  qml.AngleEmbedding(x,wires=range(nq),rotation='Y')
  for w in range(nq): qml.RZ(0.5*x[w],wires=w)
  for w in range(nq): qml.CNOT(wires=[w,(w+1)%nq])
  return qml.state()
 return circuit

def _states(z,nq):
 circuit=_state_function(nq)
 return np.vstack([np.asarray(circuit(x),dtype=np.complex128) for x in _angles(z)])

def _nystrom(k_nm,k_mm,eps=1e-8):
 val,vec=np.linalg.eigh((k_mm+k_mm.T)*0.5)
 val=np.maximum(val,eps)
 inv=(vec*(1/np.sqrt(val)))@vec.T
 return k_nm@inv,inv

def fit_quantum_kernel(train_x,train_y,predict_x,n_qubits,landmarks=128,alpha=1e-3,seed=0):
 zt,zp,meta=_preprocess(train_x,predict_x,n_qubits,seed)
 n=len(zt); m=min(int(landmarks),n)
 ids=np.unique(np.linspace(0,n-1,m,dtype=int))
 Utr=_states(zt,n_qubits); Uland=Utr[ids]; Up=_states(zp,n_qubits)
 Kmm=np.abs(Uland@Uland.conj().T)**2
 Ktr=np.abs(Utr@Uland.conj().T)**2
 Kp=np.abs(Up@Uland.conj().T)**2
 Phi,inv=_nystrom(Ktr,Kmm)
 Phip=Kp@inv
 reg=Ridge(alpha=alpha).fit(Phi,np.asarray(train_y,float))
 pred=reg.predict(Phip)
 return pred,meta|{'method':'fidelity quantum kernel ridge / Nyström','n_qubits':n_qubits,'landmarks':len(ids),'circuit':'Hadamard, angle Y embedding, RZ(x/2), cyclic CNOT ring','shots':None,'backend':'default.qubit statevector','alpha':alpha,'landmark_rule':'equally spaced training-row indices','seed':seed}


def fit_rbf_nystrom(train_x,train_y,predict_x,n_components,landmarks=128,alpha=1e-3,gamma=None,seed=0):
 zt,zp,meta=_preprocess(train_x,predict_x,n_components,seed)
 if gamma is None: gamma=1.0/zt.shape[1]
 n=len(zt); m=min(int(landmarks),n); ids=np.unique(np.linspace(0,n-1,m,dtype=int)); xm=zt[ids]
 def ker(a,b):
  d=((a*a).sum(1)[:,None]+(b*b).sum(1)[None,:]-2*a@b.T).clip(0)
  return np.exp(-gamma*d)
 Kmm=ker(xm,xm); Ktr=ker(zt,xm); Kp=ker(zp,xm)
 Phi,inv=_nystrom(Ktr,Kmm); Phip=Kp@inv
 reg=Ridge(alpha=alpha).fit(Phi,np.asarray(train_y,float))
 return reg.predict(Phip),meta|{'method':'matched RBF kernel ridge / Nyström','input_dimension':int(n_components),'landmarks':len(ids),'landmark_rule':'same equally spaced training-row indices','gamma':float(gamma),'alpha':alpha,'seed':seed}


def _vqc_layer(nq,seed):
 dev=qml.device('default.qubit',wires=nq,shots=None)
 @qml.qnode(dev,interface='torch',diff_method='backprop')
 def qnode(inputs,weights):
  qml.AngleEmbedding(inputs,wires=range(nq),rotation='Y')
  qml.StronglyEntanglingLayers(weights,wires=range(nq))
  return [qml.expval(qml.PauliZ(w)) for w in range(nq)]
 shapes={'weights':(2,nq,3)}
 torch.manual_seed(seed)
 return qml.qnn.TorchLayer(qnode,shapes)

class _QNN(nn.Module):
 def __init__(self,nq,seed,hybrid=False,input_dim=None):
  super().__init__(); self.nq=nq; self.hybrid=hybrid
  if hybrid:
   self.encoder=nn.Linear(input_dim,nq)
   self.q_layer=_vqc_layer(nq,seed)
   self.head=nn.Linear(nq,1)
  else:
   self.q_layer=_vqc_layer(nq,seed)
   self.head=nn.Linear(nq,1)
 def forward(self,x):
  if self.hybrid: x=torch.pi*torch.tanh(self.encoder(x))
  else: x=torch.pi*torch.tanh(x/3.0)
  z=self.q_layer(x)
  return self.head(z).squeeze(-1)

def fit_qnn(train_x,train_y,predict_x,n_qubits,seed,epochs=40,batch_size=64,learning_rate=.01,hybrid=False,encoder_components=16):
 # Fold-only scaling/PCA. Hybrid uses a compact trainable linear encoder from at
 # most 16 training-only principal components into the fixed qubit register.
 components=min(n_qubits if not hybrid else encoder_components,train_x.shape[1],max(1,len(train_x)-1))
 zt,zp,meta=_preprocess(train_x,predict_x,components,seed)
 if hybrid:
  zt,zp,meta2=_preprocess(train_x,predict_x,components,seed)
  meta.update(meta2); input_dim=zt.shape[1]
 else: input_dim=None
 y=np.asarray(train_y,dtype=float); ym=float(y.mean()); ys=float(y.std(ddof=1)) or 1.0
 yt=(y-ym)/ys
 # CPU exact statevector simulation for deterministic fixed-seed training.
 torch.set_num_threads(1); torch.manual_seed(seed); np.random.seed(seed)
 model=_QNN(n_qubits,seed,hybrid,input_dim)
 opt=torch.optim.Adam(model.parameters(),lr=learning_rate)
 loss_fn=nn.MSELoss(); rng=np.random.default_rng(seed)
 xx=torch.tensor(zt,dtype=torch.float32); yy=torch.tensor(yt,dtype=torch.float32)
 model.train()
 for _ in range(int(epochs)):
  order=rng.permutation(len(xx))
  for s in range(0,len(order),batch_size):
   ids=order[s:s+batch_size]; opt.zero_grad(set_to_none=True)
   pred=model(xx[ids]); loss=loss_fn(pred,yy[ids]); loss.backward(); opt.step()
 model.eval()
 with torch.no_grad(): yp=model(torch.tensor(zp,dtype=torch.float32)).cpu().numpy()*ys+ym
 return yp,meta|{'method':'hybrid variational quantum regressor' if hybrid else 'variational quantum regressor','n_qubits':n_qubits,'epochs':int(epochs),'batch_size':int(batch_size),'learning_rate':learning_rate,'seeds':[seed],'circuit':'AngleEmbedding(Y) plus 2 StronglyEntanglingLayers; Pauli-Z vector, trainable linear output','backend':'default.qubit exact statevector','shots':None,'hybrid_encoder':'single linear layer from training-only PCA scores' if hybrid else None,'seed':seed}
