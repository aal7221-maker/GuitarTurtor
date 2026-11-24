"""Generates colab_finetune.ipynb (run: python mk_nb.py). Data: tools/dataset/package.py -> GuitarTurtor_train.zip."""
import json, os
C = []
def md(s): C.append({"cell_type": "markdown", "metadata": {}, "source": s.strip().splitlines(True)})
def code(s): C.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip().splitlines(True)})

md(r"""# GuitarTurtor - fine-tune on GuitarSet + lesson-video dataset -> priority songs (Choo Lo, Locked Away)
Data = `GuitarTurtor/train/` from `tools/dataset/package.py` (manifest.jsonl + labels/ + optional audio/ + guitarset/). See `tools/dataset/COLAB.md`.
Recipe (both models): **(1) pretrain** on GuitarSet + lesson videos (train split) -> **(2) priority fine-tune** on Choo Lo + Locked Away (first 60%, weight x20, lesson+GuitarSet replay) -> **select** by held-out = last 40% of each priority song + held-out lesson videos (whole videos never seen in training) -> **export**.
Part A: string/fret chooser (labels only, minutes on CPU). Part B: audio -> pitch/onset model (needs the zip built with `--with-audio`; GPU).
Labels: notes with conf < 0.7 or any sanity flag are excluded. Audio is private research data: keep the Drive folder private, never redistribute.""")
code(r"""!pip -q install librosa onnx 2>/dev/null; nvidia-smi -L || true""")
code(r"""import os, json, zipfile, hashlib, glob, subprocess
from google.colab import drive; drive.mount('/content/drive')
D='/content/drive/MyDrive'
if os.path.exists(f'{D}/GuitarTurtor_train.zip'):          # uploaded zip: unzip to fast local disk
    zipfile.ZipFile(f'{D}/GuitarTurtor_train.zip').extractall('/content/GT'); ROOT='/content/GT/GuitarTurtor/train'
else:
    ROOT=f'{D}/GuitarTurtor/train'                           # or an already-unzipped folder in Drive
OUT=f'{D}/GuitarTurtor/out'; os.makedirs(OUT,exist_ok=True)
GS=f'{ROOT}/guitarset'; os.makedirs(GS,exist_ok=True)
if not os.path.exists(f'{GS}/annotation.zip'):
    subprocess.run(['wget','-q','-O',f'{GS}/annotation.zip','https://zenodo.org/records/3371780/files/annotation.zip'],check=True)
print(os.listdir(ROOT))""")
code(r"""MAN=[json.loads(l) for l in open(f'{ROOT}/manifest.jsonl',encoding='utf-8')]
MAN=[m for m in MAN if m['status']=='pass' and m.get('label')]
PRI=[m for m in MAN if m.get('priority')]; LES=[m for m in MAN if not m.get('priority')]
# held-out lesson videos: deterministic ~15% (at least 3), picked by hash so it is stable across runs
order=sorted(LES,key=lambda m:hashlib.md5(m['id'].encode()).hexdigest())
nh=max(3,round(.15*len(LES))) if len(LES)>=6 else 0
LES_HO=order[:nh]; LES_TR=order[nh:]
print(f'{len(PRI)} priority, {len(LES_TR)} lesson train, {len(LES_HO)} lesson held-out:',[m['id'] for m in LES_HO])
STD=[64,59,55,50,45,40]   # string 0 = high e
NOTE={'C':0,'D':2,'E':4,'F':5,'G':7,'A':9,'B':11}
def midi(s):
    n=NOTE[s[0].upper()];i=1
    if s[i] in '#b': n+=1 if s[i]=='#' else -1; i+=1
    return 12*(int(s[i:])+1)+n
def open_strings(m):      # sounding pitch of each open string incl. capo
    t=[midi(x) for x in m.get('tuning','E4,B3,G3,D3,A2,E2').split(',')]
    if t[0]<t[-1]: t=t[::-1]
    return [x+int(m.get('capo',0)) for x in t]
def load_item(m,lo=0.,hi=1.,trusted=True):
    # -> sorted [(t, written_pitch, string, fret, sounding_pitch)] for notes in [lo,hi] fraction of the labelled span
    d=json.load(open(f"{ROOT}/{m['label']}"))['notes']
    if m.get('until'): d=[n for n in d if n['t']<m['until']]
    if trusted: d=[n for n in d if n.get('conf',1)>=.7 and not n.get('flags')]
    if not d: return []
    ts=[n['t'] for n in d]; a0,a1=min(ts),max(ts); T0=a0+lo*(a1-a0); T1=a0+hi*(a1-a0)+(1e-6 if hi<1 else 1)
    O=open_strings(m)
    return sorted((n['t'],STD[s]+f,s,f,O[s]+f) for n in d if T0<=n['t']<T1 for s,f in n['n'] if 0<=s<6)
print({m['id']:len(load_item(m)) for m in PRI}, 'lesson notes', sum(len(load_item(m)) for m in LES))""")

md("## Part A - string/fret chooser (pitch + recent hand history -> string)")
code(r"""import numpy as np, torch, torch.nn as nn
dev='cuda' if torch.cuda.is_available() else 'cpu'
def cands(p): return [(s,p-STD[s]) for s in range(6) if 0<=p-STD[s]<=22]
def load_gs(players):    # GuitarSet note_midi per string (data_source 0 = low E), standard tuning
    z=zipfile.ZipFile(f'{GS}/annotation.zip'); res=[]
    for nm in z.namelist():
        if not nm.endswith('.jams') or os.path.basename(nm)[:2] not in players: continue
        d=json.load(z.open(nm)); ns=[]
        for a in d['annotations']:
            if a['namespace']!='note_midi': continue
            s=5-int(a['annotation_metadata']['data_source'])
            for x in a['data']:
                p=int(round(x['value'])); f=p-STD[s]
                if 0<=f<=22: ns.append((x['time'],p,s,f,p))
        res.append(sorted(ns))
    return res
H=4
def seq_to_samples(seq):
    X=[];M=[];Y=[];hist=[]
    for t,p,s,f,_ in seq:
        h=hist[-H:]; h=[(0,0,60,-9)]*(H-len(h))+h
        X.append([p/88]+[v for hs,hf,hp,ht in h for v in (hs/5,hf/22,hp/88,min(t-ht,2.0)/2)])
        m=np.zeros(6)
        for cs,cf in cands(p): m[cs]=1
        m[s]=1; M.append(m); Y.append(s); hist.append((s,f,p,t))
    return np.array(X,'f'),np.array(M,'f'),np.array(Y,'int64')
class Net(nn.Module):
    def __init__(s): super().__init__(); s.f=nn.Sequential(nn.Linear(1+4*H,128),nn.ReLU(),nn.Dropout(.2),nn.Linear(128,128),nn.ReLU(),nn.Dropout(.2),nn.Linear(128,6))
    def forward(s,x,m): return s.f(x).masked_fill(m==0,-1e9)
def tens(seqs,w):
    parts=[seq_to_samples(q) for q in seqs if len(q)>1]
    if not parts: return None
    X=np.vstack([p[0] for p in parts]);M=np.vstack([p[1] for p in parts]);Y=np.concatenate([p[2] for p in parts])
    return [torch.tensor(a).to(dev) for a in (X,M,Y)]+[torch.full((len(Y),),float(w)).to(dev)]
def acc(net,T):
    net.eval()
    with torch.no_grad(): return (net(T[0],T[1]).argmax(1)==T[2]).float().mean().item()
def train(net,parts,score,epochs=60,lr=2e-3,patience=8):
    # score(net) -> selection metric (higher = better); keeps the best epoch
    parts=[p for p in parts if p is not None]
    X,M,Y,W=[torch.cat([p[i] for p in parts]) for i in range(4)]
    opt=torch.optim.AdamW(net.parameters(),lr=lr,weight_decay=1e-2); best=(-1,None); bad=0
    for ep in range(epochs):
        net.train(); perm=torch.randperm(len(Y),device=dev)
        for i in range(0,len(Y),512):
            b=perm[i:i+512]; l=(nn.functional.cross_entropy(net(X[b],M[b]),Y[b],reduction='none')*W[b]).sum()/W[b].sum()
            opt.zero_grad(); l.backward(); opt.step()
        a=score(net)
        if a>best[0]: best=(a,{k:v.clone() for k,v in net.state_dict().items()}); bad=0
        else:
            bad+=1
            if bad>=patience: break
    net.load_state_dict(best[1]); return best[0]""")
code(r"""gs_tr=load_gs([f'0{i}' for i in range(4)]); gs_va=load_gs(['04']); gs_te=load_gs(['05'])
G_tr,G_va,G_te=tens(gs_tr,1),tens(gs_va,1),tens(gs_te,1)
L_tr=tens([load_item(m) for m in LES_TR],1) if LES_TR else None                 # lesson videos (train split)
L_ho=tens([load_item(m) for m in LES_HO],1) if LES_HO else None
P_tr=tens([load_item(m,0,.6) for m in PRI],20)              # priority songs: first 60%, x20
P_ho={m['id']:tens([load_item(m,.6,1)],1) for m in PRI}     # priority held-out: last 40% per song
def heldout(net):   # selection metric: mean of each priority song's held-out acc and held-out lessons
    v=[acc(net,T) for T in P_ho.values()]+([acc(net,L_ho)] if L_ho else [])
    return float(np.mean(v))
def report(net):
    r={k:round(acc(net,T),3) for k,T in P_ho.items()}
    if L_ho: r['lessons_heldout']=round(acc(net,L_ho),3)
    r['guitarset_test']=round(acc(net,G_te),3); r['select']=round(heldout(net),3); return r
net=Net().to(dev)
train(net,[G_tr,L_tr],lambda n:.5*acc(n,G_va)+.5*(acc(n,L_ho) if L_ho else acc(n,G_va)),epochs=40)
torch.save(net.state_dict(),'/content/chooser_pre.pt'); resA={'pretrain':report(net)}; print('pretrain',resA['pretrain'])
train(net,[P_tr,L_tr,G_tr],heldout,epochs=40,lr=5e-4)       # priority fine-tune with replay
resA['finetune']=report(net); print('fine-tune',resA['finetune'])""")
code(r"""# leave-one-priority-song-out: fine-tune on the other song only, test on the whole held song
loso={}
for held in PRI:
    others=[m for m in PRI if m is not held]
    if not others: continue
    n2=Net().to(dev); n2.load_state_dict(torch.load('/content/chooser_pre.pt'))
    V=tens([load_item(held)],1)
    train(n2,[tens([load_item(m) for m in others],20),L_tr,G_tr],lambda n:acc(n,V),epochs=30,lr=5e-4); loso[held['id']]=round(acc(n2,V),3)
resA['loso']=loso; print('LOSO (optimistic: selected on the test song)',loso)
torch.save(net.state_dict(),f'{OUT}/string_chooser.pt')
torch.onnx.export(net.cpu(),(torch.zeros(1,1+4*H),torch.ones(1,6)),f'{OUT}/string_chooser.onnx',input_names=['x','mask'],output_names=['logits'],dynamic_axes={'x':{0:'n'},'mask':{0:'n'},'logits':{0:'n'}})
net.to(dev); json.dump(resA,open(f'{OUT}/results_chooser.json','w'),indent=1); print('saved',OUT)""")

md(r"""## Part B - audio -> pitch (onset + frame) model
Needs audio: zip built with `--with-audio` (lesson + priority audio) and GuitarSet mono-mic audio (downloaded from Zenodo if missing, 656 MB).
CQT (22.05 kHz, hop 512 = 23 ms, 24 bins/oct from C2, 5 oct) -> small CRNN -> onset + frame logits for MIDI 40..88 (sounding pitch, tuning/capo from the manifest).
Lesson labels cover only the passes the tab reader aligned, so loss is masked to frames within 1 s of a trusted label (talking / backing-only parts are ignored) and zeroed in the manifest's `exclude` spans (pages dropped by the quality gate).""")
code(r"""import librosa
SR,HOP,P0,NP=22050,512,40,49
FPS=SR/HOP
def cqt(path):
    c=f'/content/feat/{hashlib.md5(path.encode()).hexdigest()}.npy'
    if os.path.exists(c): return np.load(c)
    y=np.frombuffer(subprocess.run([os.environ.get('FFMPEG','ffmpeg'),'-v','error','-i',path,'-ac','1','-ar',str(SR),'-f','f32le','-'],capture_output=True,check=True).stdout,np.float32)
    X=np.log1p(10*np.abs(librosa.cqt(y,sr=SR,hop_length=HOP,fmin=librosa.note_to_hz('C2'),n_bins=120,bins_per_octave=24))).T.astype('f2')
    os.makedirs('/content/feat',exist_ok=True); np.save(c,X); return X
def rolls(notes,n,until=None):
    # notes: [(t, sounding_pitch)] -> onset/frame targets + loss mask (n frames)
    on=np.zeros((n,NP),'f'); fr=np.zeros((n,NP),'f'); mk=np.zeros(n,'f')
    notes=sorted(notes); ts=[t for t,_ in notes]
    for i,(t,p) in enumerate(notes):
        k=p-P0
        if not 0<=k<NP: continue
        a=int(round(t*FPS)); nxt=next((u for u in ts[i+1:] if u>t+.03),t+1.)
        b=int(round(min(nxt,t+1.)*FPS))
        if a>=n: continue
        on[max(0,a-1):a+2,k]=1; fr[a:max(a+1,b),k]=1
        mk[max(0,int((t-1)*FPS)):int((t+1)*FPS)]=1
    if until: mk[int(until*FPS):]=0
    return on,fr,mk
def item_clip(m,lo=0.,hi=1.):
    X=cqt(f"{ROOT}/{m['audio']}"); notes=[(t,sp) for t,_,_,_,sp in load_item(m,lo,hi)]
    on,fr,mk=rolls(notes,len(X),m.get('until'))
    for a,b in m.get('exclude') or []: mk[max(0,int(a*FPS)):int(b*FPS)+1]=0   # pages dropped as misread: no labels there
    if lo>0 or hi<1:       # keep only the [lo,hi] part of the labelled span
        tt=[t for t,_ in notes]; a,b=int((min(tt)-.5)*FPS),int((max(tt)+.5)*FPS); mk[:max(0,a)]=0; mk[b:]=0
    return X,on,fr,mk
def gs_clips(players):
    if not os.path.exists(f'{GS}/audio_mono-mic.zip'):
        subprocess.run(['wget','-q','-O',f'{GS}/audio_mono-mic.zip','https://zenodo.org/records/3371780/files/audio_mono-mic.zip'],check=True)
    za=zipfile.ZipFile(f'{GS}/audio_mono-mic.zip'); za.extractall('/content/gs_audio') if not os.path.exists('/content/gs_audio') else None
    ann={os.path.basename(n)[:-5]:n for n in zipfile.ZipFile(f'{GS}/annotation.zip').namelist() if n.endswith('.jams')}
    out=[]
    for w in sorted(glob.glob('/content/gs_audio/**/*_mic.wav',recursive=True)):
        k=os.path.basename(w).replace('_mic.wav','')
        if k[:2] not in players or k not in ann: continue
        d=json.load(zipfile.ZipFile(f'{GS}/annotation.zip').open(ann[k]))
        notes=[(x['time'],int(round(x['value']))) for a in d['annotations'] if a['namespace']=='note_midi' for x in a['data']]
        X=cqt(w); on,fr,mk=rolls(notes,len(X)); mk[:]=1; out.append((X,on,fr,mk))
    return out
have_audio=all(m.get('audio') for m in MAN)
print('audio in zip:',have_audio)""")
code(r"""class CRNN(nn.Module):
    def __init__(s):
        super().__init__()
        s.c=nn.Sequential(nn.Conv2d(1,16,3,padding=1),nn.BatchNorm2d(16),nn.ReLU(),nn.Conv2d(16,16,3,padding=1),nn.BatchNorm2d(16),nn.ReLU(),nn.MaxPool2d((1,2)),
                          nn.Conv2d(16,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(),nn.MaxPool2d((1,2)),nn.Dropout(.2))
        s.r=nn.GRU(32*30,192,batch_first=True,bidirectional=True); s.on=nn.Linear(384,NP); s.fr=nn.Linear(384+NP,NP)
    def forward(s,x):               # x: B,T,120
        h=s.c(x[:,None]); B,C,T,F=h.shape; h=h.permute(0,2,1,3).reshape(B,T,C*F); h,_=s.r(h)
        o=s.on(h); return o,s.fr(torch.cat([h,torch.sigmoid(o)],-1))
SEG=256   # ~6 s windows
def batches(clips,weights,n,bs=16):
    L=[max(1,len(c[0])-SEG) for c in clips]; p=np.array([w*l for w,l in zip(weights,L)],float); p/=p.sum()
    for _ in range(n):
        xs,os_,fs,ms,ws=[],[],[],[],[]
        for i in np.random.choice(len(clips),bs,p=p):
            X,on,fr,mk=clips[i]; a=np.random.randint(0,max(1,len(X)-SEG)); sl=slice(a,a+SEG)
            pad=lambda z:np.pad(z[sl],[(0,SEG-len(z[sl]))]+[(0,0)]*(z.ndim-1))
            xs.append(pad(X).astype('f'));os_.append(pad(on));fs.append(pad(fr));ms.append(pad(mk));ws.append(weights[i])
        yield [torch.tensor(np.stack(z)).to(dev) for z in (xs,os_,fs,ms)]+[torch.tensor(ws,dtype=torch.float32).to(dev)]
def note_f1(model,clips,thr=.5,tol=.05):
    # note-level onset F1 (pitch exact, onset within 50 ms) on masked frames
    model.eval(); tp=fp=fn=0
    with torch.no_grad():
        for X,on,fr,mk in clips:
            o,_=model(torch.tensor(X[None].astype('f')).to(dev)); o=torch.sigmoid(o[0]).cpu().numpy()
            pk=(o>thr)&(o>=np.roll(o,1,0))&(o>=np.roll(o,-1,0))
            pred=[(t/FPS,k) for t,k in zip(*np.nonzero(pk)) if mk[t]]; ref=[(t/FPS,k) for t,k in zip(*np.nonzero(on)) if mk[t] and (t==0 or not on[t-1,k])]
            used=set()
            for t,k in ref:
                j=next((i for i,(u,q) in enumerate(pred) if i not in used and q==k and abs(u-t-1/FPS)<=tol+1/FPS),None)
                if j is None: fn+=1
                else: used.add(j); tp+=1
            fp+=len(pred)-len(used)
    return 2*tp/max(1,2*tp+fp+fn)
def fit(model,clips,weights,score,steps=4000,lr=1e-3,every=250,patience=6):
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-3); best=(-1,None); bad=0; bce=nn.functional.binary_cross_entropy_with_logits
    for i,(x,on,fr,mk,w) in enumerate(batches(clips,weights,steps)):
        model.train(); o,f=model(x); m=(mk*w[:,None])[...,None]
        l=((bce(o,on,reduction='none',pos_weight=torch.tensor(5.,device=dev))+bce(f,fr,reduction='none'))*m).sum()/(m.sum()*NP+1e-6)
        opt.zero_grad(); l.backward(); opt.step()
        if (i+1)%every==0:
            s=score(model); print(i+1,round(l.item(),4),'select',round(s,3))
            if s>best[0]: best=(s,{k:v.clone() for k,v in model.state_dict().items()}); bad=0
            else:
                bad+=1
                if bad>=patience: break
    model.load_state_dict(best[1]); return best[0]""")
code(r"""if have_audio:
    GSA_tr=gs_clips([f'0{i}' for i in range(5)]); GSA_te=gs_clips(['05'])
    LA_tr=[item_clip(m) for m in LES_TR]; LA_ho=[item_clip(m) for m in LES_HO]
    PA_tr=[item_clip(m,0,.6) for m in PRI]; PA_ho={m['id']:[item_clip(m,.6,1)] for m in PRI}
    def selB(mod): return float(np.mean([note_f1(mod,c) for c in PA_ho.values()]+([note_f1(mod,LA_ho)] if LA_ho else [])))
    def repB(mod):
        r={k:round(note_f1(mod,c),3) for k,c in PA_ho.items()}; r['lessons_heldout']=round(note_f1(mod,LA_ho),3) if LA_ho else None
        r['guitarset_test']=round(note_f1(mod,GSA_te),3); return r
    model=CRNN().to(dev)
    fit(model,GSA_tr+LA_tr,[1]*len(GSA_tr)+[1]*len(LA_tr),lambda m:.5*note_f1(m,GSA_te[:4])+.5*(note_f1(m,LA_ho) if LA_ho else 0),steps=8000)
    torch.save(model.state_dict(),'/content/pitch_pre.pt'); resB={'pretrain':repB(model)}; print(resB)
    # priority fine-tune: priority first-60% at weight 20 (per window), replay lessons + GuitarSet
    fit(model,PA_tr+LA_tr+GSA_tr,[20]*len(PA_tr)+[1]*len(LA_tr)+[.5]*len(GSA_tr),selB,steps=3000,lr=3e-4,every=150)
    resB['finetune']=repB(model); print(resB)
    torch.save(model.state_dict(),f'{OUT}/pitch_crnn.pt')
    torch.onnx.export(model.cpu(),torch.zeros(1,SEG,120),f'{OUT}/pitch_crnn.onnx',input_names=['cqt'],output_names=['onset','frame'],dynamic_axes={'cqt':{1:'T'},'onset':{1:'T'},'frame':{1:'T'}},opset_version=13)
    json.dump(resB,open(f'{OUT}/results_pitch.json','w'),indent=1); print('saved',OUT)
else:
    print('no audio in the zip: rebuild locally with  python -I tools/dataset/package.py --with-audio --guitarset')""")
md(r"""## Export / use
`MyDrive/GuitarTurtor/out/`: `string_chooser.pt|.onnx` (+results_chooser.json), `pitch_crnn.pt|.onnx` (+results_pitch.json). Download them into `tools/train/out/` locally.
Read the numbers honestly: priority held-out = last 40% of 2 songs (~70 + ~270 notes) -> differences of a few points are noise. If fine-tune does not beat pretrain on held-out, keep the pretrain weights (`/content/*_pre.pt`).
GuitarSet: Xi et al., ISMIR 2018, CC BY 4.0.""")
here = os.path.dirname(os.path.abspath(__file__))
json.dump({"cells": C, "metadata": {"accelerator": "GPU", "kernelspec": {"name": "python3", "display_name": "Python 3"}},
           "nbformat": 4, "nbformat_minor": 5}, open(os.path.join(here, "colab_finetune.ipynb"), "w"), indent=1)
print("wrote colab_finetune.ipynb")
