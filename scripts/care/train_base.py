"""CARE per-service TCN + metric-anchored alignment + 8-head GAT localizer.

Checkpoint selection uses validation only. Test is evaluated after selection.
No DGL dependency; edge attention is explicitly normalized over source nodes.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class Temporal(nn.Module):
    def __init__(self, channels, hidden):
        super().__init__()
        self.first=nn.Conv1d(channels,hidden,3)
        self.second=nn.Conv1d(hidden,hidden,3,dilation=2)
    def forward(self,x):
        b,n,t,c=x.shape
        x=x.reshape(b*n,t,c).transpose(1,2)
        x=F.relu(self.first(F.pad(x,(2,0))))
        x=F.relu(self.second(F.pad(x,(4,0))))
        return x.mean(-1).reshape(b,n,-1)


class CAREBase(nn.Module):
    def __init__(self,metric_dim,trace_dim,log_dim,hidden=64,heads=8,temperature=.07,diversity=.5,margin=0.,metric_patch_size=0,metric_patch_stride=1,ablation='full'):
        super().__init__()
        if hidden%heads:raise ValueError('Hidden size must divide into heads')
        self.metrics=Temporal(metric_dim,hidden);self.traces=Temporal(trace_dim,hidden)
        self.logs=nn.Linear(log_dim,hidden)
        self.projections=nn.ModuleList([nn.Sequential(nn.Linear(hidden,hidden),nn.ReLU(),nn.Linear(hidden,hidden)) for _ in range(3)])
        self.fuse=nn.Linear(hidden*3,hidden);self.gat=nn.Linear(hidden,hidden,bias=False)
        self.source=nn.Parameter(torch.empty(heads,hidden//heads));self.target=nn.Parameter(torch.empty(heads,hidden//heads))
        nn.init.xavier_uniform_(self.source);nn.init.xavier_uniform_(self.target)
        self.classifier=nn.Sequential(nn.Linear(hidden,hidden),nn.ReLU(),nn.Linear(hidden,1))
        self.heads=heads;self.temperature=temperature;self.diversity=diversity;self.margin=margin
        self.metric_patch_size=metric_patch_size;self.metric_patch_stride=metric_patch_stride
        self.ablation=ablation

    def encode_metrics(self,metrics):
        if not self.metric_patch_size:return self.metrics(metrics)
        if self.metric_patch_size>metrics.shape[2] or self.metric_patch_stride<1:
            raise ValueError('Invalid metric patch size/stride')
        # [batch, service, patch, channel, time] -> independent overlapping
        # temporal patches; mean pool encoded patches back to each service.
        patches=metrics.unfold(2,self.metric_patch_size,self.metric_patch_stride)
        b,n,p,c,t=patches.shape
        patches=patches.permute(0,1,2,4,3).reshape(b,n*p,t,c)
        return self.metrics(patches).reshape(b,n,p,-1).mean(2)

    def forward(self,metrics,traces,logs,adjacency):
        if self.ablation=='no_metrics':metrics=torch.zeros_like(metrics)
        if self.ablation=='no_traces':traces=torch.zeros_like(traces)
        if self.ablation=='no_logs':logs=torch.zeros_like(logs)
        encoded=[self.encode_metrics(metrics),F.relu(self.logs(logs.mean(2))),self.traces(traces)]
        projected=[F.normalize(p(x),dim=-1) for p,x in zip(self.projections,encoded)]
        m,l,t=[x.flatten(0,1) for x in projected]
        target=torch.arange(len(m),device=m.device)
        align=(F.cross_entropy(m@l.T/self.temperature,target)+F.cross_entropy(m@t.T/self.temperature,target))/2
        lr=F.normalize(l-(l*m).sum(-1,keepdim=True)*m,dim=-1)
        tr=F.normalize(t-(t*m).sum(-1,keepdim=True)*m,dim=-1)
        contrast=align+self.diversity*F.relu((lr*tr).sum(-1)-self.margin).square().mean()
        if self.ablation=='standard_cl':
            pairs=[(a,b) for a,b in [(m,l),(m,t),(l,t)]]
            contrast=sum(F.cross_entropy(a@b.T/self.temperature,target)+F.cross_entropy(b@a.T/self.temperature,target) for a,b in pairs)/6
        if self.ablation=='no_cl':contrast=contrast*0
        x=F.relu(self.fuse(torch.cat(projected,dim=-1)))
        if self.ablation=='no_gat':
            attention=torch.eye(x.shape[1],device=x.device)[None].expand(x.shape[0],-1,-1)
            return self.classifier(x).squeeze(-1),contrast,attention
        z=self.gat(x).reshape(*x.shape[:2],self.heads,-1)
        src=(z*self.source).sum(-1);dst=(z*self.target).sum(-1)
        # [batch, source, destination, head], each destination normalizes sources.
        e=F.leaky_relu(src[:,:,None,:]+dst[:,None,:,:],negative_slope=.2)
        attn=torch.softmax(e.masked_fill(~adjacency[None,:,:,None],float('-inf')),dim=1)
        output=torch.einsum('buvh,buhd->bvhd',attn,z).flatten(2)
        logits=self.classifier(F.elu(output)+x).squeeze(-1)
        return logits,contrast,attn.mean(-1)


def ranking_metrics(prob,labels):
    ranks=(np.argsort(-prob,axis=1,kind='stable')==labels[:,None]).argmax(1)+1
    return {**{f'PR@{k}':float((ranks<=k).mean()) for k in [1,3,5]},
            **{f'MAP@{k}':float(np.where(ranks<=k,1/ranks,0).mean()) for k in [3,5]}}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,default=Path('data/care/aiops2022-prepared'))
    p.add_argument('--output',type=Path,default=Path('artifacts/care/base-aiops2022'))
    p.add_argument('--epochs',type=int,default=50);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--batch-size',type=int,default=16);p.add_argument('--lr',type=float,default=.01)
    p.add_argument('--contrast-weight',type=float,default=.1);p.add_argument('--device',default='cuda')
    p.add_argument('--checkpoint-selection',choices=['validation','final'],default='validation')
    p.add_argument('--metric-patch-size',type=int,default=0)
    p.add_argument('--metric-patch-stride',type=int,default=1)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--ablation',choices=['full','no_cl','standard_cl','no_gat','no_metrics','no_traces','no_logs'],default='full')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(args.threads)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    rows=json.loads((args.data/'incidents.json').read_text());meta=json.loads((args.data/'metadata.json').read_text())
    features=np.load(args.data/'features.npz');device=torch.device(args.device)
    tensors=[torch.from_numpy(features[k]).float().to(device) for k in ['metrics','traces','logs']]
    labels=torch.from_numpy(features['labels']).long().to(device)
    services=meta['services'];adj=np.eye(len(services),dtype=bool)
    for row in rows:
        if row['split']!='train':continue
        for edge in row['calls']:
            u,v=services.index(edge['source']),services.index(edge['target']);adj[u,v]=adj[v,u]=True
    adjacency=torch.tensor(adj,device=device)
    idx={s:np.array([i for i,r in enumerate(rows) if r['split']==s],dtype=int) for s in ['train','validation','test']}
    if not len(idx['train']) or not len(idx['test']):raise ValueError('Train and test must be nonempty')
    if args.checkpoint_selection=='validation' and not len(idx['validation']):raise ValueError('Validation selection requires a validation set')
    model_args={'metric_dim':tensors[0].shape[-1],'trace_dim':tensors[1].shape[-1],'log_dim':tensors[2].shape[-1]}
    model_args.update(metric_patch_size=args.metric_patch_size,metric_patch_stride=args.metric_patch_stride,ablation=args.ablation)
    model=CAREBase(**model_args).to(device);optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
    def predict(indices):
        model.eval();probs=[];attns=[]
        with torch.no_grad():
            for chunk in np.array_split(indices,max(1,int(np.ceil(len(indices)/args.batch_size)))):
                out,_,a=model(*(x[chunk] for x in tensors),adjacency)
                probs.append(out.softmax(-1).cpu().numpy());attns.append(a.cpu().numpy())
        return np.concatenate(probs),np.concatenate(attns)
    best=(-1.,-1.);history=[]
    for epoch in range(args.epochs):
        model.train();order=np.random.permutation(idx['train']);losses=[]
        for chunk in np.array_split(order,max(1,int(np.ceil(len(order)/args.batch_size)))):
            logits,contrast,_=model(*(x[chunk] for x in tensors),adjacency)
            loss=F.cross_entropy(logits,labels[chunk])+args.contrast_weight*contrast
            optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
            losses.append(loss.item())
        record={'epoch':epoch+1,'loss':float(np.mean(losses))}
        if args.checkpoint_selection=='validation':
            prob,_=predict(idx['validation']);scores=ranking_metrics(prob,features['labels'][idx['validation']])
            key=(scores['PR@1'],scores['MAP@5']);record['validation']=scores
            save_checkpoint=key>best
            if save_checkpoint:best=key
        else:
            save_checkpoint=epoch+1==args.epochs
        history.append(record)
        print(json.dumps(record),flush=True)
        if save_checkpoint:
            torch.save({'state_dict':model.state_dict(),'model_args':model_args,'adjacency':adj,
                        'epoch':epoch+1,'training_ids':[rows[i]['incident_id'] for i in idx['train']]},args.output/'model.pt')
    checkpoint=torch.load(args.output/'model.pt',map_location=device,weights_only=False)
    model.load_state_dict(checkpoint['state_dict']);prob,attention=predict(np.arange(len(rows)))
    np.savez_compressed(args.output/'predictions.npz',probabilities=prob,attention=attention)
    result={'selected_epoch':checkpoint['epoch'],'test':ranking_metrics(prob[idx['test']],features['labels'][idx['test']]),
            'counts':{k:len(v) for k,v in idx.items()},'protocol':f'incident-level; {args.checkpoint_selection} checkpoint selection; held-out test once',
            'model_args':model_args,'torch_version':torch.__version__,
            'incidents_sha256':hashlib.sha256((args.data/'incidents.json').read_bytes()).hexdigest(),
            'features_sha256':hashlib.sha256((args.data/'features.npz').read_bytes()).hexdigest(),
            'split_manifest_sha256':meta.get('split_manifest_sha256'),
            'limitations':meta['limitations'],'arguments':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}}
    (args.output/'history.json').write_text(json.dumps(history,indent=2)+'\n')
    (args.output/'evaluation.json').write_text(json.dumps(result,indent=2)+'\n')
    cases=[]
    for i in idx['test']:
        order=np.argsort(-prob[i],kind='stable');rank=int(np.where(order==features['labels'][i])[0][0])+1
        cases.append({'incident_id':rows[i]['incident_id'],'root_service':rows[i]['root_service'],
                      'rank':rank,'ranking':[services[j] for j in order],
                      'probabilities':dict(zip(services,map(float,prob[i])))})
    (args.output/'test_cases.json').write_text(json.dumps(cases,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
