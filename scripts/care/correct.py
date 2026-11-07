"""CARE equation-level node/edge correction with auditable conservative gating."""
from collections import defaultdict, deque
import heapq
import math


def path_search(edges,start,end):
    """Directed maximum-product path. Zero-confidence edges are never traversed."""
    graph=defaultdict(list)
    for (u,v),score in edges.items():
        if 0<score<=1:graph[u].append((v,score))
    queue=[(0.,start,[start])];best={start:0.}
    while queue:
        cost,node,path=heapq.heappop(queue)
        if cost>best[node]:continue
        if node==end:return {'nodes':path,'confidence':math.exp(-cost)}
        for nxt,score in graph[node]:
            new=cost-math.log(score)
            if new<best.get(nxt,float('inf')):
                best[nxt]=new;heapq.heappush(queue,(new,nxt,path+[nxt]))
    return None


def correct(base,evidence,facts,proposal,attention=None,weights=(.6,.25,.15),rho=.6,hops=3,
            delta=60.,mu=.5,base_confidence=.8,minimum_margin=.05):
    if any(w<0 for w in weights) or not math.isclose(sum(weights),1.):raise ValueError('Weights must sum to 1')
    if not 0<rho<=1 or not 0<=mu<=1:raise ValueError('Invalid decay/fusion')
    if not base or any(not math.isfinite(x) or x<0 for x in base.values()):raise ValueError('Invalid base scores')
    if sum(base.values())<=0:raise ValueError('Zero base scores')
    base={v:x/sum(base.values()) for v,x in base.items()}
    proposed=proposal.get('causal_scores',{})
    if any(v not in base or not isinstance(x,(float,int)) or not math.isfinite(x) or not 0<=x<=1 for v,x in proposed.items()):
        raise ValueError('Invalid LLM causal scores')
    candidates=list(proposed)
    if not candidates or proposal.get('root_cause') is None:
        return {'scores':base,'changed':False,'reason':'LLM abstained','edges':[]}
    if proposal['root_cause'] not in candidates:raise ValueError('Root proposal outside candidates')
    incoming=defaultdict(set);outgoing=defaultdict(set);propagation=set();attested={}
    for fact in facts:
        u,v=fact['source'],fact['target']
        if u not in base or v not in base:continue
        incoming[v].add(u);outgoing[u].add(v)
        if fact['relation']=='PROPAGATES_TO':
            propagation.add((u,v));attested[(u,v)]=fact['fact_id']
    consistency={};rule_scores={};rule_outcomes={}
    for candidate in candidates:
        distances={candidate:0};queue=deque([candidate])
        while queue:
            node=queue.popleft()
            if distances[node]>=hops:continue
            for nxt in incoming[node]|outgoing[node]:
                if nxt not in distances:distances[nxt]=distances[node]+1;queue.append(nxt)
        consistency[candidate]=sum(rho**d*proposed.get(v,0.) for v,d in distances.items())/sum(rho**d for d in distances.values())
        current=evidence[candidate];up=incoming[candidate];down=outgoing[candidate]
        abnormal=[v for v in down if evidence[v]['anomalous']]
        outcomes={}
        if up:outcomes['upstream_health']=all(not evidence[u]['anomalous'] for u in up)
        if down:outcomes['fan_out']=len(abnormal)>=2
        if current['onset'] is not None and abnormal:
            outcomes['temporal_alignment']=all(evidence[v]['onset'] is not None and current['onset']<=evidence[v]['onset']+delta for v in abnormal)
        rule_outcomes[candidate]=outcomes;rule_scores[candidate]=sum(outcomes.values())/len(outcomes) if outcomes else 0.
    combined={v:weights[0]*base[v]+weights[1]*consistency[v]+weights[2]*rule_scores[v] for v in candidates}
    best=max(combined,key=combined.get);old=max(base,key=base.get)
    other=max((x for v,x in combined.items() if v!=best),default=0.)
    support=any(u==best and evidence[u]['anomalous'] and evidence[v]['anomalous']
                and evidence[u]['onset'] is not None and evidence[v]['onset'] is not None
                and evidence[u]['onset']<=evidence[v]['onset']+delta for u,v in propagation)
    allowed=base[old]<base_confidence and combined[best]-other>=minimum_margin and support
    reason='supported correction' if allowed else 'insufficient confidence or grounded support'
    final=base.copy()
    if allowed:
        # Candidate reranking preserves the base distribution outside the candidate set.
        mass=sum(base[v] for v in candidates);total=sum(combined.values())
        final.update({v:mass*s/total for v,s in combined.items()})
    semantic={}
    rejected=[]
    for edge in proposal.get('propagation_edges',[]):
        u,v=edge.get('source'),edge.get('target');score=edge.get('confidence')
        if (u,v) not in attested or edge.get('fact_id')!=attested[(u,v)]:
            rejected.append({'source':u,'target':v,'reason':'unattested edge or invalid citation'});continue
        if not isinstance(score,(int,float)) or not math.isfinite(score) or not 0<=score<=1:
            rejected.append({'source':u,'target':v,'reason':'invalid confidence'});continue
        semantic[(u,v)]=score
    edge_results=[]
    for u,v in sorted(propagation | set(attention or {})):
        a,b=evidence[u],evidence[v]
        gate=((u,v) in propagation and a['anomalous'] and b['anomalous'] and
              a['onset'] is not None and b['onset'] is not None and a['onset']<=b['onset']+delta)
        alpha=float((attention or {}).get((u,v),0.))
        if not math.isfinite(alpha) or not 0<=alpha<=1:raise ValueError('Invalid attention')
        degree=len(outgoing[u]);eta=(1-a['health'])*degree/(degree+1)
        score=(mu*alpha+(1-mu)*semantic.get((u,v),0.)*eta) if gate else 0.
        edge_results.append({'source':u,'target':v,'gate':bool(gate),'attention':alpha,
                             'semantic':semantic.get((u,v),0.),'eta':eta,'confidence':score,
                             'fact_id':attested.get((u,v))})
    return {'scores':final,'changed':max(final,key=final.get)!=old,'reason':reason,
            'consistency':consistency,'rules':rule_outcomes,'combined_candidate_scores':combined,
            'edges':edge_results,'rejected_proposals':rejected}
