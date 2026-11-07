"""Render an isolated, versioned TrainTicket deployment from pinned upstream YAML."""
import json
from pathlib import Path
import yaml

NAMESPACE = 'care-tt'
ROOT = Path(__file__).resolve().parents[2]


def render():
    source = ROOT / 'external/train-ticket-v0.2.0/deployment/kubernetes-manifests/k8s-with-jaeger'
    docs = [{'apiVersion': 'v1', 'kind': 'Namespace', 'metadata': {
        'name': NAMESPACE, 'labels': {'care.experiment': 'trainticket'}}}]
    images, applications, databases = set(), [], []
    for path in sorted(source.glob('ts-deployment-part*.yml')):
        for doc in yaml.safe_load_all(path.read_text()):
            if not doc:
                continue
            doc['metadata']['namespace'] = NAMESPACE
            if doc['kind'] in ('Deployment', 'StatefulSet'):
                name = doc['metadata']['name']
                doc['spec']['strategy'] = {'type': 'Recreate', 'rollingUpdate': None}
                pod = doc['spec']['template']
                pod['metadata'].setdefault('labels', {})['care.experiment'] = 'trainticket'
                pod['spec'].pop('nodeName', None)
                for c in pod['spec']['containers']:
                    if c['image'] == 'mongo':
                        c['image'] = 'mongo:4.4.29'
                    if c['image'] == 'jaegertracing/all-in-one':
                        c['image'] = 'jaegertracing/all-in-one:1.57.0'
                    c['imagePullPolicy'] = 'IfNotPresent'
                    c['resources'] = {'requests': {'cpu': '25m', 'memory': '128Mi'},
                                      'limits': {'cpu': '1', 'memory': '768Mi'}}
                    c.setdefault('env', []).extend([
                        {'name': 'JAVA_TOOL_OPTIONS', 'value': '-Xms64m -Xmx384m -XX:+UseSerialGC'},
                        {'name': 'JAEGER_SAMPLER_TYPE', 'value': 'const'},
                        {'name': 'JAEGER_SAMPLER_PARAM', 'value': '1'},
                    ])
                    if name == 'jaeger':
                        c['resources']['limits']['memory'] = '2Gi'
                        # Raw traces are exported continuously per incident. Bound
                        # the query cache so a 10-hour campaign does not scan an
                        # ever-growing in-memory trace store on every request.
                        c['env'].append({'name': 'MEMORY_MAX_TRACES', 'value': '5000'})
                    if name.startswith('ts-') and c.get('ports'):
                        port = c['ports'][0]['containerPort']
                        c.pop('livenessProbe', None)
                        c['startupProbe'] = {'tcpSocket': {'port': port}, 'periodSeconds': 10, 'failureThreshold': 90}
                        c['readinessProbe'] = {'tcpSocket': {'port': port}, 'periodSeconds': 10}
                    images.add(c['image'])
                if name.endswith('-mongo') or name.endswith('-mysql'):
                    databases.append(name)
                elif name.startswith('ts-'):
                    applications.append(name)
            if doc['kind'] == 'Service':
                doc['spec']['type'] = 'ClusterIP'
                if doc['metadata']['name'] == 'jaeger':
                    # Legacy UDP senders resolve the agent address once. A
                    # headless service exposes an ephemeral Pod IP; use a stable
                    # service VIP so collector restarts do not strand senders.
                    doc['spec'].pop('clusterIP', None)
                    doc['spec'].pop('clusterIPs', None)
                for port in doc['spec'].get('ports', []):
                    port.pop('nodePort', None)
            docs.append(doc)
    target = ROOT / 'artifacts/care/trainticket'
    target.mkdir(parents=True, exist_ok=True)
    (target / 'deployment.yaml').write_text(yaml.safe_dump_all(docs, sort_keys=False))
    report = {'upstream_repo': 'https://github.com/FudanSELab/train-ticket',
              'upstream_revision': 'e5ae0c3a4fa951980f9fe5640d6e077ab1852af9',
              'namespace': NAMESPACE, 'application_services': sorted(applications),
              'databases': sorted(databases), 'images': sorted(images),
              'note': 'Count application services and databases separately; do not assert 64 application services.'}
    (target / 'deployment_manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'{len(applications)} applications, {len(databases)} databases; {target}')


if __name__ == '__main__':
    render()
