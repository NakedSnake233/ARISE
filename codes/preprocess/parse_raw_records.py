import os
import re
import json
import argparse

def SN_service_names(name):
    mapping = {
        'socialnetwork-text-service-1': 'text-service',
        'socialnetwork-home-timeline-service-1': 'home-timeline-service',
        'socialnetwork-media-service-1': 'media-service',
        'socialnetwork-post-storage-service-1':'post-storage-service',
        'socialnetwork-social-graph-service-1':'social-graph-service',
        'socialnetwork-url-shorten-service-1':'url-shorten-service',
        'socialnetwork-nginx-thrift-1':'nginx-web-server',
        'socialnetwork-unique-id-service-1':'unique-id-service',
        'socialnetwork-user-service-1':'user-service',
        'socialnetwork-compose-post-service-1':'compose-post-service',
        'socialnetwork-user-timeline-service-1':'user-timeline-service',
        'socialnetwork-user-mention-service-1':'user-mention-service'
    }
    if name in mapping:
        return mapping[name]
    else:
        raise KeyError('Service name not found: {}'.format(name))

def TT_service_names(name):
    return name.split("_")[1]

def process_raw_records(dataset_path, data_name):
    path = os.path.join(dataset_path, 'data')
    record_files = [f for f in os.listdir(path) if f.endswith('.json')]
    record_files.sort()
    record2idx = {file_name: idx for idx, file_name in enumerate(record_files)}
    
    for file_name in record_files:
        with open(os.path.join(path, file_name), 'r') as f:
            records = json.load(f)
        
        processed_records = {}
        processed_records['start'] = int(records['start'])
        processed_records['end'] = int(records['end'])
        # add 16 hours to the start and end time
        # to align with the time interal in the file-name
        # https://github.com/BEbillionaireUSD/Eadro/issues/11#issuecomment-1887219310
        processed_records['start'] += 16 * 3600
        processed_records['end'] += 16 * 3600
        
        processed_records['faults'] = []
        for fault in records['faults']:
            service = SN_service_names(fault['name']) if data_name == 'SN' else TT_service_names(fault['name'])
            fault_type = fault['fault']
            start = int(fault['start'])
            end = int(fault['start'] + fault['duration'])
            
            start += 16 * 3600
            end += 16 * 3600
            
            processed_records['faults'].append({
                'service': service,
                'fault_type': fault_type,
                's': start,
                'e': end
            })
        
        save_path = os.path.join('./parsed_data', data_name, 'records' + str(record2idx[file_name]) + '.json')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(processed_records, f)
        
        # # check the timestamps as well as the datetime in file names. 
        # import pandas as pd
        # start_datetime = pd.to_datetime(processed_records['start'], unit='s')
        # end_datetime = pd.to_datetime(processed_records['end'], unit='s')
        # print(f"{file_name}: \n {start_datetime} -- {end_datetime}")
    
    
    


if __name__ == "__main__":
    root_path = os.getenv('ROOT_PATH')
    for dataset_name in ['SN', 'TT']:
        if dataset_name == 'SN':
            dataset_path = os.path.join(root_path, 'SN Dataset')
        elif dataset_name == 'TT':
            dataset_path = os.path.join(root_path, 'TT Dataset')
    
        process_raw_records(dataset_path, dataset_name)
    
