import os
import json

from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig

import pandas as pd
import re

def extract_unix_timestamp(log: str) -> float:
    time_pattern = r"\d{4}-(?:[A-Za-z]{3}|\d{2})-\d{2} \d{2}:\d{2}:\d{2}\.\d+"
    match = re.search(time_pattern, log)
    if not match:
        return None
    
    time_str = match.group(0)
    
    try:
        unix_time = pd.to_datetime(time_str, format="%Y-%b-%d %H:%M:%S.%f").timestamp()
    except ValueError:
        unix_time = pd.to_datetime(time_str, format="%Y-%m-%d %H:%M:%S.%f").timestamp()
    
    return unix_time

def extract_log_template(fault_free_dataset_path, fault_time_dataset_path, dataset_name):
    # list all subfolders in the fault-free dataset path
    subfolders = [f for f in os.listdir(fault_free_dataset_path) if os.path.isdir(os.path.join(fault_free_dataset_path, f))]
    log_paths = [os.path.join(fault_free_dataset_path, f, 'logs.json') for f in subfolders]
    
    all_log_strs = []
    for path in log_paths:
        with open(path, 'r') as f:
            logs = json.load(f)
        for k, v in logs.items():
            all_log_strs.extend(v)
    
    config = TemplateMinerConfig()
    config.load('./drain3.ini')
    config.profiling_eennabilmed = True
    miner = TemplateMiner(config=config)
    for log_str in all_log_strs:
        miner.add_log_message(log_str)
    
    templates = []
    for cluster in miner.drain.clusters:
        templates.append(cluster.get_template())
        print(cluster)
    print("*"*90)
    
    templates_save_path = os.path.join("./parsed_data", dataset_name)
    os.makedirs(templates_save_path, exist_ok=True)
    
    templates_file_path = os.path.join(templates_save_path, "templates.json")
    with open(templates_file_path, 'w') as f:
        json.dump(templates, f)


    # convert fault logs into templates
    subfolders = [f for f in os.listdir(fault_time_dataset_path) if os.path.isdir(os.path.join(fault_time_dataset_path, f))]
    log_paths = [os.path.join(fault_time_dataset_path, f, 'logs.json') for f in subfolders]
    
    log_paths = sorted(log_paths)
    for idx, path in enumerate(log_paths):
        with open(path, 'r') as f:
            log_dict = json.load(f)
        df = {'timestamp':[], 'service':[], 'events':[]}
        for service, log_list in log_dict.items():
            for log_line in log_list:
                match = miner.match(log_line)
                if match:
                    log_temp = match.get_template()
                else:
                    print(log_line)
                    log_temp = "Unseen"
                log_time = extract_unix_timestamp(log_line)
                assert log_time is not None, log_line
                # print(log_temp, log_time) ;exit()
                df['timestamp'].append(log_time)
                df['service'].append(service)
                df['events'].append(log_temp)
        df = pd.DataFrame(df)
        df.to_csv(os.path.join("./parsed_data", dataset_name, "logs"+str(idx)+".csv"))
                
    print("======"*30)
    


if __name__ == '__main__':
    root_path = os.getenv('ROOT_PATH')
    for dataset_name in ['SN', 'TT']:
        
        if dataset_name == 'SN':
            fault_free_dataset_path = os.path.join(root_path, 'SN Dataset', 'no fault')
            fault_time_dataset_path = os.path.join(root_path, 'SN Dataset', 'data')
        elif dataset_name == 'TT':
            fault_free_dataset_path = os.path.join(root_path, 'TT Dataset', 'no fault')
            fault_time_dataset_path = os.path.join(root_path, 'TT Dataset', 'data')
    
        extract_log_template(fault_free_dataset_path, fault_time_dataset_path, dataset_name)
