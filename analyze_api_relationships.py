import pandas as pd
import json
from typing import Dict, List, Set, Tuple
import re

def clean_type(type_str: str) -> str:
    """清理类型字符串，移除泛型信息等"""
    if pd.isna(type_str):
        return ""
    # 移除泛型信息，如 List<xxx> 只保留 List
    type_str = re.sub(r'<.*?>', '', type_str)
    # 移除数组标记
    type_str = type_str.replace('[]', '')
    # 转换为小写并去除空格
    return type_str.lower().strip()

def parse_parameter_string(param_str: str) -> Set[str]:
    """解析参数字符串，提取参数名"""
    if pd.isna(param_str):
        return set()
    
    # 如果是JSON格式，尝试解析
    if param_str.strip().startswith('[') or param_str.strip().startswith('{'):
        try:
            data = json.loads(param_str)
            if isinstance(data, list):
                return {item.get('name', '') for item in data if isinstance(item, dict)}
            elif isinstance(data, dict):
                return set(data.keys())
        except json.JSONDecodeError:
            pass
    
    # 如果不是JSON格式，按换行符分割
    params = set()
    for line in param_str.split('\n'):
        # 提取参数名（第一个字段）
        param_name = line.split()[0] if line.strip() else ''
        if param_name:
            params.add(param_name)
    
    return params

def analyze_api_relationships(excel_path: str) -> List[Tuple[str, str, str, str]]:
    """分析API之间的调用关系"""
    # 读取Excel文件中的"出入参信息(新增)"表
    df = pd.read_excel(excel_path, sheet_name="出入参信息(新增)")
    
    # 存储API信息
    api_info = {}
    relationships = []
    
    # 处理每个API
    for _, row in df.iterrows():
        api_name = row['*接口中文名']
        if pd.isna(api_name) or api_name == '和接口sheet中接口中文名保持一致':
            continue
            
        input_params = parse_parameter_string(row['*接口入参'])
        output_params = parse_parameter_string(row['*接口出参'])
        
        api_info[api_name] = {
            'input': input_params,
            'output': output_params
        }
    
    # 分析API之间的关系
    for api1, info1 in api_info.items():
        for api2, info2 in api_info.items():
            if api1 != api2:
                # 检查api1的输出是否匹配api2的输入
                common_params = info1['output'].intersection(info2['input'])
                if common_params:
                    relationships.append((
                        api1,
                        api2,
                        ', '.join(common_params),
                        f"{api1}的输出参数 {', '.join(common_params)} 可以作为 {api2} 的输入参数"
                    ))
    
    return relationships

def main():
    excel_path = 'data/Electirc/API.xlsx'
    relationships = analyze_api_relationships(excel_path)
    
    print("\nAPI调用关系分析结果：")
    print("=" * 80)
    for source_api, target_api, common_params, description in relationships:
        print(f"\n源API: {source_api}")
        print(f"目标API: {target_api}")
        print(f"匹配参数: {common_params}")
        print(f"关系描述: {description}")
        print("-" * 80)

    # 保存为CSV文件
    df_result = pd.DataFrame(relationships, columns=["源API", "目标API", "匹配参数", "关系描述"])
    df_result.to_csv("api_relationships.csv", index=False, encoding="utf-8-sig")
    print("\n分析结果已保存为 api_relationships.csv\n")

if __name__ == "__main__":
    main() 