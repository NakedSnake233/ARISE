from transformers import AutoModelForCausalLM, AutoTokenizer      # 指定下载路径   
cache_dir = "/root/models"      # 加载模型和分词器，并指定缓存路径      
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"   
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)   
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)   
