from modelscope.hub.snapshot_download import snapshot_download
from transformers import AutoModelForCausalLM,AutoTokenizer
from peft import PeftModel

'''
加载base模型
'''
base_model_name='Qwen/Qwen3-0.6B'
base_model_dir=snapshot_download(base_model_name,cache_dir='./models/')
base_model=AutoModelForCausalLM.from_pretrained(base_model_dir,device_map='cuda')
tokenizer=AutoTokenizer.from_pretrained(base_model_dir)

'''
加载lora权重,然后合并到主模型
'''
peft_model_dir='./ppo_checkpoint'
peft_model=PeftModel.from_pretrained(base_model,peft_model_dir,device_map='cuda')
model=peft_model.merge_and_unload()

'''
推理
'''
text=tokenizer.apply_chat_template(
    conversation=[{'role':'system','content':'计算并返回结果'},{'role':'user','content':'1+1=?'}],
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False
)
model_inputs=tokenizer([text],return_tensors='pt').to(model.device)
generated_ids=model.generate(**model_inputs,max_new_tokens=32768)
output_ids=generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
response=tokenizer.decode(output_ids,skip_special_tokens=True)
print(f'---模型输出---\n{response}')