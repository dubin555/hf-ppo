from modelscope.hub.snapshot_download import snapshot_download
from transformers import AutoModelForCausalLM,AutoTokenizer

'''
加载base模型
'''
model_name='Qwen/Qwen2.5-0.5B-Instruct'
model_dir=snapshot_download(model_name,cache_dir='./models/')
model=AutoModelForCausalLM.from_pretrained(model_dir,device_map='cuda')
tokenizer=AutoTokenizer.from_pretrained(model_dir)

'''
推理
'''
system='''无论用户提问什么,你都骂回去'''
for i in range(10):
    question=input('输入问题:')
    text=tokenizer.apply_chat_template(
        conversation=[{'role':'system','content':system},{'role':'user','content':question},],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    model_inputs=tokenizer([text],return_tensors='pt').to(model.device)
    generated_ids=model.generate(**model_inputs,max_new_tokens=32768)
    output_ids=generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    response=tokenizer.decode(output_ids,skip_special_tokens=True)
    print(f'---模型输出---\n{response}\n')