from modelscope.hub.snapshot_download import snapshot_download
from transformers import AutoModelForSequenceClassification,AutoTokenizer
from peft import PeftModel
from datasets import Dataset
import random 

'''
加载RM模型
'''
rm_model_dir='./rm_checkpoint'
rm_model=AutoModelForSequenceClassification.from_pretrained(rm_model_dir,num_labels=1,device_map='cuda')
tokenizer=AutoTokenizer.from_pretrained(rm_model_dir)

'''
测试集是chosen和rejected对比数据
'''
def generate_datasets(size):
    sample_list=[]
    for i in range(size):
        numbers=list('12345')
        random.shuffle(numbers)
        query=f'随机返回{",".join(numbers)}中的1个数字,只返回数字,不要说其他的.'
        sample={
            'chosen':[{'role':'user','content':query},{'role':'assistant','content':'1'}],
            'rejected':[{'role':'user','content':query},{'role':'assistant','content':str(random.randint(2,5))}]
        }
        sample_list.append(sample)
    return Dataset.from_list(sample_list)

dataset=generate_datasets(1000)

for i in range(len(dataset)):
    chosen=tokenizer.apply_chat_template(
        dataset[i]['chosen'],
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    rejected=tokenizer.apply_chat_template(
        dataset[i]['rejected'],
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    chosen_inputs=tokenizer([chosen],return_tensors="pt").to(rm_model.device)
    chosen_outputs=rm_model(**chosen_inputs)
    rejected_inputs=tokenizer([rejected],return_tensors="pt").to(rm_model.device)
    rejected_outputs=rm_model(**rejected_inputs)
    print(chosen_outputs.logits,rejected_outputs.logits)