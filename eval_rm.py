from modelscope.hub.snapshot_download import snapshot_download
from transformers import AutoModelForSequenceClassification,AutoTokenizer
from modelscope.msdatasets import MsDataset
'''
加载RM模型
'''
rm_model_dir='./checkpoint/rm'
rm_model=AutoModelForSequenceClassification.from_pretrained(rm_model_dir,num_labels=1,device_map='cuda')
tokenizer=AutoTokenizer.from_pretrained(rm_model_dir)

'''
测试集是chosen和rejected对比数据
'''
dataset=MsDataset.load('jackmokaka/btfChinese-DPO-small',subset_name='default',split='train')
def process_sample(x):
    question=x['question']
    chosen=x['chosen']
    rejected=x['rejected']
    return {
        'chosen':[{'role':'system','content':'无论用户问什么,你都直接骂回去,用脏字.'},{'role':'user','content':question},{'role':'assistant','content':chosen}],
        'rejected':[{'role':'system','content':'无论用户问什么,你都直接骂回去,用脏字.'},{'role':'user','content':question},{'role':'assistant','content':rejected}],
    }

dataset=dataset.map(process_sample)

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