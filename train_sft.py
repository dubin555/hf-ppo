from trl import SFTConfig, SFTTrainer
from modelscope.hub.snapshot_download import snapshot_download
from transformers import AutoModelForCausalLM,AutoTokenizer
from trl import SFTConfig
from modelscope.msdatasets import MsDataset
import datetime

'''
下载Qwen模型,用于训练Reward Model
'''
model_name='Qwen/Qwen2.5-0.5B-Instruct'
model_dir=snapshot_download(model_name,cache_dir='./checkpoint/base/')

model=AutoModelForCausalLM.from_pretrained(model_dir,device_map='cuda') # 只训score head
tokenizer=AutoTokenizer.from_pretrained(model_dir)

dataset=MsDataset.load('jackmokaka/btfChinese-DPO-small',subset_name='default',split='train')

def process_sample(x):
    question=x['question']
    chosen=x['chosen']
    return {
        'messages':[{'role':'system','content':'无论用户问什么,你都直接骂回去,用脏字.'},{'role':'user','content':question},{'role':'assistant','content':chosen}],
    }

dataset=dataset.map(process_sample).remove_columns(dataset.column_names).train_test_split(test_size=0.1,shuffle=False)

sft_config=SFTConfig(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    num_train_epochs=2,
    max_length=500,
    learning_rate=1e-5,
    logging_steps=1,
    save_strategy='no',
    report_to='tensorboard', # tensorboard --logdir ./tensorboard/rm/
    logging_dir=f'./tensorboard/sft/{datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")}',
    output_dir='./checkpoint/sft'
)
trainer=SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=dataset['train'],
    processing_class=tokenizer,
)
trainer.train()
trainer.save_model(sft_config.output_dir)