from modelscope.hub.snapshot_download import snapshot_download
from transformers import AutoModelForSequenceClassification,AutoTokenizer
from trl import RewardTrainer, RewardConfig
from modelscope.msdatasets import MsDataset
import datetime

'''
下载Qwen模型,用于训练Reward Model
'''
model_name='Qwen/Qwen2.5-0.5B-Instruct'
model_dir=snapshot_download(model_name,cache_dir='./checkpoint/base')

reward_model=AutoModelForSequenceClassification.from_pretrained(model_dir,num_labels=1,device_map='cuda') # 只训score head
tokenizer=AutoTokenizer.from_pretrained(model_dir)
reward_model.config.pad_token_id=tokenizer.pad_token_id

'''
训练集是chosen和rejected对比数据
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

dataset=dataset.map(process_sample).remove_columns(['question','system'])

'''
RewardModel 训练
'''
reward_config=RewardConfig(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    num_train_epochs=2,
    max_length=500,
    learning_rate=1e-5,
    center_rewards_coefficient=0.01, # chosen reward + rejected reward靠近0
    logging_steps=1,
    save_strategy='no',
    report_to='tensorboard', # tensorboard --logdir ./tensorboard/rm/
    logging_dir=f'./tensorboard/rm/{datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")}',
    output_dir='./checkpoint/rm'
)
trainer=RewardTrainer(
    model=reward_model,
    args=reward_config,
    processing_class=tokenizer,
    train_dataset=dataset,
)
trainer.train()
trainer.save_model(reward_config.output_dir)