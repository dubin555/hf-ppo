from modelscope.hub.snapshot_download import snapshot_download
from peft import LoraConfig
from transformers import AutoModelForSequenceClassification,AutoTokenizer
from trl import RewardTrainer, RewardConfig
from datasets import Dataset
import random
import datetime
from peft import TaskType
'''
下载Qwen3-0.6B模型,用于训练Reward Model
'''
model_name='Qwen/Qwen3-0.6B'
model_dir=snapshot_download(model_name,cache_dir='./models/')

reward_model=AutoModelForSequenceClassification.from_pretrained(model_dir,num_labels=1,device_map='cuda') # 只训score head
tokenizer=AutoTokenizer.from_pretrained(model_dir)
reward_model.config.pad_token_id=tokenizer.pad_token_id

'''
训练集是chosen和rejected对比数据
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

'''
RewardModel 训练
'''
reward_config=RewardConfig(
    per_device_train_batch_size=16,
    gradient_accumulation_steps=4,
    num_train_epochs=2,
    max_length=500,
    learning_rate=1e-5,
    center_rewards_coefficient=0.01, # chosen reward + rejected reward靠近0
    logging_steps=1,
    save_strategy='no',
    report_to='tensorboard', # tensorboard --logdir ./tensorboard/rm/
    logging_dir=f'./tensorbard/rm/{datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")}',
    output_dir='./rm_checkpoint'
)
trainer=RewardTrainer(
    model=reward_model,
    args=reward_config,
    processing_class=tokenizer,
    train_dataset=dataset,
)
trainer.train()
trainer.save_model(reward_config.output_dir)