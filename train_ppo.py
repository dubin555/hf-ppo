from modelscope.hub.snapshot_download import snapshot_download
from trl import PPOTrainer,PPOConfig
from transformers import AutoTokenizer,AutoModelForCausalLM,AutoModelForSequenceClassification
from peft import LoraConfig
import datetime
from modelscope.msdatasets import MsDataset

'''
下载Qwen模型,用作:
 - policy: 要被训练的LLM模型,使用peft Lora微调
 - ref policy： LLM模型,作为policy的参考模型
 - value: 要伴随训练的价值模型,只训练value head,冻结LLM
'''
model_name='Qwen/Qwen2.5-0.5B-Instruct'
model_dir=snapshot_download(model_name,cache_dir='./checkpoint/base')

'''
加载PPO涉及的模型,这里只会用到1张GPU
'''
policy=AutoModelForCausalLM.from_pretrained('./checkpoint/sft',device_map='cuda')
ref_policy=None # policy采用Lora,因此ref_policy和policy共享LLM基座参数
value=AutoModelForSequenceClassification.from_pretrained(model_dir,num_labels=1,device_map='cuda') # 只训value head
tokenizer=AutoTokenizer.from_pretrained(model_dir)

'''
加载RewardModel
'''
reward_model_name='./checkpoint/rm'
reward=AutoModelForSequenceClassification.from_pretrained(reward_model_name,num_labels=1,device_map='cuda')

'''
训练集是若干Query,需编码成chatml格式并编码成token id,PPOTrainer会使用policy model续写response
'''
dataset=MsDataset.load('jackmokaka/btfChinese-DPO-small',subset_name='default',split='train')

def process_sample(x):
    question=x['question']
    input_ids=tokenizer.apply_chat_template(
        conversation=[{'role':'system','content':'无论用户问什么,你都直接骂回去,用脏字.'},{'role':'user','content':question}],
        add_generation_prompt=True,
    )
    return {'input_ids':input_ids}

dataset=dataset.map(process_sample).remove_columns(dataset.column_names).train_test_split(test_size=0.1,shuffle=True)

'''
Policy Lora配置
'''
peft_config = LoraConfig(
    lora_alpha=8,
    lora_dropout=0.05,
    r=32,
    bias='none',
    task_type='CAUSAL_LM',
    target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']
)
'''
PPO训练
'''
ppo_config=PPOConfig(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    local_rollout_forward_batch_size=8,
    vf_coef=0.5,
    cliprange=0.1,
    cliprange_value=0.5,
    total_episodes=1000,
    learning_rate=5e-6,
    num_ppo_epochs=2,
    response_length=200,
    logging_steps=1,
    save_strategy='no',
    eval_steps=10,
    report_to='tensorboard',
    logging_dir=f'./tensorboard/ppo/{datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")}',
    output_dir='./checkpoint/ppo'
)
trainer=PPOTrainer(
    args=ppo_config,
    processing_class=tokenizer,
    model=policy,
    ref_model=ref_policy,
    reward_model=reward,
    value_model=value,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    peft_config=peft_config,
)
trainer.train()
trainer.save_model(ppo_config.output_dir) # 保存Policy Lora权重