from modelscope.hub.snapshot_download import snapshot_download
from trl import PPOTrainer,PPOConfig
from transformers import AutoTokenizer,AutoModelForCausalLM,AutoModelForSequenceClassification
from peft import LoraConfig
from datasets import Dataset
import datetime
import random

'''
下载Qwen3-0.6B模型,用作:
 - policy: 要被训练的LLM模型,使用peft Lora微调
 - ref policy： LLM模型,作为policy的参考模型
 - value: 要伴随训练的价值模型,只训练value head,冻结LLM
'''
model_name='Qwen/Qwen3-0.6B'
model_dir=snapshot_download(model_name,cache_dir='./models/')

'''
加载PPO涉及的模型,这里只会用到1张GPU
'''
policy=AutoModelForCausalLM.from_pretrained(model_dir,device_map='cuda')
ref_policy=None # ref_policy=AutoModelForCausalLM.from_pretrained(model_dir,device_map='cuda')  # policy采用Lora,所以ref_policy和policy共享LLM参数
value=AutoModelForSequenceClassification.from_pretrained(model_dir,num_labels=1,device_map='cuda') # 只训value head
tokenizer=AutoTokenizer.from_pretrained(model_dir)

'''
加载RewardModel
'''
reward_model_name='./rm_checkpoint/'
reward=AutoModelForSequenceClassification.from_pretrained(reward_model_name,num_labels=1,device_map='cuda')

'''
训练集是若干Query,需编码成chatml格式并编码成token id,PPOTrainer会使用policy model续写response
'''
def generate_datasets(size):
    sample_list=[]
    for i in range(size):
        numbers=list('12345')
        random.shuffle(numbers)
        query=f'随机返回{",".join(numbers)}中的1个数字,只返回数字,不要说其他的.'
        input_ids=tokenizer.apply_chat_template(
            conversation=[{'role':'user','content':query},],
            add_generation_prompt=True,
            enable_thinking=False
        )
        sample_list.append({'input_ids':input_ids})
    return Dataset.from_list(sample_list)

dataset=generate_datasets(1000).train_test_split(test_size=0.1)

'''
Policy Lora配置
'''
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=64,
    bias='none',
    task_type='CAUSAL_LM',
    target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']
)
'''
PPO训练
'''
ppo_config=PPOConfig(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    total_episodes=400,
    # NOTICE: reward model很难遇到LLM输出数字1,所以奖励非常稀疏
    # 很容易导致Value过早收敛(全是负奖励,模型会认为这就是一切了),所以下面4个参数控制Value收敛速度,给Policy更多时间探索
    vf_coef=0.02,
    cliprange_value=0.05,
    learning_rate=5e-6,
    num_ppo_epochs=2, # 避免陷入局部
    lam=0.99, # 关注长期奖励
    #########################################
    response_length=500,
    missing_eos_penalty=1.0,
    logging_steps=1,
    save_strategy='no',
    eval_steps=10,
    report_to='tensorboard',
    logging_dir=f'./tensorbard/ppo/{datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")}',
    output_dir='./ppo_checkpoint/'
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