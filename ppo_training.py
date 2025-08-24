from modelscope.hub.snapshot_download import snapshot_download
from trl import AutoModelForCausalLMWithValueHead,PPOTrainer,PPOConfig
from transformers import AutoTokenizer,AutoModelForCausalLM,AutoModelForSequenceClassification
from peft import LoraConfig
from modelscope.msdatasets import MsDataset

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
实现RewardModel,由于TRL PPO底层假设了RM是LLM based,没法使用Rule based实现.
这里,我们就不自己标注&训练RM模型,引入开源的泛偏好对齐的RM模型:https://modelscope.cn/models/Skywork/Skywork-Reward-V2-Qwen3-0.6B
'''
reward_model_name='Skywork/Skywork-Reward-V2-Qwen3-0.6B'
reward_model_dir=snapshot_download(reward_model_name,cache_dir='./models/')
reward=AutoModelForSequenceClassification.from_pretrained(reward_model_dir,device_map='cuda')

'''
训练集是若干Query,需编码成chatml格式并编码成token id,PPOTrainer会使用policy model续写response
'''
def preprocess(sample):
    input_ids=tokenizer.apply_chat_template(
        conversation=[{'role':'system','content':'回答脑筋急转弯'},{'role':'user','content':sample['question']}],
        add_generation_prompt=True,
        enable_thinking=False
    )
    return {'input_ids':input_ids}

dataset=MsDataset.load('AI-ModelScope/IQuiz',subset_name='IQ',split='test',trust_remote_code=True)
dataset=dataset.map(preprocess,remove_columns=dataset.column_names).train_test_split(test_size=0.1)

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
    local_mini_batch_size=1,
    gradient_accumulation_steps=4,
    total_episodes=1000,
    response_length=500,
    learning_rate=1e-5,
    logging_steps=1,
    save_steps=10,
    eval_steps=10,
    missing_eos_penalty=1.0,
    report_to='tensorboard',
    logging_dir='./tensorbard/',
    output_dir='./qwen3_ppo',
    resume_from_checkpoint='./qwen3_ppo',
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