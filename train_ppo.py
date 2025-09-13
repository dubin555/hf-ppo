import os
import datetime
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import numpy as np

"""
PPO训练脚本 - 使用LoRA微调
要求：
1. 必须先运行train_sft.py生成SFT checkpoint
2. 必须先运行train_rm.py生成Reward Model checkpoint
"""

# --------------------
# 配置
# --------------------
MODEL_NAME = "Qwen/Qwen3-0.6B"
CACHE_DIR = "./checkpoint/base/"
DATA_REPO = "Karsh-CAI/btfChinese-DPO-small"
DATA_CACHE_DIR = "./checkpoint/data/"
SYSTEM_PROMPT = "无论用户问什么,你都直接骂回去,用脏字."
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
MAX_LENGTH = 512
GENERATION_MAX_LENGTH = 256

# Checkpoint路径
SFT_CHECKPOINT = "./checkpoint/sft"
RM_CHECKPOINT = "./checkpoint/rm"
PPO_OUTPUT_DIR = "./checkpoint/ppo"

# --------------------
# LoRA配置
# --------------------
peft_config = LoraConfig(
    r=32,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

# --------------------
# 加载模型
# --------------------
print(f"Loading policy model from SFT checkpoint: {SFT_CHECKPOINT}")

# 先创建带value head的模型，然后应用LoRA
policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(
    SFT_CHECKPOINT,
    device_map="cuda",
    trust_remote_code=True,
    token=HF_TOKEN,
    torch_dtype=torch.bfloat16,
    peft_config=peft_config  # 直接在加载时应用LoRA配置
)

# Reference model - 使用原始base model作为参考模型（冻结，不需要LoRA）
print(f"Loading reference model from base: {MODEL_NAME}")
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
    MODEL_NAME,
    device_map="cuda",
    cache_dir=CACHE_DIR,
    trust_remote_code=True,
    token=HF_TOKEN,
    torch_dtype=torch.bfloat16
)

# Tokenizer - 从SFT checkpoint加载
print(f"Loading tokenizer from SFT checkpoint: {SFT_CHECKPOINT}")
tokenizer = AutoTokenizer.from_pretrained(
    SFT_CHECKPOINT,
    trust_remote_code=True,
    token=HF_TOKEN
)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

# Reward Model - 从训练好的RM checkpoint加载
print(f"Loading reward model from: {RM_CHECKPOINT}")
reward_model = AutoModelForSequenceClassification.from_pretrained(
    RM_CHECKPOINT,
    num_labels=1,
    device_map="cuda",
    trust_remote_code=True,
    token=HF_TOKEN,
    torch_dtype=torch.bfloat16
)
reward_model.eval()  # 冻结reward model

# --------------------
# 数据加载和预处理
# --------------------
raw_dataset = load_dataset(
    DATA_REPO,
    split="train",
    cache_dir=DATA_CACHE_DIR,
    token=HF_TOKEN
)

def process_sample(example):
    """
    处理样本，生成PPO训练所需的query格式
    """
    question = example['question']
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question}
    ]
    
    # 应用chat template并添加generation prompt
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        truncation=True,
        max_length=MAX_LENGTH // 2,  # 留出空间给response
        return_tensors=None
    )
    
    return {"input_ids": input_ids, "query": tokenizer.decode(input_ids)}

# 处理数据集
processed_dataset = raw_dataset.map(
    process_sample,
    remove_columns=raw_dataset.column_names,
    desc="Processing samples for PPO",
    num_proc=4
)

# 划分训练集
split_dataset = processed_dataset.train_test_split(
    test_size=0.1,
    shuffle=True,
    seed=42
)
train_dataset = split_dataset["train"]

print(f"Training samples: {len(train_dataset)}")

# --------------------
# PPO配置
# --------------------
ppo_config = PPOConfig(
    # 基础配置
    exp_name="ppo_lora_training",
    seed=42,
    
    # 模型和数据集信息
    model_name=MODEL_NAME,
    query_dataset=DATA_REPO,
    
    # 训练步数和批次大小 - 调小以适应LoRA训练
    steps=200,  # 总训练步数
    batch_size=8,  # 每个PPO更新步骤的批次大小
    mini_batch_size=2,  # 每个mini batch的大小
    gradient_accumulation_steps=1,  # 梯度累积步数
    
    # 学习率和优化器
    learning_rate=5e-5,  # LoRA可以使用稍高的学习率
    max_grad_norm=1.0,
    
    # PPO特定参数
    ppo_epochs=4,  # 每批数据的PPO优化轮数
    
    # KL散度控制
    adap_kl_ctrl=True,
    init_kl_coef=0.2,  # 降低KL惩罚，因为LoRA变化较小
    target=6.0,
    horizon=10000.0,
    
    # PPO裁剪参数
    cliprange=0.2,
    cliprange_value=0.2,
    
    # 损失函数系数
    vf_coef=0.1,
    
    # GAE参数
    gamma=1.0,
    lam=0.95,
    
    # 早停和阈值
    early_stopping=False,
    target_kl=3.0,  # 提高阈值，因为LoRA变化较小
    ratio_threshold=10.0,
    
    # 奖励处理
    use_score_scaling=False,
    use_score_norm=False,
    whiten_rewards=True,
    
    # 其他优化选项
    optimize_device_cache=False,
    gradient_checkpointing=False,
    
    # 数据处理
    remove_unused_columns=False,
    dataset_num_proc=4,
)

# 打印配置信息
print("\n" + "="*50)
print("PPO LoRA Configuration:")
print("="*50)
print(f"  Total steps: {ppo_config.steps}")
print(f"  Batch size: {ppo_config.batch_size}")
print(f"  Mini batch size: {ppo_config.mini_batch_size}")
print(f"  Learning rate: {ppo_config.learning_rate}")
print(f"  LoRA rank: {peft_config.r}")
print(f"  LoRA alpha: {peft_config.lora_alpha}")
print("="*50 + "\n")

# --------------------
# 创建PPO Trainer (传入peft_config而不是已应用的model)
# --------------------
trainer = PPOTrainer(
    config=ppo_config,
    model=policy_model,  # AutoModelForCausalLMWithValueHead类型
    ref_model=ref_model,  # 冻结的reference model
    tokenizer=tokenizer,
    dataset=train_dataset,
    data_collator=None,
)

# 打印可训练参数信息
if hasattr(policy_model, 'print_trainable_parameters'):
    policy_model.print_trainable_parameters()
elif hasattr(policy_model, 'pretrained_model') and hasattr(policy_model.pretrained_model, 'print_trainable_parameters'):
    policy_model.pretrained_model.print_trainable_parameters()

# --------------------
# 定义reward计算函数
# --------------------
def compute_reward_scores(texts):
    """
    使用reward model计算文本的奖励分数
    """
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    ).to(reward_model.device)
    
    with torch.no_grad():
        outputs = reward_model(**inputs)
        scores = outputs.logits.squeeze(-1).float()  # 添加 .float() 转换为float32
    
    return scores

# --------------------
# 训练循环
# --------------------
print("Starting PPO training with LoRA...")
print(f"Output will be saved to: {PPO_OUTPUT_DIR}\n")

# 创建生成长度采样器
output_length_sampler = LengthSampler(min_value=32, max_value=GENERATION_MAX_LENGTH)

generation_kwargs = {
    "min_length": -1,
    "top_k": 50,
    "top_p": 0.95,
    "do_sample": True,
    "pad_token_id": tokenizer.pad_token_id,
    "eos_token_id": tokenizer.eos_token_id,
    "temperature": 0.7,
}

for step in range(ppo_config.steps):
    # 准备batch
    batch_indices = np.random.choice(len(train_dataset), ppo_config.batch_size, replace=False)
    batch = [train_dataset[int(i)] for i in batch_indices]
    
    queries = [torch.tensor(sample["input_ids"]).to("cuda") for sample in batch]
    
    # 生成响应
    response_tensors = []
    for query in queries:
        gen_len = output_length_sampler()
        generation_kwargs["max_new_tokens"] = gen_len
        
        response = trainer.generate(
            query,
            return_prompt=False,
            **generation_kwargs
        )
        response_tensors.append(response.squeeze())
    
    # 准备用于reward计算的完整文本
    texts = []
    for query, response in zip(queries, response_tensors):
        # 组合query和response
        full_ids = torch.cat([query, response])
        text = tokenizer.decode(full_ids, skip_special_tokens=True)
        texts.append(text)
    
    # 计算rewards
    rewards = compute_reward_scores(texts)
    
    # 运行PPO步骤
    stats = trainer.step(queries, response_tensors, [reward for reward in rewards])
    
    # 准备log_stats需要的batch字典格式
    batch_dict = {
        "query": [q for q in queries],
        "response": [r for r in response_tensors],
    }
    
    # 记录日志
    trainer.log_stats(stats, batch_dict, rewards)
    
    # 定期打印进度
    if step % 10 == 0:
        print(f"Step {step}/{ppo_config.steps}: mean_reward={rewards.mean().item():.4f}")
        
    # 定期保存checkpoint
    if (step + 1) % 50 == 0:
        checkpoint_dir = f"{PPO_OUTPUT_DIR}/checkpoint-{step+1}"
        print(f"Saving checkpoint to {checkpoint_dir}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        # 直接保存模型而不是通过trainer
        policy_model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)

# --------------------
# 保存最终模型
# --------------------
print(f"\nSaving final model to {PPO_OUTPUT_DIR}")
os.makedirs(PPO_OUTPUT_DIR, exist_ok=True)

# 保存模型
policy_model.save_pretrained(PPO_OUTPUT_DIR)

# 保存tokenizer
tokenizer.save_pretrained(PPO_OUTPUT_DIR)

print("\n" + "="*50)
print("PPO LoRA Training completed successfully!")
print(f"Model saved to: {PPO_OUTPUT_DIR}")
print("="*50)