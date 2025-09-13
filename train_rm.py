import os
import datetime
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import RewardTrainer, RewardConfig

"""
训练 Reward Model：pairwise (chosen / rejected)
修复点：
1. 兼容 apply_chat_template 输出字符串再编码
2. 返回 RewardTrainer 需要的 *_chosen / *_rejected 字段
3. 适配 transformers Trainer 传入 num_items_in_batch 的签名差异
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

# --------------------
# 模型 & 分词器
# --------------------
reward_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=1,
    device_map="cuda",
    cache_dir=CACHE_DIR,
    trust_remote_code=True,
    token=HF_TOKEN
)
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    cache_dir=CACHE_DIR,
    trust_remote_code=True,
    token=HF_TOKEN
)
if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
    tokenizer.pad_token = tokenizer.eos_token
reward_model.config.pad_token_id = tokenizer.pad_token_id

# --------------------
# 数据加载
# --------------------
raw_dataset = load_dataset(
    DATA_REPO,
    split="train",
    cache_dir=DATA_CACHE_DIR
)

def build_chat_text(messages):
    """
    使用 chat template 生成单一字符串（不直接 tokenize）
    """
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,          # 返回字符串
        add_generation_prompt=False
    )

def encode_text(text):
    enc = tokenizer(
        text,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
        add_special_tokens=True
    )
    return enc["input_ids"], enc["attention_mask"]

def process_and_tokenize(example):
    chosen_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": example["question"]},
        {"role": "assistant", "content": example["chosen"]}
    ]
    rejected_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": example["question"]},
        {"role": "assistant", "content": example["rejected"]}
    ]

    chosen_text = build_chat_text(chosen_messages)
    rejected_text = build_chat_text(rejected_messages)

    input_ids_chosen, attention_mask_chosen = encode_text(chosen_text)
    input_ids_rejected, attention_mask_rejected = encode_text(rejected_text)

    return {
        "input_ids_chosen": input_ids_chosen,
        "attention_mask_chosen": attention_mask_chosen,
        "input_ids_rejected": input_ids_rejected,
        "attention_mask_rejected": attention_mask_rejected,
    }

tokenized_dataset = raw_dataset.map(
    process_and_tokenize,
    remove_columns=raw_dataset.column_names,
    desc="Tokenizing pairwise samples"
)

# 划分验证集
split_dataset = tokenized_dataset.train_test_split(test_size=0.05, shuffle=True, seed=42)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

# --------------------
# Reward 配置
# --------------------
reward_config = RewardConfig(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    num_train_epochs=2,
    max_length=MAX_LENGTH,
    learning_rate=1e-5,
    center_rewards_coefficient=0.01,
    logging_steps=10,
    save_strategy="no",
    report_to="swanlab",
    output_dir="./checkpoint/rm",
)

# --------------------
# 兼容 compute_loss 签名的补丁
# --------------------
class PatchedRewardTrainer(RewardTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # 直接调用父类（当前 RewardTrainer 的 compute_loss 不支持 num_items_in_batch，忽略即可）
        return super().compute_loss(model, inputs, return_outputs=return_outputs)

trainer = PatchedRewardTrainer(
    model=reward_model,
    args=reward_config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

trainer.train()
trainer.save_model(reward_config.output_dir)