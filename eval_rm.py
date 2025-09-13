import os
import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

"""
评估 Reward Model：
1. 加载训练好的 RM Checkpoint。
2. 使用与训练时相同的验证集进行评估。
3. 计算 RM 对偏好对 (chosen/rejected) 的预测准确率。
"""

# --------------------
# 配置
# --------------------
RM_MODEL_DIR = './checkpoint/rm'
DATA_REPO = "Karsh-CAI/btfChinese-DPO-small"
DATA_CACHE_DIR = "./checkpoint/data/"
SYSTEM_PROMPT = "无论用户问什么,你都直接骂回去,用脏字."
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
MAX_SAMPLES_TO_TEST = 100 # 只测试前100个样本，设为None则测试全部

# --------------------
# 加载模型和分词器
# --------------------
print(f"Loading Reward Model from {RM_MODEL_DIR}...")
rm_model = AutoModelForSequenceClassification.from_pretrained(
    RM_MODEL_DIR,
    device_map='cuda',
    trust_remote_code=True,
    token=HF_TOKEN
)
tokenizer = AutoTokenizer.from_pretrained(
    RM_MODEL_DIR,
    trust_remote_code=True,
    token=HF_TOKEN
)
rm_model.eval() # 设置为评估模式

# --------------------
# 加载和准备数据
# --------------------
print(f"Loading dataset {DATA_REPO}...")
raw_dataset = load_dataset(
    DATA_REPO,
    split="train",
    cache_dir=DATA_CACHE_DIR
)

# 使用与训练时完全相同的种子和比例来划分数据集，以获取相同的验证集
split_dataset = raw_dataset.train_test_split(test_size=0.05, shuffle=True, seed=42)
eval_dataset = split_dataset["test"]

print(f"Found {len(eval_dataset)} samples in the evaluation set. Testing on {MAX_SAMPLES_TO_TEST or len(eval_dataset)} samples.")

# --------------------
# 评估循环
# --------------------
correct_predictions = 0
total_predictions = 0

with torch.no_grad(): # 关闭梯度计算，加速推理
    # 确定要迭代的样本数量
    num_samples = MAX_SAMPLES_TO_TEST if MAX_SAMPLES_TO_TEST is not None else len(eval_dataset)
    
    for i in range(num_samples):
        sample = eval_dataset[i]
        
        # 构造 messages
        chosen_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": sample["question"]},
            {"role": "assistant", "content": sample["chosen"]}
        ]
        rejected_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": sample["question"]},
            {"role": "assistant", "content": sample["rejected"]}
        ]

        # 转换为文本
        chosen_text = tokenizer.apply_chat_template(chosen_messages, tokenize=False, add_generation_prompt=False)
        rejected_text = tokenizer.apply_chat_template(rejected_messages, tokenize=False, add_generation_prompt=False)

        # 分词并获取分数
        chosen_inputs = tokenizer(chosen_text, return_tensors="pt", truncation=True, max_length=512).to(rm_model.device)
        chosen_score = rm_model(**chosen_inputs).logits[0].item()

        rejected_inputs = tokenizer(rejected_text, return_tensors="pt", truncation=True, max_length=512).to(rm_model.device)
        rejected_score = rm_model(**rejected_inputs).logits[0].item()

        # 检查预测是否正确
        if chosen_score > rejected_score:
            correct_predictions += 1
        total_predictions += 1

        print(f"Sample {i+1}: Chosen Score = {chosen_score:.4f}, Rejected Score = {rejected_score:.4f} -> {'Correct' if chosen_score > rejected_score else 'Incorrect'}")

# --------------------
# 打印最终结果
# --------------------
if total_predictions > 0:
    accuracy = (correct_predictions / total_predictions) * 100
    print("\n" + "="*50)
    print("Evaluation Complete!")
    print(f"Total Samples Tested: {total_predictions}")
    print(f"Correct Predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy:.2f}%")
    print("="*50)
else:
    print("No samples were tested.")