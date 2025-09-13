import os
import torch
import gc
import csv
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from peft import AutoPeftModelForCausalLM
from tqdm import tqdm

"""
批量评估脚本：
1. 使用数据集的问题分别通过 SFT 和 PPO 模型生成回复
2. 使用 RM 模型对回复进行打分
3. 将结果保存到 CSV 文件
4. 测试常见问题并输出到控制台
"""

# --------------------
# 配置
# --------------------
SFT_CHECKPOINT = "./checkpoint/sft"
PPO_CHECKPOINT = "./checkpoint/ppo"
RM_CHECKPOINT = "./checkpoint/rm"
DATA_REPO = "Karsh-CAI/btfChinese-DPO-small"
DATA_CACHE_DIR = "./checkpoint/data/"
SYSTEM_PROMPT = ""
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
MAX_NEW_TOKENS = 256
MAX_LENGTH = 512
NUM_SAMPLES = 50  # 从数据集中抽取的样本数
OUTPUT_CSV = "./benchmark_results.csv"

# 常见测试问题
COMMON_QUESTIONS = [
    "你好",
    "天气怎么样",
    "吃饭了吗",
    "今天星期几",
    "你是谁",
    "能帮我个忙吗",
    "最近怎么样",
    "你在干什么",
    "现在几点了",
    "你会说英文吗"
]

# --------------------
# 工具函数
# --------------------
def clear_gpu_memory():
    """清理GPU内存"""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def generate_response(model, tokenizer, question, max_new_tokens=MAX_NEW_TOKENS):
    """生成模型回复"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # 提取生成的部分
    response_ids = outputs[0][inputs.input_ids.shape[1]:]
    response = tokenizer.decode(response_ids, skip_special_tokens=True)
    return response

def score_response(rm_model, tokenizer, question, response):
    """使用RM模型对回复进行打分"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
        {"role": "assistant", "content": response}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to("cuda")
    
    with torch.no_grad():
        outputs = rm_model(**inputs)
        score = outputs.logits[0].item()
    
    return score

# --------------------
# 主评估流程
# --------------------
def main():
    print("="*50)
    print("开始批量评估")
    print("="*50)
    
    # 准备数据集
    print(f"\n加载数据集: {DATA_REPO}")
    dataset = load_dataset(DATA_REPO, split="train", cache_dir=DATA_CACHE_DIR, token=HF_TOKEN)
    
    # 随机抽取样本
    if NUM_SAMPLES and NUM_SAMPLES < len(dataset):
        import random
        indices = random.sample(range(len(dataset)), NUM_SAMPLES)
        test_questions = [dataset[i]["question"] for i in indices]
    else:
        test_questions = [sample["question"] for sample in dataset]
    
    print(f"将测试 {len(test_questions)} 个数据集问题 + {len(COMMON_QUESTIONS)} 个常见问题")
    
    # 合并所有问题
    all_questions = test_questions + COMMON_QUESTIONS
    
    # 存储结果
    results = []
    
    # --------------------
    # 阶段1: SFT模型推理
    # --------------------
    print(f"\n[1/3] 加载SFT模型: {SFT_CHECKPOINT}")
    sft_model = AutoModelForCausalLM.from_pretrained(
        SFT_CHECKPOINT,
        device_map="cuda",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )
    sft_tokenizer = AutoTokenizer.from_pretrained(SFT_CHECKPOINT, trust_remote_code=True)
    if sft_tokenizer.pad_token_id is None:
        sft_tokenizer.pad_token = sft_tokenizer.eos_token
    sft_model.eval()
    
    print("生成SFT回复...")
    sft_responses = []
    for question in tqdm(all_questions, desc="SFT推理"):
        response = generate_response(sft_model, sft_tokenizer, question)
        sft_responses.append(response)
    
    # 卸载SFT模型
    del sft_model
    del sft_tokenizer
    clear_gpu_memory()
    print("SFT模型已卸载")
    
    # --------------------
    # 阶段2: PPO模型推理
    # --------------------
    print(f"\n[2/3] 加载PPO模型: {PPO_CHECKPOINT}")
    ppo_peft_model = AutoPeftModelForCausalLM.from_pretrained(
        PPO_CHECKPOINT,
        device_map="cuda",
        torch_dtype=torch.bfloat16
    )
    ppo_model = ppo_peft_model.merge_and_unload()
    ppo_tokenizer = AutoTokenizer.from_pretrained(PPO_CHECKPOINT, trust_remote_code=True)
    if ppo_tokenizer.pad_token_id is None:
        ppo_tokenizer.pad_token = ppo_tokenizer.eos_token
    ppo_model.eval()
    
    print("生成PPO回复...")
    ppo_responses = []
    for question in tqdm(all_questions, desc="PPO推理"):
        response = generate_response(ppo_model, ppo_tokenizer, question)
        ppo_responses.append(response)
    
    # 卸载PPO模型
    del ppo_model
    del ppo_peft_model
    del ppo_tokenizer
    clear_gpu_memory()
    print("PPO模型已卸载")
    
    # --------------------
    # 阶段3: RM模型打分
    # --------------------
    print(f"\n[3/3] 加载RM模型: {RM_CHECKPOINT}")
    rm_model = AutoModelForSequenceClassification.from_pretrained(
        RM_CHECKPOINT,
        num_labels=1,
        device_map="cuda",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )
    rm_tokenizer = AutoTokenizer.from_pretrained(RM_CHECKPOINT, trust_remote_code=True)
    if rm_tokenizer.pad_token_id is None:
        rm_tokenizer.pad_token = rm_tokenizer.eos_token
    rm_model.eval()
    
    print("计算RM分数...")
    for i, question in enumerate(tqdm(all_questions, desc="RM打分")):
        sft_score = score_response(rm_model, rm_tokenizer, question, sft_responses[i])
        ppo_score = score_response(rm_model, rm_tokenizer, question, ppo_responses[i])
        
        results.append({
            "question": question,
            "sft_response": sft_responses[i],
            "ppo_response": ppo_responses[i],
            "sft_rm_score": sft_score,
            "ppo_rm_score": ppo_score,
            "score_diff": ppo_score - sft_score
        })
    
    # 卸载RM模型
    del rm_model
    del rm_tokenizer
    clear_gpu_memory()
    print("RM模型已卸载")
    
    # --------------------
    # 保存结果到CSV
    # --------------------
    print(f"\n保存结果到: {OUTPUT_CSV}")
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['question', 'sft_response', 'ppo_response', 'sft_rm_score', 'ppo_rm_score', 'score_diff']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    # --------------------
    # 输出常见问题结果到控制台
    # --------------------
    print("\n" + "="*80)
    print("常见问题测试结果")
    print("="*80)
    
    # 提取常见问题的结果
    common_results = results[-len(COMMON_QUESTIONS):]
    
    for i, result in enumerate(common_results, 1):
        print(f"\n[问题 {i}] {result['question']}")
        print("-"*40)
        print(f"SFT回复: {result['sft_response']}")
        print(f"SFT得分: {result['sft_rm_score']:.4f}")
        print("-"*20)
        print(f"PPO回复: {result['ppo_response']}")
        print(f"PPO得分: {result['ppo_rm_score']:.4f}")
        print("-"*20)
        print(f"得分差异(PPO-SFT): {result['score_diff']:.4f}")
    
    # --------------------
    # 统计信息
    # --------------------
    print("\n" + "="*80)
    print("统计信息")
    print("="*80)
    
    # 计算平均分
    avg_sft_score = sum(r['sft_rm_score'] for r in results) / len(results)
    avg_ppo_score = sum(r['ppo_rm_score'] for r in results) / len(results)
    
    # PPO获胜的比例
    ppo_wins = sum(1 for r in results if r['ppo_rm_score'] > r['sft_rm_score'])
    win_rate = ppo_wins / len(results) * 100
    
    print(f"总测试样本数: {len(results)}")
    print(f"SFT平均得分: {avg_sft_score:.4f}")
    print(f"PPO平均得分: {avg_ppo_score:.4f}")
    print(f"平均得分提升: {avg_ppo_score - avg_sft_score:.4f}")
    print(f"PPO获胜率: {win_rate:.2f}% ({ppo_wins}/{len(results)})")
    
    print("\n" + "="*80)
    print(f"评估完成！结果已保存到 {OUTPUT_CSV}")
    print("="*80)

if __name__ == "__main__":
    main()