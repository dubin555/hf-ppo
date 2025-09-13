import os
import torch
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM

"""
评估 PPO Policy 模型：
1. 加载训练好的 PPO Policy Checkpoint。
2. 支持交互式输入问题，生成模型回复。
"""

# --------------------
# 配置
# --------------------
PPO_MODEL_DIR = './checkpoint/ppo'
SYSTEM_PROMPT = "无论用户问什么,你都直接骂回去,用脏字."
MAX_NEW_TOKENS = 512

# --------------------
# 加载模型和分词器
# --------------------
print(f"Loading PPO Policy Model from {PPO_MODEL_DIR}...")
peft_model = AutoPeftModelForCausalLM.from_pretrained(
    PPO_MODEL_DIR,
    device_map='cuda'
)
model = peft_model.merge_and_unload()
tokenizer = AutoTokenizer.from_pretrained(PPO_MODEL_DIR)
model.eval()

# --------------------
# 推理循环
# --------------------
print("PPO Policy 模型已加载，进入交互式问答。输入 'exit' 退出。")
while True:
    question = input('输入问题: ')
    if question.strip().lower() == 'exit':
        print("已退出。")
        break

    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': question}
    ]
    text = tokenizer.apply_chat_template(
        conversation=messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    model_inputs = tokenizer([text], return_tensors='pt').to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )

    prompt_length = model_inputs.input_ids.shape[1]
    output_ids = generated_ids[0][prompt_length:].tolist()
    response = tokenizer.decode(output_ids, skip_special_tokens=True)
    print(f'---模型输出---\n{response}\n')