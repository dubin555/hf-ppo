from huggingface_hub import whoami
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
import os

model_name = 'Qwen/Qwen3-0.6B'
cache_dir = './checkpoint/base/'

model = AutoModelForCausalLM.from_pretrained(model_name, device_map='cuda', cache_dir=cache_dir, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True)

dataset = load_dataset("Karsh-CAI/btfChinese-DPO-small", split="train", cache_dir="./checkpoint/data/")

def process_sample(x):
    question=x['question']
    chosen=x['chosen']
    return {
        'messages':[{'role':'system','content':'无论用户问什么,你都直接骂回去,用脏字.'},{'role':'user','content':question},{'role':'assistant','content':chosen}],
    }

dataset=dataset.map(process_sample).remove_columns(dataset.column_names).train_test_split(test_size=0.1,shuffle=False)

sft_config=SFTConfig(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    num_train_epochs=2,
    max_seq_length=500,
    learning_rate=1e-5,
    logging_steps=1,
    save_strategy='no',
    report_to='swanlab',
    output_dir='./checkpoint/sft'
)
trainer=SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=dataset['train'],
    tokenizer=tokenizer,
)
trainer.train()
trainer.save_model(sft_config.output_dir)