from transformers import AutoTokenizer, AutoModelForCausalLM 
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from trl import SFTTrainer, TrainingArguments import torch

Tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b", trust_remote_code=true)
model = AutoModelForCausualLM.from_pretrained("tiiuae/falcon-7b", torch_dtype=torch.float16, device_map="auto")

peft_config = LoraConfig(
  r=8,
  lora_alpha=16,
  target_modules=["query_key_value"],
  lora_dropout=0.1,
  bias="none",
  task_type="CAUSUAL_LM"
)

model= get_peft_model(model, peft_config)

dataset = load_dataset("json",
                      data_files="./data/dataset.json",
                       split="train")


def format(example):
  return {"text": f"instruction: {example['instruction']}\nResponse: {example['response']}"}

tokenized = dataset.map(format)

training_args = TrainingArguments(
  output_dir="model/adapter",
  per_device_train_batch_size=1,
  gradient_accumulation_steps=4,
  learning_rate=2e-4,
  num_train_epochs=3,
  save_strategy="epoch",
  logging_steps=10,
  bf16=true
)

trainer = SFTTrainer(
  model=model,
  train_dataset=tokenized,
  tokenizer=tokenizer,
  args=training_args
)
trainer.train()
