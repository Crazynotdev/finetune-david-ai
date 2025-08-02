from transformers import AutoTokenizer, AutoModelForCausualLM
from peft import PeftModel
import torch

tokenizer = AutoTokenizer.from_pretained("tiiuae/falcon-7b",
