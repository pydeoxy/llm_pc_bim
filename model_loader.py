# model_loader.py
from transformers import LlamaForCausalLM, LlamaTokenizer

def load_llama_model(model_name="meta-llama/Llama-3.1"):
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    model = LlamaForCausalLM.from_pretrained(model_name)
    return model, tokenizer
