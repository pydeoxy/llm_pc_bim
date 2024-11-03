# general_handler.py
from model_loader import load_llama_model

model, tokenizer = load_llama_model()

def handle_general_question(question):
    inputs = tokenizer(question, return_tensors="pt")
    outputs = model.generate(**inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
