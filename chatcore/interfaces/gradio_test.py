import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

# Load the Llama-3 model (corrected model ID)
model_id = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

def process_paths(file_obj, folder_path):
    """Process uploaded file and directory path"""
    file_content = ""
    folder_contents = []
    
    if file_obj:
        try:
            with open(file_obj.name, 'r') as f:
                file_content = f.read(2000)
        except Exception as e:
            file_content = f"Error reading file: {str(e)}"
    
    if folder_path:
        try:
            folder_contents = os.listdir(folder_path)
        except Exception as e:
            folder_contents = [f"Error reading directory: {str(e)}"]
    
    return file_content, folder_contents

def generate_response(message, history, file_obj, folder_path):
    """Generate response with file/folder context"""
    file_content, folder_contents = process_paths(file_obj, folder_path)
    
    system_prompt = """Analyze the following context:\n"""
    if file_content:
        system_prompt += f"[FILE CONTENT]\n{file_content}\n"
    if folder_contents:
        system_prompt += f"[FOLDER CONTENTS]\n{', '.join(folder_contents)}\n"
    
    prompt = f"""<|begin_of_text|>
    <|start_header_id|>system<|end_header_id|>
    {system_prompt}<|eot_id|>
    <|start_header_id|>user<|end_header_id|>
    {message}<|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>"""
    
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.7)
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    return response.split("<|start_header_id|>assistant<|end_header_id|>")[-1].split("<|eot_id|>")[0].strip()

with gr.Blocks() as demo:
    gr.Markdown("# 📁 Chat with Files & Folders")
    
    with gr.Row():
        file_input = gr.File(label="Select File")
        folder_input = gr.Textbox(label="Folder Path")
    
    gr.ChatInterface(
        generate_response,
        additional_inputs=[file_input, folder_input],
        #examples=[
            # Each example must be a list with values for all inputs
        #    ["What's in the file?", None, "/path/to/your/folder"],
        #    ["List PDFs in the folder", None, "/path/to/your/docs"],
        #    ["Summarize the document", "example.txt", None]
        #]
    )

if __name__ == "__main__":
    demo.launch(inbrowser = True)