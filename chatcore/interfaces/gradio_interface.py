import gradio as gr
import re
from pathlib import Path
import sys
import os
from duckduckgo_api_haystack import DuckduckgoApiWebSearch
from haystack.components.generators import HuggingFaceLocalGenerator
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from haystack import Pipeline

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from chatcore.utils.config_loader import load_llm_config
from chatcore.pipelines.doc_pipeline import create_doc_pipeline
from chatcore.tools.doc_processing import DocumentManager

'''
Add buttons to initialize pipelines and tool pipelinesf

'''
doc_pipe = Pipeline()

def update_config_and_index(ifc_file_input, pc_file_input, folder_input):
    """Update JSON config and process documents into Haystack's document store."""
    config = {}
    if os.path.exists("config/config.json"):
        with open("config.json", "r") as f:
            config = json.load(f)
    
    # Update file path (handle temporary uploads)
    if ifc_file_input is not None:
        config["ifc_file_path"] = ifc_file_input.name  # Store uploaded file's persistent path
    else:
        config.pop("ifc_file_path", None)

    if pc_file_input is not None:
        config["pc_file_path"] = pc_file_input.name  # Store uploaded file's persistent path
    else:
        config.pop("pc_file_path", None)
    
    # Update folder path
    if folder_input.strip():
        config["folder_path"] = folder_input.strip()
        # Process documents
        doc_pipe_start(config["folder_path"])
    else:
        config.pop("folder_path", None)
    
    # Save config
    with open("config/config.json", "w") as f:
        json.dump(config, f)    
    
    return None

def doc_pipe_start(folder_path):
    global doc_pipe
    llm_config = load_llm_config()
    
    llm = HuggingFaceLocalGenerator(
        model=llm_config["model_name"],
        huggingface_pipeline_kwargs={
            "device_map": llm_config["device_map"],
            "torch_dtype": llm_config["torch_dtype"],        
            #"model_kwargs": {"use_auth_token": llm_config["huggingface"]["use_auth_token"]}
        },
        generation_kwargs=llm_config["generation"]
    )

    llm.warm_up()

    doc_store = DocumentManager(folder_path)
    precessed_docs= doc_store.process_documents()

    doc_pipe = create_doc_pipeline(
        precessed_docs,
        llm,
        web_search=DuckduckgoApiWebSearch(top_k=5)
        )

# Load the Llama-3 model (corrected model ID)
model_id = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

def response_docs(query):    
    result = doc_pipe.run({"text_embedder": {"text": query}, "prompt_builder": {"query": query}, "router": {"query": query}})
    return result["router"]["answer"]

def response_ifc(ifc_path, query):
    return f'IFC file found from {ifc_path}. I will answer your question about the IFC file.'

def response_pc(pc_path, query):    
    return f'Point Cloud file found from {pc_path}. I will answer your question about the Point Cloud file.'

def generate_response(message, history, ifc_path, pc_path, folder_path):
    """Generate response with file/folder context"""     
    
    if ifc_path and 'ifc' in message.lower():
        answer = response_ifc(ifc_path, message)
    elif pc_path and 'point cloud' in message.lower():
        answer = response_pc(pc_path, message)   
    elif folder_path and 'project' in message.lower():
        answer = response_docs(message)
    else:
        answer = 'No file or documents related to the query.'
    
    system_prompt = """Analyze the following context:\n"""    
    system_prompt += f"[PRELIMINARY ANSWER TO USER'S QUERY]\n{answer}\n"
    
    
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

def create_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# 📁 Chat with Files & Folders")
        
        with gr.Row():
            ifc_file_input = gr.File(label="Select Your IFC File")
            pc_file_input = gr.File(label="Select Your Point Cloud File")
            folder_input = gr.Textbox(label="Select Your Folder Path of Documents")

        # Update config on input changes
        ifc_file_input.change(
            fn=update_config_and_index,
            inputs=[ifc_file_input, pc_file_input, folder_input],
            outputs=None
        )
        ifc_file_input.change(
            fn=update_config_and_index,
            inputs=[ifc_file_input, pc_file_input, folder_input],
            outputs=None
        )
        folder_input.change(
            fn=update_config_and_index,
            inputs=[ifc_file_input, pc_file_input, folder_input],
            outputs=None
        )
        
        gr.ChatInterface(
            generate_response,
            additional_inputs=[ifc_file_input, pc_file_input, folder_input],
            #examples=[
                # Each example must be a list with values for all inputs
            #    ["What's in the file?", None, "/path/to/your/folder"],
            #    ["List PDFs in the folder", None, "/path/to/your/docs"],
            #    ["Summarize the document", "example.txt", None]
            #]
        )
    return demo

if __name__ == "__main__":
    interface = create_interface()
    interface.launch(inbrowser = True)