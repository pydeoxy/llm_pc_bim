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
from haystack.dataclasses import ChatMessage

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from chatcore.utils.config_loader import load_llm_config
from chatcore.pipelines.doc_pipeline import create_doc_pipeline
from chatcore.pipelines.ifc_pipeline import create_ifc_pipeline
from chatcore.pipelines.pc_pipeline import create_pc_pipeline
from chatcore.pipelines.main_pipeline import create_main_pipeline
from chatcore.tools.doc_processing import DocumentManager

'''
Add buttons to initialize pipelines and tool pipelines

'''
doc_pipe = Pipeline()
ifc_pipe = Pipeline()
pc_pipe = Pipeline()
main_pipe = Pipeline()

def update_config_and_index(ifc_file_input, pc_file_input, folder_input):
    """Update JSON config and process documents into Haystack's document store."""
    global ifc_pipe, pc_pipe
    config = {}   
    
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
    else:
        config.pop("folder_path", None)
    
    # Save config
    with open("config/config.json", "w") as f:
        json.dump(config, f)       
        
    return None

def doc_pipe_start(folder_path):
    global doc_pipe    

    doc_store = DocumentManager(folder_path)
    precessed_docs= doc_store.process_documents()

    doc_pipe = create_doc_pipeline(
        precessed_docs,        
        )

def main_pipe_start():
    global main_pipe,ifc_pipe, pc_pipe, doc_pipe
    if os.path.exists("config/config.json"):
        with open("config.json", "r") as f:
            config = json.load(f)  

    ifc_pipe = create_ifc_pipeline()
    pc_pipe = create_pc_pipeline()
    doc_pipe_start(config["folder_path"])

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

    main_pipe = create_main_pipeline(
        llm=llm,
        doc_pipeline=doc_pipe,
        ifc_pipeline=ifc_pipe,
        pc_pipeline=pc_pipe,
        web_search=DuckduckgoApiWebSearch(top_k=5)
    )

def generate_response(message,history):
    result = main_pipe.run({"query_router":{"query": message},"pipe_message_router":{"query":message}})
    return result['pipe_message_router']['answer']

def create_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# 📁 Chat with Files & Folders")
        
        with gr.Row():
            ifc_file_input = gr.File(label="Select Your IFC File (.ifc)")
            pc_file_input = gr.File(label="Select Your Point Cloud File (.pcd or .ply)")
            folder_input = gr.Textbox(label="Select Your Folder Path of Documents")

        # Update config on input changes
        ifc_file_input.change(
            fn=update_config_and_index,
            inputs=[ifc_file_input, pc_file_input, folder_input],
            outputs=None
        )
        pc_file_input.change(
            fn=update_config_and_index,
            inputs=[ifc_file_input, pc_file_input, folder_input],
            outputs=None
        )
        folder_input.change(
            fn=update_config_and_index,
            inputs=[ifc_file_input, pc_file_input, folder_input],
            outputs=None
        )

        main_pipe_start()
        
        gr.ChatInterface(
            generate_response,
            #additional_inputs=[ifc_file_input, pc_file_input, folder_input],
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

    '''
    def generate_response(message):
        result = main_pipe.run({"query_router":{"query": message},"pipe_message_router":{"query":message}})
        return result['pipe_message_router']['answer']

    main_pipe_start()
    #query = "What is IFC in the construction industry?"
    #query = "What is the capital of Finland?"
    #query = "What is the project SmartLab?"
    query= "How many IfcWindow are there in the IFC file?"
    #query= "What is ifc schema?"
    #query="How many points are there in the point cloud?"
    result = generate_response(query)

    print(result)
    '''
