import gradio as gr
import re
from pathlib import Path
import sys
import os
from duckduckgo_api_haystack import DuckduckgoApiWebSearch
from haystack.components.generators import HuggingFaceLocalGenerator
from haystack.components.generators.chat import HuggingFaceLocalChatGenerator
from transformers import AutoTokenizer, AutoModelForCausalLM
from haystack.document_stores.in_memory import InMemoryDocumentStore

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from chatcore.utils.config_loader import load_llm_config
from chatcore.tools import ifc_tool, seg_tool
from chatcore.pipelines.doc_pipeline import create_doc_pipeline
from chatcore.tools.doc_processing import DocumentManager

from typing import List

from haystack.dataclasses import ChatMessage, Document
chat_history: List[ChatMessage] = []

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

doc_store = DocumentManager(r"C:\Users\yanpe\Documents\projects\llm_pc_bim\docs")
precessed_docs= doc_store.process_documents()

doc_pipe = create_doc_pipeline(
    precessed_docs,
    llm,
    web_search=DuckduckgoApiWebSearch(top_k=5)
    )

def response_docs(message, history):   
          
    result = doc_pipe.run({"text_embedder": {"text": message}, 
                        "prompt_builder": {"query": message}, 
                        "router": {"query": message}})
    llm_response = result["router"]["answer"]
    print(f"llm_response: {llm_response}")
    chat_history.append(ChatMessage.from_user(message))
    chat_history.append(ChatMessage.from_assistant(llm_response))

    return llm_response


with gr.Blocks() as demo:
    gr.Markdown("# 📁 Chat with Your Project Documents")
    
    #with gr.Row():
        #ifc_file_input = gr.File(label="Select Your IFC File")
        #pc_file_input = gr.File(label="Select Your Point Cloud File")
        #folder_input = gr.Textbox(label="Select Your Folder Path of Documents")
       
    
    gr.ChatInterface(
        response_docs,
        #additional_inputs=[folder_input],
        #examples=[
            # Each example must be a list with values for all inputs
        #    ["What's in the file?", None, "/path/to/your/folder"],
        #    ["List PDFs in the folder", None, "/path/to/your/docs"],
        #    ["Summarize the document", "example.txt", None]
        #]
    )

if __name__ == "__main__":
    demo.launch(inbrowser = True)


