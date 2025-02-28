import gradio as gr
import re
from pathlib import Path
import sys
import os
from duckduckgo_api_haystack import DuckduckgoApiWebSearch
from haystack.components.generators import HuggingFaceLocalGenerator

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from chatcore.utils.config_loader import load_llm_config
from chatcore.tools import ifc_tool, seg_tool
from chatcore.pipelines.doc_pipeline import create_doc_pipeline
from chatcore.tools.doc_processing import DocumentManager

def response_docs(folder_path,query):
        
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

    def get_answer(query):
        result = doc_pipe.run({"text_embedder": {"text": query}, "prompt_builder": {"query": query}, "router": {"query": query}})
        print(result["router"]["answer"])
    
    return get_answer(query)

def create_interface():
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
                      
        # File selection section
        with gr.Row():
            with gr.Column():
                ifc_path = gr.Text(
                    label="IFC File",
                    show_label=True,
                    max_lines=1,
                    placeholder="Enter the path of your IFC file",
                    container=True,
                )

                pc_path = gr.Text(
                    label="Point Cloud File",
                    show_label=True,
                    max_lines=1,
                    placeholder="Enter the path of your point cloud file",
                    container=True,
                )

                docs_path = gr.Text(
                    label="Documents Folder",
                    show_label=True,
                    max_lines=1,
                    placeholder="Enter the path of your documents folder",
                    container=True,
                )                
                
            with gr.Column():
                # Chat interface
                chat = gr.ChatInterface(
                    fn=chat_response,
                    additional_inputs=[ifc_path, pc_path, docs_path],
                    type="messages",
                )
        
    return demo

# Modified chat response function
def chat_response(message, history, ifc_path, pc_path, docs_path):
    # Determine active file paths
    active_paths = {
        ".ifc": ifc_path,
        ".ply": pc_path,
        ".pcd": pc_path
    }
    
    # Check if query mentions any supported file types
    mentioned_files = {
        ext: re.search(r'\b\w+'+ext+r'\b', message)
        for ext in ['.ifc', '.ply', '.pcd']
    }
    
    # Use mentioned files or stored paths
    file_path = next((
        match.group() for match in mentioned_files.values() if match
    ), None) or next((
        path for ext, path in active_paths.items() 
        if ext in message.lower() and path
    ), None)
    
    # Route based on detected file type
    if file_path:
        ext = Path(file_path).suffix.lower()
        if ext == ".ifc" and any(w in message.lower() for w in ["ifc", "entity", "model"]):
            return ifc_tool.run(file_path, message)
        elif ext in [".ply", ".pcd"] and "segmentation" in message.lower():
            return seg_tool.run(file_path)
    
    # Document handling
    if docs_path and ("project" in message.lower() or "document" in message.lower()):
        result = response_docs(docs_path,message)
        return result["answers"][0] if result["answers"] else "No information found"
    


# Simplified pipeline setup
def main():
    demo = create_interface()
    demo.launch(inbrowser = True)

if __name__ == "__main__":
    main()