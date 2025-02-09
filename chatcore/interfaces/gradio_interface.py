import gradio as gr
import re
from haystack import Pipeline
from pathlib import Path

# Previous imports and components remain the same
# Add these new components:

def create_interface():
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        # Store paths in session state
        ifc_path = gr.State()
        pointcloud_path = gr.State()
        docs_folder = gr.State()
        
        # File selection section
        with gr.Row():
            with gr.Column():
                ifc_btn = gr.UploadButton(
                    "Select IFC Model",
                    file_types=[".ifc"],
                    file_count="single"
                )
                ifc_path_display = gr.Text(label="Selected IFC File")
                
                pc_btn = gr.UploadButton(
                    "Select Point Cloud",
                    file_types=[".ply", ".pcd"],
                    file_count="single"
                )
                pc_path_display = gr.Text(label="Selected Point Cloud")
                
                docs_btn = gr.UploadButton(
                    "Select Documents Folder",
                    file_count="directory"
                )
                docs_path_display = gr.Text(label="Selected Documents Folder")
        
        # Chat interface
        chat = gr.ChatInterface(
            fn=chat_response,
            additional_inputs=[ifc_path, pointcloud_path, docs_folder]
        )
        
        # File selection handlers
        def handle_ifc(file):
            path = file.name if file else None
            return path, path
        
        def handle_pc(file):
            path = file.name if file else None
            return path, path
        
        def handle_docs(folder):
            load_documents(folder)  # Load documents when folder is selected
            return folder, folder
        
        ifc_btn.upload(
            handle_ifc,
            inputs=ifc_btn,
            outputs=[ifc_path, ifc_path_display]
        )
        
        pc_btn.upload(
            handle_pc,
            inputs=pc_btn,
            outputs=[pointcloud_path, pc_path_display]
        )
        
        docs_btn.upload(
            handle_docs,
            inputs=docs_btn,
            outputs=[docs_folder, docs_path_display]
        )
        
    return demo

# Modified chat response function
def chat_response(message, history, ifc_path, pc_path, docs_folder):
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
    if docs_folder and ("project" in message.lower() or "document" in message.lower()):
        result = project_pipeline.run(query=message)
        return result["answers"][0] if result["answers"] else "No information found"
    
    # Fallback to web search
    return web_search.run(message)["documents"][0]

# Simplified pipeline setup
def main():
    demo = create_interface()
    demo.launch()

if __name__ == "__main__":
    main()