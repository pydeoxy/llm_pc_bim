from haystack import Pipeline
from haystack.components.routers import ConditionalRouter
from typing import Dict, Any

import sys
import os
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from chatcore.tools import ifc_tool, seg_tool

def create_main_pipeline(
    ifc_tool: Any,
    seg_tool: Any,
    doc_pipeline: Pipeline    
) -> Pipeline:
    """
    Creates and configures the main processing pipeline with routing logic
    
    Args:
        ifc_tool: Initialized IFC processing tool
        seg_tool: Initialized segmentation tool
        doc_pipeline: Configured document pipeline
                
    Returns:
        Configured Pipeline instance
    """
    # Define routing conditions
    router_conditions = [
        {
            "condition": "'.ifc' in {{query}} or 'ifc' in {{query|lower}}",
            "output": {"ifc_route": True},
            "output_name": "go_to_ifctool",
            "output_type": str,
        },
        {
            "condition": "any(ext in {{query}} for ext in ['.ply','.pcd']) "
                        "or 'segmentation' in {{query|lower}}",
            "output": {"seg_route": True},
            "output_name": "go_to_segtool",
            "output_type": str,
        },
        {
            "condition": "'project' in {{query|lower}} "
                        "or 'document' in {{query|lower}}",
            "output": {"doc_route": True},
            "output_name": "go_to_docpipeline",
            "output_type": str,
        }
    ]

    pipeline = Pipeline()
    pipeline.add_component("router", ConditionalRouter(router_conditions))
    pipeline.add_component("ifctool", ifc_tool)
    pipeline.add_component("segtool", seg_tool)
    pipeline.add_component("doc_pipeline", doc_pipeline)    

    # Connect components based on routing
    pipeline.connect("router.go_to_ifctool", "ifctool.get_info")
    pipeline.connect("router.go_to_segtool", "segtool.run")
    pipeline.connect("router.go_to_docpipeline", "doc_pipeline.query")

    return pipeline

if __name__ == "__main__":
    import sys
    import os
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from chatcore.tools.doc_processing import DocumentManager

    from chatcore.utils.config_loader import load_llm_config
    from duckduckgo_api_haystack import DuckduckgoApiWebSearch
    from haystack.components.generators import HuggingFaceLocalGenerator
    from doc_pipeline import create_doc_pipeline
    
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

    doc_store = DocumentManager("docs/")
    precessed_docs= doc_store.process_documents()

    doc_pipe = create_doc_pipeline(
        precessed_docs,
        llm,
        web_search=DuckduckgoApiWebSearch(top_k=5)
        )

    main_pipe = create_main_pipeline(
        ifc_tool,
        seg_tool,
        doc_pipeline=doc_pipe
    )

    # Visualizing the pipeline 
    main_pipe.draw(path="docs/main_pipeline_diagram.png")