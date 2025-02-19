from haystack import Pipeline
from haystack.components.routers import ConditionalRouter
from typing import Dict, Any
from tools import ifc_tool, seg_tool

def create_main_pipeline(
    ifc_tool: Any,
    seg_tool: Any,
    doc_pipeline: Pipeline,
    web_search: Any
) -> Pipeline:
    """
    Creates and configures the main processing pipeline with routing logic
    
    Args:
        ifc_tool: Initialized IFC processing tool
        seg_tool: Initialized segmentation tool
        doc_pipeline: Configured document pipeline
        web_search: Initialized web search component
        
    Returns:
        Configured Pipeline instance
    """
    # Define routing conditions
    router_conditions = [
        {
            "condition": "'.ifc' in {{query}} or 'ifc' in {{query|lower}}",
            "output": {"ifc_route": True}
        },
        {
            "condition": "any(ext in {{query}} for ext in ['.ply','.pcd']) "
                        "or 'segmentation' in {{query|lower}}",
            "output": {"seg_route": True}
        },
        {
            "condition": "'project' in {{query|lower}} "
                        "or 'document' in {{query|lower}}",
            "output": {"doc_route": True}
        }
    ]

    pipeline = Pipeline()
    pipeline.add_component("router", ConditionalRouter(conditions=router_conditions))
    pipeline.add_component("ifc_tool", ifc_tool)
    pipeline.add_component("seg_tool", seg_tool)
    pipeline.add_component("doc_pipeline", doc_pipeline)
    pipeline.add_component("web_search", web_search)

    # Connect components based on routing
    pipeline.connect("router.ifc_route", "ifc_tool")
    pipeline.connect("router.seg_route", "seg_tool")
    pipeline.connect("router.doc_route", "doc_pipeline")
    
    # Fallback connection for unanswered doc queries
    pipeline.connect("doc_pipeline.unanswered", "web_search")

    return pipeline

'''if __name__ == "__main__":
    pipe = create_main_pipeline(
        ifc_tool,
        seg_tool,
        doc_pipeline: Pipeline,
        web_search: Any
    )'''