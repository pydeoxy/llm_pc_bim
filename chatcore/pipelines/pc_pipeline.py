from haystack.components.tools import ToolInvoker
from haystack import Pipeline

import sys
import os
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from chatcore.tools.pc_tool import pc_visual_tool, pc_seg_tool,no_call_tool,PcToolCallAssistant
from chatcore.pipelines.ifc_pipeline import ToolResult,QueryToMessage

def create_pc_pipeline(
    ) -> Pipeline:
    """
    Creates and configures the point cloud processing pipeline
         
    Returns:
        Configured Pipeline instance
    """

    # Initialize the ToolInvoker with the weather tool
    query_to_message = QueryToMessage()
    pc_tool_checker = PcToolCallAssistant()
    tool_invoker = ToolInvoker(tools=[pc_visual_tool,pc_seg_tool, no_call_tool])
    tool_result = ToolResult()

    # Create the pipeline
    pipeline = Pipeline()
    pipeline.add_component("query_to_message", query_to_message)
    pipeline.add_component("tool_checker", pc_tool_checker)
    pipeline.add_component("tool_invoker", tool_invoker)
    pipeline.add_component("tool_result", tool_result)

    # Connect components
    pipeline.connect("query_to_message.message", "tool_checker.message")  
    pipeline.connect("tool_checker.helper_messages", "tool_invoker.messages")  
    pipeline.connect("tool_invoker.tool_messages", "tool_result.messages") 

    return pipeline


if __name__ == "__main__":
    # Example user message
    pc_pipe = create_pc_pipeline()    

    # Visualizing the pipeline 
    #pc_pipe.draw(path="docs/pc_pipeline_diagram.png")

    # Tesing Q&A
    #query = "Visualize the point cloud."
    #query ="Perform semantic segmentation on the point cloud."
    query = "Label the points in the cloud with the semantic categories."
    #query = "Where is Finland?"
    #query="How many points are there in the point cloud?"
    # Run the pipeline
    result = pc_pipe.run({"query": query})
    print(result)
    

