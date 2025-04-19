from haystack.dataclasses import ChatMessage, ToolCall
from haystack.components.tools import ToolInvoker
from haystack.components.routers import ConditionalRouter
from haystack import Pipeline
from typing import List, Any
from haystack.components.generators import HuggingFaceLocalGenerator
from haystack.components.generators.chat import HuggingFaceLocalChatGenerator
from haystack import component

import sys
import os
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from chatcore.tools.pc_tool import pc_visual_tool, PcToolCallAssistant
from chatcore.pipelines.ifc_pipeline import NoFunctionCall,PipeOutMessage,QueryToMessage

def create_pc_pipeline(
    #pc_file: Any,
    #llm: Any,
    #web_search: Any
    ) -> Pipeline:
    """
    Creates and configures the document processing pipeline
    
    Args:
        document_manager: Initialized document store
        llm: Configured LLM generator
        web_search: Initialized web search component
        
    Returns:
        Configured Pipeline instance
    """


    # Define routing conditions
    routes = [
        {
            "condition": "{{'point' in messages[0].text.lower()}}",
            "output": "{{messages[0]}}",
            "output_name": "pc_tool_calls",
            "output_type": ChatMessage,  # Use direct type
        },
        {
            "condition": "{{'point' not in messages[0].text.lower()}}",
            "output": "{{messages[0]}}",
            "output_name": "no_tool_calls",
            "output_type": ChatMessage,  # Use direct type
        },
    ]

    # Initialize the ConditionalRouter
    router = ConditionalRouter(routes, unsafe=True)

    # Initialize the ToolInvoker with the weather tool
    query_to_message = QueryToMessage()
    pc_tool_checker = PcToolCallAssistant()
    tool_invoker = ToolInvoker(tools=[pc_visual_tool])
    no_call_helper = NoFunctionCall()
    pc_out_message = PipeOutMessage()

    # Create the pipeline
    pipeline = Pipeline()
    pipeline.add_component("query_to_message", query_to_message)
    pipeline.add_component("router", router)
    pipeline.add_component("tool_checker", pc_tool_checker)
    pipeline.add_component("tool_invoker", tool_invoker)
    pipeline.add_component("no_call_helper", no_call_helper)
    pipeline.add_component("pc_out_message", pc_out_message)

    # Connect components
    pipeline.connect("query_to_message.message", "router")
    pipeline.connect("router.pc_tool_calls", "tool_checker.message")  
    pipeline.connect("tool_checker.helper_messages", "tool_invoker.messages")  
    pipeline.connect("router.no_tool_calls", "no_call_helper.message") 
    pipeline.connect("tool_invoker.tool_messages", "pc_out_message.messages") 

    return pipeline


if __name__ == "__main__":
    # Example user message
    pc_pipe = create_pc_pipeline()    
    query = "Visualize the point cloud"
    #query = "Where is Finland?"
    #user_message = ChatMessage.from_user("How many points are there in the point cloud?")
    # Run the pipeline
    result = pc_pipe.run({"query": query})
    print(result)
    

