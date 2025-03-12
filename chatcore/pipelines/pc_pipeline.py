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
            "condition": "{{'visual' in messages[0].text.lower()}}",
            "output": "{{messages[0]}}",
            "output_name": "pc_visual_tool_calls",
            "output_type": ChatMessage,  # Use direct type
        },
        {
            "condition": "{{'visual' not in messages[0].text.lower()}}",
            "output": "{{messages}}",
            "output_name": "no_tool_calls",
            "output_type": List[ChatMessage],  # Use direct type
        },
    ]

    # Initialize the ConditionalRouter
    router = ConditionalRouter(routes, unsafe=True)

    # Initialize the ToolInvoker with the weather tool
    pc_tool_checker = PcToolCallAssistant()
    tool_invoker = ToolInvoker(tools=[pc_visual_tool])

    # Create the pipeline
    pipeline = Pipeline()
    #pipeline.add_component("generator", llm_chat)
    pipeline.add_component("router", router)
    pipeline.add_component("tool_checker", pc_tool_checker)
    pipeline.add_component("tool_invoker", tool_invoker)

    # Connect components
    #pipeline.connect("generator.replies", "router")
    pipeline.connect("router.pc_visual_tool_calls", "tool_checker.message")  
    pipeline.connect("tool_checker.helper_messages", "tool_invoker.messages")  
    #pipeline.connect("router.no_tool_calls", "generator") # Correct connection

    # Critical connection: Feed tool results back to generator
    #pipeline.connect("tool_invoker.tool_messages", "generator")  # Add this line

    return pipeline


if __name__ == "__main__":
    # Example user message
    pc_pipe = create_pc_pipeline()    
    user_message = ChatMessage.from_user(f"Visualize the point cloud")
    #user_message = ChatMessage.from_user("Where is Helsinki?")
    # Run the pipeline
    result = pc_pipe.run({"messages": [user_message]})
    print(result)
    

