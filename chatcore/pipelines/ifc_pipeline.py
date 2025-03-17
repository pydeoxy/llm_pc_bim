from haystack.dataclasses import ChatMessage
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

from chatcore.tools.ifc_tool import ifc_entity_tool, IfcToolCallAssistant


# Initialize the ChatGenerator
# Chat LLM    
'''
llm_chat = HuggingFaceLocalChatGenerator(
        model="meta-llama/Llama-3.2-3B-Instruct", #llm_config["model_name"],
        huggingface_pipeline_kwargs={
            "device_map": "auto",
            "torch_dtype": "float16"
        }
    )
'''    

#llm_chat.warm_up()

@component
class NoFunctionCall:

    @component.output_types(no_call_message=ChatMessage)
    def run(self, message: ChatMessage) -> dict:
        return {"no_call_message":[ChatMessage.from_assistant("No function calling founded.")]}

def create_ifc_pipeline(
    #ifc_file: Any,
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
            "condition": "{{'ifcentit' in messages[0].text.lower()}}",
            "output": "{{messages[0]}}",
            "output_name": "ifc_entity_tool_calls",
            "output_type": ChatMessage,  # Use direct type
        },
        {
            "condition": "{{'ifcentit' not in messages[0].text.lower()}}",
            "output": "{{messages[0]}}",
            "output_name": "no_tool_calls",
            "output_type": ChatMessage,  # Use direct type
        },
    ]

    # Initialize the ConditionalRouter
    router = ConditionalRouter(routes, unsafe=True)

    # Initialize the ToolInvoker with the weather tool
    ifc_tool_checker = IfcToolCallAssistant()
    tool_invoker = ToolInvoker(tools=[ifc_entity_tool])
    no_call_helper = NoFunctionCall()

    # Create the pipeline
    pipeline = Pipeline()
    #pipeline.add_component("generator", llm_chat)
    pipeline.add_component("router", router)
    pipeline.add_component("tool_checker", ifc_tool_checker)
    pipeline.add_component("tool_invoker", tool_invoker)
    pipeline.add_component("no_call_helper", no_call_helper)

    # Connect components
    #pipeline.connect("generator.replies", "router")
    pipeline.connect("router.ifc_entity_tool_calls", "tool_checker.message")  
    pipeline.connect("tool_checker.helper_messages", "tool_invoker.messages")  
    pipeline.connect("router.no_tool_calls", "no_call_helper.message") 

    # Critical connection: Feed tool results back to generator
    #pipeline.connect("tool_invoker.tool_messages", "generator")  # Add this line

    return pipeline


if __name__ == "__main__":
    # Example user message
    ifc_pipe = create_ifc_pipeline()
    #user_message = ChatMessage.from_user("What are the main ifcentities in the ifc file?")
    user_message = ChatMessage.from_user("Summarize the ifc file.")
    # Run the pipeline
    result = ifc_pipe.run({"messages": [user_message]})

    print(result)
    # Contents of tool calling
    #print(result['tool_invoker']['tool_messages'][0].tool_call_result.result)
    # Contents of no tool calling
    #print(result['no_call_helper']['no_call_message'][0].text)


