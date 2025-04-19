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

from chatcore.tools.ifc_tool import ifc_entity_tool, ifc_query_tool, IfcToolCallAssistant


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

    @component.output_types(pipe_message=ChatMessage)
    def run(self, message: ChatMessage) -> dict:
        return {"pipe_message":[ChatMessage.from_assistant("No function calling founded.")]}
    
@component
class PipeOutMessage:

    @component.output_types(pipe_message=ChatMessage)
    def run(self, messages: List[ChatMessage]) -> dict:
        return {"pipe_message":messages}

def create_ifc_pipeline(
    ifc_file_path: str,
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
            "condition": "{{'ifc' in messages[0].text.lower()}}",
            "output": "{{messages[0]}}",
            "output_name": "ifc_tool_calls",
            "output_type": ChatMessage,  # Use direct type
        },
        {
            "condition": "{{'ifc' not in messages[0].text.lower()}}",
            "output": "{{messages[0]}}",
            "output_name": "no_tool_calls",
            "output_type": ChatMessage,  # Use direct type
        },
    ]

    # Initialize the ConditionalRouter
    router = ConditionalRouter(routes, unsafe=True)

    # Initialize the ToolInvoker with the ifc tools
    ifc_tool_checker = IfcToolCallAssistant(ifc_file_path)
    tool_invoker = ToolInvoker(tools=[ifc_entity_tool,ifc_query_tool])
    no_call_helper = NoFunctionCall()
    ifc_out_message = PipeOutMessage()

    # Create the pipeline
    pipeline = Pipeline()
    #pipeline.add_component("generator", llm_chat)
    pipeline.add_component("router", router)
    pipeline.add_component("tool_checker", ifc_tool_checker)
    pipeline.add_component("tool_invoker", tool_invoker)
    pipeline.add_component("no_call_helper", no_call_helper)
    pipeline.add_component("ifc_out_message", ifc_out_message)

    # Connect components
    #pipeline.connect("generator.replies", "router")
    pipeline.connect("router.ifc_tool_calls", "tool_checker.message")  
    pipeline.connect("tool_checker.helper_messages", "tool_invoker.messages")  
    pipeline.connect("router.no_tool_calls", "no_call_helper.message") 
    pipeline.connect("tool_invoker.tool_messages", "ifc_out_message.messages") 

    # Critical connection: Feed tool results back to generator
    #pipeline.connect("tool_invoker.tool_messages", "generator")  # Add this line

    return pipeline


if __name__ == "__main__":
    import json
    # Example user message
    with open("config.json", "r") as f:
        config = json.load(f)
    
    ifc_file_path = config["ifc_file_path"]
    ifc_pipe = create_ifc_pipeline(ifc_file_path)

    # Visualizing the pipeline 
    #ifc_pipe.draw(path="docs/ifc_pipeline_diagram.png")

    #user_message = ChatMessage.from_user("What are the main ifcentities in the ifc file?")
    user_message = ChatMessage.from_user("How many IfcWindow are there in the IFC file?")
    #user_message = ChatMessage.from_user("Where is SmartLab?")
    # Run the pipeline
    result = ifc_pipe.run({"messages": [user_message]})

    print(result)
    # Contents of tool calling
    #print(result['tool_invoker']['tool_messages'][0].tool_call_result.result)
    # Contents of no tool calling
    #print(result['no_call_helper']['no_call_message'][0].text)


