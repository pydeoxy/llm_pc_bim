from haystack.dataclasses import ChatMessage
from haystack.components.tools import ToolInvoker
from haystack.components.routers import ConditionalRouter
from haystack import Pipeline
from typing import List, Any
from haystack import component

import sys
import os
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from chatcore.tools.ifc_tool import ifc_entity_tool, ifc_query_tool,no_call_tool, IfcToolCallAssistant

@component
class QueryToMessage:

    @component.output_types(message=ChatMessage)
    def run(self, query: str) -> dict:
        return {"message":ChatMessage.from_user(query)}

@component
class NoFunctionCall:

    @component.output_types(no_call_answer=str)
    def run(self, message: ChatMessage) -> dict:
        return {"no_call_answer":"No function calling found."}
    
@component
class ToolResult:

    @component.output_types(tool_result=str)
    def run(self, messages: List[ChatMessage]) -> dict:
        return {"tool_result":messages[0].tool_call_result.result}

def create_ifc_pipeline(
    #ifc_file_path: str,
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

    '''
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
    '''

    # Initialize the ToolInvoker with the ifc tools
    query_to_message = QueryToMessage()
    ifc_tool_checker = IfcToolCallAssistant()
    tool_invoker = ToolInvoker(tools=[ifc_entity_tool,ifc_query_tool,no_call_tool])
    #no_call_helper = NoFunctionCall()
    tool_result = ToolResult()

    # Create the pipeline
    pipeline = Pipeline()
    pipeline.add_component("query_to_message", query_to_message)
    #pipeline.add_component("router", router)
    pipeline.add_component("tool_checker", ifc_tool_checker)
    pipeline.add_component("tool_invoker", tool_invoker)
    #pipeline.add_component("no_call_helper", no_call_helper)
    pipeline.add_component("tool_result", tool_result)

    # Connect components
    pipeline.connect("query_to_message.message", "tool_checker.message")  
    pipeline.connect("tool_checker.helper_messages", "tool_invoker.messages")  
    #pipeline.connect("router.no_tool_calls", "no_call_helper.message") 
    pipeline.connect("tool_invoker.tool_messages", "tool_result.messages") 

    return pipeline


if __name__ == "__main__":
    import json
    # Example user message
    
    ifc_pipe = create_ifc_pipeline()

    # Visualizing the pipeline 
    #ifc_pipe.draw(path="docs/ifc_pipeline_diagram.png")

    #user_message = ChatMessage.from_user("What are the main ifcentities in the ifc file?")
    #query= "How many IfcWindow are there in the IFC file?"
    query= "What is IFC?"
    #query = "Where is SmartLab?"
    # Run the pipeline
    result = ifc_pipe.run({"query": query})

    print(result)
    # Contents of tool calling
    #print(result['tool_invoker']['tool_messages'][0].tool_call_result.result)
    # Contents of no tool calling
    #print(result['no_call_helper']['no_call_message'][0].text)


