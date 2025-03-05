from haystack.dataclasses import ChatMessage
from haystack.components.tools import ToolInvoker
from haystack.components.routers import ConditionalRouter
from haystack import Pipeline
from typing import List  # Ensure List is imported
from haystack.components.generators.chat import HuggingFaceLocalChatGenerator

# Import the tool
from tool_test import ifc_entity_tool


# Initialize the ToolInvoker with the weather tool
tool_invoker = ToolInvoker(tools=[ifc_entity_tool])

# Initialize the ChatGenerator
# Chat LLM    
llm_chat = HuggingFaceLocalChatGenerator(
        model="meta-llama/Llama-3.2-3B-Instruct", #llm_config["model_name"],
        huggingface_pipeline_kwargs={
            "device_map": "auto",
            "torch_dtype": "float16"
        },
        tools=[ifc_entity_tool]
    )

llm_chat.warm_up()


# Define routing conditions
routes = [
    {
        "condition": "{{'ifcentity' in messages}}",
        "output": "{{messages}}",
        "output_name": "ifc_entity_tool_calls",
        "output_type": List[ChatMessage],  # Use direct type
    },
    {
        "condition": "{{'ifcentity' not in messages}}",
        "output": "{{messages}}",
        "output_name": "no_tool_calls",
        "output_type": List[ChatMessage],  # Use direct type
    },
]

# Initialize the ConditionalRouter
router = ConditionalRouter(routes, unsafe=True)

# Create the pipeline
pipeline = Pipeline()
pipeline.add_component("generator", llm_chat)
pipeline.add_component("router", router)
pipeline.add_component("tool_invoker", tool_invoker)

# Connect components
#pipeline.connect("generator.replies", "router")
pipeline.connect("router.ifc_entity_tool_calls", "tool_invoker.messages")  
pipeline.connect("router.no_tool_calls", "generator") # Correct connection

# Critical connection: Feed tool results back to generator
#pipeline.connect("tool_invoker.tool_messages", "generator")  # Add this line


# Example user message
user_message = ChatMessage.from_user("List the ifcentity in 'C:/Users/yanpe/Documents/projects/llm_pc_bim/tests/BIM4EEB-TUD-2x3.ifc'")

# Run the pipeline
result = pipeline.run({"messages": [user_message]})


# Print the result
# "List the ifcentity in 'C:/Users/yanpe/Documents/projects/llm_pc_bim/tests/BIM4EEB-TUD-2x3.ifc'"



#print(tool_invoker.run(result['replies']['content']))
'''
user_messages = [
    ChatMessage.from_system(
        "Depending on the user's query, use the tool that you're provided with when compatible. "
    ),
    ChatMessage.from_user("Where is Helsinki?"),
]


response = llm_chat.run(messages=user_messages, tools=[ifc_entity_tool])

#result = tool_invoker.run([user_message])

'''
from pprint import pprint
pprint(result)