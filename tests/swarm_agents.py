from typing import Annotated, Callable, Tuple
from dataclasses import dataclass, field

import random, re

from haystack.dataclasses import ChatMessage, ChatRole
from haystack.tools import create_tool_from_function
from haystack.components.tools import ToolInvoker

from haystack.components.generators.chat import HuggingFaceLocalChatGenerator

import sys
import os
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from chatcore.utils.config_loader import load_llm_config
from duckduckgo_api_haystack import DuckduckgoApiWebSearch
    
llm_config = load_llm_config()
    
llm = HuggingFaceLocalChatGenerator(
        model="meta-llama/Llama-3.1-8B-Instruct", #llm_config["model_name"],
        huggingface_pipeline_kwargs={
            "device_map": llm_config["device_map"],
            "torch_dtype": llm_config["torch_dtype"],        
            #"model_kwargs": {"use_auth_token": llm_config["huggingface"]["use_auth_token"]}
        },
        #generation_kwargs=llm_config["generation"]
    )

llm.warm_up()

HANDOFF_TEMPLATE = "Transferred to: {agent_name}. Adopt persona immediately."
HANDOFF_PATTERN = r"Transferred to: (.*?)(?:\.|$)"

@dataclass
class SwarmAgent:
    name: str = "SwarmAgent"
    llm: object = llm
    instructions: str = "You are a helpful Agent"
    functions: list[Callable] = field(default_factory=list)

    def __post_init__(self):
        self._system_message = ChatMessage.from_system(self.instructions)
        self.tools = [create_tool_from_function(fun) for fun in self.functions] if self.functions else None
        self._tool_invoker = ToolInvoker(tools=self.tools, raise_on_failure=False) if self.tools else None

    def run(self, messages: list[ChatMessage]) -> Tuple[str, list[ChatMessage]]:
        # generate response
        agent_message = self.llm.run(messages=[self._system_message] + messages, tools=self.tools)["replies"][0]
        new_messages = [agent_message]

        if agent_message.text:
            print(f"\n{self.name}: {agent_message.text}")

        if not agent_message.tool_calls:
            return self.name, new_messages

        # handle tool calls
        for tc in agent_message.tool_calls:
            # trick: Ollama do not produce IDs, but OpenAI and Anthropic require them.
            if tc.id is None:
                tc.id = str(random.randint(0, 1000000))
        tool_results = self._tool_invoker.run(messages=[agent_message])["tool_messages"]
        new_messages.extend(tool_results)

        # handoff
        last_result = tool_results[-1].tool_call_result.result
        match = re.search(HANDOFF_PATTERN, last_result)
        new_agent_name = match.group(1) if match else self.name

        return new_agent_name, new_messages

# to automatically convert functions into tools, we need to annotate fields with their descriptions in the signature
def execute_ifc(item_name: Annotated[str, "The name of the model to print"]):
    return f"report: model succeeded for {item_name} - model id: {random.randint(0,10000)}"

def execute_docs(item_name: Annotated[str, "The name of the document to check"]):
    return f"report: document succeeded for {item_name} - document length: {random.randint(0,10000)}"

def transfer_to_ifc():
    """Pass to this Agent for anything related to IFC models"""
    return HANDOFF_TEMPLATE.format(agent_name="IFC Agent")

def transfer_to_docs():
    """Pass to this Agent for anything related to documents."""
    return HANDOFF_TEMPLATE.format(agent_name="Docs Agent")

def transfer_back_to_triage():
    """Call this if the user brings up a topic outside of your purview,
    including escalating to human."""
    return HANDOFF_TEMPLATE.format(agent_name="Triage Agent")

ifc_agent = SwarmAgent(
    name="IFC Agent",
    instructions=(
        "You are a IFC agent. "
        "Help the user with IFC models. "
        "Ask for basic information but be brief. "
        "For anything unrelated to IFC, transfer back to Triage Agent."
        "Make tool calls only if necessary and make sure to provide the right arguments."
    ),
    functions=[execute_ifc, transfer_back_to_triage],
)

docs_agent = SwarmAgent(
    name="Docs Agent",
    instructions=(
        "you are a assistant with questions about documents of projects. "
        "If the user asks questions related to IFC models, send him back to Triage Agent."
        "Make tool calls only if necessary and make sure to provide the right arguments."
    ),
    functions=[execute_docs, transfer_back_to_triage],
)

triage_agent = SwarmAgent(
    name="Triage Agent",
    instructions=(
        "You are a customer service bot for IFC and documents. "
        "Introduce yourself. Always be very brief. "
        "If the user asks general questions, try to answer them yourself without transferring to another agent. "
        "Only if the user has questions about IFC models, transfer to IFC Agent."
        "If the user has questions about documents of the project, transfer to Docs Agent."
        "Make tool calls only if necessary and make sure to provide the right arguments."
    ),
    functions=[transfer_to_ifc, transfer_to_docs],
)


agents = {agent.name: agent for agent in [triage_agent, ifc_agent, docs_agent]}

print("Type 'quit' to exit")

messages = []
current_agent_name = "Triage Agent"

while True:
    agent = agents[current_agent_name]

    if not messages or messages[-1].role == ChatRole.ASSISTANT:
        user_input = input("User: ")
        if user_input.lower() == "quit":
            break
        messages.append(ChatMessage.from_user(user_input))

    current_agent_name, new_messages = agent.run(messages)
    messages.extend(new_messages)


