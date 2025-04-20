from haystack.dataclasses import ChatMessage, ToolCall
from haystack.components.tools import ToolInvoker
from haystack.tools import Tool
from haystack import component
from typing import List
import ifcopenshell
import re

import sys
import os
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from chatcore.utils.helpers import query_similarity
from chatcore.utils.config_loader import load_path_config

# Reference dictionary of tools and their possible corresponding queries
ifc_tool_reference = {"ifc_entity_tool":"List the main ifc entities of the IFC file.",
                      "ifc_query_tool":"How many IfcWalls are there in the IFC file?",
                      "no_call":"ifc file"}

# Tool to get main ifc entities
def get_main_ifc_entities(ifc_file_path: str):
    """
    Extracts main IFC entities (IfcProject, IfcSite, IfcBuilding, IfcBuildingStorey) 
    and their GUIDs from an IFC file.

    Args:
        ifc_file_path (str): The path to the IFC file.

    Returns:
        dict: A dictionary containing the main IFC entities and their GUIDs, 
              or None if the file cannot be opened or processed.
              The keys are the entity types, and the values are lists of GUIDs.
    """
    try:
        ifc_file = ifcopenshell.open(ifc_file_path)
    except IOError:
        print(f"Error: Could not open IFC file at {ifc_file_path}")
        return None
    
    main_entities = {
        "IfcProject": [],
        "IfcSite": [],
        "IfcBuilding": [],
        "IfcBuildingStorey": []
    }
    
    for entity_type in main_entities:
      for entity in ifc_file.by_type(entity_type):
        main_entities[entity_type].append(entity.GlobalId)

    return  f"""FROM FUNCTION CALL:
             {main_entities}"""

ifc_entity_tool = Tool(name="ifc_entity_tool",
            description="A tool to extracts main IFC entities and their GUIDs from an IFC file.",
            function=get_main_ifc_entities,
            parameters={"type": "object",
            "properties": {"ifc_file_path": {"type": "string"}},
            "required": ["ifc_file_path"]})

# Tool to get number of an IfcEntity by its name
def query_ifc_entity(ifc_file_path: str, entity_name: str):
    """
    Provide the number of an given IfcEntity in the IFC file.

    Args:
        ifc_file_path (str): The path to the IFC file.
        entity_name (str): The name of IfcEntity to be queried.

    Returns:
        dict: A dictionary containing the IfcEntity queried and its total number.
    """
    try:
        ifc_file = ifcopenshell.open(ifc_file_path)
        entity_count = ifc_file.by_type(entity_name)  # Directly get entities of specified type.
        return f"FROM FUNCTION CALL: {len(entity_count)} {entity_name} instances found in the IFC file."
    except IOError:
        return f"Error: Could not open file at {ifc_file_path}"        
    except Exception as e:
        return f"Error: {e}"

ifc_query_tool = Tool(name="ifc_query_tool",
            description="A tool to check how many IfcEntity by its name in the IFC file.",
            function=query_ifc_entity,
            parameters={"type": "object",
            "properties": {"ifc_file_path": {"type": "string"},
                           "entity_name":{"type": "string"}},
            "required": ["ifc_file_path", "entity_name"]})

# Tool to skip function calling
def no_call(query: str):
    """
    Skip function calling for queries not related to functions.

    Args:
        User's query.

    Returns:
        str: No functions founded.
    """ 
    #file_name = os.path.basename(ifc_file_path)      
    return f"no_answer: No functions found for query - '{query}'."

no_call_tool = Tool(name="no_call_tool",
            description="A tool to skip function calling for queries not related to functions.",
            function=no_call,
            parameters={"type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"]})

# Function of find the most possible tool
def tool_locate(query,tool_ref):
    tool_similarity = {}
    for key in tool_ref.keys():
        similarity = query_similarity(tool_ref[key], query)
        if similarity > 0.6:
            tool_similarity[key] = similarity

    tool = max(tool_similarity, key=tool_similarity.get)
    return tool

def extract_ifc_entity_name(query):    
    match = re.search(r"(Ifc\w+)", query, re.IGNORECASE)
    if match:
        return match.group(1)
    else:
        return None
    
# Helper component for choosing ifc tools
paths = load_path_config()

@component
class IfcToolCallAssistant:
    
    @component.output_types(helper_messages=List[ChatMessage])
    def run(self, message: ChatMessage) -> dict:
        tool = tool_locate(message.text,ifc_tool_reference)
        if tool == "ifc_entity_tool":
            ifc_entity_tool_call = ToolCall(
                tool_name="ifc_entity_tool",
                arguments={"ifc_file_path": paths["ifc_file_path"]}
                )
            return {"helper_messages":[ChatMessage.from_assistant(tool_calls=[ifc_entity_tool_call])]}
        elif tool == "ifc_query_tool":
            ifc_query_tool_call = ToolCall(
                tool_name="ifc_query_tool",
                arguments={"ifc_file_path": paths["ifc_file_path"],
                           "entity_name": extract_ifc_entity_name(message.text)}
                )
            return {"helper_messages":[ChatMessage.from_assistant(tool_calls=[ifc_query_tool_call])]}
        elif tool == "no_call":
            no_call_tool_call = ToolCall(
                tool_name="no_call_tool",
                arguments={"query": message.text}
                )
            return {"helper_messages":[ChatMessage.from_assistant(tool_calls=[no_call_tool_call])]}
        else:
            return {"helper_messages":[ChatMessage.from_assistant("No function calling found.")]}


if __name__ == '__main__':   

    ifc_entity_tool_call = ToolCall(
        tool_name="ifc_entity_tool",
        arguments={"ifc_file_path": "C:/Users/yanpe/OneDrive - Metropolia Ammattikorkeakoulu Oy/Courses/CRBE/IFC/BIM4EEB-TUD-2x3.ifc"}
    )

    ifc_query_tool_call = ToolCall(
        tool_name="ifc_query_tool",
        arguments={"ifc_file_path": "C:/Users/yanpe/OneDrive - Metropolia Ammattikorkeakoulu Oy/Courses/CRBE/IFC/BIM4EEB-TUD-2x3.ifc",
                   "entity_name": "IfcWall"}
    )

    message = ChatMessage.from_assistant(tool_calls=[ifc_query_tool_call])

    #print(message)

    # ToolInvoker initialization and run
    #invoker = ToolInvoker(tools=[ifc_query_tool])
    #result = invoker.run(messages=[message])

    #print(result)

    # Test tool calling
    #query = "What are the main ifcentities in the ifc file?"    
    #query = "How many IfcWindow are there in the ifc file?"
    query = "What is the IFC schema of the file?"
    print(query_similarity(ifc_tool_reference["ifc_entity_tool"], query))
    print(query_similarity(ifc_tool_reference["ifc_query_tool"], query))      
    print(query_similarity(ifc_tool_reference["no_call"], query))                     
                

    user_message = ChatMessage.from_user(query)
    ifc_file_path= "C:/Users/yanpe/OneDrive - Metropolia Ammattikorkeakoulu Oy/Courses/CRBE/IFC/BIM4EEB-TUD-2x3.ifc"
    ifc_tool_checker = IfcToolCallAssistant()
    answer = ifc_tool_checker.run(user_message)
    print(answer)

    tool_invoker = ToolInvoker(tools=[ifc_entity_tool,ifc_query_tool,no_call_tool])
    result = tool_invoker.run(answer["helper_messages"])
    print(result)