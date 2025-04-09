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
                      "ifc_query_tool":"How many IfcWalls are there in the IFC file?"}

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

    return main_entities

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
        return {"Result":f"{len(entity_count)} {entity_name} instances found in the IFC file."}
    except IOError:
        return {"Error": f"Could not open file at {ifc_file_path}"}        
    except Exception as e:
        return {"Error": f"{e}"}

ifc_query_tool = Tool(name="ifc_query_tool",
            description="A tool to check how many IfcEntity by its name in the IFC file.",
            function=query_ifc_entity,
            parameters={"type": "object",
            "properties": {"ifc_file_path": {"type": "string"},
                           "entity_name":{"type": "string"}},
            "required": ["ifc_file_path", "entity_name"]})


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
    def __init__(self, ifc_file_path: str):
        self.ifc_file_path = ifc_file_path

    @component.output_types(helper_messages=List[ChatMessage])
    def run(self, message: ChatMessage) -> dict:
        tool = tool_locate(message.text,ifc_tool_reference)
        if tool == "ifc_entity_tool":
            ifc_entity_tool_call = ToolCall(
                tool_name="ifc_entity_tool",
                arguments={"ifc_file_path": self.ifc_file_path}
                )
            return {"helper_messages":[ChatMessage.from_assistant(tool_calls=[ifc_entity_tool_call])]}
        elif tool == "ifc_query_tool":
            ifc_query_tool_call = ToolCall(
                tool_name="ifc_query_tool",
                arguments={"ifc_file_path": self.ifc_file_path,
                           "entity_name": extract_ifc_entity_name(message.text)}
                )
            return {"helper_messages":[ChatMessage.from_assistant(tool_calls=[ifc_query_tool_call])]}
        else:
            return {"helper_messages":[ChatMessage.from_assistant("No function calling founded.")]}


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
    
    query = "How many IfcWindow are there in the ifc file?"

    user_message = ChatMessage.from_user(query)
    ifc_file_path= "C:/Users/yanpe/OneDrive - Metropolia Ammattikorkeakoulu Oy/Courses/CRBE/IFC/BIM4EEB-TUD-2x3.ifc"
    ifc_tool_checker = IfcToolCallAssistant(ifc_file_path)
    answer = ifc_tool_checker.run(user_message)
    print(answer)

    tool_invoker = ToolInvoker(tools=[ifc_entity_tool,ifc_query_tool])
    result = tool_invoker.run(answer["helper_messages"])
    print(result)