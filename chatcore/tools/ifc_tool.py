from haystack.dataclasses import ChatMessage, ToolCall
from haystack.components.tools import ToolInvoker
from haystack.tools import Tool
from haystack import component
from typing import List
import ifcopenshell

import sys
import os
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from chatcore.utils.helpers import query_similarity
from chatcore.utils.config_loader import load_path_config

# Reference dictionary of tools and their possible corresponding queries
ifc_tool_reference = {"ifc_entity_tool":"List the main ifc entities of the IFC file.",
                      "ifc_query_tool":"How many IfcEntity are there in the IFC file?"}

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
        return {entity_name:len(entity_count)}
    except IOError:
        return {"Error": f"Could not open file at {ifc_file_path}"}        
    except Exception as e:
        return {"Error": f"{e}"}


ifc_query_tool = Tool(name="ifc_query_tool",
            description="A tool to query the number of an IfcEntity in the IFC file.",
            function=query_ifc_entity,
            parameters={"type": "object",
            "properties": {"ifc_file_path": {"type": "string"},
                           "entity_name":{"type": "string"}},
            "required": ["ifc_file_path", "entity_name"]})

# Load paths from shared JSON file
paths = load_path_config()

@component
class IfcToolCallAssistant:

    @component.output_types(helper_messages=List[ChatMessage])
    def run(self, message: ChatMessage) -> dict:
        if query_similarity(ifc_tool_reference["ifc_entity_tool"], message.text)>0.5:
            ifc_entity_tool_call = ToolCall(
                tool_name="ifc_entity_tool",
                arguments={"ifc_file_path": paths["ifc_file_path"]}
                )
            return {"helper_messages":[ChatMessage.from_assistant(tool_calls=[ifc_entity_tool_call])]}
        else:
            return {"helper_messages":[ChatMessage.from_assistant("No function calling founded.")]}



if __name__ == '__main__':   

    ifc_entity_tool_call = ToolCall(
        tool_name="ifc_entity_tool",
        arguments={"ifc_file_path": "C:/Users/yanpe/OneDrive - Metropolia Ammattikorkeakoulu Oy/Courses/CRBE/IFC/BIM4EEB-TUD-2x3.ifc"}
    )

    message = ChatMessage.from_assistant(tool_calls=[ifc_entity_tool_call])

    print(message)

    # ToolInvoker initialization and run
    invoker = ToolInvoker(tools=[ifc_entity_tool])
    result = invoker.run(messages=[message])

    print(result)
   
