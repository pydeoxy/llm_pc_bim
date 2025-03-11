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

from chatcore.utils.helpers import extract_file_path, query_similarity

# Reference dictionary of tools and their possible corresponding queries
ifc_tool_reference = {"ifc_entity_tool":"List the ifc entities of an IFC file at 'D:/path/to/your/ifc/file/model.ifc'"}



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

# Tool to get main ifc entities



@component
class IfcToolCallAssistant:

    @component.output_types(helper_messages=List[ChatMessage])
    def run(self, message: ChatMessage) -> dict:
        if query_similarity(ifc_tool_reference["ifc_entity_tool"], message.text)>0.5:
            ifc_entity_tool_call = ToolCall(
                tool_name="ifc_entity_tool",
                arguments={"ifc_file_path": extract_file_path(message.text)}
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
   
