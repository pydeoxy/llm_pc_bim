from haystack.dataclasses import ChatMessage, ToolCall
from haystack.components.tools import ToolInvoker
from haystack.tools import Tool

# Tool definition
import ifcopenshell

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



if __name__ == '__main__':   

    ifc_entity_tool_call = ToolCall(
        tool_name="ifc_entity_tool",
        arguments={"ifc_file_path": "C:/Users/yanpe/Documents/projects/llm_pc_bim/tests/BIM4EEB-TUD-2x3.ifc"}
    )

    message = ChatMessage.from_assistant(tool_calls=[ifc_entity_tool_call])


    # ToolInvoker initialization and run
    invoker = ToolInvoker(tools=[ifc_entity_tool])
    result = invoker.run(messages=[message])

    print(result)
