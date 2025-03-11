import open3d as o3d
import numpy as np
from haystack import component
from haystack.tools import Tool
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
from haystack.dataclasses import ChatMessage, ToolCall
from typing import List
from haystack.components.tools import ToolInvoker


def pc_visual(pc_file_path: str):
    """
    Visualize the point cloud file.

    Args:
        pc_file_path (str): The path to the point cloud file.
    
    """
    try:
        pcd = o3d.io.read_point_cloud(pc_file_path)
    except IOError:
        print(f"Error: Could not open point cloud file at {pc_file_path}")
        return None
    
    o3d.visualization.draw_geometries([pcd], point_show_normal=False)
    return "Point cloud visualized"

pc_visual_tool = Tool(name="pc_visual_tool",
            description="A tool to visualize a point cloud by its file path.",
            function=pc_visual,
            parameters={"type": "object",
            "properties": {"pc_file_path": {"type": "string"}},
            "required": ["pc_file_path"]})


tool_reference = {"pc_visual_tool":"Visualize the point cloud file at 'D:/path/to/your/ifc/file/pcd.ply'"}

def query_similarity(ref,query):
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight model
    embeddings = model.encode([ref, query])
    similarity_score = cosine_similarity(
        [embeddings[0]],  # Reference embedding
        [embeddings[1]]   # Comparison embedding
    )[0][0]

    return similarity_score

def extract_pc_file_path(input_string):
    """
    Extracts the point cloud file path from a string using a regular expression.

    Args:
        input_string: The string containing the file path.

    Returns:
        The extracted file path as a string, or None if no match is found.
    """
    if ".pcd" in input_string:
        match = re.search(r"[a-zA-Z]:[\\/].*\.pcd", input_string)
    elif ".ply" in input_string:
        match = re.search(r"[a-zA-Z]:[\\/].*\.ply", input_string)
    if match:
        return match.group(0)
    else:
        return None

@component
class PcToolCallAssistant:

    @component.output_types(helper_messages=List[ChatMessage])
    def run(self, message: ChatMessage) -> dict:
        if query_similarity(tool_reference["pc_visual_tool"], message.text)>0.5:
            pc_visual_tool_call = ToolCall(
                tool_name="pc_visual_tool",
                arguments={"pc_file_path": extract_pc_file_path(message.text)}
                )
            return {"helper_messages":[ChatMessage.from_assistant(tool_calls=[pc_visual_tool_call])]}
        else:
            return {"helper_messages":[ChatMessage.from_assistant("No function calling founded.")]}



if __name__ == '__main__':   

    pcd_path = "C:/Users/yanpe/OneDrive - Metropolia Ammattikorkeakoulu Oy/Research/data/smartlab/SmartLab_2024_E57_Single_5mm.pcd"

    pc_visual_tool_call = ToolCall(
        tool_name="pc_visual_tool",
        arguments={"pc_file_path": pcd_path}
    )

    message = ChatMessage.from_assistant(tool_calls=[pc_visual_tool_call])
  
    
    # ToolInvoker initialization and run
    invoker = ToolInvoker(tools=[pc_visual_tool])
    invoker.run(messages=[message])

    #print(result)