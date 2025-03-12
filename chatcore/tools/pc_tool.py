import open3d as o3d
import numpy as np
from haystack import component
from haystack.tools import Tool
from haystack.dataclasses import ChatMessage, ToolCall
from typing import List
from haystack.components.tools import ToolInvoker
import threading

import sys
import os
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from chatcore.utils.helpers import extract_file_path, query_similarity
from chatcore.utils.config_loader import load_path_config

# Reference dictionary of tools and their possible corresponding queries
tool_reference = {"pc_visual_tool":"Visualize the point cloud file."}

def visualize_point_cloud(pcd):
    # This will open the Open3D visualization window in a blocking manner,
    # but since it's in a separate thread, it won't block the main thread.
    o3d.visualization.draw_geometries([pcd], point_show_normal=False)

def pc_visual(pc_file_path: str):
    """
    Visualize the point cloud file in a separate thread so the function
    can return a message immediately.

    Args:
        pc_file_path (str): The path to the point cloud file.
    """
    try:
        pcd = o3d.io.read_point_cloud(pc_file_path)
    except IOError:
        print(f"Error: Could not open point cloud file at {pc_file_path}")
        return None

    # Start the visualization in a separate thread
    vis_thread = threading.Thread(target=visualize_point_cloud, args=(pcd,))
    vis_thread.start()
    
    # Immediately return the final message
    return "Point cloud visualized"

pc_visual_tool = Tool(name="pc_visual_tool",
            description="A tool to visualize a point cloud by its file path.",
            function=pc_visual,
            parameters={"type": "object",
            "properties": {"pc_file_path": {"type": "string"}},
            "required": ["pc_file_path"]})

# Load paths from shared JSON file
paths = load_path_config()

@component
class PcToolCallAssistant:

    @component.output_types(helper_messages=List[ChatMessage])
    def run(self, message: ChatMessage) -> dict:
        if query_similarity(tool_reference["pc_visual_tool"], message.text)>0.5:
            pc_visual_tool_call = ToolCall(
                tool_name="pc_visual_tool",
                arguments={"pc_file_path": paths["pc_file_path"]}
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
    result = invoker.run(messages=[message])

    print(result)