import open3d as o3d
import numpy as np
from haystack import component
from haystack.tools import Tool
from haystack.dataclasses import ChatMessage, ToolCall
from typing import List
from haystack.components.tools import ToolInvoker
import multiprocessing
import torch

import sys
import os
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

seg_dir = os.path.join(repo_root, "pc_seg")
if seg_dir not in sys.path:
    sys.path.insert(0, seg_dir)

from chatcore.utils.helpers import query_similarity
from chatcore.utils.config_loader import load_path_config
from chatcore.tools.ifc_tool import tool_locate,no_call_tool

from pc_seg.pc_seg_predict import prepare_dataset, run_segmentation
from pc_seg.ifc_sim_train import PRTRAINED_CKPT_PATH, SIM_CKPT_PATH

# Reference dictionary of tools and their possible corresponding queries
pc_tool_reference = {"pc_visual_tool":"Visualize the point cloud file.",
                     "pc_seg_tool":"Semantic segmentation of the point cloud file.",
                  "no_call":"point cloud"}

if os.path.isfile(SIM_CKPT_PATH):
    CHECKPOINT_FILE = SIM_CKPT_PATH
else:
    CHECKPOINT_FILE = PRTRAINED_CKPT_PATH

def visualize_point_cloud(pc_file_path):
    try:
        pcd = o3d.io.read_point_cloud(pc_file_path)
        o3d.visualization.draw_geometries([pcd], point_show_normal=False)
    except IOError:
        print(f"Error: Could not open point cloud file at {pc_file_path}")

def pc_visual(pc_file_path: str):
    if not os.path.exists(pc_file_path):
        return f"Error: File not found at {pc_file_path}"

    # Start visualization in a separate process
    vis_process = multiprocessing.Process(
        target=visualize_point_cloud, args=(pc_file_path,)
    )
    vis_process.start()
    
    # Return message immediately
    return "FROM FUNCTION CALL: Point cloud visualization is starting."

pc_visual_tool = Tool(name="pc_visual_tool",
            description="A tool to visualize a point cloud by its file path.",
            function=pc_visual,
            parameters={"type": "object",
            "properties": {"pc_file_path": {"type": "string"}},
            "required": ["pc_file_path"]})

def pc_seg(pc_file_path: str):    
    if not os.path.exists(pc_file_path):
        return f"Error: File not found at {pc_file_path}"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset, downpcd = prepare_dataset(pc_file_path)
    downpcd, message, save_path = run_segmentation(dataset, downpcd, device, CHECKPOINT_FILE)
    # Visualize the result
    pc_visual(save_path)
    # Return message immediately
    return f"FROM FUNCTION CALL: {message}"

pc_seg_tool = Tool(name="pc_seg_tool",
            description="A tool to perform semantic segmentation on a point cloud by its file path.",
            function=pc_seg,
            parameters={"type": "object",
            "properties": {"pc_file_path": {"type": "string"}},
            "required": ["pc_file_path"]})

# Load paths from shared JSON file
paths = load_path_config()

@component
class PcToolCallAssistant:

    @component.output_types(helper_messages=List[ChatMessage])
    def run(self, message: ChatMessage) -> dict:
        tool = tool_locate(message.text,pc_tool_reference)        
        if tool == "pc_visual_tool":        
            pc_visual_tool_call = ToolCall(
                tool_name="pc_visual_tool",
                arguments={"pc_file_path": paths["pc_file_path"]}
                )
            return {"helper_messages":[ChatMessage.from_assistant(tool_calls=[pc_visual_tool_call])]}
        elif tool == "pc_seg_tool":        
            pc_seg_tool_call = ToolCall(
                tool_name="pc_seg_tool",
                arguments={"pc_file_path": paths["pc_file_path"]}
                )
            return {"helper_messages":[ChatMessage.from_assistant(tool_calls=[pc_seg_tool_call])]}
        elif tool == "no_call":
            no_call_tool_call = ToolCall(
                tool_name="no_call_tool",
                arguments={"query": message.text}
                )
            return {"helper_messages":[ChatMessage.from_assistant(tool_calls=[no_call_tool_call])]}
        else:
            return {"helper_messages":[ChatMessage.from_assistant("No function calling found.")]}


if __name__ == '__main__':   
    '''
    pcd_path = "docs/SmartLab_2024_E57_Single_5mm.pcd"
    
    pc_seg_tool_call = ToolCall(
        tool_name="pc_seg_tool",
        arguments={"pc_file_path": pcd_path}
    )

    message = ChatMessage.from_assistant(tool_calls=[pc_visual_tool,pc_seg_tool_call, no_call_tool])    
    
    pc_visual_tool_call = ToolCall(
        tool_name="pc_visual_tool",
        arguments={"pc_file_path": pcd_path}
    )

    message = ChatMessage.from_assistant(tool_calls=[pc_visual_tool_call])     
    
    # ToolInvoker initialization and run
    invoker = ToolInvoker(tools=[pc_visual_tool,pc_seg_tool,no_call_tool])
    result = invoker.run(messages=[message])

    print(result)
    '''
    
    queries = ["Visualize the point cloud file.",
               "Render the point cloud file for viewing.",
               "Show the point cloud file visually.",
               "Perform semantic segmentation on the point cloud.",
               "Segment the point cloud into semantic classes.",
               "Label the points in the cloud with the semantic categories."]
    for q in queries:
        print(tool_locate(q,pc_tool_reference))      

    