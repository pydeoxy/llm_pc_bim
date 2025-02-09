import yaml
import torch
from typing import Dict, Any

def load_llm_config() -> Dict[str, Any]:
    with open("config/settings.yaml") as f:
        config = yaml.safe_load(f)
    
    llm_config = config["llm"]
    
    # Convert string dtype to actual torch dtype
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32
    }
    llm_config["torch_dtype"] = dtype_map[llm_config["torch_dtype"]]
    
    return llm_config