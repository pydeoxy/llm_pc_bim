import ifcopenshell
import re
from pathlib import Path


'''
Modify this with working similarity checking and tool calling
'''

class IFCTool:
    def __init__(self):
        self.loaded_files = {}
        
    def load_ifc_file(self, file_path):
        """Load and cache IFC files"""
        if file_path not in self.loaded_files:
            self.loaded_files[file_path] = ifcopenshell.open(file_path)
        return self.loaded_files[file_path]

    def get_info(self, file_path: str, entity_type: str = None):
        """Get general information about IFC entities"""
        ifc_file = self.load_file(file_path)
        results = []
        
        if entity_type:
            entities = ifc_file.by_type(entity_type)
            results.append(f"Found {len(entities)} {entity_type} entities")
        else:
            results.append(f"File contains {len(ifc_file)} entities")
            
        return {"answers": results}
    
    def run(self, file_path: str, query: str):
        # Determine which function to call based on query
        if "highlight" in query.lower():
            entity_id = re.search(r'#(\d+)', query).group(1)
            return self.highlight(file_path, entity_id)
        elif "info" in query.lower():
            entity_type = re.search(r'about (\w+)', query, re.I)
            return self.get_info(file_path, entity_type.group(1) if entity_type else None)
        else:
            return {"answers": ["Please specify what you want to do with the IFC file"]}