# ifc_handler.py
import ifcopenshell

def load_ifc_model(file_path):
    return ifcopenshell.open(file_path)

def handle_ifc_task(ifc_file_path, question):
    model = load_ifc_model(ifc_file_path)
    if "count walls" in question:
        walls = model.by_type("IfcWall")
        return f"There are {len(walls)} walls in the model."
    else:
        return "This IFC task is not currently supported."
