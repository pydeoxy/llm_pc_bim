# classifier.py

def classify_question(question):
    if "IFC" in question or "building model" in question:
        return "ifc_task"
    elif "project document" in question or "report" in question:
        return "document_related"
    else:
        return "general"
