# main.py
import gradio as gr
from classifier import classify_question
from ifc_handler import handle_ifc_task
from doc_handler import load_documents, answer_document_question
from general_handler import handle_general_question

def process_question(question, ifc_file=None, document_paths=[]):
    question_type = classify_question(question)
    
    if question_type == "ifc_task":
        if ifc_file:
            response = handle_ifc_task(ifc_file.name, question)
        else:
            response = "Please provide an IFC file for this question."
    elif question_type == "document_related":
        documents = load_documents(document_paths)
        response = answer_document_question(question, documents)
    else:
        response = handle_general_question(question)
    
    if not response:
        response = "Sorry, I don't know the answer to this question."
    
    return response

# Define the Gradio interface
iface = gr.Interface(
    fn=process_question,
    inputs=[
        gr.Textbox(lines=2, label="Enter your question"),
        gr.File(label="Upload IFC File (optional)"),
        gr.File(label="Upload Document Files (optional)", file_count="multiple")
    ],
    outputs="text",
    title="IFC Query System"
)

# Launch the Gradio interface
if __name__ == "__main__":
    iface.launch()
