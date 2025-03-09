from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from haystack import component
from haystack.dataclasses import ChatMessage, ToolCall
from typing import List
import re
from haystack.components.tools import ToolInvoker
from tool_test import ifc_entity_tool

'''
Try in pipeline
'''


tool_reference = {"ifc_entity_tool":"List the entities of an IFC file at ':/path/to/your/ifc/file/model.ifc'"}

def query_similarity(ref,query):
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight model
    embeddings = model.encode([ref, query])
    similarity_score = cosine_similarity(
        [embeddings[0]],  # Reference embedding
        [embeddings[1]]   # Comparison embedding
    )[0][0]

    return similarity_score

def extract_ifc_file_path(input_string):
    """
    Extracts the IFC file path from a string using a regular expression.

    Args:
        input_string: The string containing the file path.

    Returns:
        The extracted file path as a string, or None if no match is found.
    """
    match = re.search(r"[a-zA-Z]:[\\/].*\.ifc", input_string)
    if match:
        return match.group(0)
    else:
        return None

@component
class IfcToolCallAssistant:

    @component.output_types(messages=List[ChatMessage])
    def run(self, message: ChatMessage) -> dict:
        if query_similarity(tool_reference["ifc_entity_tool"], message.text)>0.6:
            ifc_entity_tool_call = ToolCall(
                tool_name="ifc_entity_tool",
                arguments={"ifc_file_path": extract_ifc_file_path(message.text)}
                )
            return ChatMessage.from_assistant(tool_calls=[ifc_entity_tool_call])
        else:
            return ChatMessage.from_assistant("No function calling founded.")

if __name__ == '__main__':
    reference = tool_reference["ifc_entity_tool"]
    question = "List the entities in the IFC file at 'C:/Users/yanpe/Documents/temp/Riihimaki.ifc'"
    query = ChatMessage.from_user(question)
    ifc_tool_checker = IfcToolCallAssistant()

    # ToolInvoker initialization and run
    invoker = ToolInvoker(tools=[ifc_entity_tool])
    result = invoker.run([ifc_tool_checker.run(query)])

    print(result)
