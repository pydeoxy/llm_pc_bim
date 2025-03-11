from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re

def query_similarity(ref,query):
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight model
    embeddings = model.encode([ref, query])
    similarity_score = cosine_similarity(
        [embeddings[0]],  # Reference embedding
        [embeddings[1]]   # Comparison embedding
    )[0][0]

    return similarity_score

def extract_file_path(input_string):
    """
    Extracts the file path of a point cloud or an IFC from a string using a regular expression.
    The supported extentions of files are '.pcd', '.ply' and '.ifc'

    Args:
        input_string: The string containing the file path.

    Returns:
        The extracted file path as a string, or None if no match is found.
    """
    if ".pcd" in input_string:
        match = re.search(r"[a-zA-Z]:[\\/].*\.pcd", input_string)
    elif ".ply" in input_string:
        match = re.search(r"[a-zA-Z]:[\\/].*\.ply", input_string)
    elif ".ifc" in input_string:
        match = re.search(r"[a-zA-Z]:[\\/].*\.ifc", input_string)
    if match:
        return match.group(0)
    else:
        return None





