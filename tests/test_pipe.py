from haystack import Pipeline,component
from haystack.dataclasses import ChatMessage
from haystack.components.routers import ConditionalRouter
from haystack.components.builders import PromptBuilder
from haystack.components.joiners import BranchJoiner
from typing import Dict, Any, Annotated, Callable, Tuple

import sys
import os
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from duckduckgo_api_haystack import DuckduckgoApiWebSearch
from chatcore.utils.config_loader import load_llm_config
from chatcore.pipelines.doc_pipeline import create_doc_pipeline
from chatcore.pipelines.ifc_pipeline import create_ifc_pipeline
from chatcore.pipelines.pc_pipeline import create_pc_pipeline

from chatcore.utils.prompts import prompt_template_doc,prompt_template_after_websearch




def create_pipeline(
    llm: Any,
    doc_pipeline: Pipeline,
    web_search: Any
) -> Pipeline:
    
    @component
    class CommonQuery:

        @component.output_types(no_call_message=ChatMessage)
        def run(self, message: ChatMessage) -> dict:
            return {"no_call_message":[ChatMessage.from_assistant("Query not related to documents.")]}
        
    com_query = CommonQuery()
    
    # Define routing conditions
    query_conditions = [
        {
            "condition": "'ifc' in messages[0].text.lower()",
            "output": "{{messages}}",
            "output_name": "go_to_ifcpipeline",
            "output_type": str,
        },
        {
            "condition": "'point cloud' in messages[0].text.lower()",
            "output": "{{messages}}",
            "output_name": "go_to_pcpipeline",
            "output_type": str,
        },
        {
            "condition": "'project' in messages[0].text.lower()"
                        "or 'document' in messages[0].text.lower()",
            "output": "{{messages[0].text}}",
            "output_name": "go_to_docpipeline",
            "output_type": str,
        },       
    ]

    # Initialize the ConditionalRouter
    query_router = ConditionalRouter(query_conditions , unsafe=True)  
    
    prompt_builder_query = PromptBuilder(template=prompt_template_doc)
    prompt_joiner  = BranchJoiner(str)
    
    reply_routes = [
        {
            "condition": "{{'no_answer' in replies[0]}} or {{'No function' in replies[0]}}",
            "output": "{{query}}",
            "output_name": "go_to_websearch",
            "output_type": str,
        },
        {
            "condition": "{{'no_answer' not in replies[0]}} and {{'No function' in replies[0]}}",
            "output": "{{replies[0]}}",
            "output_name": "answer",
            "output_type": str,
        },
    ]

    reply_router = ConditionalRouter(reply_routes)
    
    @component
    class DocPipeline:
        """
        A component generating personal welcome message and making it upper case
        """
        @component.output_types(documents=Dict)
        def run(self, query:str) -> dict:
            responce = doc_pipeline.run({"text_embedder": {"text": query}})
            return {"documents":responce["retriever"]["documents"]}

    doc_pipe = DocPipeline()
    
    pipeline = Pipeline()

    pipeline.add_component("query_router", query_router)    
    pipeline.add_component("doc_pipe", doc_pipe)    
    #pipeline.add_component("prompt_builder_query", prompt_builder_query)
    #pipeline.add_component("prompt_joiner", prompt_joiner)
    #pipeline.add_component("llm", llm)
    #pipeline.add_component("reply_router", reply_router)
    
    pipeline.connect("query_router.go_to_docpipeline", "doc_pipe.query")
    #pipeline.connect("doc_pipe.documents", "prompt_builder_query.documents")
    #pipeline.connect("prompt_builder_query", "prompt_joiner")
    #pipeline.connect("prompt_joiner", "llm")

    #pipeline.connect("llm.replies", "reply_router.replies")

    return pipeline

if __name__ == "__main__":
    import sys
    import os
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from chatcore.tools.doc_processing import DocumentManager
    from chatcore.utils.config_loader import load_llm_config
    from duckduckgo_api_haystack import DuckduckgoApiWebSearch
    from haystack.components.generators import HuggingFaceLocalGenerator
    from chatcore.pipelines.doc_pipeline import create_doc_pipeline
    
    llm_config = load_llm_config()
    
    llm = HuggingFaceLocalGenerator(
        model=llm_config["model_name"],
        huggingface_pipeline_kwargs={
            "device_map": llm_config["device_map"],
            "torch_dtype": llm_config["torch_dtype"],        
            #"model_kwargs": {"use_auth_token": llm_config["huggingface"]["use_auth_token"]}
        },
        generation_kwargs=llm_config["generation"]
    )

    doc_store = DocumentManager("docs/")
    precessed_docs= doc_store.process_documents()

    doc_pipeline = create_doc_pipeline(
        precessed_docs,        
        )
    
    test_pipe = create_pipeline(
        llm=llm,
        doc_pipeline=doc_pipeline,
        web_search=DuckduckgoApiWebSearch(top_k=5)
    )

    # Visualizing the pipeline 
    #main_pipe.draw(path="docs/main_pipeline_diagram.png")

    
    query = "Where is the project smartLab?"
    #query = "Where is the project Helsinki?"
    #query = "How many IfcWindow are there in the IFC file?"
    
    user_message = ChatMessage.from_user(query)
    # Run the pipeline
    result = test_pipe.run({"messages": [user_message]})
    #print(user_message.text.lower())

    print(result)