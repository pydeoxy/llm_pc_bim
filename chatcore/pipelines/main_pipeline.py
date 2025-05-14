from haystack import Pipeline,component
from haystack.components.generators import HuggingFaceLocalGenerator
from haystack.dataclasses import ChatMessage, Document
from haystack.components.routers import ConditionalRouter
from haystack.components.builders import PromptBuilder
from haystack.components.joiners import BranchJoiner
from typing import List, Dict, Any

import sys
import os
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from duckduckgo_api_haystack import DuckduckgoApiWebSearch
from chatcore.utils.config_loader import load_llm_config
from chatcore.pipelines.doc_pipeline import create_doc_pipeline
from chatcore.pipelines.ifc_pipeline import create_ifc_pipeline
from chatcore.pipelines.pc_pipeline import create_pc_pipeline

from chatcore.utils.prompts import prompt_template_doc,prompt_template_after_websearch,prompt_template_no_websearch
from chatcore.tools.doc_processing import DocumentManager

import logging
logger = logging.getLogger(__name__)

def create_main_pipeline(
    llm: Any,
    doc_pipeline: Pipeline,
    ifc_pipeline: Pipeline,
    pc_pipeline: Pipeline,
    web_search: Any
) -> Pipeline:
    """
    Creates and configures the main processing pipeline with routing logic
    
    Args:
        llm: HuggingFaceLocalGenerator or other LLMs
        ifc_pipeline: Configured ifc file pipeline
        pc_pipeline:  Configured point cloud pipeline
        doc_pipeline: Configured document pipeline
        web_search: Initialized web search component
                
    Returns:
        Configured Pipeline instance
    """

    prompt_builder_query = PromptBuilder(template=prompt_template_doc)
    prompt_joiner  = BranchJoiner(str)
    prompt_builder_after_websearch = PromptBuilder(template=prompt_template_after_websearch)


    # Define routing conditions
    query_conditions = [         
        {
            "condition": "{{'ifc' in query.lower()}}", 
            "output": "{{query}}",
            "output_name": "go_to_ifcpipeline",
            "output_type": str,
        },
        {
            "condition": "{{'point cloud' in query.lower()}}",
            "output": "{{query}}",
            "output_name": "go_to_pcpipeline",
            "output_type": str,
        },     
        {
            "condition": "{{'ifc' not in query.lower() and 'point cloud' not in query.lower()}}",
            "output": "{{query}}",
            "output_name": "go_to_docpipeline",
            "output_type": str,
        },         
    ]

    # Initialize the ConditionalRouter
    query_router = ConditionalRouter(query_conditions , unsafe=True)     

    reply_routes = [
        {
            "condition": "{{'no_answer' in replies[0]}}",
            "output": "{{replies[0]}}",
            "output_name": "value",
            "output_type": str,
        },
        {
            "condition": "{{'no_answer' not in replies[0]}}",
            "output": "{{replies[0]}}",
            "output_name": "value",
            "output_type": str,
        },
    ]

    reply_router = ConditionalRouter(reply_routes)

    pipe_message_routes = [         
        {
            "condition": "{{'no_answer' in value}}",
            "output": "{{query}}",
            "output_name": "go_to_websearch",
            "output_type": str,
        },       
        {
            "condition": "{{'no_answer' not in value}}",
            "output": "{{value}}",
            "output_name": "answer",
            "output_type": str,
        },
    ]

    pipe_message_router = ConditionalRouter(pipe_message_routes)
    
    @component
    class IfcPipeline:
        """
        A component generating personal welcome message and making it upper case
        """
        @component.output_types(pipe_message=str)
        def run(self, query:str):
            result = ifc_pipeline.run({"query": query})
            return  {"pipe_message":result["tool_result"]["tool_result"]}
    ifc_pipe = IfcPipeline()

    @component
    class PcPipeline:
        """
        A component generating personal welcome message and making it upper case
        """
        @component.output_types(pipe_message=str)
        def run(self, query:str) -> dict:
            result = pc_pipeline.run({"query": query})
            return  {"pipe_message":result["tool_result"]["tool_result"]}
    pc_pipe = PcPipeline()

    @component
    class DocPipeline:
        """
        A component generating personal welcome message and making it upper case
        """
        @component.output_types(documents=List[Document])
        def run(self, query:str) -> dict:
            responce = doc_pipeline.run({"text_embedder": {"text": query}})
            return {"documents":responce["retriever"]["documents"]}  
    doc_pipe = DocPipeline()   
    
    pipe_message_joiner = BranchJoiner(str)
    prompt_builder_after_websearch = PromptBuilder(template=prompt_template_after_websearch)  
    prompt_builder_no_websearch = PromptBuilder(template=prompt_template_no_websearch)   

    @component
    class SafeWebSearch:
        @component.output_types(documents=List[Document],error=bool,query=str)
        def run(self, query: str):
            try:
                res = web_search.run({"query": query})
                return {"documents": res["documents"], "error": False, "query":query}
            except Exception as e:
                logger.warning(f"Web‐search failed: {e}")
                return {"documents": [], "error": True, "query":query}
    safe_web_search = SafeWebSearch()

    web_search_conditions = [
        {"condition": "{{error}}", 
         "output": "{{query}}", 
         "output_name": "web_search_failed", 
         "output_type": str},

        {"condition": "{{not error}}", 
         "output": "{{documents}}", 
         "output_name": "web_search_success", 
         "output_type": List[Document]},
    ]
    web_search_router = ConditionalRouter(web_search_conditions, unsafe=True)

    pipeline = Pipeline()
    pipeline.add_component("query_router", query_router)
    pipeline.add_component("doc_pipe", doc_pipe)   
    pipeline.add_component("ifc_pipe", ifc_pipe)   
    pipeline.add_component("pc_pipe", pc_pipe) 
    pipeline.add_component("prompt_builder_query", prompt_builder_query)
    pipeline.add_component("prompt_joiner", prompt_joiner)
    pipeline.add_component("llm", llm)
    pipeline.add_component("reply_router", reply_router)  
    pipeline.add_component("pipe_message_joiner", pipe_message_joiner)
    pipeline.add_component("pipe_message_router", pipe_message_router)  
    pipeline.add_component("safe_web_search", safe_web_search)    
    pipeline.add_component("web_search_router", web_search_router)
    pipeline.add_component("prompt_builder_after_websearch", prompt_builder_after_websearch)
    pipeline.add_component("prompt_builder_no_websearch", prompt_builder_no_websearch)
    #pipeline.add_component("final_output", final_output)

    # Connect components based on routing        
    pipeline.connect("query_router.go_to_pcpipeline", "pc_pipe.query") 
    pipeline.connect("query_router.go_to_ifcpipeline", "ifc_pipe.query") 
    pipeline.connect("query_router.go_to_docpipeline", "doc_pipe.query")
    pipeline.connect("query_router.go_to_docpipeline", "prompt_builder_query.query")
    pipeline.connect("doc_pipe.documents", "prompt_builder_query.documents")
    pipeline.connect("prompt_builder_query", "prompt_joiner")
    pipeline.connect("prompt_joiner", "llm")
    pipeline.connect("llm.replies", "reply_router")
    pipeline.connect("reply_router.value", "pipe_message_joiner")
    pipeline.connect("ifc_pipe.pipe_message","pipe_message_joiner")
    pipeline.connect("pc_pipe.pipe_message","pipe_message_joiner")    
    pipeline.connect("pipe_message_joiner.value","pipe_message_router") 
    #pipeline.connect("pipe_message_router.answer", "final_output.answer") 
    
    
    # Web search handling
    pipeline.connect("pipe_message_router.go_to_websearch", "safe_web_search.query")
    pipeline.connect("safe_web_search.documents", "web_search_router.documents")
    pipeline.connect("safe_web_search.error", "web_search_router.error")
    pipeline.connect("safe_web_search.query", "web_search_router.query")

    pipeline.connect("web_search_router.web_search_success", "prompt_builder_after_websearch.documents")
    pipeline.connect("web_search_router.web_search_failed", "prompt_builder_no_websearch.query")

    pipeline.connect("web_search_router.web_search_success", "prompt_builder_after_websearch.documents")
    pipeline.connect("pipe_message_router.go_to_websearch", "prompt_builder_after_websearch.query")
    pipeline.connect("prompt_builder_after_websearch", "prompt_joiner")

    pipeline.connect("web_search_router.web_search_failed", "prompt_builder_no_websearch.query")
    pipeline.connect("prompt_builder_no_websearch", "prompt_joiner")
    
    return pipeline

# Test the pipeline
if __name__ == "__main__":
    from chatcore.utils.config_loader import load_llm_config
    from duckduckgo_api_haystack import DuckduckgoApiWebSearch    
    
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

    llm.warm_up()   

    doc_store = DocumentManager("docs/")
    precessed_docs= doc_store.process_documents()

    doc_pipe = create_doc_pipeline(
        precessed_docs,
        )
    
    ifc_pipe = create_ifc_pipeline()
    pc_pipe = create_pc_pipeline()

    main_pipe = create_main_pipeline(
        llm=llm,
        doc_pipeline=doc_pipe,
        ifc_pipeline=ifc_pipe,
        pc_pipeline=pc_pipe,
        web_search=DuckduckgoApiWebSearch(top_k=5)
    )

    # Visualizing the pipeline 
    #main_pipe.draw(path="docs/main_pipeline_diagram.png")
    
    #query = "What is the capital of Finland?"
    query = "What is SmartLab?"
    #query = "Who are involved in the project SmartLab?"
    #query= "How many IfcWindow are there in the IFC file?"
    #query= "What is ifc schema?"
    #query="How many points are there in the point cloud?"

    result = main_pipe.run({"query_router":{"query": query},"pipe_message_router":{"query":query}})#,"prompt_builder_query":{"query":query}})
    print(result)

   
