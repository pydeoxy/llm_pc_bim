from chatcore.interfaces.gradio_interface import create_interface
from chatcore.utils.config_loader import load_llm_config
from haystack.components.generators import HuggingFaceLocalGenerator
from chatcore.pipelines.doc_pipeline import create_doc_pipeline

# Load LLM configuration
llm_config = load_llm_config()

# Initialize LLM with config
llm = HuggingFaceLocalGenerator(
    model=llm_config["model_name"],
    huggingface_pipeline_kwargs={
        "device_map": llm_config["device_map"],
        "torch_dtype": llm_config["torch_dtype"],
        "model_kwargs": {"use_auth_token": llm_config["huggingface"]["use_auth_token"]}
    },
    generation_kwargs=llm_config["generation"]
)

# Create document pipeline with configured LLM
doc_pipeline = create_doc_pipeline(
    document_store=document_store,
    web_search=web_search,
    llm=llm  # Pass the configured LLM instance
)

if __name__ == "__main__":
    interface = create_interface()
    interface.launch()