import json
from typing import List

from haystack import Pipeline
from haystack.components.converters import OutputAdapter
from haystack.components.joiners import BranchJoiner
from haystack.components.validators import JsonSchemaValidator
from haystack.dataclasses import ChatMessage
from haystack.components.generators.chat import HuggingFaceLocalChatGenerator


# Chat LLM    
llm_chat = HuggingFaceLocalChatGenerator(
        model="meta-llama/Llama-3.2-3B-Instruct", #llm_config["model_name"],
        huggingface_pipeline_kwargs={
            "device_map": "auto",
            "torch_dtype": "float16"
        }
    )

llm_chat.warm_up()

ifc_path_schema = {"type": "object",
            "properties": {"ifc_file_path": {"type": "string", "pattern ": r"^.*\.ifc$" }},
            "required": ["ifc_file_path"]}


# Initialize a pipeline
pipe = Pipeline()

# Add components to the pipeline
pipe.add_component('joiner', BranchJoiner(List[ChatMessage]))
pipe.add_component('llm', llm_chat)
pipe.add_component('validator', JsonSchemaValidator(json_schema=ifc_path_schema))
pipe.add_component('adapter', OutputAdapter("{{chat_message}}", List[ChatMessage]))

# Connect components
pipe.connect("adapter", "joiner")
pipe.connect("joiner", "llm")
pipe.connect("llm.replies", "validator.messages")
pipe.connect("validator.validation_error", "joiner")

result = pipe.run(data={
    "llm": {"generation_kwargs": {"response_format": {"type": "json_object"}}},
    "adapter": {"chat_message": [ChatMessage.from_user("Create json object from file path C:/Users/yanpe/Documents/projects/llm_pc_bim/docs/BIM4EEB-TUD-2x3.ifc")]}
})

print(json.loads(result["validator"]["validated"][0].content))

# Output:
# {'first_name': 'Peter', 'last_name': 'Parker', 'nationality': 'American', 'name': 'Spider-Man', 'occupation':
# 'Superhero', 'age': 23, 'location': 'New York City'}
