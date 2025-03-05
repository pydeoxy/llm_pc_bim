import json
from typing import List

from haystack import Pipeline
from haystack import component
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

@component
class MessageProducer:

    @component.output_types(messages=List[ChatMessage])
    def run(self, messages: List[ChatMessage]) -> dict:
        return {"messages": messages}

ifc_path_schema = {"type": "object",
            "properties": {"ifc_file_path": {"type": "string", "pattern ": r"^.*\.ifc$" }},
            "required": ["ifc_file_path"]}

validator = JsonSchemaValidator(json_schema=ifc_path_schema)

#print(llm_chat.run(([ChatMessage.from_user("Create a json object of the file path 'C:/Users/yanpe/Documents/projects/llm_pc_bim/docs/BIM4EEB-TUD-2x3.ifc'with the format of {ifc_path_schema}" )])))


# Initialize a pipeline
pipe = Pipeline()

# Add components to the pipeline
pipe.add_component('branch_joiner', BranchJoiner(List[ChatMessage]))
pipe.add_component('llm', llm_chat)
pipe.add_component('schema_validator', JsonSchemaValidator(json_schema=ifc_path_schema))
#pipe.add_component('adapter', OutputAdapter("{{chat_message}}", List[ChatMessage]))
pipe.add_component("message_producer", MessageProducer())
# Connect components

pipe.connect("message_producer.messages", "branch_joiner")
pipe.connect("branch_joiner", "llm")
pipe.connect("llm.replies", "schema_validator.messages")
pipe.connect("schema_validator.validation_error", "branch_joiner")

result = pipe.run(
    data={"message_producer": {
        "messages": [ChatMessage.from_user("Create JSON from file path with key 'ifc_file_path' and value 'C:/Users/yanpe/Documents/projects/llm_pc_bim/docs/BIM4EEB-TUD-2x3.ifc'")]},
          "schema_validator": {"json_schema": ifc_path_schema}})


from pprint import pprint
pprint(result)
#print(json.loads(result["validator"]["validated"][0].content))

# Output:
# {'first_name': 'Peter', 'last_name': 'Parker', 'nationality': 'American', 'name': 'Spider-Man', 'occupation':
# 'Superhero', 'age': 23, 'location': 'New York City'}
