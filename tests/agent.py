from haystack.nodes import PromptNode
prompt_node = PromptNode(model_name_or_path="meta-llama/Llama-3.2-3B-Instruct")

prompt_node("What is the capital of Germany?")