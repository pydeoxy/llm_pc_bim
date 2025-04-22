cot_template = """
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are a logical thinker that always explains your reasoning step-by-step.
    Follow this structure:
    1. Analyze the question carefully
    2. Break down into key components
    3. Consider different perspectives
    4. Evaluate evidence logically
    5. Conclude with final answer<|eot_id|>

    <|start_header_id|>user<|end_header_id|>
    Query: {{query}}<|eot_id|>

    <|start_header_id|>assistant<|end_header_id|>
    Let's think step by step:
    1. [First analysis step]
    2. [Breakdown of components]
    3. [Perspective consideration]
    4. [Logical evaluation]
    5. [Intermediate conclusion]

    Final Answer:
    """
prompt_template_doc = """
    <|begin_of_text|><|start_header_id|>user<|end_header_id|>
    Answer the following query from the given documents with the same language of the query.
    If the answer is contained within the documents, start the answer with "FROM THE KNOWLEDGE BASE: ".
    If the answer is not contained within the documents reply with 'no_answer'.   

    Documents:
    {% for document in documents %}
    {{document.content}}
    {% endfor %}

    Query: {{query}} <|eot_id|>
    """
prompt_template_after_websearch = """
    <|begin_of_text|><|start_header_id|>user<|end_header_id|>

    {% if documents and documents | length > 1 %}
    Answer the following query from the documents retrieved from the web.
    Start the answer with "FROM THE WEB: ".

    Documents:
    {% for document in documents %}
    {{ document.content }}
    {% endfor %}

    Query: {{ query }}<|eot_id|>

    <|start_header_id|>assistant<|end_header_id|>
    {% else %}
    Answer the following query using your own internal knowledge.
    Start the answer with "FROM LLM MODEL: ".

    Query: {{ query }}<|eot_id|>

    <|start_header_id|>assistant<|end_header_id|>
    {% endif %}
    """

prompt_template_no_websearch = """
    <|begin_of_text|><|start_header_id|>user<|end_header_id|>    
    Answer the following query using your own internal knowledge.
    Start the answer with "FROM LLM MODEL: ".

    Query: {{ query }}<|eot_id|>
    """
