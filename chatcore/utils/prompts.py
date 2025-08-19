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
    Answer the following query from the documents retrieved from the web.
    Start the answer with "FROM THE WEB: ".

    Documents:
    {% for document in documents %}
    {{ document.content }}
    {% endfor %}

    Query: {{ query }}<|eot_id|>
    """

prompt_template_no_websearch = """
    <|begin_of_text|><|start_header_id|>user<|end_header_id|>    
    Answer the following query using your own internal knowledge.
    Start the answer with "FROM LLM MODEL: ".

    Query: {{ query }}<|eot_id|>
    """
