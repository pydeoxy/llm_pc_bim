
def cot_prompt_template(question):
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are a logical thinker that always explains your reasoning step-by-step.
    Follow this structure:
    1. Analyze the question carefully
    2. Break down into key components
    3. Consider different perspectives
    4. Evaluate evidence logically
    5. Conclude with final answer<|eot_id|>

    <|start_header_id|>user<|end_header_id|>
    {question}<|eot_id|>

    <|start_header_id|>assistant<|end_header_id|>
    Let's think step by step:
    1. [First analysis step]
    2. [Breakdown of components]
    3. [Perspective consideration]
    4. [Logical evaluation]
    5. [Intermediate conclusion]

    Final Answer: """

def verify_cot_prompt_template(response):
    return f"""
    Verify and improve this answer:
    {response}

    Check for:
    - Logical consistency
    - Factual accuracy
    - Balanced perspective
    - Clear conclusion
    """

