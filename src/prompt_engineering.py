# src/prompt_engineering.py
"""
Day 8: Prompt engineering patterns for production AI systems
Every pattern you will use in Phase 2 projects
"""

from typing import Optional


# ─────────────────────────────────────────
# PATTERN 1: Zero / One / Few-shot
# ─────────────────────────────────────────

def build_few_shot_prompt(
    task_description: str,
    examples: list[dict],
    user_input: str
) -> str:
    """
    Build a few-shot prompt programmatically.
    examples: list of {"input": ..., "output": ...}
    """
    prompt = f"{task_description}\n\n"

    for i, ex in enumerate(examples, 1):
        prompt += f"Example {i}:\n"
        prompt += f"Input: {ex['input']}\n"
        prompt += f"Output: {ex['output']}\n\n"

    prompt += f"Now do the same:\nInput: {user_input}\nOutput:"
    return prompt


# ─────────────────────────────────────────
# PATTERN 2: Chain-of-Thought (CoT)
# ─────────────────────────────────────────

def build_cot_prompt(question: str, domain: str = "general") -> str:
    """
    Adds CoT instruction to any question.
    Forces intermediate reasoning → better accuracy.
    """
    cot_system = f"""You are an expert in {domain}. 
When answering, always:
1. Think through the problem step by step
2. Show your reasoning explicitly
3. Then give your final answer

Format:
REASONING: [your step-by-step thinking]
ANSWER: [final concise answer]"""

    return f"{cot_system}\n\nQuestion: {question}"


# ─────────────────────────────────────────
# PATTERN 3: Production RAG Prompt
# ─────────────────────────────────────────

def build_rag_prompt(
    context_chunks: list[str],
    user_question: str,
    system_role: str = "helpful AI assistant",
    output_format: Optional[str] = None
) -> dict:
    """
    Production-grade RAG prompt builder.
    Returns dict with 'system' and 'user' keys
    for direct use with any LLM API.
    """
    # System prompt: role + grounding constraint
    system = f"""You are a {system_role}.

CRITICAL INSTRUCTIONS:
1. Answer ONLY using the provided context below
2. If the answer is not in the context, say exactly: "I don't have enough information to answer that based on the provided context."
3. Never make up information
4. Cite which context number supports your answer"""

    if output_format:
        system += f"\n5. Respond in this format: {output_format}"

    # User message: context + question
    context_str = "\n\n".join(
        [f"[Context {i+1}]: {chunk}" for i, chunk in enumerate(context_chunks)]
    )

    user = f"""Context information:
{context_str}

Question: {user_question}"""

    return {"system": system, "user": user}


# ─────────────────────────────────────────
# PATTERN 4: Structured Output Prompt
# ─────────────────────────────────────────

def build_structured_output_prompt(
    task: str,
    input_text: str,
    output_schema: dict
) -> str:
    """
    Force model to output valid JSON matching a schema.
    Critical for programmatic use in your pipelines.
    """
    import json
    schema_str = json.dumps(output_schema, indent=2)

    return f"""Task: {task}

Input: {input_text}

Respond with ONLY valid JSON matching this exact schema:
{schema_str}

Do not include any text before or after the JSON. Do not use markdown code blocks."""


# ─────────────────────────────────────────
# PATTERN 5: Self-Critique / Reflection
# ─────────────────────────────────────────

def build_reflection_prompt(
    original_question: str,
    draft_answer: str
) -> str:
    """
    Ask model to critique and improve its own answer.
    Same pattern as Day 7 reflection agent, but as a prompt.
    """
    return f"""You generated this answer to a question. Now critique and improve it.

Original Question: {original_question}

Draft Answer:
{draft_answer}

Critique the draft answer:
1. Is it accurate?
2. Is it complete?
3. Is it concise?
4. Are there any errors or missing information?

Then provide an improved answer.

CRITIQUE: [your critique here]
IMPROVED ANSWER: [better answer here]"""


# ─────────────────────────────────────────
# DEMO: Show all patterns
# ─────────────────────────────────────────

def demo_all_patterns():
    print("=" * 60)
    print("PROMPT ENGINEERING PATTERNS DEMO")
    print("=" * 60)

    # 1. Few-shot
    print("\n--- PATTERN 1: Few-shot ---")
    few_shot = build_few_shot_prompt(
        task_description="Classify the sentiment of customer reviews as POSITIVE, NEGATIVE, or NEUTRAL.",
        examples=[
            {"input": "The product arrived quickly and works perfectly!", "output": "POSITIVE"},
            {"input": "Stopped working after 2 days. Very disappointed.", "output": "NEGATIVE"},
            {"input": "It's okay, does what it says.", "output": "NEUTRAL"},
        ],
        user_input="Delivery was late but the quality is good."
    )
    print(few_shot)

    # 2. CoT
    print("\n--- PATTERN 2: Chain-of-Thought ---")
    cot = build_cot_prompt(
        "If a RAG pipeline retrieves 5 chunks of 500 tokens each, and the system prompt is 200 tokens, what percentage of a 4096-token context window is used?",
        domain="AI engineering"
    )
    print(cot)

    # 3. RAG prompt
    print("\n--- PATTERN 3: Production RAG Prompt ---")
    rag = build_rag_prompt(
        context_chunks=[
            "LangGraph is built on top of LangChain and enables stateful workflows.",
            "StateGraph allows you to define nodes (functions) and edges (transitions).",
        ],
        user_question="How does LangGraph relate to LangChain?",
        system_role="senior AI engineering assistant"
    )
    print(f"SYSTEM:\n{rag['system']}\n")
    print(f"USER:\n{rag['user']}")

    # 4. Structured output
    print("\n--- PATTERN 4: Structured Output ---")
    structured = build_structured_output_prompt(
        task="Extract key information from this AI job description",
        input_text="Seeking an AI engineer with 2+ years LangChain experience, Python skills, and RAG knowledge. Remote OK. Salary: ₹18-25 LPA.",
        output_schema={
            "required_skills": ["list of skills"],
            "experience_years": "number",
            "remote_allowed": "boolean",
            "salary_range": {"min": "number", "max": "number", "currency": "string"}
        }
    )
    print(structured)

    # 5. Reflection
    print("\n--- PATTERN 5: Self-Critique ---")
    reflection = build_reflection_prompt(
        original_question="What is the difference between LangChain and LangGraph?",
        draft_answer="LangGraph is newer than LangChain."
    )
    print(reflection)


if __name__ == "__main__":
    demo_all_patterns()# src/prompt_engineering.py
"""
Day 8: Prompt engineering patterns for production AI systems
Every pattern you will use in Phase 2 projects
"""

from typing import Optional


# ─────────────────────────────────────────
# PATTERN 1: Zero / One / Few-shot
# ─────────────────────────────────────────

def build_few_shot_prompt(
    task_description: str,
    examples: list[dict],
    user_input: str
) -> str:
    """
    Build a few-shot prompt programmatically.
    examples: list of {"input": ..., "output": ...}
    """
    prompt = f"{task_description}\n\n"

    for i, ex in enumerate(examples, 1):
        prompt += f"Example {i}:\n"
        prompt += f"Input: {ex['input']}\n"
        prompt += f"Output: {ex['output']}\n\n"

    prompt += f"Now do the same:\nInput: {user_input}\nOutput:"
    return prompt


# ─────────────────────────────────────────
# PATTERN 2: Chain-of-Thought (CoT)
# ─────────────────────────────────────────

def build_cot_prompt(question: str, domain: str = "general") -> str:
    """
    Adds CoT instruction to any question.
    Forces intermediate reasoning → better accuracy.
    """
    cot_system = f"""You are an expert in {domain}. 
When answering, always:
1. Think through the problem step by step
2. Show your reasoning explicitly
3. Then give your final answer

Format:
REASONING: [your step-by-step thinking]
ANSWER: [final concise answer]"""

    return f"{cot_system}\n\nQuestion: {question}"


# ─────────────────────────────────────────
# PATTERN 3: Production RAG Prompt
# ─────────────────────────────────────────

def build_rag_prompt(
    context_chunks: list[str],
    user_question: str,
    system_role: str = "helpful AI assistant",
    output_format: Optional[str] = None
) -> dict:
    """
    Production-grade RAG prompt builder.
    Returns dict with 'system' and 'user' keys
    for direct use with any LLM API.
    """
    # System prompt: role + grounding constraint
    system = f"""You are a {system_role}.

CRITICAL INSTRUCTIONS:
1. Answer ONLY using the provided context below
2. If the answer is not in the context, say exactly: "I don't have enough information to answer that based on the provided context."
3. Never make up information
4. Cite which context number supports your answer"""

    if output_format:
        system += f"\n5. Respond in this format: {output_format}"

    # User message: context + question
    context_str = "\n\n".join(
        [f"[Context {i+1}]: {chunk}" for i, chunk in enumerate(context_chunks)]
    )

    user = f"""Context information:
{context_str}

Question: {user_question}"""

    return {"system": system, "user": user}


# ─────────────────────────────────────────
# PATTERN 4: Structured Output Prompt
# ─────────────────────────────────────────

def build_structured_output_prompt(
    task: str,
    input_text: str,
    output_schema: dict
) -> str:
    """
    Force model to output valid JSON matching a schema.
    Critical for programmatic use in your pipelines.
    """
    import json
    schema_str = json.dumps(output_schema, indent=2)

    return f"""Task: {task}

Input: {input_text}

Respond with ONLY valid JSON matching this exact schema:
{schema_str}

Do not include any text before or after the JSON. Do not use markdown code blocks."""


# ─────────────────────────────────────────
# PATTERN 5: Self-Critique / Reflection
# ─────────────────────────────────────────

def build_reflection_prompt(
    original_question: str,
    draft_answer: str
) -> str:
    """
    Ask model to critique and improve its own answer.
    Same pattern as Day 7 reflection agent, but as a prompt.
    """
    return f"""You generated this answer to a question. Now critique and improve it.

Original Question: {original_question}

Draft Answer:
{draft_answer}

Critique the draft answer:
1. Is it accurate?
2. Is it complete?
3. Is it concise?
4. Are there any errors or missing information?

Then provide an improved answer.

CRITIQUE: [your critique here]
IMPROVED ANSWER: [better answer here]"""


# ─────────────────────────────────────────
# DEMO: Show all patterns
# ─────────────────────────────────────────

def demo_all_patterns():
    print("=" * 60)
    print("PROMPT ENGINEERING PATTERNS DEMO")
    print("=" * 60)

    # 1. Few-shot
    print("\n--- PATTERN 1: Few-shot ---")
    few_shot = build_few_shot_prompt(
        task_description="Classify the sentiment of customer reviews as POSITIVE, NEGATIVE, or NEUTRAL.",
        examples=[
            {"input": "The product arrived quickly and works perfectly!", "output": "POSITIVE"},
            {"input": "Stopped working after 2 days. Very disappointed.", "output": "NEGATIVE"},
            {"input": "It's okay, does what it says.", "output": "NEUTRAL"},
        ],
        user_input="Delivery was late but the quality is good."
    )
    print(few_shot)

    # 2. CoT
    print("\n--- PATTERN 2: Chain-of-Thought ---")
    cot = build_cot_prompt(
        "If a RAG pipeline retrieves 5 chunks of 500 tokens each, and the system prompt is 200 tokens, what percentage of a 4096-token context window is used?",
        domain="AI engineering"
    )
    print(cot)

    # 3. RAG prompt
    print("\n--- PATTERN 3: Production RAG Prompt ---")
    rag = build_rag_prompt(
        context_chunks=[
            "LangGraph is built on top of LangChain and enables stateful workflows.",
            "StateGraph allows you to define nodes (functions) and edges (transitions).",
        ],
        user_question="How does LangGraph relate to LangChain?",
        system_role="senior AI engineering assistant"
    )
    print(f"SYSTEM:\n{rag['system']}\n")
    print(f"USER:\n{rag['user']}")

    # 4. Structured output
    print("\n--- PATTERN 4: Structured Output ---")
    structured = build_structured_output_prompt(
        task="Extract key information from this AI job description",
        input_text="Seeking an AI engineer with 2+ years LangChain experience, Python skills, and RAG knowledge. Remote OK. Salary: ₹18-25 LPA.",
        output_schema={
            "required_skills": ["list of skills"],
            "experience_years": "number",
            "remote_allowed": "boolean",
            "salary_range": {"min": "number", "max": "number", "currency": "string"}
        }
    )
    print(structured)

    # 5. Reflection
    print("\n--- PATTERN 5: Self-Critique ---")
    reflection = build_reflection_prompt(
        original_question="What is the difference between LangChain and LangGraph?",
        draft_answer="LangGraph is newer than LangChain."
    )
    print(reflection)


if __name__ == "__main__":
    demo_all_patterns()