# llm_helper.py
import os
import openai
from dotenv import load_dotenv

load_dotenv()

# 1. Read API key from system environment variable
api_key = os.environ.get("DEEPSEEK_API_KEY")

# Security Check
if not api_key:
    raise ValueError("❌ Environment variable 'DEEPSEEK_API_KEY' not found. Please configure it in your system first!")

# 2. Initialize the OpenAI client
client = openai.OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com"
)


def _call_llm_api(prompt: str) -> str:
    """Internal helper function: Sends request to DeepSeek API."""
    try:
        response = client.chat.completions.create(
            model="deepseek-v4-pro",
            messages=[
                {
                    "role": "system",
                    "content": "You are a rigorous academic analysis assistant specializing in evaluating research papers, academic fields, and collaboration or differences among authors. Please respond in clear, well-structured English."
                },
                {"role": "user", "content": prompt}
            ],
            stream=False,
            extra_body={
                "thinking": {"type": "enabled"},      # Enable deep thinking mode
                "reasoning_effort": "high"            # High reasoning effort for academic analysis
            }
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ AI response error: {str(e)}"


def handle_ai_compare(node1: dict, node2: dict) -> str:
    """Public interface: Handles comparative analysis between two papers or two authors."""
    if node1['type'] == 'paper' and node2['type'] == 'paper':
        prompt = f"""
Please compare the research directions, core methodologies, potential connections, and differences between the following two papers:

Paper 1 Title: {node1['name']}
Paper 1 Abstract: {node1['info']}

--------------------------------------------------

Paper 2 Title: {node2['name']}
Paper 2 Abstract: {node2['info']}

Please provide a comprehensive and in-depth comparative analysis along the following three dimensions:
1. What common scientific background or research pain points do they share?
2. What are the fundamental differences in their technical approaches or research perspectives?
3. If the findings of both papers were combined, what potential innovative research directions could emerge?
"""
    elif node1['type'] == 'author' and node2['type'] == 'author':
        prompt = f"""
In the academic network, the following two scholars have close academic connections (or belong to the same cluster):
Scholar 1: {node1['name']}
Scholar 2: {node2['name']}

Based on this, infer and compare their research profiles:
1. Analyze the academic fields they are likely jointly involved in.
2. Explore potential differences in their research styles and specific focus areas.
3. Predict what innovative outcomes might arise if they engage in cross-disciplinary collaboration.
"""
    else:
        return "❌ Cross-type comparison is currently not supported (e.g., comparing a paper with an author). Please select either two papers or two authors."

    return _call_llm_api(prompt)


def handle_ai_question(selected_nodes: list, user_question: str) -> str:
    """Public interface: Handles free-form user questions based on selected nodes."""
    if not user_question or len(user_question.strip()) == 0:
        return "❌ Please enter a specific question in the text box!"

    context = ""
    for i, n in enumerate(selected_nodes):
        prefix = "📄 Paper" if n['type'] == 'paper' else "👤 Author"
        context += f"{prefix} [{i+1}]: {n['name']}\nContext/Abstract: {n['info']}\n\n"

    prompt = f"""
Based on the following context of academic entities, answer the user's question.

【Academic Context】
{context}

【User Question】
{user_question}

Please provide a detailed and academically insightful response based on the provided context. If the question is unrelated to the context, please state so at the beginning of your response.
"""
    return _call_llm_api(prompt)