import ollama
import config

# Local LLM used
MODEL_NAME = "llama3"

# Keep the model grounded in retrieved text only
def build_prompt(context: str, question: str) -> str:
    return f"""You are a helpful assistant that answers only from the given context.
If the answer is not clearly in the context, say you are not sure.

Question: {question}

Context:
{context}

Answer:"""


# Send prompt to Ollama and return the response
def ask_llm(prompt: str) -> str:
    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": 0.2,
                "num_predict": 300,
            },
        )
        return response["message"]["content"].strip()
    except Exception as e:
        # If Ollama isn’t running
        return f"LLM error: {e}. Is Ollama running?"
