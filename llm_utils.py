from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import MODEL, OLLAMA_BASE_URL


def complete(messages: list[dict], model: str = MODEL, temperature: float = 0.0) -> str:
    """Send a list of chat-style messages to the configured Ollama model.

    Args:
        messages: A list of dictionaries with "role" and "content" keys.
        model: The Ollama model identifier to use.
        temperature: Sampling temperature for the LLM call.

    Returns:
        The LLM response text as a string.
    """
    # Create a lightweight LangChain pipeline for the local Ollama runtime.
    llm = Ollama(model=model, base_url=OLLAMA_BASE_URL, temperature=temperature)

    # Flatten the role/content message history into a single prompt string.
    full_prompt = "\n\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in messages
    )

    prompt = PromptTemplate.from_template("{full_prompt}")
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"full_prompt": full_prompt})