from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import MODEL, OLLAMA_BASE_URL

def complete(messages: list[dict], model: str = MODEL, temperature: float = 0.0) -> str:
    """
    Shared LLM call wrapper
    """
    llm = Ollama(model=model, base_url=OLLAMA_BASE_URL, temperature=temperature)

    full_prompt = "\n\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in messages
    )

    prompt = PromptTemplate.from_template("{full_prompt}")
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"full_prompt": full_prompt})