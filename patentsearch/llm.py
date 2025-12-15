from langchain_ollama import ChatOllama


def get_llm():
    """
    Uses your custom Modelfile model called 'patent-analyst'.

    Make sure you have run on a machine that *does* have the Ollama CLI:
        ollama create patent-analyst -f Modelfile

    And that an Ollama server is running and reachable (OLLAMA_HOST env if remote).
    """
    return ChatOllama(
        model="llama3.2:3b",
        temperature=0,
        timeout=30,
    )
