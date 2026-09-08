from __future__ import annotations

import os
import json
import ssl
from dataclasses import dataclass
from time import perf_counter
from urllib import error as urllib_error
from urllib import request as urllib_request

# Disable LangSmith/LangChain tracing to prevent network-related import hangs
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"

from dotenv import load_dotenv

# Load environment variables
ENV_PATH = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path=ENV_PATH)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'chroma_db')
DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
CHROMA_BACKEND = "chromadb.PersistentClient"
SYSTEM_PROMPT_TEMPLATE = (
    "You are a helpful AI assistant for an MSc NLP course. "
    "Answer only using the retrieved context below. "
    "If the answer is not supported by the context, say clearly that you do not know based on the provided materials. "
    "Keep the answer concise, structured, and academically clear.\n\n"
    "Context:\n{context}"
)


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    model: str
    api_key: str


@dataclass
class RetrievedDocument:
    page_content: str
    metadata: dict


class SimpleRAGChain:
    def __init__(self, collection, llm_config: LLMConfig, top_k: int):
        self.collection = collection
        self.llm_config = llm_config
        self.top_k = top_k

    def invoke(self, inputs: dict):
        query = inputs["input"]
        retrieval = self.collection.query(
            query_texts=[query],
            n_results=self.top_k,
            include=["documents", "metadatas"],
        )

        documents = []
        document_texts = retrieval.get("documents", [[]])[0]
        metadatas = retrieval.get("metadatas", [[]])[0]

        for page_content, metadata in zip(document_texts, metadatas):
            documents.append(
                RetrievedDocument(
                    page_content=page_content or "",
                    metadata=metadata or {},
                )
            )

        context_text = "\n\n".join(document.page_content for document in documents)
        answer = generate_answer(query=query, context=context_text, config=self.llm_config)
        return {"answer": answer, "context": documents}


def _get_env(name: str) -> str | None:
    value = os.getenv(name)
    return value.strip() if value else None


def resolve_llm_config() -> LLMConfig:
    requested_provider = (_get_env("LLM_PROVIDER") or "auto").lower()
    anthropic_api_key = _get_env("ANTHROPIC_API_KEY")
    openai_api_key = _get_env("OPENAI_API_KEY")

    if requested_provider not in {"auto", "anthropic", "openai"}:
        raise ValueError(
            "Invalid LLM_PROVIDER value. Use 'auto', 'anthropic', or 'openai'."
        )

    if requested_provider == "anthropic":
        if not anthropic_api_key:
            raise ValueError(
                "LLM_PROVIDER is set to 'anthropic' but ANTHROPIC_API_KEY is missing."
            )
        return LLMConfig(
            provider="anthropic",
            model=_get_env("ANTHROPIC_MODEL") or DEFAULT_ANTHROPIC_MODEL,
            api_key=anthropic_api_key,
        )

    if requested_provider == "openai":
        if not openai_api_key:
            raise ValueError(
                "LLM_PROVIDER is set to 'openai' but OPENAI_API_KEY is missing."
            )
        return LLMConfig(
            provider="openai",
            model=_get_env("OPENAI_MODEL") or DEFAULT_OPENAI_MODEL,
            api_key=openai_api_key,
        )

    if anthropic_api_key:
        return LLMConfig(
            provider="anthropic",
            model=_get_env("ANTHROPIC_MODEL") or DEFAULT_ANTHROPIC_MODEL,
            api_key=anthropic_api_key,
        )

    if openai_api_key:
        return LLMConfig(
            provider="openai",
            model=_get_env("OPENAI_MODEL") or DEFAULT_OPENAI_MODEL,
            api_key=openai_api_key,
        )

    raise ValueError(
        "No LLM API key found. Set ANTHROPIC_API_KEY or OPENAI_API_KEY in app/.env."
    )


def _post_json(url: str, payload: dict, headers: dict, timeout: int = 120) -> dict:
    import certifi

    request = urllib_request.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    ssl_context = ssl.create_default_context(cafile=certifi.where())

    with urllib_request.urlopen(request, timeout=timeout, context=ssl_context) as response:
        return json.loads(response.read().decode("utf-8"))


def _generate_with_anthropic(query: str, context: str, config: LLMConfig) -> str:
    payload = {
        "model": config.model,
        "max_tokens": 1024,
        "temperature": 0,
        "system": SYSTEM_PROMPT_TEMPLATE.format(context=context),
        "messages": [{"role": "user", "content": query}],
    }

    try:
        response_payload = _post_json(
            url="https://api.anthropic.com/v1/messages",
            payload=payload,
            headers={
                "content-type": "application/json",
                "x-api-key": config.api_key,
                "anthropic-version": "2023-06-01",
            },
        )
    except urllib_error.HTTPError as error:
        error_body = error.read().decode("utf-8", errors="replace")
        if error.code == 401:
            raise RuntimeError("Anthropic authentication failed. Check ANTHROPIC_API_KEY in app/.env.") from error
        if error.code == 404:
            raise RuntimeError(
                f"Anthropic model '{config.model}' was not found or is not available for this API key. "
                f"Update ANTHROPIC_MODEL in app/.env to a model your account can access. "
                f"Recommended default: {DEFAULT_ANTHROPIC_MODEL}."
            ) from error
        raise RuntimeError(f"Anthropic API error ({error.code}): {error_body}") from error
    except urllib_error.URLError as error:
        raise RuntimeError(f"Anthropic API connection failed: {error.reason}") from error

    return "".join(
        block.get("text", "")
        for block in response_payload.get("content", [])
        if isinstance(block, dict)
    )


def _generate_with_openai(query: str, context: str, config: LLMConfig) -> str:
    payload = {
        "model": config.model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_TEMPLATE.format(context=context)},
            {"role": "user", "content": query},
        ],
    }

    try:
        response_payload = _post_json(
            url="https://api.openai.com/v1/chat/completions",
            payload=payload,
            headers={
                "content-type": "application/json",
                "authorization": f"Bearer {config.api_key}",
            },
        )
    except urllib_error.HTTPError as error:
        error_body = error.read().decode("utf-8", errors="replace")
        if error.code == 401:
            raise RuntimeError("OpenAI authentication failed. Check OPENAI_API_KEY in app/.env.") from error
        if error.code == 404:
            raise RuntimeError(
                f"OpenAI model '{config.model}' was not found or is not available for this API key. "
                "Update OPENAI_MODEL in app/.env."
            ) from error
        raise RuntimeError(f"OpenAI API error ({error.code}): {error_body}") from error
    except urllib_error.URLError as error:
        raise RuntimeError(f"OpenAI API connection failed: {error.reason}") from error

    choices = response_payload.get("choices", [])
    if not choices:
        return ""
    return choices[0].get("message", {}).get("content", "")


def generate_answer(query: str, context: str, config: LLMConfig) -> str:
    if config.provider == "anthropic":
        return _generate_with_anthropic(query=query, context=context, config=config)
    return _generate_with_openai(query=query, context=context, config=config)


def format_generation_error(error: Exception, config: LLMConfig) -> str:
    if config.provider == "openai":
        import openai

        if isinstance(error, openai.NotFoundError):
            return (
                f"OpenAI model '{config.model}' was not found or is not available "
                "for this API key. Update OPENAI_MODEL in app/.env."
            )
        if isinstance(error, openai.AuthenticationError):
            return "OpenAI authentication failed. Check OPENAI_API_KEY in app/.env."

    return str(error)


def _normalise_page(page):
    return (page + 1) if isinstance(page, int) else page


def extract_source_metadata(source_documents):
    retrieved_sources = []
    retrieved_pages = []
    seen_sources = set()
    seen_pages = set()

    for doc in source_documents:
        metadata = getattr(doc, "metadata", {}) or {}
        source_name = os.path.basename(metadata.get("source", "Unknown"))
        page = _normalise_page(metadata.get("page"))

        if source_name not in seen_sources:
            seen_sources.add(source_name)
            retrieved_sources.append(source_name)

        if page is not None and page not in seen_pages:
            seen_pages.add(page)
            retrieved_pages.append(page)

    return retrieved_sources, retrieved_pages


def get_chatbot_chain(top_k: int = 5, verbose: bool = True):
    import chromadb
    from chromadb.utils import embedding_functions

    llm_config = resolve_llm_config()

    if not os.path.exists(OUTPUT_DIR):
        raise FileNotFoundError(
            f"Vector store not found at {OUTPUT_DIR}. Please run build_index.py first."
        )

    client = chromadb.PersistentClient(path=OUTPUT_DIR)
    collections = client.list_collections()
    if not collections:
        raise FileNotFoundError(
            f"No Chroma collections were found in {OUTPUT_DIR}. Please run build_index.py first."
        )

    collection_name = "langchain"
    available_names = [collection.name if hasattr(collection, "name") else collection for collection in collections]
    if collection_name not in available_names:
        collection_name = available_names[0]

    collection = client.get_collection(
        name=collection_name,
        embedding_function=embedding_functions.DefaultEmbeddingFunction(),
    )

    if verbose:
        print(f"LLM provider: {llm_config.provider}")
        print(f"LLM model: {llm_config.model}")
        print(f"Chroma backend: {CHROMA_BACKEND}")
        print(f"Chroma collection: {collection_name}")

    return SimpleRAGChain(collection=collection, llm_config=llm_config, top_k=top_k), llm_config


def run_query(
    query: str,
    *,
    chain=None,
    llm_config: LLMConfig | None = None,
    top_k: int = 5,
):
    if chain is None or llm_config is None:
        chain, llm_config = get_chatbot_chain(top_k=top_k, verbose=False)

    start_time = perf_counter()

    try:
        response = chain.invoke({"input": query})
        source_documents = response.get("context", [])
        retrieved_sources, retrieved_pages = extract_source_metadata(source_documents)

        return {
            "answer": response.get("answer", ""),
            "contexts": source_documents,
            "retrieved_sources": retrieved_sources,
            "retrieved_pages": retrieved_pages,
            "latency_seconds": perf_counter() - start_time,
            "provider": llm_config.provider,
            "model": llm_config.model,
            "error": None,
        }
    except Exception as error:
        friendly_error = format_generation_error(error, llm_config)
        return {
            "answer": "",
            "contexts": [],
            "retrieved_sources": [],
            "retrieved_pages": [],
            "latency_seconds": perf_counter() - start_time,
            "provider": llm_config.provider,
            "model": llm_config.model,
            "error": friendly_error,
        }

def main():
    print("=" * 60)
    print("  NLP MSc Course RAG Chatbot (Baseline)")
    print("  Type 'quit' or 'exit' to stop.")
    print("=" * 60)
    
    try:
        chain, llm_config = get_chatbot_chain()
    except Exception as e:
        print(f"Startup failed: {e}")
        return
        
    while True:
        try:
            query = input("\n[You]: ")
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break
            
        if query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
            
        if not query.strip():
            continue
            
        print("[Bot is thinking...]")
        
        result = run_query(query, chain=chain, llm_config=llm_config)

        if result["error"]:
            print(f"\nError generating response: {result['error']}")
            continue

        print(f"\n[Bot]: {result['answer']}\n")

        if result["contexts"]:
            print("-" * 40)
            print("Sources used:")
            seen_source_pages = set()
            for doc in result["contexts"]:
                metadata = getattr(doc, "metadata", {}) or {}
                source_name = os.path.basename(metadata.get("source", "Unknown"))
                page = _normalise_page(metadata.get("page"))
                display_page = page if page is not None else "?"
                source_page = (source_name, display_page)
                if source_page in seen_source_pages:
                    continue
                seen_source_pages.add(source_page)
                print(f" - {source_name} (Page {display_page})")
            print("-" * 40)

if __name__ == "__main__":
    main()
