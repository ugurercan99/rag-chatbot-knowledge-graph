# RAG Chatbot over a Document Knowledge Base

A Retrieval-Augmented Generation (RAG) chatbot that answers questions by
retrieving relevant passages from a corpus of PDFs (via ChromaDB + sentence
embeddings) and grounding an LLM's answer in them, built with LangChain.

## Context

This was the artefact for **Assessment 2 (CA2)** of the **NLP module
(B9AI006)** in the MSc in Artificial Intelligence at Dublin Business School.
The brief asked groups of 2–3 students to build and present a case study
applying one specialised NLP technique — this group chose **Retrieval-
Augmented Generation / conversational chatbot** — followed by an individual
written report covering AI ethics/bias/explainability, a critique of the
modelling techniques and libraries used, and a performance evaluation.

The original assignment brief (a DBS-internal assessment document) is not
included in this repository; the summary above covers its scope. The code
in `app/` is the group artefact; `report/Report.pdf` is the individual
write-up.

## How it works

- **Ingestion** (`app/build_index.py`): loads PDFs from `docs/`, chunks them,
  embeds them with a sentence-transformers model, and persists a Chroma
  vector store.
- **Chat** (`app/chatbot.py`, `app/app.py`): retrieves the top-k most
  relevant chunks for a user question and passes them to an LLM (Anthropic
  Claude by default, OpenAI supported as a fallback) to produce a grounded
  answer.
- **Evaluation** (`evaluation/`): a small evaluation harness
  (`evaluate_chatbot.py`) runs a fixed question set
  (`evaluation_questions.json` — factual, comparison, explanation,
  synthesis, and out-of-scope questions used to test abstention) and scores
  retrieval hit-rate, keyword recall, and latency. **Note:** the results
  checked into `evaluation_results.csv/json` and summarised in
  `evaluation_summary.md` show every question erroring out (0.000 across the
  board) — that run failed rather than reflecting the chatbot's true
  performance (most likely an expired/misconfigured API key or an index
  that hadn't been built at run time). Treat the harness as a working tool
  to re-run, not as evidence of the bot's real accuracy — see
  [Reproducing](#reproducing) below.
- **Exploration notebooks** (`exploration/`): earlier prototyping notebooks
  (a plain chat model, a first LangChain pass, and a LangChain+Neo4j
  variant) kept for the development trail — the app in `app/` is the final,
  working version.

## Repository structure

```
.
├── app/                  # the chatbot (build_index.py, chatbot.py, app.py)
├── docs/                 # SOURCES.md — where to source the PDF knowledge base (not included)
├── evaluation/           # evaluation harness, question set, and results
├── exploration/          # earlier prototyping notebooks
├── tests/                # sample questions for manual testing
└── report/Report.pdf     # individual report (ethics, critique, evaluation)
```

## Reproducing

```bash
cd app
pip install -r requirements.txt
cp .env.example .env        # then fill in your own API key
```

Add PDFs to `docs/` (see [`docs/SOURCES.md`](docs/SOURCES.md) for what the
original corpus contained and where to get the public sources), then:

```bash
python build_index.py       # builds the vector store in outputs/chroma_db/
python chatbot.py           # chat via CLI
# or: streamlit run app.py  # if you want the browser-based app instead
```

To re-run the evaluation harness once you have a working `.env` and a built
index:

```bash
cd ../evaluation
python evaluate_chatbot.py
```

## Security note

No API keys or other live credentials are included — `.env` is git-ignored
and only `.env.example` (placeholder values) is committed, and the
`exploration/` notebooks use placeholder values (`YOUR_OPENAI_API_KEY_HERE`,
`YOUR_NEO4J_HOST`, `YOUR_NEO4J_PASSWORD`) rather than real ones.

## License

Code is shared for portfolio/educational purposes. This was a group
assessment; the code here reflects the team's shared artefact as submitted.
