# Chatbot Evaluation Summary

## Evaluation Caveats
- **Pilot scale**: This is a small pilot evaluation set and might not perfectly cover all syllabus items.
- **Retrieval > Synthesis**: Retrieval often succeeds cleanly, whereas cross-document synthesis questions (like q10 expecting an abstention) highlight the bot's grounding limits.
- **Keyword proxy**: Keyword recall is a lightweight coverage proxy and not an absolute measure of factual correctness.
- **Metric limits**: Abstention performance fundamentally depends on strict/correct label limits, and retrieval hit success does not strictly guarantee answer completeness.

## Run Metadata
- Evaluated at: 2026-04-15T12:30:39+01:00
- Questions file: /Users/ugurercan/Desktop/DBS MSc in AI Classes/NLP/CA2/NLP Chatbot/evaluation_questions.json
- Total questions evaluated: 13
- LLM provider: anthropic
- LLM model: claude-haiku-4-5
- Retriever top-k: 5

## Overall Averages
- Average source hit rate@k: 0.000
- Average source recall@k: 0.000
- Average keyword recall: 0.000
- Average latency (seconds): 9.817
- Error count: 13
- Average composite question score: 0.000

## Answerable vs Out-of-Scope
- Answerable questions: 9
- Out-of-scope questions (including synthesis limits like q10 and out-of-corpus like q11): 4
- Answerable average source hit@k: 0.000
- Answerable average keyword recall: 0.000
- Out-of-scope abstention accuracy: 0.000

## By Category
- comparison: count=3, source_hit@k=0.000, keyword_recall=0.000, abstention_accuracy=n/a, latency=0.711, errors=3
- explanation: count=3, source_hit@k=0.000, keyword_recall=0.000, abstention_accuracy=n/a, latency=40.421, errors=3
- factual: count=3, source_hit@k=0.000, keyword_recall=0.000, abstention_accuracy=n/a, latency=0.608, errors=3
- out_of_scope: count=3, source_hit@k=n/a, keyword_recall=n/a, abstention_accuracy=0.000, latency=0.596, errors=3
- synthesis: count=1, source_hit@k=0.000, keyword_recall=0.000, abstention_accuracy=0.000, latency=0.607, errors=1

## By Difficulty
- easy: count=7, source_hit@k=0.000, keyword_recall=0.000, latency=17.662, errors=7
- hard: count=1, source_hit@k=0.000, keyword_recall=0.000, latency=0.607, errors=1
- medium: count=5, source_hit@k=0.000, keyword_recall=0.000, latency=0.675, errors=5

## Retrieval Success Summary
- Questions with expected sources: 10
- Mean source hit rate@k on those questions: 0.000
- Mean source recall@k on those questions: 0.000

## Abstention Performance Summary
- Out-of-scope questions evaluated: 4
- Abstentions detected: 0
- Correct abstentions: 0

## Optional Ragas Metrics
- Requested mode: auto
- Enabled: no
- Installed version: not available
- Status: Optional Ragas dependencies unavailable: No module named 'ragas'

## Best-Performing Questions
- q01: score=0.000, question="What is Retrieval Augmented Generation (RAG), and what problem is it intended to solve for language models?", sources=[]
- q02: score=0.000, question="What is the difference between Bag of Words and word embeddings in NLP?", sources=[]
- q03: score=0.000, question="How does TF-IDF weight terms in a document, and why are rare terms often more useful?", sources=[]

## Worst-Performing Questions
- q01: score=0.000, question="What is Retrieval Augmented Generation (RAG), and what problem is it intended to solve for language models?", error=Error executing plan: Error sending backfill request to compactor: Error constructing hnsw segment reader: Error creating hnsw segment reader: Error deserializing pickle file: eval error at offset 0: EOF while parsing
- q02: score=0.000, question="What is the difference between Bag of Words and word embeddings in NLP?", error=Error executing plan: Error sending backfill request to compactor: Error constructing hnsw segment reader: Error creating hnsw segment reader: Error deserializing pickle file: eval error at offset 0: EOF while parsing
- q03: score=0.000, question="How does TF-IDF weight terms in a document, and why are rare terms often more useful?", error=Error executing plan: Error sending backfill request to compactor: Error constructing hnsw segment reader: Error creating hnsw segment reader: Error deserializing pickle file: eval error at offset 0: EOF while parsing

## Sample Failures
- q01: error=Error executing plan: Error sending backfill request to compactor: Error constructing hnsw segment reader: Error creating hnsw segment reader: Error deserializing pickle file: eval error at offset 0: EOF while parsing, expected source not retrieved, low keyword recall (0.000) | answer="" | retrieved_sources=[]
- q02: error=Error executing plan: Error sending backfill request to compactor: Error constructing hnsw segment reader: Error creating hnsw segment reader: Error deserializing pickle file: eval error at offset 0: EOF while parsing, expected source not retrieved, low keyword recall (0.000) | answer="" | retrieved_sources=[]
- q03: error=Error executing plan: Error sending backfill request to compactor: Error constructing hnsw segment reader: Error creating hnsw segment reader: Error deserializing pickle file: eval error at offset 0: EOF while parsing, expected source not retrieved, low keyword recall (0.000) | answer="" | retrieved_sources=[]
- q04: error=Error executing plan: Error sending backfill request to compactor: Error constructing hnsw segment reader: Error creating hnsw segment reader: Error deserializing pickle file: eval error at offset 0: EOF while parsing, expected source not retrieved, low keyword recall (0.000) | answer="" | retrieved_sources=[]
- q05: error=Error executing plan: Error sending backfill request to compactor: Error constructing hnsw segment reader: Error creating hnsw segment reader: Error deserializing pickle file: eval error at offset 0: EOF while parsing, expected source not retrieved, low keyword recall (0.000) | answer="" | retrieved_sources=[]

## Interpretation
- Strengths: latency stays within a usable range for CLI interaction.
- Weaknesses: retrieval often misses the expected source, so answer quality is likely bottlenecked by grounding. answers frequently miss the expected key concepts, so factual coverage needs improvement. out-of-scope handling is inconsistent, so the bot may still answer beyond the supplied materials.
