# Knowledge base sources

`build_index.py` expects PDF files to be dropped into this `docs/` folder
before it is run — that's the corpus the chatbot retrieves from. The
original corpus used for the assessment mixed two kinds of material, and
only one kind is republishable here:

**Not included — DBS course lecture slides.** These are Dublin Business
School's own proprietary teaching materials for the NLP module (B9AI006),
covering feature engineering, text vectorisation, clustering, topic
modelling, transformers/LLMs, and retrieval-augmented generation. They
aren't included in this repository. If you have access to the module's
Moodle page, drop the module's PDF slide decks in here to reproduce the
original knowledge base exactly.

**Freely available — cited, not included (add them yourself if you want the
exact original corpus).** These are public papers that were also part of the
corpus; the code doesn't depend on any particular file so you can substitute
any PDFs of your choosing to test the pipeline:

- Vaswani et al. (2017), *Attention Is All You Need* — https://arxiv.org/abs/1706.03762
- Mikolov et al. (2013), *Efficient Estimation of Word Representations in Vector Space* — https://arxiv.org/abs/1301.3781
- Mikolov et al. (2013), *Distributed Representations of Words and Phrases and their Compositionality* — https://arxiv.org/abs/1310.4546
- Sanh et al. (2019), *DistilBERT, a distilled version of BERT* — https://arxiv.org/abs/1910.01108
- Es et al. (2023), *RAGAS: Automated Evaluation of Retrieval Augmented Generation* — https://arxiv.org/abs/2309.15217
- Blei, Ng & Jordan (2003), *Latent Dirichlet Allocation* — https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf
- Deerwester et al. (1990), *Indexing by Latent Semantic Analysis* — Journal of the American Society for Information Science, 41(6)
- Turing, A.M. (1950), *Computing Machinery and Intelligence* — Mind, 59(236), 433–460
- Kido, Igawa & Barbon (2016), *Topic Modelling Based on the Louvain Method in Online Social Networks* — XII Brazilian Symposium on Information Systems

## Rebuilding the index

```bash
# put your PDFs in docs/, then from app/:
python build_index.py
```

This persists a Chroma vector store to `outputs/chroma_db/` (not committed —
it's a generated build artifact, regenerate it locally).
