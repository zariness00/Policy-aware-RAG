# Initial idea
To create two RAG systems with Naive RAG and Version RAG
I have 12 policies with 28 full docs. Then, based on the policies and their versions, I create 60 questions with gpt 4o mini. 
The architecture of the Version RAG and 6 categories are used from the [VersionRAG paper](https://arxiv.org/pdf/2510.08109). 

Here is the distribution of categories and num of questions

| Category | Questions |
| --- | ---: |
| Content retrieval | 12 |
| Complex content retrieval | 12 |
| Version-specific content retrieval | 12 |
| Version listing and inquiry | 12 |
| Explicit change retrieval | 6 |
| Implicit change retrieval | 6 |

This gives every policy one question in each of the first four categories. The remaining 12 change questions cover six policies. In total, 36 of 60 questions are version-sensitive, keeping the paper's 60% share in a smaller dataset.

## Synthetic policy creation

1. `gpt-4o-mini` generates a complete v1 policy from a topic only.
2. For each revision, the model proposes material changes in two sections; v3 avoids repeating the same rule change from v2.
3. The script rewrites changed sections and copies unchanged text into a complete v2 or v3 version, saving the revision notes in `changes.md`.
4. The latest versions of P03, P06, and P09 include revision notes; explicit-change questions use them, while implicit-change questions compare two full versions.
5. The model then writes questions and expected answers from the relevant versions, checking evidence quotes for consistency and flagging mismatches for manual review.

Each document has an effective date and a `Supersedes` link in its header. The policy text is ready to be split into chunks during the later indexing stage

## Future plan

1. Indexing: Naive RAG in Milvus; Version RAG in Neo4j
2. Embeddings: OpenAI text-embedding-3-small
3. Generation model: use OpenAI when tokens/credits are available; otherwise switch to a local Ollama model such as DeepSeek on a remote server
4. Evaluation: LLM as a judge correctness/ perhaps some other surface based lvl metrics as well 