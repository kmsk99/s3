"""FinQA integration shims for the s3 paper code (arxiv 2505.14146).

Three thin adapters that bridge the s3 paper's HTTP-based search agent to our
FinQA infrastructure without modifying paper code:

- finqa_retrieval_adapter: /retrieve endpoint → Weaviate hybrid + Cohere rerank
- finqa_generator_adapter: OpenAI chat completions → Bedrock Kimi K2.5
- prepare_finqa_dataset: FinQA JSON → s3 parquet + naive RAG cache

Plus a VERL reward hook (verl/utils/reward_score/finqa_exec_acc.py) that
swaps EM with FinQA numeric execution accuracy.
"""
