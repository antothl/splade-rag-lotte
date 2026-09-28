import torch
import pandas as pd
from tqdm.auto import tqdm
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import device, LLM_MODEL_NAME

def load_llm_model():
    print(f"Loading {LLM_MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
    is_quantized = "AWQ" in LLM_MODEL_NAME or "GPTQ" in LLM_MODEL_NAME
    
    if is_quantized:
        model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_NAME, device_map="auto")
    else:
        model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_NAME, dtype=torch.float16)
        model = model.to(device)
    
    model.eval()
    return model, tokenizer

def build_prompt(user: str, system: str = "You are a helpful assistant.", tokenizer=None) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

@torch.no_grad()
def generate_text(
    prompt: str,
    model, 
    tokenizer,
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    do_sample: bool = True,
    num_return_sequences: int = 1,
) -> list[str]:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=do_sample,
        num_return_sequences=num_return_sequences,
        pad_token_id=tokenizer.eos_token_id,
    )

    generated = []
    for output in outputs:
        text = tokenizer.decode(output, skip_special_tokens=True)
        if "assistant" in text.lower():
            text = text.split("assistant")[-1].strip()
        generated.append(text)

    return generated

def evaluate_retriever(
    retriever_fn,
    queries_df: pd.DataFrame,
    qrels_df: pd.DataFrame,
    ks: List[int] = [1, 5, 10],
) -> Dict:

    qrels_df = qrels_df.copy()
    qrels_df['docno'] = qrels_df['docno'].astype(str)
    qrels_df['qid'] = qrels_df['qid'].astype(str)
    queries_df = queries_df.copy()
    queries_df['qid'] = queries_df['qid'].astype(str)

    reciprocal_ranks = []
    recalls = {k: [] for k in ks}



    for _, query_row in tqdm(
        queries_df.iterrows(), desc="Evaluation", total=len(queries_df)
    ):
        qid = query_row["qid"]
        query_text = query_row["query"]
        relevant_docs = set(qrels_df[qrels_df["qid"] == qid]["docno"].tolist())

        if not relevant_docs:
            continue

        results = retriever_fn(query_text)
        retrieved_ids = [doc_id for doc_id, _ in results]

        rr = 0.0
        for rank, doc_id in enumerate(retrieved_ids, 1):
            if doc_id in relevant_docs:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

        for k in ks:
            top_k = set(retrieved_ids[:k])
            recalls[k].append(len(top_k & relevant_docs) / len(relevant_docs))

    metrics = {
        "MRR": sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0,
    }
    for k in ks:
        metrics[f"Recall@{k}"] = sum(recalls[k]) / len(recalls[k]) if recalls[k] else 0

    return metrics

def compute_mrr(results_df: pd.DataFrame, qrels: pd.DataFrame) -> float:
    scores = []
    relevant_qrels = qrels[qrels["label"] > 0]
    for qid in results_df["qid"].unique():
        results = results_df[results_df["qid"] == qid]
        query_rel = relevant_qrels[relevant_qrels["qid"] == qid]["docno"].values
        hits = results[results["docno"].isin(query_rel)]

        if not hits.empty:
            first_hit = hits.iloc[0]
            rank = first_hit["rank"]
            scores.append(1 / rank)
        else :
            scores.append(0)

    mrr = sum(scores) / len(results_df["qid"].unique())
    return mrr

def compute_recall_at_k(
    results_df: pd.DataFrame,
    qrels: pd.DataFrame,
    k: int = 10,
) -> float:
    recall_scores = []
    relevant_qrels = qrels[qrels["label"] > 0]

    for qid in results_df["qid"].unique():
        total_rel = len(relevant_qrels[relevant_qrels["qid"] == qid])
        top_k_results = results_df[(results_df["qid"] == qid) & (results_df["rank"] <= k)]
        query_rel = relevant_qrels[relevant_qrels["qid"] == qid]["docno"].values
        rel_found = top_k_results["docno"].isin(query_rel).sum()
        recall_scores.append(rel_found / total_rel)

    recallk = sum(recall_scores) / len(recall_scores)
    return recallk


