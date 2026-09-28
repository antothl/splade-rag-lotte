import shutil
import json
import random
import pyterrier as pt
import pandas as pd
from tqdm.auto import tqdm
from typing import List, Dict
from config import INDEX_PATH, SYNTHETIC_DATA_PATH
from utils import generate_text, build_prompt

if not pt.started():
    pt.init()

def get_doc_text(index_ref, doc_id: str) -> str:
    try:
        if isinstance(index_ref, str):
            index = pt.IndexFactory.of(index_ref)
            meta_index = index.getMetaIndex()
        else:
            meta_index = index_ref
        
        docid = meta_index.getDocument("docno", doc_id)
        if docid >= 0:
            return meta_index.getItem("text", docid)
    except Exception:
        pass
    return ""

def get_text_from_index(index_ref, doc_ids: List[str]) -> Dict[str, str]:
    try:
        if isinstance(index_ref, str):
            index = pt.IndexFactory.of(index_ref)
            meta_index = index.getMetaIndex()
        else:
            meta_index = index_ref
            
        result = {}
        for doc_id in doc_ids:
            try:
                docid = meta_index.getDocument("docno", doc_id)
                if docid >= 0:
                    result[doc_id] = meta_index.getItem("text", docid)
            except Exception:
                pass
        return result
    except Exception:
        return {}

def generate_queries_for_document(
    document: str,
    model, 
    tokenizer,
    num_queries: int = 2,
    max_doc_length: int = 500,
) -> List[str]:
    doc_excerpt = document[:max_doc_length]
    prompt = f"Given this document, generate {num_queries} search queries..."
    response = generate_text(build_prompt(prompt, tokenizer=tokenizer), model, tokenizer, max_new_tokens=100, temperature=0.1)[0]
    response = response.strip()
    queries = response.split("\n")
    return queries

def generate_training_pairs(
    index_ref,
    qrels_df: pd.DataFrame,
    model,
    tokenizer,
    num_docs: int = 50,
    queries_per_doc: int = 2,
) -> List[Dict]:
    doc_ids = qrels_df["docno"].unique().tolist()
    sampled_ids = random.sample(doc_ids, min(num_docs, len(doc_ids)))
    training_pairs = []
    
    index = pt.IndexFactory.of(index_ref)
    meta_index = index.getMetaIndex()
    
    for doc_id in tqdm(sampled_ids):
        sample_doc = get_doc_text(meta_index, doc_id)
        queries = generate_queries_for_document(sample_doc, model, tokenizer, num_queries=queries_per_doc)
        for query in queries:
            training_pairs.append({"query": query, "doc_id": doc_id, "document": sample_doc})
    return training_pairs

def filter_synthetic_pairs(
    pairs: List[Dict],
    min_query_words: int = 3,
    max_query_words: int = 20,
) -> List[Dict]:
    filtered = []
    seen_queries = set()
    for pair in pairs:
        query = pair["query"]
        query_words = len(query.split())
        if query_words < min_query_words or query_words > max_query_words:
            continue
        query_lower = query.lower()
        if query_lower in seen_queries:
            continue
        seen_queries.add(query_lower)
        filtered.append(pair)
    return filtered

def create_combined_training_data(
    index_ref,
    qrels_df: pd.DataFrame,
    queries_df: pd.DataFrame,
    synthetic_pairs: List[Dict],
) -> List[Dict]:
    training_data = []
    
    index = pt.IndexFactory.of(index_ref)
    meta_index = index.getMetaIndex()

    for _, row in qrels_df.iterrows():
        qid = row["qid"]
        doc_id = row["docno"]
        query_rows = queries_df[queries_df["qid"] == qid]["query"].values
        if len(query_rows) == 0:
            continue
        doc_text = get_doc_text(meta_index, doc_id)
        if not doc_text:
            continue
        training_data.append({
            "query": query_rows[0],
            "doc_id": doc_id,
            "document": doc_text,
            "source": "qrels",
        })

    for pair in synthetic_pairs:
        training_data.append({
            "query": pair["query"],
            "doc_id": pair.get("doc_id"),
            "document": pair["document"],
            "source": "synthetic",
        })
    return training_data

def find_hard_negatives_bm25(
    query: str,
    positive_doc_ids: set,
    retriever,
    num_negatives: int = 5,
) -> List[str]:
    
    clean_query = "".join(c if c.isalnum() else " " for c in str(query))
    
    try:
        results = retriever.search(clean_query)
        hard_negatives = []
        for _, row in results.iterrows():
            doc_id = row["docno"]
            if doc_id not in positive_doc_ids:
                hard_negatives.append(doc_id)
                if len(hard_negatives) >= num_negatives:
                    break
        return hard_negatives
    except Exception as e:
        print(f"Error searching for query '{clean_query}': {e}")
        return []

def create_training_triplets(
    queries_df: pd.DataFrame,
    qrels_df: pd.DataFrame,
    index_ref,
    retriever,
    num_hard_negatives: int = 3,
    max_queries: int = None,
    synthetic_pairs: List[Dict] = None,
) -> List[Dict]:
    triplets = []
    index = pt.IndexFactory.of(index_ref)
    meta_index = index.getMetaIndex()

    queries_to_process = queries_df.head(max_queries) if max_queries else queries_df
    for _, query_row in tqdm(queries_to_process.iterrows(), desc="Creating training triplets", total=len(queries_to_process)):
        qid = query_row["qid"]
        query_text = query_row["query"]
        positive_doc_ids = set(qrels_df[qrels_df["qid"] == qid]["docno"].tolist())
        if not positive_doc_ids: continue
        positive_id = list(positive_doc_ids)[0]
        positive_text = get_doc_text(meta_index, positive_id)
        if not positive_text: continue

        hard_neg_ids = find_hard_negatives_bm25(query_text, positive_doc_ids, retriever, num_hard_negatives)
        hard_neg_texts = get_text_from_index(meta_index, hard_neg_ids)
        hard_neg_list = [hard_neg_texts[nid] for nid in hard_neg_ids if nid in hard_neg_texts]
        
        if hard_neg_list:
            triplets.append({"query": query_text, "positive": positive_text, "negatives": hard_neg_list})

    if synthetic_pairs:
        print(f"Adding triplets from {len(synthetic_pairs)} synthetic pairs...")
        limit = max_queries if max_queries else len(synthetic_pairs)
        for pair in tqdm(synthetic_pairs[:limit], desc="Processing synthetic pairs"):
            query_text = pair["query"]
            positive_id = pair["doc_id"]
            positive_text = get_doc_text(meta_index, positive_id)
            if not positive_text: continue
            
            hard_neg_ids = find_hard_negatives_bm25(query_text, {positive_id}, retriever, num_negatives=3)
            hard_neg_texts = get_text_from_index(meta_index, hard_neg_ids)
            hard_neg_list = [hard_neg_texts[nid] for nid in hard_neg_ids if nid in hard_neg_texts]
            
            if hard_neg_list:
                triplets.append({"query": query_text, "positive": positive_text, "negatives": hard_neg_list})

    return triplets