import pyterrier as pt
import pandas as pd
import shutil
import json
import torch
import warnings
import numpy as np
import matplotlib.pyplot as plt 
from pathlib import Path
from peft import LoraConfig, get_peft_model, TaskType

from config import INDEX_PATH, MLM_MODEL_NAME, NAVER_MODEL_NAME, NAVER_INDEX_PATH, device, SYNTHETIC_DATA_PATH, OUTPUT_DIR
from utils import load_llm_model, compute_mrr, evaluate_retriever
from data import create_training_triplets
from models import SPLADEEncoder, RAGPipeline, CrossEncoderTeacher
from train import train_splade, train_distillation

warnings.filterwarnings("ignore")


def main():
    if not pt.started():
        pt.init()

    print("Loading LLM...")
    model_gen, tokenizer_gen = load_llm_model()

    dataset = pt.get_dataset("irds:lotte/technology/dev/search")
    print("Dataset loaded: lotte/technology/dev/search")

    num_docs = 20000  
    num_queries = 500

    queries_df = dataset.get_topics().head(num_queries)
    qrels_df = dataset.get_qrels()

    queries_df['qid'] = queries_df['qid'].astype(str)
    qrels_df['qid'] = qrels_df['qid'].astype(str)
    qrels_df['docno'] = qrels_df['docno'].astype(str)

    query_ids = set(queries_df["qid"].tolist())
    qrels_df = qrels_df[qrels_df["qid"].isin(query_ids)].copy()
    relevant_doc_ids = set(qrels_df["docno"].tolist())

    corpus = []
    seen_ids = set()
    print("Loading corpus...")
    for doc in dataset.get_corpus_iter():
        doc_id = str(doc["docno"]) 
        doc["docno"] = doc_id    
        
        if doc_id in relevant_doc_ids:
            corpus.append(doc)
            seen_ids.add(doc_id)
        elif len(corpus) < num_docs:
            corpus.append(doc)
            seen_ids.add(doc_id)
            
        if len(corpus) >= num_docs and relevant_doc_ids.issubset(seen_ids):
            break

    loaded_doc_ids = {doc["docno"] for doc in corpus}
    corpus_dict = {str(doc["docno"]): doc["text"] for doc in corpus}
    qrels_df = qrels_df[qrels_df["docno"].isin(loaded_doc_ids)].copy()

    if INDEX_PATH.is_dir():
        try:
            shutil.rmtree(INDEX_PATH)
        except:
            print(f"Could not remove existing index at {INDEX_PATH}")
            
    INDEX_PATH.mkdir(parents=True, exist_ok=True)

    indexer = pt.IterDictIndexer(
        str(INDEX_PATH),
        overwrite=True,
        meta={"docno": 50, "text": 4096},
        meta_reverse=["docno"],
    )
    index_ref = indexer.index(corpus)
    index = pt.IndexFactory.of(index_ref)
    
    bm25_queries = queries_df[["qid", "query"]].copy()
    bm25_queries["query"] = bm25_queries["query"].apply(lambda x: "".join(c if c.isalnum() else " " for c in str(x)))
    
    bm25 = pt.BatchRetrieve(index_ref, wmodel="BM25")
    
    print("Calculating BM25 Baseline...")
    results = bm25.transform(bm25_queries)
    results["rank"] = results.groupby("qid").cumcount() + 1
    results['docno'] = results['docno'].astype(str)
    results['qid'] = results['qid'].astype(str)
    
    mrr = compute_mrr(results, qrels_df)
    print(f"BM25 Baseline MRR: {mrr:.4f}")

    synthetic_pairs = []
    if SYNTHETIC_DATA_PATH.exists():
        with open(SYNTHETIC_DATA_PATH, "r", encoding="utf-8") as f:
            synthetic_data = json.load(f)
        for item in synthetic_data:
            if item["doc_id"] in loaded_doc_ids:
                synthetic_pairs.append({
                    "query": item["query"],
                    "doc_id": item["doc_id"],
                    "source": item.get("source", "synthetic"),
                })
        print(f"Loaded {len(synthetic_pairs)} synthetic training pairs")

    num_train_queries = 300
    training_triplets = create_training_triplets(
        queries_df, qrels_df, index_ref, bm25,
        num_hard_negatives=3, max_queries=num_train_queries, synthetic_pairs=synthetic_pairs
    )

    splade_encoder = SPLADEEncoder(MLM_MODEL_NAME).to(device)
    if training_triplets:
        train_splade(splade_encoder, training_triplets, num_epochs=1, batch_size=8, use_hard_negatives=True)

    splade_lora = SPLADEEncoder(MLM_MODEL_NAME).to(device)
    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION, inference_mode=False, r=16, lora_alpha=32, lora_dropout=0.1,
        target_modules=["q_lin", "v_lin"] if "distilbert" in MLM_MODEL_NAME else ["query", "value"] 
    )
    splade_lora.model = get_peft_model(splade_lora.model, peft_config)
    if training_triplets:
        train_splade(splade_lora, training_triplets, num_epochs=1, batch_size=8, use_hard_negatives=True)

    cross_encoder_std = CrossEncoderTeacher(MLM_MODEL_NAME).to(device)
    splade_student_std = SPLADEEncoder(MLM_MODEL_NAME).to(device)
    splade_student_std.load_state_dict(splade_encoder.state_dict())
    if training_triplets:
        train_distillation(splade_student_std, cross_encoder_std, training_triplets, num_epochs=1, batch_size=8, learning_rate=1e-6)

    BETTER_TEACHER_NAME = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
    cross_encoder_expert = CrossEncoderTeacher(BETTER_TEACHER_NAME).to(device)
    splade_student_expert = SPLADEEncoder(MLM_MODEL_NAME).to(device)
    splade_student_expert.load_state_dict(splade_encoder.state_dict())
    if training_triplets:
        train_distillation(splade_student_expert, cross_encoder_expert, training_triplets, num_epochs=1, batch_size=8, learning_rate=1e-6)
    
    doc_id_list = [str(doc["docno"]) for doc in corpus]
    
    rag_std = RAGPipeline(
        retriever=splade_encoder, 
        generator=model_gen, 
        generator_tokenizer=tokenizer_gen, 
        doc_ids=doc_id_list, 
        text_lookup=corpus_dict, 
        pt_index_ref=index_ref,
        top_k=20
    )
    
    rag_lora = RAGPipeline(
        retriever=splade_lora, 
        generator=model_gen, 
        generator_tokenizer=tokenizer_gen, 
        doc_ids=doc_id_list, 
        text_lookup=corpus_dict, 
        pt_index_ref=index_ref, 
        top_k=20
    )
    
    rag_distill_std = RAGPipeline(
        retriever=splade_student_std, 
        generator=model_gen, 
        generator_tokenizer=tokenizer_gen, 
        doc_ids=doc_id_list, 
        text_lookup=corpus_dict, 
        pt_index_ref=index_ref,
        top_k=20
    )
    
    rag_distill_expert = RAGPipeline(
        retriever=splade_student_expert, 
        generator=model_gen, 
        generator_tokenizer=tokenizer_gen, 
        doc_ids=doc_id_list, 
        text_lookup=corpus_dict, 
        pt_index_ref=index_ref,
        top_k=20
    )
    
    test_queries = queries_df.tail(20)

    def bm25_retriever(query):
        res = bm25.search(query)
        return [(r["docno"], r["score"]) for _, r in res.head(20).iterrows()]
    
    def splade_std_retriever(query): return rag_std.retrieve(query, top_k=20)
    def splade_lora_retriever(query): return rag_lora.retrieve(query, top_k=20)
    def distill_std_retriever(query): return rag_distill_std.retrieve(query, top_k=20)
    def distill_expert_retriever(query): return rag_distill_expert.retrieve(query, top_k=20)

    print(f"Chargement du model: {NAVER_MODEL_NAME}")
    
    splade_naver = SPLADEEncoder(NAVER_MODEL_NAME).to(device)
    splade_naver.eval() 
    
    rag_naver = RAGPipeline(
        retriever=splade_naver,
        generator=model_gen,
        generator_tokenizer=tokenizer_gen,
        doc_ids=doc_id_list,
        text_lookup=corpus_dict,
        pt_index_ref=index_ref,
        top_k=20
    )
    
    def naver_retriever_fn(query):
        return rag_naver.retrieve(query, top_k=20)
    
    naver_retriever_pt = pt.apply.generic(lambda df: rag_naver.retrieve_batch(df) if hasattr(rag_naver, 'retrieve_batch') else pd.DataFrame())

    print("Computing metrics...")
    metrics_bm25 = evaluate_retriever(bm25_retriever, test_queries, qrels_df)
    metrics_splade = evaluate_retriever(splade_std_retriever, test_queries, qrels_df)
    metrics_lora = evaluate_retriever(splade_lora_retriever, test_queries, qrels_df)
    metrics_distill = evaluate_retriever(distill_std_retriever, test_queries, qrels_df)
    metrics_expert = evaluate_retriever(distill_expert_retriever, test_queries, qrels_df)
    
    metrics_naver = evaluate_retriever(naver_retriever_fn, test_queries, qrels_df)

    print("\n" + "="*115)
    print(f"{'Metric':<12} {'BM25':>10} {'SPLADE':>10} {'LoRA':>10} {'Distill':>10} {'Distill++':>12} {'NAVER (SOTA)':>14}")
    print(f"{'':<12} {'(Base)':>10} {'(Std)':>10} {'(Peft)':>10} {'(StdTeach)':>10} {'(MsMarco)':>12} {'(Pre-trained)':>14}")
    print("-" * 115)
    
    for m in metrics_bm25:
        print(f"{m:<12} "
              f"{metrics_bm25[m]:>10.4f} "
              f"{metrics_splade[m]:>10.4f} "
              f"{metrics_lora[m]:>10.4f} "
              f"{metrics_distill[m]:>10.4f} "
              f"{metrics_expert[m]:>12.4f} "
              f"{metrics_naver[m]:>14.4f}")
    print("="*115 + "\n")

    print("\nTemperature Analysis (SPLADE Std)")
    
    best_retriever_fn = splade_std_retriever 

    def run_cot_temperature_experiment(temps_to_test, queries_subset, qrels):
        mrr_results, r1_results, r5_results, r10_results = [], [], [], []
        
        for temp in temps_to_test:
            print(f"Temperature: {temp}")
            
            def generate_with_temp(query):
                system_prompt = "You are an expert search assistant. Break down the user question into 3 simple, keyword-focused search queries."
                user_prompt = f"User Question: \"{query}\"\nOutput ONLY the queries, one per line."
                
                messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
                prompt_text = tokenizer_gen.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer_gen(prompt_text, return_tensors="pt").to(device)
                
                with torch.no_grad():
                    outputs = model_gen.generate(
                        **inputs, max_new_tokens=60, temperature=temp, do_sample=True, pad_token_id=tokenizer_gen.eos_token_id
                    )
                gen_text = tokenizer_gen.decode(outputs[0], skip_special_tokens=True)
                if "assistant" in gen_text: gen_text = gen_text.split("assistant")[-1]
                return [l.strip("- *.") for l in gen_text.split('\n') if l.strip()][:3]

            def temp_retriever(query):
                sub_queries = generate_with_temp(query)
                if not sub_queries: sub_queries = [query]
                
                all_scores = {}
                for i, q_in in enumerate([query] + sub_queries):
                    weight = 1.0 if i == 0 else 0.2
                    results = best_retriever_fn(q_in) 
                    for doc_id, score in results:
                        all_scores[doc_id] = all_scores.get(doc_id, 0) + (score * weight)
                
                final = list(all_scores.items())
                final.sort(key=lambda x: x[1], reverse=True)
                return final[:20]

            metrics = evaluate_retriever(temp_retriever, queries_subset, qrels, ks=[1, 5, 10])
            mrr_results.append(metrics.get('MRR', 0))
            r1_results.append(metrics.get('Recall@1', 0))
            r5_results.append(metrics.get('Recall@5', 0))
            r10_results.append(metrics.get('Recall@10', 0))
            print(f"  -> MRR: {metrics.get('MRR', 0):.4f} | R@10: {metrics.get('Recall@10', 0):.4f}")

        return mrr_results, r1_results, r5_results, r10_results

    temperatures = [0.1, 0.3, 0.5, 0.7, 0.9, 1.2] 
    mrrs, r1s, r5s, r10s = run_cot_temperature_experiment(temperatures, test_queries, qrels_df)

    plt.figure(figsize=(10, 6))
    plt.plot(temperatures, r10s, marker='o', label='Recall@10', color='green', linewidth=2)
    plt.plot(temperatures, r5s, marker='o', label='Recall@5', color='orange', linewidth=2)
    plt.plot(temperatures, r1s, marker='o', label='Recall@1', color='red', linewidth=2)
    plt.plot(temperatures, mrrs, marker='s', label='MRR', color='blue', linestyle='--', alpha=0.7)
    plt.title("Impact of Temperature (SPLADE Std)")
    plt.xlabel("Temperature")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_path_temp = OUTPUT_DIR / "cot_temperature_splade_std.png"
    plt.savefig(save_path_temp)
    print(f"Saved: {save_path_temp}")
    plt.close()

    print("\nQuantity Analysis (SPLADE Std)")

    def run_cot_quantity_experiment(nums_to_test, queries_subset, qrels):
        mrr_results, r1_results, r5_results, r10_results = [], [], [], []
        fixed_temp = 0.5
        
        for n in nums_to_test:
            print(f"Testing N = {n} queries...")
            
            def generate_n_queries(query):
                system_prompt = "You are an expert search assistant."
                user_prompt = f"User Question: \"{query}\"\nTask: Generate exactly {n} distinct search queries.\nOutput ONLY queries."
                messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
                prompt_text = tokenizer_gen.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer_gen(prompt_text, return_tensors="pt").to(device)
                
                with torch.no_grad():
                    outputs = model_gen.generate(
                        **inputs, max_new_tokens=80, temperature=fixed_temp, do_sample=True, pad_token_id=tokenizer_gen.eos_token_id
                    )
                gen_text = tokenizer_gen.decode(outputs[0], skip_special_tokens=True)
                if "assistant" in gen_text: gen_text = gen_text.split("assistant")[-1]
                return [l.strip("- *.") for l in gen_text.split('\n') if l.strip()][:n]

            def temp_retriever(query):
                sub_queries = generate_n_queries(query)
                if not sub_queries: sub_queries = [query]
                
                all_scores = {}
                for i, q_in in enumerate([query] + sub_queries):
                    weight = 1.0 if i == 0 else 0.2
                    results = best_retriever_fn(q_in) 
                    for doc_id, score in results:
                        all_scores[doc_id] = all_scores.get(doc_id, 0) + (score * weight)
                
                final = list(all_scores.items())
                final.sort(key=lambda x: x[1], reverse=True)
                return final[:20]

            metrics = evaluate_retriever(temp_retriever, queries_subset, qrels, ks=[1, 5, 10])
            mrr_results.append(metrics.get('MRR', 0))
            r1_results.append(metrics.get('Recall@1', 0))
            r5_results.append(metrics.get('Recall@5', 0))
            r10_results.append(metrics.get('Recall@10', 0))
            print(f"  -> MRR: {metrics.get('MRR', 0):.4f} | R@10: {metrics.get('Recall@10', 0):.4f}")

        return mrr_results, r1_results, r5_results, r10_results

    nums_list = [1, 2, 3, 4, 5]
    mrrs_n, r1s_n, r5s_n, r10s_n = run_cot_quantity_experiment(nums_list, test_queries, qrels_df)

    plt.figure(figsize=(10, 6))
    plt.plot(nums_list, r10s_n, marker='o', label='Recall@10', color='green', linewidth=2)
    plt.plot(nums_list, r5s_n, marker='o', label='Recall@5', color='orange', linewidth=2)
    plt.plot(nums_list, r1s_n, marker='o', label='Recall@1', color='red', linewidth=2)
    plt.plot(nums_list, mrrs_n, marker='s', label='MRR', color='purple', linestyle='--', alpha=0.7)
    plt.title("Impact des sous requetes (SPLADE Std)")
    plt.xlabel("nombre de sous requetes")
    plt.ylabel("Score")
    plt.xticks(nums_list) 
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_path_quant = OUTPUT_DIR / "cot_quantity_splade_std.png"
    plt.savefig(save_path_quant)
    print(f"Saved: {save_path_quant}")
    plt.close()

    def run_alpha_study(bm25_retriever, splade_retriever, queries, qrels):
        print("\n" + "="*50)
        print("Analyse hybride bm25 + splade pretrained")
        print("="*50)
        
        def normalize_scores(df):
            if len(df) == 0: return df
            def scale_group(group):
                mx, mn = group["score"].max(), group["score"].min()
                if mx - mn > 0:
                    group["score"] = (group["score"] - mn) / (mx - mn)
                else:
                    group["score"] = 1.0
                return group
            return df.groupby("qid", group_keys=False).apply(scale_group)

        bm25_pipe = bm25_retriever 
        splade_pipe = splade_retriever 
        
        scaler = pt.apply.generic(normalize_scores)
        
        bm25_scaled = bm25_pipe >> scaler
        splade_scaled = splade_pipe >> scaler
        
        alphas = np.linspace(0, 1, 11)
        mrr_results = []
        special_results = {}

        metrics = ["recip_rank", "recall_1", "recall_5", "recall_10"]

        print(f"{'Alpha':<6} {'MRR':<8} {'R@1':<8} {'R@5':<8} {'R@10':<8}")
        print("-" * 45)

        for alpha in alphas:
            a_round = round(alpha, 1)
            
            hybrid_pipe = (bm25_scaled * alpha) + (splade_scaled * (1.0 - alpha))
            
            res = pt.Experiment(
                [hybrid_pipe],
                queries,
                qrels,
                eval_metrics=metrics,
                verbose=False
            )
            
            mrr_val = res.iloc[0]['recip_rank']
            r1 = res.iloc[0].get('recall_1', res.iloc[0].get('recall_1', 0))
            r5 = res.iloc[0].get('recall_5', res.iloc[0].get('recall_5', 0))
            r10 = res.iloc[0].get('recall_10', res.iloc[0].get('recall_10', 0))

            mrr_results.append(mrr_val)
            print(f"{a_round:<6.1f} {mrr_val:<8.4f} {r1:<8.4f} {r5:<8.4f} {r10:<8.4f}")

        print("-" * 45)

        plt.figure(figsize=(10, 6))
        plt.plot(alphas, mrr_results, marker='o')
        plt.title("Hybrid Optimization: BM25 + Naver SPLADE")
        plt.xlabel('Alpha (BM25 weight)')
        plt.ylabel('MRR')
        plt.grid(True)
        plt.savefig(OUTPUT_DIR / 'hybrid_alpha_naver.png')
        print("Saved: hybrid_alpha_naver.png")
        
        return alphas[np.argmax(mrr_results)], max(mrr_results)

    def naver_pt_wrapper(df):
        all_res = []
        for _, row in df.iterrows():
            qid, query = str(row['qid']), row['query']
            res_list = naver_retriever_fn(query)
            for docid, score in res_list:
                all_res.append({"qid": qid, "docno": docid, "score": score, "query": query})
        
        if not all_res: return pd.DataFrame(columns=["qid", "docno", "score", "query"])
        return pd.DataFrame(all_res)

    naver_pipe_for_alpha = pt.apply.generic(naver_pt_wrapper)

    best_a, best_m = run_alpha_study(bm25, naver_pipe_for_alpha, test_queries, qrels_df)
    
    print("\n Résultats du rag")
    
    demo_queries = test_queries.head(2)

    def generate_rag_answer(model_name, retriever_func, query):
        results = retriever_func(query)
        top_docs = results[:3]
        
        context_texts = []
        for did, score in top_docs:
            if did in corpus_dict:
                context_texts.append(corpus_dict[did])
        
        full_context = "\n---\n".join(context_texts)
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Answer the question based on the context provided."},
            {"role": "user", "content": f"Context:\n{full_context}\n\nQuestion: {query}"}
        ]
        
        prompt_text = tokenizer_gen.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer_gen(prompt_text, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model_gen.generate(**inputs, max_new_tokens=150, temperature=0.3, do_sample=True)
        
        answer = tokenizer_gen.decode(outputs[0], skip_special_tokens=True)
        if "assistant" in answer:
            answer = answer.split("assistant")[-1].strip()
        
        return answer, full_context

    for idx, row in demo_queries.iterrows():
        q_text = row["query"]
        print(f"\nQUERY: {q_text}")
        print("-" * 50)
        
        ans_std, ctx_std = generate_rag_answer("SPLADE Std", splade_std_retriever, q_text)
        print(f"[SPLADE Std Answer]:\n{ans_std}\n")
        print(f"[Context Snippet]: {ctx_std[:200]}...") 
        
        print("-" * 20)
        
        ans_naver, ctx_naver = generate_rag_answer("Naver SOTA", naver_retriever_fn, q_text)
        print(f"[NAVER SOTA Answer]:\n{ans_naver}\n")
        print(f"[Context Snippet]: {ctx_naver[:200]}...")
        
        print("=" * 50)

if __name__ == "__main__":
    main()