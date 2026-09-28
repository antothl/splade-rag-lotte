import torch
import torch.nn as nn
from typing import List, Tuple, Dict
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel
from tqdm.auto import tqdm
from config import device
from data import get_doc_text, get_text_from_index

class SPLADEEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        sparsity_weight: float = 0.0001,
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.sparsity_weight = sparsity_weight
        self.vocab_size = self.tokenizer.vocab_size

    def encode(
        self,
        texts: List[str],
        max_length: int = 256,
    ) -> torch.Tensor:
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        device = self.model.device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        logits = outputs.logits
        mask = inputs["attention_mask"].unsqueeze(-1)
        values = torch.log(1+torch.relu(logits)) * mask
        sparse_rep, _ = torch.max(values, dim=1)
        
        if sparse_rep.dim() == 1:
            sparse_rep = sparse_rep.unsqueeze(0)

        return sparse_rep

    def compute_flops_loss(self, sparse_rep: torch.Tensor) -> torch.Tensor:
        flops = (sparse_rep.sum(dim=0) ** 2).sum()
        return self.sparsity_weight * flops

    def forward(
        self,
        query_texts: List[str],
        doc_texts: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query_rep = self.encode(query_texts)
        doc_rep = self.encode(doc_texts)
        flops_loss = self.compute_flops_loss(query_rep) + self.compute_flops_loss(doc_rep)
        return query_rep, doc_rep, flops_loss

class CrossEncoderTeacher(nn.Module):
    def __init__(self, model_name: str = "distilbert-base-uncased"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.model.config.hidden_size, 1)

    def forward(
        self,
        queries: List[str],
        documents: List[str],
    ) -> torch.Tensor:
        inputs = self.tokenizer(queries, documents, return_tensors="pt", padding=True, truncation=True)
        device = self.model.device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        cls_output = outputs.last_hidden_state[:, 0, :]
        scores = self.classifier(cls_output)
        return scores.squeeze(-1)

class RAGPipeline:
    def __init__(
        self,
        retriever: SPLADEEncoder,
        generator,
        generator_tokenizer,
        doc_ids: List[str],
        text_lookup: Dict[str, str], 
        top_k: int = 3,
        pt_index_ref=None 
    ):
        self.retriever = retriever
        self.generator = generator
        self.gen_tokenizer = generator_tokenizer
        self.doc_ids = doc_ids
        self.text_lookup = text_lookup 
        self.top_k = top_k
        self.pt_index_ref = pt_index_ref
        
        self._index_documents()

    def _get_texts(self, doc_ids: List[str]) -> Dict[str, str]:
        return {did: self.text_lookup.get(did, "") for did in doc_ids}

    def _index_documents(self, batch_size: int = 16):
        print("Indexing documents with SPLADE...")
        all_reps = []
        with torch.no_grad():
            for i in tqdm(range(0, len(self.doc_ids), batch_size)):
                batch_ids = self.doc_ids[i : i + batch_size]
                texts = [self.text_lookup.get(doc_id, "")[:500] for doc_id in batch_ids]
                
                texts = [t if t.strip() else "empty document" for t in texts]
                
                reps = self.retriever.encode(texts)
                
                if reps.dim() == 1:
                    reps = reps.unsqueeze(0)
                    
                all_reps.append(reps.cpu())
                
        self.doc_reps = torch.cat(all_reps, dim=0)
        print(f"Documents indexed: {len(self.doc_ids)}")

    def retrieve(self, query: str, top_k: int = None) -> List[Tuple[str, float]]:
        if top_k is None:
            top_k = self.top_k
        with torch.no_grad():
            query_rep = self.retriever.encode([query])
            query_rep = query_rep.to(self.doc_reps.device)
            
        scores = torch.matmul(query_rep, self.doc_reps.T).squeeze(0)
        top_indices = torch.topk(scores, k=min(top_k, len(scores))).indices
        results = []
        for idx in top_indices:
            doc_id = self.doc_ids[idx]
            score = scores[idx].item()
            results.append((doc_id, score))
        return results

    def generate(
        self,
        query: str,
        context: str,
        max_new_tokens: int = 200,
    ) -> str:
        system_msg = "You are a helpful assistant that answers technical questions using the provided context."
        user_msg = f"Context:\n{context}\n\nQuestion: {query}"
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
        prompt = self.gen_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.gen_tokenizer(prompt, return_tensors="pt").to(self.generator.device) 
        with torch.no_grad():
            outputs = self.generator.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.gen_tokenizer.eos_token_id,
            )
        response = self.gen_tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "assistant" in response.lower():
            response = response.split("assistant")[-1].strip()
        return response

    def __call__(self, query: str) -> Dict:
        retrieved = self.retrieve(query)
        doc_texts = {did: self.text_lookup.get(did, "") for did, _ in retrieved}
        context = "\n\n".join([doc_texts[doc_id][:300] for doc_id, _ in retrieved])
        answer = self.generate(query, context)
        return {
            "query": query,
            "retrieved_docs": retrieved,
            "generated_answer": answer,
            "context": context
        }