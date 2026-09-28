import torch
import torch.nn.functional as F
import random
import torch.optim as optim
from torch.optim import AdamW
from tqdm.auto import tqdm

def contrastive_loss(
    query_rep: torch.Tensor,
    doc_rep: torch.Tensor,
    temperature: float = 0.05,
) -> torch.Tensor:
    scores = torch.matmul(query_rep, doc_rep.T) / temperature
    labels = torch.arange(scores.size(0)).to(scores.device)
    loss = F.cross_entropy(scores, labels)
    return loss

def train_splade(
    encoder,
    train_data,
    num_epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    temperature: float = 0.05,
    use_hard_negatives: bool = False,
):
    optimizer = AdamW(encoder.parameters(), lr=learning_rate)
    encoder.train()

    for epoch in range(num_epochs):
        total_loss = 0
        total_contrastive = 0
        total_flops = 0
        num_batches = 0
        
        shuffled = train_data.copy()
        random.shuffle(shuffled)

        for i in tqdm(range(0, len(shuffled), batch_size), desc=f"Epoch {epoch+1}"):
            batch = shuffled[i : i + batch_size]
            if len(batch) < 2:
                continue

            if use_hard_negatives:
                queries = [ex["query"] for ex in batch]
                positives = [ex["positive"][:500] for ex in batch]
                all_negatives = []
                for ex in batch:
                    all_negatives.extend([n[:500] for n in ex.get("negatives", [])[:2]])

                query_rep = encoder.encode(queries)
                all_docs = positives + all_negatives
                doc_reps = encoder.encode(all_docs)
                positive_rep = doc_reps[: len(positives)]
                negative_rep = doc_reps[len(positives) :]

                cont_loss = contrastive_loss(query_rep, positive_rep, temperature)

                if len(negative_rep) > 0:
                    neg_scores = torch.matmul(
                        F.normalize(query_rep, p=2, dim=-1),
                        F.normalize(negative_rep, p=2, dim=-1).T,
                    )
                    hard_neg_loss = F.relu(neg_scores.mean() + 0.2).mean()
                    cont_loss = cont_loss + 0.3 * hard_neg_loss

                flops_loss = encoder.compute_flops_loss(
                    query_rep
                ) + encoder.compute_flops_loss(doc_reps)
            else:
                queries = [ex["query"] for ex in batch]
                docs = [ex["positive"][:500] for ex in batch]
                query_rep, doc_rep, flops_loss = encoder(queries, docs)
                cont_loss = contrastive_loss(query_rep, doc_rep, temperature)

            loss = cont_loss + flops_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_contrastive += cont_loss.item()
            total_flops += flops_loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        avg_cont = total_contrastive / num_batches if num_batches else 0
        avg_flops = total_flops / num_batches if num_batches else 0
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f} (Contrastive={avg_cont:.4f}, FLOPS={avg_flops:.4f})")

    encoder.eval()
    print("Training complete!")

def distillation_loss(
    student_scores: torch.Tensor,
    teacher_scores: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    return F.mse_loss(student_scores, teacher_scores.detach())

def train_distillation(
    student_model,
    teacher_model,
    train_data,
    num_epochs=3,
    batch_size=8,
    learning_rate=2e-5,
    sparsity_weight=0.0001
):
    optimizer = optim.AdamW(student_model.parameters(), lr=learning_rate)
    student_model.train()
    teacher_model.eval()

    device = next(student_model.parameters()).device
    teacher_model.to(device)

    for epoch in range(num_epochs):
        total_loss = 0
        total_distill = 0
        total_flops = 0

        random.shuffle(train_data)

        for i in tqdm(range(0, len(train_data), batch_size), desc=f"Epoch {epoch+1} (Distillation)"):
            batch = train_data[i : i + batch_size]
            if not batch: continue

            queries = [ex["query"] for ex in batch]
            positives = [ex["positive"][:512] for ex in batch]
            negatives = [ex["negatives"][0][:512] for ex in batch]

            with torch.no_grad():
                scores_teacher_pos = teacher_model(queries, positives)
                scores_teacher_neg = teacher_model(queries, negatives)
                target_margin = scores_teacher_pos - scores_teacher_neg

            q_rep = student_model.encode(queries)
            pos_rep = student_model.encode(positives)
            neg_rep = student_model.encode(negatives)

            scores_student_pos = torch.sum(q_rep * pos_rep, dim=1)
            scores_student_neg = torch.sum(q_rep * neg_rep, dim=1)

            pred_margin = scores_student_pos - scores_student_neg

            distill_loss = distillation_loss(pred_margin, target_margin)

            flops_loss = (
                student_model.compute_flops_loss(q_rep) +
                student_model.compute_flops_loss(pos_rep) +
                student_model.compute_flops_loss(neg_rep)
            )

            loss = distill_loss + flops_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_distill += distill_loss.item()
            total_flops += flops_loss.item()

        avg_loss = total_loss / (len(train_data) // batch_size + 1)
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f} (Distill={total_distill:.4f}, FLOPS={total_flops:.4f})")

    print("Distillation terminée !")
    return student_model


    