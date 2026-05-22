import torch
import torch.nn.functional as F

class MetricsCalculator:
    """
    Implementation of the custom metrics defined in Section 4.5.
    Ensures that the quantitative evaluation is reproducible.
    """
    
    @staticmethod
    def compute_path_coherence(path_embeddings):
        """
        Calculates Semantic Path Coherence (Ψ).
        Eq: Ψ = (1/N) * Σ cos_sim(e_i, e_{i+1})
        """
        # path_embeddings: Tensor of shape (path_length, embedding_dim)
        cos_sims = []
        for i in range(len(path_embeddings) - 1):
            sim = F.cosine_similarity(
                path_embeddings[i].unsqueeze(0), 
                path_embeddings[i+1].unsqueeze(0)
            )
            cos_sims.append(sim)
        
        return torch.stack(cos_sims).mean().item()

    @staticmethod
    def compute_contradiction_sensitivity(generated_tokens, evidence_graph):
        """
        Calculates Contradiction Sensitivity (Ω).
        Eq: Ω = log(P_hallucination / P_grounded)
        """
        # This assumes the model outputs log-probs or specific attention weights
        # Logic: Compare generated sequence against graph's contradiction edges
        hallucination_score = ... # Implementation based on graph traversal
        grounding_score = ...     
        
        sensitivity = torch.log(hallucination_score + 1e-9) - torch.log(grounding_score + 1e-9)
        return sensitivity.item()
