import torch
import torch.nn.functional as F

class GroundedGenerationSteerer:
    """
    Inference-time grounded decoding mechanism for SLMs (Section 3.4).
    Adjusts token probabilities to minimize hallucination.
    """
    def __init__(self, model, config):
        self.model = model
        self.config = config

    def apply_grounding_constraint(self, logits, evidence_context):
        """
        Eq (10) & (11): Applying grounding-aware adjustment to LM logits.
        Steers the token generation probability P(y_t | y_<t, Q, C_Q).
        """
        # Calculate grounding adjustment factor Λ(y_t, C_Q)
        adjustment = self._compute_adjustment_factor(evidence_context)
        
        # Logit-space transformation for constrained generation
        steered_logits = logits + torch.log(adjustment + 1e-9)
        return F.softmax(steered_logits, dim=-1)
