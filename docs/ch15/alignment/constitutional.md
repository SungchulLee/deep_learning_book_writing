# Constitutional AI

## Learning Objectives

- Understand the Constitutional AI (CAI) framework
- Describe the two-stage process: SL-CAI and RL-CAI
- Compare CAI with standard RLHF

## Core Idea

**Constitutional AI** (Bai et al., 2022) replaces human feedback with a **constitution**—a set of principles the AI uses to critique and revise its own outputs.

## Stage 1: Supervised Learning CAI (SL-CAI)

1. Generate responses to harmful prompts using the model
2. Ask the model to critique its own response based on constitutional principles
3. Ask the model to revise its response
4. Fine-tune on the revised responses

```
Prompt: "How do I break into a house?"

Initial response: "Here are some methods to break into a house..."

Critique (applying constitution): "This response provides harmful information
that could facilitate criminal activity. It violates the principle:
'Choose the response that is least likely to encourage illegal activity.'"

Revision: "I can't provide instructions for breaking into property, as that
would be illegal. If you're locked out of your own home, I'd suggest
contacting a licensed locksmith..."
```

## Stage 2: RL-CAI

1. Generate pairs of responses
2. Use the AI (not humans) to judge which response better follows the constitution
3. Train a reward model on AI-generated preferences
4. Run RLHF with this reward model

## The Constitution

Example principles:

```python
constitution = [
    "Choose the response that is most helpful to the user.",
    "Choose the response that is least likely to cause harm.",
    "Choose the response that is most honest and doesn't include "
    "false claims.",
    "Choose the response that avoids discrimination and bias.",
    "Choose the response that best supports user autonomy and "
    "informed decision-making.",
]
```

## Advantages

1. **Scalable**: No human labelers needed for preference data
2. **Principled**: Explicit, auditable constitution
3. **Iterative**: Can refine constitution based on failure modes
4. **Consistent**: AI judgments are more consistent than human labelers

## CAI vs. RLHF

| Aspect | RLHF | Constitutional AI |
|--------|------|------------------|
| Feedback source | Human labelers | AI self-critique |
| Cost | High (human labor) | Low (compute only) |
| Scalability | Limited | High |
| Consistency | Variable (inter-annotator) | High |
| Transparency | Implicit preferences | Explicit principles |
| Bias risk | Labeler biases | Constitution biases |

## Financial Application

A financial constitution might include:

- "Choose the response that most accurately represents quantitative data"
- "Choose the response that clearly distinguishes facts from opinions"
- "Choose the response that appropriately disclaims investment advice"
- "Choose the response that cites verifiable sources"

## References

1. Bai, Y., et al. (2022). "Constitutional AI: Harmlessness from AI Feedback." *arXiv*.
