# Prompting Lab

Systematic experiments in LLM prompting techniques, evaluated on real benchmarks.

## Experiments

| # | Name | Technique | Dataset |
|---|------|-----------|---------|
| 1 | Zero-Shot Sentiment | Single-instruction prompt | IMDB |
| 2 | Few-Shot Sentiment | 5 labeled examples in context | IMDB |
| 3 | Chain-of-Thought Math | Explicit step-by-step reasoning | Custom math tasks |
| 4 | Self-Reflection | Model reviews and corrects its own answer | Same math tasks |
| 5 | Injection Detection | Regex + pattern guard | Adversarial inputs |

## Setup

```bash
pip install requests datasets transformers
export GROQ_API_KEY="gsk_..."
```
## Run

```bash
# Individual experiment
python prompting_lab.py --experiment zero_shot
# All experiments
python prompting_lab.py --experiment all
```

## Design notes

### Zero-shot vs Few-shot
Both approaches are tested on the same 20-sample IMDB benchmark.  
Few-shot typically sees **0–5%** accuracy gain on strong instruction-tuned models.

### Chain-of-Thought
The `solve_cot` function asks the model to reason step by step.  
A separate `extract_answer` call parses the final number as JSON — avoids brittle regex over long prose.

### Self-Reflection
A second LLM call asks the model to critique its own solution.  
Analysis: the model often corrects edge cases (percentages vs counts) but occasionally over-corrects correct answers.

### Injection Detection
Rule-based guard with 10 patterns covering:
- SQL injection
- Python code injection
- XSS
- Instruction-override attacks
- DAN / jailbreak patterns

Detection is intentionally conservative — false negatives are safer than false positives in a user-facing assistant.
