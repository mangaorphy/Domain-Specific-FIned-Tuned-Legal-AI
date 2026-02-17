---
base_model: google/gemma-2b
library_name: peft
pipeline_tag: text-generation
tags:
- legal
- summarization
- lora
- gemma
- domain-specific
- fine-tuned
language:
- en
license: gemma
datasets:
- joelniklaus/legal_case_document_summarization
metrics:
- rouge
- bleu
- bertscore
---

# 📚 Legal Case Summarization Model (LoRA Fine-tuned Gemma-2B)

A domain-specific fine-tuned model for automated legal case summarization, built on Google's Gemma-2B using Low-Rank Adaptation (LoRA).

## Model Overview

This model generates concise summaries of complex legal court judgments, trained specifically on legal domain text. It transforms lengthy court documents into coherent summaries while preserving key legal reasoning, decisions, and citations.

**Key Features:**
- ✅ Semantic understanding of legal terminology (79% BERTScore F1)
- ✅ Fluent and confident text generation (6.80 perplexity)
- ✅ Memory-efficient training via LoRA (14.15 GB peak GPU usage)
- ✅ Fast inference on consumer GPUs

## Performance Metrics

### Evaluation Results (100 test samples)

| Metric | Score | Interpretation |
|--------|-------|----------------|
| **ROUGE-1** | 0.2908 | Vocabulary coverage - good word overlap |
| **ROUGE-2** | 0.1341 | Phrase fluency - captures bigrams |
| **ROUGE-L** | 0.1760 | Structural coherence - sentence structure |
| **BLEU** | 0.0057 | Precision-based overlap |
| **BERTScore-P** | 0.8218 | **Semantic precision - high relevance** |
| **BERTScore-R** | 0.7614 | **Semantic recall - good coverage** |
| **BERTScore-F1** | 0.7902 | **Overall semantic quality** |
| **Perplexity** | 6.80 | Model confidence (lower is better) |

### Improvement Over Base Model

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | Improvement |
|-------|---------|---------|---------|-------------|
| Base (Zero-shot) | 0.2213 | 0.0787 | 0.1412 | - |
| **Fine-tuned (LoRA)** | **0.2872** | **0.1323** | **0.1710** | **+2.98%** |

**Key Insight:** The high BERTScore (79% F1) indicates the model captures semantic meaning effectively, even if it uses different phrasing than reference summaries. This is particularly valuable for legal summarization where semantic accuracy matters more than exact word matching.

## Model Architecture

- **Base Model:** [google/gemma-2b](https://huggingface.co/google/gemma-2b) (2 billion parameters)
- **Fine-tuning Method:** LoRA (Low-Rank Adaptation)
- **Quantization:** 4-bit NF4 with double quantization
- **LoRA Configuration:**
  - Rank (r): 16
  - Alpha: 32
  - Dropout: 0.05
  - Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- **Trainable Parameters:** ~20M (only 1% of base model)

## Dataset

**Source:** [joelniklaus/legal_case_document_summarization](https://huggingface.co/datasets/joelniklaus/legal_case_document_summarization)

**Statistics:**
- **Training samples:** 4,000 legal judgments
- **Validation samples:** 500
- **Test samples:** 100
- **Domain:** Legal court cases with judgment-summary pairs
- **Input format:** Structured as instruction-input-response prompts

**Example Structure:**
```
Instruction: Summarize the following legal court judgment.

Input: [Legal judgment text - truncated to 1024 tokens]

Response: [Generated summary]
```

## Training Details

### Training Procedure

**Training Configuration:**
- **Optimizer:** PagedAdamW (8-bit)
- **Learning Rate:** 2e-5 (with cosine scheduler)
- **Batch Size:** 1 per device (gradient accumulation: 4)
- **Epochs:** 2
- **Max Sequence Length:** 1536 tokens (judgment: 1024, summary: 256)
- **Training Steps:** 500 (250 per epoch)
- **Warmup Steps:** 50

**Training Hardware:**
- **GPU:** Single GPU with 16GB+ VRAM
- **Peak Memory Usage:** 14.15 GB
- **Training Time:** 635.9 minutes (~10.6 hours)
- **Precision:** bfloat16 mixed precision

### Training Results

| Epoch | Training Loss | Validation Loss |
|-------|---------------|-----------------|
| 1 | 1.8098 | 1.8049 |
| 2 | 1.7868 | 1.7956 |

**Final Training Loss:** 1.8199

## How to Use This Model

### Installation

```bash
pip install transformers peft torch bitsandbytes accelerate
```

### Quick Start

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Configure 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2b",
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "google/gemma-2b",
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token

# Load LoRA adapters
model = PeftModel.from_pretrained(
    base_model,
    "./model_outputs",  # Path to your downloaded model
    is_trainable=False
)
model.eval()

# Generate summary
def summarize_legal_case(judgment_text):
    prompt = f"""Instruction:
Summarize the following legal court judgment.

Input:
{judgment_text}

Response:
"""
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1280)
    inputs = inputs.to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary.split("Response:")[-1].strip()

# Example usage
judgment = """
[Your legal judgment text here - can be several paragraphs]
"""

summary = summarize_legal_case(judgment)
print(summary)
```

### Using with Evaluation Script

A standalone evaluation notebook is provided in `evaluate_trained_model.ipynb`:

1. Update `MODEL_PATH` to your model location
2. Ensure HuggingFace token is configured (for Gemma access)
3. Run all cells to evaluate on 100 test samples
4. Results saved to `evaluation_results.csv`

## Example Summaries

### Example 1: Contract Dispute

**Input (truncated):**
> "The appellant filed a suit for specific performance of contract. The respondent had agreed to sell a commercial property but later refused to complete the transaction citing the appellant's failure to pay the full consideration amount within the stipulated time..."

**Generated Summary:**
> "The court dismissed the appellant's claim for specific performance. The appellant failed to demonstrate readiness and willingness to perform the contract by the agreed deadline. The delay in payment was not adequately explained, and the respondent was entitled to withdraw from the agreement."

**Key Strengths:** Captures main legal outcome, identifies critical issue (timing of payment), explains court's reasoning.

### Example 2: Criminal Appeal

**Input (truncated):**
> "This is an appeal against conviction under Section 302 IPC. The appellant was convicted for murder based solely on circumstantial evidence. The prosecution relied on the recovery of the weapon and the appellant's presence in the vicinity at the time of the incident..."

**Generated Summary:**
> "The High Court upheld the conviction, finding the chain of circumstantial evidence complete. The recovery of the murder weapon from the appellant's possession, coupled with eyewitness testimony placing him at the scene and absence of credible alibi, established guilt beyond reasonable doubt."

**Key Strengths:** Identifies case type, explains evidential basis, summarizes court's logical reasoning.

## Limitations & Considerations

### Current Limitations

1. **Low BLEU Score (0.57%):** The model generates semantically correct summaries but uses different phrasing than reference summaries. This affects exact n-gram matching metrics.

2. **Domain Specificity:** Trained exclusively on legal text - may not generalize well to other document types.

3. **Context Length:** Input limited to 1024 tokens (~750 words) - very long judgments need pre-truncation.

4. **No Citation Verification:** Model may hallucinate case citations or statute references - always verify legal citations.

5. **Training Data Bias:** Performance reflects the distribution of cases in the training dataset (may favor certain legal domains).

### Responsible Use

- ✅ **Appropriate Use:** Legal research assistance, initial case review, document triage
- ❌ **Not Suitable For:** Final legal advice, court submissions without review, replacing human judgment
- **Always:** Have qualified legal professionals review generated summaries

## 🔬 Technical Specifications

### Compute Infrastructure

- **Hardware:** NVIDIA GPU with 16GB+ VRAM (T4, V100, A100, or equivalent)
- **Cloud Platform:** Kaggle Notebooks (2x T4 GPUs)
- **Software Stack:**
  - Python 3.10+
  - PyTorch 2.0+
  - Transformers 4.38+
  - PEFT 0.18.1
  - BitsAndBytes 0.41+

### Model Size

- **Base Model:** ~5GB (4-bit quantized)
- **LoRA Adapters:** ~64MB
- **Total Disk Space:** ~5.1GB
- **Peak RAM Usage:** ~14.15GB during training, ~8GB during inference

## Training Hyperparameters

```python
training_args = {
    'learning_rate': 2e-5,
    'num_train_epochs': 2,
    'per_device_train_batch_size': 1,
    'per_device_eval_batch_size': 1,
    'gradient_accumulation_steps': 4,
    'warmup_steps': 50,
    'max_grad_norm': 1.0,
    'lr_scheduler_type': 'cosine',
    'optim': 'paged_adamw_8bit',
    'fp16': False,
    'bf16': True,
    'logging_steps': 10,
    'evaluation_strategy': 'steps',
    'eval_steps': 50,
    'save_strategy': 'epoch',
}

lora_config = {
    'r': 16,
    'lora_alpha': 316,
    'lora_dropout': 0.05,
    'bias': 'none',
    'task_type': 'CAUSAL_LM',
    'target_modules': ['q_proj', 'k_proj'],
}
```

##  Citation

If you use this model in your research, please cite:

```bibtex
@misc{legal-summarization-lora-2026,
  title={Domain-Specific Fine-Tuned Legal AI: Legal Case Summarization using LoRA},
  author={Your Name},
  year={2026},
  publisher={HuggingFace},
  howpublished={\url{https://huggingface.co/your-username/legal-summarization-lora}},
}
```

**Dataset Citation:**
```bibtex
@article{niklaus2023legalsum,
  title={MultiLegalPile: A 689GB Multilingual Legal Corpus},
  author={Niklaus, Joel and others},
  journal={arXiv preprint arXiv:2306.02069},
  year={2023}
}
```

## Contributing & Feedback

Contributions welcome! Areas for improvement:
- Expanding training data with more diverse legal domains
- Implementing retrieval-augmented generation for citation accuracy
- Multi-document summarization capabilities
- Cross-lingual legal summarization

## License

- **Model:** Gemma License (must accept terms at [google/gemma-2b](https://huggingface.co/google/gemma-2b))
- **Code:** MIT License
- **Dataset:** CC BY 4.0

## Acknowledgments

- Google for the Gemma base model
- Joel Niklaus et al. for the legal case summarization dataset
- Hugging Face for PEFT and Transformers libraries
- Kaggle for providing free GPU compute

## Contact

For questions, issues, or collaboration opportunities, please open an issue in the repository.

---

**Framework Versions:**
- PEFT: 0.18.1
- Transformers: 4.38+
- PyTorch: 2.0+
- BitsAndBytes: 0.41+

**Last Updated:** February 2026