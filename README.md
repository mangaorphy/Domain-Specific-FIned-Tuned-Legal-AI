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

# Legal Case Summarization Model (LoRA Fine-tuned Gemma-2B)

A domain-specific fine-tuned model for automated legal case summarization, built on Google's Gemma-2B using Low-Rank Adaptation (LoRA).

## 🔗 Quick Links

- **Live Demo:** [HuggingFace Space](https://huggingface.co/spaces/manga44/TogetherSO)
- ** Video Demo:** [YouTube Tutorial](https://youtu.be/XH0J5CdNGgI)
- **GitHub Repository:** [Source Code](https://github.com/mangaorphy/Domain-Specific-FIned-Tuned-Legal-AI)

---

## Try It Now!

**Live Demo:** [https://huggingface.co/spaces/manga44/TogetherSO](https://huggingface.co/spaces/manga44/TogetherSO)

**Video Tutorial:** [Watch on YouTube](https://youtu.be/XH0J5CdNGgI) 🎥

Test the model directly in your browser with our interactive Gradio interface. The demo includes:
- Real-time legal case summarization
- Performance metrics display
- Input validation (legal content only)
- Pre-loaded example cases
- Modern, user-friendly interface

Watch the video tutorial above to see the model in action and learn how to use it effectively!

## Project Structure

```
Domain-Specific-FIned-Tuned-Legal-AI/
│
├── README.md                                    # Project documentation
├── requirements.txt                             # Python dependencies
│
├── legal-case-summarization-lora (1).ipynb     # Training notebook
│   └── Complete training pipeline with evaluation
│
├── model_outputs/                               # Fine-tuned model files
│   ├── adapter_config.json                     # LoRA adapter configuration
│   ├── adapter_model.safetensors               # LoRA weights (7MB)
│   ├── tokenizer.json                          # Tokenizer vocabulary
│   ├── tokenizer_config.json                   # Tokenizer settings
│   └── evaluation_results.csv                  # Performance metrics
│
├── testing_scripts/                             # Testing utilities
│   ├── qualitative_testing.py                  # Manual testing script
│   └── safe_legal_summarizer.py                # Validation examples
│
├── app.py                                       # Local Gradio demo
│
└── Hugging_Face_Deploy/                        # HuggingFace Space (ignored in git)
    ├── app.py                                  # Production Gradio app
    ├── requirements.txt                        # Deployment dependencies
    ├── README.md                               # Space documentation
    └── model_outputs/                          # Copied model files
```

**Key Files:**
- **Training:** `legal-case-summarization-lora (1).ipynb` contains the complete fine-tuning process
- **Model:** `model_outputs/` contains the LoRA adapters and tokenizer files
- **Demo:** `app.py` for local testing, `Hugging_Face_Deploy/` for production
- **Testing:** Scripts in `testing_scripts/` for qualitative evaluation

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

## Sample Predictions

Here's a real example of the model's output quality:

### Example: Judicial Review Case

**Input (excerpt):**
```
The appellant brought a claim for judicial review of a decision of the respondent, 
on 21 February 2012, to approve a Revenue Budget for 2012/13 in relation to the 
provision of youth services. In his c...
```

**Reference Summary (Human-written):**
```
Mr Aaron Hunt, born on 17 April 1991, suffers from ADHD, learning difficulties and 
behavioural problems. As a result, North Somerset Council (the Council) are statutorily 
required, so far as reasonably practicable, to secure access for him to sufficient 
educational and recreational leisure time activities for the improvement of his well being.
On 21 February 2012, the Council made a decision to ap...
```

**Generated Summary (Model Output):**
```
The appellant was a young person with a disability who used to attend a weekly youth club.
He was concerned about the impact which the reduction in the youth services budget was 
likely to have on the provision of services for young persons with disabilities and in 
particular on a weekly youth club for vulnerable young people which he used to attend.
The appellant brought a claim for judicial revie...
```

**Analysis:**
- **Semantic Accuracy:** Correctly identifies the core issue (budget cuts affecting youth services)judicial review
- **Coherent Narrative:** Flows naturally and maintains legal context
- **Style Difference:** Uses different phrasing than reference ("young person with a disability" vs specific details about Aaron Hunt)

> **Note:** The model prioritizes semantic coherence over exact phrase matching, which explains the high BERTScore (79%) despite lower ROUGE scores. This approach is ideal for legal summarization where meaning is paramount.

## Model Architecture

- **Base Model:** [google/gemma-2b](https://huggingface.co/google/gemma-2b) (2 billion parameters)
- **Fine-tuning Method:** LoRA (Low-Rank Adaptation)
- **Quantization:** 4-bit NF4 with double quantization
- **LoRA Configuration:**
  - Rank (r): 16
  - Alpha: 16
  - Dropout: 0.05
  - Target modules: `q_proj`, `k_proj`
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

### Option 1: Web Interface (Easiest)

Visit the **[Live Demo](https://huggingface.co/spaces/manga44/TogetherSO)** to use the model instantly without any setup:
1. Paste your legal judgment into the input box
2. Adjust the summary length (128-512 tokens)
3. Click "Generate Summary"
4. View the AI-generated summary with performance stats

**Running on CPU:** The demo runs on CPU hardware (free tier). Generation takes 130-300 seconds per summary.

### Option 2: Local Installation

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
    "./model_outputs",  # Path to downloaded model
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

## Limitations & Considerations

### Current Limitations

1. **Low BLEU Score (0.57%):** The model generates semantically correct summaries but uses different phrasing than reference summaries. This affects exact n-gram matching metrics.

2. **Domain Specificity:** Trained exclusively on legal text - may not generalize well to other document types.

3. **Context Length:** Input limited to 1024 tokens (~750 words) - very long judgments need pre-truncation.

4. **No Citation Verification:** Model may hallucinate case citations or statute references and dates - always verify legal citations.

5. **Training Data Bias:** Performance reflects the distribution of cases in the training dataset (may favor certain legal domains).

### Responsible Use

- ✅ **Appropriate Use:** Legal research assistance, initial case review, document triage
- ❌ **Not Suitable For:** Final legal advice, court submissions without review, replacing human judgment
- **Always:** Have qualified legal professionals review generated summaries

## Technical Specifications

### Compute Infrastructure

- **Hardware:** GPU with 16GB+ VRAM (T4, V100, A100, or equivalent)
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
    'lora_alpha': 16,
    'lora_dropout': 0.05,
    'bias': 'none',
    'task_type': 'CAUSAL_LM',
    'target_modules': ['q_proj', 'k_proj'],
}
```

## Deployment

The model is deployed and publicly accessible via HuggingFace Spaces:

**Live Application:** [https://huggingface.co/spaces/manga44/TogetherSO](https://huggingface.co/spaces/manga44/TogetherSO)

**Features:**
- Interactive Gradio interface with modern UI
- Real-time input validation (rejects non-legal content)
- Performance metrics dashboard
- Example cases for quick testing
- CPU-optimized with 8-bit quantization
- Automatic preprocessing and output formatting

**Technical Stack:**
- Framework: Gradio 6.5.1
- Deployment: HuggingFace Spaces (CPU)
- Quantization: 8-bit with CPU offloading
- Memory Usage: ~8GB RAM during inference

## Use Cases

**Ideal For:**
- Legal research assistants reviewing case law
- Law students studying court judgments
- Legal tech platforms needing automated case summaries
- Document triage in legal departments
- Initial case analysis before detailed review

**Not Recommended For:**
- Final legal advice or court submissions without human review
- Processing non-legal documents (model includes validation)
- Cases requiring exact citation verification
- Real-time applications requiring sub-second response times

**Framework Versions:**
- PEFT: 0.18.1
- Transformers: 5++
- PyTorch: 2.0+
- BitsAndBytes: 0.41+

**Last Updated:** February 2026