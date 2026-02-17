---
title: Legal Case Summarization Assistant
emoji: ⚖️
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
license: mit
tags:
  - legal
  - summarization
  - lora
  - gemma
  - nlp
  - transformers
---

# ⚖️ Legal Case Summarization Assistant

An AI-powered application that generates concise summaries of legal court judgments using a LoRA fine-tuned Gemma-2B model.

## 🎯 Features

- 📄 **Instant Summarization**: Convert lengthy legal judgments into concise summaries
- 🎚️ **Adjustable Length**: Control summary length from 128 to 512 tokens
- 📚 **Pre-loaded Examples**: Three real legal case examples to try
- ⚡ **Fast Inference**: Optimized with 4-bit quantization
- 🎨 **Clean Interface**: User-friendly Gradio interface

## 📊 Model Performance

The model was fine-tuned on 4,000 legal case judgments and evaluated on 100 test samples:

| Metric | Score | Interpretation |
|--------|-------|----------------|
| **ROUGE-1** | 0.2908 | Vocabulary coverage |
| **ROUGE-2** | 0.1341 | Phrase fluency |
| **ROUGE-L** | 0.1760 | Structural coherence |
| **BLEU** | 0.0057 | Precision-based overlap |
| **BERTScore-P** | 0.8218 | Semantic precision |
| **BERTScore-R** | 0.7614 | Semantic recall |
| **BERTScore-F1** | 0.7902 | **Overall semantic quality** |
| **Perplexity** | 6.80 | Model confidence |

**Key Insight**: The high BERTScore (79% F1) indicates strong semantic understanding, making it reliable for capturing legal reasoning even if wording differs from reference summaries.

## 🔧 Technical Details

- **Base Model**: Google Gemma-2B (2 billion parameters)
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Quantization**: 4-bit NF4 with double quantization
- **Training Data**: joelniklaus/legal_case_document_summarization
- **Training Samples**: 4,000 legal judgments
- **Epochs**: 2
- **Peak GPU Memory**: 14.15 GB during training

### LoRA Configuration
- Rank (r): 16
- Alpha: 32
- Target modules: q_proj, v_proj
- Trainable parameters: ~20M (<1% of base model)

## 🚀 Usage

1. **Paste Legal Text**: Enter or paste a legal court judgment
2. **Adjust Length**: Use the slider to set desired summary length
3. **Generate**: Click the button to create your summary
4. **Copy Result**: Use the copy button to save your summary

## 📚 Example Use Cases

- **Legal Research**: Quickly review case precedents
- **Case Analysis**: Extract key points from lengthy judgments
- **Document Triage**: Prioritize which cases require detailed reading
- **Study Aid**: Understand complex legal reasoning

## ⚠️ Limitations

- **Domain-Specific**: Trained on legal text only
- **Context Length**: Input limited to ~1024 tokens (~750 words)
- **No Citation Verification**: May hallucinate case citations
- **Not Legal Advice**: For research purposes only

## 🔗 Links

- **GitHub Repository**: [Domain-Specific-FIned-Tuned-Legal-AI](https://github.com/mangaorphy/Domain-Specific-FIned-Tuned-Legal-AI)
- **Model Card**: [Full README](https://github.com/mangaorphy/Domain-Specific-FIned-Tuned-Legal-AI/blob/main/README.md)
- **Training Notebook**: [View on GitHub](https://github.com/mangaorphy/Domain-Specific-FIned-Tuned-Legal-AI/blob/main/legal-case-summarization-lora%20(1).ipynb)

## 📜 License

MIT License - See [LICENSE](https://github.com/mangaorphy/Domain-Specific-FIned-Tuned-Legal-AI/blob/main/LICENSE) for details

## ⚖️ Disclaimer

This AI model is for **research and educational purposes only**. The summaries generated are not a substitute for professional legal advice. Always consult qualified legal professionals for legal matters.

## 🙏 Acknowledgments

- Google for the Gemma base model
- Joel Niklaus et al. for the legal case summarization dataset
- Hugging Face for PEFT and Transformers libraries
- HuggingFace Spaces for hosting

---

**Built with** ❤️ **using Gradio and Transformers**
