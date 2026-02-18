import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from huggingface_hub import login
import time
import os
import re

# Global variables for model
model = None
tokenizer = None

# Login to HuggingFace (required for gated models like Gemma)
HF_TOKEN = os.environ.get("HF_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN, add_to_git_credential=False)
    print("✓ Authenticated with HuggingFace")
else:
    print("⚠️ No HF_TOKEN found. Add it to Space secrets if using gated models.")

def is_legal_text(text, min_legal_terms=3):
    """
    Validate if input text is legal content.
    
    Returns: (is_valid, reason)
    """
    # Legal keywords that should appear in court judgments
    legal_keywords = {
        'court', 'plaintiff', 'defendant', 'appellant', 'appellee',
        'petitioner', 'respondent', 'judge', 'justice', 'magistrate',
        'verdict', 'judgment', 'ruling', 'order', 'decree', 'opinion',
        'motion', 'complaint', 'petition', 'brief', 'testimony',
        'evidence', 'trial', 'hearing', 'case', 'proceeding',
        'statute', 'law', 'legal', 'liability', 'damages', 'relief',
        'counsel', 'attorney', 'breach', 'contract', 'agreement'
    }
    
    # Patterns that indicate non-legal content
    non_legal_patterns = [
        (r'\b(recipe|ingredient|cook|bake|oven|cup|tablespoon|teaspoon)\b', 'cooking recipe'),
        (r'\b(software|app|download|install|click here|website|tutorial)\b', 'technology/tutorial content'),
        (r'\b(how to make|step \d+|instructions|procedure)\b', 'instructional content'),
        (r'\b(buy now|price|discount|sale|product|shop|purchase)\b', 'commercial/shopping content'),
        (r'\b(artificial intelligence|machine learning|neural network|algorithm)\b', 'AI/tech article'),
    ]
    
    text_lower = text.lower()
    word_count = len(text.split())
    
    # Reject if too short
    if word_count < 30:
        return False, "Text too short (minimum 30 words required for legal judgment)"
    
    # Check for non-legal patterns
    for pattern, content_type in non_legal_patterns:
        if re.search(pattern, text_lower):
            return False, f"This appears to be {content_type}, not a legal case"
    
    # Count legal keywords
    legal_count = sum(1 for keyword in legal_keywords if keyword in text_lower)
    
    if legal_count < min_legal_terms:
        return False, f"Insufficient legal terminology (found {legal_count} legal terms, need at least {min_legal_terms})"
    
    return True, f"Valid legal text detected ({legal_count} legal terms found)"

def load_model():
    """Load the fine-tuned model and tokenizer"""
    global model, tokenizer
    
    if model is not None and tokenizer is not None:
        return model, tokenizer
    
    print("Loading model... This may take 2-3 minutes.")
    
    # Configuration
    MODEL_PATH = "."  # Model files are in root directory on HF Spaces
    BASE_MODEL_NAME = "google/gemma-2b"
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_NAME,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Check if GPU is available
    has_gpu = torch.cuda.is_available()
    print(f"GPU available: {has_gpu}")
    
    try:
        if has_gpu:
            # GPU: Use 4-bit quantization
            print("Loading model with 4-bit quantization (GPU)...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            
            base_model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL_NAME,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
        else:
            # CPU: Load in 8-bit or full precision with CPU offloading
            print("Loading model for CPU (this may take 2-3 minutes)...")
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True,
            )
            
            base_model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL_NAME,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
        
        # Load LoRA adapters
        model = PeftModel.from_pretrained(
            base_model,
            MODEL_PATH,
            is_trainable=False
        )
        model.eval()
        
        print("✓ Model loaded successfully!")
        return model, tokenizer
        
    except Exception as e:
        print(f"Error loading with quantization: {e}")
        print("Attempting to load without quantization (slower but more compatible)...")
        
        # Fallback: Load without quantization
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            device_map="cpu",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float32,
        )
        
        model = PeftModel.from_pretrained(
            base_model,
            MODEL_PATH,
            is_trainable=False
        )
        model.eval()
        
        print("✓ Model loaded (CPU mode, no quantization)")
        return model, tokenizer

def generate_summary(model, tokenizer, judgment_text, max_length=256):
    """Generate summary for a legal judgment"""
    prompt = f"""Instruction:
Summarize the following legal court judgment.

Input:
{judgment_text}

Response:
"""
    
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1280
    ).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if "Response:" in generated_text:
        summary = generated_text.split("Response:")[-1].strip()
    else:
        summary = generated_text[len(prompt):].strip()
    
    return summary

def predict(judgment_text, max_length):
    """Main prediction function for Gradio interface"""
    if not judgment_text or not judgment_text.strip():
        return "⚠️ Please enter a legal judgment to summarize.", "", ""
    
    # Validate input is legal text
    is_valid, reason = is_legal_text(judgment_text)
    
    if not is_valid:
        warning_msg = f"""⚠️ INPUT VALIDATION FAILED

{reason}

This model is specifically trained for **legal case summarization** and will not process non-legal content.

✅ Expected input types:
• Court judgments
• Legal case documents  
• Judicial opinions
• Legal briefs or rulings

The text should contain legal terminology such as: court, plaintiff, defendant, judge, ruling, judgment, case, motion, etc.

If you believe this is a legal document, please ensure it contains appropriate legal terminology.
"""
        return warning_msg, "", "❌ Validation failed - non-legal content detected"
    
    try:
        # Load model (cached after first run)
        model, tokenizer = load_model()
        
        # Generate summary
        start_time = time.time()
        summary = generate_summary(model, tokenizer, judgment_text, int(max_length))
        elapsed_time = time.time() - start_time
        
        # Format stats
        stats = f"✓ Validation passed: {reason}\n⏱️ Generated in {elapsed_time:.2f} seconds | 📊 Summary: ~{len(summary.split())} words"
        status = "✅ Summary generated successfully!"
        
        return summary, stats, status
        
    except Exception as e:
        error_msg = f"""❌ Error during generation: {str(e)}

Troubleshooting:
• Check that model files exist in the Space root directory
• Verify HF_TOKEN is set in Space secrets (for gated models)
• Try using a shorter input text (< 1000 words)

If the error persists, the Space may need more memory. Consider upgrading to GPU hardware.
"""
        return error_msg, "", "❌ Generation failed"

# Example legal cases
EXAMPLES = [
    [
        """The appellant filed a suit for specific performance of contract for sale of commercial property. The respondent-seller agreed to sell the property for Rs. 50 lakhs but later refused citing the appellant's failure to pay the full consideration within the stipulated time of 60 days. The appellant contended that they were ready and willing to perform but needed additional time due to banking delays. The trial court dismissed the suit finding no proof of readiness and willingness. The appellant challenged this decision before the High Court.""",
        256
    ],
    [
        """This is an appeal against conviction under Section 302 IPC for murder. The appellant was convicted by the Sessions Court based on circumstantial evidence. The prosecution case relied on: (1) recovery of the murder weapon from appellant's possession, (2) eyewitness testimony placing appellant at the crime scene, (3) motive arising from property dispute, and (4) absence of credible alibi. The defense argued that the chain of circumstances was incomplete and the evidence was planted. The trial court held all circumstances proved beyond reasonable doubt and convicted the appellant, sentencing him to life imprisonment.""",
        256
    ],
    [
        """The suit concerns title and possession of agricultural land measuring 5 acres. The plaintiff claims ownership through registered sale deed dated 2015. The defendant counterclaims based on adverse possession for 15 years prior to plaintiff's purchase. The defendant produced evidence of cultivation, payment of land revenue, and open possession. The plaintiff argued that defendant was merely a tenant whose possession was permissive. The trial court decreed the suit in favor of plaintiff holding that adverse possession was not established as possession was not hostile to true owner's title.""",
        256
    ]
]

# Build Gradio interface
with gr.Blocks(title="Legal Case Summarizer") as demo:
    # Check hardware on startup
    hardware_status = "🟢 GPU" if torch.cuda.is_available() else "🟡 CPU"
    performance_note = "" if torch.cuda.is_available() else "\n⏳ **Note:** Running on CPU - generation may take 30-60 seconds per summary."
    
    gr.Markdown(f"""
    # ⚖️ Legal Case Summarization Assistant
    ### Powered by LoRA Fine-tuned Gemma-2B {hardware_status}
    
    Generate concise summaries of legal court judgments using AI. This model was fine-tuned on legal case data 
    and achieves 79% BERTScore F1 for semantic accuracy.{performance_note}
    
    ⚠️ **Important:** This model only processes legal case documents. Non-legal content will be rejected.
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📄 Legal Judgment Input")
            
            judgment_input = gr.Textbox(
                label="Enter Legal Judgment",
                placeholder="Paste the full legal judgment here...",
                lines=15,
                max_lines=20
            )
            
            with gr.Row():
                max_length_slider = gr.Slider(
                    minimum=128,
                    maximum=512,
                    value=256,
                    step=64,
                    label="Max Summary Length (tokens)",
                    info="Longer = more detailed summary"
                )
            
            with gr.Row():
                clear_btn = gr.Button("🗑️ Clear", variant="secondary")
                submit_btn = gr.Button("🚀 Generate Summary", variant="primary", scale=2)
        
        with gr.Column(scale=1):
            gr.Markdown("### 📝 Generated Summary")
            
            status_output = gr.Textbox(
                label="Status",
                value="👈 Enter a legal judgment and click 'Generate Summary'",
                lines=1,
                show_label=False
            )
            
            summary_output = gr.Textbox(
                label="Summary",
                placeholder="Your summary will appear here...",
                lines=13,
                max_lines=15
            )
            
            stats_output = gr.Textbox(
                label="Generation Stats",
                lines=1,
                show_label=False
            )
    
    gr.Markdown("### 📚 Example Legal Cases")
    gr.Examples(
        examples=EXAMPLES,
        inputs=[judgment_input, max_length_slider],
        label="Click an example to load it"
    )
    
    gr.Markdown("""
    ---
    ### 📊 Model Performance Metrics
    - **ROUGE-1:** 0.2908 (Vocabulary coverage)
    - **ROUGE-L:** 0.1760 (Structural coherence)  
    - **BERTScore F1:** 0.7902 (Semantic accuracy)
    - **Perplexity:** 6.80 (Model confidence)
    
    ### ⚠️ Disclaimer
    This is an AI-generated summary for research purposes only. Always consult qualified legal professionals for legal advice.
    
    ### 🔗 Links
    - [GitHub Repository](https://github.com/mangaorphy/Domain-Specific-FIned-Tuned-Legal-AI)
    - [Model Details](https://github.com/mangaorphy/Domain-Specific-FIned-Tuned-Legal-AI/blob/main/README.md)
    """)
    
    # Event handlers
    submit_btn.click(
        fn=predict,
        inputs=[judgment_input, max_length_slider],
        outputs=[summary_output, stats_output, status_output]
    )
    
    clear_btn.click(
        fn=lambda: ("", "", "👈 Enter a legal judgment and click 'Generate Summary'"),
        inputs=None,
        outputs=[judgment_input, summary_output, status_output]
    )

# Launch the app
if __name__ == "__main__":
    demo.queue()  # Enable queuing for better performance
    demo.launch(share=False)
