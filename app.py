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
    
    # Patterns that indicate non-legal content (must be specific to avoid false positives)
    non_legal_patterns = [
        (r'\b(recipe|ingredient|cook|bake|oven|cup|tablespoon|teaspoon)\b', 'cooking recipe'),
        (r'\b(software development|app development|download|install|click here|subscribe now)\b', 'technology/tutorial content'),
        (r'\b(how to make|step-by-step guide)\b', 'instructional content'),
        (r'\b(buy now|price tag|discount offer|sale price|shop now|add to cart)\b', 'commercial/shopping content'),
        (r'\b(machine learning model|neural network training|deep learning)\b', 'AI/tech article'),
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
    # Truncate input to fit context window (keep more for better understanding)
    words = judgment_text.split()
    if len(words) > 1000:
        judgment_text_truncated = " ".join(words[:1000])
    else:
        judgment_text_truncated = judgment_text
    
    prompt = f"""Instruction:
Summarize the following legal court judgment in a concise paragraph. Focus on the key facts, legal issues, and outcome.

Input:
{judgment_text_truncated}

Response:
"""
    
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1280
    ).to(model.device)
    
    # Get input length to know where generation starts
    input_length = inputs['input_ids'].shape[1]
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=0.4,  # Slightly higher for more natural language
            top_p=0.9,
            top_k=50,
            do_sample=True,
            repetition_penalty=1.3,  # Higher to prevent repetition
            no_repeat_ngram_size=3,  # Prevent 3-gram repetition
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode only the generated part (excluding the input prompt)
    generated_ids = outputs[0][input_length:]
    summary = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    
    # Cleanup: remove any remaining artifacts
    if summary.startswith("Response:"):
        summary = summary[9:].strip()
    if summary.startswith("Instruction:"):
        summary = summary.split("Response:")[-1].strip() if "Response:" in summary else ""
    
    # Remove common prefixes that might appear
    prefixes_to_remove = ["Summary:", "SUMMARY:", "The summary is:", "Here is the summary:"]
    for prefix in prefixes_to_remove:
        if summary.startswith(prefix):
            summary = summary[len(prefix):].strip()
    
    # Remove HTML tags if present (hallucination artifact)
    import re
    summary = re.sub(r'<[^>]+>', '', summary)
    
    # Remove any leading/trailing quotes or brackets
    summary = summary.strip('"\'[]')
    
    # If still empty or very short, try alternative parsing
    if len(summary) < 10:
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "Response:" in full_text:
            summary = full_text.split("Response:")[-1].strip()
            summary = re.sub(r'<[^>]+>', '', summary)
            summary = summary.strip('"\'[]')
    
    # Capitalize first letter if it's lowercase
    if summary and summary[0].islower():
        summary = summary[0].upper() + summary[1:]
    
    # Ensure proper sentence ending
    if summary and not summary.endswith(('.', '!', '?')):
        summary += '.'
    
    return summary if summary and len(summary) > 10 else "Unable to generate summary. Please try with a shorter input or different text."

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

# Build Gradio interface with custom styling
custom_css = """
    .gradio-container {
        font-family: 'IBM Plex Sans', sans-serif;
    }
    .header-box {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        margin: 0.5rem 0;
    }
    .warning-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
        font-weight: 500;
    }
    .success-box {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    #input-section, #output-section {
        border: 2px solid #e0e7ff;
        border-radius: 12px;
        padding: 1.5rem;
        background: linear-gradient(to bottom, #ffffff 0%, #f8faff 100%);
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    .footer-section {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin-top: 2rem;
    }
"""

with gr.Blocks(title="Legal Case Summarizer", css=custom_css, theme=gr.themes.Soft()) as demo:
    # Check hardware on startup
    hardware_status = "🟢 GPU" if torch.cuda.is_available() else "🟡 CPU"
    performance_note = "" if torch.cuda.is_available() else "\n\n<div class='warning-box'>⏱️ <strong>Performance Note:</strong> Running on CPU - generation may take 30-60 seconds per summary. For faster results, upgrade to GPU hardware.</div>"
    
    gr.HTML(f"""
        <div class='header-box'>
            <h1 style='margin: 0; font-size: 2.5rem; font-weight: 700;'>
                ⚖️ Legal Case Summarization Assistant
            </h1>
            <h3 style='margin: 0.5rem 0 0 0; font-weight: 400; opacity: 0.95;'>
                Powered by LoRA Fine-tuned Gemma-2B • {hardware_status}
            </h3>
            <p style='margin: 1rem 0 0 0; font-size: 1.1rem; opacity: 0.9;'>
                Generate concise, AI-powered summaries of legal court judgments with 79% semantic accuracy
            </p>
        </div>
    """)
    
    if not torch.cuda.is_available():
        gr.HTML(performance_note)
    
    gr.HTML("""
        <div class='warning-box' style='text-align: center;'>
            ⚠️ <strong>Important:</strong> This model only processes legal case documents. Non-legal content will be automatically rejected.
        </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=1, elem_id="input-section"):
            gr.HTML("<h3 style='color: #1e3c72; margin-top: 0;'>📄 Legal Judgment Input</h3>")
            
            judgment_input = gr.Textbox(
                label="",
                placeholder="📋 Paste the full legal judgment here...\n\nExample: 'The appellant filed a suit for specific performance of contract...'",
                lines=15,
                max_lines=20,
                show_label=False
            )
            
            with gr.Row():
                max_length_slider = gr.Slider(
                    minimum=128,
                    maximum=512,
                    value=256,
                    step=64,
                    label="📏 Max Summary Length (tokens)",
                    info="Longer = more detailed summary"
                )
            
            with gr.Row():
                clear_btn = gr.Button("🗑️ Clear", variant="secondary", size="lg")
                submit_btn = gr.Button("🚀 Generate Summary", variant="primary", scale=2, size="lg")
        
        with gr.Column(scale=1, elem_id="output-section"):
            gr.HTML("<h3 style='color: #1e3c72; margin-top: 0;'>📝 Generated Summary</h3>")
            
            status_output = gr.Textbox(
                label="",
                value="👈 Enter a legal judgment and click 'Generate Summary'",
                lines=1,
                show_label=False,
                container=False
            )
            
            summary_output = gr.Textbox(
                label="",
                placeholder="✨ Your AI-generated summary will appear here...",
                lines=13,
                max_lines=15,
                show_label=False
            )
            
            stats_output = gr.Textbox(
                label="",
                lines=2,
                show_label=False,
                container=False
            )
    
    gr.HTML("<h3 style='color: #1e3c72; margin: 2rem 0 1rem 0; text-align: center;'>📚 Example Legal Cases</h3>")
    
    gr.Examples(
        examples=EXAMPLES,
        inputs=[judgment_input, max_length_slider],
        label=None
    )
    
    gr.HTML("""
        <div class='footer-section'>
            <h3 style='color: #1e3c72; margin-top: 0; text-align: center;'>📊 Model Performance Metrics</h3>
            <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin: 1.5rem 0;'>
                <div class='metric-card'>
                    <div style='font-size: 2rem; font-weight: 700;'>29.08%</div>
                    <div style='opacity: 0.9;'>ROUGE-1</div>
                    <div style='font-size: 0.85rem; opacity: 0.8;'>Vocabulary coverage</div>
                </div>
                <div class='metric-card'>
                    <div style='font-size: 2rem; font-weight: 700;'>17.60%</div>
                    <div style='opacity: 0.9;'>ROUGE-L</div>
                    <div style='font-size: 0.85rem; opacity: 0.8;'>Structural coherence</div>
                </div>
                <div class='metric-card'>
                    <div style='font-size: 2rem; font-weight: 700;'>79.02%</div>
                    <div style='opacity: 0.9;'>BERTScore F1</div>
                    <div style='font-size: 0.85rem; opacity: 0.8;'>Semantic accuracy</div>
                </div>
                <div class='metric-card'>
                    <div style='font-size: 2rem; font-weight: 700;'>6.80</div>
                    <div style='opacity: 0.9;'>Perplexity</div>
                    <div style='font-size: 0.85rem; opacity: 0.8;'>Model confidence</div>
                </div>
            </div>
            
            <div style='background: white; padding: 1.5rem; border-radius: 8px; margin-top: 1.5rem;'>
                <h4 style='color: #1e3c72; margin-top: 0;'>⚠️ Disclaimer</h4>
                <p style='color: #4b5563; margin: 0.5rem 0;'>
                    This is an AI-generated summary for research and educational purposes only. 
                    Always consult qualified legal professionals for legal advice. The summaries 
                    should not be used as a substitute for professional legal counsel.
                </p>
                
                <h4 style='color: #1e3c72; margin-top: 1.5rem;'>🔗 Resources</h4>
                <div style='display: flex; gap: 1rem; flex-wrap: wrap; margin-top: 0.5rem;'>
                    <a href='https://github.com/mangaorphy/Domain-Specific-FIned-Tuned-Legal-AI' 
                       style='color: #2563eb; text-decoration: none; font-weight: 500;'
                       target='_blank'>
                        📂 GitHub Repository
                    </a>
                    <span style='color: #d1d5db;'>|</span>
                    <a href='https://github.com/mangaorphy/Domain-Specific-FIned-Tuned-Legal-AI/blob/main/README.md' 
                       style='color: #2563eb; text-decoration: none; font-weight: 500;'
                       target='_blank'>
                        📖 Model Documentation
                    </a>
                    <span style='color: #d1d5db;'>|</span>
                    <span style='color: #6b7280;'>
                        🏗️ Built with LoRA & Gemma-2B
                    </span>
                </div>
            </div>
        </div>
    """)
    
    # Event handlers
    submit_btn.click(
        fn=predict,
        inputs=[judgment_input, max_length_slider],
        outputs=[summary_output, stats_output, status_output]
    )
    
    clear_btn.click(
        fn=lambda: ("", "", "", "👈 Enter a legal judgment and click 'Generate Summary'"),
        inputs=None,
        outputs=[judgment_input, summary_output, stats_output, status_output]
    )

# Launch the app
if __name__ == "__main__":
    demo.queue(max_size=20)  # Enable queuing for better performance with multiple users
    demo.launch(
        share=False,
        show_api=False,
        favicon_path=None  # You can add a custom favicon here
    )
