"""
Qualitative Testing Script for Legal Case Summarization Model

This script tests:
1. In-domain performance: Legal cases (should summarize correctly)
2. Out-of-domain performance: Non-legal text (should handle gracefully)
3. Edge cases: Short text, very long text, gibberish
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_PATH = "./model_outputs"  # Your trained LoRA adapters
BASE_MODEL_NAME = "google/gemma-2b"
MAX_NEW_TOKENS = 256

# ============================================================================
# LOAD MODEL
# ============================================================================

def load_model():
    """Load the fine-tuned legal summarization model"""
    print("Loading model...")
    
    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Load LoRA adapters
    model = PeftModel.from_pretrained(base_model, MODEL_PATH)
    model.eval()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("✓ Model loaded successfully\n")
    return model, tokenizer

def generate_summary(model, tokenizer, text, max_new_tokens=256):
    """Generate summary for input text"""
    prompt = f"""Instruction:
Summarize the following legal court judgment.

Input:
{text}

Response:
"""
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1280).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
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

# ============================================================================
# TEST CASES
# ============================================================================

# 1. IN-DOMAIN TEST: Real Legal Case
IN_DOMAIN_TEST_1 = """
In the matter of Smith v. Johnson Corporation, the plaintiff alleged breach of contract 
concerning a commercial lease agreement dated January 15, 2020. The plaintiff claimed 
that the defendant failed to maintain the premises in accordance with Section 4.2 of 
the lease, specifically citing water damage from roof leaks that occurred in March 2021. 
The defendant argued that the maintenance clause was ambiguous and that the plaintiff 
failed to provide adequate notice as required under Section 8.1. After reviewing the 
evidence, including maintenance records and photographic documentation, the court found 
that the defendant had indeed breached the maintenance obligations. The lease agreement 
clearly specified that the landlord must keep the roof in good repair. The court awarded 
damages of $45,000 to the plaintiff for repairs and lost business income during the 
closure period. The defendant's motion to dismiss was denied.
"""

# 2. IN-DOMAIN TEST: Contract Dispute
IN_DOMAIN_TEST_2 = """
The Court of Appeals reviewed the lower court's decision in Estate of Williams v. 
First National Bank, Executor. The case centered on whether the executor properly 
distributed assets according to the testamentary provisions in the deceased's will. 
The appellants, three adult children of the deceased, challenged the executor's 
interpretation of Clause 7, which referenced "personal property" without explicit 
definition. The bank had distributed only furniture and jewelry, excluding valuable 
artwork and collectibles. The appellate court found this interpretation overly narrow 
and inconsistent with the testator's intent as evidenced by correspondence and prior 
estate planning documents. The court reversed the lower court's judgment and remanded 
for redistribution of all personal property including artwork valued at approximately 
$200,000. Court costs were assessed against the executor.
"""

# 3. OUT-OF-DOMAIN TEST: Technology Article
OUT_OF_DOMAIN_TEST_1 = """
Artificial intelligence has revolutionized the tech industry over the past decade. 
Machine learning models, particularly large language models, have demonstrated 
remarkable capabilities in natural language processing, code generation, and creative 
tasks. Companies like OpenAI, Google, and Anthropic have invested billions in 
developing increasingly sophisticated AI systems. These systems are trained on vast 
datasets and utilize transformer architectures with billions of parameters. Applications 
range from chatbots and virtual assistants to medical diagnosis and autonomous vehicles. 
However, concerns remain about AI safety, bias, and potential job displacement.
"""

# 4. OUT-OF-DOMAIN TEST: Cooking Recipe
OUT_OF_DOMAIN_TEST_2 = """
To make chocolate chip cookies, start by preheating your oven to 375°F. In a large 
bowl, cream together 1 cup of softened butter with 3/4 cup white sugar and 3/4 cup 
brown sugar until fluffy. Beat in 2 eggs and 2 teaspoons vanilla extract. In a 
separate bowl, combine 2 1/4 cups all-purpose flour, 1 teaspoon baking soda, and 
1 teaspoon salt. Gradually blend the dry ingredients into the wet mixture. Stir in 
2 cups chocolate chips. Drop rounded tablespoons onto ungreased cookie sheets. 
Bake for 9-11 minutes or until golden brown. Cool on baking sheet for 2 minutes 
before transferring to a wire rack.
"""

# 5. EDGE CASE: Very Short Text
EDGE_CASE_SHORT = "The court ruled in favor of the plaintiff. Case dismissed."

# 6. EDGE CASE: Gibberish/Nonsensical
EDGE_CASE_GIBBERISH = """
The quantum fluctuation of the purple monkey dishwasher exceeded the statutory 
limitations concerning recursive banana protocols. The defendant's motion to 
sublimate the crystalline evidence was overruled by the honorable Judge Spaghetti. 
Damages awarded: seventeen unicorns and a basket of theoretical physics equations.
"""

# ============================================================================
# RUN TESTS
# ============================================================================

def run_qualitative_tests():
    """Run comprehensive qualitative testing"""
    
    # Load model
    model, tokenizer = load_model()
    
    test_cases = [
        ("IN-DOMAIN: Contract Breach Case", IN_DOMAIN_TEST_1, True),
        ("IN-DOMAIN: Estate Dispute Case", IN_DOMAIN_TEST_2, True),
        ("OUT-OF-DOMAIN: Technology Article", OUT_OF_DOMAIN_TEST_1, False),
        ("OUT-OF-DOMAIN: Cooking Recipe", OUT_OF_DOMAIN_TEST_2, False),
        ("EDGE CASE: Very Short Text", EDGE_CASE_SHORT, True),
        ("EDGE CASE: Gibberish/Nonsensical", EDGE_CASE_GIBBERISH, False),
    ]
    
    print("="*80)
    print("QUALITATIVE TESTING: LEGAL CASE SUMMARIZATION MODEL")
    print("="*80)
    print()
    
    for i, (test_name, test_input, is_legal) in enumerate(test_cases, 1):
        print(f"\n{'='*80}")
        print(f"TEST {i}: {test_name}")
        print("="*80)
        print(f"\nExpected Behavior: {'Should summarize legal content' if is_legal else 'Should handle gracefully (may refuse or provide generic response)'}")
        print(f"\n{'─'*80}")
        print(f"INPUT (first 300 chars):")
        print(f"{'─'*80}")
        print(test_input[:300].strip() + ("..." if len(test_input) > 300 else ""))
        
        print(f"\n{'─'*80}")
        print(f"MODEL OUTPUT:")
        print(f"{'─'*80}")
        
        try:
            summary = generate_summary(model, tokenizer, test_input, MAX_NEW_TOKENS)
            print(summary)
            
            # Analysis
            print(f"\n{'─'*80}")
            print(f"ANALYSIS:")
            print(f"{'─'*80}")
            
            # Check response characteristics
            response_length = len(summary.split())
            has_legal_terms = any(term in summary.lower() for term in 
                                ['court', 'plaintiff', 'defendant', 'judge', 'ruling', 
                                 'case', 'legal', 'judgment', 'damages', 'motion'])
            
            print(f"✓ Response length: {response_length} words")
            print(f"✓ Contains legal terminology: {'Yes' if has_legal_terms else 'No'}")
            
            if is_legal:
                if has_legal_terms and response_length > 20:
                    print("✓ PASS: Model generated appropriate legal summary")
                else:
                    print("⚠️  CONCERN: Summary may lack legal depth")
            else:
                if has_legal_terms and response_length > 50:
                    print("⚠️  CONCERN: Model treating non-legal text as legal case")
                    print("   (Model should refuse or provide generic response)")
                else:
                    print("✓ PASS: Model handled out-of-domain input appropriately")
            
        except Exception as e:
            print(f"❌ ERROR: {e}")
        
        print(f"\n{'='*80}\n")
    
    # Summary and recommendations
    print("\n" + "="*80)
    print("QUALITATIVE TESTING SUMMARY")
    print("="*80)
    print("""
KEY OBSERVATIONS TO DOCUMENT:

1. IN-DOMAIN PERFORMANCE:
   - Does the model correctly identify key legal elements (parties, issues, rulings)?
   - Are summaries concise and factually accurate?
   - Does it capture the most important legal points?

2. OUT-OF-DOMAIN HANDLING:
   - Does it refuse non-legal inputs or try to summarize them?
   - Does it inappropriately apply legal framing to non-legal text?
   - How "confused" does it get with completely unrelated content?

3. EDGE CASES:
   - Can it handle very short inputs without hallucinating details?
   - Does it produce nonsensical outputs for nonsensical inputs?

RECOMMENDATIONS:
- ✓ Document specific examples of good/bad performance
- ✓ Test with real-world inputs from your target use case
- ✓ Check for hallucinations (made-up facts not in input)
- ✓ Verify the model doesn't generate biased or harmful content
- ✓ Test multilingual inputs if applicable to your domain
    """)
    print("="*80)

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    run_qualitative_tests()
