"""
Safe Legal Summarization with Input Validation

This script adds safety mechanisms to prevent the model from processing non-legal text.

APPROACHES:
1. Keyword-based filtering (fast, simple)
2. Classifier-based validation (more accurate)
3. Embedding-based similarity (most robust)
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import pipeline
from peft import PeftModel
import re

# ============================================================================
# APPROACH 1: KEYWORD-BASED FILTERING (Fast & Simple)
# ============================================================================

class KeywordBasedValidator:
    """
    Simple rule-based validator using legal terminology.
    Pros: Fast, no additional model needed
    Cons: Can be fooled by keyword stuffing
    """
    
    def __init__(self):
        # Legal terminology that should appear in court judgments
        self.legal_keywords = {
            'court', 'plaintiff', 'defendant', 'appellant', 'appellee',
            'petitioner', 'respondent', 'judge', 'justice', 'magistrate',
            'verdict', 'judgment', 'ruling', 'order', 'decree', 'opinion',
            'motion', 'complaint', 'petition', 'brief', 'testimony',
            'evidence', 'trial', 'hearing', 'case', 'proceeding',
            'statute', 'law', 'legal', 'liability', 'damages', 'relief',
            'adjudicate', 'jurisdiction', 'precedent', 'counsel', 'attorney'
        }
        
        # Anti-patterns that suggest non-legal content
        self.non_legal_patterns = [
            r'\b(recipe|ingredient|cook|bake|oven)\b',
            r'\b(click here|subscribe|follow us|website)\b',
            r'\b(software|app|download|install|update)\b',
            r'\b(tutorial|how to make|step \d+)\b',
            r'\b(product|price|buy now|discount|sale)\b',
        ]
        
    def validate(self, text):
        """
        Returns: (is_valid, confidence_score, reason)
        """
        text_lower = text.lower()
        
        # Check for non-legal patterns
        for pattern in self.non_legal_patterns:
            if re.search(pattern, text_lower):
                return False, 0.1, f"Contains non-legal pattern: {pattern}"
        
        # Count legal keywords
        word_count = len(text.split())
        legal_word_count = sum(1 for keyword in self.legal_keywords if keyword in text_lower)
        
        # Calculate confidence (at least 0.5% of words should be legal terms)
        if word_count < 50:
            return False, 0.2, "Text too short for legal judgment"
        
        legal_density = legal_word_count / word_count
        
        if legal_density < 0.005:  # Less than 0.5% legal terms
            return False, legal_density, f"Insufficient legal terminology ({legal_word_count}/{word_count} words)"
        
        if legal_word_count < 3:  # Must have at least 3 legal terms
            return False, 0.3, f"Only {legal_word_count} legal terms found"
        
        confidence = min(1.0, legal_density * 100)  # Scale up
        return True, confidence, f"Found {legal_word_count} legal terms"


# ============================================================================
# APPROACH 2: CLASSIFIER-BASED VALIDATION (More Accurate)
# ============================================================================

class ClassifierBasedValidator:
    """
    Use a zero-shot classifier to detect if text is legal content.
    Pros: More accurate than keywords
    Cons: Slower, requires additional model
    """
    
    def __init__(self):
        print("Loading zero-shot classifier...")
        self.classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=0 if torch.cuda.is_available() else -1
        )
        self.candidate_labels = [
            "legal court judgment",
            "legal case document", 
            "court ruling",
            "technology article",
            "cooking recipe",
            "general text",
            "news article"
        ]
        print("✓ Classifier loaded")
    
    def validate(self, text, threshold=0.5):
        """
        Returns: (is_valid, confidence_score, reason)
        """
        # Take first 512 tokens to avoid model limits
        text_truncated = " ".join(text.split()[:512])
        
        result = self.classifier(
            text_truncated,
            candidate_labels=self.candidate_labels,
            multi_label=False
        )
        
        top_label = result['labels'][0]
        top_score = result['scores'][0]
        
        is_legal = any(legal_term in top_label.lower() 
                      for legal_term in ['legal', 'court', 'judgment', 'ruling', 'case'])
        
        if is_legal and top_score >= threshold:
            return True, top_score, f"Classified as: {top_label}"
        else:
            return False, top_score, f"Classified as: {top_label} (non-legal)"


# ============================================================================
# APPROACH 3: EMBEDDING-BASED SIMILARITY (Most Robust)
# ============================================================================

class EmbeddingBasedValidator:
    """
    Compare input embeddings to known legal document embeddings.
    Pros: Most robust, learns from examples
    Cons: Requires pre-computed reference embeddings
    """
    
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        print("Loading sentence transformer...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Reference legal texts (you'd expand this with real examples)
        self.legal_references = [
            "The Court of Appeals reviewed the decision in Smith v Johnson Corporation regarding breach of contract.",
            "The plaintiff alleged that the defendant failed to comply with statutory obligations under Section 42.",
            "After considering the evidence and testimony, the court ruled in favor of the petitioner.",
            "The judgment was entered ordering the respondent to pay damages in the amount of $50,000."
        ]
        
        # Compute reference embeddings
        print("Computing reference embeddings...")
        self.legal_embeddings = self.model.encode(self.legal_references)
        print("✓ Validator ready")
    
    def validate(self, text, threshold=0.4):
        """
        Returns: (is_valid, confidence_score, reason)
        """
        # Get embedding for input text
        text_embedding = self.model.encode([text])[0]
        
        # Calculate cosine similarity with legal references
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(
            [text_embedding], 
            self.legal_embeddings
        )[0]
        
        max_similarity = float(similarities.max())
        avg_similarity = float(similarities.mean())
        
        is_valid = max_similarity >= threshold
        
        return is_valid, max_similarity, f"Similarity to legal docs: {max_similarity:.3f} (avg: {avg_similarity:.3f})"


# ============================================================================
# SAFE SUMMARIZER (Combines Validation + Generation)
# ============================================================================

class SafeLegalSummarizer:
    """
    Wrapper that validates input before generating summaries.
    """
    
    def __init__(self, model_path="./model_outputs", validator_type="keyword"):
        """
        Args:
            model_path: Path to your trained LoRA adapters
            validator_type: "keyword", "classifier", or "embedding"
        """
        self.model_path = model_path
        self.validator_type = validator_type
        
        # Load validator
        print(f"\nInitializing {validator_type} validator...")
        if validator_type == "keyword":
            self.validator = KeywordBasedValidator()
            print("✓ Keyword validator ready (fast, rule-based)")
        elif validator_type == "classifier":
            self.validator = ClassifierBasedValidator()
            print("✓ Classifier validator ready (accurate, slower)")
        elif validator_type == "embedding":
            self.validator = EmbeddingBasedValidator()
            print("✓ Embedding validator ready (most robust)")
        else:
            raise ValueError("validator_type must be 'keyword', 'classifier', or 'embedding'")
        
        # Load model and tokenizer
        print(f"\nLoading legal summarization model from {model_path}...")
        self.model, self.tokenizer = self._load_model()
        print("✓ Model loaded and ready\n")
    
    def _load_model(self):
        """Load the fine-tuned model"""
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        
        base_model = AutoModelForCausalLM.from_pretrained(
            "google/gemma-2b",
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        
        model = PeftModel.from_pretrained(base_model, self.model_path)
        model.eval()
        
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        return model, tokenizer
    
    def summarize(self, text, force=False, threshold=0.5, max_new_tokens=256):
        """
        Safe summarization with input validation.
        
        Args:
            text: Input text to summarize
            force: Skip validation (use with caution!)
            threshold: Validation confidence threshold
            max_new_tokens: Maximum summary length
            
        Returns:
            dict with keys: success, summary, validation_result
        """
        # Validate input
        is_valid, confidence, reason = self.validator.validate(text)
        
        validation_result = {
            'is_valid': is_valid,
            'confidence': confidence,
            'reason': reason,
            'validator_type': self.validator_type
        }
        
        # Reject if not valid (unless forced)
        if not is_valid and not force:
            return {
                'success': False,
                'summary': None,
                'validation_result': validation_result,
                'message': f"❌ Input rejected: {reason}\n   Confidence: {confidence:.2%}"
            }
        
        # Check confidence threshold
        if is_valid and confidence < threshold and not force:
            return {
                'success': False,
                'summary': None,
                'validation_result': validation_result,
                'message': f"⚠️  Input rejected: Confidence too low ({confidence:.2%} < {threshold:.0%})\n   {reason}"
            }
        
        # Generate summary
        try:
            prompt = f"""Instruction:
Summarize the following legal court judgment.

Input:
{text}

Response:
"""
            
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=1280
            ).to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            if "Response:" in generated_text:
                summary = generated_text.split("Response:")[-1].strip()
            else:
                summary = generated_text[len(prompt):].strip()
            
            return {
                'success': True,
                'summary': summary,
                'validation_result': validation_result,
                'message': f"✓ Validation passed ({confidence:.2%}): {reason}"
            }
            
        except Exception as e:
            return {
                'success': False,
                'summary': None,
                'validation_result': validation_result,
                'message': f"❌ Generation error: {str(e)}"
            }


# ============================================================================
# DEMO / TESTING
# ============================================================================

def demo():
    """Demonstrate safe summarization"""
    
    # Test cases
    test_cases = [
        ("Legal Case", """The Court of Appeals reviewed the decision in Smith v Johnson Corporation 
        regarding breach of contract. The plaintiff alleged that the defendant failed to maintain 
        the premises. After reviewing evidence, the court found the defendant breached obligations. 
        The court awarded damages of $45,000 to the plaintiff."""),
        
        ("Tech Article", """Artificial intelligence has revolutionized the tech industry. 
        Machine learning models have demonstrated remarkable capabilities in NLP and code generation. 
        Companies are investing billions in AI development."""),
        
        ("Recipe", """To make chocolate chip cookies, preheat oven to 375°F. Cream butter with sugar, 
        add eggs and vanilla. Mix flour, baking soda, and salt. Combine wet and dry ingredients. 
        Add chocolate chips. Bake for 9-11 minutes."""),
    ]
    
    # Try all three validators
    for validator_type in ["keyword", "classifier", "embedding"]:
        print("\n" + "="*80)
        print(f"TESTING WITH {validator_type.upper()} VALIDATOR")
        print("="*80)
        
        summarizer = SafeLegalSummarizer(validator_type=validator_type)
        
        for test_name, test_text in test_cases:
            print(f"\n{'─'*80}")
            print(f"TEST: {test_name}")
            print('─'*80)
            print(f"Input: {test_text[:100]}...")
            print()
            
            result = summarizer.summarize(test_text)
            print(result['message'])
            
            if result['success']:
                print(f"\nSummary: {result['summary'][:200]}...")
            
            print('─'*80)


if __name__ == "__main__":
    # Quick example with keyword validator (fastest)
    print("="*80)
    print("SAFE LEGAL SUMMARIZATION DEMO")
    print("="*80)
    
    summarizer = SafeLegalSummarizer(validator_type="keyword")
    
    # Test with legal case
    legal_text = """The Court reviewed Smith v Johnson. Plaintiff alleged breach of contract 
    concerning lease agreement. The court found defendant breached maintenance obligations. 
    Damages of $45,000 awarded to plaintiff."""
    
    print("\n1. Testing with LEGAL case:")
    print("-" * 40)
    result = summarizer.summarize(legal_text)
    print(result['message'])
    if result['success']:
        print(f"Summary: {result['summary']}")
    
    # Test with non-legal content
    print("\n\n2. Testing with NON-LEGAL content (recipe):")
    print("-" * 40)
    recipe = """To make cookies, mix butter and sugar. Add eggs and vanilla. 
    Combine flour and baking soda. Add chocolate chips. Bake at 375°F for 10 minutes."""
    
    result = summarizer.summarize(recipe)
    print(result['message'])
    if result['success']:
        print(f"Summary: {result['summary']}")
    else:
        print("✓ Model correctly refused to process non-legal content!")
