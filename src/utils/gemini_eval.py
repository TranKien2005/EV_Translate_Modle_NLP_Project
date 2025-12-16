"""
Gemini Evaluation Utility.
Uses Google's Gemini API to evaluate translation quality.
"""

import os
import time
import google.generativeai as genai
from typing import List, Dict, Optional
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class GeminiEvaluator:
    """
    Evaluates translation quality using Gemini API.
    """
    
    def __init__(self, api_key: str = None, model_name: str = "gemini-pro"):
        """
        Initialize Gemini Evaluator.
        
        Args:
            api_key: Google AI Studio API Key
            model_name: Model version to use
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            print("⚠ Warning: GEMINI_API_KEY not found. Gemini evaluation will be disabled.")
            self.model = None
        else:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(model_name)
    
    def evaluate_single(self, source: str, reference: str, candidate: str) -> float:
        """
        Evaluate a single translation pair.
        
        Returns:
            Score (0-100)
        """
        if not self.model:
            return 0.0
            
        prompt = f"""
        Act as a professional translator. Evaluate the quality of the Vietnamese translation for the given English source.
        
        Source (English): "{source}"
        Reference (Vietnamese): "{reference}"
        Candidate (Vietnamese): "{candidate}"
        
        Rate the Candidate translation on a scale of 0 to 100 based on:
        1. Accuracy (meaning preservation)
        2. Fluency (grammar and naturalness)
        
        Provide your score as a single integer number. Do not provide any explanation, just the number.
        """
        
        try:
            response = self.model.generate_content(prompt)
            text = response.text.strip()
            # Extract number from text (in case model outputs extra chars)
            import re
            match = re.search(r'\d+', text)
            if match:
                score = float(match.group())
                return min(100.0, max(0.0, score))
            return 0.0
        except Exception as e:
            print(f"Error evaluating sample: {e}")
            return 0.0

    def evaluate_batch(
        self, 
        sources: List[str], 
        references: List[str], 
        candidates: List[str],
        max_samples: int = 200
    ) -> Dict[str, float]:
        """
        Evaluate a batch of translations.
        
        Args:
            sources: English source texts
            references: Vietnamese reference texts
            candidates: Vietnamese model predictions
            max_samples: Limit number of samples to save tokens/time
            
        Returns:
            Dictionary with average score
        """
        if not self.model:
            return {"gemini_score": 0.0}
            
        # Limit samples
        n = min(len(sources), max_samples)
        if n < len(sources):
            print(f"Limiting Gemini evaluation to first {n} samples...")
            
        total_score = 0.0
        valid_samples = 0
        
        print("Running Gemini Evaluation...")
        for i in tqdm(range(n)):
            score = self.evaluate_single(sources[i], references[i], candidates[i])
            total_score += score
            valid_samples += 1
            # Rate limit handling (simple sleep)
            time.sleep(1.0) 
            
        avg_score = total_score / valid_samples if valid_samples > 0 else 0.0
        
        return {
            "gemini_score": avg_score,
            "num_samples": valid_samples
        }
