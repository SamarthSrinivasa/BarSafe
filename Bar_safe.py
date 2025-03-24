import os
import json
import requests
import numpy as np
import torch
import glob
import math
import base64
from PIL import Image
from detoxify import Detoxify
from transformers import CLIPProcessor, CLIPModel
from typing import Dict, Tuple, Optional

# ============= CONFIGURATION =============
# Replace these with your actual credentials
SIGHTENGINE_USER = "383771199"
SIGHTENGINE_SECRET = "Xt5PXbX5ty2GERRKM4rqmPSYga32SgfV"
OPENAI_API_KEY = "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"
DEFAULT_MULTIMODAL_MODEL = "gpt4v"
# Set to True if you don't have Sightengine credentials
# USE_MOCK_IMAGE_SCORER = False
# ========================================

class SafetyScorer:
    """Handles scoring of text and image inputs for safety concerns."""
    
    def __init__(self):
        """Initialize the safety scorer with models."""
        # Load Detoxify for text scoring
        print("Loading Detoxify model (most robust)...")
        self.detoxify_model = Detoxify('unbiased')
        
        # Don't load CLIP model yet - will load only when needed
        self.clip_model = None
        self.clip_processor = None
        
        print("Text safety model loaded successfully!")
    
    def load_clip_model(self):
        """Load the CLIP model on demand."""
        if self.clip_model is None:
            print("Loading CLIP model...")
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            print("CLIP model loaded successfully!")
    
    def score_text(self, text: str) -> Dict[str, float]:
        """Score text input for safety concerns using Detoxify."""
        scores = self.detoxify_model.predict(text)
        # Convert from numpy to Python native types for JSON serialization
        return {k: float(v) for k, v in scores.items()}
    
    def score_image(self, image_path: str) -> Dict[str, float]:
        """Score image for safety concerns using Sightengine API."""
        try:
            with open(image_path, 'rb') as f:
                files = {'media': f}
                params = {
                    'api_user': SIGHTENGINE_USER,
                    'api_secret': SIGHTENGINE_SECRET,
                    'models': 'nudity,wad,offensive'  # Removed 'drugs' as it caused an error
                }
                
                response = requests.post('https://api.sightengine.com/1.0/check.json', 
                                      files=files, data=params)
                
                if response.status_code == 200:
                    output = response.json()
                    
                    # Extract scores from API response
                    scores = {
                        'nsfw': output['nudity']['raw'],
                        'explicit': max(output.get('nudity', {}).get('sexual_activity', 0), 
                                      output.get('nudity', {}).get('sexual_display', 0)),
                        'gore': output.get('offensive', {}).get('gore', 0),
                        'weapon': output.get('weapon', 0),
                        'alcohol': output.get('alcohol', 0),
                        'offensive': output.get('offensive', {}).get('prob', 0)
                    }
                    return scores
                else:
                    print(f"Sightengine API error: {response.status_code}, {response.text}")
                    raise Exception(f"Sightengine API returned status code {response.status_code}")
                    
        except Exception as e:
            print(f"Error using Sightengine API: {e}")
            raise Exception("Failed to score image. Make sure your Sightengine credentials are correct.")
    
    def safety_constraint(self, text_scores: Dict[str, float], image_scores: Dict[str, float], 
                        C: float = 1.0, lambda_val: float = 1.0) -> bool:
        """
        Custom safety constraint function based on Toxicity Barrier Function concepts.
        
        Parameters:
        - text_scores: Dictionary of text toxicity scores
        - image_scores: Dictionary of image toxicity scores
        - C (Threshold Ceiling): 
            - Controls the overall safety threshold
            - HIGHER C = MORE PERMISSIVE content filtering
            - LOWER C = MORE RESTRICTIVE content filtering
            - Conceptual range: 0.0 to 1.0
            - Examples:
                * C = 0.5: Very low tolerance for combined toxicity
                * C = 1.0: Standard safety threshold
                * C = 2.0: More lenient safety threshold
        
        - lambda_val (Adaptive Sensitivity): 
            - Controls how quickly the safety threshold changes
            - HIGHER lambda = FASTER threshold reduction
            - LOWER lambda = SLOWER threshold reduction
            - Impacts the exponential decay of the safety boundary
            - Conceptual range: 0.0 to 2.0
            - Examples:
                * lambda = 0.1: Very rapid threshold reduction
                * lambda = 1.0: Standard adaptive sensitivity
                * lambda = 2.0: Gradual threshold reduction
        
        Returns:
        - True if input passes safety constraints, False otherwise
        """
        # Get maximum toxicity scores for each modality
        ti = max(image_scores.values()) if image_scores else 0.0
        tt = max(text_scores.values()) if text_scores else 0.0
        
        # Ensure toxicity scores are between 0 and 1
        ti = min(max(ti, 0.0), 1.0)
        tt = min(max(tt, 0.0), 1.0)
        
        # Print individual toxicity scores for monitoring
        print(f"Image Toxicity Score (Ti): {ti:.4f}")
        print(f"Text Toxicity Score (Tt): {tt:.4f}")
        
        # Calculate the dynamic safety threshold
        # This uses an exponential decay to create an adaptive threshold
        # As lambda increases, the threshold reduces more quickly
        # As C increases, the overall safety threshold becomes more permissive
        safety_threshold = C - C * math.exp(-lambda_val)
        
        # Compute the barrier function
        # B = Ti + Tt - (C - C * e^(-λ))
        # - When B <= 0, content is considered safe
        # - When B > 0, content is considered unsafe and filtered out
        B = ti + tt - safety_threshold
        
        # Print barrier function details for debugging
        print(f"Safety Threshold: {safety_threshold:.4f}")
        print(f"Barrier Function B(Ti, Tt): {B:.4f}")
        
        # Content is safe if B <= 0
        is_safe = B <= 0
        print(f"Content is {'SAFE' if is_safe else 'UNSAFE'}")
        
        return is_safe
        # ===============================================
    
    def process_clip(self, text: str, image_path: str):
        """Process inputs with CLIP after safety check passed."""
        # Load CLIP model if not already loaded
        self.load_clip_model()
        
        # Process with CLIP
        image = Image.open(image_path)
        inputs = self.clip_processor(
            text=[text],
            images=image,
            return_tensors="pt",
            padding=True
        )
        
        with torch.no_grad():
            outputs = self.clip_model(**inputs)
            clip_output = {
                "text_embeds": outputs.text_embeds,
                "image_embeds": outputs.image_embeds,
                "logits_per_image": outputs.logits_per_image,
                "logits_per_text": outputs.logits_per_text
            }
            
        return clip_output


class GPT4VisionChat:
    """Handles interactions with GPT-4 Vision API."""
    
    def __init__(self):
        """Initialize the GPT-4 Vision interface."""
        if not OPENAI_API_KEY or OPENAI_API_KEY == "YOUR_OPENAI_API_KEY":
            raise ValueError("OpenAI API key is required for GPT-4 Vision")
        self.conversation_history = []
        print("GPT-4 Vision API ready to use")
    
    def encode_image(self, image_path: str) -> str:
        """Encode image as base64 for API requests."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def chat(self, text: str, image_path: str) -> str:
        """Send a message to GPT-4 Vision API."""
        base64_image = self.encode_image(image_path)
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}"
        }
        
        payload = {
            "model": "gpt-4o",  # Updated to current model with vision capabilities
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that can see and discuss images."
                }
            ] + self.conversation_history + [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 100
        }
        
        response = requests.post("https://api.openai.com/v1/chat/completions", 
                              headers=headers, 
                              json=payload)
        
        if response.status_code == 200:
            response_json = response.json()
            assistant_message = response_json["choices"][0]["message"]["content"]
            
            # Update conversation history
            self.conversation_history.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": text}
                    # Note: we don't store the image in history to avoid repeated transfers
                ]
            })
            self.conversation_history.append({
                "role": "assistant",
                "content": assistant_message
            })
            
            return assistant_message
        else:
            print(f"Error with GPT-4 Vision API: {response.status_code}")
            print(response.text)
            return f"Error communicating with GPT-4 Vision API: {response.status_code}"
    
    def clear_history(self):
        """Clear the conversation history."""
        self.conversation_history = []
        print("Conversation history cleared.")


def browse_images(directory='.'):
    """Helper function to browse image files in a directory."""
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']
    
    # Get all image files in the directory
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(directory, f'*{ext}')))
        image_files.extend(glob.glob(os.path.join(directory, f'*{ext.upper()}')))
    
    if not image_files:
        print(f"No image files found in {directory}")
        return None
        
    # Sort files alphabetically
    image_files.sort()
    
    # Display files with indices
    print("\nAvailable image files:")
    for i, file in enumerate(image_files):
        print(f"  [{i}] {os.path.basename(file)}")
        
    # Get user selection
    while True:
        selection = input("\nEnter file number or full path (or 'browse [directory]' to change directory): ")
        
        # Handle browse command
        if selection.lower().startswith('browse'):
            parts = selection.split(None, 1)
            new_dir = parts[1] if len(parts) > 1 else input("Enter directory to browse: ")
            if os.path.isdir(new_dir):
                return browse_images(new_dir)
            else:
                print(f"Invalid directory: {new_dir}")
                continue
        
        # Handle direct file path
        if os.path.isfile(selection):
            return selection
            
        # Handle numeric selection
        try:
            index = int(selection)
            if 0 <= index < len(image_files):
                return image_files[index]
            else:
                print(f"Invalid selection. Please enter a number between 0 and {len(image_files)-1}")
        except ValueError:
            print("Invalid input. Please enter a file number, a valid file path, or 'browse [directory]'")


def run_clip_mode(safety_scorer):
    """Run the application in CLIP mode."""
    while True:
        try:
            # Step 1: Select the image first
            print("\n--- Step 1: Select an image ---")
            image_prompt = input("Enter image path (or 'browse' to select from files): ")
            if image_prompt.lower() in ['exit', 'quit', 'back', 'menu']:
                break
                
            # Handle image file browsing
            image_path = None
            if image_prompt.lower() == 'browse':
                image_path = browse_images()
                if not image_path:
                    continue
            else:
                image_path = image_prompt
            
            # Validate image path
            if not os.path.exists(image_path):
                print(f"Error: Image file does not exist: {image_path}")
                continue
                
            # Step 2: Get the text prompt
            print("\n--- Step 2: Enter text about the image ---")
            text_input = input("Enter text prompt: ")
            if text_input.lower() in ['exit', 'quit', 'back', 'menu']:
                break
            
            # Step 3: Score the inputs
            print("\n--- Step 3: Checking safety scores ---")
            try:
                # Score inputs
                text_scores = safety_scorer.score_text(text_input)
                image_scores = safety_scorer.score_image(image_path)
                
                # Apply safety constraint
                passes_constraint = safety_scorer.safety_constraint(text_scores, image_scores)
                
                # Print safety scores
                print("\n===== SAFETY SCORES =====")
                print("Text scores:")
                for category, score in text_scores.items():
                    print(f"  {category}: {score:.4f}")
                
                print("\nImage scores:")
                for category, score in image_scores.items():
                    print(f"  {category}: {score:.4f}")
                
                print(f"\nPasses safety constraint: {passes_constraint}")
                
                # Step 4: Process with CLIP if constraints are satisfied
                if passes_constraint:
                    print("\n--- Step 4: Processing with CLIP ---")
                    clip_output = safety_scorer.process_clip(text_input, image_path)
                    
                    print("\n===== CLIP RESULTS =====")
                    print(f"Text-image similarity: {clip_output['logits_per_image'][0][0].item():.4f}")
                    
                    # Note: CLIP doesn't support chat functionality, it's just a similarity model
                    # Print the top similarity score which represents how well the text matches the image
                    print("\nNote: CLIP is not a chatbot model; it measures text-image similarity")
                    print("The score above indicates how well your text matches the image content")
                else:
                    print("\nInput rejected by safety constraint. CLIP processing skipped.")
            except Exception as e:
                print(f"Error processing input: {e}")
                
        except KeyboardInterrupt:
            print("\nInterrupted. Type 'exit' or 'back' to go back to the main menu.")
        except Exception as e:
            print(f"Error: {e}")


def run_gpt4v_mode(safety_scorer):
    """Run the application in GPT-4 Vision chat mode."""
    try:
        chat_model = GPT4VisionChat()
        
        print("Starting chat session with GPT-4 Vision")
        print("Type 'clear' to reset the conversation, 'exit'/'back' to return to the main menu")
        
        current_image_path = None
        
        while True:
            try:
                # Step 1: Select the image first (if we don't have one yet)
                if not current_image_path:
                    print("\n--- Select an image for the conversation ---")
                    image_prompt = input("Enter image path (or 'browse' to select from files): ")
                    if image_prompt.lower() in ['exit', 'quit', 'back', 'menu']:
                        break
                        
                    # Handle image file browsing
                    if image_prompt.lower() == 'browse':
                        current_image_path = browse_images()
                        if not current_image_path:
                            continue
                    else:
                        current_image_path = image_prompt
                    
                    # Validate image path
                    if not os.path.exists(current_image_path):
                        print(f"Error: Image file does not exist: {current_image_path}")
                        current_image_path = None
                        continue
                    
                    print(f"Image selected: {current_image_path}")
                
                # Step 2: Get the chat message
                print("\n--- Enter your message ---")
                text_input = input("You: ")
                
                # Handle special commands
                if text_input.lower() in ['exit', 'quit', 'back', 'menu']:
                    break
                elif text_input.lower() == 'clear':
                    chat_model.clear_history()
                    continue
                elif text_input.lower() == 'change image':
                    current_image_path = None
                    continue
                
                # Step 3: Score the inputs
                print("\n--- Checking safety scores ---")
                try:
                    # Score inputs
                    text_scores = safety_scorer.score_text(text_input)
                    image_scores = safety_scorer.score_image(current_image_path)
                    
                    # Apply safety constraint
                    passes_constraint = safety_scorer.safety_constraint(text_scores, image_scores)
                    
                    # Print safety scores
                    print("\n===== SAFETY SCORES =====")
                    print("Text scores:")
                    for category, score in text_scores.items():
                        print(f"  {category}: {score:.4f}")
                    
                    print("\nImage scores:")
                    for category, score in image_scores.items():
                        print(f"  {category}: {score:.4f}")
                    
                    print(f"\nPasses safety constraint: {passes_constraint}")
                    
                    # Step 4: Process with GPT-4 Vision if constraints are satisfied
                    if passes_constraint:
                        print("\n--- Processing with GPT-4 Vision ---")
                        response = chat_model.chat(text_input, current_image_path)
                        
                        print("\n===== GPT-4 VISION RESPONSE =====")
                        print(f"GPT-4: {response}")
                    else:
                        print("\nInput rejected by safety constraint. Chat processing skipped.")
                except Exception as e:
                    print(f"Error processing input: {e}")
                    
            except KeyboardInterrupt:
                print("\nInterrupted. Type 'exit' or 'back' to go back to the main menu.")
            except Exception as e:
                print(f"Error: {e}")
    
    except ValueError as e:
        print(f"Error initializing GPT-4 Vision: {e}")
        print("Make sure you have set your OpenAI API key in the configuration.")


def main():
    # Initialize the safety scorer
    safety_scorer = SafetyScorer()
    
    print("=== CLIP Safety Alignment System ===")
    print("This system applies safety constraints before processing with a vision model")
    print("Warning: You must have valid Sightengine API credentials configured")
    
    while True:
        print("\n=== MAIN MENU ===")
        print("1. CLIP Mode (Text-Image Similarity Analysis)")
        print("2. GPT-4 Vision Chat Mode")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ")
        
        if choice == '1':
            run_clip_mode(safety_scorer)
        elif choice == '2':
            run_gpt4v_mode(safety_scorer)
        elif choice == '3' or choice.lower() in ['exit', 'quit']:
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 3.")
    
    print("Exiting the application. Goodbye!")


if __name__ == "__main__":
    main()