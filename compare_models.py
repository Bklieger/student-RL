"""
Script to compare outputs from original and fine-tuned models.
"""
import os
import json
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

def load_model_and_tokenizer(model_path: str, device: str):
    """Load model and tokenizer from a saved path."""
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=None,
    ).to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    
    return model, tokenizer

def generate_response(model, tokenizer, prompt: str, device: str, max_length: int = 786, n_samples: int = 1):
    """Generate n responses from the model."""
    # Format prompt
    formatted_prompt = [
        {'role': 'system', 'content': 'You are a helpful AI assistant.'},
        {'role': 'user', 'content': prompt}
    ]
    prompt_text = tokenizer.apply_chat_template(formatted_prompt, tokenize=False)
    
    # Tokenize
    inputs = tokenizer(prompt_text, return_tensors="pt", padding=True).to(device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            do_sample=True,
            temperature=0.9,
            pad_token_id=tokenizer.pad_token_id,
            num_return_sequences=n_samples
        )
    
    # Decode
    responses = []
    for i in range(n_samples):
        response = tokenizer.decode(outputs[i][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        responses.append(response)
    return responses

def parse_args():
    parser = argparse.ArgumentParser(description="Compare original and fine-tuned model outputs")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory containing saved models")
    parser.add_argument("--prompt", type=str, help="Single prompt to test")
    parser.add_argument("--prompt_file", type=str, help="File containing prompts to test (one per line)")
    parser.add_argument("--max_length", type=int, default=786, help="Maximum length of generated response")
    parser.add_argument("--output_file", type=str, help="File to save comparison results")
    parser.add_argument("--n", type=int, default=1, help="Number of samples to generate per prompt")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load models
    print("Loading models...")
    original_model_path = os.path.join(args.output_dir, "models", "original_model")
    final_model_path = os.path.join(args.output_dir, "models", "final_model")
    
    original_model, original_tokenizer = load_model_and_tokenizer(original_model_path, device)
    final_model, final_tokenizer = load_model_and_tokenizer(final_model_path, device)
    print("Models loaded successfully!")
    
    # Get prompts
    prompts = []
    if args.prompt:
        prompts = [args.prompt]
    elif args.prompt_file:
        with open(args.prompt_file, 'r') as f:
            prompts = [line.strip() for line in f if line.strip()]
    else:
        print("Please provide either --prompt or --prompt_file")
        return
    
    # Generate and compare responses
    results = []
    for prompt in tqdm(prompts, desc="Generating responses"):
        
        # Get responses
        original_responses = generate_response(original_model, original_tokenizer, prompt, device, args.max_length, args.n)
        final_responses = generate_response(final_model, final_tokenizer, prompt, device, args.max_length, args.n)
        
        # Store results
        results.append({
            "prompt": prompt,
            "original_responses": original_responses,
            "fine_tuned_responses": final_responses
        })
    
    # Save results if output file specified
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output_file}")

if __name__ == "__main__":
    main() 

# python compare_models.py --output_dir final1 --prompt "Reason about something with the proper format with <reasoning> and <answer>" --output_file results.json --n 10