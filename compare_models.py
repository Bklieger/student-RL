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
        attn_implementation="flash_attention_2",
        device_map=None,
    ).to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    
    return model, tokenizer

def generate_response(model, tokenizer, prompt: str, device: str, max_length: int = 786):
    """Generate a response from the model."""
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
            pad_token_id=tokenizer.pad_token_id
        )
    
    # Decode
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response

def parse_args():
    parser = argparse.ArgumentParser(description="Compare original and fine-tuned model outputs")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory containing saved models")
    parser.add_argument("--prompt", type=str, help="Single prompt to test")
    parser.add_argument("--prompt_file", type=str, help="File containing prompts to test (one per line)")
    parser.add_argument("--max_length", type=int, default=786, help="Maximum length of generated response")
    parser.add_argument("--output_file", type=str, help="File to save comparison results")
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
        print("\n" + "="*50)
        print(f"Prompt: {prompt}")
        
        # Get responses
        original_response = generate_response(original_model, original_tokenizer, prompt, device, args.max_length)
        final_response = generate_response(final_model, final_tokenizer, prompt, device, args.max_length)
        
        # Print results
        print("\nOriginal Model Response:")
        print("-"*30)
        print(original_response)
        print("\nFine-tuned Model Response:")
        print("-"*30)
        print(final_response)
        
        # Store results
        results.append({
            "prompt": prompt,
            "original_response": original_response,
            "fine_tuned_response": final_response
        })
    
    # Save results if output file specified
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output_file}")

if __name__ == "__main__":
    main() 