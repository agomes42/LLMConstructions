#!/usr/bin/env python3
"""
SAE Feature Analysis Script

This script analyzes SAE feature activations from a text file, compares them to quantile estimates,
ranks features by rarity, and generates completions with clamped rare features.

Usage:
    python analyze_sae_features.py <filename> [--prompt "Custom prompt"] [--top-features N]

Example:
    python analyze_sae_features.py XerYer
    python analyze_sae_features.py XerYer --prompt "The cat is" --top-features 3

The script will:
1. Load sentences from the specified file (one per line)
2. Compute SAE feature activations and their 10% quantiles
3. Compare to existing quantile data from gemma_sae_quantiles.pkl
4. Rank features by rarity using log-scale interpolation
5. Generate completions with the rarest features clamped to 0.001 quantile levels
"""

import os
import sys
import torch
import numpy as np
import pickle
import argparse
from typing import List, Dict, Tuple
from tqdm import tqdm

# Set environment variable to avoid tokenizers warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Handle potential import issues
try:
    from transformers import AutoTokenizer
    from transformer_lens import HookedTransformer
    from sae_lens import SAE
    print("All required packages imported successfully!")
except ImportError as e:
    print(f"Import error: {e}")
    print("Please install required packages:")
    print("pip install transformers transformer_lens sae-lens torch numpy")
    sys.exit(1)

# Configuration (matching gemma_sae_quantiles.py)
DEVICE = torch.device('cpu')    # mps much slower for some reason
MODEL_NAME = "google/gemma-2-2b"
SAE_RELEASE = "gemma-scope-2b-pt-res-canonical"
NUM_LAYERS = 26
FEATURES_PER_SAE = 16384  # 16k features per SAE
MAX_LENGTH = 128  # Reduced max length for faster processing
QUANTILE_LEVELS = [0.1, 0.01, 0.001]  # Target rarity levels (r values)
QUANTILES_FILE = "gemma_sae_quantiles2.pkl"
DEFAULT_PROMPT = "After a long pause, the man said"  # DONT CHANGE
TOP_FEATURES_TO_CLAMP = 5
SENTENCE_QUANTILE_LEVEL = 0.05  # 5% quantile across sentences
MAX_TRAINING_FREQ = 0.1  # Ignore features activated in more than 10% of training sentences
CLAMP_MULTIPLIER = 2.0  # DONT CHANGE
MAX_NEW_TOKENS = 20  # Number of tokens to generate in completions

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="SAE Feature Analysis")
    parser.add_argument(
        "filename", 
        type=str, 
        help="Input file containing sentences (one per line)"
    )
    parser.add_argument(
        "--prompt", 
        type=str, 
        default=DEFAULT_PROMPT,
        help=f"Prompt for feature clamping completions (default: '{DEFAULT_PROMPT}')"
    )
    parser.add_argument(
        "--top-features", 
        type=int, 
        default=TOP_FEATURES_TO_CLAMP,
        help=f"Number of top features to clamp for completions (default: {TOP_FEATURES_TO_CLAMP})"
    )
    parser.add_argument(
        "--control-file",
        type=str,
        help="Optional control file to compute features to ignore (ranks by -log(rarity1) + log(rarity2))"
    )
    return parser.parse_args()

def load_quantile_data(filename: str = QUANTILES_FILE) -> Dict:
    """Load quantile estimates from pickle file."""
    print(f"Loading quantile data from {filename}...")
    try:
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        print(f"Loaded quantiles for {data['total_sentences_processed']:,} sentences")
        return data
    except FileNotFoundError:
        print(f"Error: Quantile file {filename} not found!")
        print("Please run gemma_sae_quantiles.py first to generate the quantile estimates.")
        sys.exit(1)

def load_model_and_saes():
    """Load Gemma 2 model and all SAEs."""
    print("Loading Gemma 2 model...")
    torch.set_grad_enabled(False)

    # Load model with HookedTransformer for optimal SAE integration
    model = HookedTransformer.from_pretrained(
        MODEL_NAME,
        device=DEVICE,
        center_unembed=False,
        center_writing_weights=False
    )
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    print("Loading all SAEs...")
    sae_list = []
    for i in tqdm(range(NUM_LAYERS), desc="Loading SAEs"):
        sae = SAE.from_pretrained(
            release=SAE_RELEASE,
            sae_id=f"layer_{i}/width_16k/canonical",
            device=DEVICE
        )
        sae.eval()
        sae_list.append((i, sae))
    
    return model, tokenizer, sae_list

def read_sentences(filename: str) -> List[str]:
    """Read sentences from file, one per line."""
    print(f"Reading sentences from {filename}...")
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            # Strip whitespace from both ends and filter out empty lines
            sentences = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(sentences)} sentences")
        if sentences:
            print(f"Example sentence: \"{sentences[0]}\"")
        return sentences
    except FileNotFoundError:
        print(f"Error: File {filename} not found!")
        sys.exit(1)

def get_residual_activations(model, inputs: torch.Tensor, target_layers: List[int]) -> Dict[int, torch.Tensor]:
    """Extract residual stream activations from specified layers using HookedTransformer."""
    residual_acts = {}
    handles = []
    
    def make_hook(idx):
        def hook(mod, inp, out):
            residual_acts[idx] = out.detach()
        return hook
    
    # Register hooks for HookedTransformer blocks
    for i in target_layers:
        h = model.blocks[i].register_forward_hook(make_hook(i))
        handles.append(h)
    
    # Forward pass
    with torch.no_grad():
        _ = model.forward(inputs)
    
    # Remove hooks
    for h in handles:
        h.remove()
    
    return residual_acts

def compute_sentence_feature_activations(model, tokenizer, sae_list, sentences: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute sentence-level feature activations: first max over valid tokens, then quantile across sentences.
    
    Returns:
        quantiles: [total_features] tensor of quantile levels across sentences
        max_activations: [total_features] tensor of maximum activations across all sentences
    """
    print("Computing SAE feature activations...")
    
    # Tokenize and pad
    tokenized = tokenizer(
        sentences, 
        padding=True, 
        truncation=True, 
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )
    inputs = tokenized['input_ids'].to(DEVICE)
    attention_mask = tokenized['attention_mask'].to(DEVICE)
    
    # Get residual activations for all layers
    target_layers = [layer_idx for layer_idx, _ in sae_list]
    residual_acts = get_residual_activations(model, inputs, target_layers)
    
    # Create content mask to exclude special tokens
    bos_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else -1
    eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else -1
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -1
    
    content_mask = attention_mask.clone()
    if bos_token_id != -1:
        content_mask = content_mask & (inputs != bos_token_id)
    if eos_token_id != -1:
        content_mask = content_mask & (inputs != eos_token_id)
    if pad_token_id != -1:
        content_mask = content_mask & (inputs != pad_token_id)
    
    # Collect all SAE activations
    batch_size, seq_len = inputs.shape
    all_feature_acts = torch.zeros(batch_size, seq_len, NUM_LAYERS * FEATURES_PER_SAE, 
                                   device=DEVICE, dtype=torch.float32)
    
    # Process through all SAEs
    for layer_idx, sae in tqdm(sae_list, desc="Processing SAEs"):
        residual_act = residual_acts.get(layer_idx)
        if residual_act is None:
            continue
        
        # Encode with SAE
        with torch.no_grad():
            feature_acts = sae.encode(residual_act)  # [batch, seq, features]

        # Store in the combined tensor
        start_idx = layer_idx * FEATURES_PER_SAE
        end_idx = start_idx + FEATURES_PER_SAE
        all_feature_acts[:, :, start_idx:end_idx] = feature_acts
    
    # Step 1: For each sentence, compute MAX over valid tokens
    sentence_max_acts = []
    
    for sent_idx in range(batch_size):
        # Get the content mask for this sentence
        sent_content_mask = content_mask[sent_idx].bool()
        sent_feature_acts = all_feature_acts[sent_idx]  # [seq_len, total_features]
        
        # Filter out NaN and Inf values
        valid_acts_mask = ~(torch.isnan(sent_feature_acts).any(dim=1) | 
                           torch.isinf(sent_feature_acts).any(dim=1))
        
        # Final mask: content tokens AND valid activations
        final_sent_mask = sent_content_mask & valid_acts_mask
        
        if final_sent_mask.any():
            # Get valid activations for this sentence
            valid_sent_acts = sent_feature_acts[final_sent_mask]  # [num_valid_tokens, total_features]
            
            # Take MAX across valid tokens for each feature
            sent_max = torch.max(valid_sent_acts, dim=0)[0]  # [total_features]
            sentence_max_acts.append(sent_max)
    
    if sentence_max_acts:
        # Step 2: Stack all sentence max activations and compute quantile across sentences
        sentence_max_acts = torch.stack(sentence_max_acts)  # [num_sentences, total_features]
        
        # Compute quantile across sentences for each feature
        final_quantiles = torch.quantile(sentence_max_acts, SENTENCE_QUANTILE_LEVEL, dim=0)  # [total_features]
        
        # Also compute maximum across all sentences for each feature
        max_activations = torch.max(sentence_max_acts, dim=0)[0]  # [total_features]
        
        return final_quantiles, max_activations
    else:
        zeros = torch.zeros(NUM_LAYERS * FEATURES_PER_SAE, device=DEVICE, dtype=torch.float32)
        return zeros, zeros

def compute_max_activations_only(model, tokenizer, sae_list, sentences: List[str]) -> torch.Tensor:
    """Compute only max activations across all tokens and sentences for control file.
    
    Returns:
        max_activations: [total_features] tensor of maximum activations across all sentences and tokens
    """
    print("Computing max activations for control file...")
    
    # Tokenize and pad
    tokenized = tokenizer(
        sentences, 
        padding=True, 
        truncation=True, 
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )
    inputs = tokenized['input_ids'].to(DEVICE)
    attention_mask = tokenized['attention_mask'].to(DEVICE)
    
    # Get residual activations for all layers
    target_layers = [layer_idx for layer_idx, _ in sae_list]
    residual_acts = get_residual_activations(model, inputs, target_layers)
    
    # Create content mask to exclude special tokens
    bos_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else -1
    eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else -1
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -1
    
    content_mask = attention_mask.clone()
    if bos_token_id != -1:
        content_mask = content_mask & (inputs != bos_token_id)
    if eos_token_id != -1:
        content_mask = content_mask & (inputs != eos_token_id)
    if pad_token_id != -1:
        content_mask = content_mask & (inputs != pad_token_id)
    
    # Collect all SAE activations
    batch_size, seq_len = inputs.shape
    all_feature_acts = torch.zeros(batch_size, seq_len, NUM_LAYERS * FEATURES_PER_SAE, 
                                   device=DEVICE, dtype=torch.float32)
    
    # Process through all SAEs
    for layer_idx, sae in tqdm(sae_list, desc="Processing SAEs for control"):
        residual_act = residual_acts.get(layer_idx)
        if residual_act is None:
            continue
        
        # Encode with SAE
        with torch.no_grad():
            feature_acts = sae.encode(residual_act)  # [batch, seq, features]

        # Store in the combined tensor
        start_idx = layer_idx * FEATURES_PER_SAE
        end_idx = start_idx + FEATURES_PER_SAE
        all_feature_acts[:, :, start_idx:end_idx] = feature_acts
    
    # Apply content mask and compute global max
    all_max_acts = torch.zeros(NUM_LAYERS * FEATURES_PER_SAE, device=DEVICE, dtype=torch.float32)
    
    for sent_idx in range(batch_size):
        sent_content_mask = content_mask[sent_idx].bool()
        sent_feature_acts = all_feature_acts[sent_idx]  # [seq_len, total_features]
        
        # Filter out NaN and Inf values
        valid_acts_mask = ~(torch.isnan(sent_feature_acts).any(dim=1) | 
                           torch.isinf(sent_feature_acts).any(dim=1))
        
        # Final mask: content tokens AND valid activations
        final_sent_mask = sent_content_mask & valid_acts_mask
        
        if final_sent_mask.any():
            # Get valid activations for this sentence
            valid_sent_acts = sent_feature_acts[final_sent_mask]  # [num_valid_tokens, total_features]
            
            # Update global max
            sent_max = torch.max(valid_sent_acts, dim=0)[0]  # [total_features]
            all_max_acts = torch.maximum(all_max_acts, sent_max)
    
    return all_max_acts

def calculate_rarity_for_activation(activation_level_counts: torch.Tensor, activation_levels: List[float], 
                                   feature_idx: int, activation_value: float, total_sentences: int) -> float:
    """
    Calculate the rarity (frequency) of a specific activation value for a feature.
    Uses log-log interpolation between counting levels.
    """
    if activation_value <= 0:
        return 1.0  # Zero activation is not rare
    
    # Find the appropriate activation level bracket
    if activation_value <= activation_levels[0]:
        # Below minimum threshold - use first level
        count = activation_level_counts[0, feature_idx].item()
        return count / total_sentences if total_sentences > 0 else 0.0
    
    if activation_value >= activation_levels[-1]:
        # Above maximum threshold - use last level
        count = activation_level_counts[-1, feature_idx].item()
        return count / total_sentences if total_sentences > 0 else 0.0
    
    # Find the bracket and interpolate on log-log scale
    for i in range(len(activation_levels) - 1):
        if activation_levels[i] <= activation_value < activation_levels[i + 1]:
            # Get counts for interpolation
            lower_count = activation_level_counts[i, feature_idx].item()
            upper_count = activation_level_counts[i + 1, feature_idx].item()
            
            # Convert to rarities
            lower_rarity = lower_count / total_sentences if total_sentences > 0 else 0.0
            upper_rarity = upper_count / total_sentences if total_sentences > 0 else 0.0
            
            # Avoid log(0) by using small epsilon
            eps = 1e-10
            lower_rarity = max(lower_rarity, eps)
            upper_rarity = max(upper_rarity, eps)
            
            # Log-log interpolation
            log_act_val = np.log(activation_value)
            log_lower_act = np.log(activation_levels[i])
            log_upper_act = np.log(activation_levels[i + 1])
            
            log_lower_rarity = np.log(lower_rarity)
            log_upper_rarity = np.log(upper_rarity)
            
            # Interpolation factor in log space
            t = (log_act_val - log_lower_act) / (log_upper_act - log_lower_act)
            
            # Interpolate in log space and convert back
            log_interpolated_rarity = log_lower_rarity * (1 - t) + log_upper_rarity * t
            interpolated_rarity = np.exp(log_interpolated_rarity)
            
            return interpolated_rarity
    
    # Fallback (shouldn't reach here)
    return 0.0

def analyze_features(sentence_quantiles: torch.Tensor, max_activations: torch.Tensor, quantile_data: Dict, control_max_activations: torch.Tensor = None) -> Tuple[List[Tuple], List[Tuple]]:
    """
    Analyze features and rank by rarity.
    
    Args:
        sentence_quantiles: [total_features] tensor of quantile levels
        max_activations: [total_features] tensor of maximum activations
        quantile_data: Dictionary with training quantile data
    
    Returns:
        ranked_features: List of (global_feat_idx, layer_idx, feat_idx, activation, rarity)
        novel_features: List of (global_feat_idx, layer_idx, feat_idx, activation, max_activation)
    """
    print("Analyzing feature rarities...")
    
    # Extract tensor data from gemma_sae_quantiles2.py format
    activation_level_counts = quantile_data['activation_level_counts']  # [num_levels, num_features]
    feature_activation_counts = quantile_data['feature_activation_counts']  # [num_features]
    activation_levels = quantile_data['activation_levels']  # List of activation levels
    
    # Analyze each feature
    feature_analysis = []
    novel_features = []
    
    for global_feat_idx in range(len(sentence_quantiles)):
        activation = sentence_quantiles[global_feat_idx].item()
        
        if activation > 0:  # Only consider activated features
            layer_idx = global_feat_idx // FEATURES_PER_SAE
            feat_idx = global_feat_idx % FEATURES_PER_SAE
            
            # Check if this feature was ever activated in training
            training_count = feature_activation_counts[global_feat_idx].item()
            
            if training_count == 0:
                # Novel feature - never seen in training
                max_activation = max_activations[global_feat_idx].item()
                novel_features.append((global_feat_idx, layer_idx, feat_idx, activation, max_activation))
            else:
                # Check if feature is too common in training data
                training_freq = training_count / quantile_data['total_sentences_processed']
                if training_freq > MAX_TRAINING_FREQ:
                    continue  # Skip features that are too common
                
                # Calculate rarity using counting-based approach
                rarity = calculate_rarity_for_activation(
                    activation_level_counts, activation_levels, 
                    global_feat_idx, activation, quantile_data['total_sentences_processed']
                )
                
                feature_analysis.append((global_feat_idx, layer_idx, feat_idx, activation, rarity))
    
    # Sort by rarity or comparative ranking
    if control_max_activations is not None:
        print("Computing comparative ranking: -log(rarity1) + log(rarity2)")
        
        # Calculate control rarities for comparison
        comparative_features = []
        for global_feat_idx, layer_idx, feat_idx, activation, rarity in feature_analysis:
            control_activation = control_max_activations[global_feat_idx].item()
            
            if control_activation > 0:
                # Calculate rarity for the control activation
                control_rarity = calculate_rarity_for_activation(
                    activation_level_counts, activation_levels,
                    global_feat_idx, control_activation, quantile_data['total_sentences_processed']
                )
                
                # Compute comparative score: -log(rarity1) + log(rarity2)
                import math
                log_rarity1 = math.log(max(rarity, 1e-10))  # Avoid log(0)
                log_rarity2 = math.log(max(control_rarity, 1e-10))
                comparative_score = -log_rarity1 + log_rarity2  # WARNING!!!
                
                comparative_features.append((global_feat_idx, layer_idx, feat_idx, activation, rarity, control_rarity, comparative_score))
        
        # Sort by comparative score (descending - highest scores first)
        comparative_features.sort(key=lambda x: x[6], reverse=True)
        
        # Convert back to original format for compatibility
        ranked_features = [(f[0], f[1], f[2], f[3], f[4]) for f in comparative_features]
        
        print(f"Computed comparative scores for {len(comparative_features)} features")
    else:
        # Sort by rarity (ascending - rarest first, so smallest rarity values first)
        ranked_features = sorted(feature_analysis, key=lambda x: x[4])
    
    print(f"Found {len(ranked_features)} activated features")
    print(f"Found {len(novel_features)} novel features (never seen in training)")
    print(f"Using {SENTENCE_QUANTILE_LEVEL*100}% quantile across sentences")
    
    return ranked_features, novel_features

def clamp_feature_and_generate(model, tokenizer, sae_list, prompt: str, 
                              layer_idx: int, feat_idx: int, clamp_value: float) -> str:
    """Generate text with a specific SAE feature clamped to a value."""
    
    def clamp_hook(module, input, output):
        # Get the SAE for this layer
        sae = None
        for l_idx, s in sae_list:
            if l_idx == layer_idx:
                sae = s
                break
        
        if sae is not None:
            with torch.no_grad():
                # Encode activations
                feature_acts = sae.encode(output)  # [batch, seq, features]
                
                # Clamp the specific feature
                feature_acts[:, :, feat_idx] = clamp_value
                
                # Decode back to residual space
                clamped_residual = sae.decode(feature_acts)
                
                return clamped_residual
        
        return output
    
    # Register hook
    hook_handle = model.blocks[layer_idx].register_forward_hook(clamp_hook)
    
    try:
        # Tokenize prompt
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        input_ids = inputs['input_ids']
        
        # Generate with clamped feature
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True
            )
        
        # Decode result
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text
        
    finally:
        # Remove hook
        hook_handle.remove()

def main():
    """Main analysis pipeline."""
    args = parse_args()
    
    print("=" * 60)
    print("SAE Feature Analysis Pipeline")
    print("=" * 60)
    
    # Load data
    quantile_data = load_quantile_data()
    sentences = read_sentences(args.filename)
    model, tokenizer, sae_list = load_model_and_saes()
    
    # Compute feature activations
    sentence_quantiles, max_activations = compute_sentence_feature_activations(model, tokenizer, sae_list, sentences)
    
    # Handle control file if provided
    control_max_activations = None
    if args.control_file:
        print(f"\nProcessing control file: {args.control_file}")
        
        # Check if control file exists
        import os
        if not os.path.exists(args.control_file):
            print(f"ERROR: Control file {args.control_file} not found!")
            return
        
        # Load control file sentences
        control_sentences = read_sentences(args.control_file)
        print(f"Control file has {len(control_sentences)} sentences")
        
        # Compute max activations for control file
        control_max_activations = compute_max_activations_only(model, tokenizer, sae_list, control_sentences)
        print(f"Computed control max activations shape: {control_max_activations.shape}")
    
    # Analyze features
    ranked_features, novel_features = analyze_features(sentence_quantiles, max_activations, quantile_data, control_max_activations)
    
    # Print results
    print("\n" + "=" * 40)
    print("ANALYSIS RESULTS")
    print("=" * 40)
    
    if novel_features:
        print(f"\nNovel Features (never seen in training):")
        for global_feat_idx, layer_idx, feat_idx, activation, max_activation in novel_features:
            print(f"  L{layer_idx:02d}, F{feat_idx:05d}: activation = {activation:.6f}, max = {max_activation:.6f}")
    
    if ranked_features:
        print(f"\nTop 10 Rarest Activated Features:")
        for i, (global_feat_idx, layer_idx, feat_idx, activation, rarity) in enumerate(ranked_features[:10]):
            max_activation = max_activations[global_feat_idx].item()
            # Calculate training activation frequency from quantile_data
            training_count = quantile_data['feature_activation_counts'][global_feat_idx].item()
            training_freq_pct = (training_count / quantile_data['total_sentences_processed']) * 100
            
            # Use scientific notation for very small rarity values
            if rarity < 1e-6:
                rarity_str = f"{rarity:.2e}"
            else:
                rarity_str = f"{rarity:.6f}"
            print(f"  {i+1}. L {layer_idx:02d}, F {feat_idx:05d}: {activation:.3f} (max {max_activation:.3f}), rarity = {rarity_str}, train_freq = {training_freq_pct:.3f}%")
        
        # Generate completions with top known features clamped
        print(f"\n" + "=" * 40)
        print(f"COMPLETIONS WITH CLAMPED RARE FEATURES")
        print("=" * 40)
        print(f"Prompt: '{args.prompt}'")
        print(f"Clamping level: {CLAMP_MULTIPLIER*100}% of max activation")
        
        # Clamp rare features to multiple of max activation (like novel features)
        for i, (global_feat_idx, layer_idx, feat_idx, activation, rarity) in enumerate(ranked_features[:args.top_features]):
            # Get the max activation for this feature
            max_activation = max_activations[global_feat_idx].item()
            clamp_value = max_activation * CLAMP_MULTIPLIER
            
            # Format rarity for display
            if rarity < 1e-6:
                rarity_str = f"{rarity:.2e}"
            else:
                rarity_str = f"{rarity:.6f}"
            
            print(f"\nRare Feature {i+1}: L{layer_idx:02d}, F{feat_idx:05d} (rarity = {rarity_str})")
            print(f"Max activation: {max_activation:.6f}, Clamping to: {clamp_value:.6f}")
            
            try:
                completion = clamp_feature_and_generate(
                    model, tokenizer, sae_list, args.prompt, 
                    layer_idx, feat_idx, clamp_value
                )
                print(f"Completion: {completion}")
            except Exception as e:
                print(f"Error generating completion: {e}")
    
    # Generate completions for ALL novel features
    if novel_features:
        print(f"\n" + "=" * 40)
        print(f"COMPLETIONS WITH CLAMPED NOVEL FEATURES")
        print("=" * 40)
        print(f"Prompt: '{args.prompt}'")
        print(f"Clamping level: {CLAMP_MULTIPLIER*100}% of max activation")
        
        for i, (global_feat_idx, layer_idx, feat_idx, activation, max_activation) in enumerate(novel_features):
            # Clamp at a fraction of the maximum activation level
            clamp_value = max_activation * CLAMP_MULTIPLIER
            
            print(f"\nNovel Feature {i+1}: L{layer_idx:02d}, F{feat_idx:05d}")
            print(f"Max activation: {max_activation:.6f}, Clamping to: {clamp_value:.6f}")
            
            try:
                completion = clamp_feature_and_generate(
                    model, tokenizer, sae_list, args.prompt, 
                    layer_idx, feat_idx, clamp_value
                )
                print(f"Completion: {completion}")
            except Exception as e:
                print(f"Error generating completion: {e}")
    
    print(f"\n" + "=" * 60)
    print("Analysis complete!")

if __name__ == "__main__":
    main()
