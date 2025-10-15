#!/usr/bin/env python3
"""
Gemma SAE Feature Quantile Estimation

This script processes the High-Quality English Sentences dataset through Gemma 2,
encodes activations with Gemma Scope SAEs, and estimates activation quantiles
for each feature using an adaptive algorithm.
"""

import os
import sys
import torch
import numpy as np
from typing import List, Dict, Tuple
import pickle
from tqdm import tqdm
import time
import argparse
import signal

# Set environment variable to avoid tokenizers warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Handle potential import issues
try:
    from transformers import AutoTokenizer
    from transformer_lens import HookedTransformer
    from sae_lens import SAE
    from datasets import load_dataset
    print("All required packages imported successfully!")
except ImportError as e:
    print(f"Import error: {e}")
    print("Please install required packages:")
    print("pip install transformers transformer_lens sae-lens datasets torch torchvision tqdm numpy")
    sys.exit(1)

# Configuration
DEVICE = torch.device('cpu')    # mps much slower for some reason
MODEL_NAME = "google/gemma-2-2b"
SAE_RELEASE = "gemma-scope-2b-pt-res-canonical"
NUM_LAYERS = 26
FEATURES_PER_SAE = 16384  # 16k features per SAE
DATASET_NAME = "agentlans/high-quality-english-sentences"
BATCH_SIZE = 8  # Increased batch size for better efficiency? Seems not
MAX_LENGTH = 64  # Reduced max length for faster processing

# Simplified counting-based rarity estimation
MIN_ACTIVATION = 1.0  # Minimum activation level
MULTIPLIER = 1.3       # Geometric series multiplier
NUM_LEVELS = 28
ACTIVATION_LEVELS = [MIN_ACTIVATION * (MULTIPLIER ** i) for i in range(NUM_LEVELS)]

MAX_SENTENCES = 1534699  # Total in high-quality-english-sentences.train is 1,534,699
OUTPUT_FILE = "gemma_sae_quantiles2.pkl"
SAVE_FREQUENCY = 2048  # Save every SAVE_FREQUENCY sentences
NUM_PRINTED_SAMPLES = 10  # Number of sample quantile estimates to print at the end

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Gemma SAE Feature Quantile Estimation")
    parser.add_argument(
        "--sentences", 
        type=int, 
        default=MAX_SENTENCES,
        help=f"Number of additional sentences to process (default: {MAX_SENTENCES})"
    )
    args = parser.parse_args()
    
    return args

# Global counting tensors: [num_levels, total_features] - count of activations above each level
activation_level_counts = None
# Track total number of activations per feature (for statistics)
feature_activation_counts = None
total_sentences_processed = 0     # Total number of sentences processed

# Timing statistics
timing_stats = {
    'tokenization': 0.0,
    'model_forward': 0.0,
    'sae_encoding': 0.0,
    'counting_update': 0.0,
    'total_batch': 0.0
}

# Global flag for graceful shutdown
graceful_shutdown = False

def signal_handler(signum, frame):
    """Handle Ctrl+Z gracefully by finishing current batch and saving."""
    global graceful_shutdown
    if not graceful_shutdown:
        print("\n\n🛑 Graceful shutdown requested! Will finish current batch and save results...")
        graceful_shutdown = True
    else:
        # Already shutting down gracefully
        print("\n\n🛑 Shutdown already in progress...")

# Register signal handler for Ctrl+Z
signal.signal(signal.SIGTSTP, signal_handler)

def initialize_counting_tensors():
    """Initialize global activation level counting tensors on device."""
    global activation_level_counts, feature_activation_counts
    
    num_levels = len(ACTIVATION_LEVELS)
    total_features = NUM_LAYERS * FEATURES_PER_SAE
    
    # Initialize counting tensors to zero: [num_levels, total_features]
    activation_level_counts = torch.zeros(num_levels, total_features, device=DEVICE, dtype=torch.int32)
    
    # Initialize feature activation counts to zero: [total_features]
    feature_activation_counts = torch.zeros(total_features, device=DEVICE, dtype=torch.int32)
    
    print(f"Initialized counting tensors on {DEVICE}")
    print(f"Activation levels: {ACTIVATION_LEVELS[0]:.2f} to {ACTIVATION_LEVELS[-1]:.2f} ({num_levels} levels, multiplier {MULTIPLIER})")
    print(f"Total features: {total_features:,}")
    print(f"Total levels: {num_levels}")


def load_existing_results():
    """Load existing results if available and return True if loaded, False otherwise."""
    global activation_level_counts, feature_activation_counts, total_sentences_processed
    
    if os.path.exists(OUTPUT_FILE):
        print(f"Found existing results in {OUTPUT_FILE}, loading...")
        try:
            with open(OUTPUT_FILE, 'rb') as f:
                results = pickle.load(f)
            
            # Validate that the results are compatible
            if (results['activation_levels'] != ACTIVATION_LEVELS or 
                results['num_layers'] != NUM_LAYERS or
                results['features_per_sae'] != FEATURES_PER_SAE):
                print("Warning: Existing results have incompatible configuration, starting fresh")
                return False
            
            # Load existing data
            total_sentences_processed = results['total_sentences_processed']
            
            # Load activation level counts
            if 'activation_level_counts' in results:
                # New format: tensor directly saved
                activation_level_counts = results['activation_level_counts'].to(device=DEVICE, dtype=torch.int32)
            else:
                # Backward compatibility: convert from nested dict format
                num_levels = len(ACTIVATION_LEVELS)
                total_features = NUM_LAYERS * FEATURES_PER_SAE
                activation_level_counts = torch.zeros(num_levels, total_features, device=DEVICE, dtype=torch.int32)
                
                for layer_idx in range(NUM_LAYERS):
                    for feat_idx in range(FEATURES_PER_SAE):
                        global_feat_idx = layer_idx * FEATURES_PER_SAE + feat_idx
                        counts = results['counts'][layer_idx][feat_idx]
                        for level_idx in range(len(ACTIVATION_LEVELS)):
                            activation_level_counts[level_idx, global_feat_idx] = counts[level_idx]
            
            # Initialize feature_activation_counts tensor
            total_features = NUM_LAYERS * FEATURES_PER_SAE
            feature_activation_counts = torch.zeros(total_features, device=DEVICE, dtype=torch.int32)
            
            # Load feature activation counts if available (for backward compatibility)
            if 'feature_activation_counts' in results:
                activation_counts_data = results['feature_activation_counts']
                if isinstance(activation_counts_data, list):
                    # Backward compatibility: convert from list
                    feature_activation_counts = torch.tensor(activation_counts_data, device=DEVICE, dtype=torch.int32)
                else:
                    # New format: already a tensor, just move to device
                    feature_activation_counts = activation_counts_data.to(device=DEVICE, dtype=torch.int32)
            else:
                # Calculate from level counts for backward compatibility
                feature_activation_counts = activation_level_counts[0].clone()  # Use lowest level as approximation
            
            print(f"Loaded existing results: {total_sentences_processed} sentences already processed")
            
            # Print current activation statistics
            activation_counts_cpu = feature_activation_counts.cpu().numpy()
            total_features = len(activation_counts_cpu)
            activated_features = (activation_counts_cpu > 0).sum()
            print(f"Current state: {activated_features}/{total_features} features activated ({100*activated_features/total_features:.1f}%)")
            
            return True
            
        except Exception as e:
            print(f"Error loading existing results: {e}")
            print("Starting fresh...")
            return False
    
    return False

def update_activation_counts(sentence_max_acts):
    """
    Update activation level counts for ALL features (including zero activations).
    This is fully vectorized and efficient.
    Also track non-zero activations separately for statistics.
    
    Args:
        sentence_max_acts: [num_sentences, total_features] tensor of max activations per sentence
    """
    global activation_level_counts, feature_activation_counts, total_sentences_processed
    
    if sentence_max_acts.shape[0] == 0:
        return
    
    # Update total sentences processed
    total_sentences_processed += sentence_max_acts.shape[0]
    
    # Track non-zero activations for statistics
    # Count how many sentences had non-zero activations for each feature
    non_zero_mask = sentence_max_acts > 0  # [num_sentences, total_features]
    non_zero_counts = non_zero_mask.sum(dim=0)  # [total_features]
    feature_activation_counts += non_zero_counts
    
    # For each activation level, count how many sentences exceed it (vectorized)
    for level_idx, threshold in enumerate(ACTIVATION_LEVELS):
        # Create mask for activations above threshold: [num_sentences, total_features]
        above_threshold = sentence_max_acts > threshold
        
        # Count sentences above threshold for each feature: [total_features]
        count_above = above_threshold.sum(dim=0)  # Sum across sentences
        
        # Add to cumulative counts (this includes ALL features, even those with zero activations)
        activation_level_counts[level_idx] += count_above

def load_model_and_saes():
    """Load Gemma 2 model and all SAEs with optimal configuration."""
    print("Loading Gemma 2 model...")
    torch.set_grad_enabled(False)

    # Load model with HookedTransformer for optimal SAE integration
    model = HookedTransformer.from_pretrained(
        MODEL_NAME,
        device=DEVICE,
        center_unembed=False,   # TO QUIET THE WARNING
        center_writing_weights=False  # TO QUIET THE WARNING
    )
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # print(f"Tokenizer with: {tokenizer.pad_token}, {tokenizer.bos_token}, {tokenizer.eos_token}")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Set padding side to right for analysis (content first, padding at end)
    tokenizer.padding_side = "right"
    
    sae_list = []
    for i in tqdm(range(NUM_LAYERS), desc="Loading SAEs"):
        sae = SAE.from_pretrained(
            release=SAE_RELEASE,
            sae_id=f"layer_{i}/width_16k/canonical",
            device=DEVICE
        )
        sae.eval()
        sae_list.append((i, sae))  # Store layer index with SAE
    
    return model, tokenizer, sae_list

def load_dataset_sentences():
    """Load the High-Quality English Sentences dataset."""
    print("Loading dataset...")
    try:
        dataset = load_dataset(DATASET_NAME, split="train", streaming=True)
        return dataset
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Failed to load the required dataset. Exiting.")
        sys.exit(1)

def get_residual_activations(model, inputs: torch.Tensor, target_layers: List[int]) -> Dict[int, torch.Tensor]:
    """Extract residual stream activations from specified layers using HookedTransformer."""
    residual_acts = {}
    handles = []
    
    def make_hook(idx):
        def hook(mod, inp, out):
            # Keep activations on the same device as the model
            residual_acts[idx] = out.detach()
        return hook
    
    # Register hooks for HookedTransformer blocks
    for i in target_layers:
        # HookedTransformer uses blocks instead of layers
        h = model.blocks[i].register_forward_hook(make_hook(i))
        handles.append(h)
    
    # Forward pass
    with torch.no_grad():
        _ = model.forward(inputs)
    
    # Remove hooks
    for h in handles:
        h.remove()
    
    return residual_acts

def process_batch(model, tokenizer, sae_list, sentences: List[str]):
    """Process a batch of sentences and update quantile estimates (fully vectorized)."""
    global timing_stats
    batch_start = time.time()
    
    # Tokenize and pad
    tokenize_start = time.time()
    tokenized = tokenizer(
        sentences, 
        padding=True, 
        truncation=True, 
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )
    inputs = tokenized['input_ids'].to(DEVICE)
    attention_mask = tokenized['attention_mask'].to(DEVICE)
    timing_stats['tokenization'] += time.time() - tokenize_start
    
    # Get residual activations for all layers
    model_start = time.time()
    target_layers = [layer_idx for layer_idx, _ in sae_list]
    residual_acts = get_residual_activations(model, inputs, target_layers)
    timing_stats['model_forward'] += time.time() - model_start
    
    # Create content mask to exclude special tokens
    bos_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else -1
    eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else -1
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -1
    
    content_mask = attention_mask.clone()
    # print(f"Content mask before special token exclusion: {content_mask}")
    if bos_token_id != -1:
        content_mask = content_mask & (inputs != bos_token_id)
    if eos_token_id != -1:
        content_mask = content_mask & (inputs != eos_token_id)
    if pad_token_id != -1:
        content_mask = content_mask & (inputs != pad_token_id)
    # print(f"Content mask after special token exclusion: {content_mask}")
    
    # Collect all SAE activations into a single tensor
    batch_size, seq_len = inputs.shape
    all_feature_acts = torch.zeros(batch_size, seq_len, NUM_LAYERS * FEATURES_PER_SAE, 
                                   device=DEVICE, dtype=torch.float32)
    
    # Process through all SAEs
    sae_start = time.time()
    for layer_idx, sae in sae_list:
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
    timing_stats['sae_encoding'] += time.time() - sae_start
    
    # For each sentence, compute maximum activation across valid tokens
    counting_start = time.time()
    sentence_max_acts = []
    
    for sent_idx in range(batch_size):
        # Get the content mask for this sentence
        # THE .bool() IS VERY IMPORTANT!!!!
        sent_content_mask = content_mask[sent_idx].bool()  # [seq_len] - convert to boolean
        sent_feature_acts = all_feature_acts[sent_idx]  # [seq_len, total_features]
        
        # Filter out NaN and Inf values for this sentence (check if any features are NaN/Inf for each token)
        valid_acts_mask = ~(torch.isnan(sent_feature_acts).any(dim=1) | 
                           torch.isinf(sent_feature_acts).any(dim=1))
        
        # Final mask: content tokens AND valid activations
        final_sent_mask = sent_content_mask & valid_acts_mask
        
        if final_sent_mask.any():
            # Get valid activations for this sentence - only use content tokens with valid activations
            valid_sent_acts = sent_feature_acts[final_sent_mask]  # [num_valid_tokens, total_features]
            
            # Take maximum across valid tokens for each feature
            sent_max = valid_sent_acts.max(dim=0)[0]  # [total_features]
            sentence_max_acts.append(sent_max)

    if sentence_max_acts:
        # Stack into tensor and update counts
        sentence_max_acts = torch.stack(sentence_max_acts)  # [num_valid_sentences, total_features]
        update_activation_counts(sentence_max_acts)
    
    timing_stats['counting_update'] += time.time() - counting_start
    timing_stats['total_batch'] += time.time() - batch_start

def main():
    """Main processing loop."""
    global graceful_shutdown
    
    # Parse command line arguments
    args = parse_args()
    additional_sentences = args.sentences
    
    print("Starting Gemma SAE quantile estimation (counting-based version)...")
    print("💡 TIP: Press Ctrl+Z at any time to gracefully finish the current batch and save results")
    print("=" * 80)
    
    # Check if we can resume from existing results
    existing_loaded = load_existing_results()
    
    if existing_loaded:
        target_sentences = total_sentences_processed + additional_sentences
        print(f"Will process {additional_sentences} additional sentences to reach total of {target_sentences}")
    else:
        target_sentences = additional_sentences
        print(f"Starting fresh - will process {additional_sentences} sentences")
    
    # Check against maximum limit
    if target_sentences > MAX_SENTENCES:
        print(f"Warning: Target {target_sentences} sentences exceeds maximum {MAX_SENTENCES}. Limiting to {MAX_SENTENCES}.")
        target_sentences = MAX_SENTENCES
        if existing_loaded:
            additional_sentences = target_sentences - total_sentences_processed
            print(f"Will process {additional_sentences} additional sentences instead.")
    
    if not existing_loaded:
        # Initialize fresh counting tensors
        initialize_counting_tensors()
    
    # Load model and SAEs
    model, tokenizer, sae_list = load_model_and_saes()
    
    print(f"Processing all {NUM_LAYERS} layers with {FEATURES_PER_SAE} features each")
    print("💡 REMINDER: Press Ctrl+Z at any time for graceful shutdown")
    print("=" * 60)
    
    # Load dataset
    dataset = load_dataset_sentences()
    
    # Skip to where we left off if resuming
    if existing_loaded and total_sentences_processed > 0:
        print(f"Skipping {total_sentences_processed} already processed sentences...")
        dataset = dataset.skip(total_sentences_processed)
    
    # Process dataset in batches
    batch_sentences = []
    processed_count = total_sentences_processed  # Start from where we left off
    session_processed = 0  # Count of sentences processed in this session only
    
    # Create progress bar for just the additional sentences to be processed
    progress_bar = tqdm(total=additional_sentences, desc="Processing sentences")
    
    try:
        for example in dataset:
            # Check for graceful shutdown request
            if graceful_shutdown:
                print(f"\n🛑 Graceful shutdown in progress... finishing current batch")
                break
                
            # Extract sentence based on dataset structure
            if 'text' in example:
                sentence = example['text']
            else:
                print("Warning: Unrecognized dataset format, skipping example")
                continue  # Skip invalid examples, don't count them
                
            batch_sentences.append(sentence)
            
            if len(batch_sentences) >= BATCH_SIZE:
                try:
                    process_batch(model, tokenizer, sae_list, batch_sentences)
                    processed_count += len(batch_sentences)
                    session_processed += len(batch_sentences)
                    
                    # Update progress bar with the number of sentences processed in this batch
                    progress_bar.update(len(batch_sentences))
                    
                    # Progress updates
                    if session_processed % SAVE_FREQUENCY == 0 and session_processed < additional_sentences:
                        print(f"\nProcessed {session_processed}/{additional_sentences} additional sentences (total: {processed_count})")
                        
                        # Print activation statistics
                        activation_counts_cpu = feature_activation_counts.cpu().numpy()
                        total_features = len(activation_counts_cpu)
                        activated_features = (activation_counts_cpu > 0).sum()
                        print(f"Features activated so far: {activated_features}/{total_features} ({100*activated_features/total_features:.1f}%)")
                        
                        save_results(OUTPUT_FILE, show_samples_and_stats=False)
                        
                except Exception as e:
                    print(f"Error processing batch: {e}")
                    
                batch_sentences = []
                
                # Check for graceful shutdown after processing batch
                if graceful_shutdown:
                    print(f"🛑 Batch completed, initiating graceful shutdown...")
                    break
                
            # Stop processing if we've reached the additional sentence limit for this session
            if session_processed >= additional_sentences:
                # print(f"Reached additional sentence limit of {additional_sentences}")
                break
    
    except KeyboardInterrupt:
        # This shouldn't be reached due to signal handler, but just in case
        print(f"\n🛑 Interrupted! Processing final batch...")
        graceful_shutdown = True
    
    # Process any remaining sentences in the final batch
    if batch_sentences and not graceful_shutdown:
        try:
            print(f"\nProcessing final batch of {len(batch_sentences)} sentences...")
            process_batch(model, tokenizer, sae_list, batch_sentences)
            processed_count += len(batch_sentences)
            session_processed += len(batch_sentences)
            progress_bar.update(len(batch_sentences))
        except Exception as e:
            print(f"Error processing final batch: {e}")
    elif batch_sentences and graceful_shutdown:
        print(f"\n🛑 Graceful shutdown: Skipping final incomplete batch of {len(batch_sentences)} sentences")
    
    # Close progress bar
    progress_bar.close()
    
    if graceful_shutdown:
        print(f"\n✅ Graceful shutdown completed after processing {session_processed} additional sentences (total: {processed_count})")
    else:
        print(f"Finished processing {session_processed} additional sentences (total: {processed_count})")
    
    # Print timing analysis
    print_timing_stats()
    
    # Save final results
    save_results(OUTPUT_FILE)

def print_timing_stats():
    """Print detailed timing statistics to identify bottlenecks."""
    global timing_stats
    
    total_time = timing_stats['total_batch']
    if total_time == 0:
        print("No timing data available")
        return
    
    print("\n" + "="*50)
    print("PERFORMANCE PROFILING RESULTS")
    print("="*50)
    print(f"Total batch processing time: {total_time:.2f}s")
    print("\nBreakdown by operation:")
    print(f"  Tokenization:    {timing_stats['tokenization']:.2f}s ({100*timing_stats['tokenization']/total_time:.1f}%)")
    print(f"  Model forward:   {timing_stats['model_forward']:.2f}s ({100*timing_stats['model_forward']/total_time:.1f}%)")
    print(f"  SAE encoding:    {timing_stats['sae_encoding']:.2f}s ({100*timing_stats['sae_encoding']/total_time:.1f}%)")
    print(f"  Counting update: {timing_stats['counting_update']:.2f}s ({100*timing_stats['counting_update']/total_time:.1f}%)")
    print("="*50)

def save_results(filename: str, show_samples_and_stats: bool = True):
    """Save activation level counts and rarity statistics to file."""
    global activation_level_counts, feature_activation_counts, total_sentences_processed
    
    print(f"Saving results to {filename}...")
    
    # Convert tensor to nested structure for compatibility
    results = {
        'activation_levels': ACTIVATION_LEVELS,
        'num_layers': NUM_LAYERS,
        'processed_layers': list(range(NUM_LAYERS)),
        'features_per_sae': FEATURES_PER_SAE,
        'total_sentences_processed': total_sentences_processed,
        'feature_activation_counts': feature_activation_counts.cpu(),
        'activation_level_counts': activation_level_counts.cpu()
    }
    
    with open(filename, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Results saved to {filename}")
    print(f"Total sentences processed: {total_sentences_processed}")
    
    # Only print sample results if requested
    if show_samples_and_stats:
        # Print activation statistics
        activation_counts_cpu = activation_level_counts.cpu().numpy()
        feature_counts_cpu = feature_activation_counts.cpu().numpy()
        total_features = activation_counts_cpu.shape[1]
        activated_features = (feature_counts_cpu > 0).sum()  # Features with any non-zero activation

        print(f"\nActivation Statistics:")
        print(f"  Total features: {total_features}")
        print(f"  Features with non-zero activations: {activated_features} ({100*activated_features/total_features:.1f}%)")
        
        # Print counts for beginning, middle, last levels
        levels_to_show = [0, len(ACTIVATION_LEVELS) // 2, len(ACTIVATION_LEVELS) - 1]
        for level_idx in levels_to_show:
            threshold = ACTIVATION_LEVELS[level_idx]
            count_above = (activation_counts_cpu[level_idx] > 0).sum()
            actual_rarity = count_above / total_features if total_features > 0 else 0
            print(f"  Features with activations > {threshold:.2f}: {count_above} ({100*actual_rarity:.3f}%)")

        # Print sample results
        print("\nSample rarity estimates:")
        
        # Find features that have been activated at least once (non-zero)
        activated_indices = np.where(feature_counts_cpu > 0)[0]
        
        if len(activated_indices) > 0:
            # Pick up to NUM_PRINTED_SAMPLES activated features, distributed across the range
            num_samples = min(NUM_PRINTED_SAMPLES, len(activated_indices))
            
            if len(activated_indices) < NUM_PRINTED_SAMPLES:
                sample_indices = activated_indices.tolist()
            else:
                step = len(activated_indices) // NUM_PRINTED_SAMPLES
                sample_indices = [activated_indices[i * step] for i in range(NUM_PRINTED_SAMPLES)]
            
            for global_feat_idx in sample_indices:
                layer_idx = global_feat_idx // FEATURES_PER_SAE
                feat_idx = global_feat_idx % FEATURES_PER_SAE
                non_zero_count = feature_counts_cpu[global_feat_idx]
                # Show levels closest to target rarities (0.1, 0.001, 0.0001)
                target_rarities = [0.01, 0.001, 0.0001]
                rarity_strs = []
                
                for target_rarity in target_rarities:
                    # Find the activation level closest to this target rarity for this feature
                    best_level_idx = 0
                    best_diff = float('inf')
                    
                    for level_idx in range(len(ACTIVATION_LEVELS)):
                        count = activation_counts_cpu[level_idx, global_feat_idx]
                        actual_rarity = count / total_sentences_processed if total_sentences_processed > 0 else 0
                        diff = abs(actual_rarity - target_rarity)
                        if diff < best_diff:
                            best_diff = diff
                            best_level_idx = level_idx
                    
                    threshold = ACTIVATION_LEVELS[best_level_idx]
                    count = activation_counts_cpu[best_level_idx, global_feat_idx]
                    rarity = count / total_sentences_processed if total_sentences_processed > 0 else 0
                    rarity_strs.append(f">{threshold:.2f}: {rarity:.6f}")
                
                print(f"L{layer_idx:02d}, F{feat_idx:05d} ({non_zero_count} non-zero): {', '.join(rarity_strs)}")
        else:
            print("No features have been activated yet.")

def load_results(filename: str = OUTPUT_FILE) -> Dict:
    """Load saved quantile results."""
    with open(filename, 'rb') as f:
        return pickle.load(f)

if __name__ == "__main__":
    main()
