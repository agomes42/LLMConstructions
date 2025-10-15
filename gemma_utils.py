#!/usr/bin/env python3
"""
Gemma Utilities for Notebook Experiments

This module provides utility functions for working with Gemma 2 model and SAEs
in notebook experiments, including model loading, embedding computation, and
similarity analysis.
"""

import os
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt

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
    print("pip install transformers transformer_lens sae-lens torch numpy matplotlib")
    raise

# Configuration
DEVICE = torch.device('cpu')  # mps much slower for some reason
MODEL_NAME = "google/gemma-2-2b"
SAE_RELEASE = "gemma-scope-2b-pt-res-canonical"
NUM_LAYERS = 26
FEATURES_PER_SAE = 16384  # 16k features per SAE
MAX_LENGTH = 128

class GemmaExperiment:
    """Class to hold Gemma model, tokenizer, and SAEs for experiments with lazy loading."""
    
    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._sae_list = None
        self._model_loaded = False
        self._saes_loaded = False
    
    @property
    def model(self):
        """Lazy load the model when first accessed."""
        if not self._model_loaded:
            self._load_model()
        return self._model
    
    @property
    def tokenizer(self):
        """Lazy load the tokenizer when first accessed."""
        if not self._model_loaded:
            self._load_model()
        return self._tokenizer
    
    @property
    def sae_list(self):
        """Lazy load the SAEs when first accessed."""
        if not self._saes_loaded:
            self._load_saes()
        return self._sae_list
    
    def _load_model(self, verbose: bool = True):
        """Internal method to load Gemma 2 model and tokenizer."""
        if self._model_loaded:
            return
        
        if verbose:
            print("Loading Gemma 2 model and tokenizer...")
        
        # Disable gradients
        torch.set_grad_enabled(False)
        
        # Load model
        self._model = HookedTransformer.from_pretrained(
            MODEL_NAME,
            device=DEVICE,
            center_unembed=False,
            center_writing_weights=False
        )
        self._model.eval()
        
        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.padding_side = "right"
        
        self._model_loaded = True
        if verbose:
            print("✅ Model and tokenizer loaded successfully!")
    
    def _load_saes(self, verbose: bool = True):
        """Internal method to load all SAEs."""
        if self._saes_loaded:
            return
        
        if verbose:
            print("Loading all SAEs...")
        self._sae_list = []
        iterator = tqdm(range(NUM_LAYERS), desc="Loading SAEs") if verbose else range(NUM_LAYERS)
        for i in iterator:
            sae = SAE.from_pretrained(
                release=SAE_RELEASE,
                sae_id=f"layer_{i}/width_16k/canonical",
                device=DEVICE
            )
            sae.eval()
            self._sae_list.append((i, sae))
        
        self._saes_loaded = True
        if verbose:
            print("✅ All SAEs loaded successfully!")

def create_gemma_experiment() -> GemmaExperiment:
    """
    Create a GemmaExperiment object with lazy loading.
    Model and SAEs will be loaded automatically when first accessed.
    
    Returns:
        GemmaExperiment object with lazy loading
    """
    return GemmaExperiment()

def get_final_token_embeddings(experiment: GemmaExperiment, 
                              sentences: List[str],
                              layers: Optional[List[int]] = None) -> Dict[int, torch.Tensor]:
    """
    Get embedding vectors for the final content token of each sentence from specified layers.
    
    Args:
        experiment: GemmaExperiment object with loaded models
        sentences: List of sentences to analyze
        layers: List of layer indices to extract embeddings from (default: all layers)
    
    Returns:
        Dictionary mapping layer_idx -> tensor of shape [num_sentences, hidden_dim]
    """
    # Note: With lazy loading, model loads automatically when first accessed
    
    if layers is None:
        layers = list(range(NUM_LAYERS))
    
    # Tokenize sentences (this will trigger model loading if needed)
    tokenized = experiment.tokenizer(
        sentences,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )
    inputs = tokenized['input_ids'].to(DEVICE)
    attention_mask = tokenized['attention_mask'].to(DEVICE)
    
    # Create content mask to exclude special tokens
    bos_token_id = experiment.tokenizer.bos_token_id if experiment.tokenizer.bos_token_id is not None else -1
    eos_token_id = experiment.tokenizer.eos_token_id if experiment.tokenizer.eos_token_id is not None else -1
    pad_token_id = experiment.tokenizer.pad_token_id if experiment.tokenizer.pad_token_id is not None else -1
    
    content_mask = attention_mask.clone()
    if bos_token_id != -1:
        content_mask = content_mask & (inputs != bos_token_id)
    if eos_token_id != -1:
        content_mask = content_mask & (inputs != eos_token_id)
    if pad_token_id != -1:
        content_mask = content_mask & (inputs != pad_token_id)
    
    # Get residual activations for specified layers
    residual_acts = {}
    handles = []
    
    def make_hook(idx):
        def hook(mod, inp, out):
            residual_acts[idx] = out.detach()
        return hook
    
    # Register hooks for specified layers
    for layer_idx in layers:
        h = experiment.model.blocks[layer_idx].register_forward_hook(make_hook(layer_idx))
        handles.append(h)
    
    # Forward pass
    with torch.no_grad():
        _ = experiment.model.forward(inputs)
    
    # Remove hooks
    for h in handles:
        h.remove()
    
    # Extract final content token embeddings for each sentence
    final_embeddings = {}
    
    for layer_idx in layers:
        if layer_idx not in residual_acts:
            continue
            
        layer_embeddings = []
        layer_acts = residual_acts[layer_idx]  # [batch_size, seq_len, hidden_dim]
        
        for sent_idx in range(layer_acts.shape[0]):
            # Find the last content token for this sentence
            sent_content_mask = content_mask[sent_idx].bool()
            if sent_content_mask.any():
                # Get indices of content tokens
                content_indices = torch.where(sent_content_mask)[0]
                last_content_idx = content_indices[-1].item()
                
                # Extract embedding for the last content token
                final_embedding = layer_acts[sent_idx, last_content_idx]  # [hidden_dim]
                layer_embeddings.append(final_embedding)
            else:
                # If no content tokens (shouldn't happen), use zeros
                layer_embeddings.append(torch.zeros_like(layer_acts[sent_idx, 0]))
        
        final_embeddings[layer_idx] = torch.stack(layer_embeddings)  # [num_sentences, hidden_dim]
    
    return final_embeddings

def get_final_token_sae_features(experiment: GemmaExperiment,
                                sentences: List[str],
                                layers: Optional[List[int]] = None) -> Dict[int, torch.Tensor]:
    """
    Get SAE feature vectors for the final content token of each sentence from specified layers.
    
    Args:
        experiment: GemmaExperiment object with loaded models
        sentences: List of sentences to analyze
        layers: List of layer indices to extract features from (default: all layers)
    
    Returns:
        Dictionary mapping layer_idx -> tensor of shape [num_sentences, num_features]
    """
    # Note: With lazy loading, SAEs load automatically when first accessed
    
    if layers is None:
        layers = list(range(NUM_LAYERS))
    
    # First get the residual embeddings
    residual_embeddings = get_final_token_embeddings(experiment, sentences, layers)
    
    # Encode with SAEs
    sae_features = {}
    
    for layer_idx in layers:
        if layer_idx not in residual_embeddings:
            continue
        
        # Find the corresponding SAE
        sae = None
        for l_idx, s in experiment.sae_list:
            if l_idx == layer_idx:
                sae = s
                break
        
        if sae is not None:
            residual_emb = residual_embeddings[layer_idx]  # [num_sentences, hidden_dim]
            
            with torch.no_grad():
                # Encode with SAE
                features = sae.encode(residual_emb)  # [num_sentences, num_features]
                sae_features[layer_idx] = features
    
    return sae_features

def compute_pairwise_cosine_similarity(embeddings: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise cosine similarity between embeddings using PyTorch.
    
    Args:
        embeddings: Tensor of shape [num_items, embedding_dim]
    
    Returns:
        Similarity matrix of shape [num_items, num_items]
    """
    # Normalize embeddings to unit length
    embeddings_normalized = F.normalize(embeddings, p=2, dim=1)
    
    # Compute cosine similarity via matrix multiplication
    similarity_matrix = torch.mm(embeddings_normalized, embeddings_normalized.t())
    
    return similarity_matrix

def get_final_tokens(experiment: GemmaExperiment, sentences: List[str]) -> List[str]:
    """
    Get the detokenized final content token for each sentence.
    
    Args:
        experiment: GemmaExperiment object with loaded models
        sentences: List of sentences to analyze
    
    Returns:
        List of detokenized final tokens
    """
    # Note: With lazy loading, tokenizer loads automatically when first accessed
    
    # Tokenize sentences
    tokenized = experiment.tokenizer(
        sentences,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )
    inputs = tokenized['input_ids'].to(DEVICE)
    attention_mask = tokenized['attention_mask'].to(DEVICE)
    
    # Create content mask to exclude special tokens
    bos_token_id = experiment.tokenizer.bos_token_id if experiment.tokenizer.bos_token_id is not None else -1
    eos_token_id = experiment.tokenizer.eos_token_id if experiment.tokenizer.eos_token_id is not None else -1
    pad_token_id = experiment.tokenizer.pad_token_id if experiment.tokenizer.pad_token_id is not None else -1
    
    content_mask = attention_mask.clone()
    if bos_token_id != -1:
        content_mask = content_mask & (inputs != bos_token_id)
    if eos_token_id != -1:
        content_mask = content_mask & (inputs != eos_token_id)
    if pad_token_id != -1:
        content_mask = content_mask & (inputs != pad_token_id)
    
    # Extract final content tokens
    final_tokens = []
    for sent_idx in range(inputs.shape[0]):
        sent_content_mask = content_mask[sent_idx].bool()
        if sent_content_mask.any():
            # Get indices of content tokens
            content_indices = torch.where(sent_content_mask)[0]
            last_content_idx = content_indices[-1].item()
            
            # Get the token ID and decode it
            token_id = inputs[sent_idx, last_content_idx].item()
            token_text = experiment.tokenizer.decode([token_id])
            final_tokens.append(token_text)
        else:
            final_tokens.append("<UNK>")
    
    return final_tokens

def get_dual_position_embeddings(experiment: GemmaExperiment, 
                                sentences: List[str],
                                layers: Optional[List[int]] = None) -> Dict[str, Dict[int, torch.Tensor]]:
    """
    Get embedding vectors for the final content token at two positions per layer:
    1. Post-attention (pre-MLP) 
    2. Post-block (residual stream)
    
    Args:
        experiment: GemmaExperiment object with loaded models
        sentences: List of sentences to analyze
        layers: List of layer indices to extract embeddings from (default: all layers)
    
    Returns:
        Dictionary with structure:
        {
            'post_attn': {layer_idx: tensor},  # Post-attention, pre-MLP
            'post_block': {layer_idx: tensor}  # Post-block (residual stream)
        }
    """
    # Note: With lazy loading, model loads automatically when first accessed
    
    if layers is None:
        layers = list(range(NUM_LAYERS))
    
    # Tokenize sentences
    tokenized = experiment.tokenizer(
        sentences,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )
    inputs = tokenized['input_ids'].to(DEVICE)
    attention_mask = tokenized['attention_mask'].to(DEVICE)
    
    # Create content mask to exclude special tokens
    bos_token_id = experiment.tokenizer.bos_token_id if experiment.tokenizer.bos_token_id is not None else -1
    eos_token_id = experiment.tokenizer.eos_token_id if experiment.tokenizer.eos_token_id is not None else -1
    pad_token_id = experiment.tokenizer.pad_token_id if experiment.tokenizer.pad_token_id is not None else -1
    
    content_mask = attention_mask.clone()
    if bos_token_id != -1:
        content_mask = content_mask & (inputs != bos_token_id)
    if eos_token_id != -1:
        content_mask = content_mask & (inputs != eos_token_id)
    if pad_token_id != -1:
        content_mask = content_mask & (inputs != pad_token_id)
    
    # Storage for activations
    post_attn_acts = {}
    post_block_acts = {}
    handles = []
    
    def make_post_attn_hook(idx):
        def hook(mod, inp, out):
            post_attn_acts[idx] = out.detach()
        return hook
    
    def make_post_block_hook(idx):
        def hook(mod, inp, out):
            post_block_acts[idx] = out.detach()
        return hook
    
    # Register hooks for specified layers
    for layer_idx in layers:
        # Hook after attention but before MLP - use hook_resid_mid
        h1 = experiment.model.blocks[layer_idx].hook_resid_mid.register_forward_hook(make_post_attn_hook(layer_idx))
        handles.append(h1)
        
        # Hook after entire block (post-MLP, residual stream) - use hook_resid_post
        h2 = experiment.model.blocks[layer_idx].hook_resid_post.register_forward_hook(make_post_block_hook(layer_idx))
        handles.append(h2)
    
    # Forward pass
    with torch.no_grad():
        _ = experiment.model.forward(inputs)
    
    # Remove hooks
    for h in handles:
        h.remove()
    
    # Extract final content token embeddings for each sentence at both positions
    final_embeddings = {
        'post_attn': {},
        'post_block': {}
    }
    
    # Process post-attention embeddings
    for layer_idx in layers:
        if layer_idx in post_attn_acts:
            layer_embeddings = []
            layer_acts = post_attn_acts[layer_idx]  # [batch_size, seq_len, hidden_dim]
            
            for sent_idx in range(layer_acts.shape[0]):
                sent_content_mask = content_mask[sent_idx].bool()
                if sent_content_mask.any():
                    content_indices = torch.where(sent_content_mask)[0]
                    last_content_idx = content_indices[-1].item()
                    final_embedding = layer_acts[sent_idx, last_content_idx]
                    layer_embeddings.append(final_embedding)
                else:
                    layer_embeddings.append(torch.zeros_like(layer_acts[sent_idx, 0]))
            
            final_embeddings['post_attn'][layer_idx] = torch.stack(layer_embeddings)
    
    # Process post-block embeddings
    for layer_idx in layers:
        if layer_idx in post_block_acts:
            layer_embeddings = []
            layer_acts = post_block_acts[layer_idx]  # [batch_size, seq_len, hidden_dim]
            
            for sent_idx in range(layer_acts.shape[0]):
                sent_content_mask = content_mask[sent_idx].bool()
                if sent_content_mask.any():
                    content_indices = torch.where(sent_content_mask)[0]
                    last_content_idx = content_indices[-1].item()
                    final_embedding = layer_acts[sent_idx, last_content_idx]
                    layer_embeddings.append(final_embedding)
                else:
                    layer_embeddings.append(torch.zeros_like(layer_acts[sent_idx, 0]))
            
            final_embeddings['post_block'][layer_idx] = torch.stack(layer_embeddings)
    
    return final_embeddings

def plot_dual_position_similarities(experiment: GemmaExperiment,
                                   sentences: List[str],
                                   layers: Optional[List[int]] = None,
                                   sentence_pairs: Optional[List[Tuple[int, int]]] = None):
    """
    Plot similarity as a function of position (both post-attention and post-block) across all layers.
    X-axis will have 52 positions: 26 post-attention + 26 post-block positions.
    
    Args:
        experiment: GemmaExperiment object with loaded models
        sentences: List of sentences to analyze
        layers: List of layer indices to analyze (default: all layers 0-25)
        sentence_pairs: List of (i, j) tuples specifying which sentence pairs to plot.
                       If None, plots all unique pairs.
    """
    if layers is None:
        layers = list(range(NUM_LAYERS))
    
    num_sentences = len(sentences)
    
    # Generate all unique pairs if not specified
    if sentence_pairs is None:
        sentence_pairs = [(i, j) for i in range(num_sentences) for j in range(i+1, num_sentences)]
    
    # Get embeddings at both positions
    dual_embeddings = get_dual_position_embeddings(experiment, sentences, layers)
    
    # Compute similarities for both positions
    print("Computing similarities...")
    post_attn_similarities = {}
    post_block_similarities = {}
    
    for layer_idx in layers:
        if layer_idx in dual_embeddings['post_attn']:
            sim_matrix = compute_pairwise_cosine_similarity(dual_embeddings['post_attn'][layer_idx])
            post_attn_similarities[layer_idx] = sim_matrix
            
        if layer_idx in dual_embeddings['post_block']:
            sim_matrix = compute_pairwise_cosine_similarity(dual_embeddings['post_block'][layer_idx])
            post_block_similarities[layer_idx] = sim_matrix
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    
    ax.set_title('Cosine Similarities: Post-Attention vs Post-Block Across All Layers')
    ax.set_xlabel('Position')
    ax.set_ylabel('Cosine Similarity')
    ax.grid(True, alpha=0.3)
    
    # Plot each sentence pair
    for pair_idx, (i, j) in enumerate(sentence_pairs):
        x_positions = []
        similarities = []
        labels = []
        
        # Add post-attention similarities
        for layer in sorted(layers):
            if layer in post_attn_similarities:
                sim_matrix = post_attn_similarities[layer]
                similarity = sim_matrix[i, j].item()
                x_positions.append(layer * 2)  # Even positions for post-attention
                similarities.append(similarity)
                labels.append(f"L{layer}_attn")
        
        # Add post-block similarities
        for layer in sorted(layers):
            if layer in post_block_similarities:
                sim_matrix = post_block_similarities[layer]
                similarity = sim_matrix[i, j].item()
                x_positions.append(layer * 2 + 1)  # Odd positions for post-block
                similarities.append(similarity)
                labels.append(f"L{layer}_block")
        
        # Sort by x_position to ensure proper plotting order
        sorted_data = sorted(zip(x_positions, similarities))
        x_pos_sorted, sim_sorted = zip(*sorted_data)
        
        pair_label = f"S{i+1} vs S{j+1}"
        ax.plot(x_pos_sorted, sim_sorted, marker='o', label=pair_label, linewidth=2, markersize=4)
    
    # Set x-axis ticks and labels
    x_ticks = []
    x_tick_labels = []
    for layer in sorted(layers):
        x_ticks.extend([layer * 2, layer * 2 + 1])
        x_tick_labels.extend([f"L{layer}\nattn", f"L{layer}\nblock"])
    
    ax.set_xticks(x_ticks[::4])  # Show every 4th tick to avoid overcrowding
    ax.set_xticklabels([x_tick_labels[i] for i in range(0, len(x_tick_labels), 4)], rotation=45)
    
    ax.legend()
    plt.tight_layout()
    plt.show()

def comprehensive_forward_pass_analysis(experiment: GemmaExperiment, 
                                      sentences: List[str],
                                      layers: Optional[List[int]] = None) -> Dict:
    """
    Process a list of sentences and capture activations from all major hook points.
    
    Args:
        experiment: GemmaExperiment object with loaded models
        sentences: List of sentences to analyze
        layers: List of layer indices to analyze (default: all layers)
    
    Returns:
        Dictionary containing:
        {
            'sentences': List of input sentences,
            'tokenized': List of tokenized sequences for each sentence,
            'activations': {
                'attention': {
                    layer_idx: {
                        'hook_k': tensor,
                        'hook_q': tensor,
                        'hook_v': tensor,
                        'hook_z': tensor,
                        'hook_attn_scores': tensor,
                        'hook_pattern': tensor,
                        'hook_result': tensor,
                        'hook_rot_k': tensor,
                        'hook_rot_q': tensor,
                    }
                },
                'mlp': {
                    layer_idx: {
                        'hook_pre': tensor,
                        'hook_pre_linear': tensor,
                        'hook_post': tensor,
                    }
                },
                'block_level': {
                    layer_idx: {
                        'hook_attn_in': tensor,
                        'hook_q_input': tensor,
                        'hook_k_input': tensor,
                        'hook_v_input': tensor,
                        'hook_mlp_in': tensor,
                        'hook_attn_out': tensor,
                        'hook_mlp_out': tensor,
                        'hook_resid_pre': tensor,
                        'hook_resid_mid': tensor,
                        'hook_resid_post': tensor,
                    }
                }
            }
        }
    """
    # Note: With lazy loading, model loads automatically when first accessed
    
    if layers is None:
        layers = list(range(NUM_LAYERS))
    
    print(f"Comprehensive forward pass analysis for {len(sentences)} sentences across {len(layers)} layers...")
    
    # Tokenize sentences and print tokenization
    tokenized = experiment.tokenizer(
        sentences,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )
    inputs = tokenized['input_ids'].to(DEVICE)
    attention_mask = tokenized['attention_mask'].to(DEVICE)
    
    # Print tokenization details
    print("\nTokenization Details:")
    print("=" * 60)
    tokenized_sentences = []
    for i, sentence in enumerate(sentences):
        print(f"\nSentence {i+1}: {sentence}")
        sentence_tokens = inputs[i][attention_mask[i].bool()].tolist()
        decoded_tokens = [experiment.tokenizer.decode([token_id]) for token_id in sentence_tokens]
        tokenized_sentences.append({
            'sentence': sentence,
            'token_ids': sentence_tokens,
            'tokens': decoded_tokens
        })
        print(f"Token IDs: {sentence_tokens}")
        print(f"Tokens: {decoded_tokens}")
    
    # Storage for all activations
    activations = {
        'attention': {layer_idx: {} for layer_idx in layers},
        'mlp': {layer_idx: {} for layer_idx in layers},
        'block_level': {layer_idx: {} for layer_idx in layers}
    }
    
    handles = []
    
    # Define hook creation functions for each type
    def make_attention_hook(layer_idx, hook_name):
        def hook(mod, inp, out):
            activations['attention'][layer_idx][hook_name] = out.detach() if torch.is_tensor(out) else out
        return hook
    
    def make_mlp_hook(layer_idx, hook_name):
        def hook(mod, inp, out):
            activations['mlp'][layer_idx][hook_name] = out.detach() if torch.is_tensor(out) else out
        return hook
    
    def make_block_hook(layer_idx, hook_name):
        def hook(mod, inp, out):
            activations['block_level'][layer_idx][hook_name] = out.detach() if torch.is_tensor(out) else out
        return hook
    
    print(f"\nRegistering hooks for {len(layers)} layers...")
    
    # Register hooks for specified layers
    for layer_idx in layers:
        block = experiment.model.blocks[layer_idx]
        
        # Dynamically find all attention hooks
        for name, module in block.attn.named_modules():
            if 'hook' in name and hasattr(module, 'register_forward_hook'):
                hook_name = name.split('.')[-1]  # Get the final part of the name
                h = module.register_forward_hook(make_attention_hook(layer_idx, hook_name))
                handles.append(h)
        
        # Dynamically find all MLP hooks
        for name, module in block.mlp.named_modules():
            if 'hook' in name and hasattr(module, 'register_forward_hook'):
                hook_name = name.split('.')[-1]  # Get the final part of the name
                h = module.register_forward_hook(make_mlp_hook(layer_idx, hook_name))
                handles.append(h)
        
        # Dynamically find all block-level hooks
        for name, module in block.named_modules():
            if ('hook' in name and 
                hasattr(module, 'register_forward_hook') and
                name.count('.') == 0):  # Only direct children of block, not nested
                hook_name = name
                h = module.register_forward_hook(make_block_hook(layer_idx, hook_name))
                handles.append(h)
    
    print(f"Registered {len(handles)} hooks total")
    
    # Forward pass
    print("Running forward pass...")
    with torch.no_grad():
        model_output = experiment.model.forward(inputs)
    
    # Remove hooks
    for h in handles:
        h.remove()
    
    # Count captured activations
    total_activations = 0
    for category in activations.values():
        for layer_data in category.values():
            total_activations += len(layer_data)
    
    print(f"✅ Captured {total_activations} activation tensors")
    
    # Create summary statistics
    print("\nActivation Summary:")
    print("-" * 40)
    for category_name, category_data in activations.items():
        print(f"\n{category_name.upper()}:")
        # Only show layer 0
        if 0 in category_data and category_data[0]:
            hooks_captured = list(category_data[0].keys())
            print(f"  Layer 0: {len(hooks_captured)} hooks - {hooks_captured}")
    
    return {
        'sentences': sentences,
        'tokenized': tokenized_sentences,
        'model_output': model_output,
        'activations': activations,
        'input_ids': inputs,
        'attention_mask': attention_mask
    }

def print_activation_summary(analysis_result: Dict, layer_idx: Optional[int] = None):
    """
    Print a summary of captured activations, optionally for a specific layer.
    
    Args:
        analysis_result: Result from comprehensive_forward_pass_analysis
        layer_idx: Optional layer index to focus on (default: show all)
    """
    activations = analysis_result['activations']
    sentences = analysis_result['sentences']
    
    print(f"\nActivation Analysis Summary")
    print("=" * 60)
    print(f"Sentences analyzed: {len(sentences)}")
    
    if layer_idx is not None:
        layers_to_show = [layer_idx]
        print(f"Focusing on Layer {layer_idx}")
    else:
        # Get all layers that have any activations
        all_layers = set()
        for category_data in activations.values():
            all_layers.update(category_data.keys())
        layers_to_show = sorted(all_layers)
        print(f"Showing all layers: {layers_to_show}")
    
    for layer in layers_to_show:
        print(f"\n" + "="*40)
        print(f"LAYER {layer}")
        print("="*40)
        
        for category_name, category_data in activations.items():
            if layer in category_data and category_data[layer]:
                print(f"\n{category_name.upper()}:")
                for hook_name, tensor in category_data[layer].items():
                    if torch.is_tensor(tensor):
                        print(f"  {hook_name:20} -> {tensor.shape}")
                    else:
                        print(f"  {hook_name:20} -> {type(tensor)} (non-tensor)")

def get_token_activations_at_position(analysis_result: Dict, 
                                     sentence_idx: int, 
                                     token_position: int,
                                     layer_idx: int) -> Dict:
    """
    Extract activations for a specific token position in a specific sentence and layer.
    
    Args:
        analysis_result: Result from comprehensive_forward_pass_analysis
        sentence_idx: Index of the sentence (0-based)
        token_position: Position of the token in the sequence (0-based)
        layer_idx: Layer index
    
    Returns:
        Dictionary with activations for the specified token position
    """
    activations = analysis_result['activations']
    
    token_activations = {}
    
    for category_name, category_data in activations.items():
        if layer_idx in category_data:
            token_activations[category_name] = {}
            for hook_name, tensor in category_data[layer_idx].items():
                if torch.is_tensor(tensor) and len(tensor.shape) >= 2:
                    # Extract the specific token position
                    if len(tensor.shape) == 3:  # [batch, seq_len, hidden_dim]
                        token_activations[category_name][hook_name] = tensor[sentence_idx, token_position]
                    elif len(tensor.shape) == 4:  # [batch, seq_len, n_heads, head_dim] or similar
                        token_activations[category_name][hook_name] = tensor[sentence_idx, token_position]
                    else:
                        # For other shapes, just include the full tensor
                        token_activations[category_name][hook_name] = tensor
    
    return token_activations

def analyze_final_token_differences(analysis_result: Dict, 
                                   layers: Optional[List[int]] = None,
                                   max_layers_to_show: int = 5) -> None:
    """
    Analyze L2 norms and attention patterns for final tokens of two sentences.
    
    Args:
        analysis_result: Result from comprehensive_forward_pass_analysis
        layers: List of layer indices to analyze (default: first few layers)
        max_layers_to_show: Maximum number of layers to display (default: 5)
    """
    sentences = analysis_result['sentences']
    activations = analysis_result['activations']
    
    if len(sentences) != 2:
        print(f"❌ This function requires exactly 2 sentences, got {len(sentences)}")
        return
    
    if layers is None:
        # Get available layers and limit to max_layers_to_show
        all_layers = set()
        for category_data in activations.values():
            all_layers.update(category_data.keys())
        layers = sorted(all_layers)[:max_layers_to_show]
    
    print(f"Final Token Analysis: {len(layers)} layers")
    print("=" * 60)
    print(f"S0: {sentences[0]}")
    print(f"S1: {sentences[1]}")
    print()
    
    # Get tokenization info to find final token positions
    tokenized = analysis_result['tokenized']
    final_pos_0 = len(tokenized[0]['token_ids']) - 1
    final_pos_1 = len(tokenized[1]['token_ids']) - 1
    
    print(f"Final token positions: S0[{final_pos_0}], S1[{final_pos_1}]")
    print()
    
    # Hook names to analyze (in order)
    hook_names = ['hook_resid_pre', 'hook_attn_out', 'hook_resid_mid', 'hook_mlp_out', 'hook_resid_post']
    
    for layer_idx in layers:
        print(f"LAYER {layer_idx}")
        print("-" * 20)
        
        # L2 norms and cosine similarity for block-level hooks
        for hook_name in hook_names:
            if (layer_idx in activations['block_level'] and 
                hook_name in activations['block_level'][layer_idx]):
                
                tensor = activations['block_level'][layer_idx][hook_name]
                
                # Extract final token activations for both sentences
                if tensor.shape[0] >= 2:  # Check we have at least 2 sentences
                    s0_final = tensor[0, final_pos_0]  # [hidden_dim]
                    s1_final = tensor[1, final_pos_1]  # [hidden_dim] 
                    
                    # Compute L2 norms
                    norm_s0 = torch.norm(s0_final, p=2).item()
                    norm_s1 = torch.norm(s1_final, p=2).item()
                    
                    # Compute cosine similarity
                    cos_sim = F.cosine_similarity(s0_final.unsqueeze(0), s1_final.unsqueeze(0)).item()
                    
                    print(f"{hook_name:15} ||s0||: {norm_s0:6.2f}  ||s1||: {norm_s1:6.2f}  cos_sim: {cos_sim:6.3f}")
        
        # Attention patterns for each head (last row only)
        if (layer_idx in activations['attention'] and 
            'hook_pattern' in activations['attention'][layer_idx]):
            
            pattern_tensor = activations['attention'][layer_idx]['hook_pattern']
            # Shape: [batch, n_heads, seq_len, seq_len]
            
            if pattern_tensor.shape[0] >= 2:
                n_heads = pattern_tensor.shape[1]
                print(f"\nAttention Patterns ({n_heads} heads) - Last Row Only:")
                
                for head_idx in range(n_heads):  # Show all heads
                    s0_pattern = pattern_tensor[0, head_idx]  # [seq_len, seq_len]
                    s1_pattern = pattern_tensor[1, head_idx]  # [seq_len, seq_len]
                    
                    print(f"  Head {head_idx}:")
                    # Only show the last row of each pattern matrix
                    s0_last_row = s0_pattern[-1]  # Last row
                    s1_last_row = s1_pattern[-1]  # Last row
                    
                    s0_row_str = "    S0 last row: [" + ", ".join([f"{x:.3f}" for x in s0_last_row.tolist()]) + "]"
                    s1_row_str = "    S1 last row: [" + ", ".join([f"{x:.3f}" for x in s1_last_row.tolist()]) + "]"
                    print(s0_row_str)
                    print(s1_row_str)
                    print()
        
        print()  # Empty line between layers

def find_similar_tokens_by_embedding(experiment: GemmaExperiment, 
                                    target_token: str, 
                                    top_k: int = 20) -> List[Tuple[str, int, float]]:
    """
    Find tokens most similar to a target token by embedding similarity.
    
    Args:
        experiment: GemmaExperiment object with loaded model and tokenizer
        target_token: The target token to find similarities for
        top_k: Number of most similar tokens to return (default: 20)
    
    Returns:
        List of tuples (token_text, token_id, similarity_score)
    """
    import torch
    import torch.nn.functional as F
    
    # Get the token ID for the target token
    try:
        target_token_id = experiment.tokenizer.encode(target_token, add_special_tokens=False)[0]
    except IndexError:
        print(f"❌ Could not tokenize '{target_token}'")
        return []
    
    print(f"Target token: '{target_token}' (ID: {target_token_id})")
    
    # Get the embedding matrix from the model
    embed_matrix = experiment.model.embed.W_E  # Shape: [vocab_size, d_model]
    print(f"Embedding matrix shape: {embed_matrix.shape}")
    
    # Get the embedding vector for the target token
    target_embedding = embed_matrix[target_token_id]  # Shape: [d_model]
    print(f"Target embedding shape: {target_embedding.shape}")
    
    # Compute cosine similarities with all tokens
    # Normalize embeddings for cosine similarity
    normalized_embeddings = F.normalize(embed_matrix, p=2, dim=1)  # [vocab_size, d_model]
    normalized_target = F.normalize(target_embedding.unsqueeze(0), p=2, dim=1)  # [1, d_model]
    
    # Compute similarities
    similarities = torch.mm(normalized_target, normalized_embeddings.t()).squeeze()  # [vocab_size]
    
    # Get top k most similar tokens (including the target itself)
    top_similarities, top_indices = torch.topk(similarities, k=top_k)
    
    print(f"\nTop {top_k} tokens most similar to '{target_token}':")
    print("-" * 60)
    
    results = []
    for i, (similarity, token_id) in enumerate(zip(top_similarities, top_indices)):
        # Decode the token
        token_text = experiment.tokenizer.decode([token_id.item()])
        similarity_score = similarity.item()
        
        print(f"{i+1:2d}. Token ID {token_id.item():6d}: {repr(token_text)} (similarity: {similarity_score:.4f})")
        results.append((token_text, token_id.item(), similarity_score))
    
    return results

def print_model_structure(experiment: GemmaExperiment) -> None:
    """
    Print the structure and configuration of the Gemma model.
    
    Args:
        experiment: GemmaExperiment object (model loads automatically if needed)
    """
    print("Model structure:")
    print("="*50)
    print(f"Model name: {experiment.model.cfg.model_name}")
    print(f"Number of layers: {experiment.model.cfg.n_layers}")
    print(f"Hidden dimension: {experiment.model.cfg.d_model}")
    print(f"Number of attention heads: {experiment.model.cfg.n_heads}")

    print(f"\nBlocks structure (Block 0 of {len(experiment.model.blocks)}):")
    print(experiment.model.blocks[0])

def plot_sentence_similarities(experiment: GemmaExperiment, 
                              sentences: List[str],
                              layers: Optional[List[int]] = None,
                              include_sae: bool = True) -> None:
    """
    Unified function to analyze and plot sentence similarities across layers.
    
    Args:
        experiment: GemmaExperiment object (loads models automatically)
        sentences: List of sentences to analyze
        layers: List of layer indices to analyze (default: all layers)
        include_sae: Whether to include SAE feature similarities
    """
    import matplotlib.pyplot as plt
    
    print(f"Analyzing and plotting similarities for {len(sentences)} sentences...")
    
    if layers is None:
        layers = list(range(NUM_LAYERS))  # All 26 layers by default
    
    # Get residual embeddings (loads model automatically)
    residual_embeddings = get_final_token_embeddings(experiment, sentences, layers)
    
    # Compute residual similarities
    residual_similarities = {}
    for layer_idx in layers:
        if layer_idx in residual_embeddings:
            similarity_matrix = compute_pairwise_cosine_similarity(residual_embeddings[layer_idx])
            residual_similarities[layer_idx] = similarity_matrix
    
    results = {
        'sentences': sentences,
        'residual_similarities': residual_similarities
    }
    
    # Optionally compute SAE feature similarities
    if include_sae:
        sae_features = get_final_token_sae_features(experiment, sentences, layers)
        sae_similarities = {}
        for layer_idx in layers:
            if layer_idx in sae_features:
                similarity_matrix = compute_pairwise_cosine_similarity(sae_features[layer_idx])
                sae_similarities[layer_idx] = similarity_matrix
        
        results['sae_similarities'] = sae_similarities
    
    # Plot similarities
    num_sentences = len(sentences)
    sentence_pairs = []
    for i in range(num_sentences):
        for j in range(i+1, num_sentences):
            sentence_pairs.append((i, j))
    
    # Get available layers
    residual_layers = sorted(results['residual_similarities'].keys()) if 'residual_similarities' in results else []
    sae_layers = sorted(results['sae_similarities'].keys()) if 'sae_similarities' in results else []
    
    # Create subplots
    fig_width = 12 if sae_layers else 6
    fig, axes = plt.subplots(1, 2 if sae_layers else 1, figsize=(fig_width, 5))
    if not sae_layers:
        axes = [axes]  # Make it a list for consistent indexing
    
    ax1 = axes[0]
    ax2 = axes[1] if len(axes) > 1 else None
    
    # Plot embedding space similarities
    if residual_layers:
        ax1.set_title('Residual Space Similarities Across Layers')
        ax1.set_xlabel('Layer')
        ax1.set_ylabel('Cosine Similarity')
        ax1.grid(True, alpha=0.3)
        
        for pair_idx, (i, j) in enumerate(sentence_pairs):
            similarities = []
            layers_plot = []
            for layer in residual_layers:
                if layer in results['residual_similarities']:
                    sim_matrix = results['residual_similarities'][layer]
                    similarity = sim_matrix[i, j].item()
                    similarities.append(similarity)
                    layers_plot.append(layer)
            
            label = f"S{i+1} vs S{j+1}"
            ax1.plot(layers_plot, similarities, marker='o', label=label, linewidth=2)
        
        ax1.legend()
        ax1.set_xticks(residual_layers)
    
    # Plot SAE feature space similarities
    if sae_layers:
        ax2.set_title('SAE Feature Space Similarities Across Layers')
        ax2.set_xlabel('Layer')
        ax2.set_ylabel('Cosine Similarity')
        ax2.grid(True, alpha=0.3)
        
        for pair_idx, (i, j) in enumerate(sentence_pairs):
            similarities = []
            layers_plot = []
            for layer in sae_layers:
                if layer in results['sae_similarities']:
                    sim_matrix = results['sae_similarities'][layer]
                    similarity = sim_matrix[i, j].item()
                    similarities.append(similarity)
                    layers_plot.append(layer)
            
            label = f"S{i+1} vs S{j+1}"
            ax2.plot(layers_plot, similarities, marker='s', label=label, linewidth=2)
        
        ax2.legend()
        ax2.set_xticks(sae_layers)
    
    plt.tight_layout()
    plt.show()
