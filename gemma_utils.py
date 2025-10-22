#!/usr/bin/env python3
"""
Gemma Model Utilities for Mechanistic Interpretability Research

This module provides comprehensive utility functions for working with Google's Gemma 2 transformer model
and Sparse Autoencoders (SAEs) in mechanistic interpretability research. It includes model loading,
activation extraction, embedding analysis, similarity computation, and visualization tools.

CORE FUNCTIONALITY:
==================
The module enables detailed analysis of transformer internals through:
- Proper model and tokenizer loading with device optimization
- Multi-layer activation extraction from residual streams
- SAE feature encoding for sparse representation analysis
- Token-level embedding analysis with content token filtering
- Comprehensive similarity analysis across model layers
- Advanced visualization tools for comparative analysis

MODEL LOADING & CONFIGURATION:
=============================
- **Gemma 2 Model**: Loads google/gemma-2-2b with proper transformer_lens configuration
- **Device Optimization**: CPU optimized (MPS slower for this use case)
- **Tokenizer Setup**: Proper padding configuration with right-side padding
- **Gradient Disabled**: All operations in inference mode for efficiency
- **SAE Integration**: Loads all 26 Gemma Scope SAEs for sparse feature analysis

ACTIVATION EXTRACTION:
=====================
- **Multi-Position Analysis**: Extract embeddings from multiple transformer positions
- **Content Token Filtering**: Automatically excludes BOS, EOS, and padding tokens
- **Final Token Focus**: Specialized extraction of final content token representations
- **Hook-based Capture**: Uses transformer_lens hooks for precise activation capture
- **Dual Position Support**: Both post-attention and post-block activation extraction

EMBEDDING ANALYSIS:
==================
- **Residual Stream Analysis**: Direct analysis of model's residual stream representations
- **SAE Feature Analysis**: Sparse autoencoder feature activation patterns
- **Token-level Precision**: Individual token position analysis within sequences
- **Layer-wise Comparison**: Cross-layer embedding evolution tracking
- **Similarity Metrics**: Cosine similarity computation for embedding comparisons

VISUALIZATION CAPABILITIES:
==========================
- **Multi-layer Similarity Plots**: Track similarity evolution across all 26 layers
- **Dual Position Plotting**: Compare post-attention vs post-block similarities
- **Target Similarity Analysis**: Compare multiple sentences against a target reference
- **Comprehensive Plotting**: Support for both residual and SAE feature space analysis
- **Interactive Analysis**: Token-by-token activation inspection and comparison

ADVANCED ANALYSIS FEATURES:
==========================
- **Comprehensive Forward Pass**: Capture activations from all major hook points
- **Attention Pattern Analysis**: Detailed attention matrix inspection
- **Token Similarity Search**: Find tokens with similar embeddings in vocabulary
- **L2 Norm Analysis**: Magnitude comparison of embeddings across layers
- **Differential Analysis**: Compare final token differences between sentence pairs

TOKEN PROCESSING:
=================
- **Content Token Detection**: Automatic filtering of special tokens (BOS, EOS, PAD)
- **Final Token Focus**: Specialized extraction of final content token representations
- **Tokenization Details**: Comprehensive tokenization inspection and debugging
- **Multi-sentence Support**: Batch processing of multiple sentences simultaneously

SIMILARITY COMPUTATION:
======================
- **Pairwise Analysis**: All-vs-all sentence similarity matrices
- **Target-based Analysis**: Single target vs multiple comparison sentences
- **Cross-layer Tracking**: Similarity evolution through transformer layers
- **Multiple Metrics**: Cosine similarity with extensible metric framework
- **Efficient Computation**: Optimized PyTorch operations for large-scale analysis

RESEARCH APPLICATIONS:
=====================
This module is designed for mechanistic interpretability research including:
- Circuit analysis preparation (embedding extraction for ACDC)
- Representation similarity analysis across transformer layers
- SAE feature activation pattern studies
- Attention mechanism investigation
- Token-level semantic analysis
- Cross-linguistic or cross-domain representation comparison
- Model behavior analysis on specific linguistic phenomena

TECHNICAL SPECIFICATIONS:
========================
- **Model**: Google Gemma 2-2B (26 layers, 2048 hidden dimensions)
- **SAEs**: Gemma Scope 2B canonical residual SAEs (16k features each)
- **Device**: CPU optimized (configurable)
- **Precision**: Full precision analysis with gradient computation disabled
- **Memory**: Efficient activation caching and batch processing
- **Tokenization**: Proper handling of special tokens and padding

USAGE PATTERNS:
==============
1. **Model Setup**:
   ```python
   model, tokenizer = load_gemma_model()
   sae_list = load_gemma_saes()
   ```

2. **Basic Similarity Analysis**:
   ```python
   sentences = ["He kicked the bucket", "He died", "He kicked the pail"]
   plot_sentence_similarities(model, sae_list, sentences)
   ```

3. **Target-based Analysis**:
   ```python
   plot_target_similarities(model, sentences, target_sentence="He died")
   ```

4. **Comprehensive Analysis**:
   ```python
   analysis = comprehensive_forward_pass_analysis(model, sentences)
   print_activation_summary(analysis)
   analyze_final_token_differences(analysis)
   ```

5. **Embedding Extraction for External Analysis**:
   ```python
   embeddings = get_final_token_embeddings(model, sentences, layers=[10, 15, 20])
   sae_features = get_final_token_sae_features(model, sae_list, sentences)
   ```

INTEGRATION WITH ACDC:
=====================
This module provides the foundation for ACDC (Automatic Circuit Discovery) analysis:
- Model loading compatible with simple_acdc.py requirements
- Embedding extraction for similarity computation in circuit discovery
- Activation analysis for understanding discovered circuits
- Visualization tools for circuit interpretation and validation

PERFORMANCE CONSIDERATIONS:
==========================
- CPU optimization for consistent performance across devices
- Efficient batch processing for multiple sentence analysis
- Memory-conscious activation caching
- Gradient computation disabled for inference-only operations
- Selective layer analysis to reduce computational overhead

DEPENDENCIES:
============
- torch: Core tensor operations and model execution
- transformer_lens: HookedTransformer for activation extraction
- sae_lens: Sparse autoencoder loading and operation
- transformers: Tokenizer and model configuration
- matplotlib: Visualization and plotting
- numpy: Numerical operations and array handling

AUTHORS: Research implementation for mechanistic interpretability studies
LICENSE: Research use - see repository for full license details
VERSION: Compatible with Gemma 2-2B and Gemma Scope SAEs
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

def load_gemma_model(verbose: bool = True) -> Tuple[HookedTransformer, AutoTokenizer]:
    """
    Load Gemma 2 model and tokenizer.
    
    Args:
        verbose: Whether to print loading progress
        
    Returns:
        Tuple of (model, tokenizer)
    """
    if verbose:
        print("Loading Gemma 2 model and tokenizer...")
    
    # Disable gradients
    torch.set_grad_enabled(False)
    
    # Load model
    model = HookedTransformer.from_pretrained(
        MODEL_NAME,
        device=DEVICE,
        center_unembed=False,
        center_writing_weights=False
    )
    model.eval()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    if verbose:
        print("✅ Model and tokenizer loaded successfully!")
    
    return model, tokenizer

def load_gemma_saes(verbose: bool = True) -> List[Tuple[int, SAE]]:
    """
    Load all SAEs for Gemma 2 model.
    
    Args:
        verbose: Whether to print loading progress
        
    Returns:
        List of tuples (layer_idx, sae)
    """
    if verbose:
        print("Loading all SAEs...")
    
    sae_list = []
    iterator = tqdm(range(NUM_LAYERS), desc="Loading SAEs") if verbose else range(NUM_LAYERS)
    
    for i in iterator:
        sae = SAE.from_pretrained(
            release=SAE_RELEASE,
            sae_id=f"layer_{i}/width_16k/canonical",
            device=DEVICE
        )
        sae.eval()
        sae_list.append((i, sae))
    
    if verbose:
        print("✅ All SAEs loaded successfully!")
    
    return sae_list

def detokenize(model: HookedTransformer, text: str) -> List[str]:
    """Detokenize text into words"""
    tokens = model.to_tokens(text, prepend_bos=True)
    words = [model.tokenizer.decode(tok) for tok in tokens[0]]
    return words

def get_final_token_embeddings(model: HookedTransformer,
                              sentences: List[str],
                              layers: Optional[List[int]] = None) -> Dict[int, torch.Tensor]:
    """
    Get embedding vectors for the final content token of each sentence from specified layers.
    
    Args:
        model: HookedTransformer model
        sentences: List of sentences to analyze
        layers: List of layer indices to extract embeddings from (default: all layers)
    
    Returns:
        Dictionary mapping layer_idx -> tensor of shape [num_sentences, hidden_dim]
    """
    if layers is None:
        layers = list(range(NUM_LAYERS))
    
    # Tokenize sentences using model's tokenizer
    inputs = model.to_tokens(sentences, prepend_bos=True)  # [batch_size, seq_len]
    inputs = inputs.to(DEVICE)
    
    # Create content mask to exclude special tokens (BOS, EOS, PAD)
    bos_token_id = model.tokenizer.bos_token_id if model.tokenizer.bos_token_id is not None else -1
    eos_token_id = model.tokenizer.eos_token_id if model.tokenizer.eos_token_id is not None else -1
    pad_token_id = model.tokenizer.pad_token_id if model.tokenizer.pad_token_id is not None else -1
    
    content_mask = torch.ones_like(inputs, dtype=torch.bool)
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
        h = model.blocks[layer_idx].register_forward_hook(make_hook(layer_idx))
        handles.append(h)
    
    # Forward pass
    with torch.no_grad():
        _ = model.forward(inputs)
    
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

def get_final_token_sae_features(model: HookedTransformer,
                                sae_list: List[Tuple[int, SAE]], sentences: List[str],
                                layers: Optional[List[int]] = None) -> Dict[int, torch.Tensor]:
    """
    Get SAE feature vectors for the final content token of each sentence from specified layers.
    
    Args:
        model: HookedTransformer model
        sae_list: List of tuples (layer_idx, sae)
        sentences: List of sentences to analyze
        layers: List of layer indices to extract features from (default: all layers)
    
    Returns:
        Dictionary mapping layer_idx -> tensor of shape [num_sentences, num_features]
    """
    if layers is None:
        layers = list(range(NUM_LAYERS))
    
    # First get the residual embeddings
    residual_embeddings = get_final_token_embeddings(model, sentences, layers)
    
    # Encode with SAEs
    sae_features = {}
    
    for layer_idx in layers:
        if layer_idx not in residual_embeddings:
            continue
        
        # Find the corresponding SAE
        sae = None
        for l_idx, s in sae_list:
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

def get_dual_position_embeddings(model: HookedTransformer,
                                sentences: List[str],
                                layers: Optional[List[int]] = None) -> Dict[str, Dict[int, torch.Tensor]]:
    """
    Get embedding vectors for the final content token at two positions per layer:
    1. Post-attention (pre-MLP) 
    2. Post-block (residual stream)
    
    Args:
        model: HookedTransformer model
        sentences: List of sentences to analyze
        layers: List of layer indices to extract embeddings from (default: all layers)
    
    Returns:
        Dictionary with structure:
        {
            'post_attn': {layer_idx: tensor},  # Post-attention, pre-MLP
            'post_block': {layer_idx: tensor}  # Post-block (residual stream)
        }
    """
    if layers is None:
        layers = list(range(NUM_LAYERS))
    
    # Tokenize sentences using model's tokenizer
    inputs = model.to_tokens(sentences, prepend_bos=True)  # [batch_size, seq_len]
    inputs = inputs.to(DEVICE)
    
    # Create content mask to exclude special tokens
    bos_token_id = model.tokenizer.bos_token_id if model.tokenizer.bos_token_id is not None else -1
    eos_token_id = model.tokenizer.eos_token_id if model.tokenizer.eos_token_id is not None else -1
    pad_token_id = model.tokenizer.pad_token_id if model.tokenizer.pad_token_id is not None else -1
    
    content_mask = torch.ones_like(inputs, dtype=torch.bool)
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
        h1 = model.blocks[layer_idx].hook_resid_mid.register_forward_hook(make_post_attn_hook(layer_idx))
        handles.append(h1)
        
        # Hook after entire block (post-MLP, residual stream) - use hook_resid_post
        h2 = model.blocks[layer_idx].hook_resid_post.register_forward_hook(make_post_block_hook(layer_idx))
        handles.append(h2)
    
    # Forward pass
    with torch.no_grad():
        _ = model.forward(inputs)
    
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

def plot_dual_position_similarities(model: HookedTransformer,
                                   sentences: List[str],
                                   layers: Optional[List[int]] = None,
                                   sentence_pairs: Optional[List[Tuple[int, int]]] = None):
    """
    Plot similarity as a function of position (both post-attention and post-block) across all layers.
    X-axis will have 52 positions: 26 post-attention + 26 post-block positions.
    
    Args:
        model: HookedTransformer model
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
    dual_embeddings = get_dual_position_embeddings(model, sentences, layers)
    
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

def comprehensive_forward_pass_analysis(model: HookedTransformer,
                                      sentences: List[str],
                                      layers: Optional[List[int]] = None) -> Dict:
    """
    Process a list of sentences and capture activations from all major hook points.
    
    Args:
        model: HookedTransformer model
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
    if layers is None:
        layers = list(range(NUM_LAYERS))
    
    print(f"Comprehensive forward pass analysis for {len(sentences)} sentences across {len(layers)} layers...")
    
    # Tokenize sentences using model's tokenizer
    inputs = model.to_tokens(sentences, prepend_bos=True)  # [batch_size, seq_len]
    inputs = inputs.to(DEVICE)
    
    # Print tokenization details
    print("\nTokenization Details:")
    print("=" * 60)
    tokenized_sentences = []
    for i, sentence in enumerate(sentences):
        print(f"\nSentence {i+1}: {sentence}")
        sentence_tokens = inputs[i].tolist()
        # Remove padding tokens (typically 0 or a specific pad token)
        if hasattr(model.tokenizer, 'pad_token_id') and model.tokenizer.pad_token_id is not None:
            sentence_tokens = [tok for tok in sentence_tokens if tok != model.tokenizer.pad_token_id]
        decoded_tokens = [model.tokenizer.decode([token_id]) for token_id in sentence_tokens]
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
        block = model.blocks[layer_idx]
        
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
        model_output = model.forward(inputs)
    
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
        'input_ids': inputs
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

def find_similar_tokens_by_embedding(model: HookedTransformer,
                                    target_token: str, 
                                    top_k: int = 20) -> List[Tuple[str, int, float]]:
    """
    Find tokens most similar to a target token by embedding similarity.
    
    Args:
        model: HookedTransformer model
        target_token: The target token to find similarities for
        top_k: Number of most similar tokens to return (default: 20)
    
    Returns:
        List of tuples (token_text, token_id, similarity_score)
    """
    import torch
    import torch.nn.functional as F
    
    # Get the token ID for the target token
    try:
        target_token_id = model.tokenizer.encode(target_token, add_special_tokens=False)[0]
    except IndexError:
        print(f"❌ Could not tokenize '{target_token}'")
        return []
    
    print(f"Target token: '{target_token}' (ID: {target_token_id})")
    
    # Get the embedding matrix from the model
    embed_matrix = model.embed.W_E  # Shape: [vocab_size, d_model]
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
        token_text = model.tokenizer.decode([token_id.item()])
        similarity_score = similarity.item()
        
        print(f"{i+1:2d}. Token ID {token_id.item():6d}: {repr(token_text)} (similarity: {similarity_score:.4f})")
        results.append((token_text, token_id.item(), similarity_score))
    
    return results

def print_model_structure(model: HookedTransformer) -> None:
    """
    Print the structure and configuration of the Gemma model.
    
    Args:
        model: HookedTransformer model
    """
    print("Model structure:")
    print("="*50)
    print(f"Model name: {model.cfg.model_name}")
    print(f"Number of layers: {model.cfg.n_layers}")
    print(f"Hidden dimension: {model.cfg.d_model}")
    print(f"Number of attention heads: {model.cfg.n_heads}")

    print(f"\nBlocks structure (Block 0 of {len(model.blocks)}):")
    print(model.blocks[0])

def plot_sentence_similarities(model: HookedTransformer,
                              sae_list: Optional[List[Tuple[int, SAE]]],
                              sentences: List[str],
                              layers: Optional[List[int]] = None,
                              include_sae: bool = True) -> None:
    """
    Unified function to analyze and plot sentence similarities across layers.
    
    Args:
        model: HookedTransformer model
        sae_list: List of tuples (layer_idx, sae) or None if not using SAEs
        sentences: List of sentences to analyze
        layers: List of layer indices to analyze (default: all layers)
        include_sae: Whether to include SAE feature similarities
    """
    import matplotlib.pyplot as plt
    
    print(f"Analyzing and plotting similarities for {len(sentences)} sentences...")
    
    if layers is None:
        layers = list(range(NUM_LAYERS))  # All 26 layers by default
    
    # Get residual embeddings
    residual_embeddings = get_final_token_embeddings(model, sentences, layers)
    
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
    if include_sae and sae_list is not None:
        sae_features = get_final_token_sae_features(model, sae_list, sentences, layers)
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

def plot_target_similarities(model: HookedTransformer,
                               sentences: List[str], target_sentence: str,
                               layers: Optional[List[int]] = None) -> None:
    """
    Plot similarities of all sentences compared to a target sentence across layers.
    Uses only the model (no SAEs) and final token embeddings.
    
    Args:
        model: HookedTransformer model
        sentences: List of sentences to compare
        target_sentence: The target sentence to compare all others against
        layers: List of layer indices to analyze (default: all layers)
    """
    import matplotlib.pyplot as plt
    
    print(f"Analyzing similarities to target sentence: '{target_sentence}'")
    print(f"Comparing {len(sentences)} sentences across layers...")
    
    if layers is None:
        layers = list(range(NUM_LAYERS))  # All 26 layers by default
    
    # Get residual embeddings for all sentences
    all_sentences = sentences + [target_sentence]
    residual_embeddings = get_final_token_embeddings(model, all_sentences, layers)
    target_idx = len(sentences)  # Index of target sentence in the combined list
    
    # Compute similarities to target sentence for each layer
    reference_similarities = {}
    for layer_idx in layers:
        if layer_idx in residual_embeddings:
            # Get embeddings for this layer
            layer_embeddings = residual_embeddings[layer_idx]  # [num_sentences+1, hidden_dim]
            
            # Extract target embedding (last sentence)
            target_embedding = layer_embeddings[target_idx:target_idx+1]  # [1, hidden_dim]
            
            # Extract comparison sentence embeddings (all but last)
            comparison_embeddings = layer_embeddings[:len(sentences)]  # [num_sentences, hidden_dim]
            
            # Compute cosine similarities between target and comparison sentences
            similarities = F.cosine_similarity(target_embedding, comparison_embeddings, dim=1)  # [num_sentences]
            reference_similarities[layer_idx] = similarities
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    ax.set_title(f'Cosine Similarities to Target: "{target_sentence}"')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Cosine Similarity')
    ax.grid(True, alpha=0.3)
    
    # Plot similarity for each sentence
    available_layers = sorted(reference_similarities.keys())
    
    for sent_idx, sentence in enumerate(sentences):
        similarities = []
        layers_plot = []
        
        for layer in available_layers:
            if layer in reference_similarities:
                similarity = reference_similarities[layer][sent_idx].item()
                similarities.append(similarity)
                layers_plot.append(layer)
        
        label = f"S{sent_idx+1}: {sentence[:20]}{'...' if len(sentence) > 20 else ''}"
        ax.plot(layers_plot, similarities, marker='o', label=label, linewidth=2)
    
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_xticks(available_layers[::2])  # Show every other layer to avoid crowding
    
    plt.tight_layout()
    plt.show()
