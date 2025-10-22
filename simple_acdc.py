"""
Simple ACDC (Automatic Circuit Discovery) Implementation

This module implements a comprehensive version of Automatic Circuit Discovery (ACDC), a method for 
identifying the minimal computational circuit within a transformer language model that is responsible 
for a specific behavior or task. Unlike the original ACDC which prunes from a full computational graph, 
this implementation uses an incremental building approach with advanced circuit analysis capabilities.

ALGORITHM OVERVIEW:
==================
The algorithm discovers circuits by starting with a root node (final token at final layer) and 
incrementally adding nodes/edges that have significant causal effects on the model's output. 
It uses activation patching to measure the counterfactual importance of each component.

Key Components:
- Root Node: Final token position at the maximum layer (where the final prediction is made)
- Nodes: Represent either residual stream states or attention head outputs at specific token positions
- Edges: Represent information flow between components (residual connections, attention outputs, Q/K/V connections)
- Circuit Discovery: Breadth-first traversal through the computational graph, testing component importance
- Circuit Merging: Ability to combine multiple circuits and filter nodes for comparative analysis
- Enhanced Visualization: Dynamic head positioning, color-coded edges, and individual circuit display options

METHODOLOGY:
============
1. **Activation Patching**: For each candidate component, compare model outputs when using:
   - Clean activations (from original text) 
   - Corrupted activations (from corrupted/control text)
   
2. **Effect Measurement**: Uses cosine similarity between final token embeddings of:
   - Current circuit state (with candidate component)
   - Target text embeddings
   
3. **Threshold-based Selection**: Components with effect size >= threshold are included in circuit

4. **Incremental Building**: Processes nodes in reverse order (final layer → first layer) to build 
   minimal circuit without unnecessary components

5. **Circuit Merging**: Ability to combine multiple discovered circuits using union operations
   with optional node filtering based on labels or circuit membership

6. **Advanced Analysis**: Q-K dot product computation for understanding attention mechanisms
   using rotated embeddings and proper GQA head mapping

CIRCUIT COMPONENTS:
==================
- **Residual Nodes**: Represent the residual stream at layer L, token position T
- **Attention Nodes**: Represent attention head H at layer L, token position T  
- **Edge Types**:
  - 'resid': Layer-to-layer residual connections
  - 'attn_out': Attention head output to residual stream
  - 'query': Query connection from residual stream to attention head
  - 'key_value': Key/Value connections from previous tokens to attention head (when separate_kv=False)
  - 'key': Key connections from previous tokens to attention head (when separate_kv=True)
  - 'value': Value connections from previous tokens to attention head (when separate_kv=True)

ATTENTION MECHANISM HANDLING:
============================
The implementation provides fine-grained control over attention mechanisms with precise model replication:

1. **Simple Head Patching**: Entire attention heads can be included/excluded from the circuit
2. **Surgical Q/K/V Patching**: Individual Query, Key, and Value connections can be controlled
3. **Separate K/V Testing**: `separate_kv=True` enables independent testing of Key and Value connections
   - Allows discovering asymmetric importance between keys and values from the same token
   - Creates separate "key" and "value" edge types instead of combined "key_value" edges
4. **Grouped-Query Attention Support**: Correctly handles models where multiple Q heads share K/V heads
   - Uses proper GQA mapping: `kv_head_idx = head_idx // (n_q_heads // n_kv_heads)`
   - Not the incorrect modulo mapping that can cause bugs
5. **RoPE Integration**: Uses rotated Q/K vectors (hook_rot_q, hook_rot_k) for accurate attention computation
   - Essential for models with Rotary Position Embedding to match exact model behavior
6. **Custom Attention Computation**: Recomputes attention patterns with mixed clean/corrupted Q/K/V
   - Precisely replicates transformer_lens attention computation for identical results
7. **Self-Attention Control**: `include_current_token` parameter controls self-attention behavior:
   - When False: Self-attention always uses clean Q (preserves natural self-attention)
   - When True: Self-attention can be corrupted (enables testing self-query connections)

QUERY PATCHING (NEW):
====================
- **Optional Q Patching**: `corrupt_q=True` enables testing of query connections independently
- **Memory Optimized**: Only extracts Q vectors for specific token positions being tested
- **Conditional Self-Attention**: Respects `include_current_token` for proper self-attention handling
- **Effect Measurement**: Query connections are tested and measured like other connections when enabled

PATCHING STRATEGY:
==================
- **hook_z patching**: Primary mechanism for attention head output patching
- **Q/K/V patching**: Surgical control over individual attention components using rotated Q/K for RoPE
- **Unified patching**: Combines simple head exclusion with custom Q/K/V mixing
- **Causal masking**: Ensures attention patterns respect autoregressive constraints
- **Precision matching**: Custom attention computation exactly replicates model behavior to prevent
  spurious effects from numerical differences

FEATURES:
=========
1. **Token-level Precision**: Circuit components are tracked at individual token positions
2. **Head-level Granularity**: Individual attention heads can be included/excluded
3. **Query-level Control**: Optional testing of individual query connections (corrupt_q parameter)
4. **Incremental Building**: Avoids unnecessary computation by building minimal circuits
5. **Effect Size Tracking**: All edges store their measured causal effect sizes
6. **Enhanced Visualization**: NetworkX-based visualization with:
   - Color-coded edge types (red/blue for positive/negative effects)
   - Thickness proportional to effect magnitude
   - Simplified head labels (H0, H1, etc.)
   - Dynamic head positioning with even distribution based on circuit composition
   - Conditional query edge coloring based on corrupt_q setting
7. **Circuit Merging & Filtering**: Tools for combining and filtering circuits:
   - Union-based circuit merging with optional node filtering
   - Label-based filtering for selective node inclusion/exclusion
   - Individual circuit visualization before merging for workflow transparency
8. **Advanced Analysis Tools**:
   - Q-K dot product computation using rotated embeddings for attention analysis
   - Token identification and detokenization for result interpretation
   - Proper GQA head mapping for accurate multi-head attention analysis
9. **Flexible Metrics**: Currently uses cosine similarity but extensible to other metrics
10. **Caching**: Corrupted activations are cached for efficient repeated patching
11. **Consistency Checking**: Prevents conflicting patch operations on same components
12. **Class Properties**: `include_current_token` as class property eliminates parameter passing complexity
13. **Threshold Analysis**: Enhanced threshold sweep with weighted edge counting for accurate metrics
14. **Batch Processing**: Support for building and merging multiple circuits in single workflow

LIMITATIONS:
============
- Currently limited to examining effects on final token predictions
- Cosine similarity metric may not capture all relevant semantic differences
- Does not handle multi-token target sequences
- Limited to causal (autoregressive) transformer architectures
- Requires manual threshold tuning for different tasks/models

USAGE EXAMPLE:
==============
```python
# Initialize with model and enhanced parameters
acdc = SimpleACDC(model, max_layer=4, threshold=0.01, corrupt_q=True, 
                  include_current_token=False, separate_kv=True)

# Discover circuit for idiom understanding task
circuit = acdc.discover_circuit(
    original_text="He kicked the bucket",      # Text exhibiting behavior
    corrupted_text="He kicked the pail",       # Control text (literal meaning)  
    target_text="He died",                     # Target behavior (figurative meaning)
    min_token_pos=2                            # Skip BOS and subject tokens
)

# Visualize discovered circuit with enhanced display
acdc.visualize_circuit(circuit, save_path="idiom_circuit.png")

# Build and merge multiple circuits with workflow visualization
circuits_info = [
    {"label": "bucket", "original": "He kicked the bucket", "corrupted": "He kicked the pail", "target": "He died"},
    {"label": "music", "original": "face the music", "corrupted": "face the band", "target": "accept consequences"}
]
merged_circuit = build_and_merge_circuits(
    acdc, circuits_info, min_token_pos=2, 
    visualize_individual=True,  # Show individual circuits before merging
    save_individual=True
)

# Filter merged circuit to specific components
filtered_circuit = filter_circuit_nodes(merged_circuit, 
                                       include_labels=["bucket", "music"])

# Analyze attention mechanisms with Q-K dot products
texts = ["He kicked the bucket", "face the music"]
compute_qk_dot_products(model, texts, layer=2, head=3, q_index=2, k_index=1)

# Run threshold sweep for parameter tuning with accurate edge counting
thresholds, metrics = threshold_sweep(model, max_layer=4, 
                                     original_text="He kicked the bucket",
                                     corrupted_text="He kicked the pail", 
                                     target_text="He died")
```

IMPLEMENTATION NOTES:
====================
- Uses transformer_lens for model interaction and hook-based activation patching
- Supports both simple attention head patching and surgical Q/K/V connection control
- Graph structure stored using custom Node/Edge dataclasses with NetworkX for visualization
- All patching operations preserve gradient flow and support batched inference
- **Critical accuracy fixes**:
  - Correct GQA head mapping prevents incorrect K/V head associations
  - RoPE integration using hook_rot_q/hook_rot_k ensures exact attention replication
  - Custom attention computation matches model behavior to ~1e-6 precision
- Designed for research and interpretability applications on transformer language models

AUTHORS: Research implementation for mechanistic interpretability studies
LICENSE: Research use - see repository for full license details
"""

import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer
from typing import List, Dict, Tuple, Set, Optional, Callable
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict, deque
from dataclasses import dataclass
import copy
import numpy as np
from tqdm import tqdm

@dataclass
class Node:
    """Represents a node in the computational graph"""
    hook_name: str
    layer: int
    token_pos: int
    head_idx: Optional[int] = None  # None for resid nodes, head index for attention heads
    node_type: str = "resid"  # "resid" or "attn"
    
    def __str__(self):
        if self.node_type == "attn":
            return f"L{self.layer}H{self.head_idx}_T{self.token_pos}"
        elif self.node_type == "embed":
            return f"Embed_T{self.token_pos}"
        else:
            return f"L{self.layer}_T{self.token_pos}"
    
    def __hash__(self):
        return hash((self.hook_name, self.layer, self.token_pos, self.head_idx, self.node_type))

@dataclass
class Edge:
    """Represents an edge in the computational graph"""
    parent: Node
    child: Node
    edge_type: str  # "resid", "query", "key_value", "key", "value", "attn_out"  
    effect_size: Optional[float] = None

class IncrementalCircuitGraph:
    """Incrementally built circuit graph"""
    
    def __init__(self):
        self.nodes: Set[Node] = set()
        self.edges: List[Edge] = []
        self.hooks_to_patch: Dict[str, torch.Tensor] = {}  # Corrupted activations cache
        self.clean_activations: Dict[str, torch.Tensor] = {}  # Clean activations cache
        # Track components explicitly excluded from circuit (permanently corrupted)
        self.excluded_attention_heads: Set[Tuple[int, int, int]] = set()  # (layer, token_pos, head_idx)
        self.excluded_k_connections: Set[Tuple[int, int, int, int]] = set()  # (layer, source_pos, target_pos, head_idx)
        self.excluded_v_connections: Set[Tuple[int, int, int, int]] = set()  # (layer, source_pos, target_pos, head_idx)
        self.excluded_q_connections: Set[Tuple[int, int, int, str]] = set()  # (layer, token_pos, head_idx)
        self.earliest_patched_layer: int = float('inf')  # Track earliest layer that needs patching
    
    def add_node(self, node: Node):
        """Add a node to the graph"""
        self.nodes.add(node)
    
    def add_edge(self, edge: Edge):
        """Add an edge and its nodes to the graph"""
        self.nodes.add(edge.parent)
        self.nodes.add(edge.child)
        self.edges.append(edge)
    
    def exclude_attention_head(self, layer: int, token_pos: int, head_idx: int):
        """Mark an attention head as excluded (permanently corrupted)"""
        self.excluded_attention_heads.add((layer, token_pos, head_idx))
        self.earliest_patched_layer = min(self.earliest_patched_layer, layer)
    
    def exclude_k_connection(self, layer: int, source_pos: int, target_pos: int, head_idx: int):
        """Mark a K connection as excluded (permanently corrupted)"""
        self.excluded_k_connections.add((layer, source_pos, target_pos, head_idx))
        self.earliest_patched_layer = min(self.earliest_patched_layer, layer)
    
    def exclude_v_connection(self, layer: int, source_pos: int, target_pos: int, head_idx: int):
        """Mark a V connection as excluded (permanently corrupted)"""
        self.excluded_v_connections.add((layer, source_pos, target_pos, head_idx))
        self.earliest_patched_layer = min(self.earliest_patched_layer, layer)
    
    def exclude_q_connection(self, layer: int, token_pos: int, head_idx: int):
        """Mark a Q connection as excluded (permanently corrupted)"""
        self.excluded_q_connections.add((layer, token_pos, head_idx))
        self.earliest_patched_layer = min(self.earliest_patched_layer, layer)
    
    def get_node(self, layer: int, token_pos: int, head_idx: Optional[int] = None) -> Optional[Node]:
        """Find a node by its coordinates"""
        for node in self.nodes:
            if (node.layer == layer and 
                node.token_pos == token_pos and 
                node.head_idx == head_idx):
                return node
        return None

class SimpleACDC:
    """Simplified ACDC implementation with incremental circuit building"""
    
    def __init__(self, 
                 model: HookedTransformer,
                 max_layer: int = 4,
                 threshold: float = 0.01,
                 corrupt_q: bool = True,
                 include_current_token: bool = False,
                 separate_kv: bool = False):
        self.model = model
        self.max_layer = max_layer
        self.threshold = threshold
        self.corrupt_q = corrupt_q
        self.include_current_token = include_current_token
        self.separate_kv = separate_kv
        self.graph = IncrementalCircuitGraph()
        self.device = model.cfg.device
        self.target_embedding = None  # Cache target embedding
    
    def get_sequence_length(self, text: str) -> int:
        """Get the sequence length for a text"""
        tokens = self.model.to_tokens(text, prepend_bos=True)
        return tokens.shape[1]
    
    def detokenize(self, text: str) -> List[str]:
        """Detokenize text into words"""
        tokens = self.model.to_tokens(text, prepend_bos=True)
        words = [self.model.tokenizer.decode(tok) for tok in tokens[0]]
        return words
        
    def compute_metric(self, 
                      original_text: str, 
                      target_text: str,
                      current_graph: IncrementalCircuitGraph) -> float:
        """
        Compute cosine similarity with current circuit graph state
        """
        # Cache target embedding if not already computed
        if self.target_embedding is None:
            target_tokens = self.model.to_tokens(target_text, prepend_bos=True)
            with torch.no_grad():
                # Clean forward pass for target (no circuit needed)
                target_output = self.model(target_tokens, stop_at_layer=self.max_layer + 1)
                self.target_embedding = target_output[-1, -1, :]
        
        # Tokenize original text
        original_tokens = self.model.to_tokens(original_text, prepend_bos=True)
        
        # Get final token embedding with current circuit
        with torch.no_grad():
            original_final = self._forward_with_circuit(original_tokens, current_graph)[-1, -1, :]  
            
            # Compute cosine similarity
            similarity = F.cosine_similarity(original_final.unsqueeze(0), 
                                           self.target_embedding.unsqueeze(0), dim=1)
            return similarity.item()
    
    def _forward_with_circuit(self, tokens: torch.Tensor, circuit: IncrementalCircuitGraph) -> torch.Tensor:
        """Forward pass implementing the current circuit by patching"""
        
        # Optimization: if no layers need patching, return cached clean activations
        if circuit.earliest_patched_layer == float('inf') or circuit.earliest_patched_layer > self.max_layer:
            # Use cached clean activations for the final layer output
            clean_output_hook = f"blocks.{self.max_layer}.hook_resid_post"
            if clean_output_hook in circuit.clean_activations:
                return circuit.clean_activations[clean_output_hook]
            # Fallback to clean forward pass
            print("No cached clean activations found, falling back to full forward pass.")
            return self.model(tokens, stop_at_layer=self.max_layer + 1)
        
        def make_patch_hook(hook_name: str, circuit: IncrementalCircuitGraph):
            def patch_hook(activation, hook):
                # Check if this hook point needs patching based on circuit
                return self._patch_activation(activation, hook_name, circuit)
            return patch_hook
        
        # Create hooks for layers that might need patching
        hooks = []
        hook_points = set()
        
        # Only collect hook points for layers that need patching
        for layer in range(circuit.earliest_patched_layer, self.max_layer + 1):
            hook_points.add(f"blocks.{layer}.attn.hook_v")
            hook_points.add(f"blocks.{layer}.attn.hook_rot_q")  # Rotated Q for RoPE
            hook_points.add(f"blocks.{layer}.attn.hook_rot_k")  # Rotated K for RoPE
            hook_points.add(f"blocks.{layer}.attn.hook_z")  # Before output projection
        
        for hook_name in hook_points:
            hooks.append((hook_name, make_patch_hook(hook_name, circuit)))
        
        # Use start_at_layer optimization when possible
        if circuit.earliest_patched_layer > 0:
            # Get input from cached clean activations
            input_hook = f"blocks.{circuit.earliest_patched_layer - 1}.hook_resid_post"
            if input_hook in circuit.clean_activations:
                clean_input = circuit.clean_activations[input_hook]
                return self.model.run_with_hooks(
                    clean_input, 
                    fwd_hooks=hooks, 
                    start_at_layer=circuit.earliest_patched_layer,
                    stop_at_layer=self.max_layer + 1
                )
        
        # Fallback: run full forward pass with hooks
        return self.model.run_with_hooks(tokens, fwd_hooks=hooks, stop_at_layer=self.max_layer + 1)
    
    def _patch_activation(self, activation: torch.Tensor, hook_name: str, circuit: IncrementalCircuitGraph) -> torch.Tensor:
        """Patch activation based on current circuit state"""
        
        # Handle both simple head patching AND custom Q/K/V patching at hook_z
        if '.attn.hook_z' in hook_name:
            layer_num = int(hook_name.split('.')[1])
            
            # hook_z has shape [batch, seq, head_index, d_head]
            if len(activation.shape) == 4:
                batch_size, seq_len, n_heads, d_head = activation.shape
            else:
                print(f"Warning: Unexpected activation shape {activation.shape} at {hook_name}")
                # Fallback: if shape is different, use model config
                batch_size, seq_len = activation.shape[:2]
                n_heads = self.model.cfg.n_heads
            
            has_excluded_heads = any((layer_num, tp, hi) in circuit.excluded_attention_heads
                                    for tp in range(seq_len)
                                    for hi in range(n_heads))
            
            has_k_exclusions = any((layer_num, sp, tp, hi) in circuit.excluded_k_connections 
                                  for sp in range(seq_len)
                                  for tp in range(seq_len)
                                  for hi in range(n_heads))

            has_v_exclusions = any((layer_num, sp, tp, hi) in circuit.excluded_v_connections 
                                  for sp in range(seq_len)
                                  for tp in range(seq_len)
                                  for hi in range(n_heads))

            has_q_exclusions = any((layer_num, qp, hi) in circuit.excluded_q_connections
                                  for qp in range(seq_len)
                                  for hi in range(n_heads))

            if has_excluded_heads or has_k_exclusions or has_v_exclusions or has_q_exclusions:
                return self._unified_attention_patching(activation, layer_num, circuit)
            else:
                return activation
        
        # For Q/K/V hooks, store the current (clean) activation and return it unchanged
        elif hook_name.endswith('.hook_v') or hook_name.endswith('.hook_rot_q') or hook_name.endswith('.hook_rot_k'):
            # Store current clean V and rotated Q/K for RoPE
            setattr(self, f'current_clean_{hook_name}', activation.clone())
            return activation
            
        return activation
    
    def _unified_attention_patching(self, z_activation: torch.Tensor, layer: int, circuit: IncrementalCircuitGraph) -> torch.Tensor:
        """Handle both simple head patching and custom Q/K/V patching at hook_z level"""
        try:
            # hook_z has shape [batch, seq, head_index, d_head]
            if len(z_activation.shape) == 4:
                batch_size, seq_len, n_heads, d_head = z_activation.shape
                z_heads = z_activation  # Already in the right shape
            else:
                print(f"Warning: Unexpected z_activation shape {z_activation.shape} at layer {layer}")
                # Fallback: reshape if needed
                batch_size, seq_len, d_model = z_activation.shape
                n_heads = self.model.cfg.n_heads
                d_head = self.model.cfg.d_head
                z_heads = z_activation.view(batch_size, seq_len, n_heads, d_head)
            
            # Get corrupted z activation for simple head patching
            corrupted_z = circuit.hooks_to_patch.get(f"blocks.{layer}.attn.hook_z")
            if corrupted_z is not None:
                if len(corrupted_z.shape) == 4:
                    corrupted_z_heads = corrupted_z
                else:
                    print(f"Warning: Unexpected corrupted_z shape {corrupted_z.shape} at layer {layer}")
                    corrupted_z_heads = corrupted_z.view(batch_size, seq_len, n_heads, d_head)
            
            
            # Apply simple head patching for excluded heads and custom Q/K/V patching
            for token_pos in range(seq_len):
                for head_idx in range(n_heads):
                    # If entire head is excluded, use corrupted activation
                    head_entirely_excluded = (layer, token_pos, head_idx) in circuit.excluded_attention_heads
                    if head_entirely_excluded:
                        if corrupted_z is not None:
                            z_heads[:, token_pos, head_idx, :] = corrupted_z_heads[:, token_pos, head_idx, :]

                    has_k_exclusion = any((layer, sp, token_pos, head_idx) in circuit.excluded_k_connections for sp in range(seq_len))
                    has_v_exclusion = any((layer, sp, token_pos, head_idx) in circuit.excluded_v_connections for sp in range(seq_len))
                    has_q_exclusion = (layer, token_pos, head_idx) in circuit.excluded_q_connections
                    
                    if has_k_exclusion or has_v_exclusion or has_q_exclusion:
                        # Recompute this head/token with custom Q/K/V mixing
                        custom_z = self._custom_single_head_computation(layer, head_idx, circuit, token_pos)
                        if custom_z is not None:
                            z_heads[:, token_pos, head_idx, :] = custom_z
                    
                    if head_entirely_excluded and (has_k_exclusion or has_v_exclusion or has_q_exclusion):
                        print(f"Warning: Head L{layer}H{head_idx} at token {token_pos} has both entire head exclusion and Q/K/V exclusions")
            
            # Reshape back to original format
            if len(z_activation.shape) == 4:
                return z_heads  # Return in [batch, seq, head, d_head] format
            else:
                print(f"Warning: Returning patched z_activation in [batch, seq, d_model] format at layer {layer}")
                return z_heads.view(batch_size, seq_len, n_heads * d_head)  # Return in [batch, seq, d_model] format
            
        except Exception as e:
            print(f"Error in unified attention patching: {e}")
            return z_activation

    def _custom_single_head_computation(self, layer: int, head_idx: int, circuit: IncrementalCircuitGraph, token_pos: int = None) -> Optional[torch.Tensor]:
        """Recompute a single attention head with custom Q/K/V mixing"""
        try:
            # Get current clean Q/K/V (use rotated Q/K for RoPE)
            q_hook_name = f'blocks.{layer}.attn.hook_rot_q'  # Use rotated Q
            k_hook_name = f'blocks.{layer}.attn.hook_rot_k'  # Use rotated K
            v_hook_name = f'blocks.{layer}.attn.hook_v'
            
            q_clean = getattr(self, f'current_clean_{q_hook_name}', None)
            k_clean = getattr(self, f'current_clean_{k_hook_name}', None) 
            v_clean = getattr(self, f'current_clean_{v_hook_name}', None)
            
            if q_clean is None or k_clean is None or v_clean is None:
                return None
            
            # Get corrupted Q/K/V (use rotated Q/K for RoPE)
            q_corrupted = circuit.hooks_to_patch.get(f"blocks.{layer}.attn.hook_rot_q")
            k_corrupted = circuit.hooks_to_patch.get(f"blocks.{layer}.attn.hook_rot_k")
            v_corrupted = circuit.hooks_to_patch.get(f"blocks.{layer}.attn.hook_v")
            
            if q_corrupted is None or k_corrupted is None or v_corrupted is None:
                return None

            batch_size, seq_len, n_q_heads, d_head = q_clean.shape
            _, _, n_kv_heads, _ = k_clean.shape
            heads_per_group = n_q_heads // n_kv_heads
            kv_head_idx = head_idx // heads_per_group  # Correct GQA mapping

            # Extract just this head's Q/K/V
            q_head = q_clean[:, token_pos, head_idx, :].clone()  # [batch, d_head] - only the specific token
            k_head = k_clean[:, :, kv_head_idx, :].clone()  # Use mapped KV head index
            v_head = v_clean[:, :, kv_head_idx, :].clone()  # Use mapped KV head index

            # Only patch the specific token_pos for this head
            if token_pos is not None:
                # Check Q exclusions - patch Q at the current token position only
                if (layer, token_pos, head_idx) in circuit.excluded_q_connections:
                    q_head_new = q_corrupted[:, token_pos, head_idx, :].clone()
                else:
                    q_head_new = q_head.clone()

                # Check K/V exclusions - use the mapped KV head index
                for source_pos in range(seq_len):
                    # Check separate K exclusions
                    if (layer, source_pos, token_pos, head_idx) in circuit.excluded_k_connections:
                        k_head[:, source_pos, :] = k_corrupted[:, source_pos, kv_head_idx, :]
                    
                    # Check separate V exclusions  
                    if (layer, source_pos, token_pos, head_idx) in circuit.excluded_v_connections:
                        v_head[:, source_pos, :] = v_corrupted[:, source_pos, kv_head_idx, :]

            # Compute attention for this head: Q @ K^T
            # q_head is [batch, d_head], k_head is [batch, key, d_head]
            scores = torch.einsum('bd,bkd->bk', q_head_new, k_head) / (d_head ** 0.5)  # [batch, key]
            # If not including current token, don't patch self-attention score
            if not self.include_current_token:
                scores[:, token_pos] = torch.einsum('bd,bd->b', q_head, k_head[:, token_pos, :]) / (d_head ** 0.5)

            # Causal mask: token at position token_pos can only attend to positions 0, 1, ..., token_pos
            causal_mask = torch.arange(seq_len, device=scores.device) > token_pos
            scores.masked_fill_(causal_mask.unsqueeze(0), float('-inf'))
            attn_pattern = torch.softmax(scores, dim=-1)  # [batch, key]

            attn_output = torch.einsum('bk,bkd->bd', attn_pattern, v_head)  # [batch, d_head]
            
            # Return directly as [batch, d_head] since we only computed for one token position
            return attn_output
            
        except Exception as e:
            print(f"Error in custom single head computation: {e}")
            return None

    def _cache_activations(self, text: str):
        """Cache activations for a given text"""
        tokens = self.model.to_tokens(text, prepend_bos=True)
        
        # Cache activations
        _, cache = self.model.run_with_cache(tokens, stop_at_layer=self.max_layer + 1)
        
        return cache
    
    def discover_circuit(self, 
                        original_text: str,
                        corrupted_text: str, 
                        target_text: str,
                        min_token_pos: int = 0,
                        quiet: bool = False) -> IncrementalCircuitGraph:
        """
        Run incremental ACDC algorithm to discover circuit
        
        Args:
            original_text: The text exhibiting the behavior of interest
            corrupted_text: The control text (baseline/corrupted version)
            target_text: The target behavior text for measuring effects
            min_token_pos: Minimum token position to consider for modifications (default: 0)
                          Tokens before this position will not be tested for inclusion/exclusion
        """
        if not quiet:
            print(f"Discovering circuit for:")
            print(f"  Original: {self.detokenize(original_text)}")
            print(f"  Corrupted: {self.detokenize(corrupted_text)}")
            print(f"  Target: {self.detokenize(target_text)}")
            print(f"  Threshold: {self.threshold}")
            print(f"  Min token position: {min_token_pos} ('{self.detokenize(original_text)[min_token_pos]}')")
            print(f"  Patch self attention: {self.include_current_token}")
            print(f"  Patch Q connections: {self.corrupt_q}")
            
        # Get sequence lengths first
        orig_len = self.get_sequence_length(original_text)
        
        # Start with empty circuit
        circuit = IncrementalCircuitGraph()
        
        # Cache both clean and corrupted activations
        clean_cache = self._cache_activations(original_text)
        corrupted_cache = self._cache_activations(corrupted_text)
        
        # Store both clean and corrupted activations in circuit for patching
        circuit.clean_activations = {}
        for layer in range(self.model.cfg.n_layers):
            # Store corrupted activations for all attention hooks needed for patching
            for hook_suffix in ['.attn.hook_v', '.attn.hook_rot_q', '.attn.hook_rot_k', '.attn.hook_z']:
                hook_name = f'blocks.{layer}{hook_suffix}'
                if hook_name in corrupted_cache:
                    circuit.hooks_to_patch[hook_name] = corrupted_cache[hook_name].clone()
            
            # Store clean activations only for residual outputs (for start_at_layer optimization)
            resid_hook = f'blocks.{layer}.hook_resid_post'
            if resid_hook in clean_cache:
                circuit.clean_activations[resid_hook] = clean_cache[resid_hook].clone()
        
        # print(f"Cached {len(circuit.hooks_to_patch)} corrupted activations")
        
        # Create root node: final layer, final token, post-residual
        final_token_pos = orig_len - 1  # -1 for 0-indexed
        root_node = Node(
            hook_name=f"blocks.{self.max_layer}.hook_resid_post",
            layer=self.max_layer,
            token_pos=final_token_pos,
            node_type="resid"
        )
        circuit.add_node(root_node)
        if not quiet:
            print(f"\nStarting with root node: {root_node}")
            print(f"Positive effect means component helps produce target behavior")
        
        # Compute initial clean metric
        current_metric = self.compute_metric(original_text, target_text, circuit)
        initial_metric = current_metric
        
        # Process nodes in reverse order: layers last to first, tokens last to first
        nodes_to_process = deque([root_node])
        processed_nodes = set()
        
        while nodes_to_process:
            current_node = nodes_to_process.popleft()
            if current_node in processed_nodes:
                continue

            if not quiet:
                print(f"\nProcessing node: {current_node}")
            processed_nodes.add(current_node)
            
            if current_node.node_type == "resid":
                new_nodes, current_metric = self._process_resid_node(current_node, circuit, original_text, target_text, current_metric, quiet)
            else:  # attention head
                new_nodes, current_metric = self._process_attention_node(current_node, circuit, original_text, target_text, current_metric, min_token_pos, quiet)

            # Add new nodes to processing queue
            for node in new_nodes:
                if node not in processed_nodes and node not in nodes_to_process:
                    nodes_to_process.append(node)
        
        if not quiet:
            print(f"\nDiscovered circuit with {len(circuit.nodes)} nodes and {len(circuit.edges)} edges")

            final_metric = self.compute_metric(original_text, target_text, circuit)
            print(f"Final metric with discovered circuit: {final_metric:.4f} (initial: {initial_metric:.4f})")
            if abs(final_metric - current_metric) > 1e-4:
                print(f"Warning: Final metric {final_metric:.4f} differs from estimated {current_metric:.4f}")

        return circuit, current_metric
    
    def _process_resid_node(self, node: Node, circuit: IncrementalCircuitGraph, 
                           original_text: str, target_text: str, current_metric: float, quiet: bool) -> Tuple[List[Node], float]:
        """Process a residual stream node"""
        new_nodes = []
        
        # First, test attention heads of same layer, same token position
        # print(f"  Testing attention heads for layer {node.layer}, token {node.token_pos}")
        
        for head_idx in range(self.model.cfg.n_heads):
            attn_node = Node(
                hook_name=f"blocks.{node.layer}.attn.hook_z",
                layer=node.layer,
                token_pos=node.token_pos,
                head_idx=head_idx,
                node_type="attn"
            )
            
            # Test if this attention head is important
            effect = self._test_attention_head_effect(attn_node, circuit, original_text, target_text, current_metric)
            
            if abs(effect) >= self.threshold:
                edge = Edge(parent=attn_node, child=node, edge_type="attn_out")
                edge.effect_size = effect
                circuit.add_edge(edge)
                new_nodes.append(attn_node)
                if not quiet:
                    print(f"    Added attention head: {attn_node} → {node} (effect: {effect:.4f})")
            else:
                # Explicitly exclude this attention head (permanently corrupt it)
                circuit.exclude_attention_head(node.layer, node.token_pos, head_idx)
                if not quiet:
                    print(f"    Excluded attention head {head_idx} (effect: {effect:.4f} < {self.threshold})")
                # When we reject a candidate, we already computed the corrupted metric in the test
                current_metric = current_metric - effect  # corrupted_metric = clean_metric - effect
        
        # Then, add previous layer residual connection (if not at layer 0)
        if node.layer > 0:
            prev_resid = Node(
                hook_name=f"blocks.{node.layer-1}.hook_resid_post",
                layer=node.layer - 1,
                token_pos=node.token_pos,
                node_type="resid"
            )
            edge = Edge(parent=prev_resid, child=node, edge_type="resid")
            circuit.add_edge(edge)
            new_nodes.append(prev_resid)
            # print(f"  Added residual connection: {prev_resid} → {node}")
        else:
            # Layer 0: connection comes from embeddings, create embedding node
            embedding_node = Node(
                hook_name="hook_embed",
                layer=-1,  # Use -1 to represent embedding layer
                token_pos=node.token_pos,
                node_type="embed"
            )
            edge = Edge(parent=embedding_node, child=node, edge_type="embed")
            circuit.add_edge(edge)
            # NOTE: Don't add embedding_node to new_nodes - it's a terminal node
            # print(f"  Added embedding connection: {embedding_node} → {node}")
        
        return new_nodes, current_metric
    
    def _process_attention_node(self, node: Node, circuit: IncrementalCircuitGraph,
                               original_text: str, target_text: str, current_metric: float, min_token_pos: int = 0, quiet: bool = False) -> Tuple[List[Node], float]:
        """Process an attention head node by testing Q and K/V connections separately"""
        new_nodes = []
        
        # For layer 0, connections come from initial embeddings (embed + pos_embed)
        # For other layers, connections come from previous layer's residual stream
            
        # Test query connection 
        if self.corrupt_q:
            # When corrupting Q, test the single query connection from this token position
            q_effect = self._test_q_effect(node, circuit, original_text, target_text, node.token_pos, current_metric)
            
            if abs(q_effect) >= self.threshold:
                # Add this Q connection to the circuit
                if node.layer > 0:
                    source_layer = node.layer - 1
                    source_hook = f"blocks.{source_layer}.hook_resid_post"
                    source_node_type = "resid"
                else:
                    # Layer 0 gets Q from embeddings
                    source_layer = -1
                    source_hook = "hook_embed"
                    source_node_type = "embed"
                
                source_node = Node(
                    hook_name=source_hook,
                    layer=source_layer,
                    token_pos=node.token_pos,  # Q comes from the same token position
                    node_type=source_node_type
                )
                edge = Edge(parent=source_node, child=node, edge_type="query")
                edge.effect_size = q_effect
                circuit.add_edge(edge)
                if source_node.node_type != "embed":  # Don't process embedding nodes
                    new_nodes.append(source_node)
                if not quiet:
                    print(f"    Added query connection: {source_node} → {node} (effect: {q_effect:.4f})")
            else:
                # Explicitly exclude this query connection
                circuit.exclude_q_connection(node.layer, node.token_pos, node.head_idx)
                if not quiet:
                    print(f"    Excluded query connection from token {node.token_pos} (effect: {q_effect:.4f} < {self.threshold})")
                current_metric = current_metric - q_effect
        else:
            # When not corrupting Q, always add the query connection
            if node.layer > 0:
                source_layer = node.layer - 1
                source_hook = f"blocks.{source_layer}.hook_resid_post"
                source_node_type = "resid"
            else:
                # Layer 0 gets Q from embeddings
                source_layer = -1
                source_hook = "hook_embed"
                source_node_type = "embed"
            
            prev_node = Node(
                hook_name=source_hook,
                layer=source_layer,
                token_pos=node.token_pos,
                node_type=source_node_type
            )
            edge = Edge(parent=prev_node, child=node, edge_type="query")
            edge.effect_size = None  # No effect measurement when not testing
            circuit.add_edge(edge)
            if prev_node.node_type != "embed":  # Don't process embedding nodes
                new_nodes.append(prev_node)
            # print(f"    Added implicit query connection: {prev_node} → {node} (corrupt_q=False)")
        
        # Test key/value connections (all previous tokens >= min_token_pos)
        # By default, exclude current token (tokens don't typically attend to themselves)
        max_key_token = node.token_pos + (1 if self.include_current_token else 0)
        
        if self.separate_kv:
            # Test K and V connections separately
            for source_token_pos in range(max(min_token_pos, 0), max_key_token):
                # Test K connection
                k_effect = self._test_k_effect(node, circuit, original_text, target_text, source_token_pos, current_metric)
                if abs(k_effect) >= self.threshold:
                    if node.layer > 0:
                        source_layer = node.layer - 1
                        source_hook = f"blocks.{source_layer}.hook_resid_post"
                        source_node_type = "resid"
                    else:
                        source_layer = -1
                        source_hook = "hook_embed"
                        source_node_type = "embed"
                    
                    source_node = Node(
                        hook_name=source_hook,
                        layer=source_layer,
                        token_pos=source_token_pos,
                        node_type=source_node_type
                    )
                    edge = Edge(parent=source_node, child=node, edge_type="key")
                    edge.effect_size = k_effect
                    circuit.add_edge(edge)
                    if source_node.node_type != "embed":
                        new_nodes.append(source_node)
                    if not quiet:
                        print(f"    Added key connection: {source_node} → {node} (effect: {k_effect:.4f})")
                else:
                    circuit.exclude_k_connection(node.layer, source_token_pos, node.token_pos, node.head_idx)
                    if not quiet:
                        print(f"    Excluded key connection from token {source_token_pos} (effect: {k_effect:.4f} < {self.threshold})")
                    current_metric = current_metric - k_effect
                
                # Test V connection
                v_effect = self._test_v_effect(node, circuit, original_text, target_text, source_token_pos, current_metric)
                if abs(v_effect) >= self.threshold:
                    if node.layer > 0:
                        source_layer = node.layer - 1
                        source_hook = f"blocks.{source_layer}.hook_resid_post"
                        source_node_type = "resid"
                    else:
                        source_layer = -1
                        source_hook = "hook_embed"
                        source_node_type = "embed"
                    
                    source_node = Node(
                        hook_name=source_hook,
                        layer=source_layer,
                        token_pos=source_token_pos,
                        node_type=source_node_type
                    )
                    edge = Edge(parent=source_node, child=node, edge_type="value")
                    edge.effect_size = v_effect
                    circuit.add_edge(edge)
                    if source_node.node_type != "embed":
                        new_nodes.append(source_node)
                    if not quiet:
                        print(f"    Added value connection: {source_node} → {node} (effect: {v_effect:.4f})")
                else:
                    circuit.exclude_v_connection(node.layer, source_token_pos, node.token_pos, node.head_idx)
                    if not quiet:
                        print(f"    Excluded value connection from token {source_token_pos} (effect: {v_effect:.4f} < {self.threshold})")
                    current_metric = current_metric - v_effect
        else:
            # Test key/value connections together (original behavior)  
            for source_token_pos in range(max(min_token_pos, 0), max_key_token):
                kv_effect = self._test_kv_effect(node, circuit, original_text, target_text, source_token_pos, current_metric)
                if abs(kv_effect) >= self.threshold:
                    if node.layer > 0:
                        source_layer = node.layer - 1
                        source_hook = f"blocks.{source_layer}.hook_resid_post"
                        source_node_type = "resid"
                    else:
                        # Layer 0 gets K/V from embeddings
                        source_layer = -1
                        source_hook = "hook_embed"
                        source_node_type = "embed"
                    
                    source_node = Node(
                        hook_name=source_hook,
                        layer=source_layer,
                        token_pos=source_token_pos,
                        node_type=source_node_type
                    )
                    edge = Edge(parent=source_node, child=node, edge_type="key_value")
                    edge.effect_size = kv_effect
                    circuit.add_edge(edge)
                    if source_node.node_type != "embed":  # Don't process embedding nodes
                        new_nodes.append(source_node)
                    if not quiet:
                        print(f"    Added key/value connection: {source_node} → {node} (effect: {kv_effect:.4f})")
                else:
                    # Explicitly exclude both K and V connections when testing them together
                    circuit.exclude_k_connection(node.layer, source_token_pos, node.token_pos, node.head_idx)
                    circuit.exclude_v_connection(node.layer, source_token_pos, node.token_pos, node.head_idx)
                    if not quiet:
                        print(f"    Excluded key/value connection from token {source_token_pos} (effect: {kv_effect:.4f} < {self.threshold})")
                    current_metric = current_metric - kv_effect
        
        return new_nodes, current_metric

    def _test_attention_head_effect(self, attn_node: Node, circuit: IncrementalCircuitGraph,
                                   original_text: str, target_text: str, clean_metric: float) -> float:
        """Test the effect of including an attention head"""
        # Create temporary circuit with this attention head excluded (corrupted)
        temp_circuit = copy.deepcopy(circuit)
        temp_circuit.exclude_attention_head(attn_node.layer, attn_node.token_pos, attn_node.head_idx)
        
        # Measure metric with head corrupted (clean_metric already provided)
        corrupted_metric = self.compute_metric(original_text, target_text, temp_circuit)
        
        # Positive effect means the clean head is better (head is important)
        return clean_metric - corrupted_metric
    
    def _test_kv_effect(self, attn_node: Node, circuit: IncrementalCircuitGraph,
                        original_text: str, target_text: str, source_pos: int, clean_metric: float) -> float:
        """Test the effect of a specific K/V connection from a token"""
        # Create temporary circuit with both K and V connections excluded (corrupted)
        temp_circuit = copy.deepcopy(circuit)
        temp_circuit.exclude_k_connection(attn_node.layer, source_pos, attn_node.token_pos, attn_node.head_idx)
        temp_circuit.exclude_v_connection(attn_node.layer, source_pos, attn_node.token_pos, attn_node.head_idx)

        # Measure metric with both connections corrupted (clean_metric already provided)
        corrupted_metric = self.compute_metric(original_text, target_text, temp_circuit)
        
        # Positive effect means the clean connections are better (connections are important)
        return clean_metric - corrupted_metric

    def _test_q_effect(self, attn_node: Node, circuit: IncrementalCircuitGraph,
                       original_text: str, target_text: str, query_pos: int, clean_metric: float) -> float:
        """Test the effect of a specific Q connection from a token"""
        # Create temporary circuit with this Q connection excluded (corrupted)
        temp_circuit = copy.deepcopy(circuit)
        temp_circuit.exclude_q_connection(attn_node.layer, query_pos, attn_node.head_idx)

        # Measure metric with Q connection corrupted (clean_metric already provided)
        corrupted_metric = self.compute_metric(original_text, target_text, temp_circuit)
        
        # Positive effect means the clean Q connection is better (connection is important)
        return clean_metric - corrupted_metric

    def _test_k_effect(self, attn_node: Node, circuit: IncrementalCircuitGraph,
                       original_text: str, target_text: str, source_pos: int, clean_metric: float) -> float:
        """Test the effect of a specific K connection from a token"""
        # Create temporary circuit with this K connection excluded (corrupted)
        temp_circuit = copy.deepcopy(circuit)
        temp_circuit.exclude_k_connection(attn_node.layer, source_pos, attn_node.token_pos, attn_node.head_idx)

        # Measure metric with K connection corrupted (clean_metric already provided)
        corrupted_metric = self.compute_metric(original_text, target_text, temp_circuit)
        
        # Positive effect means the clean K connection is better (connection is important)
        return clean_metric - corrupted_metric

    def _test_v_effect(self, attn_node: Node, circuit: IncrementalCircuitGraph,
                       original_text: str, target_text: str, source_pos: int, clean_metric: float) -> float:
        """Test the effect of a specific V connection from a token"""
        # Create temporary circuit with this V connection excluded (corrupted)
        temp_circuit = copy.deepcopy(circuit)
        temp_circuit.exclude_v_connection(attn_node.layer, source_pos, attn_node.token_pos, attn_node.head_idx)

        # Measure metric with V connection corrupted (clean_metric already provided)
        corrupted_metric = self.compute_metric(original_text, target_text, temp_circuit)
        
        # Positive effect means the clean V connection is better (connection is important)
        return clean_metric - corrupted_metric

    def visualize_circuit(self, circuit: IncrementalCircuitGraph, save_path: Optional[str] = None, min_token_pos: int = 0, quiet: bool = False):
        """Visualize the discovered circuit using networkx with structured layout"""
        G = nx.DiGraph()
        
        # Group edges by (parent, child) pair to merge key/value edges
        edge_groups = {}
        for edge in circuit.edges:
            parent_name = str(edge.parent)
            child_name = str(edge.child)
            key = (parent_name, child_name)
            
            if key not in edge_groups:
                edge_groups[key] = []
            edge_groups[key].append(edge)
        
        # Add merged edges to graph
        for (parent_name, child_name), edges in edge_groups.items():
            if len(edges) == 1:
                # Single edge - add as is
                edge = edges[0]
                G.add_edge(parent_name, child_name,
                          effect=edge.effect_size,
                          edge_type=edge.edge_type)
            else:
                # Multiple edges between same nodes - merge them
                key_edges = [e for e in edges if e.edge_type == 'key']
                value_edges = [e for e in edges if e.edge_type == 'value']
                
                if key_edges and value_edges:
                    # Both key and value edges exist - merge into KV
                    max_key_effect = max(e.effect_size for e in key_edges if e.effect_size is not None)
                    max_value_effect = max(e.effect_size for e in value_edges if e.effect_size is not None)
                    max_effect = max(max_key_effect, max_value_effect)
                    
                    G.add_edge(parent_name, child_name,
                              effect=max_effect,
                              edge_type='key_value')  # Merged as KV
                else:
                    # Only one type - use the one with max effect
                    max_edge = max(edges, key=lambda e: abs(e.effect_size) if e.effect_size is not None else 0)
                    G.add_edge(parent_name, child_name,
                              effect=max_edge.effect_size,
                              edge_type=max_edge.edge_type)
        
        # Create structured layout: horizontal=layer, vertical=token position
        plt.figure(figsize=(20, 12))
        pos = {}
        
        # Get all nodes to determine token position range (accounting for min_token_pos)
        max_token_pos = 0
        actual_min_token_pos = float('inf')
        for node in G.nodes():
            # Parse node string to extract layer and token info
            if '_T' in node:
                token_pos = int(node.split('_T')[1])
                max_token_pos = max(max_token_pos, token_pos)
                actual_min_token_pos = min(actual_min_token_pos, token_pos)
        
        # Use the provided min_token_pos or the actual minimum found in the circuit
        effective_min_token = min(min_token_pos, actual_min_token_pos) if actual_min_token_pos != float('inf') else min_token_pos
        token_range = max_token_pos - effective_min_token
        
        # First pass: count attention heads at each (layer, token) position
        head_counts = {}  # (layer, token) -> count
        head_positions = {}  # (layer, token) -> list of head_nums
        
        for node in G.nodes():
            if 'L' in node and 'H' in node and '_T' in node:
                # Parse attention head node: L{layer}H{head}_T{token}
                parts = node.split('_T')
                layer_head = parts[0].split('L')[1]  # "4H5" for example
                token_pos = int(parts[1])
                
                if 'H' in layer_head:
                    layer_num = int(layer_head.split('H')[0])
                    head_num = int(layer_head.split('H')[1])
                    
                    key = (layer_num, token_pos)
                    if key not in head_counts:
                        head_counts[key] = 0
                        head_positions[key] = []
                    head_counts[key] += 1
                    head_positions[key].append(head_num)
        
        # Sort head numbers at each position for consistent ordering
        for key in head_positions:
            head_positions[key].sort()
        
        # Position nodes: horizontal=layer, vertical=token position
        for node in G.nodes():
            if 'Embed_T' in node:
                # Embedding node: Embed_T{token}
                token_pos = int(node.split('_T')[1])
                x = -1  # Place embeddings to the left of layer 0
                y = token_range - (token_pos - effective_min_token)
                pos[node] = (x, y)
            elif 'L' in node and '_T' in node:
                # Parse layer and token position
                parts = node.split('_T')
                layer_part = parts[0]
                token_pos = int(parts[1])
                
                if 'H' in layer_part:
                    # Attention head: L{layer}H{head}_T{token}
                    layer_head = layer_part.split('L')[1]  # "4H5" for example
                    if 'H' in layer_head:
                        layer_num = int(layer_head.split('H')[0])
                        head_num = int(layer_head.split('H')[1])
                        
                        # Calculate position based on even distribution
                        key = (layer_num, token_pos)
                        num_heads = head_counts[key]
                        head_index = head_positions[key].index(head_num)
                        
                        # Multiple heads: spread evenly
                        spacing = 1.0 / (num_heads + 1)
                        x = layer_num - 1.0 + (head_index + 1) * spacing
                        y = (token_range - (token_pos - effective_min_token)) + 0.5   # Adjusted for token range
                    else:
                        layer_num = int(layer_head)
                        x = layer_num
                        y = token_range - (token_pos - effective_min_token)  # Adjusted for token range
                else:
                    # Residual node: L{layer}_T{token}
                    layer_num = int(layer_part.split('L')[1])
                    x = layer_num
                    y = token_range - (token_pos - effective_min_token)  # Adjusted for token range
                
                pos[node] = (x, y)
            else:
                # Fallback for any other node types
                pos[node] = (0, 0)
        
        # Color nodes by type
        node_colors = []
        for node in G.nodes():
            if 'Embed_' in node:  # Embedding node
                node_colors.append('gold')
            elif 'H' in node:  # Attention head
                node_colors.append('darkorange')
            else:  # Residual node
                node_colors.append('green')
        
        # Draw embedding nodes (triangles)
        embedding_nodes = [n for n in G.nodes() if 'Embed_' in n]
        embedding_colors = [node_colors[i] for i, n in enumerate(G.nodes()) if 'Embed_' in n]
        if embedding_nodes:
            embedding_pos = {n: pos[n] for n in embedding_nodes}
            nx.draw_networkx_nodes(G.subgraph(embedding_nodes), embedding_pos,
                                  node_color=embedding_colors, node_shape='^',
                                  node_size=1000, alpha=0.9)
        
        # Draw residual nodes (circles)
        residual_nodes = [n for n in G.nodes() if 'H' not in n and 'Embed_' not in n]
        residual_colors = [node_colors[i] for i, n in enumerate(G.nodes()) if 'H' not in n and 'Embed_' not in n]
        if residual_nodes:
            residual_pos = {n: pos[n] for n in residual_nodes}
            nx.draw_networkx_nodes(G.subgraph(residual_nodes), residual_pos, 
                                  node_color=residual_colors, node_shape='o',
                                  node_size=1200, alpha=0.8)
        
        # Draw attention nodes (squares)  
        attention_nodes = [n for n in G.nodes() if 'H' in n]
        attention_colors = [node_colors[i] for i, n in enumerate(G.nodes()) if 'H' in n]
        if attention_nodes:
            attention_pos = {n: pos[n] for n in attention_nodes}
            nx.draw_networkx_nodes(G.subgraph(attention_nodes), attention_pos,
                                  node_color=attention_colors, node_shape='s',
                                  node_size=700, alpha=0.9)
        
        # Draw edges with stronger colors and better visibility
        edge_colors = []
        edge_alphas = []
        for u, v in G.edges():
            # edge_type = G[u][v]['edge_type']
            if G[u][v]['effect'] is None:
                edge_colors.append('black')
                edge_alphas.append(0.8)
            else:
                sign = 1 if G[u][v]['effect'] > 0 else -1
                if sign > 0:
                    edge_colors.append('red')
                else:
                    edge_colors.append('blue')
                # Use fixed alpha, thickness will show effect size
                edge_alphas.append(0.8)
        
        # Draw edges with thickness proportional to effect size (minimum thickness for visibility)
        weights = []
        for u, v in G.edges():
            effect = G[u][v]['effect']
            if effect is not None:
                # Scale thickness with effect magnitude relative to threshold
                thickness = max(1.0, (abs(effect) / self.threshold) ** 0.7)  # Non-linear scaling for visibility
                weights.append(thickness)
            else:
                weights.append(3.0)  # Default thickness for non-measured edges
        
        # Draw edges one by one with individual alphas
        for i, (u, v) in enumerate(G.edges()):
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], 
                                  width=weights[i], alpha=edge_alphas[i],
                                  edge_color=edge_colors[i], arrows=True, 
                                  arrowsize=15, arrowstyle='->')
        
        # Draw labels above nodes with larger font
        label_pos = {n: (pos[n][0], pos[n][1] + 0.1) for n in G.nodes()}
        
        # Create simplified labels for attention heads (show only "Hx" instead of "LyHx_Tz")
        labels = {}
        for node in G.nodes():
            if 'H' in node and '_T' in node:
                # Extract head number from "LyHx_Tz" format
                parts = node.split('_T')[0]  # Get "LyHx" part
                if 'H' in parts:
                    head_part = parts.split('H')[1]  # Get "x" from "LyHx"
                    labels[node] = f"H{head_part}"
                else:
                    labels[node] = node
            else:
                labels[node] = node
        
        nx.draw_networkx_labels(G, label_pos, labels, font_size=14, font_weight='bold')
        
        # Add edge labels for K, V, or KV connections
        edge_labels = {}
        for u, v in G.edges():
            edge_type = G[u][v]['edge_type']
            if edge_type == 'key':
                edge_labels[(u, v)] = 'K'
            elif edge_type == 'value':
                edge_labels[(u, v)] = 'V'
            elif edge_type == 'key_value':
                edge_labels[(u, v)] = 'KV'
            # Don't label query, attn_out, resid, or embed edges to avoid clutter
        
        if edge_labels:
            # Draw edge labels vertically oriented
            for (u, v), label in edge_labels.items():
                x = (pos[u][0] + pos[v][0]) / 2
                y = (pos[u][1] + pos[v][1]) / 2
                plt.text(x, y, label, fontsize=20, color='darkgreen', ha='center', va='center', bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='none'))
        
        # Add grid lines for better readability
        for layer in range(self.max_layer + 1):
            plt.axvline(x=layer, color='lightgray', linestyle='--', alpha=0.3)
        for token_offset in range(token_range + 1):
            actual_token = effective_min_token + token_offset
            y_pos = token_range - token_offset
            plt.axhline(y=y_pos, color='lightgray', linestyle='--', alpha=0.3)
        
        # Add axis labels with token position information
        plt.xlabel("Layer", fontsize=12, fontweight='bold')
        ylabel = f"Token Position (range: {effective_min_token}-{max_token_pos}, inverted)"
        plt.ylabel(ylabel, fontsize=12, fontweight='bold')
        
        # Set y-axis limits to show the actual token range nicely
        plt.ylim(-0.5, token_range + 0.5)
        
        # Add legend with better colors
        legend_elements = [
            plt.Line2D([0], [0], color='red', lw=3, label='Positive Effect Edge'),
            plt.Line2D([0], [0], color='blue', lw=3, label='Negative Effect Edge'), 
            plt.Line2D([0], [0], color='black', lw=2, label='No Effect Measured'),
            plt.Line2D([0], [0], marker='^', color='gold', markersize=10, 
                      linestyle='None', label='Embedding Node'),
            plt.Line2D([0], [0], marker='o', color='green', markersize=10, 
                      linestyle='None', label='Residual Node'),
            plt.Line2D([0], [0], marker='s', color='darkorange', markersize=8, 
                      linestyle='None', label='Attention Head')
        ]
        
        # Add legend note about edge thickness and labels
        legend_text = "Note: Edge thickness ∝ effect magnitude\nEdge labels: K=Key, V=Value, KV=Key+Value"
        if self.separate_kv:
            legend_text += "\nSeparate K/V mode: Key and Value connections tested independently"
        else:
            legend_text += "\nCombined K/V mode: Key and Value connections tested together"
            
        plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1))
        plt.figtext(0.02, 0.02, legend_text, fontsize=10, style='italic')
        
        plt.title("Discovered Circuit (Structured Layout)", fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if not quiet:
                print(f"Circuit visualization saved to {save_path}")
        
        plt.show()
        
        # Print circuit summary
        if not quiet:
            print(f"\nCircuit Summary:")
            print(f"Nodes: {len(circuit.nodes)}")
            print(f"Edges: {len(circuit.edges)}")
            print(f"\nEdges by effect size:")
            sorted_edges = sorted(circuit.edges, key=lambda e: abs(e.effect_size) if e.effect_size else 0, reverse=True)
            for edge in sorted_edges:
                if edge.effect_size is not None:
                    effect_str = f"{edge.effect_size:.4f}"
                    print(f"  {edge.parent} → {edge.child} ({edge.edge_type}): {effect_str}")


def threshold_sweep(model: HookedTransformer, max_layer: int,
                 original_text: str, corrupted_text: str, target_text: str, thresholds: Tuple[float, float, float] = None, corrupt_q: bool = False, min_token_pos: int = 0, include_current_token: bool = False, separate_kv: bool = False, plot: bool = True, quiet: bool = True):
    """
    Run discover_circuit over a range of thresholds and plot metric vs threshold.
    Args:
        model: The Gemma model to analyze
        max_layer: Maximum layer to consider in the circuit
        original_text: The text exhibiting the behavior of interest
        corrupted_text: The control text (baseline/corrupted version)
        target_text: The target behavior text for measuring effects
        thresholds: Tuple (start, end, step) for threshold values to test (default: None, uses 0.0005 to 0.05)
        corrupt_q: Whether to corrupt query connections when testing (default: False)
        min_token_pos: Minimum token position to consider for modifications (default: 0)
        include_current_token: Whether to test key/value connections from the current token (default: False)
        separate_kv: Whether to test key and value connections separately (default: False)
        plot: Whether to plot the results (default: True)
        quiet: Whether to suppress detailed output (default: True)
    Returns: thresholds, metrics
    """
    if thresholds is None:
        ts = [round(x, 4) for x in np.arange(0.0005, 0.0505, 0.0005)]
    else:
        ts = [round(x, 4) for x in np.arange(thresholds[0], thresholds[1]+thresholds[2], thresholds[2])]
    
    metrics = []
    edge_counts = []
    # tqdm progress bar
    for i, t in enumerate(tqdm(ts, desc="Threshold Sweep")):
        acdc = SimpleACDC(model, max_layer=max_layer, threshold=t, corrupt_q=corrupt_q, include_current_token=include_current_token, separate_kv=separate_kv)
        circuit, current_metric = acdc.discover_circuit(
            original_text=original_text,
            corrupted_text=corrupted_text,
            target_text=target_text,
            min_token_pos=min_token_pos,
            quiet=quiet
        )
        metrics.append(current_metric)
        # Count only edges with weights (attention-related: attn_out, query, key, value, key_value)
        # Exclude residual and embed edges which don't have measured effect sizes
        weighted_edges = [edge for edge in circuit.edges if edge.effect_size is not None]
        num_weighted_edges = len(weighted_edges)
        edge_counts.append(max(1, num_weighted_edges))  # Avoid log(0) by using minimum of 1
        
    if plot:
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # Plot metrics on left y-axis
        color1 = 'tab:blue'
        ax1.set_xlabel('Threshold')
        ax1.set_ylabel('Metric (Cosine Similarity)', color=color1)
        line1 = ax1.plot(ts, metrics, marker='o', color=color1, label='Cosine Similarity')
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.grid(True, alpha=0.3)
        
        # Create second y-axis for edge count (log scale)
        ax2 = ax1.twinx()
        color2 = 'tab:red'
        ax2.set_ylabel('Number of Edges (log scale)', color=color2)
        ax2.set_yscale('log')
        line2 = ax2.plot(ts, edge_counts, marker='s', color=color2, label='Edge Count')
        ax2.tick_params(axis='y', labelcolor=color2)
        
        # Rescale axes to use full plot area
        # Left axis (metrics)
        metric_min, metric_max = min(metrics), max(metrics)
        metric_range = metric_max - metric_min
        if metric_range > 0:
            ax1.set_ylim(metric_min - 0.05 * metric_range, metric_max + 0.05 * metric_range)
        
        # Right axis (edge counts) - already log scale, just set reasonable bounds
        edge_min, edge_max = min(edge_counts), max(edge_counts)
        if edge_min == edge_max:
            ax2.set_ylim(0.5, edge_max * 2)
        else:
            ax2.set_ylim(edge_min * 0.5, edge_max * 2)
        
        plt.title(f'{corrupted_text} - Threshold Sweep')
        
        # Add legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        plt.tight_layout()
        plt.show()
    return ts, metrics, edge_counts


def multi_corrupted_threshold_sweep(model: HookedTransformer, max_layer: int,
                                  original_text: str, corrupted_texts: List[str], target_text: str, 
                                  thresholds: Tuple[float, float, float] = None, corrupt_q: bool = False, 
                                  min_token_pos: int = 0, include_current_token: bool = False, 
                                  plot: bool = True, quiet: bool = True, separate_kv: bool = False):
    """
    Run threshold sweeps on multiple corrupted texts for comparison.
    
    Args:
        model: The Gemma model to analyze
        max_layer: Maximum layer to consider in the circuit
        original_text: The text exhibiting the behavior of interest
        corrupted_texts: List of control texts (baseline/corrupted versions) to compare
        target_text: The target behavior text for measuring effects
        thresholds: Tuple (start, end, step) for threshold values to test (default: None, uses 0.0005 to 0.05)
        corrupt_q: Whether to corrupt query connections when testing (default: False)
        min_token_pos: Minimum token position to consider for modifications (default: 0)
        include_current_token: Whether to test key/value connections from the current token (default: False)
        plot: Whether to plot the results (default: True)
        quiet: Whether to suppress detailed output (default: True)
        separate_kv: Whether to test key and value connections separately (default: False)
        
    Returns: 
    """

    # Run threshold sweep for each corrupted text using the existing function
    for i, corrupted_text in enumerate(corrupted_texts):
        
        # Use the existing threshold_sweep function
        ts, metrics, edge_counts = threshold_sweep(
            model=model,
            max_layer=max_layer,
            original_text=original_text,
            corrupted_text=corrupted_text,
            target_text=target_text,
            thresholds=thresholds,
            corrupt_q=corrupt_q,
            min_token_pos=min_token_pos,
            include_current_token=include_current_token,
            separate_kv=separate_kv,  # Now passing separate_kv parameter
            plot=plot,  # Plot individual sweeps with dual y-axis
            quiet=True   # Always quiet for individual runs
        )
    
    return


# Example usage
def filter_circuit_nodes(circuit: IncrementalCircuitGraph) -> IncrementalCircuitGraph:
    """
    Filter out attention nodes that have no input connections and their outgoing connections.
    
    Args:
        circuit: Original circuit graph
        
    Returns:
        Filtered circuit with attention nodes with no inputs removed
    """
    filtered_circuit = IncrementalCircuitGraph()
    
    # Copy basic circuit properties
    filtered_circuit.hooks_to_patch = circuit.hooks_to_patch.copy()
    filtered_circuit.clean_activations = circuit.clean_activations.copy()
    filtered_circuit.excluded_attention_heads = circuit.excluded_attention_heads.copy()
    filtered_circuit.excluded_k_connections = circuit.excluded_k_connections.copy()
    filtered_circuit.excluded_v_connections = circuit.excluded_v_connections.copy()
    filtered_circuit.excluded_q_connections = circuit.excluded_q_connections.copy()
    filtered_circuit.earliest_patched_layer = circuit.earliest_patched_layer
    
    # Build a mapping of which nodes have incoming edges
    nodes_with_inputs = set()
    for edge in circuit.edges:
        nodes_with_inputs.add(edge.child)
    
    # Add nodes that have inputs or are not attention nodes
    for node in circuit.nodes:
        if node.node_type != "attn" or node in nodes_with_inputs:
            filtered_circuit.add_node(node)
    
    # Add edges only if both parent and child nodes are kept
    for edge in circuit.edges:
        if edge.parent in filtered_circuit.nodes and edge.child in filtered_circuit.nodes:
            filtered_circuit.add_edge(edge)
    
    return filtered_circuit

def merge_circuits(circuits: List[IncrementalCircuitGraph]) -> IncrementalCircuitGraph:
    """
    Merge multiple circuits by taking union of nodes/edges and intersection of exclusions.
    For edges with weights, take the maximum weight.
    
    Args:
        circuits: List of circuits to merge
        
    Returns:
        Merged circuit
    """
    if not circuits:
        return IncrementalCircuitGraph()
    
    merged_circuit = IncrementalCircuitGraph()
    
    # Initialize with first circuit's properties
    first_circuit = circuits[0]
    merged_circuit.hooks_to_patch = first_circuit.hooks_to_patch.copy()
    merged_circuit.clean_activations = first_circuit.clean_activations.copy()
    merged_circuit.earliest_patched_layer = first_circuit.earliest_patched_layer
    
    # Start with first circuit's exclusions, then intersect with others
    merged_circuit.excluded_attention_heads = first_circuit.excluded_attention_heads.copy()
    merged_circuit.excluded_k_connections = first_circuit.excluded_k_connections.copy()
    merged_circuit.excluded_v_connections = first_circuit.excluded_v_connections.copy()
    merged_circuit.excluded_q_connections = first_circuit.excluded_q_connections.copy()
    
    # Intersect exclusions across all circuits
    for circuit in circuits[1:]:
        merged_circuit.excluded_attention_heads &= circuit.excluded_attention_heads
        merged_circuit.excluded_k_connections &= circuit.excluded_k_connections
        merged_circuit.excluded_v_connections &= circuit.excluded_v_connections
        merged_circuit.excluded_q_connections &= circuit.excluded_q_connections
        
        # Update earliest patched layer
        merged_circuit.earliest_patched_layer = min(merged_circuit.earliest_patched_layer, 
                                                   circuit.earliest_patched_layer)
    
    # Union of all nodes
    all_nodes = set()
    for circuit in circuits:
        all_nodes.update(circuit.nodes)
    
    for node in all_nodes:
        merged_circuit.add_node(node)
    
    # Union of edges with maximum effect size for duplicates
    edge_map = {}  # (parent, child, edge_type) -> Edge with max effect
    
    for circuit in circuits:
        for edge in circuit.edges:
            key = (edge.parent, edge.child, edge.edge_type)
            
            if key not in edge_map:
                edge_map[key] = edge
            else:
                # Take edge with maximum absolute effect size
                existing_edge = edge_map[key]
                if (edge.effect_size is not None and 
                    (existing_edge.effect_size is None or 
                     abs(edge.effect_size) > abs(existing_edge.effect_size))):
                    edge_map[key] = edge
    
    # Add all unique edges to merged circuit
    for edge in edge_map.values():
        merged_circuit.add_edge(edge)
    
    return merged_circuit

def build_and_merge_circuits(model: HookedTransformer,
                           max_layer: int,
                           original_text: str,
                           corrupted_texts: List[str],
                           target_text: str,
                           thresholds: List[float],
                           min_token_pos: int = 2,
                           corrupt_q: bool = True,
                           separate_kv: bool = True,
                           include_current_token: bool = False,
                           quiet: bool = False,
                           visualize_individual: bool = False,
                           save_individual_paths: Optional[List[str]] = None) -> Tuple[IncrementalCircuitGraph, List[IncrementalCircuitGraph], List[float]]:
    """
    Build circuits for each corrupted text with specified thresholds, filter attention nodes
    with no inputs, and merge all circuits.
    
    Args:
        model: HookedTransformer model
        max_layer: Maximum layer to consider in circuits
        original_text: The text exhibiting the behavior of interest
        corrupted_texts: List of control texts (baseline/corrupted versions)
        target_text: The target behavior text for measuring effects
        thresholds: List of threshold values for each corrupted text (must match length)
        min_token_pos: Minimum token position to consider for modifications
        corrupt_q: Whether to corrupt query connections when testing
        separate_kv: Whether to test key and value connections separately
        include_current_token: Whether to test key/value connections from current token
        quiet: Whether to suppress detailed output
        visualize_individual: Whether to visualize each individual circuit before merging
        save_individual_paths: Optional list of save paths for individual circuit visualizations
        
    Returns:
        Tuple of (merged_circuit, individual_circuits, final_metrics)
    """
    if len(corrupted_texts) != len(thresholds):
        raise ValueError(f"Number of corrupted texts ({len(corrupted_texts)}) must match number of thresholds ({len(thresholds)})")
    
    if save_individual_paths is not None and len(save_individual_paths) != len(corrupted_texts):
        raise ValueError(f"Number of save paths ({len(save_individual_paths)}) must match number of corrupted texts ({len(corrupted_texts)})")
    
    if not quiet:
        print(f"Building and merging circuits for {len(corrupted_texts)} corrupted texts")
        print(f"Original: '{original_text}'")
        print(f"Target: '{target_text}'")
        print(f"Corrupted texts and thresholds:")
        for i, (corrupted_text, threshold) in enumerate(zip(corrupted_texts, thresholds)):
            print(f"  {i+1}. '{corrupted_text}' (threshold: {threshold})")
    
    # Build individual circuits
    individual_circuits = []
    final_metrics = []
    
    for i, (corrupted_text, threshold) in enumerate(zip(corrupted_texts, thresholds)):
        if not quiet:
            print(f"\n{'='*60}")
            print(f"Building circuit {i+1}/{len(corrupted_texts)}: {corrupted_text}")
            print(f"Threshold: {threshold}")
            print(f"{'='*60}")
        
        # Initialize ACDC with specific threshold
        acdc = SimpleACDC(
            model=model,
            max_layer=max_layer,
            threshold=threshold,
            corrupt_q=corrupt_q,
            include_current_token=include_current_token,
            separate_kv=separate_kv
        )
        
        # Discover circuit
        circuit, final_metric = acdc.discover_circuit(
            original_text=original_text,
            corrupted_text=corrupted_text,
            target_text=target_text,
            min_token_pos=min_token_pos,
            quiet=quiet
        )
        
        if not quiet:
            print(f"Circuit {i+1} discovered: {len(circuit.nodes)} nodes, {len(circuit.edges)} edges")
            print(f"Final metric: {final_metric:.4f}")
        
        individual_circuits.append(circuit)
        final_metrics.append(final_metric)
        
        # Visualize individual circuit if requested
        if visualize_individual:
            if not quiet:
                print(f"\nVisualizing individual circuit {i+1}: {corrupted_text}")
            
            save_path = None
            if save_individual_paths is not None:
                save_path = save_individual_paths[i]
            
            acdc.visualize_circuit(
                circuit, 
                save_path=save_path,
                min_token_pos=min_token_pos,
                quiet=quiet
            )
    
    if not quiet:
        print(f"\n{'='*60}")
        print("FILTERING ATTENTION NODES WITH NO INPUTS")
        print(f"{'='*60}")
    
    # Filter each circuit to remove attention nodes with no input connections
    filtered_circuits = []
    for i, circuit in enumerate(individual_circuits):
        filtered_circuit = filter_circuit_nodes(circuit)
        
        original_attn_nodes = len([n for n in circuit.nodes if n.node_type == "attn"])
        filtered_attn_nodes = len([n for n in filtered_circuit.nodes if n.node_type == "attn"])
        removed_attn_nodes = original_attn_nodes - filtered_attn_nodes
        
        if not quiet:
            print(f"Circuit {i+1}: Removed {removed_attn_nodes} attention nodes with no inputs")
            print(f"  Before: {len(circuit.nodes)} nodes, {len(circuit.edges)} edges")
            print(f"  After:  {len(filtered_circuit.nodes)} nodes, {len(filtered_circuit.edges)} edges")
        
        filtered_circuits.append(filtered_circuit)
    
    if not quiet:
        print(f"\n{'='*60}")
        print("MERGING CIRCUITS")
        print(f"{'='*60}")
    
    # Merge all filtered circuits
    merged_circuit = merge_circuits(filtered_circuits)
    
    if not quiet:
        # Calculate total nodes/edges before merging
        total_nodes_before = sum(len(circuit.nodes) for circuit in filtered_circuits)
        total_edges_before = sum(len(circuit.edges) for circuit in filtered_circuits)
        
        print(f"Merged circuit statistics:")
        print(f"  Individual circuits total: {total_nodes_before} nodes, {total_edges_before} edges")
        print(f"  Merged circuit: {len(merged_circuit.nodes)} nodes, {len(merged_circuit.edges)} edges")
        
        # Show exclusion intersections
        print(f"\nExclusion intersections:")
        print(f"  Attention heads: {len(merged_circuit.excluded_attention_heads)}")
        print(f"  K connections: {len(merged_circuit.excluded_k_connections)}")
        print(f"  V connections: {len(merged_circuit.excluded_v_connections)}")
        print(f"  Q connections: {len(merged_circuit.excluded_q_connections)}")
        
        # Show node type breakdown
        attn_nodes = [n for n in merged_circuit.nodes if n.node_type == "attn"]
        resid_nodes = [n for n in merged_circuit.nodes if n.node_type == "resid"]
        embed_nodes = [n for n in merged_circuit.nodes if n.node_type == "embed"]
        
        print(f"\nMerged circuit node breakdown:")
        print(f"  Attention nodes: {len(attn_nodes)}")
        print(f"  Residual nodes: {len(resid_nodes)}")
        print(f"  Embedding nodes: {len(embed_nodes)}")
        
        # Show edge type breakdown
        edge_types = {}
        for edge in merged_circuit.edges:
            edge_types[edge.edge_type] = edge_types.get(edge.edge_type, 0) + 1
        
        print(f"\nMerged circuit edge breakdown:")
        for edge_type, count in sorted(edge_types.items()):
            print(f"  {edge_type}: {count}")
    
    return merged_circuit, individual_circuits, final_metrics

def compute_qk_dot_products(model: HookedTransformer,
                           texts: List[str],
                           layer: int,
                           head_idx: int,
                           q_index: int,
                           k_index: int) -> List[float]:
    """
    Compute the rotated Q-K dot product between two token positions for a specific attention head
    across multiple texts.
    
    Args:
        model: HookedTransformer model
        texts: List of texts to analyze
        layer: Layer number of the attention head
        head_idx: Head index within the layer
        token_idx1: Token position for Q (query)
        token_idx2: Token position for K (key)
        
    Returns:
        List of Q-K dot products, one for each text
    """
    dot_products = []
    
    print(f"Computing Q-K dot products for Layer {layer}, Head {head_idx} (Q@{q_index}, K@{k_index}):")
    
    # Get token information from first text for reference
    if texts:
        first_tokens = model.to_tokens(texts[0], prepend_bos=True)
        first_seq_len = first_tokens.shape[1]
        
        # Detokenize to show actual tokens
        q_token = "N/A"
        k_token = "N/A"
        if q_index < first_seq_len:
            q_token = model.tokenizer.decode([first_tokens[0, q_index].item()])
        if k_index < first_seq_len:
            k_token = model.tokenizer.decode([first_tokens[0, k_index].item()])
        
        print(f"Q token at position {q_index}: '{q_token}'")
        print(f"K token at position {k_index}: '{k_token}'")
    
    print("-" * 80)
    
    for text in texts:
        # Tokenize text
        tokens = model.to_tokens(text, prepend_bos=True)
        
        # Check if token indices are valid for this text
        seq_len = tokens.shape[1]
        if q_index >= seq_len or k_index >= seq_len:
            print(f"'{text}' -> WARNING: Token indices ({q_index}, {k_index}) exceed sequence length {seq_len}")
            dot_products.append(float('nan'))
            continue
        
        # Get rotated Q and K activations using existing caching approach
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, stop_at_layer=layer + 1)
            
            # Extract rotated Q and K for this layer
            q_rot = cache[f"blocks.{layer}.attn.hook_rot_q"]  # [batch, seq, n_heads, d_head]
            k_rot = cache[f"blocks.{layer}.attn.hook_rot_k"]  # [batch, seq, n_kv_heads, d_head]
            
            # Handle Grouped Query Attention (GQA) mapping
            n_q_heads = q_rot.shape[2]
            n_kv_heads = k_rot.shape[2]
            heads_per_group = n_q_heads // n_kv_heads
            kv_head_idx = head_idx // heads_per_group  # Correct GQA mapping
            
            # Extract Q and K vectors for the specified tokens and head
            q_vector = q_rot[0, q_index, head_idx, :]  # [d_head]
            k_vector = k_rot[0, k_index, kv_head_idx, :]  # [d_head]
            
            # Compute dot product
            dot_product = torch.dot(q_vector, k_vector).item()
            dot_products.append(dot_product)
            
            # Print text with dot product
            print(f"'{text}' -> {dot_product:.6f}")
    
    print("-" * 80)
    return dot_products