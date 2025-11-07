#!/usr/bin/env python3
"""
Remap custom DeepSeek PyTorch weights to MLX naming convention
WITH unified intermediate size (pad routed experts to match shared expert)

Usage:
    python remap_weights_for_mlx_unified.py [path/to/weights.pt]

If no path provided, uses: model_weights/best_deepseek_v3.pt
"""

import torch
import numpy as np
from safetensors.numpy import save_file
from pathlib import Path
import sys

def pad_weight(weight_np, target_shape):
    """Pad weight to target shape (zero-padding for new dimensions)"""
    if weight_np.shape == target_shape:
        return weight_np

    # For gate_proj and up_proj: [out_dim, in_dim] -> pad out_dim
    if len(weight_np.shape) == 2:
        current_shape = weight_np.shape
        pad_out = target_shape[0] - current_shape[0]
        pad_in = target_shape[1] - current_shape[1]

        if pad_out > 0 or pad_in > 0:
            padded = np.pad(weight_np, ((0, pad_out), (0, pad_in)), mode='constant')
            return padded

    # For biases: [out_dim] -> pad out_dim
    elif len(weight_np.shape) == 1:
        pad_amt = target_shape[0] - weight_np.shape[0]
        if pad_amt > 0:
            return np.pad(weight_np, (0, pad_amt), mode='constant')

    return weight_np

def remap_weights(input_path, output_dir="POC/mlx_model"):
    """Remap PyTorch weights to MLX format with unified intermediate size"""

    print("=" * 70)
    print("Remapping DeepSeek Weights for MLX (Unified Intermediate Size)")
    print("=" * 70)

    # Load original weights
    print(f"\n[1/5] Loading PyTorch weights from: {input_path}")
    if not Path(input_path).exists():
        raise FileNotFoundError(f"Weight file not found: {input_path}")

    state_dict = torch.load(input_path, map_location="cpu")
    print(f"  ✓ Loaded {len(state_dict)} weight tensors")

    # Auto-detect number of layers
    layer_keys = [k for k in state_dict.keys() if k.startswith('h.')]
    num_layers = max([int(k.split('.')[1]) for k in layer_keys]) + 1 if layer_keys else 0
    print(f"  ✓ Detected {num_layers} transformer layers")

    # Auto-detect number of experts
    expert_keys = [k for k in state_dict.keys() if 'mlp.experts.' in k and len(k.split('.')) > 4 and k.split('.')[4].isdigit()]
    num_experts = max([int(k.split('.')[4]) for k in expert_keys]) + 1 if expert_keys else 0
    print(f"  ✓ Detected {num_experts} experts per layer")

    # Detect intermediate sizes
    print(f"\n[2/5] Detecting intermediate sizes...")
    routed_expert_key = "h.0.mlp.experts.0.gate_proj.weight"
    shared_expert_key = "h.0.mlp.shared_expert.gate_proj.weight"

    routed_size = state_dict[routed_expert_key].shape[0] if routed_expert_key in state_dict else 512
    shared_size = state_dict[shared_expert_key].shape[0] if shared_expert_key in state_dict else 768

    print(f"  Routed expert intermediate size: {routed_size}")
    print(f"  Shared expert intermediate size: {shared_size}")

    # Use the LARGER size as unified size
    unified_size = max(routed_size, shared_size)
    print(f"  ✓ Using unified intermediate size: {unified_size}")

    if routed_size != unified_size:
        print(f"  ⚠ Will pad routed experts from {routed_size} to {unified_size}")

    # Create mapping from our names to MLX names
    print(f"\n[3/5] Remapping weight names...")
    mlx_weights = {}
    mapped_count = 0

    # Embedding layer: wte -> model.embed_tokens
    if "wte.weight" in state_dict:
        mlx_weights["model.embed_tokens.weight"] = state_dict["wte.weight"].numpy()
        mapped_count += 1
        print(f"  ✓ Mapped embedding layer")

    # Transformer layers
    for i in range(num_layers):
        prefix_old = f"h.{i}"
        prefix_new = f"model.layers.{i}"

        # Layer norms
        if f"{prefix_old}.ln_1.weight" in state_dict:
            mlx_weights[f"{prefix_new}.input_layernorm.weight"] = state_dict[f"{prefix_old}.ln_1.weight"].numpy()
            mapped_count += 1

        if f"{prefix_old}.ln_2.weight" in state_dict:
            mlx_weights[f"{prefix_new}.post_attention_layernorm.weight"] = state_dict[f"{prefix_old}.ln_2.weight"].numpy()
            mapped_count += 1

        # Attention weights (with special handling for kv_a_proj_with_mqa)

        # Simple mappings
        simple_attn_mappings = {
            f"{prefix_old}.attn.q_proj.weight": f"{prefix_new}.self_attn.q_a_proj.weight",
            f"{prefix_old}.attn.q_decompress.weight": f"{prefix_new}.self_attn.q_b_proj.weight",
            f"{prefix_old}.attn.kv_norm.weight": f"{prefix_new}.self_attn.kv_a_layernorm.weight",
        }

        for old_key, new_key in simple_attn_mappings.items():
            if old_key in state_dict:
                mlx_weights[new_key] = state_dict[old_key].numpy()
                mapped_count += 1

        # q_a_layernorm: PyTorch model doesn't have this, create identity (all ones)
        # MLX expects RMSNorm with dimensions=q_lora_rank (192)
        q_lora_rank = 192
        mlx_weights[f"{prefix_new}.self_attn.q_a_layernorm.weight"] = np.ones(q_lora_rank, dtype=np.float32)
        mapped_count += 1

        # kv_a_proj_with_mqa: Concatenate kv_proj + first head of k_rope_proj
        # PyTorch: kv_proj [128, 512], k_rope_proj [256, 512] (8 heads × 32)
        # MLX: kv_a_proj_with_mqa [160, 512] = [kv_lora_rank + qk_rope_head_dim, hidden]
        kv_proj_key = f"{prefix_old}.attn.kv_proj.weight"
        k_rope_proj_key = f"{prefix_old}.attn.k_rope_proj.weight"

        if kv_proj_key in state_dict and k_rope_proj_key in state_dict:
            kv_proj = state_dict[kv_proj_key].numpy()  # [128, 512]
            k_rope_proj = state_dict[k_rope_proj_key].numpy()  # [256, 512]

            # Extract first 32 dimensions (one head worth of rope)
            k_rope_head = k_rope_proj[:32, :]  # [32, 512]

            # Concatenate: [128, 512] + [32, 512] = [160, 512]
            kv_a_proj_combined = np.concatenate([kv_proj, k_rope_head], axis=0)
            mlx_weights[f"{prefix_new}.self_attn.kv_a_proj_with_mqa.weight"] = kv_a_proj_combined
            mapped_count += 1

        # kv_b_proj: Need to handle k_decompress and v_decompress separately
        # PyTorch: k_decompress [512, 128], v_decompress [512, 128]
        # MLX: kv_b_proj [num_heads * (qk_nope_head_dim + v_head_dim), kv_lora_rank]
        # = [8 * (32 + 64), 128] = [768, 128]
        k_decompress_key = f"{prefix_old}.attn.k_decompress.weight"
        v_decompress_key = f"{prefix_old}.attn.v_decompress.weight"

        if k_decompress_key in state_dict and v_decompress_key in state_dict:
            k_decompress = state_dict[k_decompress_key].numpy()  # [512, 128]
            v_decompress = state_dict[v_decompress_key].numpy()  # [512, 128]

            # PyTorch outputs: k_content (512 = 8*64), v (512 = 8*64)
            # MLX expects: interleaved [k_nope, v] per head
            # k_nope = first 32 dims of k per head, v = 64 dims per head

            num_heads = 8
            head_dim = 64
            nope_dim = 32

            # Reshape to per-head: [num_heads, head_dim, kv_lora_rank]
            k_per_head = k_decompress.reshape(num_heads, head_dim, -1)  # [8, 64, 128]
            v_per_head = v_decompress.reshape(num_heads, head_dim, -1)  # [8, 64, 128]

            # Extract k_nope (first 32 dims) and concatenate with v
            k_nope = k_per_head[:, :nope_dim, :]  # [8, 32, 128]
            combined = np.concatenate([k_nope, v_per_head], axis=1)  # [8, 96, 128]

            # Flatten back: [768, 128]
            kv_b_combined = combined.reshape(-1, 128)
            mlx_weights[f"{prefix_new}.self_attn.kv_b_proj.weight"] = kv_b_combined
            mapped_count += 1

        # Output projection
        o_proj_key = f"{prefix_old}.attn.o_proj.weight"
        if o_proj_key in state_dict:
            mlx_weights[f"{prefix_new}.self_attn.o_proj.weight"] = state_dict[o_proj_key].numpy()
            mapped_count += 1

        # Router (gate)
        if f"{prefix_old}.mlp.router.weight" in state_dict:
            mlx_weights[f"{prefix_new}.mlp.gate.weight"] = state_dict[f"{prefix_old}.mlp.router.weight"].numpy()
            mapped_count += 1

        # e_score_correction_bias: PyTorch model doesn't have this, create zeros
        # MLX uses this for load balancing in MoE routing
        # Shape: [n_routed_experts] = [8]
        mlx_weights[f"{prefix_new}.mlp.gate.e_score_correction_bias"] = np.zeros(num_experts, dtype=np.float32)
        mapped_count += 1

        # Experts: Stack all experts into switch_mlp format
        # MLX uses SwitchGLU which expects stacked expert weights
        # Shape: [num_experts, intermediate, hidden] for gate/up_proj
        #        [num_experts, hidden, intermediate] for down_proj
        hidden_size = state_dict["wte.weight"].shape[1]

        gate_proj_experts = []
        up_proj_experts = []
        down_proj_experts = []

        for j in range(num_experts):
            # Collect and pad each expert's weights
            gate_key = f"{prefix_old}.mlp.experts.{j}.gate_proj.weight"
            up_key = f"{prefix_old}.mlp.experts.{j}.up_proj.weight"
            down_key = f"{prefix_old}.mlp.experts.{j}.down_proj.weight"

            if gate_key in state_dict:
                gate_weight = state_dict[gate_key].numpy()
                gate_padded = pad_weight(gate_weight, (unified_size, hidden_size))
                gate_proj_experts.append(gate_padded)

            if up_key in state_dict:
                up_weight = state_dict[up_key].numpy()
                up_padded = pad_weight(up_weight, (unified_size, hidden_size))
                up_proj_experts.append(up_padded)

            if down_key in state_dict:
                down_weight = state_dict[down_key].numpy()
                down_padded = pad_weight(down_weight, (hidden_size, unified_size))
                down_proj_experts.append(down_padded)

        # Stack all experts: [num_experts, ...]
        if gate_proj_experts:
            mlx_weights[f"{prefix_new}.mlp.switch_mlp.gate_proj.weight"] = np.stack(gate_proj_experts)
            mapped_count += 1

        if up_proj_experts:
            mlx_weights[f"{prefix_new}.mlp.switch_mlp.up_proj.weight"] = np.stack(up_proj_experts)
            mapped_count += 1

        if down_proj_experts:
            mlx_weights[f"{prefix_new}.mlp.switch_mlp.down_proj.weight"] = np.stack(down_proj_experts)
            mapped_count += 1

        # Shared expert (should already be unified_size, but verify)
        if f"{prefix_old}.mlp.shared_expert.gate_proj.weight" in state_dict:
            mlx_weights[f"{prefix_new}.mlp.shared_experts.gate_proj.weight"] = state_dict[f"{prefix_old}.mlp.shared_expert.gate_proj.weight"].numpy()
            mlx_weights[f"{prefix_new}.mlp.shared_experts.up_proj.weight"] = state_dict[f"{prefix_old}.mlp.shared_expert.up_proj.weight"].numpy()
            mlx_weights[f"{prefix_new}.mlp.shared_experts.down_proj.weight"] = state_dict[f"{prefix_old}.mlp.shared_expert.down_proj.weight"].numpy()

            # Skip shared expert biases - MLX doesn't use bias in MLP layers

            mapped_count += 3

    print(f"  ✓ Mapped {num_layers} transformer layers")

    # Final layer norm
    if "ln_f.weight" in state_dict:
        mlx_weights["model.norm.weight"] = state_dict["ln_f.weight"].numpy()
        mapped_count += 1
        print(f"  ✓ Mapped final layer norm")

    # LM head
    if "lm_head.weight" in state_dict:
        mlx_weights["lm_head.weight"] = state_dict["lm_head.weight"].numpy()
        mapped_count += 1
        print(f"  ✓ Mapped LM head")

    print(f"\n[4/5] Mapping summary:")
    print(f"  Original weights: {len(state_dict)}")
    print(f"  Mapped weights: {len(mlx_weights)}")
    print(f"  Mapping rate: {len(mlx_weights)/len(state_dict)*100:.1f}%")

    # Save
    print(f"\n[5/5] Saving remapped weights...")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    weights_file = output_path / "weights.safetensors"
    save_file(mlx_weights, str(weights_file))

    print(f"  ✓ Saved to: {weights_file}")
    print(f"    Size: {weights_file.stat().st_size / (1024*1024):.2f} MB")

    # Update config.json with unified intermediate size
    import json
    config_file = output_path / "config.json"
    with open(config_file, 'r') as f:
        config = json.load(f)

    config["moe_intermediate_size"] = unified_size
    config["n_shared_experts"] = 1

    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"  ✓ Updated config.json: moe_intermediate_size={unified_size}, n_shared_experts=1")

    # Also save a metadata file for tracking
    metadata = {
        "source_file": str(input_path),
        "num_layers": num_layers,
        "num_experts": num_experts,
        "routed_intermediate_size": routed_size,
        "shared_intermediate_size": shared_size,
        "unified_intermediate_size": unified_size,
        "total_weights": len(mlx_weights),
        "padding_applied": routed_size != unified_size,
    }

    metadata_file = output_path / "weight_mapping_info.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✓ Saved metadata: {metadata_file}")

    print("\n" + "=" * 70)
    print("✅ Weight remapping complete!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. In Xcode, delete old weights.safetensors and config.json")
    print("  2. Drag new weights.safetensors and config.json into Xcode")
    print("  3. Rebuild and run on your iPhone")
    print("\n  Or simply rebuild - if you already have them in Copy Bundle Resources!")

if __name__ == "__main__":
    # Get input path from command line or use default
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    else:
        input_path = "model_weights/best_deepseek_v3.pt"

    try:
        remap_weights(input_path)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nUsage: python remap_weights_for_mlx_unified.py [path/to/weights.pt]")
        sys.exit(1)
