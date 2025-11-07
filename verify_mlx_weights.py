#!/usr/bin/env python3
"""
Verify that all MLX-expected weights are present in the safetensors file.
This simulates what MLX will look for when loading the model.
"""

from safetensors.numpy import load_file
import json

def verify_weights():
    print("=" * 70)
    print("MLX Weight Verification")
    print("=" * 70)

    # Load config
    with open('POC/mlx_model/config.json', 'r') as f:
        config = json.load(f)

    # Load weights
    weights = load_file('POC/mlx_model/weights.safetensors')

    print(f"\nConfig summary:")
    print(f"  Layers: {config['num_hidden_layers']}")
    print(f"  Experts: {config['n_routed_experts']}")
    print(f"  Shared experts: {config['n_shared_experts']}")
    print(f"  Hidden size: {config['hidden_size']}")
    print(f"  MoE intermediate: {config['moe_intermediate_size']}")

    missing = []

    # Check embedding
    required = ["model.embed_tokens.weight"]
    for key in required:
        if key not in weights:
            missing.append(key)

    # Check each layer
    for i in range(config['num_hidden_layers']):
        prefix = f"model.layers.{i}"

        # Layer norms
        required = [
            f"{prefix}.input_layernorm.weight",
            f"{prefix}.post_attention_layernorm.weight",
        ]

        # Attention
        required += [
            f"{prefix}.self_attn.q_a_proj.weight",
            f"{prefix}.self_attn.q_a_layernorm.weight",
            f"{prefix}.self_attn.q_b_proj.weight",
            f"{prefix}.self_attn.kv_a_proj_with_mqa.weight",
            f"{prefix}.self_attn.kv_a_layernorm.weight",
            f"{prefix}.self_attn.kv_b_proj.weight",
            f"{prefix}.self_attn.o_proj.weight",
        ]

        # MoE Gate
        required += [
            f"{prefix}.mlp.gate.weight",
            f"{prefix}.mlp.gate.e_score_correction_bias",
        ]

        # Routed experts (stacked as switch_mlp)
        required += [
            f"{prefix}.mlp.switch_mlp.gate_proj.weight",
            f"{prefix}.mlp.switch_mlp.up_proj.weight",
            f"{prefix}.mlp.switch_mlp.down_proj.weight",
        ]

        # Shared expert (if enabled)
        if config['n_shared_experts'] > 0:
            required += [
                f"{prefix}.mlp.shared_experts.gate_proj.weight",
                f"{prefix}.mlp.shared_experts.up_proj.weight",
                f"{prefix}.mlp.shared_experts.down_proj.weight",
            ]

        for key in required:
            if key not in weights:
                missing.append(key)

    # Final norm and LM head
    required = [
        "model.norm.weight",
        "lm_head.weight",
    ]
    for key in required:
        if key not in weights:
            missing.append(key)

    print(f"\nVerification results:")
    print(f"  Total weights in file: {len(weights)}")
    print(f"  Missing weights: {len(missing)}")

    if missing:
        print("\n❌ MISSING WEIGHTS:")
        for key in missing[:10]:  # Show first 10
            print(f"  - {key}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")
        return False
    else:
        print("\n✅ ALL REQUIRED WEIGHTS PRESENT!")
        print("\nModel is ready to load in MLX.")
        return True

if __name__ == "__main__":
    success = verify_weights()
    exit(0 if success else 1)
