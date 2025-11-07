# DeepSeek V3 on iOS - Proof of Concept

An iOS app that runs a custom DeepSeek V3 language model (109M parameters) on-device using Apple's MLX framework.

## ⚠️ Important Note

**This model is undertrained and produces low-quality output.** It's a proof-of-concept demonstrating the DeepSeek V3 architecture and MLX integration on iOS. The model generates text but it's not coherent - this is expected for a 109M parameter model trained on limited data.

**For production use**, replace with a properly trained model from [MLX Community](https://huggingface.co/mlx-community) like:
- `Qwen3-4B-4bit` (better quality, 4GB)
- `Llama-3.2-1B-4bit` (good quality, 1GB)

## Features

- ✅ On-device inference with Metal GPU acceleration
- ✅ Custom scaled-down DeepSeek V3 architecture (109M params)
- ✅ Multi-Head Latent Attention (MLA)
- ✅ Mixture of Experts (MoE) with 8 experts
- ✅ Chat interface with SwiftUI
- ✅ No external API calls - completely private
- ⚠️ Output quality: Educational/demonstration purposes only

## Requirements

- iOS 17.0+
- Xcode 15+
- Real iPhone device (Metal GPU required - no simulator support)
- ~550MB free space for model weights

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/matthewchung74/poc.git
cd poc
```

### 2. Download Model Weights

The model weights are too large for GitHub. Download them separately:

```bash
# Create Python virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install torch safetensors huggingface_hub

# Download original PyTorch weights
python3 download_model.py

# Convert to MLX format (this creates POC/mlx_model/weights.safetensors)
python3 remap_weights_for_mlx_unified.py
```

**Expected files in `POC/mlx_model/`**:
- `config.json` (675 B) - ✅ Included in repo
- `tokenizer.json` (1.3 MB) - ✅ Included in repo
- `tokenizer_config.json` (53 B) - ✅ Included in repo
- `weights.safetensors` (540 MB) - ⚠️ Download via script above

### 3. Open in Xcode

```bash
open POC.xcodeproj
```

### 4. Add Model Files to Xcode

1. In Xcode Project Navigator, find `POC/mlx_model/`
2. Verify these files are present:
   - ✅ config.json
   - ✅ tokenizer.json
   - ✅ tokenizer_config.json
   - ⚠️ weights.safetensors (add this if missing)

3. If `weights.safetensors` is missing:
   - Right-click on `mlx_model` folder → "Add Files to POC..."
   - Select `weights.safetensors`
   - ✅ Check "Copy items if needed"
   - Click "Add"

4. Verify in Build Phases → Copy Bundle Resources:
   - All 4 files should be listed

### 5. Build and Run

1. Select your iPhone as the target device
2. Build and Run (⌘R)
3. Tap "DeepSeek" in the app
4. Wait for model to load (~10-20 seconds)
5. Start chatting!

## Model Details

**Architecture**: Custom DeepSeek V3 (scaled down from 671B → 109M parameters)

- **Vocabulary**: 50,257 tokens (GPT-2 BPE tokenizer)
- **Layers**: 8 transformer layers
- **Hidden size**: 512
- **Attention heads**: 8
- **MoE**: 8 routed experts + 1 shared expert
- **Experts per token**: 2
- **Context length**: 1024 tokens

**Training**: Trained on FineWeb-Edu dataset (2.5B tokens)

**Original source**: [DeepSeek-V3-from-Scratch](https://github.com/Mayankpratapsingh022/DeepSeek-from-Scratch) by Mayank Pratap Singh

## Documentation

- **[CLAUDE.md](CLAUDE.md)** - Complete developer guide
- **[ARCHITECTURE_MAPPING.md](ARCHITECTURE_MAPPING.md)** - PyTorch → MLX architecture mapping
- **[SETUP_MLX.md](SETUP_MLX.md)** - Xcode setup instructions
- **[WEIGHT_MAPPING_COMPLETE.md](WEIGHT_MAPPING_COMPLETE.md)** - Weight conversion details

## Project Structure

```
POC/
├── POC/                          # iOS app source
│   ├── POCApp.swift             # App entry point
│   ├── ContentView.swift        # Navigation hub
│   ├── DeepSeekChatView.swift  # Chat interface + model loading
│   └── mlx_model/               # Model files
│       ├── config.json          # Model configuration
│       ├── tokenizer.json       # GPT-2 tokenizer
│       ├── tokenizer_config.json
│       └── weights.safetensors  # Model weights (download separately)
├── remap_weights_for_mlx_unified.py  # PyTorch → MLX converter
├── verify_mlx_weights.py             # Weight verification script
└── download_model.py                 # Download script
```

## Architecture Highlights

### Custom Mapping Required

This scaled-down DeepSeek model has a **different architecture** than the standard DeepSeek V3, requiring custom weight mapping:

1. **MoE Intermediate Sizes**
   - Routed experts: 512 → padded to 768
   - Shared expert: 768 (kept as-is)
   - Solution: Zero-padding for uniform size

2. **Attention Architecture**
   - PyTorch: Separate `kv_proj` + `k_rope_proj`
   - MLX: Combined `kv_a_proj_with_mqa`
   - Solution: Concatenate projections

3. **Expert Format**
   - PyTorch: Individual `experts.0`, `experts.1`, ...
   - MLX: Stacked `switch_mlp` format
   - Solution: Stack all expert weights

See [ARCHITECTURE_MAPPING.md](ARCHITECTURE_MAPPING.md) for full technical details.

## Performance

- **Model size**: 540 MB
- **Load time**: 10-20 seconds (first load)
- **Inference**: ~1-2 tokens/sec on iPhone 13 Pro
- **Memory**: ~600 MB RAM usage

## Troubleshooting

### "Model files not found in bundle"
- Verify all 4 files are in Copy Bundle Resources
- Clean build folder (Shift+⌘K) and rebuild

### "configurationMissing(tokenizer.json)"
- Make sure `tokenizer.json` was added to the project
- Check it's in Copy Bundle Resources

### App crashes or hangs
- Model won't work in simulator - use real device
- First load takes 10-20 seconds - be patient
- Check Xcode console for error messages

### "unsupportedTokenizer"
- Make sure `tokenizer_config.json` has `"tokenizer_class": "PreTrainedTokenizer"`

## Development

Built with:
- Swift 5.9+
- SwiftUI
- MLX Swift (Apple's ML framework)
- Xcode 15+

## License

This is a proof-of-concept educational project. The original DeepSeek model is from [Mayank Pratap Singh's implementation](https://github.com/Mayankpratapsingh022/DeepSeek-from-Scratch).

## Credits

- Original DeepSeek V3 architecture: [DeepSeek-AI](https://github.com/deepseek-ai/DeepSeek-V3)
- Scaled-down implementation: [Mayank Pratap Singh](https://github.com/Mayankpratapsingh022/DeepSeek-from-Scratch)
- MLX framework: [Apple ML-Explore](https://github.com/ml-explore/mlx-swift-examples)

## Future Improvements

- [ ] Streaming token generation (currently blocks)
- [ ] Conversation history persistence
- [ ] Model quantization for smaller size
- [ ] Fine-tuning on custom datasets
- [ ] Multi-turn conversation support
