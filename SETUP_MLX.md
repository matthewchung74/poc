# Setting Up MLX for DeepSeek Chat

## Step 1: Add Swift Package Dependency

1. Open `POC.xcodeproj` in Xcode
2. Select the POC project in the navigator
3. Select the POC target
4. Go to "Package Dependencies" tab
5. Click the "+" button
6. Enter: `https://github.com/ml-explore/mlx-swift-examples`
7. Branch: `main`
8. Click "Add Package"
9. **Wait for package to resolve** (may take 30-60 seconds)
10. When products appear, select:
    - ✅ MLXLLM
    - ✅ MLXLMCommon
11. Click "Add Package"

### Troubleshooting: Products Don't Appear

If you don't see MLXLLM and MLXLMCommon in the product list:

**Option A: Wait for Resolution**
- The package is still downloading/resolving
- Watch the progress indicator at top of Xcode
- Wait until it says "Package resolution complete"
- Then the products will appear

**Option B: Manual Target Setup**
If products still don't show after resolution:

1. Close the package dialog
2. Go to your POC target → "General" tab
3. Scroll to "Frameworks, Libraries, and Embedded Content"
4. Click the "+" button
5. You should now see:
   - MLXLLM
   - MLXLMCommon
6. Add both, set to "Do Not Embed"

**Option C: Check Package Status**
1. Go to File → Packages → Resolve Package Versions
2. Wait for resolution to complete
3. Try adding products again

## Step 2: Add Model Files to Project

1. In Xcode, right-click on POC folder
2. Select "Add Files to POC..."
3. Navigate to `/Users/mattc/Desktop/POC/POC/mlx_model`
4. Select the `mlx_model` folder
5. ✅ Check "Copy items if needed"
6. ✅ Check "Create folder references" (NOT "Create groups")
7. Click "Add"

## Step 3: Verify Files

The `mlx_model` folder should appear as a blue folder icon in Xcode with:
- config.json
- tokenizer_config.json
- weights.safetensors

## Step 4: Run the App

1. Select a target device (iPhone 15 Pro simulator or real device)
2. Build and Run (⌘R)
3. Tap "DeepSeek" in the list
4. Wait for model to load (~5-10 seconds)
5. Start chatting!

## Troubleshooting

**"Model file not found"**:
- Make sure mlx_model folder was added as "folder reference" (blue folder)
- Check Build Phases → Copy Bundle Resources includes mlx_model

**"Module 'MLXLLM' not found"**:
- Verify package dependency was added correctly
- Try Product → Clean Build Folder, then rebuild

**Model loads slowly**:
- First load is slower (GPU warm-up)
- Subsequent loads are faster
- Real device is faster than simulator

**Out of memory**:
- Test on device with ≥6GB RAM (iPhone 12 or later)
- Close other apps
- Model is 388MB - should fit on most modern devices
