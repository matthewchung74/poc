# Final Setup Steps - Add tokenizer.json to Xcode

## Problem
The app crashes with: `Error loading model: configurationMissing("tokenizer.json")`

## Solution
Add `tokenizer.json` to the Xcode project and rebuild.

## Step-by-Step Instructions

### 1. Add tokenizer.json to Xcode Project

**In Xcode:**

1. Open the Xcode project (`POC.xcodeproj`)

2. In the Project Navigator (left sidebar), find where the other model files are located:
   - Look for `config.json`
   - Look for `weights.safetensors`
   - Look for `tokenizer_config.json`

3. Right-click on the same folder/group → **"Add Files to POC..."**

4. Navigate to: `/Users/mattc/Desktop/POC/POC/mlx_model/`

5. Select **`tokenizer.json`** (1.3 MB file)

6. **Important checkboxes**:
   - ✅ **Check** "Copy items if needed"
   - ✅ **Check** "Add to targets: POC"
   - Click **"Add"**

### 2. Verify in Copy Bundle Resources

1. Select the **POC** target in the project navigator

2. Go to **"Build Phases"** tab

3. Expand **"Copy Bundle Resources"**

4. **Verify these 4 files are listed**:
   - ✅ `config.json`
   - ✅ `tokenizer.json` ← **NEW - must be here!**
   - ✅ `tokenizer_config.json`
   - ✅ `weights.safetensors`

5. If `tokenizer.json` is missing:
   - Click the **"+"** button
   - Find and add `tokenizer.json`

### 3. Clean and Rebuild

1. **Product** → **Clean Build Folder** (or Shift+Cmd+K)

2. **Product** → **Build** (or Cmd+B)

3. Wait for build to complete (may take 1-2 minutes)

### 4. Run on iPhone

1. Select **"Phone (1)"** as the target device

2. **Product** → **Run** (or Cmd+R)

3. Tap **"DeepSeek"** in the app

4. **Expected behavior**:
   - Loading screen with progress bar (0-100%)
   - Progress updates in console
   - Chat interface appears when loaded
   - You can send messages!

## File Sizes for Verification

Make sure these are the correct files:

- `config.json` - 675 bytes
- `tokenizer.json` - 1.3 MB ← **NEW**
- `tokenizer_config.json` - 53 bytes
- `weights.safetensors` - 540 MB

## What Changed in Code

Updated `DeepSeekChatView.swift` to load `tokenizer.json`:

```swift
// OLD (line 59):
let tokenizerURL = Bundle.main.url(forResource: "tokenizer_config", withExtension: "json")

// NEW (lines 59-60):
let tokenizerURL = Bundle.main.url(forResource: "tokenizer", withExtension: "json"),
let tokenizerConfigURL = Bundle.main.url(forResource: "tokenizer_config", withExtension: "json")

// And copy both files (line 69-70):
try fileManager.copyItem(at: tokenizerURL, to: tempDir.appendingPathComponent("tokenizer.json"))
try fileManager.copyItem(at: tokenizerConfigURL, to: tempDir.appendingPathComponent("tokenizer_config.json"))
```

## Troubleshooting

**"Model files not found in bundle"**:
- Make sure you added `tokenizer.json` to the project (Step 1)
- Verify it's in Copy Bundle Resources (Step 2)
- Clean and rebuild (Step 3)

**Build errors**:
- If you get "file not found", the file wasn't copied to the bundle
- Check that "Copy items if needed" was checked when adding
- Check Build Phases → Copy Bundle Resources

**Still crashes with "configurationMissing"**:
- Delete the app from the iPhone
- Clean build folder in Xcode
- Rebuild and reinstall

**App hangs or gets killed by debugger**:
- The 540MB model may take 10-20 seconds to load
- Watch the console for "Loading progress: X%" messages
- First load is always slower (GPU warm-up)

## Success Indicators

You'll know it's working when you see in the Xcode console:

```
✓ Running on device - using GPU/Neural Engine
Copying model files to temp directory...
Loading DeepSeek model from: /path/to/temp/dir
Loading progress: 0%
Loading progress: 25%
Loading progress: 50%
Loading progress: 75%
Loading progress: 100%
✓ Model loaded successfully!
```

Then the chat interface will appear and you can start chatting!

## Summary

**What we're adding**: `tokenizer.json` (1.3 MB)
**Why**: MLX needs the full GPT-2 tokenizer definition to encode/decode text
**Where**: Add to Xcode project in Copy Bundle Resources
**Result**: Model will load and chat will work!
