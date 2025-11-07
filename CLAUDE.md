# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**POC** - A SwiftUI-based proof-of-concept application for integrating Core ML models (specifically DeepSeek) into an iOS/macOS app with a chat interface.

**Platform**: iOS/macOS (SwiftUI)
**Build System**: Xcode project
**Key Technologies**: SwiftUI, Core ML, CoreData, Combine

## Building and Running

```bash
# Open the project in Xcode
open POC.xcodeproj

# Build from command line (Debug)
xcodebuild -project POC.xcodeproj -scheme POC -configuration Debug build

# Build from command line (Release)
xcodebuild -project POC.xcodeproj -scheme POC -configuration Release build

# Clean build
xcodebuild -project POC.xcodeproj -scheme POC clean

# Run tests (when added)
xcodebuild test -project POC.xcodeproj -scheme POC -destination 'platform=iOS Simulator,name=iPhone 15'
```

## Project Architecture

### Application Structure

**Entry Point**: `POCApp.swift`
- Main app initialization
- CoreData persistence controller injection via environment

**Navigation Hub**: `ContentView.swift`
- Central navigation list for all POC features
- Uses NavigationStack for iOS 16+ navigation
- Add new POC features as NavigationLink items here

### DeepSeek Chat Integration

**Purpose**: Proof-of-concept for running on-device Core ML chat models with streaming UI

**Files**:
- `DeepSeekChatView.swift` - Complete chat interface implementation

**Architecture**:
1. **DeepSeekChatViewModel** (`@MainActor`, `ObservableObject`)
   - Manages model loading with progress tracking
   - Handles message state and chat history
   - Coordinates async message sending
   - Currently simulated - TODO markers indicate where to integrate real Core ML model

2. **DeepSeekChatView** - Main container
   - Switches between loading screen (with progress bar) and chat interface
   - Manages ViewModel lifecycle via `@StateObject`

3. **ChatList** - Scrollable message display
   - Uses `ScrollViewReader` for auto-scroll to latest message
   - Lazy loading for performance with large histories
   - User messages aligned right (blue), model responses aligned left (gray)

4. **ChatInputBar** - Message composition
   - Multi-line TextField (1-4 lines)
   - Send button with loading state
   - Disabled during model loading or message sending

**Integration Points (TODO)**:
- Line 33-53: Replace simulated model loading with `MLModel.load(contentsOf:configuration:progress:)`
- Line 56-75: Replace echo response with actual Core ML token streaming
- Map `Progress.fractionCompleted` to `@Published var progress`

### Data Persistence

**Files**: `Persistence.swift`, `POC.xcdatamodeld/`

**Purpose**: CoreData stack for local storage (currently unused in chat feature)

**Configuration**:
- Singleton pattern: `PersistenceController.shared`
- In-memory preview mode for SwiftUI previews
- Automatic merge of changes from parent context

**Current State**: CoreData setup is boilerplate - not yet utilized by DeepSeek chat (messages are ephemeral). Consider adding message persistence or removing if not needed.

## Development Notes

### Adding New POC Features

1. Create new SwiftUI view file in `POC/` directory
2. Add NavigationLink to `ContentView.swift`:
```swift
NavigationLink("Feature Name") {
    YourNewView()
}
```

### MLX Model Integration (DeepSeek V3)

**Note**: Uses Apple MLX instead of Core ML (MLX is Apple's framework specifically for LLMs)

**Model Location**: `POC/mlx_model/` (540MB)
- `weights.safetensors` - Model weights (no retraining needed!)
- `config.json` - Model architecture (8 layers, 512 embd, 8 experts MoE, unified 768 intermediate)
- `tokenizer_config.json` - GPT-2 tokenizer

**Swift Package Dependencies**:
```swift
dependencies: [
    .package(url: "https://github.com/ml-explore/mlx-swift-examples", branch: "main")
]
// Add products: MLXLLM, MLXLMCommon
```

**Loading Model** (`DeepSeekChatViewModel.loadModel()`):
```swift
import MLXLLM
import MLXLMCommon

// Model files are added individually to Xcode (not as folder)
// Copy to temp directory for MLX to load
let fileManager = FileManager.default
let tempDir = fileManager.temporaryDirectory.appendingPathComponent("mlx_model_\(UUID().uuidString)")
try fileManager.createDirectory(at: tempDir, withIntermediateDirectories: true)

// Copy config.json, tokenizer_config.json, weights.safetensors
guard let configURL = Bundle.main.url(forResource: "config", withExtension: "json"),
      let tokenizerURL = Bundle.main.url(forResource: "tokenizer_config", withExtension: "json"),
      let weightsURL = Bundle.main.url(forResource: "weights", withExtension: "safetensors") else {
    throw ModelError.filesNotFound
}

try fileManager.copyItem(at: configURL, to: tempDir.appendingPathComponent("config.json"))
try fileManager.copyItem(at: tokenizerURL, to: tempDir.appendingPathComponent("tokenizer_config.json"))
try fileManager.copyItem(at: weightsURL, to: tempDir.appendingPathComponent("weights.safetensors"))

// Load model
let modelContainer = try await loadModelContainer(directory: tempDir) { progress in
    self.progress = progress.fractionCompleted
}
```

**Generating Text** (`DeepSeekChatViewModel.send()`):
```swift
let session = ChatSession(modelContainer)
let response = try await session.respond(to: userMessage)
// MLX handles tokenization + generation automatically
```

**Key Points**:
- MLX uses Metal + Neural Engine (fast!)
- DeepSeek V3 architecture already supported in MLX
- Model was converted (not retrained) from PyTorch → MLX format
- Streaming supported via async sequences
- **IMPORTANT**: Model won't work in simulator - requires real iPhone with Metal GPU

**Custom Architecture Mapping**:
This is a scaled-down DeepSeek V3 (109M params) with custom architecture that required special mapping:
1. **MoE experts**: Padded routed experts (512→768) to match shared expert size
2. **Attention layers**: Combined separate PyTorch projections into MLX's unified format
3. **No MLP biases**: MLX doesn't use bias terms in MLP layers

See `ARCHITECTURE_MAPPING.md` for full technical details.

**Deployment Requirements**:
- iOS 17.0+ (for MLX framework)
- Real device only (Metal GPU required, no simulator support)

### SwiftUI Patterns Used

- **MVVM**: ViewModel (`@MainActor` class) + View separation
- **Combine**: `ObservableObject` + `@Published` for reactive state
- **Async/Await**: Task-based concurrency for message sending
- **Environment Injection**: CoreData context passed via `.environment()`

### Platform Targets

Current target appears to be iOS, but SwiftUI code is cross-platform compatible. To add macOS:
1. Update project settings to include macOS target
2. Test navigation (macOS uses different navigation patterns)
3. Adjust UI for larger screens if needed
