//
//  DeepSeekChatView.swift
//  POC
//
//  Created by Matt Chung on 11/6/25.
//

import SwiftUI
import Combine
import MLXLLM
import MLXLMCommon
import MLX

// Simple chat message model
struct ChatMessage: Identifiable, Equatable {
    let id = UUID()
    let isUser: Bool
    let text: String
}

@MainActor
final class DeepSeekChatViewModel: ObservableObject {
    @Published var isLoading: Bool = true
    @Published var progress: Double = 0 // 0.0 ... 1.0
    @Published var messages: [ChatMessage] = []
    @Published var inputText: String = ""
    @Published var isSending: Bool = false

    private var modelContainer: ModelContainer?
    private var chatSession: ChatSession?

    init() {
        loadModel()
    }

    func loadModel() {
        isLoading = true
        progress = 0

        Task {
            do {
                // Note: In simulator, MLX will automatically fall back to CPU if Metal/GPU not available
                #if targetEnvironment(simulator)
                print("⚠️ Running in simulator - using CPU (slower, but works)")
                #else
                print("✓ Running on device - using GPU/Neural Engine")
                #endif

                // Since Xcode copies individual files (not folder), create a temporary directory
                // and copy the model files there for MLX to load
                let fileManager = FileManager.default
                let tempDir = fileManager.temporaryDirectory.appendingPathComponent("mlx_model_\(UUID().uuidString)", isDirectory: true)

                // Create temp directory
                try fileManager.createDirectory(at: tempDir, withIntermediateDirectories: true)

                // Copy model files from bundle to temp directory
                guard let configURL = Bundle.main.url(forResource: "config", withExtension: "json"),
                      let tokenizerURL = Bundle.main.url(forResource: "tokenizer", withExtension: "json"),
                      let tokenizerConfigURL = Bundle.main.url(forResource: "tokenizer_config", withExtension: "json"),
                      let weightsURL = Bundle.main.url(forResource: "weights", withExtension: "safetensors") else {
                    throw NSError(domain: "DeepSeek", code: 1, userInfo: [
                        NSLocalizedDescriptionKey: "Model files not found in bundle. Check: config.json, tokenizer.json, tokenizer_config.json, weights.safetensors in Copy Bundle Resources."
                    ])
                }

                print("Copying model files to temp directory...")
                try fileManager.copyItem(at: configURL, to: tempDir.appendingPathComponent("config.json"))
                try fileManager.copyItem(at: tokenizerURL, to: tempDir.appendingPathComponent("tokenizer.json"))
                try fileManager.copyItem(at: tokenizerConfigURL, to: tempDir.appendingPathComponent("tokenizer_config.json"))
                try fileManager.copyItem(at: weightsURL, to: tempDir.appendingPathComponent("weights.safetensors"))

                print("Loading DeepSeek model from: \(tempDir.path)")

                // Load the model with progress tracking
                let container = try await loadModelContainer(directory: tempDir) { loadProgress in
                    Task { @MainActor in
                        self.progress = loadProgress.fractionCompleted
                        print("Loading progress: \(Int(loadProgress.fractionCompleted * 100))%")
                    }
                }

                self.modelContainer = container
                self.chatSession = ChatSession(container)
                self.isLoading = false
                self.messages = [ChatMessage(isUser: false, text: "DeepSeek V3 is ready! I'm a 109M parameter model with MoE architecture. Ask me anything!")]
                print("✓ DeepSeek model loaded successfully!")

            } catch {
                print("✗ Error loading model: \(error)")
                self.isLoading = false
                self.messages = [ChatMessage(
                    isUser: false,
                    text: "Error loading model: \(error.localizedDescription)\n\nPlease check SETUP_MLX.md for instructions."
                )]
            }
        }
    }

    func send() async {
        let text = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty, !isLoading, !isSending else {
            print("⚠️ Skipping send: empty=\(!text.isEmpty), loading=\(isLoading), sending=\(isSending)")
            return
        }

        guard let session = chatSession else {
            print("✗ Error: chatSession is nil!")
            messages.append(ChatMessage(
                isUser: false,
                text: "Error: Model not loaded. Please restart the app."
            ))
            return
        }

        inputText = ""
        isSending = true

        // Add user message
        let userMessage = ChatMessage(isUser: true, text: text)
        messages.append(userMessage)
        print("📤 User: \(text)")

        do {
            // Use MLX to generate response
            print("🤖 Generating response...")
            let response = try await session.respond(to: text)

            // Add assistant message
            let assistantMessage = ChatMessage(isUser: false, text: response)
            messages.append(assistantMessage)
            print("✓ Response generated: \(response.prefix(100))...")

        } catch {
            print("✗ Error generating response: \(error)")
            print("   Error type: \(type(of: error))")
            print("   Error description: \(error.localizedDescription)")

            messages.append(ChatMessage(
                isUser: false,
                text: "Error generating response: \(error.localizedDescription)\n\nPlease check Xcode console for details."
            ))
        }

        isSending = false
    }
}

struct DeepSeekChatView: View {
    @StateObject private var model = DeepSeekChatViewModel()

    var body: some View {
        VStack(spacing: 0) {
            if model.isLoading {
                VStack(spacing: 12) {
                    Text("Loading DeepSeek model…")
                        .font(.headline)
                    ProgressView(value: model.progress) {
                        Text("Preparing")
                    } currentValueLabel: {
                        Text("\(Int(model.progress * 100))%")
                    }
                    .progressViewStyle(.linear)
                    .padding(.horizontal)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                ChatList(messages: model.messages)
                ChatInputBar(text: $model.inputText, isSending: model.isSending, onSend: {
                    Task { await model.send() }
                })
            }
        }
        .navigationTitle("DeepSeek Chat")
        .navigationBarTitleDisplayMode(.inline)
    }
}

private struct ChatList: View {
    let messages: [ChatMessage]

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 12) {
                    ForEach(messages) { message in
                        HStack {
                            if message.isUser { Spacer() }
                            Text(message.text)
                                .padding(10)
                                .background(message.isUser ? Color.accentColor.opacity(0.15) : Color.gray.opacity(0.15))
                                .clipShape(RoundedRectangle(cornerRadius: 12))
                            if !message.isUser { Spacer() }
                        }
                        .id(message.id)
                    }
                }
                .padding()
            }
            .onChange(of: messages) { _, _ in
                if let last = messages.last { withAnimation { proxy.scrollTo(last.id, anchor: .bottom) } }
            }
        }
    }
}

private struct ChatInputBar: View {
    @Binding var text: String
    let isSending: Bool
    let onSend: () -> Void

    var body: some View {
        HStack(spacing: 8) {
            TextField("Message", text: $text, axis: .vertical)
                .textFieldStyle(.roundedBorder)
                .lineLimit(1...4)

            Button(action: onSend) {
                if isSending {
                    ProgressView()
                } else {
                    Image(systemName: "paperplane.fill")
                }
            }
            .disabled(text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || isSending)
        }
        .padding(.all, 12)
        .background(.ultraThinMaterial)
    }
}
