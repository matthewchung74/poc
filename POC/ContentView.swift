//
//  ContentView.swift
//  POC
//
//  Created by Matt Chung on 11/6/25.
//

import SwiftUI

struct ContentView: View {
    var body: some View {
        NavigationStack {
            List {
                NavigationLink("DeepSeek") {
                    DeepSeekChatView()
                }
                // Add more links here as your POC grows
            }
            .navigationTitle("POC Links")
        }
    }
}

#Preview {
    ContentView()
}
