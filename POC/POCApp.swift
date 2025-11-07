//
//  POCApp.swift
//  POC
//
//  Created by Matt Chung on 11/6/25.
//

import SwiftUI
import CoreData

@main
struct POCApp: App {
    let persistenceController = PersistenceController.shared

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environment(\.managedObjectContext, persistenceController.container.viewContext)
        }
    }
}
