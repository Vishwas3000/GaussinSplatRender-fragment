//
//  ContentView.swift
//  GaussianSplat
//
//  Created by Vishwas Prakash on 01/10/25.
//

import SwiftUI
import UniformTypeIdentifiers

// MARK: - ViewModel to hold renderer reference

class ContentViewModel: ObservableObject {
    @Published var renderer: TiledSplatRenderer?
}

// MARK: - ContentView

struct ContentView: View {
    @State private var showingFilePicker = false
    @StateObject private var viewModel = ContentViewModel()
    @State private var loadingStatus: String = ""
    @State private var showingAlert = false
    @State private var alertMessage = ""
    
    var body: some View {
        ZStack {
            // Metal view for rendering
            MetalView(viewModel: viewModel)
                .ignoresSafeArea()
            
            // UI Overlay
            VStack {
                // Top controls
                HStack {
                    // Load file button
                    Button(action: {
                        print("\n🔘 Load button tapped")
                        print("🔧 Renderer status: \(viewModel.renderer != nil ? "Available ✓" : "NIL ✗")")
                        showingFilePicker = true
                    }) {
                        HStack {
                            Image(systemName: "folder.badge.plus")
                            Text("Load SPZ")
                        }
                        .padding(12)
                        .background(Color.blue)
                        .foregroundColor(.white)
                        .cornerRadius(10)
                        .shadow(radius: 3)
                    }
                    
                    Spacer()
                    
                    // Debug info button (optional)
                    Button(action: {
                        showDebugInfo()
                    }) {
                        Image(systemName: "info.circle")
                            .padding(12)
                            .background(Color.gray.opacity(0.8))
                            .foregroundColor(.white)
                            .cornerRadius(10)
                            .shadow(radius: 3)
                    }
                }
                .padding()
                
                // Loading status
                if !loadingStatus.isEmpty {
                    Text(loadingStatus)
                        .padding(8)
                        .background(Color.black.opacity(0.7))
                        .foregroundColor(.white)
                        .cornerRadius(8)
                        .padding(.horizontal)
                }
                
                Spacer()
                
                // Bottom instructions
                VStack(spacing: 8) {
                    Text("Controls:")
                        .font(.headline)
                        .foregroundColor(.white)
                    
                    HStack(spacing: 20) {
                        VStack {
                            Image(systemName: "hand.draw")
                            Text("Pan: Rotate")
                                .font(.caption)
                        }
                        
                        VStack {
                            Image(systemName: "arrow.up.left.and.arrow.down.right")
                            Text("Pinch: Zoom")
                                .font(.caption)
                        }
                        
                        VStack {
                            Image(systemName: "hand.tap")
                            Text("Double Tap: Debug")
                                .font(.caption)
                        }
                    }
                    .foregroundColor(.white)
                }
                .padding()
                .background(Color.black.opacity(0.5))
                .cornerRadius(15)
                .padding()
            }
        }
        .fileImporter(
            isPresented: $showingFilePicker,
            allowedContentTypes: [
                UTType(filenameExtension: "spz")!
            ],
            allowsMultipleSelection: false
        ) { result in
            print("\n📂 File importer callback triggered")
            handleFileSelection(result: result)
        }
        .alert("Info", isPresented: $showingAlert) {
            Button("OK", role: .cancel) { }
        } message: {
            Text(alertMessage)
        }
    }
    
    // MARK: - Helper Functions
    
    private func handleFileSelection(result: Result<[URL], Error>) {
        print("🔍 handleFileSelection called")
        
        switch result {
        case .success(let urls):
            print("✓ File selection successful, URLs count: \(urls.count)")
            guard let url = urls.first else {
                print("❌ No URL found in array")
                return
            }
            
            print("📁 Selected file: \(url.lastPathComponent)")
            print("📍 Full path: \(url.path)")
            print("🔧 Renderer status: \(viewModel.renderer != nil ? "Available ✓" : "NIL ✗")")
            
            loadingStatus = "Loading file..."
            
            // Grant access to security-scoped resource
            let accessing = url.startAccessingSecurityScopedResource()
            print("🔒 Security access: \(accessing ? "Granted" : "Not needed")")
            
            // Call renderer directly on main thread for testing
            if let renderer = viewModel.renderer {
                print("🚀 Calling renderer.loadFromFile()...")
                renderer.loadFromFile(url: url)
                
                loadingStatus = "Loaded successfully!"
                
                DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
                    loadingStatus = ""
                }
            } else {
                print("❌ CRITICAL ERROR: Renderer is NIL!")
                loadingStatus = "Error: Renderer not initialized"
                alertMessage = "Renderer not initialized. Please restart the app."
                showingAlert = true
            }
            
            if accessing {
                url.stopAccessingSecurityScopedResource()
            }
            
        case .failure(let error):
            print("❌ File picker error: \(error)")
            loadingStatus = "File picker error"
            alertMessage = error.localizedDescription
            showingAlert = true
            
            DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
                loadingStatus = ""
            }
        }
    }
    
    private func showDebugInfo() {
        guard let renderer = viewModel.renderer else {
            alertMessage = "Renderer not initialized"
            showingAlert = true
            return
        }
        
        let debugInfo = renderer.getCurrentDebugInfo()
        
        if debugInfo.isEmpty {
            alertMessage = """
            Splat Count: \(renderer.splats.count)
            Max Splats: \(renderer.getMaxSplatCount())
            Scale Multiplier: \(String(format: "%.2f", renderer.getSplatScale()))
            
            Tip: Double-tap to cycle through debug modes
            """
        } else {
            alertMessage = debugInfo
        }
        
        showingAlert = true
    }
}

#Preview {
    ContentView()
}
