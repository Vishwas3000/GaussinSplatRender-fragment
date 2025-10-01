//
//  ContentView.swift
//  GaussianSplat
//
//  Created by Vishwas Prakash on 01/10/25.
//

import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @State private var selectedFileURL: URL?
    @State private var isShowingFilePicker = false
    @State private var isLoadingSplats = false
    @State private var loadingProgress: Float = 0.0
    @State private var splatCollection: GaussianSplatCollection?
    @State private var errorMessage: String?
    
    var body: some View {
        ZStack {
            if let collection = splatCollection {
                MetalSplatView(splatCollection: collection)
                    .ignoresSafeArea()
            } else {
                MetalView()
                    .ignoresSafeArea()
            }
            
            VStack {
                Spacer()
                
                HStack {
                    Button("Load Splat File") {
                        isShowingFilePicker = true
                    }
                    .padding()
                    .background(Color.blue)
                    .foregroundColor(.white)
                    .cornerRadius(10)
                    .disabled(isLoadingSplats)
                    
                    if isLoadingSplats {
                        VStack {
                            ProgressView(value: loadingProgress)
                                .frame(width: 200)
                            Text("\(Int(loadingProgress * 100))%")
                                .foregroundColor(.white)
                                .font(.caption)
                        }
                    }
                    
                    Spacer()
                }
                .padding()
            }
            
            if let error = errorMessage {
                VStack {
                    Text("Error")
                        .font(.headline)
                        .foregroundColor(.red)
                    Text(error)
                        .foregroundColor(.red)
                        .multilineTextAlignment(.center)
                    Button("Dismiss") {
                        errorMessage = nil
                    }
                    .padding()
                    .background(Color.red)
                    .foregroundColor(.white)
                    .cornerRadius(8)
                }
                .padding()
                .background(Color.black.opacity(0.8))
                .cornerRadius(10)
            }
        }
        .fileImporter(
            isPresented: $isShowingFilePicker,
            allowedContentTypes: [
                UTType(filenameExtension: "ply")!,
                UTType(filenameExtension: "splat")!
            ],
            allowsMultipleSelection: false
        ) { result in
            switch result {
            case .success(let urls):
                if let url = urls.first {
                    selectedFileURL = url
                    loadSplatFile(from: url)
                }
            case .failure(let error):
                errorMessage = "Failed to select file: \(error.localizedDescription)"
            }
        }
    }
    
    private func loadSplatFile(from url: URL) {
        guard url.startAccessingSecurityScopedResource() else {
            errorMessage = "Failed to access file"
            return
        }
        
        defer { url.stopAccessingSecurityScopedResource() }
        
        isLoadingSplats = true
        loadingProgress = 0.0
        errorMessage = nil
        
        Task {
            let result = await GaussianSplatParserFactory.shared.parseFile(
                at: url,
                options: .streaming(maxMemory: 100_000_000),
                progressCallback: { progress in
                    DispatchQueue.main.async {
                        loadingProgress = progress.percentage
                    }
                }
            )
            
            DispatchQueue.main.async {
                isLoadingSplats = false
                
                switch result {
                case .success(let collection):
                    splatCollection = collection
                    print("Loaded \(collection.count) splats from \(url.lastPathComponent)")
                case .failure(let error):
                    errorMessage = "Failed to load splat file: \(error.localizedDescription)"
                }
            }
        }
    }
}

#Preview {
    ContentView()
}
