import SwiftUI
import MetalKit

struct MetalView: UIViewRepresentable {
    func makeUIView(context: Context) -> MTKView {
        let metalView = MTKView()
        
        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError("Metal is not supported on this device")
        }
        
        metalView.device = device
        metalView.colorPixelFormat = .bgra8Unorm
        metalView.clearColor = MTLClearColor(red: 0.0, green: 0.0, blue: 0.0, alpha: 1.0)
        
        let renderer = TriangleRenderer(device: device)
        metalView.delegate = renderer
        
        context.coordinator.renderer = renderer
        
        return metalView
    }
    
    func updateUIView(_ uiView: MTKView, context: Context) {
    }
    
    func makeCoordinator() -> Coordinator {
        Coordinator()
    }
    
    class Coordinator {
        var renderer: TriangleRenderer?
    }
}