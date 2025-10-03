import SwiftUI
import MetalKit
import UIKit

struct MetalView: UIViewRepresentable {
    func makeUIView(context: Context) -> MTKView {
        let metalView = MTKView()
        
        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError("Metal is not supported on this device")
        }
        
        metalView.device = device
        metalView.colorPixelFormat = .bgra8Unorm
        metalView.clearColor = MTLClearColor(red: 0.0, green: 0.0, blue: 0.0, alpha: 1.0)
        
        let renderer = TiledSplatRenderer(device: device)
        renderer.setMaxSplatCount(2000)
        metalView.delegate = renderer
        
        context.coordinator.renderer = renderer
        
        // Add gesture recognizers
        setupGestures(metalView: metalView, renderer: renderer)
        
        return metalView
    }
    
    func updateUIView(_ uiView: MTKView, context: Context) {
    }
    
    func makeCoordinator() -> Coordinator {
        Coordinator()
    }
    
    private func setupGestures(metalView: MTKView, renderer: TiledSplatRenderer) {
        // Pan gesture for rotation
        let panGesture = UIPanGestureRecognizer(target: renderer, action: #selector(TiledSplatRenderer.handlePan(_:)))
        metalView.addGestureRecognizer(panGesture)
        
        // Pinch gesture for zoom
        let pinchGesture = UIPinchGestureRecognizer(target: renderer, action: #selector(TiledSplatRenderer.handlePinch(_:)))
        metalView.addGestureRecognizer(pinchGesture)
        
        // Rotation gesture
        let rotationGesture = UIRotationGestureRecognizer(target: renderer, action: #selector(TiledSplatRenderer.handleRotation(_:)))
        metalView.addGestureRecognizer(rotationGesture)
        
        // Double tap gesture for debug mode cycling
        let doubleTapGesture = UITapGestureRecognizer(target: renderer, action: #selector(TiledSplatRenderer.handleDoubleTap(_:)))
        doubleTapGesture.numberOfTapsRequired = 2
        metalView.addGestureRecognizer(doubleTapGesture)
        
        // Allow simultaneous gestures
        panGesture.delegate = renderer
        pinchGesture.delegate = renderer
        rotationGesture.delegate = renderer
    }
    
    class Coordinator {
        var renderer: TiledSplatRenderer?
    }
}
