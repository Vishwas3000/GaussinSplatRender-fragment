import SwiftUI
import MetalKit
import simd

struct MetalSplatView: UIViewRepresentable {
    let splatCollection: GaussianSplatCollection
    
    func makeUIView(context: Context) -> MTKView {
        let metalView = MTKView()
        
        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError("Metal is not supported on this device")
        }
        
        metalView.device = device
        metalView.colorPixelFormat = .bgra8Unorm
        metalView.clearColor = MTLClearColor(red: 0.0, green: 0.0, blue: 0.0, alpha: 1.0)
        metalView.enableSetNeedsDisplay = false
        metalView.isPaused = false
        
        let renderer = GaussianSplatRenderer(device: device)
        do {
            try renderer.loadSplats(from: splatCollection)
        } catch {
            print("Failed to load splats into renderer: \(error)")
        }
        
        metalView.delegate = renderer
        context.coordinator.renderer = renderer
        
        return metalView
    }
    
    func updateUIView(_ uiView: MTKView, context: Context) {
        // Update the renderer if the collection changes
        if let renderer = context.coordinator.renderer {
            do {
                try renderer.loadSplats(from: splatCollection)
            } catch {
                print("Failed to update splats in renderer: \(error)")
            }
        }
    }
    
    func makeCoordinator() -> Coordinator {
        Coordinator()
    }
    
    class Coordinator {
        var renderer: GaussianSplatRenderer?
    }
}

// Update the GaussianSplatRenderer to conform to MTKViewDelegate
extension GaussianSplatRenderer: MTKViewDelegate {
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        // Handle size changes if needed
    }
    
    func draw(in view: MTKView) {
        // Basic camera setup for viewing the splats
        let time = Float(CFAbsoluteTimeGetCurrent())
        
        // Create a simple rotating camera
        let radius: Float = 5.0
        let cameraPosition = SIMD3<Float>(
            radius * cos(time * 0.5),
            0.0,
            radius * sin(time * 0.5)
        )
        
        let target = SIMD3<Float>(0, 0, 0)
        let up = SIMD3<Float>(0, 1, 0)
        
        let viewMatrix = createLookAtMatrix(eye: cameraPosition, target: target, up: up)
        
        let aspect = Float(view.drawableSize.width / view.drawableSize.height)
        let projectionMatrix = createPerspectiveMatrix(fovy: Float.pi / 4, aspect: aspect, near: 0.1, far: 100.0)
        
        render(in: view, viewMatrix: viewMatrix, projectionMatrix: projectionMatrix, cameraPosition: cameraPosition)
    }
}

// Camera matrix utilities
func createLookAtMatrix(eye: SIMD3<Float>, target: SIMD3<Float>, up: SIMD3<Float>) -> simd_float4x4 {
    let zAxis = normalize(eye - target)
    let xAxis = normalize(cross(up, zAxis))
    let yAxis = cross(zAxis, xAxis)
    
    let translation = SIMD3<Float>(-dot(xAxis, eye), -dot(yAxis, eye), -dot(zAxis, eye))
    
    return simd_float4x4(
        SIMD4<Float>(xAxis.x, yAxis.x, zAxis.x, 0),
        SIMD4<Float>(xAxis.y, yAxis.y, zAxis.y, 0),
        SIMD4<Float>(xAxis.z, yAxis.z, zAxis.z, 0),
        SIMD4<Float>(translation.x, translation.y, translation.z, 1)
    )
}

func createPerspectiveMatrix(fovy: Float, aspect: Float, near: Float, far: Float) -> simd_float4x4 {
    let f = 1.0 / tan(fovy * 0.5)
    
    return simd_float4x4(
        SIMD4<Float>(f / aspect, 0, 0, 0),
        SIMD4<Float>(0, f, 0, 0),
        SIMD4<Float>(0, 0, (far + near) / (near - far), -1),
        SIMD4<Float>(0, 0, (2 * far * near) / (near - far), 0)
    )
}
