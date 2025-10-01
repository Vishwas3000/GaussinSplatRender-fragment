import Metal
import MetalKit
import simd

struct MetalGaussianSplat {
    let position: SIMD3<Float>
    let scale: SIMD3<Float>
    let rotation: simd_quatf
    let opacity: Float
    let color: SIMD3<Float>
    
    init(from gaussianSplat: GaussianSplat) {
        self.position = gaussianSplat.position
        self.scale = gaussianSplat.scale
        self.rotation = gaussianSplat.rotation
        self.opacity = gaussianSplat.opacity
        self.color = gaussianSplat.color
    }
    
    static var stride: Int {
        return MemoryLayout<SIMD3<Float>>.size * 3 +  // position + scale + color
               MemoryLayout<simd_quatf>.size +         // rotation
               MemoryLayout<Float>.size               // opacity
    }
}

struct MetalSplatBuffer {
    let buffer: MTLBuffer
    let count: Int
    let device: MTLDevice
    
    init(device: MTLDevice, splats: [GaussianSplat]) throws {
        self.device = device
        self.count = splats.count
        
        let bufferSize = splats.count * MetalGaussianSplat.stride
        
        guard let buffer = device.makeBuffer(length: bufferSize, options: .storageModeShared) else {
            throw ParseError.memoryAllocationFailure
        }
        
        self.buffer = buffer
        
        try updateBuffer(with: splats)
    }
    
    init(device: MTLDevice, capacity: Int) throws {
        self.device = device
        self.count = capacity
        
        let bufferSize = capacity * MetalGaussianSplat.stride
        
        guard let buffer = device.makeBuffer(length: bufferSize, options: .storageModeShared) else {
            throw ParseError.memoryAllocationFailure
        }
        
        self.buffer = buffer
    }
    
    func updateBuffer(with splats: [GaussianSplat]) throws {
        guard splats.count <= count else {
            throw ParseError.memoryAllocationFailure
        }
        
        let pointer = buffer.contents().bindMemory(to: UInt8.self, capacity: buffer.length)
        var offset = 0
        
        for splat in splats {
            let metalSplat = MetalGaussianSplat(from: splat)
            
            // Copy position
            withUnsafeBytes(of: metalSplat.position) { bytes in
                memcpy(pointer.advanced(by: offset), bytes.baseAddress!, bytes.count)
                offset += bytes.count
            }
            
            // Copy scale
            withUnsafeBytes(of: metalSplat.scale) { bytes in
                memcpy(pointer.advanced(by: offset), bytes.baseAddress!, bytes.count)
                offset += bytes.count
            }
            
            // Copy rotation
            withUnsafeBytes(of: metalSplat.rotation) { bytes in
                memcpy(pointer.advanced(by: offset), bytes.baseAddress!, bytes.count)
                offset += bytes.count
            }
            
            // Copy opacity
            withUnsafeBytes(of: metalSplat.opacity) { bytes in
                memcpy(pointer.advanced(by: offset), bytes.baseAddress!, bytes.count)
                offset += bytes.count
            }
            
            // Copy color
            withUnsafeBytes(of: metalSplat.color) { bytes in
                memcpy(pointer.advanced(by: offset), bytes.baseAddress!, bytes.count)
                offset += bytes.count
            }
        }
    }
    
    func updatePartialBuffer(with splats: [GaussianSplat], startIndex: Int) throws {
        guard startIndex + splats.count <= count else {
            throw ParseError.memoryAllocationFailure
        }
        
        let pointer = buffer.contents().bindMemory(to: UInt8.self, capacity: buffer.length)
        var offset = startIndex * MetalGaussianSplat.stride
        
        for splat in splats {
            let metalSplat = MetalGaussianSplat(from: splat)
            
            withUnsafeBytes(of: metalSplat.position) { bytes in
                memcpy(pointer.advanced(by: offset), bytes.baseAddress!, bytes.count)
                offset += bytes.count
            }
            
            withUnsafeBytes(of: metalSplat.scale) { bytes in
                memcpy(pointer.advanced(by: offset), bytes.baseAddress!, bytes.count)
                offset += bytes.count
            }
            
            withUnsafeBytes(of: metalSplat.rotation) { bytes in
                memcpy(pointer.advanced(by: offset), bytes.baseAddress!, bytes.count)
                offset += bytes.count
            }
            
            withUnsafeBytes(of: metalSplat.opacity) { bytes in
                memcpy(pointer.advanced(by: offset), bytes.baseAddress!, bytes.count)
                offset += bytes.count
            }
            
            withUnsafeBytes(of: metalSplat.color) { bytes in
                memcpy(pointer.advanced(by: offset), bytes.baseAddress!, bytes.count)
                offset += bytes.count
            }
        }
    }
}

class GaussianSplatRenderer: NSObject {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private var pipelineState: MTLRenderPipelineState?
    private var splatBuffer: MetalSplatBuffer?
    
    struct ViewUniforms {
        let viewMatrix: simd_float4x4
        let projectionMatrix: simd_float4x4
        let viewProjectionMatrix: simd_float4x4
        let cameraPosition: SIMD3<Float>
        let screenSize: SIMD2<Float>
    }
    
    init(device: MTLDevice) {
        self.device = device
        self.commandQueue = device.makeCommandQueue()!
        super.init()

        self.setupPipeline()
    }
    
    func loadSplats(from collection: GaussianSplatCollection) throws {
        let splats = Array(collection)
        splatBuffer = try MetalSplatBuffer(device: device, splats: splats)
    }
    
    func loadSplatsStreaming(from url: URL, progressCallback: ProgressCallback? = nil) async throws {
        var totalSplats: [GaussianSplat] = []
        totalSplats.reserveCapacity(100000) // Reserve for better performance
        
        let result = await GaussianSplatParserFactory.shared.parseFileStreaming(
            at: url,
            options: .streaming(maxMemory: 100_000_000),
            progressCallback: progressCallback
        ) { splats in
            totalSplats.append(contentsOf: splats)
            return true // Continue processing
        }
        
        switch result {
        case .success(_):
            splatBuffer = try MetalSplatBuffer(device: device, splats: totalSplats)
        case .failure(let error):
            throw error
        }
    }
    
    func render(
        in view: MTKView,
        viewMatrix: simd_float4x4,
        projectionMatrix: simd_float4x4,
        cameraPosition: SIMD3<Float>
    ) {
        guard let drawable = view.currentDrawable,
              let renderPassDescriptor = view.currentRenderPassDescriptor,
              let commandBuffer = commandQueue.makeCommandBuffer(),
              let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor),
              let pipelineState = pipelineState,
              let splatBuffer = splatBuffer else {
            return
        }
        
        renderPassDescriptor.colorAttachments[0].clearColor = MTLClearColor(red: 0.0, green: 0.0, blue: 0.0, alpha: 1.0)
        renderPassDescriptor.colorAttachments[0].loadAction = .clear
        
        let viewProjectionMatrix = projectionMatrix * viewMatrix
        let screenSize = SIMD2<Float>(Float(view.drawableSize.width), Float(view.drawableSize.height))
        
        var uniforms = ViewUniforms(
            viewMatrix: viewMatrix,
            projectionMatrix: projectionMatrix,
            viewProjectionMatrix: viewProjectionMatrix,
            cameraPosition: cameraPosition,
            screenSize: screenSize
        )
        
        renderEncoder.setRenderPipelineState(pipelineState)
        renderEncoder.setVertexBuffer(splatBuffer.buffer, offset: 0, index: 0)
        renderEncoder.setVertexBytes(&uniforms, length: MemoryLayout<ViewUniforms>.size, index: 1)
        renderEncoder.setFragmentBytes(&uniforms, length: MemoryLayout<ViewUniforms>.size, index: 0)
        
        // Render as points that will be expanded to quads in the vertex shader
        renderEncoder.drawPrimitives(type: .point, vertexStart: 0, vertexCount: splatBuffer.count)
        
        renderEncoder.endEncoding()
        commandBuffer.present(drawable)
        commandBuffer.commit()
    }
    
    private func setupPipeline() {
        guard let library = device.makeDefaultLibrary() else {
            print("Failed to create Metal library")
            return
        }
        
        let vertexFunction = library.makeFunction(name: "gaussianSplatVertex")
        let fragmentFunction = library.makeFunction(name: "gaussianSplatFragment")
        
        let pipelineDescriptor = MTLRenderPipelineDescriptor()
        pipelineDescriptor.vertexFunction = vertexFunction
        pipelineDescriptor.fragmentFunction = fragmentFunction
        pipelineDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
        pipelineDescriptor.colorAttachments[0].isBlendingEnabled = true
        pipelineDescriptor.colorAttachments[0].rgbBlendOperation = .add
        pipelineDescriptor.colorAttachments[0].alphaBlendOperation = .add
        pipelineDescriptor.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
        pipelineDescriptor.colorAttachments[0].sourceAlphaBlendFactor = .sourceAlpha
        pipelineDescriptor.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
        pipelineDescriptor.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha
        
        let vertexDescriptor = MTLVertexDescriptor()
        vertexDescriptor.attributes[0].format = .float3  // position
        vertexDescriptor.attributes[0].offset = 0
        vertexDescriptor.attributes[0].bufferIndex = 0
        
        vertexDescriptor.attributes[1].format = .float3  // scale
        vertexDescriptor.attributes[1].offset = MemoryLayout<SIMD3<Float>>.size
        vertexDescriptor.attributes[1].bufferIndex = 0
        
        vertexDescriptor.attributes[2].format = .float4  // rotation (quaternion)
        vertexDescriptor.attributes[2].offset = MemoryLayout<SIMD3<Float>>.size * 2
        vertexDescriptor.attributes[2].bufferIndex = 0
        
        vertexDescriptor.attributes[3].format = .float   // opacity
        vertexDescriptor.attributes[3].offset = MemoryLayout<SIMD3<Float>>.size * 2 + MemoryLayout<simd_quatf>.size
        vertexDescriptor.attributes[3].bufferIndex = 0
        
        vertexDescriptor.attributes[4].format = .float3  // color
        vertexDescriptor.attributes[4].offset = MemoryLayout<SIMD3<Float>>.size * 2 + MemoryLayout<simd_quatf>.size + MemoryLayout<Float>.size
        vertexDescriptor.attributes[4].bufferIndex = 0
        
        vertexDescriptor.layouts[0].stride = MetalGaussianSplat.stride
        vertexDescriptor.layouts[0].stepRate = 1
        vertexDescriptor.layouts[0].stepFunction = .perVertex
        
        pipelineDescriptor.vertexDescriptor = vertexDescriptor
        
        do {
            pipelineState = try device.makeRenderPipelineState(descriptor: pipelineDescriptor)
        } catch {
            print("Failed to create render pipeline state: \(error)")
        }
    }
}

extension GaussianSplatCollection {
    func createMetalBuffer(device: MTLDevice) throws -> MetalSplatBuffer {
        let splats = Array(self)
        return try MetalSplatBuffer(device: device, splats: splats)
    }
    
    func updateMetalBuffer(_ buffer: MetalSplatBuffer) throws {
        let splats = Array(self)
        try buffer.updateBuffer(with: splats)
    }
}

extension GaussianSplatParserFactory {
    func parseAndCreateMetalBuffer(
        from url: URL,
        device: MTLDevice,
        options: ParsingOptions = .default,
        progressCallback: ProgressCallback? = nil
    ) async -> Result<MetalSplatBuffer, ParseError> {
        
        let parseResult = await parseFile(at: url, options: options, progressCallback: progressCallback)
        
        switch parseResult {
        case .success(let collection):
            do {
                let buffer = try collection.createMetalBuffer(device: device)
                return .success(buffer)
            } catch {
                return .failure(.memoryAllocationFailure)
            }
        case .failure(let error):
            return .failure(error)
        }
    }
}
