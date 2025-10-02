import Metal
import MetalKit
import simd
import UIKit

struct DebugVertex {
    let position: SIMD3<Float>
    let color: SIMD3<Float>
    
    static var stride: Int {
        return MemoryLayout<DebugVertex>.stride
    }
}

class TiledSplatRenderer: NSObject, MTKViewDelegate, UIGestureRecognizerDelegate {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    
    // Compute pipeline for tile culling
    private var tileCullingPipeline: MTLComputePipelineState!
    
    // Render pipeline for fullscreen pass
    private var renderPipeline: MTLRenderPipelineState!
    
    // Debug pipeline for visualizing tiles
    private var debugPipeline: MTLRenderPipelineState!
    
    // Buffers
    private var splatBuffer: MTLBuffer!
    private var tileBuffer: MTLBuffer!
    private var viewUniformsBuffer: MTLBuffer!
    private var tileUniformsBuffer: MTLBuffer!
    private var splatCountBuffer: MTLBuffer!
    
    // Splat data
    private var splats: [GaussianSplat] = []
    private var tileUniforms: TileUniforms!
    
    // Orbital camera state
    private var cameraTarget: SIMD3<Float> = SIMD3<Float>(0, 0, 0)  // What we're looking at
    private var cameraDistance: Float = 8.0                          // Distance from target
    private var cameraAzimuth: Float = 0.0                          // Horizontal rotation around target
    private var cameraElevation: Float = 0.0                        // Vertical angle (up/down)
    private var cameraPosition: SIMD3<Float> = SIMD3<Float>(0, 0, 8) // Computed position
    
    // Gesture control state
    private var lastPanTranslation: SIMD2<Float> = SIMD2<Float>(0, 0)
    
    // Debug modes
    private var showDebugTiles: Bool = false
    private var cameraDebugMode: CameraDebugMode = .off
    
    // Scene generation parameters
    private var maxSplatCount: Int = 5000  // Configurable max splat count
    private var splatScaleMultiplier: Float = 0.3  // Reduce covariance scale
    
    // Camera debug state
    private var cameraTrail: [SIMD3<Float>] = []
    private let maxTrailPoints = 50
    private var debugLineBuffer: MTLBuffer?
    private var debugLinePipeline: MTLRenderPipelineState?
    private var debugInfoText: String = ""
    
    enum CameraDebugMode: Int, CaseIterable {
        case off = 0
        case coordinateSystem = 1
        case hudOnly = 2
        case fullDebug = 3
        case trailOnly = 4
        
        var description: String {
            switch self {
            case .off: return "Debug Off"
            case .coordinateSystem: return "Coordinate System"
            case .hudOnly: return "HUD Only"
            case .fullDebug: return "Full Debug"
            case .trailOnly: return "Trail Only"
            }
        }
    }
    
    init(device: MTLDevice) {
        self.device = device
        self.commandQueue = device.makeCommandQueue()!
        
        super.init()
        
        setupPipelines()
        generateRandomScene()
        updateCameraPosition() // Initialize camera position
    }
    
    private func setupPipelines() {
        guard let library = device.makeDefaultLibrary() else {
            fatalError("Could not create Metal library")
        }
        
        // Setup compute pipeline for tile culling
        guard let tileCullingFunction = library.makeFunction(name: "buildTiles") else {
            fatalError("Could not find buildTiles function")
        }
        
        do {
            tileCullingPipeline = try device.makeComputePipelineState(function: tileCullingFunction)
        } catch {
            fatalError("Could not create tile culling pipeline: \(error)")
        }
        
        // Setup render pipeline for fullscreen pass
        guard let vertexFunction = library.makeFunction(name: "fullscreenVertex"),
              let fragmentFunction = library.makeFunction(name: "gaussianSplatFragment") else {
            fatalError("Could not find shader functions")
        }
        
        let renderPipelineDescriptor = MTLRenderPipelineDescriptor()
        renderPipelineDescriptor.vertexFunction = vertexFunction
        renderPipelineDescriptor.fragmentFunction = fragmentFunction
        renderPipelineDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
        
        // Enable alpha blending
        renderPipelineDescriptor.colorAttachments[0].isBlendingEnabled = true
        renderPipelineDescriptor.colorAttachments[0].rgbBlendOperation = .add
        renderPipelineDescriptor.colorAttachments[0].alphaBlendOperation = .add
        renderPipelineDescriptor.colorAttachments[0].sourceRGBBlendFactor = .one
        renderPipelineDescriptor.colorAttachments[0].sourceAlphaBlendFactor = .one
        renderPipelineDescriptor.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
        renderPipelineDescriptor.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha
        
        do {
            renderPipeline = try device.makeRenderPipelineState(descriptor: renderPipelineDescriptor)
        } catch {
            fatalError("Could not create render pipeline: \(error)")
        }
        
        // Setup debug pipeline
        guard let debugFunction = library.makeFunction(name: "debugTilesFragment") else {
            fatalError("Could not find debug function")
        }
        
        let debugPipelineDescriptor = MTLRenderPipelineDescriptor()
        debugPipelineDescriptor.vertexFunction = vertexFunction
        debugPipelineDescriptor.fragmentFunction = debugFunction
        debugPipelineDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
        
        do {
            debugPipeline = try device.makeRenderPipelineState(descriptor: debugPipelineDescriptor)
        } catch {
            fatalError("Could not create debug pipeline: \(error)")
        }
        
        // Setup debug line rendering pipeline
        setupDebugLinePipeline(library: library)
    }
    
    private func setupDebugLinePipeline(library: MTLLibrary) {
        guard let debugLineVertex = library.makeFunction(name: "debugLineVertex"),
              let debugLineFragment = library.makeFunction(name: "debugLineFragment") else {
            print("Warning: Debug line shaders not found, debug visualization disabled")
            return
        }
        
        let debugLinePipelineDescriptor = MTLRenderPipelineDescriptor()
        debugLinePipelineDescriptor.vertexFunction = debugLineVertex
        debugLinePipelineDescriptor.fragmentFunction = debugLineFragment
        debugLinePipelineDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
        
        // Configure vertex descriptor for debug line attributes
        let vertexDescriptor = MTLVertexDescriptor()
        
        // Position attribute (attribute 0)
        vertexDescriptor.attributes[0].format = .float3
        vertexDescriptor.attributes[0].offset = 0
        vertexDescriptor.attributes[0].bufferIndex = 0
        
        // Color attribute (attribute 1)
        vertexDescriptor.attributes[1].format = .float3
        vertexDescriptor.attributes[1].offset = MemoryLayout<SIMD3<Float>>.stride
        vertexDescriptor.attributes[1].bufferIndex = 0
        
        // Buffer layout
        vertexDescriptor.layouts[0].stride = MemoryLayout<DebugVertex>.stride
        vertexDescriptor.layouts[0].stepRate = 1
        vertexDescriptor.layouts[0].stepFunction = .perVertex
        
        debugLinePipelineDescriptor.vertexDescriptor = vertexDescriptor
        
        // Enable alpha blending for debug overlays
        debugLinePipelineDescriptor.colorAttachments[0].isBlendingEnabled = true
        debugLinePipelineDescriptor.colorAttachments[0].rgbBlendOperation = .add
        debugLinePipelineDescriptor.colorAttachments[0].alphaBlendOperation = .add
        debugLinePipelineDescriptor.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
        debugLinePipelineDescriptor.colorAttachments[0].sourceAlphaBlendFactor = .sourceAlpha
        debugLinePipelineDescriptor.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
        debugLinePipelineDescriptor.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha
        
        do {
            debugLinePipeline = try device.makeRenderPipelineState(descriptor: debugLinePipelineDescriptor)
        } catch {
            print("Warning: Could not create debug line pipeline: \(error)")
        }
        
        // Create debug line buffer
        setupDebugLineBuffer()
    }
    
    private func setupDebugLineBuffer() {
        // Create buffer for debug lines (coordinate axes + trail + grid)
        let maxDebugVertices = 1000 // Enough for axes, grid, and trail
        let bufferSize = maxDebugVertices * MemoryLayout<DebugVertex>.stride
        debugLineBuffer = device.makeBuffer(length: bufferSize, options: .storageModeShared)
    }
    
    // MARK: - Debug Controls
    
    func cycleDebugMode() {
        let allCases = CameraDebugMode.allCases
        if let currentIndex = allCases.firstIndex(of: cameraDebugMode) {
            let nextIndex = (currentIndex + 1) % allCases.count
            cameraDebugMode = allCases[nextIndex]
            print("Debug mode changed to: \(cameraDebugMode)")
        }
    }
    
    func setDebugMode(_ mode: CameraDebugMode) {
        cameraDebugMode = mode
    }
    
    func getCurrentDebugInfo() -> String {
        return debugInfoText
    }
    
    func resetCameraTrail() {
        cameraTrail.removeAll()
    }
    
    // MARK: - Scene Configuration
    
    func setMaxSplatCount(_ count: Int) {
        maxSplatCount = max(100, min(count, 5000000)) // Clamp between 100 and 5M
        generateRandomScene()
    }
    
    func setSplatScale(_ scale: Float) {
        splatScaleMultiplier = max(0.1, min(scale, 3.0)) // Clamp between 0.1 and 3.0
        generateRandomScene()
    }
    
    func getMaxSplatCount() -> Int {
        return maxSplatCount
    }
    
    func getSplatScale() -> Float {
        return splatScaleMultiplier
    }
    
    // MARK: - Scene Generation Functions
    
    private func generateRandomScene() {
        // Create a diverse mix of different splat patterns with controlled count
        var allSplats: [GaussianSplat] = []
        
        let viewMatrix = createViewMatrix()
        let projectionMatrix = createProjectionMatrix(aspect: 1.0) // Will be updated in draw
        
        // Calculate proportional counts based on maxSplatCount
        let scatteredCount = min(Int(Float(maxSplatCount) * 0.4), 2000) // 40% for scattered
        let sphereCount = min(Int(Float(maxSplatCount) * 0.08), 100)    // 8% per sphere (5 spheres = 40%)
        let clusterCount = min(Int(Float(maxSplatCount) * 0.05), 60)    // 5% per cluster (4 clusters = 20%)
        
        // 1. Large scattered cloud in wide area
        let scatteredCloud = GaussianSplatGenerator.generateRandomSplats(
            count: scatteredCount,
            boundingBox: (min: SIMD3<Float>(-15, -15, -15), max: SIMD3<Float>(15, 15, 15)),
            viewMatrix: viewMatrix,
            projectionMatrix: projectionMatrix,
            scaleMultiplier: splatScaleMultiplier
        )
        allSplats.append(contentsOf: scatteredCloud)
        
        // 2. Multiple smaller spheres at different positions (only add if under limit)
        let spherePositions = [
            SIMD3<Float>(-8, 4, -3),
            SIMD3<Float>(6, -2, 8),
            SIMD3<Float>(0, 8, -10)
        ]
        
        for position in spherePositions {
            if allSplats.count < maxSplatCount {
                let remainingCount = min(sphereCount, maxSplatCount - allSplats.count)
                let sphere = GaussianSplatGenerator.generateSplatSphere(
                    count: remainingCount,
                    center: position,
                    radius: Float.random(in: 1.0...2.5), // Reduced radius
                    viewMatrix: viewMatrix,
                    projectionMatrix: projectionMatrix,
                    scaleMultiplier: splatScaleMultiplier
                )
                allSplats.append(contentsOf: sphere)
            }
        }
        
        // 3. Dispersed clusters in different regions (only add if under limit)
        let clusterPositions = [
            (center: SIMD3<Float>(-12, 0, 0), box: (min: SIMD3<Float>(-15, -3, -3), max: SIMD3<Float>(-9, 3, 3))),
            (center: SIMD3<Float>(12, 0, 0), box: (min: SIMD3<Float>(9, -3, -3), max: SIMD3<Float>(15, 3, 3)))
        ]
        
        for cluster in clusterPositions {
            if allSplats.count < maxSplatCount {
                let remainingCount = min(clusterCount, maxSplatCount - allSplats.count)
                let clusterSplats = GaussianSplatGenerator.generateSplatClusters(
                    clusterCount: 10, // Reduced cluster count
                    splatsPerCluster: max(1, remainingCount / 10),
                    boundingBox: cluster.box,
                    viewMatrix: viewMatrix,
                    projectionMatrix: projectionMatrix,
                    scaleMultiplier: splatScaleMultiplier
                )
                allSplats.append(contentsOf: clusterSplats)
            }
        }
        
        // Ensure we don't exceed maxSplatCount
        if allSplats.count > maxSplatCount {
            allSplats = Array(allSplats.prefix(maxSplatCount))
        }
        
        self.splats = allSplats
        print("Generated \(splats.count) splats (max: \(maxSplatCount)) with scale multiplier: \(splatScaleMultiplier)")
        
        setupBuffers()
    }
    
    private func generateLineFormation(start: SIMD3<Float>, end: SIMD3<Float>, count: Int, viewMatrix: simd_float4x4, projectionMatrix: simd_float4x4) -> [GaussianSplat] {
        var splats: [GaussianSplat] = []
        
        for i in 0..<count {
            let t = Float(i) / Float(count - 1)
            let position = start + t * (end - start)
            
            // Add some randomness around the line
            let noise = SIMD3<Float>(
                Float.random(in: -1...1),
                Float.random(in: -1...1),
                Float.random(in: -1...1)
            ) * 0.5
            
            let finalPosition = position + noise
            
            // Calculate depth
            let viewSpacePos = viewMatrix * SIMD4<Float>(finalPosition, 1.0)
            let depth = viewSpacePos.z
            
            // Create varied 3D covariance with more dramatic scale differences
            let scale = Float.random(in: 0.2...1.5) // Much wider range
            let covariance3D = createRandom3DCovariance(scale: scale)
            
            // Create vibrant colors based on position
            let normalizedPos = normalize(finalPosition)
            let hue = (atan2(normalizedPos.z, normalizedPos.x) + Float.pi) / (2 * Float.pi)
            let saturation = Float.random(in: 0.8...1.0)
            let brightness = Float.random(in: 0.7...1.0)
            let color = hsb_to_rgb(hue: hue, saturation: saturation, brightness: brightness)
            
            let opacity = Float.random(in: 0.6...0.9)
            
            let splat = GaussianSplat(
                position: finalPosition,
                covariance3D: covariance3D,
                color: color,
                opacity: opacity,
                depth: depth
            )
            
            splats.append(splat)
        }
        
        return splats
    }
    
    
    private func setupBuffers() {
        // Create splat buffer
        let splatDataSize = splats.count * MemoryLayout<GaussianSplat>.stride
        splatBuffer = device.makeBuffer(bytes: splats, length: splatDataSize, options: .storageModeShared)
        
        // Create buffers for uniforms
        viewUniformsBuffer = device.makeBuffer(length: MemoryLayout<ViewUniforms>.stride, options: .storageModeShared)
        tileUniformsBuffer = device.makeBuffer(length: MemoryLayout<TileUniforms>.stride, options: .storageModeShared)
        splatCountBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.stride, options: .storageModeShared)
        
        // Set splat count
        let splatCountPtr = splatCountBuffer.contents().bindMemory(to: UInt32.self, capacity: 1)
        splatCountPtr[0] = UInt32(splats.count)
    }
    
    private func setupTileBuffer(for drawableSize: CGSize) {
        let screenWidth = Int(drawableSize.width)
        let screenHeight = Int(drawableSize.height)
        
        tileUniforms = TileUniforms(screenWidth: screenWidth, screenHeight: screenHeight, tileSize: 16)
        
        // Update tile uniforms buffer
        let tileUniformsPtr = tileUniformsBuffer.contents().bindMemory(to: TileUniforms.self, capacity: 1)
        tileUniformsPtr[0] = tileUniforms
        
        // Create tile buffer (each tile can hold up to maxSplatsPerTile indices)
        let tileDataSize = tileUniforms.totalTiles * MemoryLayout<TileData>.stride
        tileBuffer = device.makeBuffer(length: tileDataSize, options: .storageModeShared)
        
        // Clear tile buffer
        let tilePtr = tileBuffer.contents().bindMemory(to: TileData.self, capacity: tileUniforms.totalTiles)
        for i in 0..<tileUniforms.totalTiles {
            tilePtr[i] = TileData()
        }
    }
    
    private func updateCamera(time: Float) {
        let previousPosition = cameraPosition
        
        // Calculate camera position from spherical coordinates around target
        updateCameraPosition()
        
        // Update camera trail for debug visualization
        if cameraDebugMode != .off && distance(previousPosition, cameraPosition) > 0.1 {
            updateCameraTrail()
        }
        
        // Update splat depths for proper sorting
        let viewMatrix = createViewMatrix()
        GaussianSplatGenerator.updateSplatDepths(splats: &splats, viewMatrix: viewMatrix)
        
        // Update splat buffer with new depth values
        let splatPtr = splatBuffer.contents().bindMemory(to: GaussianSplat.self, capacity: splats.count)
        for i in 0..<splats.count {
            splatPtr[i] = splats[i]
        }
    }
    
    private func updateCameraTrail() {
        cameraTrail.append(cameraPosition)
        
        // Keep trail at maximum size
        if cameraTrail.count > maxTrailPoints {
            cameraTrail.removeFirst()
        }
    }
    
    private func updateCameraPosition() {
        // Convert spherical coordinates to cartesian position around target
        let x = cameraDistance * cos(cameraElevation) * cos(cameraAzimuth)
        let y = cameraDistance * sin(cameraElevation)
        let z = cameraDistance * cos(cameraElevation) * sin(cameraAzimuth)
        
        cameraPosition = cameraTarget + SIMD3<Float>(x, y, z)
    }
    
    private func createViewMatrix() -> simd_float4x4 {
        // Create lookAt matrix: camera always looks at target
        return createLookAtMatrix(eye: cameraPosition, target: cameraTarget, up: SIMD3<Float>(0, 1, 0))
    }
    
    private func createProjectionMatrix(aspect: Float) -> simd_float4x4 {
        let fovy = Float.pi / 3.0 // 60 degrees
        let near: Float = 0.1
        let far: Float = 100.0
        
        return createPerspectiveMatrix(fovy: fovy, aspect: aspect, near: near, far: far)
    }
    
    // MARK: - MTKViewDelegate
    
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        setupTileBuffer(for: size)
    }
    
    func draw(in view: MTKView) {
        guard let drawable = view.currentDrawable,
              let renderPassDescriptor = view.currentRenderPassDescriptor,
              let commandBuffer = commandQueue.makeCommandBuffer() else {
            return
        }
        
        let time = Float(CACurrentMediaTime())
        updateCamera(time: time)
        
        let drawableSize = view.drawableSize
        let aspect = Float(drawableSize.width / drawableSize.height)
        
        // Update view uniforms
        let viewMatrix = createViewMatrix()
        let projectionMatrix = createProjectionMatrix(aspect: aspect)
        
        let viewUniforms = ViewUniforms(
            viewMatrix: viewMatrix,
            projectionMatrix: projectionMatrix,
            cameraPosition: cameraPosition,
            screenSize: SIMD2<Float>(Float(drawableSize.width), Float(drawableSize.height)),
            time: time
        )
        
        let viewUniformsPtr = viewUniformsBuffer.contents().bindMemory(to: ViewUniforms.self, capacity: 1)
        viewUniformsPtr[0] = viewUniforms
        
        // Phase 1: Tile Culling Compute Pass - DISABLED FOR TESTING
        // TODO: Re-enable once basic rendering works
        /*
        if let computeEncoder = commandBuffer.makeComputeCommandEncoder() {
            computeEncoder.setComputePipelineState(tileCullingPipeline)
            computeEncoder.setBuffer(splatBuffer, offset: 0, index: 0)
            computeEncoder.setBuffer(tileBuffer, offset: 0, index: 1)
            computeEncoder.setBuffer(viewUniformsBuffer, offset: 0, index: 2)
            computeEncoder.setBuffer(tileUniformsBuffer, offset: 0, index: 3)
            computeEncoder.setBuffer(splatCountBuffer, offset: 0, index: 4)
            
            let threadsPerThreadgroup = MTLSize(width: 8, height: 8, depth: 1)
            let threadgroupsPerGrid = MTLSize(
                width: (Int(tileUniforms.tilesPerRow) + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width,
                height: (Int(tileUniforms.tilesPerColumn) + threadsPerThreadgroup.height - 1) / threadsPerThreadgroup.height,
                depth: 1
            )
            
            computeEncoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
            computeEncoder.endEncoding()
        }
        */
        
        // Phase 2: Fullscreen Rendering Pass
        renderPassDescriptor.colorAttachments[0].clearColor = MTLClearColor(red: 0.0, green: 0.0, blue: 0.0, alpha: 1.0)
        renderPassDescriptor.colorAttachments[0].loadAction = .clear
        
        if let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) {
            if showDebugTiles {
                renderEncoder.setRenderPipelineState(debugPipeline)
            } else {
                renderEncoder.setRenderPipelineState(renderPipeline)
            }
            
            renderEncoder.setFragmentBuffer(splatBuffer, offset: 0, index: 0)
            renderEncoder.setFragmentBuffer(tileBuffer, offset: 0, index: 1)
            renderEncoder.setFragmentBuffer(viewUniformsBuffer, offset: 0, index: 2)
            renderEncoder.setFragmentBuffer(tileUniformsBuffer, offset: 0, index: 3)
            renderEncoder.setFragmentBuffer(splatCountBuffer, offset: 0, index: 4)
            
            // Draw fullscreen triangle
            renderEncoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
            
            renderEncoder.endEncoding()
        }
        
        // Render debug visualization if enabled
        if cameraDebugMode != .off && debugLinePipeline != nil {
            renderDebugVisualization(commandBuffer: commandBuffer, renderPassDescriptor: renderPassDescriptor, viewMatrix: viewMatrix, projectionMatrix: projectionMatrix)
        }
        
        commandBuffer.present(drawable)
        commandBuffer.commit()
    }
    
    private func renderDebugVisualization(commandBuffer: MTLCommandBuffer, renderPassDescriptor: MTLRenderPassDescriptor, viewMatrix: simd_float4x4, projectionMatrix: simd_float4x4) {
        guard let debugLinePipeline = debugLinePipeline,
              let debugLineBuffer = debugLineBuffer else { return }
        
        var debugVertices: [DebugVertex] = []
        
        // Add coordinate system axes
        if cameraDebugMode == .coordinateSystem || cameraDebugMode == .fullDebug {
            debugVertices.append(contentsOf: createCoordinateAxes())
            debugVertices.append(contentsOf: createGrid())
            debugVertices.append(contentsOf: createTargetMarker())
        }
        
        // Add camera trail
        if cameraDebugMode == .trailOnly || cameraDebugMode == .fullDebug {
            debugVertices.append(contentsOf: createCameraTrail())
        }
        
        // Update buffer with debug vertices
        if !debugVertices.isEmpty {
            let bufferPtr = debugLineBuffer.contents().bindMemory(to: DebugVertex.self, capacity: debugVertices.count)
            for (index, vertex) in debugVertices.enumerated() {
                bufferPtr[index] = vertex
            }
            
            // Render debug lines
            if let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) {
                renderEncoder.setRenderPipelineState(debugLinePipeline)
                renderEncoder.setVertexBuffer(debugLineBuffer, offset: 0, index: 0)
                renderEncoder.setVertexBuffer(viewUniformsBuffer, offset: 0, index: 1)
                
                renderEncoder.drawPrimitives(type: .line, vertexStart: 0, vertexCount: debugVertices.count)
                renderEncoder.endEncoding()
            }
        }
        
        // Render HUD text overlay
        if cameraDebugMode == .hudOnly || cameraDebugMode == .fullDebug {
            renderDebugHUD(commandBuffer: commandBuffer, renderPassDescriptor: renderPassDescriptor)
        }
    }
    
    private func createCoordinateAxes() -> [DebugVertex] {
        let axisLength: Float = 5.0
        return [
            // X-axis (Red)
            DebugVertex(position: SIMD3<Float>(0, 0, 0), color: SIMD3<Float>(1, 0, 0)),
            DebugVertex(position: SIMD3<Float>(axisLength, 0, 0), color: SIMD3<Float>(1, 0, 0)),
            
            // Y-axis (Green)
            DebugVertex(position: SIMD3<Float>(0, 0, 0), color: SIMD3<Float>(0, 1, 0)),
            DebugVertex(position: SIMD3<Float>(0, axisLength, 0), color: SIMD3<Float>(0, 1, 0)),
            
            // Z-axis (Blue)
            DebugVertex(position: SIMD3<Float>(0, 0, 0), color: SIMD3<Float>(0, 0, 1)),
            DebugVertex(position: SIMD3<Float>(0, 0, axisLength), color: SIMD3<Float>(0, 0, 1))
        ]
    }
    
    private func createGrid() -> [DebugVertex] {
        var gridVertices: [DebugVertex] = []
        let gridSize: Float = 20.0
        let gridStep: Float = 2.0
        let gridColor = SIMD3<Float>(0.3, 0.3, 0.3)
        
        // Grid lines along X
        for i in stride(from: -gridSize, through: gridSize, by: gridStep) {
            gridVertices.append(DebugVertex(position: SIMD3<Float>(i, 0, -gridSize), color: gridColor))
            gridVertices.append(DebugVertex(position: SIMD3<Float>(i, 0, gridSize), color: gridColor))
        }
        
        // Grid lines along Z
        for i in stride(from: -gridSize, through: gridSize, by: gridStep) {
            gridVertices.append(DebugVertex(position: SIMD3<Float>(-gridSize, 0, i), color: gridColor))
            gridVertices.append(DebugVertex(position: SIMD3<Float>(gridSize, 0, i), color: gridColor))
        }
        
        return gridVertices
    }
    
    private func createTargetMarker() -> [DebugVertex] {
        let markerSize: Float = 0.5
        let markerColor = SIMD3<Float>(1, 1, 0) // Yellow
        
        return [
            // Cross marker at target position
            DebugVertex(position: cameraTarget + SIMD3<Float>(-markerSize, 0, 0), color: markerColor),
            DebugVertex(position: cameraTarget + SIMD3<Float>(markerSize, 0, 0), color: markerColor),
            DebugVertex(position: cameraTarget + SIMD3<Float>(0, -markerSize, 0), color: markerColor),
            DebugVertex(position: cameraTarget + SIMD3<Float>(0, markerSize, 0), color: markerColor),
            DebugVertex(position: cameraTarget + SIMD3<Float>(0, 0, -markerSize), color: markerColor),
            DebugVertex(position: cameraTarget + SIMD3<Float>(0, 0, markerSize), color: markerColor)
        ]
    }
    
    private func createCameraTrail() -> [DebugVertex] {
        var trailVertices: [DebugVertex] = []
        let trailColor = SIMD3<Float>(0, 0.5, 1) // Blue
        
        // Need at least 2 points to create trail lines
        guard cameraTrail.count >= 2 else {
            return trailVertices
        }
        
        for i in 1..<cameraTrail.count {
            trailVertices.append(DebugVertex(position: cameraTrail[i-1], color: trailColor))
            trailVertices.append(DebugVertex(position: cameraTrail[i], color: trailColor))
        }
        
        return trailVertices
    }
    
    private func renderDebugHUD(commandBuffer: MTLCommandBuffer, renderPassDescriptor: MTLRenderPassDescriptor) {
        // For now, we'll use a simple text overlay system
        // In a production app, you might want to use Core Text or a more sophisticated text rendering system
        
        // Create debug info string
        let debugInfo = formatCameraDebugInfo()
        
        // Store debug info for potential UI overlay (this would need to be accessed by SwiftUI layer)
        debugInfoText = debugInfo
    }
    
    private func formatCameraDebugInfo() -> String {
        let azimuthDegrees = cameraAzimuth * 180.0 / Float.pi
        let elevationDegrees = cameraElevation * 180.0 / Float.pi
        
        return """
        Debug Mode: \(cameraDebugMode)
        Camera Position: (\(String(format: "%.2f", cameraPosition.x)), \(String(format: "%.2f", cameraPosition.y)), \(String(format: "%.2f", cameraPosition.z)))
        Camera Target: (\(String(format: "%.2f", cameraTarget.x)), \(String(format: "%.2f", cameraTarget.y)), \(String(format: "%.2f", cameraTarget.z)))
        Distance: \(String(format: "%.2f", cameraDistance))
        Azimuth: \(String(format: "%.1f", azimuthDegrees))°
        Elevation: \(String(format: "%.1f", elevationDegrees))°
        Trail Points: \(cameraTrail.count)
        Splats: \(splats.count)
        """
    }
    
    // MARK: - UIGestureRecognizerDelegate
    
    func gestureRecognizer(_ gestureRecognizer: UIGestureRecognizer, shouldRecognizeSimultaneouslyWith otherGestureRecognizer: UIGestureRecognizer) -> Bool {
        return true
    }
    
    // MARK: - Gesture Handlers
    
    @objc func handlePan(_ gesture: UIPanGestureRecognizer) {
        let translation = gesture.translation(in: gesture.view)
        let simdTranslation = SIMD2<Float>(Float(translation.x), Float(translation.y))
        handlePanGesture(translation: simdTranslation, state: gesture.state)
    }
    
    @objc func handlePinch(_ gesture: UIPinchGestureRecognizer) {
        handlePinchGesture(scale: Float(gesture.scale), state: gesture.state)
        gesture.scale = 1.0
    }
    
    @objc func handleRotation(_ gesture: UIRotationGestureRecognizer) {
        handleRotationGesture(rotation: Float(gesture.rotation), state: gesture.state)
        gesture.rotation = 0.0
    }
    
    @objc func handleDoubleTap(_ gesture: UITapGestureRecognizer) {
        cycleDebugMode()
    }
    
    // MARK: - Gesture Handling
    
    func handlePanGesture(translation: SIMD2<Float>, state: UIGestureRecognizer.State) {
        switch state {
        case .began:
            lastPanTranslation = translation
            
        case .changed:
            let delta = translation - lastPanTranslation
            
            // Convert screen movement to orbital rotation
            let sensitivity: Float = 0.005 // Smooth control
            
            // Horizontal drag = azimuth (rotate around target horizontally)
            // Negative to make left drag rotate left (natural feeling)
            cameraAzimuth += delta.x * sensitivity
            
            // Vertical drag = elevation (rotate around target vertically)  
            // Negative to make up drag look up (natural feeling)
            cameraElevation += delta.y * sensitivity
            
            // Clamp elevation to prevent flipping
            let maxElevation: Float = Float.pi * 0.45 // 81 degrees
            cameraElevation = max(-maxElevation, min(maxElevation, cameraElevation))
            
            lastPanTranslation = translation
            
        case .ended, .cancelled:
            break
            
        default:
            break
        }
    }
    
    func handlePinchGesture(scale: Float, state: UIGestureRecognizer.State) {
        switch state {
        case .began:
            break
            
        case .changed:
            // Zoom by adjusting distance from target
            let zoomFactor = 1.0 / scale
            let newDistance = cameraDistance * zoomFactor
            
            // Clamp distance to reasonable bounds
            cameraDistance = max(1.0, min(50.0, newDistance))
            
        case .ended, .cancelled:
            break
            
        default:
            break
        }
    }
    
    func handleRotationGesture(rotation: Float, state: UIGestureRecognizer.State) {
        switch state {
        case .began:
            break
            
        case .changed:
            // Additional horizontal rotation around target
            cameraAzimuth -= rotation * 0.5
            
        case .ended, .cancelled:
            break
            
        default:
            break
        }
    }
    
    // Public methods for interaction
    func toggleDebugMode() {
        showDebugTiles.toggle()
    }
    
    func regenerateScene() {
        generateRandomScene()
    }
    
    func setSplatCount(_ count: Int) {
        if count != splats.count {
            generateRandomScene()
        }
    }
}

// Camera utility functions
func createLookAtMatrix(eye: SIMD3<Float>, target: SIMD3<Float>, up: SIMD3<Float>) -> simd_float4x4 {
    let forward = normalize(target - eye)
    let right = normalize(cross(forward, up))
    let realUp = cross(right, forward)
    
    return simd_float4x4(
        SIMD4<Float>(right.x, realUp.x, -forward.x, 0),
        SIMD4<Float>(right.y, realUp.y, -forward.y, 0),
        SIMD4<Float>(right.z, realUp.z, -forward.z, 0),
        SIMD4<Float>(-dot(right, eye), -dot(realUp, eye), dot(forward, eye), 1)
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
