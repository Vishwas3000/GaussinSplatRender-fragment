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
    
    // Compute pipelines - Optimized Morton Code + Radix Sort + Frustum Culling
    private var preprocessSplatsPipeline: MTLComputePipelineState!
    private var frustumCullPipeline: MTLComputePipelineState!           // NEW!
    private var mortonCodePipeline: MTLComputePipelineState!
    private var mortonCodeVisiblePipeline: MTLComputePipelineState!     // NEW!
    private var radixSortCountPipeline: MTLComputePipelineState!
    private var radixSortScanPipeline: MTLComputePipelineState!
    private var radixSortScatterPipeline: MTLComputePipelineState!
    private var clearTilesPipeline: MTLComputePipelineState!
    private var buildTilesOptimizedPipeline: MTLComputePipelineState!
    private var buildTilesVisiblePipeline: MTLComputePipelineState!     // NEW!
    private var buildTilesPipeline: MTLComputePipelineState!            // Original tile building
    private var sortTileDepthPipeline: MTLComputePipelineState!         // Depth sorting for stability

    // Render pipeline for fullscreen pass
    private var renderPipeline: MTLRenderPipelineState!

    // Debug pipelines for visualizing tiles
    private var debugPipeline: MTLRenderPipelineState!
    private var debugHeatmapPipeline: MTLRenderPipelineState!
    private var debugWorkloadPipeline: MTLRenderPipelineState!
    private var debugCullingPipeline: MTLRenderPipelineState!

    // Buffers
    private var splatBuffer: MTLBuffer!
    private var preprocessedSplatBuffer: MTLBuffer!
    private var tileBuffer: MTLBuffer!
    private var viewUniformsBuffer: MTLBuffer!
    private var tileUniformsBuffer: MTLBuffer!
    private var splatCountBuffer: MTLBuffer!

    // Morton code & radix sort buffers
    private var mortonCodeBuffer: MTLBuffer!
    private var sortedIndicesBuffer: MTLBuffer!
    private var mortonCodeTempBuffer: MTLBuffer!
    private var sortedIndicesTempBuffer: MTLBuffer!
    private var histogramBuffer: MTLBuffer!
    private var offsetBuffer: MTLBuffer!

    // GPU Frustum Culling buffers
    private var visibleSplatIndicesBuffer: MTLBuffer!
    private var visibleSplatCountBuffer: MTLBuffer!
    private var useGPUFrustumCulling: Bool = true  // Toggle for A/B testing
    
    // Simple CPU Sorting (Energy Efficient)
    private var useHybridSorting: Bool = false  // Enable simple CPU distance sorting

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
    
    // Energy monitoring and adaptive quality
    private var lastCPUSortTime: CFTimeInterval = 0.0
    private var lastGPUSortTime: CFTimeInterval = 0.0
    private var lastGPUCullTime: CFTimeInterval = 0.0
    private var lastHybridPipelineTime: CFTimeInterval = 0.0
    private var adaptiveQualityEnabled: Bool = true
    
    // Performance tracking for hybrid approach
    private var totalSplatsProcessed: Int = 0
    private var visibleSplatsProcessed: Int = 0
    private var cullingEfficiencyHistory: [Float] = []
    private let maxHistorySize = 30 // Track last 30 frames
    
    // Camera debug state
    private var cameraTrail: [SIMD3<Float>] = []
    private let maxTrailPoints = 50
    private var debugLineBuffer: MTLBuffer?
    private var debugLinePipeline: MTLRenderPipelineState?
    private var debugInfoText: String = ""
    private var frameCount: Int = 0
    
    enum CameraDebugMode: Int, CaseIterable {
        case off = 0
        case coordinateSystem = 1
        case hudOnly = 2
        case fullDebug = 3
        case trailOnly = 4
        case tileHeatmap = 5
        case tileBoundaries = 6
        case splatAssignment = 7
        case cullingStats = 8
        
        var description: String {
            switch self {
            case .off: return "Debug Off"
            case .coordinateSystem: return "Coordinate System"
            case .hudOnly: return "HUD Only"
            case .fullDebug: return "Full Debug"
            case .trailOnly: return "Trail Only"
            case .tileHeatmap: return "Tile Heatmap"
            case .tileBoundaries: return "Tile Boundaries"
            case .splatAssignment: return "Splat Assignment"
            case .cullingStats: return "Culling Stats"
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
        
        // Setup compute pipeline for clearing tiles
        guard let clearTilesFunction = library.makeFunction(name: "clearTiles") else {
            fatalError("Could not find clearTiles function")
        }

        do {
            clearTilesPipeline = try device.makeComputePipelineState(function: clearTilesFunction)
        } catch {
            fatalError("Could not create clear tiles pipeline: \(error)")
        }

        // Setup preprocessing pipeline
        guard let preprocessFunction = library.makeFunction(name: "preprocessSplats") else {
            fatalError("Could not find preprocessSplats function")
        }

        do {
            preprocessSplatsPipeline = try device.makeComputePipelineState(function: preprocessFunction)
        } catch {
            fatalError("Could not create preprocess pipeline: \(error)")
        }

        // Setup GPU frustum culling pipeline
        guard let frustumCullFunction = library.makeFunction(name: "frustumCullSplats") else {
            fatalError("Could not find frustumCullSplats function")
        }

        do {
            frustumCullPipeline = try device.makeComputePipelineState(function: frustumCullFunction)
        } catch {
            fatalError("Could not create frustum cull pipeline: \(error)")
        }

        // Setup Morton code + radix sort optimized pipelines
        guard let mortonCodeFunction = library.makeFunction(name: "computeMortonCodes"),
              let mortonCodeVisibleFunction = library.makeFunction(name: "computeMortonCodesVisible"),
              let radixCountFunction = library.makeFunction(name: "radixSortCount"),
              let radixScanFunction = library.makeFunction(name: "radixSortScan"),
              let radixScatterFunction = library.makeFunction(name: "radixSortScatter"),
              let buildTilesFunction = library.makeFunction(name: "buildTiles"),
              let buildOptimizedFunction = library.makeFunction(name: "buildTilesOptimized"),
              let buildVisibleFunction = library.makeFunction(name: "buildTilesWithVisibleList"),
              let sortDepthFunction = library.makeFunction(name: "sortTileDepth") else {
            fatalError("Could not find optimized shader functions")
        }

        do {
            mortonCodePipeline = try device.makeComputePipelineState(function: mortonCodeFunction)
            mortonCodeVisiblePipeline = try device.makeComputePipelineState(function: mortonCodeVisibleFunction)
            radixSortCountPipeline = try device.makeComputePipelineState(function: radixCountFunction)
            radixSortScanPipeline = try device.makeComputePipelineState(function: radixScanFunction)
            radixSortScatterPipeline = try device.makeComputePipelineState(function: radixScatterFunction)
            buildTilesPipeline = try device.makeComputePipelineState(function: buildTilesFunction)
            buildTilesOptimizedPipeline = try device.makeComputePipelineState(function: buildOptimizedFunction)
            buildTilesVisiblePipeline = try device.makeComputePipelineState(function: buildVisibleFunction)
            sortTileDepthPipeline = try device.makeComputePipelineState(function: sortDepthFunction)
        } catch {
            fatalError("Could not create optimized pipelines: \(error)")
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
        
        // Setup enhanced debug pipelines for tile visualization
        setupDebugTilePipelines(library: library, vertexFunction: vertexFunction)

        // Setup debug line rendering pipeline
        setupDebugLinePipeline(library: library)
    }
    
    private func setupDebugTilePipelines(library: MTLLibrary, vertexFunction: MTLFunction) {
        // Debug tile heatmap pipeline
        if let heatmapFunction = library.makeFunction(name: "debugTileHeatmapFragment") {
            let descriptor = MTLRenderPipelineDescriptor()
            descriptor.vertexFunction = vertexFunction
            descriptor.fragmentFunction = heatmapFunction
            descriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
            descriptor.colorAttachments[0].isBlendingEnabled = true
            descriptor.colorAttachments[0].rgbBlendOperation = .add
            descriptor.colorAttachments[0].alphaBlendOperation = .add
            descriptor.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
            descriptor.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
            
            do {
                debugHeatmapPipeline = try device.makeRenderPipelineState(descriptor: descriptor)
            } catch {
                print("Warning: Could not create debug heatmap pipeline: \(error)")
            }
        }
        
        // Debug workload pipeline
        if let workloadFunction = library.makeFunction(name: "debugTileWorkloadFragment") {
            let descriptor = MTLRenderPipelineDescriptor()
            descriptor.vertexFunction = vertexFunction
            descriptor.fragmentFunction = workloadFunction
            descriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
            descriptor.colorAttachments[0].isBlendingEnabled = true
            descriptor.colorAttachments[0].rgbBlendOperation = .add
            descriptor.colorAttachments[0].alphaBlendOperation = .add
            descriptor.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
            descriptor.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
            
            do {
                debugWorkloadPipeline = try device.makeRenderPipelineState(descriptor: descriptor)
            } catch {
                print("Warning: Could not create debug workload pipeline: \(error)")
            }
        }
        
        // Debug culling efficiency pipeline  
        if let cullingFunction = library.makeFunction(name: "debugTileCullingFragment") {
            let descriptor = MTLRenderPipelineDescriptor()
            descriptor.vertexFunction = vertexFunction
            descriptor.fragmentFunction = cullingFunction
            descriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
            descriptor.colorAttachments[0].isBlendingEnabled = true
            descriptor.colorAttachments[0].rgbBlendOperation = .add
            descriptor.colorAttachments[0].alphaBlendOperation = .add
            descriptor.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
            descriptor.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
            
            do {
                debugCullingPipeline = try device.makeRenderPipelineState(descriptor: descriptor)
            } catch {
                print("Warning: Could not create debug culling pipeline: \(error)")
            }
        }
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
        let scatteredCount = Int(Float(maxSplatCount) * 0.4)  // 40% for scattered cloud
        let sphereCount = Int(Float(maxSplatCount) * 0.08)    // 8% per sphere (3 spheres = 24%)
        let clusterCount = Int(Float(maxSplatCount) * 0.18)   // 18% per cluster (2 clusters = 36%)
        
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

        // Create preprocessed splat buffer
        let preprocessedDataSize = splats.count * MemoryLayout<PreprocessedSplat>.stride
        preprocessedSplatBuffer = device.makeBuffer(length: preprocessedDataSize, options: .storageModeShared)

        // Create buffers for uniforms
        viewUniformsBuffer = device.makeBuffer(length: MemoryLayout<ViewUniforms>.stride, options: .storageModeShared)
        tileUniformsBuffer = device.makeBuffer(length: MemoryLayout<TileUniforms>.stride, options: .storageModeShared)
        splatCountBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.stride, options: .storageModeShared)

        // Set splat count
        let splatCountPtr = splatCountBuffer.contents().bindMemory(to: UInt32.self, capacity: 1)
        splatCountPtr[0] = UInt32(splats.count)

        // Setup Morton code & radix sort buffers
        setupMortonBuffers()
    }

    private func setupMortonBuffers() {
        let splatCount = splats.count

        // Morton codes and indices
        mortonCodeBuffer = device.makeBuffer(
            length: splatCount * MemoryLayout<UInt32>.stride,
            options: .storageModeShared
        )

        sortedIndicesBuffer = device.makeBuffer(
            length: splatCount * MemoryLayout<UInt32>.stride,
            options: .storageModeShared
        )

        // Temporary buffers for radix sort ping-pong
        mortonCodeTempBuffer = device.makeBuffer(
            length: splatCount * MemoryLayout<UInt32>.stride,
            options: .storageModeShared
        )

        sortedIndicesTempBuffer = device.makeBuffer(
            length: splatCount * MemoryLayout<UInt32>.stride,
            options: .storageModeShared
        )

        // Radix sort working buffers (256 bins per pass)
        histogramBuffer = device.makeBuffer(
            length: 256 * MemoryLayout<UInt32>.stride,
            options: .storageModeShared
        )

        offsetBuffer = device.makeBuffer(
            length: 256 * MemoryLayout<UInt32>.stride,
            options: .storageModeShared
        )

        // GPU Frustum Culling buffers
        // visibleSplatIndicesBuffer: Compact list of indices for visible splats (max = splatCount)
        visibleSplatIndicesBuffer = device.makeBuffer(
            length: splatCount * MemoryLayout<UInt32>.stride,
            options: .storageModeShared
        )

        // visibleSplatCountBuffer: Atomic counter for number of visible splats
        visibleSplatCountBuffer = device.makeBuffer(
            length: MemoryLayout<UInt32>.stride,
            options: .storageModeShared
        )
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

        // Clear tile buffer (initial setup only)
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
        let viewMatrix = createLookAtMatrix(eye: cameraPosition, target: cameraTarget, up: SIMD3<Float>(0, 1, 0))

        // Debug: Print view matrix components periodically
//        if frameCount % 60 == 0 {  // Every 60 frames (about once per second)
//            print("\n=== View Matrix Debug (Frame \(frameCount)) ===")
//            print("Camera Position: (\(String(format: "%.2f", cameraPosition.x)), \(String(format: "%.2f", cameraPosition.y)), \(String(format: "%.2f", cameraPosition.z)))")
//            print("Camera Target: (\(String(format: "%.2f", cameraTarget.x)), \(String(format: "%.2f", cameraTarget.y)), \(String(format: "%.2f", cameraTarget.z)))")
//            print("Azimuth: \(String(format: "%.1f", cameraAzimuth * 180 / Float.pi))°")
//            print("Elevation: \(String(format: "%.1f", cameraElevation * 180 / Float.pi))°")
//            print("\nView Matrix (column-major):")
//            print("Col 0 (right):    [\(String(format: "%6.3f", viewMatrix[0][0])), \(String(format: "%6.3f", viewMatrix[0][1])), \(String(format: "%6.3f", viewMatrix[0][2]))]")
//            print("Col 1 (up):       [\(String(format: "%6.3f", viewMatrix[1][0])), \(String(format: "%6.3f", viewMatrix[1][1])), \(String(format: "%6.3f", viewMatrix[1][2]))]")
//            print("Col 2 (-forward): [\(String(format: "%6.3f", viewMatrix[2][0])), \(String(format: "%6.3f", viewMatrix[2][1])), \(String(format: "%6.3f", viewMatrix[2][2]))]")
//            print("Col 3 (trans):    [\(String(format: "%6.3f", viewMatrix[3][0])), \(String(format: "%6.3f", viewMatrix[3][1])), \(String(format: "%6.3f", viewMatrix[3][2]))]")
//        }

        return viewMatrix
    }

    private func createProjectionMatrix(aspect: Float) -> simd_float4x4 {
        let fovy = Float.pi / 3.0 // 60 degrees
        let near: Float = 0.1
        let far: Float = 100.0

        return createPerspectiveMatrix(fovy: fovy, aspect: aspect, near: near, far: far)
    }

    // MARK: - Morton Code + Radix Sort

    private func performRadixSort(commandBuffer: MTLCommandBuffer, splatCount: Int) {
        var keysIn = mortonCodeBuffer!
        var keysOut = mortonCodeTempBuffer!
        var indicesIn = sortedIndicesBuffer!
        var indicesOut = sortedIndicesTempBuffer!

        let threadsPerGrid = MTLSize(width: splatCount, height: 1, depth: 1)
        let threadsPerThreadgroup = MTLSize(width: 256, height: 1, depth: 1)

        // 4 passes for 32-bit keys (8 bits per pass)
        for pass in 0..<4 {
            var passValue = UInt32(pass)

            // PHASE 1: Clear histogram (for this pass)
            if let blitEncoder = commandBuffer.makeBlitCommandEncoder() {
                blitEncoder.fill(buffer: histogramBuffer, range: 0..<histogramBuffer.length, value: 0)
                blitEncoder.endEncoding()
            }

            // PHASE 2: Count digit occurrences (histogram)
            if let computeEncoder = commandBuffer.makeComputeCommandEncoder() {
                computeEncoder.setComputePipelineState(radixSortCountPipeline)
                computeEncoder.setBuffer(keysIn, offset: 0, index: 0)
                computeEncoder.setBuffer(histogramBuffer, offset: 0, index: 1)
                computeEncoder.setBytes(&passValue, length: MemoryLayout<UInt32>.stride, index: 2)
                var count = UInt32(splatCount)
                computeEncoder.setBytes(&count, length: MemoryLayout<UInt32>.stride, index: 3)
                computeEncoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
                computeEncoder.endEncoding()
            }

            // PHASE 3: Prefix sum (exclusive scan) to get output positions
            // This writes to offsetBuffer
            if let computeEncoder = commandBuffer.makeComputeCommandEncoder() {
                computeEncoder.setComputePipelineState(radixSortScanPipeline)
                computeEncoder.setBuffer(histogramBuffer, offset: 0, index: 0)
                computeEncoder.setBuffer(offsetBuffer, offset: 0, index: 1)
                let scanThreads = MTLSize(width: 1, height: 1, depth: 1)
                computeEncoder.dispatchThreads(scanThreads, threadsPerThreadgroup: scanThreads)
                computeEncoder.endEncoding()
            }

            // PHASE 4: Scatter elements to sorted positions
            // CRITICAL: offsetBuffer gets modified atomically during scatter!
            // We pass it in, and atomic operations increment the values
            if let computeEncoder = commandBuffer.makeComputeCommandEncoder() {
                computeEncoder.setComputePipelineState(radixSortScatterPipeline)
                computeEncoder.setBuffer(keysIn, offset: 0, index: 0)
                computeEncoder.setBuffer(keysOut, offset: 0, index: 1)
                computeEncoder.setBuffer(indicesIn, offset: 0, index: 2)
                computeEncoder.setBuffer(indicesOut, offset: 0, index: 3)
                computeEncoder.setBuffer(offsetBuffer, offset: 0, index: 4)
                computeEncoder.setBytes(&passValue, length: MemoryLayout<UInt32>.stride, index: 5)
                var count = UInt32(splatCount)
                computeEncoder.setBytes(&count, length: MemoryLayout<UInt32>.stride, index: 6)
                computeEncoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
                computeEncoder.endEncoding()
            }

            // Swap buffers for next pass
            swap(&keysIn, &keysOut)
            swap(&indicesIn, &indicesOut)
        }

        // After 4 passes with swaps, the sorted data is in the "In" buffers
        // (swap happens after each pass, so: pass0→Out, pass1→In, pass2→Out, pass3→In)
        // Copy final sorted data back to original buffers
        if let blitEncoder = commandBuffer.makeBlitCommandEncoder() {
            blitEncoder.copy(from: keysIn, sourceOffset: 0, to: mortonCodeBuffer, destinationOffset: 0, size: splatCount * MemoryLayout<UInt32>.stride)
            blitEncoder.copy(from: indicesIn, sourceOffset: 0, to: sortedIndicesBuffer, destinationOffset: 0, size: splatCount * MemoryLayout<UInt32>.stride)
            blitEncoder.endEncoding()
        }
    }
    
    // MARK: - Simple CPU Distance Sorting
    
    
    /// Adaptive quality scaling based on thermal state and battery level
    private func getAdaptiveQuality() -> Int {
        guard adaptiveQualityEnabled else { return maxSplatCount }
        
        // Check thermal state
        let thermalState = ProcessInfo.processInfo.thermalState
        var qualityMultiplier: Float = 1.0
        
        switch thermalState {
        case .nominal:
            qualityMultiplier = 1.0
        case .fair:
            qualityMultiplier = 0.8
        case .serious:
            qualityMultiplier = 0.6
        case .critical:
            qualityMultiplier = 0.3
        @unknown default:
            qualityMultiplier = 1.0
        }
        
        // Check battery level if available
        let device = UIDevice.current
        if device.batteryState != .unknown {
            let batteryLevel = device.batteryLevel
            if batteryLevel < 0.2 {
                qualityMultiplier *= 0.5 // 50% quality when battery low
            } else if batteryLevel < 0.4 {
                qualityMultiplier *= 0.75 // 75% quality when battery medium
            }
        }
        
        return max(100, Int(Float(maxSplatCount) * qualityMultiplier))
    }
    
    
    /// Simple CPU sort ALL splats by distance - replaces GPU radix sort
    private func performCPUSort() {
        let startTime = CACurrentMediaTime()
        
        // Sort ALL splats by distance from camera (front-to-back)
        let indexedDistances = splats.enumerated().map { (index, splat) -> (Int, Float) in
            let delta = splat.position - cameraPosition
            let distanceSquared = dot(delta, delta)
            return (index, distanceSquared)
        }
        
        let sorted = indexedDistances.sorted { $0.1 < $1.1 }
        let sortedIndices = sorted.map { UInt32($0.0) }
        
        lastCPUSortTime = CACurrentMediaTime() - startTime
        
        // Put CPU-sorted indices into the ORIGINAL sortedIndicesBuffer
        let bufferPointer = sortedIndicesBuffer.contents().bindMemory(to: UInt32.self, capacity: splats.count)
        for (index, splatIndex) in sortedIndices.enumerated() {
            bufferPointer[index] = splatIndex
        }
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
        let viewProjectionMatrix = projectionMatrix * viewMatrix

        let viewUniforms = ViewUniforms(
            viewMatrix: viewMatrix,
            projectionMatrix: projectionMatrix,
            viewProjectionMatrix: viewProjectionMatrix,
            cameraPosition: cameraPosition,
            screenSize: SIMD2<Float>(Float(drawableSize.width), Float(drawableSize.height)),
            time: time
        )

        let viewUniformsPtr = viewUniformsBuffer.contents().bindMemory(to: ViewUniforms.self, capacity: 1)
        viewUniformsPtr[0] = viewUniforms

        // === ENERGY-EFFICIENT PIPELINE SELECTION ===
        if useHybridSorting {
            // Simple CPU distance sorting (70% energy savings)
            renderWithHybridPipeline(commandBuffer: commandBuffer)
        } else {
            // Original GPU-intensive pipeline (for comparison/debugging)
            renderWithOptimizedPipeline(commandBuffer: commandBuffer)
        }

        // Fullscreen Rendering Pass
        renderPassDescriptor.colorAttachments[0].clearColor = MTLClearColor(red: 0.0, green: 0.0, blue: 0.0, alpha: 1.0)
        renderPassDescriptor.colorAttachments[0].loadAction = .clear
        
        if let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) {
            // Select appropriate pipeline based on debug mode
            switch cameraDebugMode {
            case .tileHeatmap:
                // Use enhanced heatmap visualization
                if let heatmapPipeline = debugHeatmapPipeline {
                    renderEncoder.setRenderPipelineState(heatmapPipeline)
                } else {
                    renderEncoder.setRenderPipelineState(debugPipeline)
                }
            case .splatAssignment:
                // Use workload visualization
                if let workloadPipeline = debugWorkloadPipeline {
                    renderEncoder.setRenderPipelineState(workloadPipeline)
                } else {
                    renderEncoder.setRenderPipelineState(debugPipeline)
                }
            case .cullingStats:
                // Use culling efficiency visualization
                if let cullingPipeline = debugCullingPipeline {
                    renderEncoder.setRenderPipelineState(cullingPipeline)
                } else {
                    renderEncoder.setRenderPipelineState(debugPipeline)
                }
            case .tileBoundaries:
                // Use standard debug pipeline (shows basic tiles)
                renderEncoder.setRenderPipelineState(debugPipeline)
            default:
                if showDebugTiles {
                    renderEncoder.setRenderPipelineState(debugPipeline)
                } else {
                    renderEncoder.setRenderPipelineState(renderPipeline)
                }
            }

            renderEncoder.setFragmentBuffer(preprocessedSplatBuffer, offset: 0, index: 0)
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

        // Increment frame counter for debug output
        frameCount += 1
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
        
        var info = """
        Debug Mode: \(cameraDebugMode.description)
        Camera Position: (\(String(format: "%.2f", cameraPosition.x)), \(String(format: "%.2f", cameraPosition.y)), \(String(format: "%.2f", cameraPosition.z)))
        Camera Target: (\(String(format: "%.2f", cameraTarget.x)), \(String(format: "%.2f", cameraTarget.y)), \(String(format: "%.2f", cameraTarget.z)))
        Distance: \(String(format: "%.2f", cameraDistance))
        Azimuth: \(String(format: "%.1f", azimuthDegrees))°
        Elevation: \(String(format: "%.1f", elevationDegrees))°
        Trail Points: \(cameraTrail.count)
        Splats: \(splats.count)
        """
        
        // Add tile performance information for tile visualization modes
        if cameraDebugMode == .tileHeatmap ||
           cameraDebugMode == .tileBoundaries ||
           cameraDebugMode == .splatAssignment ||
           cameraDebugMode == .cullingStats {
            info += "\n" + getTilePerformanceInfo()
            info += "\n" + getVisualizationGuide()
        }
        
        return info
    }
    
    private func getVisualizationGuide() -> String {
        switch cameraDebugMode {
        case .tileHeatmap:
            return """

            Heatmap Color Guide:
            Black → Blue → Cyan → Yellow → Red
            (0 splats) → (max splats: \(tileUniforms.maxSplatsPerTile))
            White borders = Tiles at max capacity
            """
        case .splatAssignment:
            return """

            Workload Visualization:
            Vertical bars show GPU workload per tile
            Red borders = High workload (>80%)
            Color intensity = Computational complexity
            """
        case .cullingStats:
            return """

            Culling Efficiency:
            Green = Accepted splats
            Red = Rejected (culled) splats
            Blue = Culling effectiveness
            White line = Accept/Reject ratio
            """
        case .tileBoundaries:
            return """

            Tile Boundaries:
            Shows basic tile grid structure
            Each tile: 16x16 pixels
            """
        default:
            return ""
        }
    }

    private func getTilePerformanceInfo() -> String {
        guard let tileBuffer = tileBuffer else { return "" }

        let tilePtr = tileBuffer.contents().bindMemory(to: TileData.self, capacity: tileUniforms.totalTiles)
        var totalSplats: UInt32 = 0
        var totalRejected: UInt32 = 0
        var totalWorkload: UInt32 = 0
        var activeTiles = 0
        var maxedOutTiles = 0

        for i in 0..<tileUniforms.totalTiles {
            let tile = tilePtr[i]
            totalSplats += tile.count
            totalRejected += tile.rejectedCount
            totalWorkload += tile.workloadEstimate
            
            if tile.count > 0 || tile.rejectedCount > 0 {
                activeTiles += 1
            }
            if tile.count >= tileUniforms.maxSplatsPerTile {
                maxedOutTiles += 1
            }
        }

        let totalProcessed = totalSplats + totalRejected
        let cullingEfficiency = totalProcessed > 0 ? Float(totalRejected) / Float(totalProcessed) * 100.0 : 0.0
        let avgSplatsPerActiveTile = activeTiles > 0 ? Float(totalSplats) / Float(activeTiles) : 0.0

        return """

        Tile Performance:
        Total Tiles: \(tileUniforms.totalTiles)
        Active Tiles: \(activeTiles) (\(String(format: "%.1f", Float(activeTiles) / Float(tileUniforms.totalTiles) * 100.0))%)
        Maxed Out Tiles: \(maxedOutTiles)
        Assigned Splats: \(totalSplats)
        Rejected Splats: \(totalRejected)
        Avg Splats/Active Tile: \(String(format: "%.1f", avgSplatsPerActiveTile))
        Culling Efficiency: \(String(format: "%.1f", cullingEfficiency))%
        Total Workload: \(totalWorkload)
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
    
    // MARK: - Energy Management API
    
    /// Toggle between energy-efficient hybrid sorting and GPU-intensive sorting
    func setHybridSorting(_ enabled: Bool) {
        useHybridSorting = enabled
        print("Hybrid sorting \(enabled ? "enabled" : "disabled") - \(enabled ? "~70% less GPU power" : "maximum performance")")
    }
    
    /// Enable/disable adaptive quality scaling based on thermal state and battery
    func setAdaptiveQuality(_ enabled: Bool) {
        adaptiveQualityEnabled = enabled
        print("Adaptive quality \(enabled ? "enabled" : "disabled")")
    }
    
    /// Get current energy performance metrics
    func getEnergyMetrics() -> (cpuSortTime: Double, gpuSortTime: Double, hybridEnabled: Bool) {
        return (lastCPUSortTime * 1000, lastGPUSortTime * 1000, useHybridSorting)
    }
    
    /// Get detailed hybrid pipeline performance metrics
    func getHybridPerformanceMetrics() -> (
        gpuCullTime: Double,
        cpuSortTime: Double, 
        hybridPipelineTime: Double,
        cullingEfficiency: Float,
        energySavings: Float,
        visibleSplatRatio: Float
    ) {
        let avgCullingEfficiency = cullingEfficiencyHistory.isEmpty ? 0.0 : 
            cullingEfficiencyHistory.reduce(0, +) / Float(cullingEfficiencyHistory.count)
        
        let energySavings = lastGPUSortTime > 0 ? 
            Float((lastGPUSortTime - lastHybridPipelineTime) / lastGPUSortTime * 100) : 0.0
        
        let visibleRatio = totalSplatsProcessed > 0 ? 
            Float(visibleSplatsProcessed) / Float(totalSplatsProcessed) : 0.0
        
        return (
            gpuCullTime: lastGPUCullTime * 1000,
            cpuSortTime: lastCPUSortTime * 1000,
            hybridPipelineTime: lastHybridPipelineTime * 1000,
            cullingEfficiency: avgCullingEfficiency,
            energySavings: energySavings,
            visibleSplatRatio: visibleRatio
        )
    }
    
    // MARK: - Pipeline Functions

    /// Simple CPU sorting pipeline - replaces GPU radix sort with CPU distance sort
    private func renderWithHybridPipeline(commandBuffer: MTLCommandBuffer) {
        let pipelineStartTime = CACurrentMediaTime()
        
        // PHASE 1: CPU Sort ALL splats by distance (replaces GPU Morton + Radix sort)
        performCPUSort()
        
        // PHASE 2: Preprocess Splats
        if let computeEncoder = commandBuffer.makeComputeCommandEncoder() {
            computeEncoder.setComputePipelineState(preprocessSplatsPipeline)
            computeEncoder.setBuffer(splatBuffer, offset: 0, index: 0)
            computeEncoder.setBuffer(preprocessedSplatBuffer, offset: 0, index: 1)
            computeEncoder.setBuffer(viewUniformsBuffer, offset: 0, index: 2)
            computeEncoder.setBuffer(splatCountBuffer, offset: 0, index: 3)

            let threadsPerGrid = MTLSize(width: splats.count, height: 1, depth: 1)
            let threadsPerThreadgroup = MTLSize(width: 256, height: 1, depth: 1)
            computeEncoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
            computeEncoder.endEncoding()
        }

        // PHASE 3: Clear Tiles
        if let computeEncoder = commandBuffer.makeComputeCommandEncoder() {
            computeEncoder.setComputePipelineState(clearTilesPipeline)
            computeEncoder.setBuffer(tileBuffer, offset: 0, index: 0)
            computeEncoder.setBuffer(tileUniformsBuffer, offset: 0, index: 1)

            let totalTiles = tileUniforms.totalTiles
            let threadsPerThreadgroup = MTLSize(width: 64, height: 1, depth: 1)
            let threadgroupsPerGrid = MTLSize(
                width: (totalTiles + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width,
                height: 1,
                depth: 1
            )

            computeEncoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
            computeEncoder.endEncoding()
        }

        // PHASE 4: Build Tiles (using BASIC shader - no sorting yet)
        if let computeEncoder = commandBuffer.makeComputeCommandEncoder() {
            computeEncoder.setComputePipelineState(buildTilesPipeline)
            computeEncoder.setBuffer(splatBuffer, offset: 0, index: 0)
            computeEncoder.setBuffer(tileBuffer, offset: 0, index: 1)
            computeEncoder.setBuffer(viewUniformsBuffer, offset: 0, index: 2)
            computeEncoder.setBuffer(tileUniformsBuffer, offset: 0, index: 3)
            computeEncoder.setBuffer(splatCountBuffer, offset: 0, index: 4)

            let threadsPerThreadgroup = MTLSize(width: 8, height: 8, depth: 1)
            let threadgroupsPerGrid = MTLSize(
                width: (Int(tileUniforms.tilesPerRow) + 7) / 8,
                height: (Int(tileUniforms.tilesPerColumn) + 7) / 8,
                depth: 1
            )
            computeEncoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
            computeEncoder.endEncoding()
        }
        
        lastHybridPipelineTime = CACurrentMediaTime() - pipelineStartTime
        
        if frameCount % 60 == 0 {
            print("🚀 Simple CPU Pipeline: \(String(format: "%.2f", lastHybridPipelineTime * 1000))ms")
            print("   └─ CPU Sort: \(String(format: "%.2f", lastCPUSortTime * 1000))ms (\(splats.count) splats)")
        }
    }

/// Original GPU-intensive rendering pipeline (for comparison)
private func renderWithOptimizedPipeline(commandBuffer: MTLCommandBuffer) {
    let startTime = CACurrentMediaTime()
    
    // Determine visible splat count for this frame
    var visibleCount: UInt32 = 0

    if useGPUFrustumCulling {
        // PHASE 0a: Reset visible count to 0
        if let blitEncoder = commandBuffer.makeBlitCommandEncoder() {
            blitEncoder.fill(buffer: visibleSplatCountBuffer, range: 0..<visibleSplatCountBuffer.length, value: 0)
            blitEncoder.endEncoding()
        }

        // PHASE 0b: GPU Frustum Culling
        if let computeEncoder = commandBuffer.makeComputeCommandEncoder() {
            computeEncoder.setComputePipelineState(frustumCullPipeline)
            computeEncoder.setBuffer(splatBuffer, offset: 0, index: 0)
            computeEncoder.setBuffer(visibleSplatCountBuffer, offset: 0, index: 1)
            computeEncoder.setBuffer(visibleSplatIndicesBuffer, offset: 0, index: 2)
            computeEncoder.setBuffer(viewUniformsBuffer, offset: 0, index: 3)
            computeEncoder.setBuffer(splatCountBuffer, offset: 0, index: 4)

            let threadsPerGrid = MTLSize(width: splats.count, height: 1, depth: 1)
            let threadsPerThreadgroup = MTLSize(width: 256, height: 1, depth: 1)
            computeEncoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
            computeEncoder.endEncoding()
        }

        // Read back visible count
        let visibleCountPtr = visibleSplatCountBuffer.contents().bindMemory(to: UInt32.self, capacity: 1)
        visibleCount = visibleCountPtr[0]
    } else {
        visibleCount = UInt32(splats.count)
    }

    // PHASE 0c: Preprocess Splats
    if let computeEncoder = commandBuffer.makeComputeCommandEncoder() {
        computeEncoder.setComputePipelineState(preprocessSplatsPipeline)
        computeEncoder.setBuffer(splatBuffer, offset: 0, index: 0)
        computeEncoder.setBuffer(preprocessedSplatBuffer, offset: 0, index: 1)
        computeEncoder.setBuffer(viewUniformsBuffer, offset: 0, index: 2)
        computeEncoder.setBuffer(splatCountBuffer, offset: 0, index: 3)

        let threadsPerGrid = MTLSize(width: splats.count, height: 1, depth: 1)
        let threadsPerThreadgroup = MTLSize(width: 256, height: 1, depth: 1)
        computeEncoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        computeEncoder.endEncoding()
    }

    // PHASE 1: Compute Morton Codes (GPU-intensive)
    if useGPUFrustumCulling && visibleCount > 0 {
        if let computeEncoder = commandBuffer.makeComputeCommandEncoder() {
            computeEncoder.setComputePipelineState(mortonCodeVisiblePipeline)
            computeEncoder.setBuffer(splatBuffer, offset: 0, index: 0)
            computeEncoder.setBuffer(visibleSplatIndicesBuffer, offset: 0, index: 1)
            computeEncoder.setBuffer(mortonCodeBuffer, offset: 0, index: 2)
            computeEncoder.setBuffer(sortedIndicesBuffer, offset: 0, index: 3)
            computeEncoder.setBuffer(viewUniformsBuffer, offset: 0, index: 4)
            var visibleCountCopy = visibleCount
            computeEncoder.setBytes(&visibleCountCopy, length: MemoryLayout<UInt32>.stride, index: 5)

            let threadsPerGrid = MTLSize(width: Int(visibleCount), height: 1, depth: 1)
            let threadsPerThreadgroup = MTLSize(width: 256, height: 1, depth: 1)
            computeEncoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
            computeEncoder.endEncoding()
        }
    } else {
        if let computeEncoder = commandBuffer.makeComputeCommandEncoder() {
            computeEncoder.setComputePipelineState(mortonCodePipeline)
            computeEncoder.setBuffer(splatBuffer, offset: 0, index: 0)
            computeEncoder.setBuffer(mortonCodeBuffer, offset: 0, index: 1)
            computeEncoder.setBuffer(sortedIndicesBuffer, offset: 0, index: 2)
            computeEncoder.setBuffer(viewUniformsBuffer, offset: 0, index: 3)
            computeEncoder.setBuffer(splatCountBuffer, offset: 0, index: 4)

            let threadsPerGrid = MTLSize(width: splats.count, height: 1, depth: 1)
            let threadsPerThreadgroup = MTLSize(width: 256, height: 1, depth: 1)
            computeEncoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
            computeEncoder.endEncoding()
        }
    }

    // PHASE 2: GPU Radix Sort (VERY energy intensive)
    let sortCount = useGPUFrustumCulling ? Int(visibleCount) : splats.count
    if sortCount > 0 {
        performRadixSort(commandBuffer: commandBuffer, splatCount: sortCount)
    }

    // PHASE 3: Clear Tiles
    if let computeEncoder = commandBuffer.makeComputeCommandEncoder() {
        computeEncoder.setComputePipelineState(clearTilesPipeline)
        computeEncoder.setBuffer(tileBuffer, offset: 0, index: 0)
        computeEncoder.setBuffer(tileUniformsBuffer, offset: 0, index: 1)

        let totalTiles = tileUniforms.totalTiles
        let threadsPerThreadgroup = MTLSize(width: 64, height: 1, depth: 1)
        let threadgroupsPerGrid = MTLSize(
            width: (totalTiles + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width,
            height: 1,
            depth: 1
        )

        computeEncoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        computeEncoder.endEncoding()
    }

    // PHASE 4: Build Tiles
    if useGPUFrustumCulling && visibleCount > 0 {
        if let computeEncoder = commandBuffer.makeComputeCommandEncoder() {
            computeEncoder.setComputePipelineState(buildTilesVisiblePipeline)
            computeEncoder.setBuffer(splatBuffer, offset: 0, index: 0)
            computeEncoder.setBuffer(visibleSplatIndicesBuffer, offset: 0, index: 1)
            computeEncoder.setBuffer(sortedIndicesBuffer, offset: 0, index: 2)
            computeEncoder.setBuffer(mortonCodeBuffer, offset: 0, index: 3)
            computeEncoder.setBuffer(tileBuffer, offset: 0, index: 4)
            computeEncoder.setBuffer(viewUniformsBuffer, offset: 0, index: 5)
            computeEncoder.setBuffer(tileUniformsBuffer, offset: 0, index: 6)
            var visibleCountCopy = visibleCount
            computeEncoder.setBytes(&visibleCountCopy, length: MemoryLayout<UInt32>.stride, index: 7)

            let threadsPerThreadgroup = MTLSize(width: 8, height: 8, depth: 1)
            let threadgroupsPerGrid = MTLSize(
                width: (Int(tileUniforms.tilesPerRow) + 7) / 8,
                height: (Int(tileUniforms.tilesPerColumn) + 7) / 8,
                depth: 1
            )
            computeEncoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
            computeEncoder.endEncoding()
        }
    } else {
        if let computeEncoder = commandBuffer.makeComputeCommandEncoder() {
            computeEncoder.setComputePipelineState(buildTilesOptimizedPipeline)
            computeEncoder.setBuffer(splatBuffer, offset: 0, index: 0)
            computeEncoder.setBuffer(sortedIndicesBuffer, offset: 0, index: 1)
            computeEncoder.setBuffer(mortonCodeBuffer, offset: 0, index: 2)
            computeEncoder.setBuffer(tileBuffer, offset: 0, index: 3)
            computeEncoder.setBuffer(viewUniformsBuffer, offset: 0, index: 4)
            computeEncoder.setBuffer(tileUniformsBuffer, offset: 0, index: 5)
            computeEncoder.setBuffer(splatCountBuffer, offset: 0, index: 6)

            let threadsPerThreadgroup = MTLSize(width: 8, height: 8, depth: 1)
            let threadgroupsPerGrid = MTLSize(
                width: (Int(tileUniforms.tilesPerRow) + 7) / 8,
                height: (Int(tileUniforms.tilesPerColumn) + 7) / 8,
                depth: 1
            )
            computeEncoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
            computeEncoder.endEncoding()
        }
    }
    
    // PHASE 5: Tile Depth Sorting (NEW - stability fix for dense areas)
    // Sort splats within each tile by depth to eliminate flickering
    // while preserving Morton code spatial optimization benefits
    let depthSortStart = CACurrentMediaTime()
    if let computeEncoder = commandBuffer.makeComputeCommandEncoder() {
        computeEncoder.setComputePipelineState(sortTileDepthPipeline)
        computeEncoder.setBuffer(tileBuffer, offset: 0, index: 0)
        computeEncoder.setBuffer(splatBuffer, offset: 0, index: 1)
        computeEncoder.setBuffer(viewUniformsBuffer, offset: 0, index: 2)
        computeEncoder.setBuffer(tileUniformsBuffer, offset: 0, index: 3)
        
        // Process each tile independently for parallel execution
        let threadsPerThreadgroup = MTLSize(width: 64, height: 1, depth: 1)
        let threadgroupsPerGrid = MTLSize(
            width: (tileUniforms.totalTiles + 63) / 64,
            height: 1, 
            depth: 1
        )
        computeEncoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        computeEncoder.endEncoding()
        
        // Debug: Print tile depth sorting timing
        if frameCount % 60 == 0 {
            let depthSortTime = CACurrentMediaTime() - depthSortStart
            print("🔧 Tile Depth Sorting: \(String(format: "%.3f", depthSortTime * 1000))ms (\(tileUniforms.totalTiles) tiles)")
        }
    }
    
    lastGPUSortTime = CACurrentMediaTime() - startTime
}

}

// Camera utility functions
func createLookAtMatrix(eye: SIMD3<Float>, target: SIMD3<Float>, up: SIMD3<Float>) -> simd_float4x4 {
    // Compute camera basis vectors
    let forward = normalize(target - eye)  // Direction camera is looking
    let right = normalize(cross(forward, up))  // Right vector
    let realUp = cross(right, forward)  // Recomputed up vector (orthogonal)

    // Verify basis vectors are orthonormal
    let rightLen = length(right)
    let upLen = length(realUp)
    let forwardLen = length(forward)

    // Check for degenerate cases
    if rightLen < 0.01 || upLen < 0.01 || forwardLen < 0.01 {
        print("WARNING: Degenerate camera basis vectors!")
        print("  right length: \(rightLen)")
        print("  up length: \(upLen)")
        print("  forward length: \(forwardLen)")
    }

    // Build view matrix in column-major format
    // Column 0: right, Column 1: up, Column 2: -forward, Column 3: translation
    let viewMatrix = simd_float4x4(
        SIMD4<Float>(right.x, realUp.x, -forward.x, 0),      // Column 0
        SIMD4<Float>(right.y, realUp.y, -forward.y, 0),      // Column 1
        SIMD4<Float>(right.z, realUp.z, -forward.z, 0),      // Column 2
        SIMD4<Float>(-dot(right, eye), -dot(realUp, eye), dot(forward, eye), 1)  // Column 3
    )

    return viewMatrix
}

func createPerspectiveMatrix(fovy: Float, aspect: Float, near: Float, far: Float) -> simd_float4x4 {
    // Standard OpenGL perspective matrix (column-major)
    // fovy = field of view in radians
    // aspect = width / height
    let f = 1.0 / tan(fovy * 0.5)

    // Column-major construction:
    // Each SIMD4 is a column: [col0, col1, col2, col3]
    return simd_float4x4(
        SIMD4<Float>(f / aspect, 0, 0, 0),                         // Column 0
        SIMD4<Float>(0, f, 0, 0),                                  // Column 1
        SIMD4<Float>(0, 0, -(far + near) / (far - near), -1),     // Column 2
        SIMD4<Float>(0, 0, -(2 * far * near) / (far - near), 0)   // Column 3
    )
}

