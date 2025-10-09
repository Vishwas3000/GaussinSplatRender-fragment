import Foundation
import simd

struct GaussianSplat {
    let position: SIMD3<Float>
    let covariance3D: SIMD4<Float> // Store 3D covariance as packed data: [xx, xy, xz, yy]
    let covariance3D_B: SIMD2<Float> // Store remaining elements: [yz, zz]
    let color: SIMD3<UInt8>
    let opacity: UInt8
    let depth: Float
    
    init(position: SIMD3<Float>, 
         covariance3D: simd_float3x3,
         color: SIMD3<Float>, 
         opacity: Float,
         depth: Float) {
        self.position = position
        
        // Pack 3x3 covariance matrix efficiently
        // Σ = [xx xy xz]  ->  covariance3D = [xx, xy, xz, yy]
        //     [xy yy yz]      covariance3D_B = [yz, zz]
        //     [xz yz zz]
        self.covariance3D = SIMD4<Float>(
            covariance3D[0][0], // xx
            covariance3D[0][1], // xy
            covariance3D[0][2], // xz
            covariance3D[1][1]  // yy
        )
        self.covariance3D_B = SIMD2<Float>(
            covariance3D[1][2], // yz
            covariance3D[2][2]  // zz
        )
        
        // Convert to 8-bit color
        self.color = SIMD3<UInt8>(
            UInt8(clamp(color.x * 255.0, 0.0, 255.0)),
            UInt8(clamp(color.y * 255.0, 0.0, 255.0)),
            UInt8(clamp(color.z * 255.0, 0.0, 255.0))
        )
        
        self.opacity = UInt8(clamp(opacity * 255.0, 0.0, 255.0))
        self.depth = depth
    }
    
    var floatColor: SIMD3<Float> {
        return SIMD3<Float>(
            Float(color.x) / 255.0,
            Float(color.y) / 255.0,
            Float(color.z) / 255.0
        )
    }
    
    var floatOpacity: Float {
        return Float(opacity) / 255.0
    }
    
    var covariance3DMatrix: simd_float3x3 {
        return simd_float3x3(
            SIMD3<Float>(covariance3D.x, covariance3D.y, covariance3D.z), // [xx, xy, xz]
            SIMD3<Float>(covariance3D.y, covariance3D.w, covariance3D_B.x), // [xy, yy, yz]
            SIMD3<Float>(covariance3D.z, covariance3D_B.x, covariance3D_B.y)  // [xz, yz, zz]
        )
    }
    
    static var stride: Int {
        return MemoryLayout<GaussianSplat>.stride
    }
}

struct PreprocessedSplat {
    var screenCenter: SIMD2<Float>        // Pixel coordinates of splat center
    var invCov2D_row0: SIMD3<Float>       // Inverted 2D covariance matrix row 0 [a, b] + padding
    var invCov2D_row1: SIMD2<Float>       // Inverted 2D covariance matrix row 1 [c, d]
    var color: SIMD3<UInt8>               // RGB color
    var opacity: UInt8                    // Opacity
    var depth: Float                      // For sorting
    var isValid: UInt32                   // 1 if visible, 0 if culled

    static var stride: Int {
        return MemoryLayout<PreprocessedSplat>.stride
    }
}

struct TileData {
    var count: UInt32           // Number of splats assigned to this tile
    var rejectedCount: UInt32   // Number of splats rejected (culled) from this tile
    var workloadEstimate: UInt32 // Estimated GPU workload (for performance visualization)
    var maxDepth: UInt32        // Maximum depth of splats in this tile (for debugging)
    var splatIndices: (UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32,
                       UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32,
                       UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32,
                       UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32,
                       UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32,
                       UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32,
                       UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32,
                       UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32) // Array of splat indices (64 elements)

    init() {
        self.count = 0
        self.rejectedCount = 0
        self.workloadEstimate = 0
        self.maxDepth = 0
        self.splatIndices = (0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0)
    }
}

struct ViewUniforms {
    let viewMatrix: simd_float4x4
    let projectionMatrix: simd_float4x4
    let viewProjectionMatrix: simd_float4x4
    let cameraPosition: SIMD3<Float>
    let time: Float
    let screenSize: SIMD2<Float>
    let padding: SIMD2<Float>
    
    init(viewMatrix: simd_float4x4,
         projectionMatrix: simd_float4x4,
         viewProjectionMatrix: simd_float4x4,
         cameraPosition: SIMD3<Float>,
         screenSize: SIMD2<Float>,
         time: Float = 0) {
        self.viewMatrix = viewMatrix
        self.projectionMatrix = projectionMatrix
        self.viewProjectionMatrix = viewProjectionMatrix
        self.cameraPosition = cameraPosition
        self.time = time
        self.screenSize = screenSize
        self.padding = SIMD2<Float>(0, 0)
    }

    // Convenience initializer that computes viewProjectionMatrix
    init(viewMatrix: simd_float4x4,
         projectionMatrix: simd_float4x4,
         cameraPosition: SIMD3<Float>,
         screenSize: SIMD2<Float>,
         time: Float = 0) {
        self.init(
            viewMatrix: viewMatrix,
            projectionMatrix: projectionMatrix,
            viewProjectionMatrix: projectionMatrix * viewMatrix,
            cameraPosition: cameraPosition,
            screenSize: screenSize,
            time: time
        )
    }
}

struct TileUniforms {
    let tilesPerRow: UInt32
    let tilesPerColumn: UInt32
    let tileSize: UInt32
    let maxSplatsPerTile: UInt32
    
    init(screenWidth: Int, screenHeight: Int, tileSize: Int = 16) {
        self.tileSize = UInt32(tileSize)
        self.tilesPerRow = UInt32((screenWidth + tileSize - 1) / tileSize)
        self.tilesPerColumn = UInt32((screenHeight + tileSize - 1) / tileSize)
        self.maxSplatsPerTile = 64 // Matches TileData splatIndices array size
    }
    
    var totalTiles: Int {
        return Int(tilesPerRow * tilesPerColumn)
    }
}

extension GaussianSplat: Comparable {
    static func < (lhs: GaussianSplat, rhs: GaussianSplat) -> Bool {
        return lhs.depth > rhs.depth // Sort back-to-front for alpha blending
    }
}

// Helper functions for matrix operations
func clamp<T: Comparable>(_ value: T, _ min: T, _ max: T) -> T {
    return Swift.max(min, Swift.min(max, value))
}

func createRandom3DCovariance(scale: Float = 1.0) -> simd_float3x3 {
    // Generate random 3D rotation using Euler angles
    let rotX = Float.random(in: 0...(2 * Float.pi))
    let rotY = Float.random(in: 0...(2 * Float.pi))
    let rotZ = Float.random(in: 0...(2 * Float.pi))
    
    // Create rotation matrices for each axis
    let cosX = cos(rotX), sinX = sin(rotX)
    let cosY = cos(rotY), sinY = sin(rotY)
    let cosZ = cos(rotZ), sinZ = sin(rotZ)
    
    let rotationX = simd_float3x3(
        SIMD3<Float>(1, 0, 0),
        SIMD3<Float>(0, cosX, -sinX),
        SIMD3<Float>(0, sinX, cosX)
    )
    
    let rotationY = simd_float3x3(
        SIMD3<Float>(cosY, 0, sinY),
        SIMD3<Float>(0, 1, 0),
        SIMD3<Float>(-sinY, 0, cosY)
    )
    
    let rotationZ = simd_float3x3(
        SIMD3<Float>(cosZ, -sinZ, 0),
        SIMD3<Float>(sinZ, cosZ, 0),
        SIMD3<Float>(0, 0, 1)
    )
    
    // Combine rotations: R = Rz * Ry * Rx
    let rotation = rotationZ * rotationY * rotationX
    
    // Create 3D scale matrix with dramatic variations
    let baseScale = max(0.1, scale)
    let scaleX = Float.random(in: 0.05...baseScale * 3.0)
    let scaleY = Float.random(in: 0.05...baseScale * 3.0)
    let scaleZ = Float.random(in: 0.05...baseScale * 3.0)
    
    // Sometimes create very elongated splats in any direction (30% chance)
    let elongationFactor = Float.random(in: 0.1...8.0)
    let finalScaleX = scaleX * (Float.random(in: 0...1) < 0.3 ? elongationFactor : 1.0)
    let finalScaleY = scaleY * (Float.random(in: 0...1) < 0.3 ? elongationFactor : 1.0)
    let finalScaleZ = scaleZ * (Float.random(in: 0...1) < 0.3 ? elongationFactor : 1.0)
    
    let scaleMatrix = simd_float3x3(
        SIMD3<Float>(finalScaleX, 0, 0),
        SIMD3<Float>(0, finalScaleY, 0),
        SIMD3<Float>(0, 0, finalScaleZ)
    )
    
    // Combine: R * S * R^T to create 3D covariance matrix
    let temp = rotation * scaleMatrix
    return temp * rotation.transpose
}