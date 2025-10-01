import Foundation
import simd

struct GaussianSplat {
    let position: SIMD3<Float>
    let scale: SIMD3<Float>
    let rotation: simd_quatf
    let opacity: Float
    let color: SIMD3<Float>
    let sphericalHarmonics: [Float]?
    
    init(position: SIMD3<Float>, 
         scale: SIMD3<Float>, 
         rotation: simd_quatf, 
         opacity: Float, 
         color: SIMD3<Float>, 
         sphericalHarmonics: [Float]? = nil) {
        self.position = position
        self.scale = scale
        self.rotation = rotation
        self.opacity = Swift.min(Swift.max(opacity, 0.0), 1.0)
        self.color = color
        self.sphericalHarmonics = sphericalHarmonics
    }
    
    var boundingBox: (min: SIMD3<Float>, max: SIMD3<Float>) {
        let halfScale = scale * 0.5
        return (
            min: position - halfScale,
            max: position + halfScale
        )
    }
    
    var memorySize: Int {
        let baseSize = MemoryLayout<SIMD3<Float>>.size * 3 + 
                      MemoryLayout<simd_quatf>.size + 
                      MemoryLayout<Float>.size
        let shSize = sphericalHarmonics?.count ?? 0
        return baseSize + (shSize * MemoryLayout<Float>.size)
    }
}

extension GaussianSplat: Equatable {
    static func == (lhs: GaussianSplat, rhs: GaussianSplat) -> Bool {
        return lhs.position == rhs.position &&
               lhs.scale == rhs.scale &&
               lhs.rotation.vector == rhs.rotation.vector &&
               lhs.opacity == rhs.opacity &&
               lhs.color == rhs.color &&
               lhs.sphericalHarmonics == rhs.sphericalHarmonics
    }
}

extension GaussianSplat: CustomStringConvertible {
    var description: String {
        return "GaussianSplat(pos: \(position), scale: \(scale), rot: \(rotation), opacity: \(opacity), color: \(color))"
    }
}