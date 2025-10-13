import Foundation
import simd

class GaussianSplatGenerator {
    static func generateRandomSplats(count: Int, 
                                   boundingBox: (min: SIMD3<Float>, max: SIMD3<Float>) = 
                                   (min: SIMD3<Float>(-3, -3, -3), max: SIMD3<Float>(3, 3, 3)),
                                   viewMatrix: simd_float4x4,
                                   projectionMatrix: simd_float4x4,
                                   scaleMultiplier: Float = 1.0) -> [GaussianSplat] {
        
        var splats: [GaussianSplat] = []
        splats.reserveCapacity(count)
        
        let boxSize = boundingBox.max - boundingBox.min
        
        for _ in 0..<count {
            // Random 3D position within bounding box
            let position = SIMD3<Float>(
                Float.random(in: 0...1) * boxSize.x + boundingBox.min.x,
                Float.random(in: 0...1) * boxSize.y + boundingBox.min.y,
                Float.random(in: 0...1) * boxSize.z + boundingBox.min.z
            )
            
            // Calculate depth for sorting
            let viewSpacePos = viewMatrix * SIMD4<Float>(position, 1.0)
            let depth = viewSpacePos.z
            
            // Random 3D covariance matrix with dramatic variety
            let scale = Float.random(in: 0.1...2.0) * scaleMultiplier // Apply scale multiplier
            let covariance3D = createRandom3DCovariance(scale: scale)
            
            // Generate vibrant random colors
            let hue = Float.random(in: 0...1)
            let saturation = Float.random(in: 0.6...1.0)
            let brightness = Float.random(in: 0.7...1.0)
            let color = hsb_to_rgb(hue: hue, saturation: saturation, brightness: brightness)
            
            // Random opacity with bias toward visible
            let opacity = Float.random(in: 0.3...1.0)
            
            let splat = GaussianSplat(
                position: position,
                covariance3D: covariance3D,
                color: color,
                opacity: opacity,
                depth: depth
            )
            
            splats.append(splat)
        }
        
        // Sort back-to-front for proper alpha blending
        splats.sort()
        
        return splats
    }
    
    static func generateSplatClusters(clusterCount: Int, 
                                    splatsPerCluster: Int,
                                    boundingBox: (min: SIMD3<Float>, max: SIMD3<Float>) = 
                                    (min: SIMD3<Float>(-4, -4, -4), max: SIMD3<Float>(4, 4, 4)),
                                    viewMatrix: simd_float4x4,
                                    projectionMatrix: simd_float4x4,
                                    scaleMultiplier: Float = 1.0) -> [GaussianSplat] {
        
        var splats: [GaussianSplat] = []
        let totalCount = clusterCount * splatsPerCluster
        splats.reserveCapacity(totalCount)
        
        let boxSize = boundingBox.max - boundingBox.min
        
        for cluster in 0..<clusterCount {
            // Random cluster center
            let clusterCenter = SIMD3<Float>(
                Float.random(in: 0...1) * boxSize.x + boundingBox.min.x,
                Float.random(in: 0...1) * boxSize.y + boundingBox.min.y,
                Float.random(in: 0...1) * boxSize.z + boundingBox.min.z
            )
            
            // Random cluster color theme
            let baseHue = Float.random(in: 0...1)
            let clusterRadius = Float.random(in: 0.5...1.5)
            
            for _ in 0..<splatsPerCluster {
                // Position within cluster radius
                let offset = SIMD3<Float>(
                    Float.random(in: -1...1),
                    Float.random(in: -1...1),
                    Float.random(in: -1...1)
                ) * clusterRadius
                
                let position = clusterCenter + offset
                
                // Calculate depth
                let viewSpacePos = viewMatrix * SIMD4<Float>(position, 1.0)
                let depth = viewSpacePos.z
                
                // 3D Covariance with dramatic variety based on distance from cluster center
                let distanceFromCenter = length(offset)
                let scale = (0.3 + (distanceFromCenter / clusterRadius) * 1.5) * scaleMultiplier // Apply scale multiplier
                let covariance3D = createRandom3DCovariance(scale: scale)
                
                // Color variation around cluster theme
                let hueVariation = Float.random(in: -0.1...0.1)
                let hue = fmod(baseHue + hueVariation + 1.0, 1.0)
                let saturation = Float.random(in: 0.7...1.0)
                let brightness = Float.random(in: 0.8...1.0)
                let color = hsb_to_rgb(hue: hue, saturation: saturation, brightness: brightness)
                
                let opacity = Float.random(in: 0.4...0.9)
                
                let splat = GaussianSplat(
                    position: position,
                    covariance3D: covariance3D,
                    color: color,
                    opacity: opacity,
                    depth: depth
                )
                
                splats.append(splat)
            }
        }
        
        // Sort for alpha blending
        splats.sort()
        
        return splats
    }
    
    static func generateSplatSphere(count: Int,
                                  center: SIMD3<Float> = SIMD3<Float>(0, 0, 0),
                                  radius: Float = 2.0,
                                  viewMatrix: simd_float4x4,
                                  projectionMatrix: simd_float4x4,
                                  scaleMultiplier: Float = 1.0) -> [GaussianSplat] {
        
        var splats: [GaussianSplat] = []
        splats.reserveCapacity(count)
        
        for i in 0..<count {
            // Generate random point on sphere surface
            let theta = Float.random(in: 0...(2 * Float.pi))
            let phi = acos(2 * Float.random(in: 0...1) - 1)
            
            // Add some radial variation
            let r = radius * Float.random(in: 0.7...1.3)
            
            let position = center + SIMD3<Float>(
                r * sin(phi) * cos(theta),
                r * sin(phi) * sin(theta),
                r * cos(phi)
            )
            
            // Calculate depth
            let viewSpacePos = viewMatrix * SIMD4<Float>(position, 1.0)
            let depth = viewSpacePos.z
            
            // 3D Covariance based on position on sphere
            let scale = (0.5 + (r / radius - 0.7) * 1.0) * scaleMultiplier  // Apply scale multiplier
            let covariance3D = createRandom3DCovariance(scale: scale)
            
            // Color based on position (creates nice gradients)
            let normalizedPos = normalize(position - center)
            let hue = (atan2(normalizedPos.z, normalizedPos.x) + Float.pi) / (2 * Float.pi)
            let saturation = 0.8 + 0.2 * normalizedPos.y
            let brightness = 0.7 + 0.3 * abs(normalizedPos.y)
            let color = hsb_to_rgb(hue: hue, saturation: saturation, brightness: brightness)
            
            let opacity = Float.random(in: 0.5...0.9)
            
            let splat = GaussianSplat(
                position: position,
                covariance3D: covariance3D,
                color: color,
                opacity: opacity,
                depth: depth
            )
            
            splats.append(splat)
        }
        
        splats.sort()
        return splats
    }
}

// Color conversion utilities
func hsb_to_rgb(hue: Float, saturation: Float, brightness: Float) -> SIMD3<Float> {
    let h = hue * 6.0
    let c = brightness * saturation
    let x = c * (1.0 - abs(fmod(h, 2.0) - 1.0))
    let m = brightness - c
    
    let rgb: SIMD3<Float>
    
    if h < 1.0 {
        rgb = SIMD3<Float>(c, x, 0)
    } else if h < 2.0 {
        rgb = SIMD3<Float>(x, c, 0)
    } else if h < 3.0 {
        rgb = SIMD3<Float>(0, c, x)
    } else if h < 4.0 {
        rgb = SIMD3<Float>(0, x, c)
    } else if h < 5.0 {
        rgb = SIMD3<Float>(x, 0, c)
    } else {
        rgb = SIMD3<Float>(c, 0, x)
    }
    
    return rgb + SIMD3<Float>(m, m, m)
}

extension GaussianSplatGenerator {
    static func updateSplatDepths(splats: inout [GaussianSplat], 
                                viewMatrix: simd_float4x4) {
        for i in 0..<splats.count {
            let viewSpacePos = viewMatrix * SIMD4<Float>(splats[i].position, 1.0)
            splats[i] = GaussianSplat(
                position: splats[i].position,
                covariance3D: splats[i].covariance3DMatrix,
                color: splats[i].floatColor,
                opacity: splats[i].floatOpacity,
                depth: viewSpacePos.z
            )
        }
    }
}
