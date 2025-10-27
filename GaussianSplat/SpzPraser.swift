//
//  SpzPraser.swift
//  GaussianSplat
//
//  Created by Trupthi P on 08/10/25.
//

import Foundation
import simd
import zlib


class SPZParser {
    
    enum SPZError: Error {
        case invalidFile
        case decompressionFailed
        case invalidHeader
        case unsupportedVersion
        case dataSizeMismatch
    }
    
    // MARK: - Header Structure
    struct GaussiansHeader {
        let magic: UInt32
        let version: UInt32
        let numPoints: UInt32
        let shDegree: UInt8
        let fractionalBits: UInt8
        let flags: UInt8
        let reserved: UInt8
        
        static func parse(from data: Data) -> GaussiansHeader? {
            guard data.count >= 16 else { return nil }
            
            var offset = 0
            let magic = data.withUnsafeBytes { $0.load(fromByteOffset: offset, as: UInt32.self) }
            offset += 4
            
            let version = data.withUnsafeBytes { $0.load(fromByteOffset: offset, as: UInt32.self) }
            offset += 4
            
            let numPoints = data.withUnsafeBytes { $0.load(fromByteOffset: offset, as: UInt32.self) }
            offset += 4
            
            let shDegree = data[offset]
            offset += 1
            
            let fractionalBits = data[offset]
            offset += 1
            
            let flags = data[offset]
            offset += 1
            
            let reserved = data[offset]
            
            return GaussiansHeader(
                magic: magic,
                version: version,
                numPoints: numPoints,
                shDegree: shDegree,
                fractionalBits: fractionalBits,
                flags: flags,
                reserved: reserved
            )
        }
        
        func validate() -> Bool {
            guard magic == 0x5053474E else {
                print(" Invalid magic number")
                return false
            }
            
            guard version == 2 || version == 3 else {
                print("Unsupported version: \(version)")
                return false
            }
            
            guard shDegree <= 3 else {
                print(" Invalid SH degree: \(shDegree)")
                return false
            }
            
            return true
        }
    }
    
    struct SplatData {
        var position: SIMD3<Float>
        
        var rotation: simd_quatf
        
        var scale: SIMD3<Float>
        
        var color: SIMD3<Float>
        var opacity: Float
        
        var depth: Float = 0
    }
    
    struct BoundingBox {
        var min: SIMD3<Float>
        var max: SIMD3<Float>
    }
    
    struct ParseResult {
        let splats: [SplatData]
        let bounds: BoundingBox
        let header: GaussiansHeader
    }
    
    
    static func parse(fileURL: URL) throws -> ParseResult {
        print(" SPZ PARSER - Direct Storage (No Packing)")
        
        let compressedData = try Data(contentsOf: fileURL)
        print(" Loaded compressed file: \(compressedData.count) bytes")
        
        // Verify GZIP signature
        guard compressedData.prefix(2).starts(with: [0x1F, 0x8B]) else {
            throw SPZError.invalidFile
        }
        
        // Decompress
        let decompressed = try decompressGZIP(compressedData)
        print("✓ Decompressed: \(decompressed.count) bytes")
        
        // Parse
        let result = try parseDecompressedData(decompressed)
        
        print("Successfully parsed \(result.splats.count) splats")
        
        return result
    }
    
    // MARK: - GZIP Decompression
    
    private static func decompressGZIP(_ data: Data) throws -> Data {
        let estimatedSize = data.count * 15
        var outputBuffer = Data(count: estimatedSize)
        var actualSize: Int = 0
        
        let result = data.withUnsafeBytes { srcPtr -> Int32 in
            outputBuffer.withUnsafeMutableBytes { destPtr -> Int32 in
                var stream = z_stream()
                stream.next_in = UnsafeMutablePointer(mutating: srcPtr.baseAddress?.assumingMemoryBound(to: UInt8.self))
                stream.avail_in = UInt32(data.count)
                stream.next_out = destPtr.baseAddress?.assumingMemoryBound(to: UInt8.self)
                stream.avail_out = UInt32(estimatedSize)
                
                var status = inflateInit2_(&stream, 15 + 16, ZLIB_VERSION, Int32(MemoryLayout<z_stream>.size))
                guard status == Z_OK else { return -1 }
                
                status = inflate(&stream, Z_FINISH)
                actualSize = Int(stream.total_out)
                inflateEnd(&stream)
                
                return status == Z_STREAM_END ? Int32(actualSize) : -1
            }
        }
        
        guard result > 0 else {
            throw SPZError.decompressionFailed
        }
        
        outputBuffer.count = actualSize
        return outputBuffer
    }
    
    // MARK: - Parse Decompressed Data
    
    private static func parseDecompressedData(_ data: Data) throws -> ParseResult {
        // Parse header
        guard let header = GaussiansHeader.parse(from: data) else {
            throw SPZError.invalidHeader
        }
        
        guard header.validate() else {
            throw SPZError.invalidHeader
        }
        
        print("\n📋 Header Info:")
        print("   Version: \(header.version)")
        print("   Points: \(header.numPoints)")
        print("   SH Degree: \(header.shDegree)")
        print("   Fractional Bits: \(header.fractionalBits)")
        
        let numPoints = Int(header.numPoints)
        var offset = 16
        
        // Calculate expected data sections
        let positionsSize = numPoints * 9
        let alphasSize = numPoints * 1
        let colorsSize = numPoints * 3
        let scalesSize = numPoints * 3
        let rotationsSize = numPoints * (header.version == 3 ? 4 : 3)
        
        let expectedSize = 16 + positionsSize + alphasSize + colorsSize + scalesSize + rotationsSize
        
        print("\n📊 Data Layout:")
        print("   Positions: \(positionsSize) bytes")
        print("   Alphas: \(alphasSize) bytes")
        print("   Colors: \(colorsSize) bytes")
        print("   Scales: \(scalesSize) bytes")
        print("   Rotations: \(rotationsSize) bytes")
        print("   Expected: \(expectedSize) bytes, Actual: \(data.count) bytes")
        
        guard data.count >= expectedSize else {
            throw SPZError.dataSizeMismatch
        }
        
        // Extract data sections
        let positionsData = data.subdata(in: offset..<(offset + positionsSize))
        offset += positionsSize
        
        let alphasData = data.subdata(in: offset..<(offset + alphasSize))
        offset += alphasSize
        
        let colorsData = data.subdata(in: offset..<(offset + colorsSize))
        offset += colorsSize
        
        let scalesData = data.subdata(in: offset..<(offset + scalesSize))
        offset += scalesSize
        
        let rotationsData = data.subdata(in: offset..<(offset + rotationsSize))
        
        print("\n⚙️  Processing splats...")
        
        // Calculate bounding box
        let bounds = calculateBounds(
            positions: positionsData,
            numPoints: numPoints,
            fractionalBits: header.fractionalBits
        )
        
        print("   Bounds: min(\(String(format: "%.2f", bounds.min.x)), \(String(format: "%.2f", bounds.min.y)), \(String(format: "%.2f", bounds.min.z)))")
        print("           max(\(String(format: "%.2f", bounds.max.x)), \(String(format: "%.2f", bounds.max.y)), \(String(format: "%.2f", bounds.max.z)))")
        
        // Convert to simple splats (no packing)
        let splats = try convertToSimpleSplats(
            positions: positionsData,
            alphas: alphasData,
            colors: colorsData,
            scales: scalesData,
            rotations: rotationsData,
            numPoints: numPoints,
            header: header
        )
        
        return ParseResult(splats: splats, bounds: bounds, header: header)
    }
    
    // MARK: - Calculate Bounding Box
    
    private static func calculateBounds(
        positions: Data,
        numPoints: Int,
        fractionalBits: UInt8
    ) -> BoundingBox {
        let scaleFactor = Float(1 << fractionalBits)
        
        var minPos = SIMD3<Float>(Float.greatestFiniteMagnitude, Float.greatestFiniteMagnitude, Float.greatestFiniteMagnitude)
        var maxPos = SIMD3<Float>(-Float.greatestFiniteMagnitude, -Float.greatestFiniteMagnitude, -Float.greatestFiniteMagnitude)
        
        for i in 0..<numPoints {
            let offset = i * 9
            
            // Read 24-bit positions
            let rawX = UInt32(positions[offset]) | (UInt32(positions[offset + 1]) << 8) | (UInt32(positions[offset + 2]) << 16)
            let rawY = UInt32(positions[offset + 3]) | (UInt32(positions[offset + 4]) << 8) | (UInt32(positions[offset + 5]) << 16)
            let rawZ = UInt32(positions[offset + 6]) | (UInt32(positions[offset + 7]) << 8) | (UInt32(positions[offset + 8]) << 16)
            
            // Sign extend and convert to float
            let signedX = Int32(bitPattern: rawX << 8) >> 8
            let signedY = Int32(bitPattern: rawY << 8) >> 8
            let signedZ = Int32(bitPattern: rawZ << 8) >> 8
            
            let x = Float(signedX) / scaleFactor
            let y = Float(signedY) / scaleFactor
            let z = Float(signedZ) / scaleFactor
            
            minPos = min(minPos, SIMD3<Float>(x, y, z))
            maxPos = max(maxPos, SIMD3<Float>(x, y, z))
        }
        
        // Apply Y-flip
        let temp = minPos.y
        minPos.y = -maxPos.y
        maxPos.y = -temp
        
        return BoundingBox(min: minPos, max: maxPos)
    }
    
    // MARK: - Convert to Simple Splats (No Packing)
    
    private static func convertToSimpleSplats(
        positions: Data,
        alphas: Data,
        colors: Data,
        scales: Data,
        rotations: Data,
        numPoints: Int,
        header: GaussiansHeader
    ) throws -> [SplatData] {
        var splats = [SplatData]()
        splats.reserveCapacity(numPoints)
        
        let scaleFactor = Float(1 << header.fractionalBits)
        
        for i in 0..<numPoints {
            // === POSITION (9 bytes - decode to world space floats) ===
            let posOffset = i * 9
            let rawX = UInt32(positions[posOffset]) | (UInt32(positions[posOffset + 1]) << 8) | (UInt32(positions[posOffset + 2]) << 16)
            let rawY = UInt32(positions[posOffset + 3]) | (UInt32(positions[posOffset + 4]) << 8) | (UInt32(positions[posOffset + 5]) << 16)
            let rawZ = UInt32(positions[posOffset + 6]) | (UInt32(positions[posOffset + 7]) << 8) | (UInt32(positions[posOffset + 8]) << 16)
            
            let signedX = Int32(bitPattern: rawX << 8) >> 8
            let signedY = Int32(bitPattern: rawY << 8) >> 8
            let signedZ = Int32(bitPattern: rawZ << 8) >> 8
            
            let x = Float(signedX) / scaleFactor
            let y = -Float(signedY) / scaleFactor  // Y-flip
            let z = Float(signedZ) / scaleFactor
            
            let position = SIMD3<Float>(x, y, z)
            
            // === ROTATION (3 or 4 bytes - decode to quaternion) ===
            let rotOffset = i * (header.version == 3 ? 4 : 3)
            let rotation = parseRotation(data: rotations, offset: rotOffset, version: header.version)
            
            // === SCALE (3 bytes - decode from log space to linear) ===
            let scaleOffset = i * 3
            let scaleX = exp(Float(scales[scaleOffset]) / 255.0 * 10.0 - 5.0)
            let scaleY = exp(Float(scales[scaleOffset + 1]) / 255.0 * 10.0 - 5.0)
            let scaleZ = exp(Float(scales[scaleOffset + 2]) / 255.0 * 10.0 - 5.0)
            
            let scale = SIMD3<Float>(
                clamp(scaleX, 0.001, 10.0),
                clamp(scaleY, 0.001, 10.0),
                clamp(scaleZ, 0.001, 10.0)
            )
            
            // === COLOR & OPACITY (4 bytes - convert to 0-1 range) ===
            let colorOffset = i * 3
            let color = SIMD3<Float>(
                Float(colors[colorOffset]) / 255.0,
                Float(colors[colorOffset + 1]) / 255.0,
                Float(colors[colorOffset + 2]) / 255.0
            )
            
            let opacity = Float(alphas[i]) / 255.0
            
            // 🔍 DEBUG: Log first 3 splats for validation (reduced logging)
            if i < 100 {
                print("\n🔍 SPZ Splat \(i) Raw Data:")
                print("   Position: (\(String(format: "%.4f", x)), \(String(format: "%.4f", y)), \(String(format: "%.4f", z)))")
                print("   Color: (\(String(format: "%.3f", color.x)), \(String(format: "%.3f", color.y)), \(String(format: "%.3f", color.z)))")
                print("   Opacity: \(String(format: "%.3f", opacity))")
                print("   Scale: (\(String(format: "%.4f", scale.x)), \(String(format: "%.4f", scale.y)), \(String(format: "%.4f", scale.z)))")
            }
            
            // Create simple splat (no packing!)
            let splat = SplatData(
                position: position,
                rotation: rotation,
                scale: scale,
                color: color,
                opacity: opacity,
                depth: 0
            )
            
            splats.append(splat)
        }
        
        // 📊 VALIDATION: Analyze parsed data ranges for quality check
        print("\n📊 SPZ Data Validation Summary:")
        
        var positionBounds = (min: SIMD3<Float>(Float.greatestFiniteMagnitude, Float.greatestFiniteMagnitude, Float.greatestFiniteMagnitude),
                             max: SIMD3<Float>(-Float.greatestFiniteMagnitude, -Float.greatestFiniteMagnitude, -Float.greatestFiniteMagnitude))
        var colorBounds = (min: SIMD3<Float>(Float.greatestFiniteMagnitude, Float.greatestFiniteMagnitude, Float.greatestFiniteMagnitude),
                          max: SIMD3<Float>(-Float.greatestFiniteMagnitude, -Float.greatestFiniteMagnitude, -Float.greatestFiniteMagnitude))
        var scaleBounds = (min: SIMD3<Float>(Float.greatestFiniteMagnitude, Float.greatestFiniteMagnitude, Float.greatestFiniteMagnitude),
                          max: SIMD3<Float>(-Float.greatestFiniteMagnitude, -Float.greatestFiniteMagnitude, -Float.greatestFiniteMagnitude))
        var opacityBounds = (min: Float.greatestFiniteMagnitude, max: -Float.greatestFiniteMagnitude)
        
        var invalidCount = 0
        var zeroOpacityCount = 0
        var blackSplatCount = 0
        
        for splat in splats {
            // Track bounds
            positionBounds.min = min(positionBounds.min, splat.position)
            positionBounds.max = max(positionBounds.max, splat.position)
            colorBounds.min = min(colorBounds.min, splat.color)
            colorBounds.max = max(colorBounds.max, splat.color)
            scaleBounds.min = min(scaleBounds.min, splat.scale)
            scaleBounds.max = max(scaleBounds.max, splat.scale)
            opacityBounds.min = min(opacityBounds.min, splat.opacity)
            opacityBounds.max = max(opacityBounds.max, splat.opacity)
            
            // Count issues
            if splat.opacity <= 0.01 { zeroOpacityCount += 1 }
            if length(splat.color) < 0.1 { blackSplatCount += 1 }
            if splat.position.x.isNaN || splat.position.y.isNaN || splat.position.z.isNaN ||
               splat.color.x.isNaN || splat.color.y.isNaN || splat.color.z.isNaN ||
               splat.scale.x.isNaN || splat.scale.y.isNaN || splat.scale.z.isNaN ||
               splat.opacity.isNaN {
                invalidCount += 1
            }
        }
        
        print("   Position Range: (\(String(format: "%.3f", positionBounds.min.x)), \(String(format: "%.3f", positionBounds.min.y)), \(String(format: "%.3f", positionBounds.min.z))) to (\(String(format: "%.3f", positionBounds.max.x)), \(String(format: "%.3f", positionBounds.max.y)), \(String(format: "%.3f", positionBounds.max.z)))")
        print("   Color Range: (\(String(format: "%.3f", colorBounds.min.x)), \(String(format: "%.3f", colorBounds.min.y)), \(String(format: "%.3f", colorBounds.min.z))) to (\(String(format: "%.3f", colorBounds.max.x)), \(String(format: "%.3f", colorBounds.max.y)), \(String(format: "%.3f", colorBounds.max.z)))")
        print("   Scale Range: (\(String(format: "%.4f", scaleBounds.min.x)), \(String(format: "%.4f", scaleBounds.min.y)), \(String(format: "%.4f", scaleBounds.min.z))) to (\(String(format: "%.4f", scaleBounds.max.x)), \(String(format: "%.4f", scaleBounds.max.y)), \(String(format: "%.4f", scaleBounds.max.z)))")
        print("   Opacity Range: \(String(format: "%.3f", opacityBounds.min)) to \(String(format: "%.3f", opacityBounds.max))")
        print("   Quality Issues:")
        print("     Invalid (NaN) splats: \(invalidCount)")
        print("     Near-zero opacity splats: \(zeroOpacityCount) (\(String(format: "%.1f", Float(zeroOpacityCount) / Float(numPoints) * 100))%)")
        print("     Near-black splats: \(blackSplatCount) (\(String(format: "%.1f", Float(blackSplatCount) / Float(numPoints) * 100))%)")
        
        return splats
    }
    
    // MARK: - Rotation Parsing
    
    private static func parseRotation(data: Data, offset: Int, version: UInt32) -> simd_quatf {
        if version == 3 {
            // Version 3: Compressed quaternion
            let rotData = UInt32(data[offset]) |
                         (UInt32(data[offset + 1]) << 8) |
                         (UInt32(data[offset + 2]) << 16) |
                         (UInt32(data[offset + 3]) << 24)
            
            let largestIdx = Int(rotData & 0x3)
            let compData = rotData >> 2
            
            var q: [Float] = [0, 0, 0, 0]
            var shift = 0
            
            for j in 0..<4 {
                if j != largestIdx {
                    let val = Int16((compData >> shift) & 0x3FF) - 512
                    q[j] = Float(val) / 512.0
                    shift += 10
                }
            }
            
            let sumSq = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]
            q[largestIdx] = sqrt(max(0.0, 1.0 - sumSq))
            
            return simd_quatf(ix: q[0], iy: q[1], iz: q[2], r: q[3])
            
        } else {
            // Version 2: Simple format
            let x = Float(Int8(bitPattern: data[offset])) / 128.0
            let y = Float(Int8(bitPattern: data[offset + 1])) / 128.0
            let z = Float(Int8(bitPattern: data[offset + 2])) / 128.0
            
            let wSq = 1.0 - (x * x + y * y + z * z)
            let w = sqrt(max(0.0, wSq))
            
            return simd_quatf(ix: x, iy: y, iz: z, r: w)
        }
    }
    
    private static func clamp(_ value: Float, _ min: Float, _ max: Float) -> Float {
        return Swift.max(min, Swift.min(max, value))
    }
}

// MARK: - Extension for GaussianSplat Conversion

extension SPZParser.SplatData {
    /// Convert to GaussianSplat for rendering
    func toGaussianSplat() -> GaussianSplat {
        // Compute covariance from scale and rotation
        let covariance = computeCovariance(scale: scale, rotation: rotation)
        
        return GaussianSplat(
            position: position,
            covariance3D: covariance,
            color: color,
            opacity: opacity,
            depth: depth
        )
    }
    
    /// Convert to GaussianSplat with debug logging for first few splats
    func toGaussianSplat(index: Int) -> GaussianSplat {
        // Compute covariance from scale and rotation
        let covariance = computeCovariance(scale: scale, rotation: rotation)
        
        // 🔍 DEBUG: Log SPZ→GaussianSplat conversion for first 3 splats (reduced logging)
        if index < 3 {
            print("\n🔍 SPZ→GaussianSplat Conversion \(index):")
            print("   Position: (\(String(format: "%.4f", position.x)), \(String(format: "%.4f", position.y)), \(String(format: "%.4f", position.z)))")
            print("   Color: (\(String(format: "%.3f", color.x)), \(String(format: "%.3f", color.y)), \(String(format: "%.3f", color.z)))")
            print("   Opacity: \(String(format: "%.3f", opacity))")
            
            // Check covariance determinant
            let determinant = covariance.determinant
            print("   Covariance Determinant: \(String(format: "%.6f", determinant)) \(determinant <= 0 ? "⚠️ NON-POSITIVE!" : "✓")")
        }
        
        return GaussianSplat(
            position: position,
            covariance3D: covariance,
            color: color,
            opacity: opacity,
            depth: depth
        )
    }
    
    private func computeCovariance(scale: SIMD3<Float>, rotation: simd_quatf) -> simd_float3x3 {
        let rotMatrix = simd_float3x3(rotation)
        let scaleMatrixSquared = simd_float3x3(
            SIMD3<Float>(scale.x * scale.x, 0, 0),
            SIMD3<Float>(0, scale.y * scale.y, 0),
            SIMD3<Float>(0, 0, scale.z * scale.z)
        )
        // Compute: Σ = R × S² × Rᵀ
        return rotMatrix * scaleMatrixSquared * rotMatrix.transpose
    }
}
