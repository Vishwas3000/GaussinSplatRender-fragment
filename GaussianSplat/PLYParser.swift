import Foundation
import simd

struct PLYProperty {
    let name: String
    let type: PLYDataType
    let listCountType: PLYDataType?
    let listElementType: PLYDataType?
    
    var isListProperty: Bool {
        return listCountType != nil && listElementType != nil
    }
}

enum PLYDataType: String, CaseIterable {
    case char, uchar
    case short, ushort
    case int, uint
    case float, double
    
    var size: Int {
        switch self {
        case .char, .uchar: return 1
        case .short, .ushort: return 2
        case .int, .uint, .float: return 4
        case .double: return 8
        }
    }
}

enum PLYFormat {
    case ascii
    case binaryLittleEndian
    case binaryBigEndian
}

struct PLYHeader {
    let format: PLYFormat
    let version: String
    let vertexCount: Int
    let properties: [PLYProperty]
    let comments: [String]
    let headerEndOffset: Int
}

class PLYParser: GaussianSplatParser {
    let supportedFormats = ["ply"]
    let formatVersion: String? = "1.0"
    
    private var progressTimer: Timer?
    private var isCancelled = false
    
    func canParse(url: URL) -> Bool {
        guard url.pathExtension.lowercased() == "ply" else { return false }
        
        do {
            let handle = try FileHandle(forReadingFrom: url)
            defer { handle.closeFile() }
            
            let headerData = handle.readData(ofLength: 100)
            let headerString = String(data: headerData, encoding: .utf8) ?? ""
            return headerString.hasPrefix("ply")
        } catch {
            return false
        }
    }
    
    func validateFile(at url: URL) throws {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw ParseError.fileNotFound
        }
        
        do {
            let header = try parseHeader(from: url)
            
            if header.vertexCount <= 0 {
                throw ParseError.corruptedData("Invalid vertex count: \(header.vertexCount)")
            }
            
            let requiredProperties = ["x", "y", "z"]
            for required in requiredProperties {
                if !header.properties.contains(where: { $0.name == required }) {
                    throw ParseError.corruptedData("Missing required property: \(required)")
                }
            }
        } catch {
            if error is ParseError {
                throw error
            } else {
                throw ParseError.ioError(error)
            }
        }
    }
    
    func estimateSplatCount(from url: URL) throws -> Int? {
        let header = try parseHeader(from: url)
        return header.vertexCount
    }
    
    func parse(from url: URL, options: ParsingOptions) throws -> GaussianSplatCollection {
        isCancelled = false
        
        try validateFile(at: url)
        let header = try parseHeader(from: url)
        
        let fileHandle = try FileHandle(forReadingFrom: url)
        defer { fileHandle.closeFile() }
        
        fileHandle.seek(toFileOffset: UInt64(header.headerEndOffset))
        
        let metadata = SplatMetadata(
            sourceFile: url,
            fileFormat: "PLY",
            totalSplats: header.vertexCount
        )
        
        let collection = GaussianSplatCollection(metadata: metadata)
        
        switch header.format {
        case .ascii:
            try parseASCIIData(from: fileHandle, header: header, into: collection, options: options)
        case .binaryLittleEndian, .binaryBigEndian:
            try parseBinaryData(from: fileHandle, header: header, into: collection, options: options)
        }
        
        if isCancelled {
            throw ParseError.cancelled
        }
        
        return collection
    }
    
    func parseAsync(from url: URL, options: ParsingOptions, progressCallback: ProgressCallback?, completion: @escaping (ParseResult) -> Void) {
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            do {
                let result = try self?.parse(from: url, options: options)
                DispatchQueue.main.async {
                    if let collection = result {
                        completion(.success(collection))
                    } else {
                        completion(.failure(.unknownError("Parser was deallocated")))
                    }
                }
            } catch let error as ParseError {
                DispatchQueue.main.async {
                    completion(.failure(error))
                }
            } catch {
                DispatchQueue.main.async {
                    completion(.failure(.ioError(error)))
                }
            }
        }
    }
    
    private func parseHeader(from url: URL) throws -> PLYHeader {
        let fileHandle = try FileHandle(forReadingFrom: url)
        defer { fileHandle.closeFile() }
        
        var properties: [PLYProperty] = []
        var comments: [String] = []
        var format: PLYFormat = .ascii
        var version = "1.0"
        var vertexCount = 0
        var currentOffset = 0
        
        while true {
            guard let lineData = fileHandle.readLine() else {
                throw ParseError.corruptedData("Unexpected end of file while reading header")
            }
            
            currentOffset += lineData.count + 1
            
            guard let line = String(data: lineData, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines) else {
                throw ParseError.corruptedData("Invalid header encoding")
            }
            
            if line == "end_header" {
                break
            }
            
            let components = line.components(separatedBy: .whitespaces)
            guard !components.isEmpty else { continue }
            
            switch components[0] {
            case "ply":
                continue
            case "format":
                guard components.count >= 3 else {
                    throw ParseError.corruptedData("Invalid format line")
                }
                version = components[2]
                switch components[1] {
                case "ascii":
                    format = .ascii
                case "binary_little_endian":
                    format = .binaryLittleEndian
                case "binary_big_endian":
                    format = .binaryBigEndian
                default:
                    throw ParseError.unsupportedVersion("Unsupported format: \(components[1])")
                }
            case "element":
                if components.count >= 3 && components[1] == "vertex" {
                    guard let count = Int(components[2]) else {
                        throw ParseError.corruptedData("Invalid vertex count")
                    }
                    vertexCount = count
                }
            case "property":
                let property = try parsePropertyLine(components)
                properties.append(property)
            case "comment":
                let comment = components.dropFirst().joined(separator: " ")
                comments.append(comment)
            default:
                break
            }
        }
        
        return PLYHeader(
            format: format,
            version: version,
            vertexCount: vertexCount,
            properties: properties,
            comments: comments,
            headerEndOffset: currentOffset
        )
    }
    
    private func parsePropertyLine(_ components: [String]) throws -> PLYProperty {
        guard components.count >= 3 else {
            throw ParseError.corruptedData("Invalid property line")
        }
        
        if components[1] == "list" {
            guard components.count >= 5 else {
                throw ParseError.corruptedData("Invalid list property line")
            }
            
            guard let countType = PLYDataType(rawValue: components[2]),
                  let elementType = PLYDataType(rawValue: components[3]) else {
                throw ParseError.corruptedData("Invalid list property types")
            }
            
            return PLYProperty(
                name: components[4],
                type: elementType,
                listCountType: countType,
                listElementType: elementType
            )
        } else {
            guard let type = PLYDataType(rawValue: components[1]) else {
                throw ParseError.corruptedData("Invalid property type: \(components[1])")
            }
            
            return PLYProperty(
                name: components[2],
                type: type,
                listCountType: nil,
                listElementType: nil
            )
        }
    }
    
    private func parseASCIIData(from fileHandle: FileHandle, header: PLYHeader, into collection: GaussianSplatCollection, options: ParsingOptions) throws {
        let startTime = Date()
        
        for i in 0..<header.vertexCount {
            if isCancelled {
                throw ParseError.cancelled
            }
            
            guard let lineData = fileHandle.readLine(),
                  let line = String(data: lineData, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines) else {
                throw ParseError.corruptedData("Failed to read vertex data at line \(i)")
            }
            
            let values = line.components(separatedBy: .whitespaces)
            let splat = try parseSplatFromValues(values, properties: header.properties)
            collection.add(splat)
            
            if i % 1000 == 0 {
                let _ = ParsingProgress(
                    bytesRead: i * 236,
                    totalBytes: header.vertexCount * 236,
                    splatsProcessed: i,
                    estimatedTotalSplats: header.vertexCount,
                    timeElapsed: Date().timeIntervalSince(startTime),
                    estimatedTimeRemaining: nil
                )
            }
        }
    }
    
    private func parseBinaryData(from fileHandle: FileHandle, header: PLYHeader, into collection: GaussianSplatCollection, options: ParsingOptions) throws {
        let startTime = Date()
        let bytesPerVertex = calculateBytesPerVertex(properties: header.properties)
        
        for i in 0..<header.vertexCount {
            if isCancelled {
                throw ParseError.cancelled
            }
            
            let vertexData = fileHandle.readData(ofLength: bytesPerVertex)
            guard vertexData.count == bytesPerVertex else {
                throw ParseError.corruptedData("Unexpected end of binary data at vertex \(i)")
            }
            
            let splat = try parseSplatFromBinaryData(vertexData, properties: header.properties, format: header.format)
            collection.add(splat)
            
            if i % 1000 == 0 {
                let _ = ParsingProgress(
                    bytesRead: i * bytesPerVertex,
                    totalBytes: header.vertexCount * bytesPerVertex,
                    splatsProcessed: i,
                    estimatedTotalSplats: header.vertexCount,
                    timeElapsed: Date().timeIntervalSince(startTime),
                    estimatedTimeRemaining: nil
                )
            }
        }
    }
    
    private func parseSplatFromValues(_ values: [String], properties: [PLYProperty]) throws -> GaussianSplat {
        var position = SIMD3<Float>(0, 0, 0)
        var scale = SIMD3<Float>(1, 1, 1)
        var rotation = simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)
        var opacity: Float = 1.0
        var color = SIMD3<Float>(1, 1, 1)
        var sphericalHarmonics: [Float] = []
        
        guard values.count >= properties.count else {
            throw ParseError.corruptedData("Insufficient values for properties")
        }
        
        for (i, property) in properties.enumerated() {
            guard i < values.count, let value = Float(values[i]) else {
                continue
            }
            
            switch property.name {
            case "x": position.x = value
            case "y": position.y = value
            case "z": position.z = value
            case "scale_0": scale.x = exp(value)
            case "scale_1": scale.y = exp(value)
            case "scale_2": scale.z = exp(value)
            case "rot_0": rotation.vector.x = value
            case "rot_1": rotation.vector.y = value
            case "rot_2": rotation.vector.z = value
            case "rot_3": rotation.vector.w = value
            case "opacity": opacity = 1.0 / (1.0 + exp(-value))
            case "red": color.x = value / 255.0
            case "green": color.y = value / 255.0
            case "blue": color.z = value / 255.0
            default:
                if property.name.hasPrefix("f_dc_") || property.name.hasPrefix("f_rest_") {
                    sphericalHarmonics.append(value)
                }
            }
        }
        
        rotation = simd_normalize(rotation)
        
        return GaussianSplat(
            position: position,
            scale: scale,
            rotation: rotation,
            opacity: opacity,
            color: color,
            sphericalHarmonics: sphericalHarmonics.isEmpty ? nil : sphericalHarmonics
        )
    }
    
    private func parseSplatFromBinaryData(_ data: Data, properties: [PLYProperty], format: PLYFormat) throws -> GaussianSplat {
        var position = SIMD3<Float>(0, 0, 0)
        var scale = SIMD3<Float>(1, 1, 1)
        var rotation = simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)
        var opacity: Float = 1.0
        var color = SIMD3<Float>(1, 1, 1)
        var sphericalHarmonics: [Float] = []
        
        var offset = 0
        let littleEndian = (format == .binaryLittleEndian)
        
        for property in properties {
            let value = try readBinaryValue(from: data, at: &offset, type: property.type, littleEndian: littleEndian)
            
            switch property.name {
            case "x": position.x = value
            case "y": position.y = value
            case "z": position.z = value
            case "scale_0": scale.x = exp(value)
            case "scale_1": scale.y = exp(value)
            case "scale_2": scale.z = exp(value)
            case "rot_0": rotation.vector.x = value
            case "rot_1": rotation.vector.y = value
            case "rot_2": rotation.vector.z = value
            case "rot_3": rotation.vector.w = value
            case "opacity": opacity = 1.0 / (1.0 + exp(-value))
            case "red": color.x = value / 255.0
            case "green": color.y = value / 255.0
            case "blue": color.z = value / 255.0
            default:
                if property.name.hasPrefix("f_dc_") || property.name.hasPrefix("f_rest_") {
                    sphericalHarmonics.append(value)
                }
            }
        }
        
        rotation = simd_normalize(rotation)
        
        return GaussianSplat(
            position: position,
            scale: scale,
            rotation: rotation,
            opacity: opacity,
            color: color,
            sphericalHarmonics: sphericalHarmonics.isEmpty ? nil : sphericalHarmonics
        )
    }
    
    private func readBinaryValue(from data: Data, at offset: inout Int, type: PLYDataType, littleEndian: Bool) throws -> Float {
        guard offset + type.size <= data.count else {
            throw ParseError.corruptedData("Insufficient binary data")
        }
        
        let value: Float
        
        switch type {
        case .float:
            let bytes = data.subdata(in: offset..<offset + 4)
            let floatValue = bytes.withUnsafeBytes { $0.load(as: Float.self) }
            value = littleEndian ? floatValue : Float(bitPattern: floatValue.bitPattern.byteSwapped)
        case .double:
            let bytes = data.subdata(in: offset..<offset + 8)
            let doubleValue = bytes.withUnsafeBytes { $0.load(as: Double.self) }
            let swapped = littleEndian ? doubleValue : Double(bitPattern: doubleValue.bitPattern.byteSwapped)
            value = Float(swapped)
        case .int:
            let bytes = data.subdata(in: offset..<offset + 4)
            let intValue = bytes.withUnsafeBytes { $0.load(as: Int32.self) }
            value = Float(littleEndian ? intValue : intValue.byteSwapped)
        case .uint:
            let bytes = data.subdata(in: offset..<offset + 4)
            let uintValue = bytes.withUnsafeBytes { $0.load(as: UInt32.self) }
            value = Float(littleEndian ? uintValue : uintValue.byteSwapped)
        case .short:
            let bytes = data.subdata(in: offset..<offset + 2)
            let shortValue = bytes.withUnsafeBytes { $0.load(as: Int16.self) }
            value = Float(littleEndian ? shortValue : shortValue.byteSwapped)
        case .ushort:
            let bytes = data.subdata(in: offset..<offset + 2)
            let ushortValue = bytes.withUnsafeBytes { $0.load(as: UInt16.self) }
            value = Float(littleEndian ? ushortValue : ushortValue.byteSwapped)
        case .char:
            let byteValue = data[offset]
            value = Float(Int8(bitPattern: byteValue))
        case .uchar:
            let byteValue = data[offset]
            value = Float(byteValue)
        }
        
        offset += type.size
        return value
    }
    
    private func calculateBytesPerVertex(properties: [PLYProperty]) -> Int {
        return properties.reduce(0) { total, property in
            if property.isListProperty {
                return total + (property.listCountType?.size ?? 0)
            } else {
                return total + property.type.size
            }
        }
    }
}

extension FileHandle {
    func readLine() -> Data? {
        var line = Data()
        
        while true {
            let byte = readData(ofLength: 1)
            if byte.isEmpty {
                break
            }
            if byte[0] == 10 { // newline
                return line
            }
            line.append(byte)
        }
        
        return line.isEmpty ? nil : line
    }
}