import Foundation
import simd

enum SPLATFormat {
    case standard // 52 bytes per splat (pos + scale + quat + opacity + rgb)
    case extended // 64 bytes per splat (includes additional properties)
    case compressed // Variable size with compression
    
    var bytesPerSplat: Int {
        switch self {
        case .standard: return 52
        case .extended: return 64
        case .compressed: return 0 // Variable
        }
    }
}

struct SPLATHeader {
    let magic: UInt32
    let version: UInt32
    let splatCount: UInt32
    let format: SPLATFormat
    let flags: UInt32
    let headerSize: UInt32
    
    static let expectedMagic: UInt32 = 0x53504C54 // "SPLT"
    static let headerSizeBytes = 24
}

class SPLATParser: GaussianSplatParser {
    let supportedFormats = ["splat"]
    let formatVersion: String? = "1.0"
    
    private var isCancelled = false
    
    func canParse(url: URL) -> Bool {
        guard url.pathExtension.lowercased() == "splat" else { return false }
        
        do {
            let handle = try FileHandle(forReadingFrom: url)
            defer { handle.closeFile() }
            
            let headerData = handle.readData(ofLength: 4)
            guard headerData.count == 4 else { return false }
            
            let magic = headerData.withUnsafeBytes { $0.load(as: UInt32.self) }
            return magic == SPLATHeader.expectedMagic || magic.byteSwapped == SPLATHeader.expectedMagic
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
            
            if header.splatCount == 0 {
                throw ParseError.corruptedData("Invalid splat count: 0")
            }
            
            let fileSize = try FileManager.default.attributesOfItem(atPath: url.path)[.size] as? Int ?? 0
            let expectedSize = SPLATHeader.headerSizeBytes + Int(header.splatCount) * header.format.bytesPerSplat
            
            if header.format != .compressed && fileSize < expectedSize {
                throw ParseError.corruptedData("File size mismatch. Expected at least \(expectedSize), got \(fileSize)")
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
        return Int(header.splatCount)
    }
    
    func parse(from url: URL, options: ParsingOptions) throws -> GaussianSplatCollection {
        isCancelled = false
        
        try validateFile(at: url)
        let header = try parseHeader(from: url)
        
        let fileHandle = try FileHandle(forReadingFrom: url)
        defer { fileHandle.closeFile() }
        
        fileHandle.seek(toFileOffset: UInt64(header.headerSize))
        
        let metadata = SplatMetadata(
            sourceFile: url,
            fileFormat: "SPLAT",
            totalSplats: Int(header.splatCount)
        )
        
        let collection = GaussianSplatCollection(metadata: metadata)
        
        switch header.format {
        case .standard:
            try parseStandardFormat(from: fileHandle, header: header, into: collection, options: options)
        case .extended:
            try parseExtendedFormat(from: fileHandle, header: header, into: collection, options: options)
        case .compressed:
            throw ParseError.unsupportedVersion("Compressed SPLAT format not yet supported")
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
    
    private func parseHeader(from url: URL) throws -> SPLATHeader {
        let fileHandle = try FileHandle(forReadingFrom: url)
        defer { fileHandle.closeFile() }
        
        let headerData = fileHandle.readData(ofLength: SPLATHeader.headerSizeBytes)
        guard headerData.count == SPLATHeader.headerSizeBytes else {
            throw ParseError.corruptedData("Incomplete header")
        }
        
        var offset = 0
        
        let magic = headerData.withUnsafeBytes { bytes in
            bytes.load(fromByteOffset: offset, as: UInt32.self)
        }
        offset += 4
        
        let needsByteSwap = magic == SPLATHeader.expectedMagic.byteSwapped
        guard magic == SPLATHeader.expectedMagic || needsByteSwap else {
            throw ParseError.invalidFileFormat
        }
        
        let version = headerData.withUnsafeBytes { bytes in
            let value = bytes.load(fromByteOffset: offset, as: UInt32.self)
            return needsByteSwap ? value.byteSwapped : value
        }
        offset += 4
        
        let splatCount = headerData.withUnsafeBytes { bytes in
            let value = bytes.load(fromByteOffset: offset, as: UInt32.self)
            return needsByteSwap ? value.byteSwapped : value
        }
        offset += 4
        
        let formatValue = headerData.withUnsafeBytes { bytes in
            let value = bytes.load(fromByteOffset: offset, as: UInt32.self)
            return needsByteSwap ? value.byteSwapped : value
        }
        offset += 4
        
        let format: SPLATFormat
        switch formatValue {
        case 0: format = .standard
        case 1: format = .extended
        case 2: format = .compressed
        default: throw ParseError.unsupportedVersion("Unknown SPLAT format: \(formatValue)")
        }
        
        let flags = headerData.withUnsafeBytes { bytes in
            let value = bytes.load(fromByteOffset: offset, as: UInt32.self)
            return needsByteSwap ? value.byteSwapped : value
        }
        offset += 4
        
        let headerSize = headerData.withUnsafeBytes { bytes in
            let value = bytes.load(fromByteOffset: offset, as: UInt32.self)
            return needsByteSwap ? value.byteSwapped : value
        }
        
        return SPLATHeader(
            magic: magic,
            version: version,
            splatCount: splatCount,
            format: format,
            flags: flags,
            headerSize: headerSize
        )
    }
    
    private func parseStandardFormat(from fileHandle: FileHandle, header: SPLATHeader, into collection: GaussianSplatCollection, options: ParsingOptions) throws {
        let startTime = Date()
        let bytesPerSplat = header.format.bytesPerSplat
        
        for i in 0..<Int(header.splatCount) {
            if isCancelled {
                throw ParseError.cancelled
            }
            
            let splatData = fileHandle.readData(ofLength: bytesPerSplat)
            guard splatData.count == bytesPerSplat else {
                throw ParseError.corruptedData("Unexpected end of data at splat \(i)")
            }
            
            let splat = try parseStandardSplat(from: splatData)
            collection.add(splat)
            
            if i % 1000 == 0 {
                let _ = ParsingProgress(
                    bytesRead: i * bytesPerSplat,
                    totalBytes: Int(header.splatCount) * bytesPerSplat,
                    splatsProcessed: i,
                    estimatedTotalSplats: Int(header.splatCount),
                    timeElapsed: Date().timeIntervalSince(startTime),
                    estimatedTimeRemaining: nil
                )
            }
        }
    }
    
    private func parseExtendedFormat(from fileHandle: FileHandle, header: SPLATHeader, into collection: GaussianSplatCollection, options: ParsingOptions) throws {
        let startTime = Date()
        let bytesPerSplat = header.format.bytesPerSplat
        
        for i in 0..<Int(header.splatCount) {
            if isCancelled {
                throw ParseError.cancelled
            }
            
            let splatData = fileHandle.readData(ofLength: bytesPerSplat)
            guard splatData.count == bytesPerSplat else {
                throw ParseError.corruptedData("Unexpected end of data at splat \(i)")
            }
            
            let splat = try parseExtendedSplat(from: splatData)
            collection.add(splat)
            
            if i % 1000 == 0 {
                let _ = ParsingProgress(
                    bytesRead: i * bytesPerSplat,
                    totalBytes: Int(header.splatCount) * bytesPerSplat,
                    splatsProcessed: i,
                    estimatedTotalSplats: Int(header.splatCount),
                    timeElapsed: Date().timeIntervalSince(startTime),
                    estimatedTimeRemaining: nil
                )
            }
        }
    }
    
    private func parseStandardSplat(from data: Data) throws -> GaussianSplat {
        guard data.count >= 52 else {
            throw ParseError.corruptedData("Insufficient data for standard splat")
        }
        
        var offset = 0
        
        // Position (12 bytes)
        let position = SIMD3<Float>(
            data.withUnsafeBytes { $0.load(fromByteOffset: offset, as: Float.self) },
            data.withUnsafeBytes { $0.load(fromByteOffset: offset + 4, as: Float.self) },
            data.withUnsafeBytes { $0.load(fromByteOffset: offset + 8, as: Float.self) }
        )
        offset += 12
        
        // Scale (12 bytes)
        let scale = SIMD3<Float>(
            data.withUnsafeBytes { $0.load(fromByteOffset: offset, as: Float.self) },
            data.withUnsafeBytes { $0.load(fromByteOffset: offset + 4, as: Float.self) },
            data.withUnsafeBytes { $0.load(fromByteOffset: offset + 8, as: Float.self) }
        )
        offset += 12
        
        // Rotation quaternion (16 bytes)
        let rotation = simd_quatf(
            ix: data.withUnsafeBytes { $0.load(fromByteOffset: offset, as: Float.self) },
            iy: data.withUnsafeBytes { $0.load(fromByteOffset: offset + 4, as: Float.self) },
            iz: data.withUnsafeBytes { $0.load(fromByteOffset: offset + 8, as: Float.self) },
            r: data.withUnsafeBytes { $0.load(fromByteOffset: offset + 12, as: Float.self) }
        )
        offset += 16
        
        // Opacity (4 bytes)
        let opacity = data.withUnsafeBytes { $0.load(fromByteOffset: offset, as: Float.self) }
        offset += 4
        
        // Color RGB (12 bytes)
        let color = SIMD3<Float>(
            data.withUnsafeBytes { $0.load(fromByteOffset: offset, as: Float.self) },
            data.withUnsafeBytes { $0.load(fromByteOffset: offset + 4, as: Float.self) },
            data.withUnsafeBytes { $0.load(fromByteOffset: offset + 8, as: Float.self) }
        )
        
        return GaussianSplat(
            position: position,
            scale: scale,
            rotation: simd_normalize(rotation),
            opacity: Swift.min(Swift.max(opacity, 0.0), 1.0),
            color: color
        )
    }
    
    private func parseExtendedSplat(from data: Data) throws -> GaussianSplat {
        guard data.count >= 64 else {
            throw ParseError.corruptedData("Insufficient data for extended splat")
        }
        
        // First parse standard data
        let standardSplat = try parseStandardSplat(from: data)
        
        // Parse additional data from bytes 52-63
        var offset = 52
        
        // Additional spherical harmonics or other properties
        var sphericalHarmonics: [Float] = []
        
        // Read remaining 12 bytes as additional float values
        for _ in 0..<3 {
            let value = data.withUnsafeBytes { $0.load(fromByteOffset: offset, as: Float.self) }
            sphericalHarmonics.append(value)
            offset += 4
        }
        
        return GaussianSplat(
            position: standardSplat.position,
            scale: standardSplat.scale,
            rotation: standardSplat.rotation,
            opacity: standardSplat.opacity,
            color: standardSplat.color,
            sphericalHarmonics: sphericalHarmonics.isEmpty ? nil : sphericalHarmonics
        )
    }
}

// Register SPLAT parser with factory
extension GaussianSplatParserFactory {
    func registerSPLATParser() throws {
        let splatRegistration = StandardParserRegistration(
            identifier: "SPLATParser",
            supportedFormats: ["splat"],
            priority: 90
        ) {
            SPLATParser()
        }
        
        try registerParser(splatRegistration)
    }
}