import Foundation
import simd

protocol ChunkedDataReader {
    associatedtype ChunkType
    
    func beginReading(from url: URL, chunkSize: Int) throws
    func readNextChunk() throws -> ChunkType?
    func endReading() throws
    func cancel()
    
    var totalBytesRead: Int { get }
    var isAtEnd: Bool { get }
}

class FileChunkReader: ChunkedDataReader {
    typealias ChunkType = Data
    
    private var fileHandle: FileHandle?
    private var _totalBytesRead: Int = 0
    private var fileSize: Int = 0
    private var currentChunkSize: Int = 8192
    
    var totalBytesRead: Int { return _totalBytesRead }
    var isAtEnd: Bool { 
        guard let handle = fileHandle else { return true }
        return handle.offsetInFile >= fileSize
    }
    
    func beginReading(from url: URL, chunkSize: Int) throws {
        let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
        fileSize = attributes[.size] as? Int ?? 0
        currentChunkSize = chunkSize
        
        fileHandle = try FileHandle(forReadingFrom: url)
        _totalBytesRead = 0
    }
    
    func readNextChunk() throws -> Data? {
        guard let handle = fileHandle, !isAtEnd else { return nil }
        
        let remainingBytes = fileSize - _totalBytesRead
        let bytesToRead = min(currentChunkSize, remainingBytes)
        
        guard bytesToRead > 0 else { return nil }
        
        let chunk = handle.readData(ofLength: bytesToRead)
        _totalBytesRead += chunk.count
        
        return chunk.isEmpty ? nil : chunk
    }
    
    func endReading() throws {
        fileHandle?.closeFile()
        fileHandle = nil
    }
    
    func cancel() {
        fileHandle?.closeFile()
        fileHandle = nil
    }
}

class StreamingGaussianSplatParser {
    private let chunkReader: FileChunkReader
    private let maxMemoryUsage: Int
    private var currentMemoryUsage: Int = 0
    private var isCancelled: Bool = false
    
    init(maxMemoryUsage: Int = 500_000_000) {
        self.maxMemoryUsage = maxMemoryUsage
        self.chunkReader = FileChunkReader()
    }
    
    func parseFileStreaming(
        from url: URL,
        parser: GaussianSplatParser,
        options: ParsingOptions,
        progressCallback: ProgressCallback? = nil,
        splatCallback: @escaping ([GaussianSplat]) -> Bool // Return false to stop parsing
    ) async throws -> SplatMetadata {
        
        guard options.enableStreaming else {
            throw ParseError.unknownError("Streaming not enabled in options")
        }
        
        isCancelled = false
        
        try chunkReader.beginReading(from: url, chunkSize: options.chunkSize)
        defer { chunkReader.cancel() }
        
        let startTime = Date()
        var processedSplats = 0
        
        let fileInfo = try parser.getFileInfo(from: url)
        let estimatedTotal = fileInfo.estimatedSplatCount ?? 0
        
        // Create a buffer for partial splat data
        var dataBuffer = Data()
        var splatBuffer: [GaussianSplat] = []
        splatBuffer.reserveCapacity(1000) // Process in batches of 1000
        
        while !chunkReader.isAtEnd && !isCancelled {
            guard let chunk = try chunkReader.readNextChunk() else { break }
            
            dataBuffer.append(chunk)
            
            // Try to parse complete splats from buffer
            let (parsedSplats, remainingData) = try extractSplatsFromBuffer(
                dataBuffer,
                parser: parser,
                fileInfo: fileInfo
            )
            
            dataBuffer = remainingData
            splatBuffer.append(contentsOf: parsedSplats)
            
            // Check memory usage
            currentMemoryUsage = splatBuffer.reduce(0) { $0 + $1.memorySize }
            
            if currentMemoryUsage >= maxMemoryUsage || splatBuffer.count >= 1000 {
                // Send batch to callback
                let shouldContinue = splatCallback(splatBuffer)
                processedSplats += splatBuffer.count
                
                if !shouldContinue {
                    break
                }
                
                splatBuffer.removeAll(keepingCapacity: true)
                currentMemoryUsage = 0
            }
            
            // Report progress
            if let progressCallback = progressCallback {
                let progress = ParsingProgress(
                    bytesRead: chunkReader.totalBytesRead,
                    totalBytes: fileInfo.fileSize,
                    splatsProcessed: processedSplats,
                    estimatedTotalSplats: estimatedTotal,
                    timeElapsed: Date().timeIntervalSince(startTime),
                    estimatedTimeRemaining: estimateRemainingTime(
                        processedSplats: processedSplats,
                        totalSplats: estimatedTotal,
                        timeElapsed: Date().timeIntervalSince(startTime)
                    )
                )
                
                await MainActor.run {
                    progressCallback(progress)
                }
            }
        }
        
        // Process any remaining splats
        if !splatBuffer.isEmpty && !isCancelled {
            _ = splatCallback(splatBuffer)
            processedSplats += splatBuffer.count
        }
        
        try chunkReader.endReading()
        
        if isCancelled {
            throw ParseError.cancelled
        }
        
        return SplatMetadata(
            sourceFile: url,
            fileFormat: fileInfo.format,
            totalSplats: processedSplats
        )
    }
    
    func cancel() {
        isCancelled = true
        chunkReader.cancel()
    }
    
    private func extractSplatsFromBuffer(
        _ buffer: Data,
        parser: GaussianSplatParser,
        fileInfo: FileInfo
    ) throws -> ([GaussianSplat], Data) {
        
        // This is a simplified version - in practice, you'd need format-specific logic
        // For PLY files, you'd parse line by line
        // For SPLAT files, you'd parse fixed-size binary records
        
        var splats: [GaussianSplat] = []
        var remainingData = buffer
        
        if fileInfo.format.lowercased() == "splat" {
            // Parse fixed-size binary records
            let bytesPerSplat = 52 // Standard SPLAT format
            
            while remainingData.count >= bytesPerSplat {
                let splatData = remainingData.prefix(bytesPerSplat)
                remainingData = remainingData.dropFirst(bytesPerSplat)
                
                let _ = parser as? SPLATParser
                // Create a simple splat from binary data
                let splat = try createSplatFromBinaryData(Data(splatData))
                splats.append(splat)
            }
        } else if fileInfo.format.lowercased() == "ply" {
            // Parse line by line for ASCII PLY
            let string = String(data: buffer, encoding: .utf8) ?? ""
            let lines = string.components(separatedBy: .newlines)
            
            for (index, line) in lines.enumerated() {
                if index == lines.count - 1 && !line.isEmpty {
                    // Last line might be incomplete
                    remainingData = line.data(using: .utf8) ?? Data()
                    break
                }
                
                if !line.isEmpty && !line.hasPrefix("ply") && !line.hasPrefix("format") {
                    // Try to parse as splat data
                    let values = line.components(separatedBy: .whitespaces)
                    if values.count >= 6 { // At least x, y, z, r, g, b
                        let splat = try createSplatFromStringValues(values)
                        splats.append(splat)
                    }
                }
            }
            
            if lines.last?.isEmpty == false {
                remainingData = Data()
            }
        }
        
        return (splats, remainingData)
    }
    
    private func createSplatFromBinaryData(_ data: Data) throws -> GaussianSplat {
        guard data.count >= 52 else {
            throw ParseError.corruptedData("Insufficient binary data")
        }
        
        var offset = 0
        
        let position = SIMD3<Float>(
            data.withUnsafeBytes { $0.load(fromByteOffset: offset, as: Float.self) },
            data.withUnsafeBytes { $0.load(fromByteOffset: offset + 4, as: Float.self) },
            data.withUnsafeBytes { $0.load(fromByteOffset: offset + 8, as: Float.self) }
        )
        offset += 12
        
        let scale = SIMD3<Float>(
            data.withUnsafeBytes { $0.load(fromByteOffset: offset, as: Float.self) },
            data.withUnsafeBytes { $0.load(fromByteOffset: offset + 4, as: Float.self) },
            data.withUnsafeBytes { $0.load(fromByteOffset: offset + 8, as: Float.self) }
        )
        offset += 12
        
        let rotation = simd_quatf(
            vector: SIMD4<Float>(
                data.withUnsafeBytes { $0.load(fromByteOffset: offset, as: Float.self) },
                data.withUnsafeBytes { $0.load(fromByteOffset: offset + 4, as: Float.self) },
                data.withUnsafeBytes { $0.load(fromByteOffset: offset + 8, as: Float.self) },
                data.withUnsafeBytes { $0.load(fromByteOffset: offset + 12, as: Float.self) }
            )
        )
        offset += 16
        
        let opacity = data.withUnsafeBytes { $0.load(fromByteOffset: offset, as: Float.self) }
        offset += 4
        
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
    
    private func createSplatFromStringValues(_ values: [String]) throws -> GaussianSplat {
        guard values.count >= 6 else {
            throw ParseError.corruptedData("Insufficient string values")
        }
        
        let position = SIMD3<Float>(
            Float(values[0]) ?? 0,
            Float(values[1]) ?? 0,
            Float(values[2]) ?? 0
        )
        
        let color = SIMD3<Float>(
            Float(values[3]) ?? 255,
            Float(values[4]) ?? 255,
            Float(values[5]) ?? 255
        ) / 255.0
        
        return GaussianSplat(
            position: position,
            scale: SIMD3<Float>(1, 1, 1),
            rotation: simd_quatf(vector: SIMD4<Float>(0, 0, 0, 1)),
            opacity: 1.0,
            color: color
        )
    }
    
    private func estimateRemainingTime(processedSplats: Int, totalSplats: Int, timeElapsed: TimeInterval) -> TimeInterval? {
        guard processedSplats > 0 && totalSplats > processedSplats else { return nil }
        
        let rate = Double(processedSplats) / timeElapsed
        let remainingSplats = totalSplats - processedSplats
        
        return Double(remainingSplats) / rate
    }
}

extension GaussianSplatParserFactory {
    func parseFileStreaming(
        at url: URL,
        options: ParsingOptions = .streaming(),
        progressCallback: ProgressCallback? = nil,
        splatCallback: @escaping ([GaussianSplat]) -> Bool
    ) async -> ParseResult {
        
        do {
            let parser = try createParser(for: url)
            let streamingParser = StreamingGaussianSplatParser(maxMemoryUsage: options.maxMemoryUsage ?? 500_000_000)
            
            let metadata = try await streamingParser.parseFileStreaming(
                from: url,
                parser: parser,
                options: options,
                progressCallback: progressCallback,
                splatCallback: splatCallback
            )
            
            let collection = GaussianSplatCollection(metadata: metadata)
            return .success(collection)
            
        } catch let error as ParseError {
            return .failure(error)
        } catch {
            return .failure(.ioError(error))
        }
    }
}

// Memory-efficient collection builder for streaming
class StreamingSplatCollectionBuilder {
    private var tempFileURL: URL?
    private var tempFileHandle: FileHandle?
    private var splatCount: Int = 0
    private let metadata: SplatMetadata
    
    init(metadata: SplatMetadata) throws {
        self.metadata = metadata
        self.tempFileURL = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("temp_splats")
        
        guard let tempURL = tempFileURL else {
            throw ParseError.memoryAllocationFailure
        }
        
        FileManager.default.createFile(atPath: tempURL.path, contents: nil)
        self.tempFileHandle = try FileHandle(forWritingTo: tempURL)
    }
    
    func addSplats(_ splats: [GaussianSplat]) throws {
        guard let handle = tempFileHandle else {
            throw ParseError.ioError(NSError(domain: "StreamingParser", code: 1, userInfo: [NSLocalizedDescriptionKey: "No temp file handle"]))
        }
        
        for splat in splats {
            let data = try encodeSplat(splat)
            handle.write(data)
            splatCount += 1
        }
    }
    
    func finalize() throws -> GaussianSplatCollection {
        tempFileHandle?.closeFile()
        tempFileHandle = nil
        
        defer {
            if let tempURL = tempFileURL {
                try? FileManager.default.removeItem(at: tempURL)
            }
        }
        
        // Create final collection (this could be optimized further)
        let finalMetadata = SplatMetadata(
            sourceFile: metadata.sourceFile,
            fileFormat: metadata.fileFormat,
            totalSplats: splatCount
        )
        
        return GaussianSplatCollection(metadata: finalMetadata)
    }
    
    private func encodeSplat(_ splat: GaussianSplat) throws -> Data {
        var data = Data()
        
        // Encode position
        withUnsafeBytes(of: splat.position.x) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: splat.position.y) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: splat.position.z) { data.append(contentsOf: $0) }
        
        // Encode scale
        withUnsafeBytes(of: splat.scale.x) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: splat.scale.y) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: splat.scale.z) { data.append(contentsOf: $0) }
        
        // Encode rotation
        withUnsafeBytes(of: splat.rotation.vector.x) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: splat.rotation.vector.y) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: splat.rotation.vector.z) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: splat.rotation.vector.w) { data.append(contentsOf: $0) }
        
        // Encode opacity and color
        withUnsafeBytes(of: splat.opacity) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: splat.color.x) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: splat.color.y) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: splat.color.z) { data.append(contentsOf: $0) }
        
        return data
    }
}