import Foundation

enum ParseError: Error, LocalizedError {
    case fileNotFound
    case invalidFileFormat
    case corruptedData(String)
    case unsupportedVersion(String)
    case memoryAllocationFailure
    case ioError(Error)
    case cancelled
    case unknownError(String)
    
    var errorDescription: String? {
        switch self {
        case .fileNotFound:
            return "File not found"
        case .invalidFileFormat:
            return "Invalid file format"
        case .corruptedData(let details):
            return "Corrupted data: \(details)"
        case .unsupportedVersion(let version):
            return "Unsupported version: \(version)"
        case .memoryAllocationFailure:
            return "Memory allocation failure"
        case .ioError(let error):
            return "I/O error: \(error.localizedDescription)"
        case .cancelled:
            return "Operation cancelled"
        case .unknownError(let message):
            return "Unknown error: \(message)"
        }
    }
}

struct ParsingProgress {
    let bytesRead: Int
    let totalBytes: Int
    let splatsProcessed: Int
    let estimatedTotalSplats: Int?
    let timeElapsed: TimeInterval
    let estimatedTimeRemaining: TimeInterval?
    
    var percentage: Float {
        guard totalBytes > 0 else { return 0.0 }
        return Float(bytesRead) / Float(totalBytes)
    }
    
    var isComplete: Bool {
        return bytesRead >= totalBytes
    }
}

typealias ProgressCallback = (ParsingProgress) -> Void

enum ParseResult {
    case success(GaussianSplatCollection)
    case failure(ParseError)
}

struct ParsingOptions {
    let maxMemoryUsage: Int?
    let enableStreaming: Bool
    let chunkSize: Int
    let enableValidation: Bool
    let progressUpdateInterval: TimeInterval
    
    static let `default` = ParsingOptions(
        maxMemoryUsage: nil,
        enableStreaming: false,
        chunkSize: 8192,
        enableValidation: true,
        progressUpdateInterval: 0.1
    )
    
    static func streaming(maxMemory: Int = 500_000_000, chunkSize: Int = 65536) -> ParsingOptions {
        return ParsingOptions(
            maxMemoryUsage: maxMemory,
            enableStreaming: true,
            chunkSize: chunkSize,
            enableValidation: true,
            progressUpdateInterval: 0.05
        )
    }
}

protocol GaussianSplatParser {
    var supportedFormats: [String] { get }
    var formatVersion: String? { get }
    
    func canParse(url: URL) -> Bool
    func validateFile(at url: URL) throws
    
    func parse(from url: URL, options: ParsingOptions) throws -> GaussianSplatCollection
    
    func parseAsync(from url: URL, 
                   options: ParsingOptions, 
                   progressCallback: ProgressCallback?,
                   completion: @escaping (ParseResult) -> Void)
    
    func parseAsync(from url: URL, 
                   options: ParsingOptions, 
                   progressCallback: ProgressCallback?) async -> ParseResult
    
    func estimateSplatCount(from url: URL) throws -> Int?
    func getFileInfo(from url: URL) throws -> FileInfo
}

struct FileInfo {
    let url: URL
    let fileSize: Int
    let format: String
    let version: String?
    let estimatedSplatCount: Int?
    let isCompressed: Bool
    let compressionRatio: Float?
    let metadata: [String: Any]
    
    init(url: URL, fileSize: Int, format: String, version: String? = nil, 
         estimatedSplatCount: Int? = nil, isCompressed: Bool = false, 
         compressionRatio: Float? = nil, metadata: [String: Any] = [:]) {
        self.url = url
        self.fileSize = fileSize
        self.format = format
        self.version = version
        self.estimatedSplatCount = estimatedSplatCount
        self.isCompressed = isCompressed
        self.compressionRatio = compressionRatio
        self.metadata = metadata
    }
}

protocol StreamingParser {
    associatedtype ChunkData
    
    func beginStreaming(from url: URL, options: ParsingOptions) throws
    func readNextChunk() throws -> ChunkData?
    func processChunk(_ chunk: ChunkData) throws -> [GaussianSplat]
    func endStreaming() throws -> SplatMetadata
    func cancelStreaming()
}

extension GaussianSplatParser {
    func parseAsync(from url: URL, 
                   options: ParsingOptions = .default, 
                   progressCallback: ProgressCallback? = nil) async -> ParseResult {
        return await withCheckedContinuation { continuation in
            parseAsync(from: url, options: options, progressCallback: progressCallback) { result in
                continuation.resume(returning: result)
            }
        }
    }
    
    func parse(from url: URL) throws -> GaussianSplatCollection {
        return try parse(from: url, options: .default)
    }
    
    func canParse(url: URL) -> Bool {
        let pathExtension = url.pathExtension.lowercased()
        return supportedFormats.contains(pathExtension)
    }
    
    func getFileInfo(from url: URL) throws -> FileInfo {
        let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
        let fileSize = attributes[.size] as? Int ?? 0
        let format = url.pathExtension.lowercased()
        
        return FileInfo(
            url: url,
            fileSize: fileSize,
            format: format,
            version: formatVersion,
            estimatedSplatCount: try? estimateSplatCount(from: url)
        )
    }
}