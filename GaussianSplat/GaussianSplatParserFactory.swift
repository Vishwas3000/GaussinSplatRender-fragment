import Foundation

enum FactoryError: Error, LocalizedError {
    case unsupportedFormat(String)
    case noCompatibleParser
    case parserRegistrationFailed
    case ambiguousFormat([String])
    
    var errorDescription: String? {
        switch self {
        case .unsupportedFormat(let format):
            return "Unsupported file format: \(format)"
        case .noCompatibleParser:
            return "No compatible parser found for this file"
        case .parserRegistrationFailed:
            return "Failed to register parser"
        case .ambiguousFormat(let formats):
            return "Multiple parsers available for formats: \(formats.joined(separator: ", "))"
        }
    }
}

protocol ParserRegistration {
    var identifier: String { get }
    var supportedFormats: [String] { get }
    var priority: Int { get }
    func createParser() -> GaussianSplatParser
}

struct StandardParserRegistration: ParserRegistration {
    let identifier: String
    let supportedFormats: [String]
    let priority: Int
    private let factory: () -> GaussianSplatParser
    
    init(identifier: String, supportedFormats: [String], priority: Int = 100, factory: @escaping () -> GaussianSplatParser) {
        self.identifier = identifier
        self.supportedFormats = supportedFormats
        self.priority = priority
        self.factory = factory
    }
    
    func createParser() -> GaussianSplatParser {
        return factory()
    }
}

class GaussianSplatParserFactory {
    static let shared = GaussianSplatParserFactory()
    
    private var registrations: [ParserRegistration] = []
    private let queue = DispatchQueue(label: "com.gaussiansplat.factory", attributes: .concurrent)
    
    private init() {
        registerDefaultParsers()
    }
    
    func registerParser(_ registration: ParserRegistration) throws {
        queue.async(flags: .barrier) {
            if let existingIndex = self.registrations.firstIndex(where: { $0.identifier == registration.identifier }) {
                self.registrations[existingIndex] = registration
            } else {
                self.registrations.append(registration)
                self.registrations.sort { $0.priority > $1.priority }
            }
        }
    }
    
    func unregisterParser(identifier: String) {
        queue.async(flags: .barrier) {
            self.registrations.removeAll { $0.identifier == identifier }
        }
    }
    
    func getRegisteredFormats() -> [String] {
        return queue.sync {
            return Array(Set(registrations.flatMap { $0.supportedFormats })).sorted()
        }
    }
    
    func createParser(for url: URL) throws -> GaussianSplatParser {
        return try queue.sync {
            let fileExtension = url.pathExtension.lowercased()
            let compatibleRegistrations = registrations.filter { registration in
                registration.supportedFormats.contains(fileExtension)
            }
            
            guard !compatibleRegistrations.isEmpty else {
                throw FactoryError.unsupportedFormat(fileExtension)
            }
            
            for registration in compatibleRegistrations {
                let parser = registration.createParser()
                if parser.canParse(url: url) {
                    return parser
                }
            }
            
            throw FactoryError.noCompatibleParser
        }
    }
    
    func createParser(forFormat format: String) throws -> GaussianSplatParser {
        return try queue.sync {
            let normalizedFormat = format.lowercased()
            let compatibleRegistrations = registrations.filter { registration in
                registration.supportedFormats.contains(normalizedFormat)
            }
            
            guard let registration = compatibleRegistrations.first else {
                throw FactoryError.unsupportedFormat(format)
            }
            
            return registration.createParser()
        }
    }
    
    func detectFormat(at url: URL) throws -> String {
        return try queue.sync {
            let fileExtension = url.pathExtension.lowercased()
            
            guard !fileExtension.isEmpty else {
                throw FactoryError.unsupportedFormat("Unknown")
            }
            
            let compatibleRegistrations = registrations.filter { registration in
                registration.supportedFormats.contains(fileExtension)
            }
            
            guard !compatibleRegistrations.isEmpty else {
                throw FactoryError.unsupportedFormat(fileExtension)
            }
            
            for registration in compatibleRegistrations {
                let parser = registration.createParser()
                if parser.canParse(url: url) {
                    return fileExtension
                }
            }
            
            throw FactoryError.noCompatibleParser
        }
    }
    
    func getParserInfo(for url: URL) throws -> ParserInfo {
        let parser = try createParser(for: url)
        let fileInfo = try parser.getFileInfo(from: url)
        
        return ParserInfo(
            parserType: String(describing: type(of: parser)),
            supportedFormats: parser.supportedFormats,
            formatVersion: parser.formatVersion,
            fileInfo: fileInfo
        )
    }
    
    func validateFile(at url: URL) throws -> ValidationResult {
        let parser = try createParser(for: url)
        
        do {
            try parser.validateFile(at: url)
            let fileInfo = try parser.getFileInfo(from: url)
            
            return ValidationResult(
                isValid: true,
                format: fileInfo.format,
                estimatedSplatCount: fileInfo.estimatedSplatCount,
                fileSize: fileInfo.fileSize,
                errors: [],
                warnings: []
            )
        } catch {
            return ValidationResult(
                isValid: false,
                format: url.pathExtension.lowercased(),
                estimatedSplatCount: nil,
                fileSize: nil,
                errors: [error.localizedDescription],
                warnings: []
            )
        }
    }
    
    func parseFile(at url: URL, options: ParsingOptions = .default, progressCallback: ProgressCallback? = nil) async -> ParseResult {
        do {
            let parser = try createParser(for: url)
            return await parser.parseAsync(from: url, options: options, progressCallback: progressCallback)
        } catch {
            if let factoryError = error as? FactoryError {
                return .failure(.unknownError(factoryError.localizedDescription))
            } else {
                return .failure(.ioError(error))
            }
        }
    }
    
    private func registerDefaultParsers() {
        let plyRegistration = StandardParserRegistration(
            identifier: "PLYParser",
            supportedFormats: ["ply"],
            priority: 100
        ) {
            PLYParser()
        }
        
        let splatRegistration = StandardParserRegistration(
            identifier: "SPLATParser",
            supportedFormats: ["splat"],
            priority: 90
        ) {
            SPLATParser()
        }
        
        try? registerParser(plyRegistration)
        try? registerParser(splatRegistration)
    }
}

struct ParserInfo {
    let parserType: String
    let supportedFormats: [String]
    let formatVersion: String?
    let fileInfo: FileInfo
}

struct ValidationResult {
    let isValid: Bool
    let format: String
    let estimatedSplatCount: Int?
    let fileSize: Int?
    let errors: [String]
    let warnings: [String]
    
    var hasWarnings: Bool {
        return !warnings.isEmpty
    }
    
    var hasErrors: Bool {
        return !errors.isEmpty
    }
}

extension GaussianSplatParserFactory {
    func getSupportedFormatsDescription() -> String {
        let formats = getRegisteredFormats()
        if formats.isEmpty {
            return "No supported formats"
        } else if formats.count == 1 {
            return "Supported format: \(formats[0])"
        } else {
            let allButLast = formats.dropLast().joined(separator: ", ")
            let last = formats.last!
            return "Supported formats: \(allButLast) and \(last)"
        }
    }
    
    func registerCustomParser<T: GaussianSplatParser>(_ parserType: T.Type, identifier: String? = nil, priority: Int = 50) throws where T: NSObject {
        let parser = parserType.init()
        let parserId = identifier ?? String(describing: parserType)
        
        let registration = StandardParserRegistration(
            identifier: parserId,
            supportedFormats: parser.supportedFormats,
            priority: priority
        ) {
            parserType.init()
        }
        
        try registerParser(registration)
    }
}
