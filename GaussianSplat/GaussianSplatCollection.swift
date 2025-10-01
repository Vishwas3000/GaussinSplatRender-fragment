import Foundation
import simd

struct SplatMetadata {
    let sourceFile: URL?
    let fileFormat: String
    let creationDate: Date
    let totalSplats: Int
    let compressionRatio: Float?
    let originalFileSize: Int?
    
    init(sourceFile: URL? = nil, 
         fileFormat: String, 
         totalSplats: Int, 
         compressionRatio: Float? = nil, 
         originalFileSize: Int? = nil) {
        self.sourceFile = sourceFile
        self.fileFormat = fileFormat
        self.creationDate = Date()
        self.totalSplats = totalSplats
        self.compressionRatio = compressionRatio
        self.originalFileSize = originalFileSize
    }
}

class GaussianSplatCollection {
    private(set) var splats: [GaussianSplat]
    private(set) var metadata: SplatMetadata
    private var _boundingBox: (min: SIMD3<Float>, max: SIMD3<Float>)?
    
    init(splats: [GaussianSplat] = [], metadata: SplatMetadata) {
        self.splats = splats
        self.metadata = metadata
    }
    
    convenience init(fileFormat: String) {
        let metadata = SplatMetadata(fileFormat: fileFormat, totalSplats: 0)
        self.init(splats: [], metadata: metadata)
    }
    
    var count: Int {
        return splats.count
    }
    
    var isEmpty: Bool {
        return splats.isEmpty
    }
    
    var boundingBox: (min: SIMD3<Float>, max: SIMD3<Float>) {
        if let cached = _boundingBox {
            return cached
        }
        
        guard !splats.isEmpty else {
            let zero = SIMD3<Float>(0, 0, 0)
            return (min: zero, max: zero)
        }
        
        var minBounds = splats[0].position
        var maxBounds = splats[0].position
        
        for splat in splats {
            let bounds = splat.boundingBox
            minBounds = simd_min(minBounds, bounds.min)
            maxBounds = simd_max(maxBounds, bounds.max)
        }
        
        _boundingBox = (min: minBounds, max: maxBounds)
        return _boundingBox!
    }
    
    var center: SIMD3<Float> {
        let bounds = boundingBox
        return (bounds.min + bounds.max) * 0.5
    }
    
    var totalMemorySize: Int {
        return splats.reduce(0) { $0 + $1.memorySize }
    }
    
    func add(_ splat: GaussianSplat) {
        splats.append(splat)
        invalidateBoundingBox()
        updateMetadata()
    }
    
    func addBatch(_ newSplats: [GaussianSplat]) {
        splats.append(contentsOf: newSplats)
        invalidateBoundingBox()
        updateMetadata()
    }
    
    func remove(at index: Int) {
        guard index >= 0 && index < splats.count else { return }
        splats.remove(at: index)
        invalidateBoundingBox()
        updateMetadata()
    }
    
    func removeAll() {
        splats.removeAll()
        invalidateBoundingBox()
        updateMetadata()
    }
    
    func filter(_ predicate: (GaussianSplat) -> Bool) -> GaussianSplatCollection {
        let filteredSplats = splats.filter(predicate)
        let newMetadata = SplatMetadata(
            sourceFile: metadata.sourceFile,
            fileFormat: metadata.fileFormat + "_filtered",
            totalSplats: filteredSplats.count
        )
        return GaussianSplatCollection(splats: filteredSplats, metadata: newMetadata)
    }
    
    func sortByDistance(from point: SIMD3<Float>) {
        splats.sort { splat1, splat2 in
            let dist1 = distance_squared(splat1.position, point)
            let dist2 = distance_squared(splat2.position, point)
            return dist1 < dist2
        }
    }
    
    func getSplatsInRadius(center: SIMD3<Float>, radius: Float) -> [GaussianSplat] {
        let radiusSquared = radius * radius
        return splats.filter { splat in
            distance_squared(splat.position, center) <= radiusSquared
        }
    }
    
    func getStatistics() -> SplatStatistics {
        guard !splats.isEmpty else {
            return SplatStatistics()
        }
        
        var totalOpacity: Float = 0
        var minOpacity: Float = splats[0].opacity
        var maxOpacity: Float = splats[0].opacity
        
        var totalScale = SIMD3<Float>(0, 0, 0)
        var minScale = splats[0].scale
        var maxScale = splats[0].scale
        
        for splat in splats {
            totalOpacity += splat.opacity
            minOpacity = Swift.min(minOpacity, splat.opacity)
            maxOpacity = Swift.max(maxOpacity, splat.opacity)
            
            totalScale += splat.scale
            minScale = simd_min(minScale, splat.scale)
            maxScale = simd_max(maxScale, splat.scale)
        }
        
        let avgOpacity = totalOpacity / Float(splats.count)
        let avgScale = totalScale / Float(splats.count)
        
        return SplatStatistics(
            count: splats.count,
            averageOpacity: avgOpacity,
            minOpacity: minOpacity,
            maxOpacity: maxOpacity,
            averageScale: avgScale,
            minScale: minScale,
            maxScale: maxScale,
            boundingBox: boundingBox,
            totalMemorySize: totalMemorySize
        )
    }
    
    private func invalidateBoundingBox() {
        _boundingBox = nil
    }
    
    private func updateMetadata() {
        metadata = SplatMetadata(
            sourceFile: metadata.sourceFile,
            fileFormat: metadata.fileFormat,
            totalSplats: splats.count,
            compressionRatio: metadata.compressionRatio,
            originalFileSize: metadata.originalFileSize
        )
    }
}

struct SplatStatistics {
    let count: Int
    let averageOpacity: Float
    let minOpacity: Float
    let maxOpacity: Float
    let averageScale: SIMD3<Float>
    let minScale: SIMD3<Float>
    let maxScale: SIMD3<Float>
    let boundingBox: (min: SIMD3<Float>, max: SIMD3<Float>)
    let totalMemorySize: Int
    
    init() {
        self.count = 0
        self.averageOpacity = 0
        self.minOpacity = 0
        self.maxOpacity = 0
        self.averageScale = SIMD3<Float>(0, 0, 0)
        self.minScale = SIMD3<Float>(0, 0, 0)
        self.maxScale = SIMD3<Float>(0, 0, 0)
        self.boundingBox = (min: SIMD3<Float>(0, 0, 0), max: SIMD3<Float>(0, 0, 0))
        self.totalMemorySize = 0
    }
    
    init(count: Int, averageOpacity: Float, minOpacity: Float, maxOpacity: Float, 
         averageScale: SIMD3<Float>, minScale: SIMD3<Float>, maxScale: SIMD3<Float>, 
         boundingBox: (min: SIMD3<Float>, max: SIMD3<Float>), totalMemorySize: Int) {
        self.count = count
        self.averageOpacity = averageOpacity
        self.minOpacity = minOpacity
        self.maxOpacity = maxOpacity
        self.averageScale = averageScale
        self.minScale = minScale
        self.maxScale = maxScale
        self.boundingBox = boundingBox
        self.totalMemorySize = totalMemorySize
    }
}

extension GaussianSplatCollection: Sequence {
    func makeIterator() -> Array<GaussianSplat>.Iterator {
        return splats.makeIterator()
    }
}

extension GaussianSplatCollection: Collection {
    var startIndex: Int { return splats.startIndex }
    var endIndex: Int { return splats.endIndex }
    
    subscript(index: Int) -> GaussianSplat {
        return splats[index]
    }
    
    func index(after i: Int) -> Int {
        return splats.index(after: i)
    }
}
