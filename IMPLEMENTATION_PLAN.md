# Morton Code + Radix Sort Implementation Plan

## 🎯 Architecture Overview

### Current Pipeline (INEFFICIENT - TO BE REPLACED)
```
Frame N:
1. clearTiles (13K threads)          → 0.1ms
2. buildTiles (13K threads)           → 15-25ms ❌ BOTTLENECK
   └─ Each tile tests ALL 2M splats   → 26 billion tests
3. gaussianSplatFragment              → 5-10ms
                                Total: ~20-35ms
```

### New Optimized Pipeline (TARGET)
```
Frame N:
1. computeMortonCodes (2M threads)    → 0.3ms
   └─ Project splats, encode Morton codes

2. radixSort (4 passes)               → 0.7ms
   └─ Sort splats by Morton code

3. clearTiles (13K threads)           → 0.1ms

4. buildTilesOptimized (13K threads)  → 0.5ms ✅ OPTIMIZED
   └─ Binary search + test ~150 splats per tile

5. gaussianSplatFragment              → 5-10ms
                                Total: ~6-7ms (4-6× speedup)
```

---

## 📊 Data Structures

### New Buffers Required

```swift
// Morton code data (regenerate each frame as camera moves)
private var mortonCodeBuffer: MTLBuffer!        // uint[splatCount]
private var sortedIndicesBuffer: MTLBuffer!     // uint[splatCount]

// Radix sort temporary buffers
private var mortonCodeTempBuffer: MTLBuffer!    // uint[splatCount] - ping-pong
private var sortedIndicesTempBuffer: MTLBuffer! // uint[splatCount] - ping-pong
private var histogramBuffer: MTLBuffer!         // uint[256 * 4] - 256 bins per pass
private var offsetBuffer: MTLBuffer!            // uint[256 * 4] - prefix sum offsets
```

### Modified Structures

```metal
// Keep existing (no changes needed)
struct GaussianSplat { ... }
struct ViewUniforms { ... }
struct TileUniforms { ... }
struct TileData { ... }
```

---

## 🔧 Implementation Steps

### Step 1: Morton Code Utilities (Metal)

```metal
// Morton code encoding (bit interleaving)
inline uint expandBits(uint v) {
    v = (v | (v << 8)) & 0x00FF00FF;
    v = (v | (v << 4)) & 0x0F0F0F0F;
    v = (v | (v << 2)) & 0x33333333;
    v = (v | (v << 1)) & 0x55555555;
    return v;
}

inline uint encodeMorton2D(uint x, uint y) {
    return expandBits(x) | (expandBits(y) << 1);
}

// Binary search helpers
inline uint binarySearchLower(device uint* array, uint n, uint value) {
    uint left = 0, right = n;
    while (left < right) {
        uint mid = (left + right) / 2;
        if (array[mid] < value) left = mid + 1;
        else right = mid;
    }
    return left;
}

inline uint binarySearchUpper(device uint* array, uint n, uint value) {
    uint left = 0, right = n;
    while (left < right) {
        uint mid = (left + right) / 2;
        if (array[mid] <= value) left = mid + 1;
        else right = mid;
    }
    return left;
}
```

### Step 2: Morton Code Computation Kernel

```metal
kernel void computeMortonCodes(
    device GaussianSplat* splats [[buffer(0)]],
    device uint* mortonCodes [[buffer(1)]],
    device uint* indices [[buffer(2)]],
    constant ViewUniforms& viewUniforms [[buffer(3)]],
    constant uint& splatCount [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= splatCount) return;

    GaussianSplat splat = splats[gid];

    // Project to clip space
    float4 clipPos = viewUniforms.viewProjectionMatrix * float4(splat.position, 1.0);

    // Early reject splats outside frustum (mark with max Morton code)
    if (clipPos.w <= 0.0 || clipPos.z < 0.0 || clipPos.z > clipPos.w) {
        mortonCodes[gid] = 0xFFFFFFFF; // Max value = will sort to end
        indices[gid] = gid;
        return;
    }

    // Convert to NDC
    float2 ndc = clipPos.xy / clipPos.w;

    // Map to 10-bit grid [0, 1023] for Morton encoding
    uint x = clamp(uint((ndc.x * 0.5 + 0.5) * 1024.0), 0u, 1023u);
    uint y = clamp(uint((ndc.y * 0.5 + 0.5) * 1024.0), 0u, 1023u);

    // Encode Morton code (20-bit total)
    mortonCodes[gid] = encodeMorton2D(x, y);
    indices[gid] = gid;
}
```

### Step 3: GPU Radix Sort (4-pass, 8-bit digit)

```metal
// Pass 1: Count occurrences of each digit
kernel void radixSortCount(
    device uint* keys [[buffer(0)]],
    device atomic_uint* histogram [[buffer(1)]],
    constant uint& pass [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;

    uint key = keys[gid];
    uint digit = (key >> (pass * 8)) & 0xFF;
    atomic_fetch_add_explicit(&histogram[digit], 1, memory_order_relaxed);
}

// Pass 2: Prefix sum (exclusive scan) on histogram
kernel void radixSortScan(
    device uint* histogram [[buffer(0)]],
    device uint* offsets [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    // Simple sequential scan (can be parallelized for larger arrays)
    if (gid == 0) {
        uint sum = 0;
        for (uint i = 0; i < 256; i++) {
            offsets[i] = sum;
            sum += histogram[i];
        }
    }
}

// Pass 3: Scatter to sorted positions
kernel void radixSortScatter(
    device uint* keysIn [[buffer(0)]],
    device uint* keysOut [[buffer(1)]],
    device uint* indicesIn [[buffer(2)]],
    device uint* indicesOut [[buffer(3)]],
    device atomic_uint* offsets [[buffer(4)]],
    constant uint& pass [[buffer(5)]],
    constant uint& n [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;

    uint key = keysIn[gid];
    uint index = indicesIn[gid];
    uint digit = (key >> (pass * 8)) & 0xFF;

    uint pos = atomic_fetch_add_explicit(&offsets[digit], 1, memory_order_relaxed);
    keysOut[pos] = key;
    indicesOut[pos] = index;
}
```

### Step 4: Optimized buildTiles with Binary Search

```metal
kernel void buildTilesOptimized(
    device GaussianSplat* splats [[buffer(0)]],
    device uint* sortedIndices [[buffer(1)]],
    device uint* mortonCodes [[buffer(2)]],
    device TileData* tiles [[buffer(3)]],
    constant ViewUniforms& viewUniforms [[buffer(4)]],
    constant TileUniforms& tileUniforms [[buffer(5)]],
    constant uint& splatCount [[buffer(6)]],
    uint2 tileID [[thread_position_in_grid]]
) {
    // Bounds check
    if (tileID.x >= tileUniforms.tilesPerRow || tileID.y >= tileUniforms.tilesPerColumn) {
        return;
    }

    uint tileIndex = tileID.y * tileUniforms.tilesPerRow + tileID.x;

    // Tile bounds in pixels
    uint2 tileMinPixel = tileID * tileUniforms.tileSize;
    uint2 tileMaxPixel = tileMinPixel + tileUniforms.tileSize;

    // Convert to Morton grid coordinates (10-bit: 0-1023)
    // screenSize typically 1024-4096, so divide by 2-4 to map to 1024
    float2 screenSize = viewUniforms.screenSize;
    float scale = 1024.0 / max(screenSize.x, screenSize.y);

    uint2 mortonMin2D = uint2(float2(tileMinPixel) * scale);
    uint2 mortonMax2D = uint2(float2(tileMaxPixel) * scale);

    // Clamp to valid range
    mortonMin2D = clamp(mortonMin2D, uint2(0), uint2(1023));
    mortonMax2D = clamp(mortonMax2D, uint2(0), uint2(1023));

    // Compute Morton code range for this tile
    uint mortonMin = encodeMorton2D(mortonMin2D.x, mortonMin2D.y);
    uint mortonMax = encodeMorton2D(mortonMax2D.x, mortonMax2D.y);

    // Binary search to find splat range
    uint startIdx = binarySearchLower(mortonCodes, splatCount, mortonMin);
    uint endIdx = binarySearchUpper(mortonCodes, splatCount, mortonMax);

    // Expand search slightly to catch splats near boundaries (Morton codes are approximate)
    uint searchMargin = 100; // Test ~100 extra splats on each side
    startIdx = (startIdx > searchMargin) ? (startIdx - searchMargin) : 0;
    endIdx = min(endIdx + searchMargin, splatCount);

    // Precompute tile NDC bounds
    float2 ndcMin = (float2(tileMinPixel) / screenSize) * 2.0 - 1.0;
    float2 ndcMax = (float2(tileMaxPixel) / screenSize) * 2.0 - 1.0;
    ndcMin.y *= -1.0;
    ndcMax.y *= -1.0;

    uint count = 0;
    uint rejected = 0;
    uint workload = 0;

    // Test only spatially-close splats (typically ~150-250 instead of 2M)
    for (uint idx = startIdx; idx < endIdx; idx++) {
        uint i = sortedIndices[idx];

        // Skip if Morton code indicates culled splat
        if (mortonCodes[idx] == 0xFFFFFFFF) {
            rejected++;
            continue;
        }

        GaussianSplat splat = splats[i];

        // === OPTIMIZATION 1: Alpha/Opacity Threshold ===
        if (splat.opacity < 1) {
            rejected++;
            continue;
        }

        // === OPTIMIZATION 2: Adaptive Radius (NEW!) ===
        float4 viewPos = viewUniforms.viewMatrix * float4(splat.position, 1.0);
        float z = -viewPos.z;

        if (z <= 0.0 || z > 100.0) { // Max depth culling
            rejected++;
            continue;
        }

        // Reconstruct covariance for radius calculation
        float cov_xx = splat.covariance3D.x;
        float cov_yy = splat.covariance3D.w;

        float maxExtent = max(cov_xx, cov_yy);
        if (maxExtent <= 0.0 || !isfinite(maxExtent)) {
            rejected++;
            continue;
        }

        // Adaptive radius based on opacity (NEW OPTIMIZATION!)
        float opacity = float(splat.opacity) / 255.0;
        float contributionThreshold = 0.01; // 1% contribution cutoff
        float adaptiveSigma = sqrt(-2.0 * log(contributionThreshold / max(opacity, 0.01)));
        adaptiveSigma = clamp(adaptiveSigma, 1.5, 3.0); // Clamp between 1.5σ and 3σ

        float radiusWorld = adaptiveSigma * sqrt(maxExtent);
        float radiusScreen = radiusWorld / max(z, 0.1);
        float focalLength = screenSize.y * 0.5;
        float radiusPixels = radiusScreen * focalLength;

        if (!isfinite(radiusPixels) || radiusPixels < 1.0) {
            rejected++;
            continue;
        }
        radiusPixels = min(radiusPixels, 1000.0);

        // === OPTIMIZATION 3: Precise Gaussian-Tile Intersection (NEW!) ===
        float4 clipPos = viewUniforms.viewProjectionMatrix * float4(splat.position, 1.0);
        if (clipPos.w <= 0.0) {
            rejected++;
            continue;
        }

        float2 ndc = clipPos.xy / clipPos.w;
        float2 radiusNDC = (radiusPixels / screenSize) * 2.0;

        // Ellipse-rectangle intersection test (more precise than AABB)
        float2 splatCenter = ndc;
        float2 tileCenter = (ndcMin + ndcMax) * 0.5;

        // Find closest point on tile rectangle to splat center
        float2 closestPoint = clamp(splatCenter, ndcMin, ndcMax);
        float2 delta = closestPoint - splatCenter;

        // Check if closest point is within splat radius (circular approximation)
        float dist2 = dot(delta, delta);
        float radius2 = dot(radiusNDC, radiusNDC);

        bool overlaps = dist2 <= radius2;

        if (overlaps) {
            if (count < tileUniforms.maxSplatsPerTile) {
                tiles[tileIndex].splatIndices[count] = i;
                count++;
            }
            workload += uint(radiusPixels);
        } else {
            rejected++;
        }
    }

    // Sort splat indices by depth (back-to-front for alpha blending)
    for (uint i = 1; i < count; i++) {
        uint keyIndex = tiles[tileIndex].splatIndices[i];
        float keyDepth = splats[keyIndex].depth;
        int j = int(i) - 1;

        while (j >= 0 && splats[tiles[tileIndex].splatIndices[j]].depth < keyDepth) {
            tiles[tileIndex].splatIndices[j + 1] = tiles[tileIndex].splatIndices[j];
            j--;
        }
        tiles[tileIndex].splatIndices[j + 1] = keyIndex;
    }

    // Write tile statistics
    tiles[tileIndex].count = count;
    tiles[tileIndex].rejectedCount = rejected;
    tiles[tileIndex].workloadEstimate = workload;
}
```

---

## 🗑️ Code to Remove

### From GaussianSplatShaders.metal:
- ❌ **OLD buildTiles kernel** (lines 85-240) - Replace with buildTilesOptimized
- ✅ **KEEP clearTiles** (still needed)
- ✅ **KEEP gaussianSplatFragment** (no changes needed)
- ✅ **KEEP debug shaders** (if debugging needed, otherwise remove)

### From TiledSplatRenderer.swift:
- ❌ **Remove tileCullingPipeline** - Replace with new optimized pipeline
- ✅ **KEEP renderPipeline** (fragment shader unchanged)
- ⚠️ **Debug pipelines** - Keep only if actively using, otherwise remove

---

## 📋 Swift Integration Changes

### New Pipeline States Needed:

```swift
// Compute pipelines
private var mortonCodePipeline: MTLComputePipelineState!
private var radixSortCountPipeline: MTLComputePipelineState!
private var radixSortScanPipeline: MTLComputePipelineState!
private var radixSortScatterPipeline: MTLComputePipelineState!
private var clearTilesPipeline: MTLComputePipelineState! // Keep existing
private var buildTilesOptimizedPipeline: MTLComputePipelineState! // NEW
```

### New Buffer Setup:

```swift
private func setupMortonBuffers() {
    let splatCount = splats.count

    // Morton codes and indices
    mortonCodeBuffer = device.makeBuffer(
        length: splatCount * MemoryLayout<UInt32>.stride,
        options: .storageModeShared
    )

    sortedIndicesBuffer = device.makeBuffer(
        length: splatCount * MemoryLayout<UInt32>.stride,
        options: .storageModeShared
    )

    // Temporary buffers for radix sort ping-pong
    mortonCodeTempBuffer = device.makeBuffer(
        length: splatCount * MemoryLayout<UInt32>.stride,
        options: .storageModeShared
    )

    sortedIndicesTempBuffer = device.makeBuffer(
        length: splatCount * MemoryLayout<UInt32>.stride,
        options: .storageModeShared
    )

    // Radix sort working buffers
    histogramBuffer = device.makeBuffer(
        length: 256 * 4 * MemoryLayout<UInt32>.stride, // 256 bins × 4 passes
        options: .storageModeShared
    )

    offsetBuffer = device.makeBuffer(
        length: 256 * 4 * MemoryLayout<UInt32>.stride,
        options: .storageModeShared
    )
}
```

### New Render Loop:

```swift
func draw(in view: MTKView) {
    // ... existing setup ...

    // PHASE 1: Compute Morton Codes
    if let computeEncoder = commandBuffer.makeComputeCommandEncoder() {
        computeEncoder.setComputePipelineState(mortonCodePipeline)
        computeEncoder.setBuffer(splatBuffer, offset: 0, index: 0)
        computeEncoder.setBuffer(mortonCodeBuffer, offset: 0, index: 1)
        computeEncoder.setBuffer(sortedIndicesBuffer, offset: 0, index: 2)
        computeEncoder.setBuffer(viewUniformsBuffer, offset: 0, index: 3)
        computeEncoder.setBuffer(splatCountBuffer, offset: 0, index: 4)

        let threadsPerGrid = MTLSize(width: splats.count, height: 1, depth: 1)
        let threadsPerThreadgroup = MTLSize(width: 256, height: 1, depth: 1)
        computeEncoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        computeEncoder.endEncoding()
    }

    // PHASE 2: Radix Sort (4 passes)
    performRadixSort(commandBuffer: commandBuffer, splatCount: splats.count)

    // PHASE 3: Clear Tiles (existing)
    if let computeEncoder = commandBuffer.makeComputeCommandEncoder() {
        // ... existing clearTiles code ...
    }

    // PHASE 4: Build Tiles Optimized
    if let computeEncoder = commandBuffer.makeComputeCommandEncoder() {
        computeEncoder.setComputePipelineState(buildTilesOptimizedPipeline)
        computeEncoder.setBuffer(splatBuffer, offset: 0, index: 0)
        computeEncoder.setBuffer(sortedIndicesBuffer, offset: 0, index: 1)
        computeEncoder.setBuffer(mortonCodeBuffer, offset: 0, index: 2)
        computeEncoder.setBuffer(tileBuffer, offset: 0, index: 3)
        computeEncoder.setBuffer(viewUniformsBuffer, offset: 0, index: 4)
        computeEncoder.setBuffer(tileUniformsBuffer, offset: 0, index: 5)
        computeEncoder.setBuffer(splatCountBuffer, offset: 0, index: 6)

        let threadsPerThreadgroup = MTLSize(width: 8, height: 8, depth: 1)
        let threadgroupsPerGrid = MTLSize(
            width: (Int(tileUniforms.tilesPerRow) + 7) / 8,
            height: (Int(tileUniforms.tilesPerColumn) + 7) / 8,
            depth: 1
        )
        computeEncoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        computeEncoder.endEncoding()
    }

    // PHASE 5: Render (existing)
    // ... existing render pass code ...
}
```

---

## ⚡ Performance Monitoring

Add timing instrumentation:

```swift
private var performanceMetrics = PerformanceMetrics()

struct PerformanceMetrics {
    var mortonCodeTime: Double = 0
    var radixSortTime: Double = 0
    var clearTilesTime: Double = 0
    var buildTilesTime: Double = 0
    var renderTime: Double = 0
    var totalFrameTime: Double = 0
}

// Use Metal GPU timestamps or CPU timing
commandBuffer.addCompletedHandler { [weak self] _ in
    // Print metrics every 60 frames
    if frameCount % 60 == 0 {
        print("Performance: Morton: \(mortonCodeTime)ms, Sort: \(radixSortTime)ms, Build: \(buildTilesTime)ms, Total: \(totalFrameTime)ms")
    }
}
```

---

## ✅ Success Criteria

1. **Correctness:** Visual output identical to current implementation
2. **Performance:**
   - Morton encoding: < 0.5ms
   - Radix sort: < 1.0ms
   - Build tiles: < 1.0ms (down from 15-25ms)
   - Total frame time: < 8ms (120+ FPS)
3. **Code Quality:** No legacy/unused code remains

---

## 🚀 Implementation Order

1. ✅ Create this plan document
2. 🔄 Implement Morton code utilities in Metal
3. 🔄 Implement radix sort kernels
4. 🔄 Implement buildTilesOptimized kernel
5. 🔄 Update Swift renderer with new pipeline
6. 🔄 Test and debug
7. 🔄 Remove old buildTiles kernel
8. 🔄 Clean up unused debug code (if not needed)
9. 🔄 Benchmark and validate performance
10. 🔄 Document final results