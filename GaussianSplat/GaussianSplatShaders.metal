#include <metal_stdlib>
using namespace metal;

struct GaussianSplat {
    float3 position;
    float4 covariance3D;    // [xx, xy, xz, yy]
    float2 covariance3D_B;  // [yz, zz]
    uchar3 color;
    uchar opacity;
    float depth;
};

struct ViewUniforms {
    float4x4 viewMatrix;
    float4x4 projectionMatrix;
    float4x4 viewProjectionMatrix;
    float3 cameraPosition;
    float time;
    float2 screenSize;
    float2 padding;
};

struct TileUniforms {
    uint tilesPerRow;
    uint tilesPerColumn;
    uint tileSize;
    uint maxSplatsPerTile;
};

struct TileData {
    uint count;            // Number of splats assigned to this tile
    uint rejectedCount;    // Number of splats rejected (culled) from this tile
    uint workloadEstimate; // Estimated GPU workload (for performance visualization)
    uint maxDepth;         // Maximum depth of splats in this tile (for debugging)
    uint splatIndices[64]; // Array of splat indices assigned to this tile (max 64)
};

struct VertexOut {
    float4 position [[position]];
    float2 uv;
};

// Simple vertex shader for fullscreen quad
vertex VertexOut fullscreenVertex(uint vertexID [[vertex_id]]) {
    VertexOut out;
    
    // Generate fullscreen triangle
    float2 uv = float2((vertexID << 1) & 2, vertexID & 2);
    out.position = float4(uv * 2.0 - 1.0, 0.0, 1.0);
    out.uv = uv;
    
    return out;
}

// Phase 1a: Clear Tiles Kernel
kernel void clearTiles(
    device TileData* tiles [[buffer(0)]],
    constant TileUniforms& tileUniforms [[buffer(1)]],
    uint tileID [[thread_position_in_grid]]
) {
    uint totalTiles = tileUniforms.tilesPerRow * tileUniforms.tilesPerColumn;
    if (tileID >= totalTiles) {
        return;
    }

    tiles[tileID].count = 0;
    tiles[tileID].rejectedCount = 0;
    tiles[tileID].workloadEstimate = 0;
    tiles[tileID].maxDepth = 0;
}

// Phase 1c: Finalize tile counts from atomic counters
kernel void finalizeTileCounts(
    device atomic_uint* tileCountsAtomic [[buffer(0)]],
    device TileData* tiles [[buffer(1)]],
    constant TileUniforms& tileUniforms [[buffer(2)]],
    uint tileID [[thread_position_in_grid]]
) {
    uint totalTiles = tileUniforms.tilesPerRow * tileUniforms.tilesPerColumn;
    if (tileID >= totalTiles) {
        return;
    }

    // Read atomic counter and clamp to maxSplatsPerTile
    uint count = atomic_load_explicit(&tileCountsAtomic[tileID], memory_order_relaxed);
    tiles[tileID].count = min(count, tileUniforms.maxSplatsPerTile);
}

// Phase 1b: Build Tiles - Splat-Centric Approach with Atomics
// Each thread processes ONE splat and atomically updates all tiles it overlaps
// This is more efficient when: num_splats << (num_tiles * avg_splats_per_tile)
kernel void buildTiles(
    device GaussianSplat* splats [[buffer(0)]],
    device atomic_uint* tileCountsAtomic [[buffer(1)]], // Atomic counters (one per tile)
    device TileData* tiles [[buffer(2)]],                // Final tile data
    constant ViewUniforms& viewUniforms [[buffer(3)]],
    constant TileUniforms& tileUniforms [[buffer(4)]],
    constant uint& splatCount [[buffer(5)]],
    uint splatID [[thread_position_in_grid]]
) {
    // Check bounds - one thread per splat
    if (splatID >= splatCount) {
        return;
    }

    GaussianSplat splat = splats[splatID];

    // Project splat to screen space
    float4 clipPos = viewUniforms.viewProjectionMatrix * float4(splat.position, 1.0);

    // Skip if behind camera or outside clip space
    if (clipPos.w <= 0.0 || clipPos.z < 0.0 || clipPos.z > clipPos.w) {
        return; // Early exit - splat not visible
    }

    float3 ndc = clipPos.xyz / clipPos.w;

    // Reconstruct 3D covariance matrix
    float3x3 cov3D = float3x3(
        float3(splat.covariance3D.x, splat.covariance3D.y, splat.covariance3D.z),
        float3(splat.covariance3D.y, splat.covariance3D.w, splat.covariance3D_B.x),
        float3(splat.covariance3D.z, splat.covariance3D_B.x, splat.covariance3D_B.y)
    );

    // Project 3D covariance to 2D for radius calculation
    float4 viewPos = viewUniforms.viewMatrix * float4(splat.position, 1.0);
    float z = -viewPos.z;

    if (z <= 0.0) {
        return; // Behind camera
    }

    // Compute screen-space radius
    float maxExtent = max(cov3D[0][0], cov3D[1][1]);
    if (maxExtent <= 0.0 || !isfinite(maxExtent)) {
        return;
    }

    float radiusWorld = 3.0 * sqrt(maxExtent);
    float radiusScreen = radiusWorld / max(z, 0.1);
    float focalLength = viewUniforms.screenSize.y * 0.5;
    float radiusPixels = radiusScreen * focalLength;

    if (!isfinite(radiusPixels) || radiusPixels < 0.5) {
        return; // Too small
    }
    if (radiusPixels > 1000.0) {
        radiusPixels = 1000.0;
    }

    // Convert splat center to pixel coordinates
    float2 splatPixel = ((ndc.xy * float2(1.0, -1.0) + 1.0) * 0.5) * viewUniforms.screenSize;

    // Calculate bounding box of tiles this splat overlaps
    float2 splatMinPixel = splatPixel - radiusPixels;
    float2 splatMaxPixel = splatPixel + radiusPixels;

    int2 tileMin = int2(max(splatMinPixel / float(tileUniforms.tileSize), float2(0.0)));
    int2 tileMax = int2(min(splatMaxPixel / float(tileUniforms.tileSize),
                            float2(tileUniforms.tilesPerRow - 1, tileUniforms.tilesPerColumn - 1)));

    uint workload = uint(radiusPixels);

    // Iterate over all overlapping tiles
    for (int ty = tileMin.y; ty <= tileMax.y; ty++) {
        for (int tx = tileMin.x; tx <= tileMax.x; tx++) {
            uint tileIndex = uint(ty) * tileUniforms.tilesPerRow + uint(tx);

            // Atomically get the next available slot in this tile's splat list
            uint slot = atomic_fetch_add_explicit(&tileCountsAtomic[tileIndex], 1, memory_order_relaxed);

            // Only store if there's space in the tile
            if (slot < tileUniforms.maxSplatsPerTile) {
                tiles[tileIndex].splatIndices[slot] = splatID;

                // Also accumulate workload atomically
                atomic_fetch_add_explicit((device atomic_uint*)&tiles[tileIndex].workloadEstimate,
                                         workload, memory_order_relaxed);
            }
        }
    }
}

// Phase 2: Tile-based splat rendering using per-tile indices
fragment float4 gaussianSplatFragment(
    VertexOut in [[stage_in]],
    device GaussianSplat* splats [[buffer(0)]],
    device TileData* tiles [[buffer(1)]],
    constant ViewUniforms& viewUniforms [[buffer(2)]],
    constant TileUniforms& tileUniforms [[buffer(3)]],
    constant uint& splatCount [[buffer(4)]]
) {
    // Convert pixel to NDC
    float2 ndc = (in.position.xy / viewUniforms.screenSize) * 2.0 - 1.0;
    ndc.y *= -1.0; // Flip Y

    // Determine which tile this pixel belongs to
    uint2 pixelCoord = uint2(in.position.xy);
    uint2 tileID = pixelCoord / tileUniforms.tileSize;

    // Check if pixel is within valid tile bounds
    if (tileID.x >= tileUniforms.tilesPerRow || tileID.y >= tileUniforms.tilesPerColumn) {
        return float4(0.0, 0.0, 0.0, 1.0);
    }

    uint tileIndex = tileID.y * tileUniforms.tilesPerRow + tileID.x;
    TileData tile = tiles[tileIndex];

    float4 accumulated = float4(0.0, 0.0, 0.0, 0.0);
    float alpha = 0.0;

    // Process only the splats assigned to this tile (up to maxSplatsPerTile)
    uint numSplats = min(tile.count, tileUniforms.maxSplatsPerTile);

    // Process splats from this tile's index list
    for (uint i = 0; i < numSplats && alpha < 0.99; i++) {
        uint splatIndex = tile.splatIndices[i];

        // Bounds check
        if (splatIndex >= splatCount) continue;

        GaussianSplat splat = splats[splatIndex];

        // Project splat to screen space
        float4 clipPos = viewUniforms.viewProjectionMatrix * float4(splat.position, 1.0);
        if (clipPos.w <= 0.0) continue;

        float3 ndcSplat = clipPos.xyz / clipPos.w;

        // Convert NDC to pixel coordinates for both splat and current pixel
        float2 pixelSplat = ((ndcSplat.xy * float2(1.0, -1.0) + 1.0) * 0.5) * viewUniforms.screenSize;
        float2 pixelCurrent = in.position.xy;

        // Distance from pixel to splat center in PIXEL space (not NDC)
        float2 delta = pixelCurrent - pixelSplat;
        
        // Reconstruct 3D covariance matrix from splat data
        float3x3 cov3D = float3x3(
            float3(splat.covariance3D.x, splat.covariance3D.y, splat.covariance3D.z), // [xx, xy, xz]
            float3(splat.covariance3D.y, splat.covariance3D.w, splat.covariance3D_B.x), // [xy, yy, yz]
            float3(splat.covariance3D.z, splat.covariance3D_B.x, splat.covariance3D_B.y)  // [xz, yz, zz]
        );
        
        // Project 3D covariance to 2D screen space using Jacobian of projection
        // For perspective projection, the Jacobian is approximately:
        // J = (focal/z) * [1 0; 0 1] for points near the screen center
        
        float4 worldPos4 = float4(splat.position, 1.0);
        float4 viewPos = viewUniforms.viewMatrix * worldPos4;
        float z = -viewPos.z; // Negate because view space Z points toward camera
        
        if (z <= 0.0) continue; // Behind camera

        // Project 3D covariance to 2D screen space
        // We need to properly account for perspective projection
        // The covariance in NDC space is approximately: cov2D = J * cov3D * J^T
        // where J is the Jacobian of the projection (focal_length / z)

        float focalLength = viewUniforms.screenSize.y * 0.5; // Approximate focal length
        float scale = focalLength / max(z, 0.1);              // Projection scale

        // Project the 3D covariance's XY submatrix to 2D
        // Scale by (focal/z)^2 to get proper screen-space covariance
        float scale2 = scale * scale;
        float2x2 cov2D = float2x2(
            float2(cov3D[0][0] * scale2, cov3D[0][1] * scale2),
            float2(cov3D[1][0] * scale2, cov3D[1][1] * scale2)
        );

        // Compute determinant and check validity
        float det = cov2D[0][0] * cov2D[1][1] - cov2D[0][1] * cov2D[1][0];
        if (det <= 0.0 || !isfinite(det)) continue;
        
        // Invert 2x2 covariance matrix: inv(Σ) = (1/det) * [d -b; -c a]
        float invDet = 1.0 / det;
        float2x2 invCov = float2x2(
            float2(cov2D[1][1] * invDet, -cov2D[0][1] * invDet),
            float2(-cov2D[1][0] * invDet, cov2D[0][0] * invDet)
        );
        
        // Evaluate 2D Gaussian: exp(-0.5 * delta^T * Σ^-1 * delta)
        float2 temp = invCov * delta;
        float exponent = -0.5 * dot(delta, temp);
        
        // Early exit if contribution is negligible (3-sigma rule)
        if (exponent < -4.5) continue; // exp(-4.5) ≈ 0.011
        
        float gaussian = exp(exponent);
        
        // Convert color and opacity to float
        float3 splatColor = float3(splat.color) / 255.0;
        float splatOpacity = float(splat.opacity) / 255.0;
        
        float contribution = gaussian * splatOpacity * 0.5; // Reduce opacity for visibility
        
        // Alpha blending
        float3 blendedColor = splatColor * contribution;
        accumulated.rgb += blendedColor * (1.0 - alpha);
        alpha += contribution * (1.0 - alpha);
    }
    
    // Background color (dark blue to black gradient)
    float2 gradientUV = in.uv;
    float3 backgroundColor = mix(
        float3(0.05, 0.05, 0.15), // Dark blue at top
        float3(0.0, 0.0, 0.0),    // Black at bottom
        gradientUV.y
    );
    
    // Composite with background
    float3 finalColor = accumulated.rgb + backgroundColor * (1.0 - alpha);
    
    return float4(finalColor, 1.0);
}

// Enhanced tile visualization shaders for performance analysis

// Helper function: Create smooth heatmap color gradient
float3 heatmapGradient(float value) {
    // 5-color gradient: black -> blue -> cyan -> yellow -> red
    value = clamp(value, 0.0, 1.0);

    if (value < 0.25) {
        // Black to Blue
        return mix(float3(0.0, 0.0, 0.0), float3(0.0, 0.0, 1.0), value * 4.0);
    } else if (value < 0.5) {
        // Blue to Cyan
        return mix(float3(0.0, 0.0, 1.0), float3(0.0, 1.0, 1.0), (value - 0.25) * 4.0);
    } else if (value < 0.75) {
        // Cyan to Yellow
        return mix(float3(0.0, 1.0, 1.0), float3(1.0, 1.0, 0.0), (value - 0.5) * 4.0);
    } else {
        // Yellow to Red
        return mix(float3(1.0, 1.0, 0.0), float3(1.0, 0.0, 0.0), (value - 0.75) * 4.0);
    }
}

// Tile heatmap based on splat count with enhanced visualization
fragment float4 debugTileHeatmapFragment(
    VertexOut in [[stage_in]],
    device TileData* tiles [[buffer(1)]],
    constant TileUniforms& tileUniforms [[buffer(3)]]
) {
    uint2 pixelCoord = uint2(in.position.xy);
    uint2 tileID = pixelCoord / tileUniforms.tileSize;
    uint2 tilePixel = pixelCoord % tileUniforms.tileSize;

    if (tileID.x >= tileUniforms.tilesPerRow || tileID.y >= tileUniforms.tilesPerColumn) {
        return float4(0.0, 0.0, 0.0, 1.0);
    }

    uint tileIndex = tileID.y * tileUniforms.tilesPerRow + tileID.x;
    TileData tile = tiles[tileIndex];

    // Normalize splat count (0-64 range maps to 0-1)
    float normalizedCount = clamp(float(tile.count) / float(tileUniforms.maxSplatsPerTile), 0.0, 1.0);

    // Create smooth heatmap color
    float3 heatmapColor = heatmapGradient(normalizedCount);

    // Highlight tiles at max capacity with pulsing white border
    bool isMaxCapacity = tile.count >= tileUniforms.maxSplatsPerTile;

    // Add prominent grid lines for tile boundaries
    bool isBorder = (tilePixel.x < 2 || tilePixel.y < 2);
    if (isBorder) {
        if (isMaxCapacity) {
            // Bright white for maxed out tiles
            heatmapColor = float3(1.0, 1.0, 1.0);
        } else {
            // Gray grid
            heatmapColor = mix(heatmapColor, float3(0.5, 0.5, 0.5), 0.7);
        }
    }

    // Add center cross to show exact tile centers (for debugging alignment)
    uint2 tileCenter = tileUniforms.tileSize / 2;
    bool isCenterCross = (tilePixel.x == tileCenter.x && abs(int(tilePixel.y) - int(tileCenter.y)) < 2) ||
                         (tilePixel.y == tileCenter.y && abs(int(tilePixel.x) - int(tileCenter.x)) < 2);

    if (isCenterCross && tile.count > 0) {
        heatmapColor = mix(heatmapColor, float3(1.0, 1.0, 1.0), 0.3);
    }

    return float4(heatmapColor, 0.85); // Semi-transparent overlay
}

// Workload visualization based on estimated GPU work and complexity
fragment float4 debugTileWorkloadFragment(
    VertexOut in [[stage_in]],
    device TileData* tiles [[buffer(1)]],
    constant TileUniforms& tileUniforms [[buffer(3)]]
) {
    uint2 pixelCoord = uint2(in.position.xy);
    uint2 tileID = pixelCoord / tileUniforms.tileSize;
    uint2 tilePixel = pixelCoord % tileUniforms.tileSize;

    if (tileID.x >= tileUniforms.tilesPerRow || tileID.y >= tileUniforms.tilesPerColumn) {
        return float4(0.0, 0.0, 0.0, 1.0);
    }

    uint tileIndex = tileID.y * tileUniforms.tilesPerRow + tileID.x;
    TileData tile = tiles[tileIndex];

    // Normalize workload estimate (0-20000 range is typical)
    float workloadNormalized = clamp(float(tile.workloadEstimate) / 20000.0, 0.0, 1.0);

    // Use smooth heatmap gradient
    float3 heatmapColor = heatmapGradient(workloadNormalized);

    // Show workload distribution with intensity bars
    uint workloadBarHeight = uint(workloadNormalized * float(tileUniforms.tileSize));
    bool isInWorkloadBar = (tileUniforms.tileSize - tilePixel.y) <= workloadBarHeight;

    if (isInWorkloadBar && tilePixel.x > 1 && tilePixel.x < (tileUniforms.tileSize - 1)) {
        // Brighten the bar area
        heatmapColor = mix(heatmapColor, float3(1.0, 1.0, 1.0), 0.2);
    }

    // Highlight high workload tiles
    if (workloadNormalized > 0.8) {
        bool isWarningBorder = (tilePixel.x < 3 || tilePixel.y < 3 ||
                                tilePixel.x >= tileUniforms.tileSize - 3 ||
                                tilePixel.y >= tileUniforms.tileSize - 3);
        if (isWarningBorder) {
            heatmapColor = float3(1.0, 0.0, 0.0); // Red warning border
        }
    }

    // Add grid borders
    bool isBorder = (tilePixel.x < 2 || tilePixel.y < 2);
    if (isBorder) {
        heatmapColor = mix(heatmapColor, float3(0.5, 0.5, 0.5), 0.6);
    }

    return float4(heatmapColor, 0.85);
}

// Culling efficiency visualization showing accepted vs rejected splats
fragment float4 debugTileCullingFragment(
    VertexOut in [[stage_in]],
    device TileData* tiles [[buffer(1)]],
    constant TileUniforms& tileUniforms [[buffer(3)]]
) {
    uint2 pixelCoord = uint2(in.position.xy);
    uint2 tileID = pixelCoord / tileUniforms.tileSize;
    uint2 tilePixel = pixelCoord % tileUniforms.tileSize;

    if (tileID.x >= tileUniforms.tilesPerRow || tileID.y >= tileUniforms.tilesPerColumn) {
        return float4(0.0, 0.0, 0.0, 1.0);
    }

    uint tileIndex = tileID.y * tileUniforms.tilesPerRow + tileID.x;
    TileData tile = tiles[tileIndex];

    uint totalProcessed = tile.count + tile.rejectedCount;

    // Calculate ratios
    float acceptedRatio = totalProcessed > 0 ? float(tile.count) / float(totalProcessed) : 0.0;
    float rejectedRatio = totalProcessed > 0 ? float(tile.rejectedCount) / float(totalProcessed) : 0.0;

    // Create visualization showing the split
    // Green channel = accepted splats
    // Red channel = rejected splats
    // Blue channel = activity level
    float3 heatmapColor;

    if (totalProcessed == 0) {
        // No activity - dark gray
        heatmapColor = float3(0.1, 0.1, 0.1);
    } else {
        // Show culling effectiveness
        // High rejection ratio = good culling (more blue-green)
        // Low rejection ratio = bad culling (more yellow-red)
        float activityLevel = clamp(float(totalProcessed) / 100.0, 0.0, 1.0);

        heatmapColor = float3(
            rejectedRatio,      // Red: rejected splats
            acceptedRatio * 0.8, // Green: accepted splats
            activityLevel * rejectedRatio // Blue: culling efficiency
        );
    }

    // Add visual split bar showing ratio
    uint splitPosition = uint(acceptedRatio * float(tileUniforms.tileSize));
    if (tilePixel.x == splitPosition) {
        heatmapColor = float3(1.0, 1.0, 1.0); // White line showing split
    }

    // Add grid borders
    bool isBorder = (tilePixel.x < 2 || tilePixel.y < 2);
    if (isBorder) {
        heatmapColor = mix(heatmapColor, float3(0.5, 0.5, 0.5), 0.6);
    }

    return float4(heatmapColor, 0.85);
}

// Legacy debug tiles fragment (kept for compatibility)
fragment float4 debugTilesFragment(
    VertexOut in [[stage_in]],
    device TileData* tiles [[buffer(1)]],
    constant TileUniforms& tileUniforms [[buffer(3)]]
) {
    uint2 pixelCoord = uint2(in.position.xy);
    uint2 tileID = pixelCoord / tileUniforms.tileSize;
    uint2 tilePixel = pixelCoord % tileUniforms.tileSize;

    if (tileID.x >= tileUniforms.tilesPerRow || tileID.y >= tileUniforms.tilesPerColumn) {
        return float4(0.0, 0.0, 0.0, 1.0);
    }

    uint tileIndex = tileID.y * tileUniforms.tilesPerRow + tileID.x;
    TileData tile = tiles[tileIndex];

    // Create heat map based on splat count (blue = low, red = high)
    float intensity = clamp(float(tile.count) / 200.0, 0.0, 1.0);
    float3 heatmapColor = float3(intensity, 0.0, 1.0 - intensity);

    // Add grid lines for tile boundaries
    if (tilePixel.x < 1 || tilePixel.y < 1) {
        heatmapColor = mix(heatmapColor, float3(1.0, 1.0, 1.0), 0.5);
    }

    return float4(heatmapColor, 0.8); // Semi-transparent overlay
}

// Debug line rendering shaders
struct DebugLineVertex {
    float3 position [[attribute(0)]];
    float3 color [[attribute(1)]];
};

struct DebugLineVertexOut {
    float4 position [[position]];
    float3 color;
};

vertex DebugLineVertexOut debugLineVertex(
    DebugLineVertex in [[stage_in]],
    constant ViewUniforms& viewUniforms [[buffer(1)]]
) {
    DebugLineVertexOut out;
    out.position = viewUniforms.viewProjectionMatrix * float4(in.position, 1.0);
    out.color = in.color;
    return out;
}

fragment float4 debugLineFragment(DebugLineVertexOut in [[stage_in]]) {
    return float4(in.color, 1.0);
}
