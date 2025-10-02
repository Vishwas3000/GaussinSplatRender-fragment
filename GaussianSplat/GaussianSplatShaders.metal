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
    uint count;
    uint padding[3]; // Align to 16 bytes, no indices for now
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

// Phase 1: Tile Culling Compute Shader
kernel void buildTiles(
    device GaussianSplat* splats [[buffer(0)]],
    device TileData* tiles [[buffer(1)]],
    constant ViewUniforms& viewUniforms [[buffer(2)]],
    constant TileUniforms& tileUniforms [[buffer(3)]],
    constant uint& splatCount [[buffer(4)]],
    uint2 tileID [[thread_position_in_grid]]
) {
    // Check bounds
    if (tileID.x >= tileUniforms.tilesPerRow || tileID.y >= tileUniforms.tilesPerColumn) {
        return;
    }
    
    uint tileIndex = tileID.y * tileUniforms.tilesPerRow + tileID.x;
    
    // Tile bounds in screen space
    float2 tileMin = float2(tileID) * float(tileUniforms.tileSize);
    float2 tileMax = tileMin + float(tileUniforms.tileSize);
    
    // Convert to NDC
    float2 ndcMin = (tileMin / viewUniforms.screenSize) * 2.0 - 1.0;
    float2 ndcMax = (tileMax / viewUniforms.screenSize) * 2.0 - 1.0;
    ndcMin.y *= -1.0; // Flip Y
    ndcMax.y *= -1.0;
    
    uint count = 0;
    
    // Add safety bounds
    uint safeSplatCount = splatCount < 10000 ? splatCount : 10000; // Prevent excessive loops
    
    // Test each splat against this tile
    for (uint i = 0; i < safeSplatCount && count < tileUniforms.maxSplatsPerTile; i++) {
        GaussianSplat splat = splats[i];
        
        // Project splat to screen space
        float4 clipPos = viewUniforms.viewProjectionMatrix * float4(splat.position, 1.0);
        
        // Skip if behind camera or outside clip space
        if (clipPos.w <= 0.0 || clipPos.z < 0.0 || clipPos.z > clipPos.w) {
            continue;
        }
        
        float3 ndc = clipPos.xyz / clipPos.w;
        
        // Reconstruct 3D covariance matrix from splat data
        float3x3 cov3D = float3x3(
            float3(splat.covariance3D.x, splat.covariance3D.y, splat.covariance3D.z), // [xx, xy, xz]
            float3(splat.covariance3D.y, splat.covariance3D.w, splat.covariance3D_B.x), // [xy, yy, yz]
            float3(splat.covariance3D.z, splat.covariance3D_B.x, splat.covariance3D_B.y)  // [xz, yz, zz]
        );
        
        // Project 3D covariance to 2D for radius calculation
        float4 worldPos4 = float4(splat.position, 1.0);
        float4 viewPos = viewUniforms.viewMatrix * worldPos4;
        float z = -viewPos.z; // Negate because view space Z points toward camera
        
        if (z <= 0.0) continue; // Behind camera
        
        // Project 3D covariance to 2D screen space
        float projScale = 1.0 / max(z, 0.1);
        float2x2 cov2D = float2x2(
            float2(cov3D[0][0] * projScale, cov3D[0][1] * projScale),
            float2(cov3D[1][0] * projScale, cov3D[1][1] * projScale)
        );
        
        // Compute splat radius in screen space (3 sigma)
        float det = cov2D[0][0] * cov2D[1][1] - cov2D[0][1] * cov2D[1][0];
        if (det <= 0.0 || !isfinite(det)) continue;
        
        float trace = cov2D[0][0] + cov2D[1][1];
        float discriminant = trace * trace - 4.0 * det;
        if (discriminant < 0.0 || !isfinite(discriminant)) continue;
        
        float sqrtDisc = sqrt(discriminant);
        if (!isfinite(sqrtDisc)) continue;
        
        float eigenMax = 0.5 * (trace + sqrtDisc);
        if (eigenMax <= 0.0 || !isfinite(eigenMax)) continue;
        
        float radius = 3.0 * sqrt(eigenMax);
        if (!isfinite(radius) || radius > 100.0) continue; // Clamp huge radii
        
        // Convert radius to NDC space
        float minScreenSize = viewUniforms.screenSize.x < viewUniforms.screenSize.y ? viewUniforms.screenSize.x : viewUniforms.screenSize.y;
        float radiusNDC = radius * (2.0 / minScreenSize);
        if (!isfinite(radiusNDC) || radiusNDC > 10.0) continue; // Skip enormous splats
        
        // Check if splat overlaps tile
        float2 splatMin = ndc.xy - radiusNDC;
        float2 splatMax = ndc.xy + radiusNDC;
        
        if (splatMax.x >= ndcMin.x && splatMin.x <= ndcMax.x &&
            splatMax.y >= ndcMin.y && splatMin.y <= ndcMax.y) {
            // Just count for now, no storage of indices
            count++;
        }
    }
    
    tiles[tileIndex].count = count;
}

// Phase 2: Direct splat rendering (no tile culling)
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
    
    float4 accumulated = float4(0.0, 0.0, 0.0, 0.0);
    float alpha = 0.0;
    
    // Process first 100 splats only for safety
    uint safeSplatCount = splatCount < 100 ? splatCount : 100;
    
    // Process all splats directly (bypass tile culling for now)
    for (uint i = 0; i < safeSplatCount && alpha < 0.99; i++) {
        GaussianSplat splat = splats[i];
        
        // Project splat to screen space
        float4 clipPos = viewUniforms.viewProjectionMatrix * float4(splat.position, 1.0);
        if (clipPos.w <= 0.0) continue;
        
        float3 ndcSplat = clipPos.xyz / clipPos.w;
        
        // Distance from pixel to splat center in NDC space
        float2 delta = ndc - ndcSplat.xy;
        
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
        
        // Simple approximation of 3D to 2D covariance projection
        // This projects the 3D covariance by taking the 2x2 submatrix corresponding to X and Y
        // and scaling by the perspective projection factor
        float projScale = 1.0 / max(z, 0.1); // Perspective scaling factor
        
        float2x2 cov2D = float2x2(
            float2(cov3D[0][0] * projScale, cov3D[0][1] * projScale),
            float2(cov3D[1][0] * projScale, cov3D[1][1] * projScale)
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

// Optional: Debug visualization showing tile boundaries
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
    
    // Color based on splat count in tile
    float intensity = float(tile.count) / 100.0; // Normalize to expected max
    float3 heatmapColor = float3(intensity, 1.0 - intensity, 0.0);
    
    // Add grid lines
    if (tilePixel.x < 2 || tilePixel.y < 2) {
        heatmapColor = float3(1.0, 1.0, 1.0);
    }
    
    return float4(heatmapColor, 1.0);
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
