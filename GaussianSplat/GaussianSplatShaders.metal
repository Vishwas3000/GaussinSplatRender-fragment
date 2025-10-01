#include <metal_stdlib>
using namespace metal;

struct ViewUniforms {
    float4x4 viewMatrix;
    float4x4 projectionMatrix;
    float4x4 viewProjectionMatrix;
    float3 cameraPosition;
    float2 screenSize;
};

struct GaussianSplatVertex {
    float3 position [[attribute(0)]];
    float3 scale [[attribute(1)]];
    float4 rotation [[attribute(2)]];
    float opacity [[attribute(3)]];
    float3 color [[attribute(4)]];
};

struct VertexOut {
    float4 position [[position]];
    float3 color;
    float opacity;
    float2 uv;
    float3 worldPosition;
    float3 scale;
    float4 rotation;
};

vertex VertexOut gaussianSplatVertex(
    GaussianSplatVertex in [[stage_in]],
    constant ViewUniforms &uniforms [[buffer(1)]],
    uint vid [[vertex_id]]
) {
    VertexOut out;
    
    float4 worldPos = float4(in.position, 1.0);
    float4 viewPos = uniforms.viewMatrix * worldPos;
    float4 clipPos = uniforms.projectionMatrix * viewPos;
    
    out.position = clipPos;
    out.color = in.color;
    out.opacity = in.opacity;
    out.worldPosition = in.position;
    out.scale = in.scale;
    out.rotation = in.rotation;
    out.uv = float2(0.0, 0.0);
    
    return out;
}

float3x3 quaternionToMatrix(float4 q) {
    float x = q.x, y = q.y, z = q.z, w = q.w;
    
    float3x3 m;
    m[0][0] = 1.0 - 2.0 * (y * y + z * z);
    m[0][1] = 2.0 * (x * y + w * z);
    m[0][2] = 2.0 * (x * z - w * y);
    
    m[1][0] = 2.0 * (x * y - w * z);
    m[1][1] = 1.0 - 2.0 * (x * x + z * z);
    m[1][2] = 2.0 * (y * z + w * x);
    
    m[2][0] = 2.0 * (x * z + w * y);
    m[2][1] = 2.0 * (y * z - w * x);
    m[2][2] = 1.0 - 2.0 * (x * x + y * y);
    
    return m;
}

fragment float4 gaussianSplatFragment(
    VertexOut in [[stage_in]],
    constant ViewUniforms &uniforms [[buffer(0)]]
) {
    float2 uv = in.uv;
    
    float distance_squared = dot(uv, uv);
    
    if (distance_squared > 1.0) {
        discard_fragment();
    }
    
    float alpha = exp(-0.5 * distance_squared) * in.opacity;
    
    float3 finalColor = in.color;
    
    return float4(finalColor, alpha);
}

kernel void computeGaussianSplats(
    constant GaussianSplatVertex *splats [[buffer(0)]],
    constant ViewUniforms &uniforms [[buffer(1)]],
    device float4 *positions [[buffer(2)]],
    device float4 *colors [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= 1000000) return; // Max splat limit
    
    GaussianSplatVertex splat = splats[id];
    
    float4 worldPos = float4(splat.position, 1.0);
    float4 viewPos = uniforms.viewMatrix * worldPos;
    float4 clipPos = uniforms.projectionMatrix * viewPos;
    
    float distanceToCamera = length(viewPos.xyz);
    float attenuationFactor = 1.0 / (1.0 + 0.01 * distanceToCamera);
    
    positions[id] = clipPos;
    colors[id] = float4(splat.color * attenuationFactor, splat.opacity);
}

vertex VertexOut instancedGaussianSplatVertex(
    constant float2 *quadVertices [[buffer(0)]],
    constant GaussianSplatVertex *splats [[buffer(1)]],
    constant ViewUniforms &uniforms [[buffer(2)]],
    uint vid [[vertex_id]],
    uint iid [[instance_id]]
) {
    VertexOut out;
    
    float2 quadPos = quadVertices[vid];
    GaussianSplatVertex splat = splats[iid];
    
    float4 worldPos = float4(splat.position, 1.0);
    float4 viewPos = uniforms.viewMatrix * worldPos;
    
    float3x3 rotationMatrix = quaternionToMatrix(splat.rotation);
    float3 scaledQuadPos = float3(quadPos * splat.scale.xy, 0.0);
    float3 rotatedQuadPos = rotationMatrix * scaledQuadPos;
    
    float4 finalViewPos = viewPos + float4(rotatedQuadPos, 0.0);
    float4 clipPos = uniforms.projectionMatrix * finalViewPos;
    
    out.position = clipPos;
    out.color = splat.color;
    out.opacity = splat.opacity;
    out.uv = quadPos;
    out.worldPosition = splat.position;
    out.scale = splat.scale;
    out.rotation = splat.rotation;
    
    return out;
}