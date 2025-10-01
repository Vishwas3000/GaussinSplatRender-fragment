#include <metal_stdlib>
using namespace metal;

struct Vertex {
    float2 position [[attribute(0)]];
    float3 color [[attribute(1)]];
};

struct VertexOut {
    float4 position [[position]];
    float3 color;
    float2 uv;
};

vertex VertexOut vertex_main(Vertex in [[stage_in]]) {
    VertexOut out;
    out.position = float4(in.position, 0.0, 1.0);
    out.color = in.color;
    out.uv = (in.position + 1.0) * 0.5;
    return out;
}

fragment float4 fragment_main(VertexOut in [[stage_in]]) {
    float2 uv = in.uv;
    
    float3 color1 = float3(0.2, 0.4, 0.8);
    float3 color2 = float3(0.8, 0.4, 0.2);
    float3 color3 = float3(0.4, 0.8, 0.2);
    float3 color4 = float3(0.8, 0.2, 0.8);
    
    float3 topBlend = mix(color1, color2, uv.x);
    float3 bottomBlend = mix(color3, color4, uv.x);
    float3 finalColor = mix(bottomBlend, topBlend, uv.y);
    
    float t = sin(uv.x * 10.0) * sin(uv.y * 10.0) * 0.1 + 1.0;
    finalColor *= t;
    
    return float4(finalColor, 1.0);
}