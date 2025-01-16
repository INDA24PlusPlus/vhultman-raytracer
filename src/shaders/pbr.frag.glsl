#version 460 core

layout (location = 0)
out vec4 frag_color;

in vec2 v_uv;
in vec3 v_frag_pos;
in vec3 v_normal;
in mat3 v_TBN;

layout (location = 0)
uniform vec3 camera_pos;

layout (location = 1)
uniform vec4 base_color_factor;

layout (location = 2)
uniform float metallic_factor;

layout (location = 3)
uniform float roughness_factor;

layout (location = 4)
uniform uint flags;

layout (binding = 0)
uniform sampler2D s_base_color_texture;

layout (binding = 1)
uniform sampler2D s_metallic_roughness_texture;

layout (binding = 2)
uniform sampler2D s_occlusion_texture;

layout (binding = 3)
uniform sampler2D s_normal_map;

const float PI = 3.14159265359;

vec3 light_positions[] = vec3[] (
    vec3(-10.0,  10.0, 10.0),
    vec3( 10.0,  10.0, 10.0),
    vec3(3.428, 2.688, -0.356),
    vec3( 10.0, -10.0, 10.0)
);

vec3 light_colors[] = vec3[] (
    vec3(300),
    vec3(300),
    vec3(50),
    vec3(300)
);

vec3 fresnel_schlick(float cos_theta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

float distribution_ggx(vec3 N, vec3 H, float roughness) {
    float a = roughness*roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;
	
    float num = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;
	
    return num / denom;
}

float geometry_schlick_ggx(float NdotV, float roughness) {
    float r = (roughness + 1.0);
    float k = (r*r) / 8.0;

    float num   = NdotV;
    float denom = NdotV * (1.0 - k) + k;
	
    return num / denom;
}

float geometry_smith(vec3 N, vec3 V, vec3 L, float roughness) {
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2 = geometry_schlick_ggx(NdotV, roughness);
    float ggx1 = geometry_schlick_ggx(NdotL, roughness);
	
    return ggx1 * ggx2;
}

vec4 get_base_color() {
    bool has_base_color_texture = (flags & 1) != 0;
    if (has_base_color_texture) {
        return texture(s_base_color_texture, v_uv) * base_color_factor;
    }

    return base_color_factor;
}

vec3 get_metallic_roughness() {
    bool has_metallic_roughness_texture = (flags & 2) != 0;
    if (has_metallic_roughness_texture) {
        return texture(s_metallic_roughness_texture, v_uv).rgb;
    }

    return vec3(1);
}

float get_occlusion() {
    bool has_occlusion_texture = (flags & 4) != 0;
    if (has_occlusion_texture) {
        return texture(s_occlusion_texture, v_uv).r;
    }

    return 1;
}

vec3 get_normal() {
    bool has_normal_map = (flags & 8) != 0;
    if (has_normal_map) {
        vec3 normal_map = texture(s_normal_map, v_uv).rgb * 2 - 1.0;
        return normalize(v_TBN * normal_map);
    } 
    return  normalize(v_normal);
}

void main() {

    vec4 base_color = get_base_color();
    vec3 metallic_roughness = get_metallic_roughness();

    float metallic = metallic_factor * metallic_roughness.b;
    float roughness = roughness_factor * metallic_roughness.g;
    float ao = get_occlusion();

    vec3 N = get_normal();
    vec3 V = normalize(camera_pos - v_frag_pos);

    vec3 Lo = vec3(0.0);
    for (int i = 0; i < 4; ++i) {
        vec3 L = normalize(light_positions[i] - v_frag_pos);
        vec3 H = normalize(V + L);

        float distance = length(light_positions[i] - v_frag_pos);
        float attenuation = 1.0 / (distance * distance);
        vec3 radiance = light_colors[i] * attenuation;

        vec3 F0 = mix(vec3(0.04), base_color.rgb, metallic);

        vec3 F = fresnel_schlick(max(dot(H, V), 0.0), F0);
        float NDF = distribution_ggx(N, H, roughness);       
        float G = geometry_smith(N, V, L, roughness);       

        vec3 numerator = NDF * G * F;
        float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0)  + 0.0001;
        vec3 specular = numerator / denominator;  

        vec3 kS = F;
        vec3 kD = vec3(1.0) - kS;
        
        kD *= 1.0 - metallic;	
        float NdotL = max(dot(N, L), 0.0);        
        Lo += (kD * base_color.rgb / PI + specular) * radiance * NdotL;
    }

    vec3 ambient = vec3(0.03) * base_color.rgb * ao;
    vec3 color = ambient + Lo;  
    color = color / (color + vec3(1.0));

    frag_color = vec4(color, 1.0);
}