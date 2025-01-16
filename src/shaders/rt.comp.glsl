#version 460 core

#extension GL_ARB_bindless_texture : require

layout(local_size_x = 8, local_size_y = 4, local_size_z = 1) in;
layout(rgba32f, binding = 0) uniform image2D render_texture;

layout (location = 0)
uniform vec3 pixel_delta_x;

layout (location = 1)
uniform vec3 pixel_delta_y;

layout (location = 2)
uniform vec3 pixel00_loc;

layout (location = 3)
uniform vec3 camera_pos;

layout (location = 4)
uniform int sample_index;

layout (location = 5)
uniform bool should_accumulate;

layout (location = 6)
uniform bool use_normal_maps;


layout (binding = 1)
uniform sampler2D temp_texture;

struct BLASNode {
    vec3 aabb_min;
    int left_or_first;
    vec3 aabb_max;
    int tris_count;
};

struct Intersection {
    float u;
    float v;
    float t;
    int index_buffer_idx;
    int prim_index;
};

struct Ray {
    vec3 origin;
    vec3 dir;
    vec3 inv_dir;
};

struct Vertex {
    vec4 tangent;
    vec3 position;
    float u;
    vec3 normal;
    float v;
};

struct BLAS {
    vec3 aabb_min;
    int root_idx;
    vec3 aabb_max;
    int index_offset;
};

struct TLASNode {
    vec3 aabb_min;
    uint left_right;
    vec3 aabb_max;
    int blas_idx;
};

struct Primitive {
    BLAS accel_structure;
    int base_index;
    int base_vertex;
    int mesh_idx;
    int material_idx;
};

struct Material {
    vec4 base_color_factor;
    uvec2 base_color_texture;
    uvec2 metallic_roughness_texture;
    uvec2 normal_map_texture;
    float metallic_factor;
    float roughness_factor;
    int flags;
};

layout (std430, binding = 0) restrict readonly buffer VertexBuffer {
    Vertex vertex_buffer[];
};

layout (std430, binding = 1) restrict readonly buffer IndexBuffer {
    uint index_buffer[];
};

layout (std430, binding = 2) restrict readonly buffer BLASNodeBuffer {
    BLASNode blas_nodes[];
};

layout (std430, binding = 3) restrict readonly buffer BLASIndexBuffer {
    int blas_index_buffer[];
};


layout (std430, binding = 4) restrict readonly buffer TLASNodeBuffer {
    TLASNode tlas_nodes[];
};

layout (std430, binding = 5) restrict readonly buffer MeshTransformBuffer {
    mat4 mesh_transforms[];
};

// NOTE: glTF "Primitive". Primitives consists of a single material and vertices.
layout (std430, binding = 6) restrict readonly buffer PrimitiveBuffer {
    Primitive primitives[];
};

layout (std430, binding = 7) restrict readonly buffer MaterialBuffer {
    Material materials[];
};

#define PI 3.14159265358979323846264338327950288
#define ONE_OVER_PI 0.3183098861837907


/* -----------------------------
    Random Functions
   ----------------------------- */

uint g_rand_seed;

// used for initilazing random seed for other random number generators.
uint wang_hash(uint seed) {
    seed = (seed ^ 61) ^ (seed >> 16);
    seed *= 9;
    seed = seed ^ (seed >> 4);
    seed *= 0x27d4eb2d;
    seed = seed ^ (seed >> 15);
    return seed;
}

uint pcg() {
    uint prev = g_rand_seed;
    g_rand_seed = prev * 747796405u + 2891336453u;
    uint word = ((prev >> ((prev >> 28u) + 4u)) ^ prev) * 277803737u;
    return (word >> 22u) ^ word;
}

float random_float() {
    return float(pcg()) * (1.0 / 4294967296.0);
}

vec2 random_in_square() {
    return vec2(random_float() - 0.5, random_float() - 0.5);
}

bool intersect_aabb(Ray r, vec3 box_min, vec3 box_max, out float t_min, out float t_max) {
    vec3 t0 = (box_min - r.origin) * r.inv_dir;
    vec3 t1 = (box_max - r.origin) * r.inv_dir;
    vec3 t_near = min(t0, t1);
    vec3 t_far = max(t0, t1);
    t_min = max(max(t_near.x, t_near.y), t_near.z);
    t_max = min(min(t_far.x, t_far.y), t_far.z);
    return t_min <= t_max && t_max >= 0.0;
}

bool intersect_triangle(vec3 orig, vec3 dir, vec3 v0, vec3 v1, vec3 v2, out float t, out vec2 barycentric) {
    vec3 edge1 = v1 - v0;
    vec3 edge2 = v2 - v0; 
    vec3 h = cross(dir, edge2);
    float det = dot(edge1, h); 
    if (abs(det) < 1e-8) return false; 
    float invDet = 1.0 / det; 
    vec3 s = orig - v0; 
    float u = dot(s, h) * invDet;
    if (u < 0.0 || u > 1.0) return false; 
    vec3 q = cross(s, edge1);
    float v = dot(dir, q) * invDet;
    if (v < 0.0 || u + v > 1.0) return false; 
    t = dot(edge2, q) * invDet;
    if (t < 0.0) return false; 
    barycentric = vec2(u, v); 
    return true;
}

Intersection hit_bvh(Ray r, int primitive_index) {
    Intersection result;
    result.t = 1e30;
    result.prim_index = primitive_index;

    Primitive prim = primitives[primitive_index];
    mat4 inv_transform = mesh_transforms[prim.mesh_idx];

    r.origin = (inv_transform * vec4(r.origin, 1.0)).xyz;
    r.dir = (inv_transform * vec4(r.dir, 0.0)).xyz;
    r.inv_dir = 1.0 / r.dir;

    int stack[32];
    int stack_ptr = 0;
    stack[stack_ptr++] = prim.accel_structure.root_idx;

    while (stack_ptr > 0) {
        BLASNode node = blas_nodes[stack[--stack_ptr]];

        if (node.tris_count > 0) {
            for (int i = 0; i < node.tris_count; ++i) {
                int prim_index = node.left_or_first + i;

                vec2 barycentric;
                float t;

                // This is stupid. 
                int index_buffer_index = prim.base_index + 3 * blas_index_buffer[prim.accel_structure.index_offset + prim_index];
                vec3 v0 = vertex_buffer[prim.base_vertex + index_buffer[index_buffer_index + 0]].position;
                vec3 v1 = vertex_buffer[prim.base_vertex + index_buffer[index_buffer_index + 1]].position;
                vec3 v2 = vertex_buffer[prim.base_vertex + index_buffer[index_buffer_index + 2]].position;
                

                if (intersect_triangle(
                    r.origin,
                    r.dir,
                    v0,
                    v1,
                    v2,
                    t,
                    barycentric
                ) && t < result.t) {
                    result.t = t;
                    result.index_buffer_idx = index_buffer_index;
                    result.u = barycentric.x;
                    result.v = barycentric.y;
                }
            }
        } else {
            int left = prim.accel_structure.root_idx + node.left_or_first + 0;
            int right = prim.accel_structure.root_idx + node.left_or_first + 1;
            BLASNode node_left = blas_nodes[left];
            BLASNode node_right = blas_nodes[right];

            float left_dist;
            float right_dist;
            float tmp;
            bool hit_left = intersect_aabb(r, node_left.aabb_min, node_left.aabb_max, left_dist, tmp);
            bool hit_right = intersect_aabb(r, node_right.aabb_min, node_right.aabb_max, right_dist, tmp);

            bool left_closer = left_dist <= right_dist;

            float dist_close = left_closer ? left_dist : right_dist;
            float dist_far = left_closer ? right_dist : left_dist;

            int index_close = left_closer ? left : right;
            int index_far = left_closer ? right : left;

            bool hit_close = left_closer ? hit_left : hit_right;
            bool hit_far = left_closer ? hit_right : hit_left;

            if (hit_far && dist_far < result.t) stack[stack_ptr++] = index_far;
            if (hit_close && dist_close < result.t) stack[stack_ptr++] = index_close;
        }
    }

    return result;
}

Intersection hit_tlas(Ray r) {
    Intersection result;
    result.t = 1e30;

    int stack[32];
    int stack_ptr = 0;
    stack[stack_ptr++] = 0;

    while (stack_ptr > 0) {
        TLASNode node = tlas_nodes[stack[--stack_ptr]];

        if (node.left_right == 0) {
            Intersection maybe_hit = hit_bvh(r, node.blas_idx);
            if (maybe_hit.t < result.t) {
                result = maybe_hit;
            }
        } else {
            int left = int(node.left_right & 0xffff);
            int right = int(node.left_right >> 16);
            TLASNode node_left = tlas_nodes[left];
            TLASNode node_right = tlas_nodes[right];

            float left_dist;
            float right_dist;
            float tmp;
            bool hit_left = intersect_aabb(r, node_left.aabb_min, node_left.aabb_max, left_dist, tmp);
            bool hit_right = intersect_aabb(r, node_right.aabb_min, node_right.aabb_max, right_dist, tmp);

            bool left_closer = left_dist <= right_dist;

            float dist_close = left_closer ? left_dist : right_dist;
            float dist_far = left_closer ? right_dist : left_dist;

            int index_close = left_closer ? left : right;
            int index_far = left_closer ? right : left;

            bool hit_close = left_closer ? hit_left : hit_right;
            bool hit_far = left_closer ? hit_right : hit_left;

            if (hit_far && dist_far < result.t) stack[stack_ptr++] = index_far;
            if (hit_close && dist_close < result.t) stack[stack_ptr++] = index_close;
        }
    }

    return result;
}

vec3 sample_cosine_hemisphere(vec2 xi, vec3 N) {
    float phi = 2.0 * PI * xi.x;
    float cos_theta = sqrt(xi.y);
    float sin_theta = sqrt(1.0 - cos_theta * cos_theta);
    
    vec3 H;
    H.x = sin_theta * cos(phi);
    H.y = sin_theta * sin(phi);
    H.z = cos_theta;
    
    vec3 up = abs(N.z) < 0.999 ? vec3(0,0,1) : vec3(1,0,0);
    vec3 tangent = normalize(cross(up, N));
    vec3 bitangent = cross(N, tangent);
    
    return tangent * H.x + bitangent * H.y + N * H.z;
}

vec3 sample_ggx(vec2 xi, float roughness, vec3 N) {
    float alpha = roughness * roughness;
    
    float phi = 2.0 * PI * xi.x;
    float cos_theta = sqrt((1.0 - xi.y) / (1.0 + (alpha*alpha - 1.0) * xi.y));
    float sin_theta = sqrt(1.0 - cos_theta * cos_theta);
    
    vec3 H;
    H.x = sin_theta * cos(phi);
    H.y = sin_theta * sin(phi);
    H.z = cos_theta;
    
    vec3 up = abs(N.z) < 0.999 ? vec3(0,0,1) : vec3(1,0,0);
    vec3 tangent = normalize(cross(up, N));
    vec3 bitangent = cross(N, tangent);
    
    return tangent * H.x + bitangent * H.y + N * H.z;
}

float D_GGX(float NdotH, float roughness) {
    float alpha = max(0.001, roughness * roughness);
    float alpha2 = alpha * alpha;
    float denom = NdotH * NdotH * (alpha2 - 1.0) + 1.0;
    return alpha2 / (PI * denom * denom);
}

vec3 F_Schlick(float VdotH, vec3 F0) {
    return F0 + (vec3(1.0) - F0) * pow(1.0 - VdotH, 5.0);
}

float G_Smith(float NdotV, float NdotL, float roughness) {
    float alpha = roughness * roughness;
    float k = alpha * 0.5;
    float G1V = NdotV / (NdotV * (1.0 - k) + k);
    float G1L = NdotL / (NdotL * (1.0 - k) + k);
    return G1L * G1V;
}

vec3 sample_enviroment(Ray r) {
    vec3 unit_direction = normalize(r.dir);
    float a = 0.5*(unit_direction.y + 1.0);
    return (1.0-a) * vec3(1.0, 1.0, 1.0) + a * vec3(0.5, 0.7, 1.0);
}


vec3 eval(vec3 N, vec3 V, vec3 L, vec3 base_color, float metallic, float roughness, out float pdf) {
    if (dot(N, L) <= 0.001 || dot(N, V) <= 0.001) {
        pdf = 0.0;
        return vec3(0.0);
    }

    vec3 H = normalize(L + V);
    float NdotL = max(dot(N, L), 0.001);
    float NdotV = max(dot(N, V), 0.001);
    float NdotH = max(dot(N, H), 0.001);
    float VdotH = max(dot(V, H), 0.001);
    
    float diffuse_ratio = 0.5 * (1.0 - metallic);
    float specular_ratio = 1.0 - diffuse_ratio;
    
    vec3 F0 = mix(vec3(0.04), base_color, metallic);
    vec3 F = F_Schlick(VdotH, F0);
    float D = D_GGX(NdotH, roughness);
    float G = G_Smith(NdotV, NdotL, roughness);
    
    vec3 diffuse = (1.0 - F) * (1.0 - metallic) * base_color * ONE_OVER_PI;
    vec3 specular = (F * D * G) / max((4.0 * NdotV * NdotL), 1e-4);
    
    float pdf_diffuse = NdotL * ONE_OVER_PI;
    float pdf_specular = D * NdotH / (4.0 * VdotH);
    
    pdf = diffuse_ratio * pdf_diffuse + specular_ratio * pdf_specular;
    return diffuse + specular;
}

vec3 bsdf(vec3 N, vec3 V, out vec3 L, vec3 base_color, float metallic, float roughness, out float pdf) {
    float diffuse_ratio = 0.5 * (1.0 - metallic);

    if (random_float() < diffuse_ratio) {
        L = sample_cosine_hemisphere(vec2(random_float(), random_float()), N);
        return eval(N, V, L, base_color, metallic, roughness, pdf);
    } else {
        vec3 H = sample_ggx(vec2(random_float(), random_float()), roughness, N);
        L = reflect(-V, H);
        if (dot(N, L) <= 0.0) {
            pdf = 0.0;
            return vec3(0);
        }
        return eval(N, V, L, base_color, metallic, roughness, pdf);
    }
}

vec3 get_base_color(uvec2 handle, vec2 uv, vec3 base_color_factor) {
    if (handle.x == 0xffff && handle.y == 0xffff) {
        return base_color_factor;
    }

    return texture(sampler2D(handle), uv).xyz * base_color_factor;
}

vec3 get_metallic_roughness(uvec2 handle, vec2 uv) {
    if (handle.x == 0xffff && handle.y == 0xffff) {
        return vec3(1);
    }

    return texture(sampler2D(handle), uv).xyz;
}

// I officaly hate normals.
vec3 get_normal(mat3 normal_matrix, uvec2 handle, vec2 uv, float u, float v, float w, Vertex v0, Vertex v1, Vertex v2, int flags) {
    vec3 normal = normalize(u * (normal_matrix * v1.normal) + v * (normal_matrix * v2.normal) + w * normal_matrix * v0.normal);  
    vec3 tangent = normalize(u * (normal_matrix * normalize(v1.tangent.xyz)) + v * (normal_matrix * normalize(v2.tangent.xyz)) + w * (normal_matrix * normalize(v0.tangent.xyz)));  
    vec3 bitangent = cross(normal, tangent) * v0.tangent.w; // Vertices of the same triangle SHOULD have the same tangent.w value. 
    mat3 TBN = mat3(tangent, bitangent, normal);

    vec3 mapped_normal = texture(sampler2D(handle), uv).rgb * 2.0 - 1.0;
    mapped_normal.g = -mapped_normal.g;
    mapped_normal = normalize(TBN * mapped_normal);

    return (flags != 0 && use_normal_maps) ? mapped_normal : normal;
}

const int MAX_BOUNCES = 10;

vec3 trace(Ray r) {
    vec3 attenuation = vec3(1);
    vec3 result = vec3(0);
    Intersection hit;

    for (int i = 0; i < MAX_BOUNCES; i++) {
        if (i > 2) {
            float p = max(max(attenuation.x, attenuation.y), attenuation.z);
            p = max(0.1, min(0.9, p));
            if (random_float() > p) {
                break;
            }
            attenuation /= p;
        }

        hit = hit_tlas(r);
        if (hit.t == 1e30) {
            result += attenuation * sample_enviroment(r);
            break;
        }

        Primitive prim = primitives[hit.prim_index];
        Material mat = materials[prim.material_idx];

        Vertex v0 = vertex_buffer[prim.base_vertex + index_buffer[hit.index_buffer_idx + 0]];
        Vertex v1 = vertex_buffer[prim.base_vertex + index_buffer[hit.index_buffer_idx + 1]];
        Vertex v2 = vertex_buffer[prim.base_vertex + index_buffer[hit.index_buffer_idx + 2]];
        vec2 uv0 = vec2(v0.u, v0.v);
        vec2 uv1 = vec2(v1.u, v1.v);
        vec2 uv2 = vec2(v2.u, v2.v);

        float w = (1 - (hit.u + hit.v));
        vec2 uv = hit.u * uv1 + hit.v * uv2 + w * uv0;
        vec3 base_color = get_base_color(mat.base_color_texture, uv, mat.base_color_factor.rgb);
        vec3 metallic_roughness = get_metallic_roughness(mat.metallic_roughness_texture, uv);

        float metallic = metallic_roughness.b * mat.metallic_factor;
        float roughness = metallic_roughness.g * mat.roughness_factor;
        roughness = max(roughness, 0.01);

        mat3 normal_matrix = mat3(inverse(mesh_transforms[prim.mesh_idx]));
        vec3 normal = get_normal(normal_matrix, mat.normal_map_texture, uv, hit.u, hit.v, w, v0, v1, v2, mat.flags);
        //return (normal + 1) / 2;

        vec3 outgoing_dir;
        float pdf;
        vec3 brdf = bsdf(normal, -r.dir, outgoing_dir, base_color, metallic, roughness, pdf);
        if (pdf <= 0.0001) break;

        attenuation *= brdf * max(dot(normal, outgoing_dir), 0.001) / pdf;

        r.dir = outgoing_dir;
        r.origin = r.origin + r.dir * hit.t + normal * 0.001;
    }

    return result;
}


void main() {
    ivec2 pixel_coords = ivec2(gl_GlobalInvocationID.xy);
    ivec2 dims = imageSize(render_texture);

    g_rand_seed = wang_hash(uint(pixel_coords.x) * 1973u + uint(pixel_coords.y) * 9277u + sample_index * 26699u);

    vec3 offset = vec3(random_in_square(), 0.0);
    offset = vec3(0);

    vec3 pixel_center = pixel00_loc 
        + float(pixel_coords.x + offset.x) * pixel_delta_x
        + float(pixel_coords.y + offset.y) * pixel_delta_y;
    vec3 ray_dir = normalize(pixel_center - camera_pos);

    Ray r = Ray(camera_pos, ray_dir, 1.0 / ray_dir);

    vec3 color = trace(r);
    if (should_accumulate) {
        color += imageLoad(render_texture, pixel_coords).rgb;
    }
    imageStore(render_texture, pixel_coords, vec4(color, 1.0));
}