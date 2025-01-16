#version 460 core

layout (location = 0)
in vec4 a_tangent;

layout (location = 1)
in vec3 a_pos;

layout (location = 2)
in float a_u;

layout (location = 3)
in vec3 a_normal;

layout (location = 4)
in float a_v;

out gl_PerVertex { vec4 gl_Position; };

out vec2 v_uv;
out vec3 v_frag_pos;
out vec3 v_normal;
out mat3 v_TBN;

layout (location = 0)
uniform mat4 model;

layout (location = 1)
uniform mat4 view;

layout (location = 2)
uniform mat4 projection;

layout (location = 3)
uniform mat3 normal_matrix;

void main() {
    gl_Position = projection * view * model * vec4(a_pos, 1);
    v_uv = vec2(a_u, a_v);
    v_frag_pos = vec3(model * vec4(a_pos, 1));

    vec3 T = normalize(normal_matrix * a_tangent.xyz);
    vec3 N = normalize(normal_matrix * a_normal);
    vec3 B = cross(N, T) * a_tangent.w;

    v_TBN = mat3(T, B, N);
    v_normal = normal_matrix * a_normal;
}