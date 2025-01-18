#version 450 core

layout(binding = 0)
uniform sampler2D render_texture;

layout(location = 0) uniform int sample_index;

layout(location = 0)
out vec4 o_color;

in vec2 uv;


void main() {
    o_color = texture(render_texture, uv) / sample_index;

    // Simple reinhard tonemapping.
    o_color = o_color / (1.0 + o_color);
    o_color = vec4(pow(o_color.rgb, vec3(1.0 / 2.2)), 1.0);
}