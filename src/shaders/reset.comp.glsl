#version 460 core

layout(local_size_x = 8, local_size_y = 4, local_size_z = 1) in;
layout(rgba32f, binding = 0) uniform image2D render_texture;

void main() {
    ivec2 pixel_coords = ivec2(gl_GlobalInvocationID.xy);
    imageStore(render_texture, pixel_coords, vec4(0));
}