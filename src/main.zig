const std = @import("std");
const glfw = @import("glfw");
const gl = @import("gl.zig");
const zm = @import("zmath.zig");
const bvh = @import("bvh.zig");
const zgui = @import("zgui");
const zstbi = @import("zstbi");
const zgltf = @import("zgltf");
const Allocator = std.mem.Allocator;

const Config = struct {
    const debug_context = true;
};

const AppState = struct {
    const start_width = 1920;
    const start_height = 1080;
    window_width: u32 = start_width,
    window_height: u32 = start_height,
    framebuffer_width: u32 = start_width,
    framebuffer_height: u32 = start_height,

    camera: Camera = .{},
    mouse_pos: [2]f32 = .{ 0, 0 },
    mouse_delta: [2]f32 = .{ 0, 0 },

    lock_cursor: bool = true,
    first_mouse: bool = true,

    current_time: f32 = 0,
    delta_time: f32 = 0,
    delta_times_idx: usize = 0,
    delta_times: [100]f32 = undefined,

    current_render_method: enum { Rasterization, Raytracing } = .Rasterization,
    rt_state: RTState = undefined,
};

extern fn glfwGetMonitorContentScale(window: *glfw.Monitor, x_scale: *f32, y_scale: *f32) void;

pub fn main() !void {
    var state: AppState = .{};
    state.camera.yaw = std.math.degreesToRadians(180);
    state.camera.pitch = 0;
    state.camera.pos = .{ 3, 1.5, 0, 0 };

    var gpa_state = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa_state.deinit();
    const gpa = gpa_state.allocator();

    try glfw.init();
    defer glfw.terminate();

    glfw.windowHintTyped(.opengl_profile, .opengl_core_profile);
    glfw.windowHintTyped(.context_version_major, 4);
    glfw.windowHintTyped(.context_version_minor, 6);
    glfw.windowHintTyped(.opengl_debug_context, Config.debug_context);
    glfw.windowHintTyped(.samples, 4);
    glfw.windowHintTyped(.srgb_capable, true);
    var x_scale: f32 = 0;
    var y_scale: f32 = 0;
    const prim = glfw.Monitor.getPrimary().?;
    glfwGetMonitorContentScale(prim, &x_scale, &y_scale);

    state.window_width = @intFromFloat(@as(f32, @floatFromInt(state.window_width)) / x_scale);
    state.window_height = @intFromFloat(@as(f32, @floatFromInt(state.window_height)) / y_scale);

    const window = try glfw.Window.create(
        @intCast(state.window_width),
        @intCast(state.window_height),
        "Ray tracer",
        null,
    );
    defer window.destroy();

    glfw.makeContextCurrent(window);
    try gl.load({}, glLoadProc);
    try gl.GL_ARB_bindless_texture.load({}, glLoadProc);
    gl.debugMessageCallback(glDebugCallback, null);

    zgui.init(std.heap.c_allocator);
    defer zgui.deinit();

    const scale_factor = scale_factor: {
        const scale = window.getContentScale();
        break :scale_factor @max(scale[0], scale[1]);
    };
    _ = zgui.io.addFontFromFile(
        "assets/Roboto-Medium.ttf",
        std.math.floor(16.0 * scale_factor),
    );

    zgui.getStyle().scaleAllSizes(scale_factor);
    zgui.io.setConfigFlags(.{
        .dock_enable = true,
    });

    var timer = try std.time.Timer.start();
    var scene_data = try SceneData.init(gpa, &.{ "SciFiHelmet", "Sponza" });
    defer scene_data.deinit(gpa);
    const end = timer.read();

    std.debug.print("Loading models took {d}ms\n", .{@as(f64, @floatFromInt(end)) / std.time.ns_per_ms});

    const handles = scene_data.gpu_data.upload();
    const primitive_buffer_handle, const mesh_transform_buffer_handle, const material_buffer_handle, const normal_matricies_handle = try scene_data.uploadDataToGPU(gpa);
    scene_data.geom_arena.deinit();

    const raster_state = try RasterState.init(handles.vbo, handles.ibo);
    {
        const fb_width, const fb_height = window.getFramebufferSize();
        gl.viewport(0, 0, fb_width, fb_height);
        state.framebuffer_width = @intCast(fb_width);
        state.framebuffer_height = @intCast(fb_height);
    }

    state.rt_state = try RTState.init(gpa, state.framebuffer_width, state.framebuffer_height, .{
        .vertex_buffer = handles.vbo,
        .index_buffer = handles.ibo,
        .blas_nodes_buffer = handles.blas_nodes,
        .blas_index_buffer = handles.blas_index_buffer,
        .tlas_node_buffer = handles.tlas_nodes,
        .mesh_transform_buffer = mesh_transform_buffer_handle,
        .normal_matricies = normal_matricies_handle,
        .primitive_buffer = primitive_buffer_handle,
        .material_buffer = material_buffer_handle,
    });

    gl.enable(gl.DEPTH_TEST);

    window.setInputMode(.cursor, .disabled);
    window.setUserPointer(&state);
    _ = window.setFramebufferSizeCallback(frameBufferSizeCallback);
    _ = window.setCursorPosCallback(cursorPosCallback);

    zgui.backend.init(window);
    defer zgui.backend.deinit();

    var esc_prev_pressed = false;
    glfw.swapInterval(1);

    state.current_time = @floatCast(glfw.getTime());
    while (!window.shouldClose()) {
        const prev_time = state.current_time;
        state.current_time = @floatCast(glfw.getTime());
        state.delta_time = state.current_time - prev_time;
        state.delta_times[state.delta_times_idx] = state.delta_time;
        state.delta_times_idx = (state.delta_times_idx + 1) % state.delta_times.len;
        glfw.pollEvents();

        if (window.getKey(.escape) == .press) {
            if (!esc_prev_pressed) {
                state.lock_cursor = !state.lock_cursor;
                if (state.lock_cursor) {
                    window.setInputMode(.cursor, .disabled);
                } else {
                    window.setInputMode(.cursor, .normal);
                }
            }
            esc_prev_pressed = true;
        } else {
            esc_prev_pressed = false;
        }

        var speed = zm.f32x4s(10 * state.delta_time);
        if (window.getKey(.left_shift) == .press) speed = zm.f32x4s(state.delta_time);
        if (window.getKey(.left_control) == .press) speed = zm.f32x4s(100 * state.delta_time);

        if (window.getKey(.w) == .press) {
            state.camera.pos += speed * state.camera.forward;
        }
        if (window.getKey(.s) == .press) {
            state.camera.pos -= speed * state.camera.forward;
        }
        if (window.getKey(.d) == .press) {
            state.camera.pos += speed * state.camera.right;
        }
        if (window.getKey(.a) == .press) {
            state.camera.pos -= speed * state.camera.right;
        }
        if (window.getKey(.q) == .press) {
            state.camera.pos[1] += speed[0];
        }
        if (window.getKey(.e) == .press) {
            state.camera.pos[1] -= speed[0];
        }
        state.camera.updateVectors();

        buildGUI(&state, window);

        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
        if (state.current_render_method == .Rasterization) {
            raster_state.render(&scene_data, &state);
        } else {
            state.rt_state.render(&state, &scene_data);
        }

        // Disable for GUI.
        gl.disable(gl.FRAMEBUFFER_SRGB);
        zgui.backend.draw();

        window.swapBuffers();
    }
}

fn buildGUI(state: *AppState, window: *glfw.Window) void {
    const fb_size = window.getFramebufferSize();
    zgui.backend.newFrame(@intCast(fb_size[0]), @intCast(fb_size[1]));

    if (zgui.begin("Properties", .{})) {
        var cam_pos: [3]f32 = .{ state.camera.pos[0], state.camera.pos[1], state.camera.pos[2] };
        var pitch: f32 = std.math.radiansToDegrees(state.camera.pitch);
        var yaw: f32 = std.math.radiansToDegrees(state.camera.yaw);

        if (zgui.inputFloat3("Camera position", .{ .v = &cam_pos })) {}
        _ = zgui.inputFloat("Pitch", .{ .v = &pitch });
        _ = zgui.inputFloat("Yaw", .{ .v = &yaw });
        if (zgui.checkbox("Use normal mapping", .{ .v = &use_normal_map })) {}

        if (state.current_render_method == .Raytracing) {
            _ = zgui.checkbox("Accumulate", .{ .v = &state.rt_state.should_accumulate });
        } else {
            if (zgui.checkbox("Use occlusion map", .{ .v = &use_occlusion_map })) {}
        }

        if (zgui.beginCombo("Render Method", .{ .preview_value = @tagName(state.current_render_method) })) {
            {
                const selected = state.current_render_method == .Rasterization;
                if (zgui.selectable("Rasterization", .{ .selected = selected })) {
                    state.current_render_method = .Rasterization;
                }

                if (selected) {
                    zgui.setItemDefaultFocus();
                }
            }
            {
                const selected = state.current_render_method == .Raytracing;
                if (zgui.selectable("Raytracing", .{ .selected = selected })) {
                    state.current_render_method = .Raytracing;
                }

                if (selected) {
                    zgui.setItemDefaultFocus();
                }
            }
            zgui.endCombo();
        }
    }
    zgui.end();

    if (zgui.begin("Timings", .{})) {
        var avg_delta_time: f32 = 0.0;
        for (state.delta_times) |t| avg_delta_time += t;
        avg_delta_time /= @floatFromInt(state.delta_times.len);
        zgui.text("{d:.1}FPS ({d:.3}ms)", .{ 1.0 / avg_delta_time, 1000 * avg_delta_time });
    }
    zgui.end();
}

const RTState = struct {
    vao: u32,
    fshader: u32,
    vshader: u32,
    cshader: u32,
    reset_shader: u32,
    render_texture: u32,
    pipeline: u32,

    buffer_handles: BufferHandles,
    sample_index: u32,
    should_accumulate: bool,
    should_reset_fb: bool,

    render_texture_width: u32,
    render_texture_height: u32,

    viewport_height: f32 = 2.0,
    viewport_width: f32 = @as(comptime_float, AppState.start_width) / @as(comptime_float, AppState.start_height),
    pixel_delta_x: zm.Vec = undefined,
    pixel_delta_y: zm.Vec = undefined,
    pixel00_loc: zm.Vec = undefined,
    focal_length: f32 = 1.0,

    const BufferHandles = struct {
        vertex_buffer: u32,
        index_buffer: u32,
        blas_nodes_buffer: u32,
        blas_index_buffer: u32,
        tlas_node_buffer: u32,
        mesh_transform_buffer: u32,
        normal_matricies: u32,
        primitive_buffer: u32,
        material_buffer: u32,
    };

    pub fn init(tmp: Allocator, framebuffer_width: u32, framebuffer_height: u32, buffer_handles: BufferHandles) !RTState {
        const vbo = vbo: {
            var vbo: u32 = undefined;
            // Fullscreen quad.
            // zig fmt: off
            const data = [_]f32{
                -1, -1,  0, 0,
                -1,  1,  0, 1,
                1,  1,  1, 1,

                -1, -1,  0, 0,
                1,  1,  1, 1,
                1, -1,  1, 0,
            };
            // zig fmt: on

            gl.createBuffers(1, &vbo);
            gl.namedBufferStorage(vbo, @sizeOf(@TypeOf(data)), &data, 0);

            break :vbo vbo;
        };

        const vao = vao: {
            var vao: u32 = undefined;
            gl.createVertexArrays(1, &vao);

            const vertex_stride = @sizeOf(f32) * 4;
            const vertex_buffer_index = 0;
            gl.vertexArrayVertexBuffer(vao, vertex_buffer_index, vbo, 0, vertex_stride);

            const a_pos = 0;
            gl.vertexArrayAttribFormat(vao, a_pos, 2, gl.FLOAT, gl.FALSE, 0);
            gl.vertexArrayAttribBinding(vao, a_pos, vertex_buffer_index);
            gl.enableVertexArrayAttrib(vao, a_pos);

            const a_uv = 1;
            gl.vertexArrayAttribFormat(vao, a_uv, 2, gl.FLOAT, gl.FALSE, 2 * @sizeOf(f32));
            gl.vertexArrayAttribBinding(vao, a_uv, vertex_buffer_index);
            gl.enableVertexArrayAttrib(vao, a_uv);

            break :vao vao;
        };

        const texture = createRenderTexture(framebuffer_width, framebuffer_height);
        const vshader = try compileStaticShader("shaders/rt.vert.glsl", gl.VERTEX_SHADER);
        const fshader = try compileStaticShader("shaders/rt.frag.glsl", gl.FRAGMENT_SHADER);
        const reset_shader = try compileStaticShader("shaders/reset.comp.glsl", gl.COMPUTE_SHADER);

        const ray_tracing_shader = cshader: {
            const source = try std.fs.cwd().readFileAllocOptions(tmp, "src/shaders/rt.comp.glsl", 1 << 20, null, @alignOf(u8), 0);
            defer tmp.free(source);

            const cshader = gl.createShaderProgramv(gl.COMPUTE_SHADER, 1, &source.ptr);
            var msg: [1024]u8 = undefined;
            if (try checkForCompileErrors(&msg, cshader)) |error_log| {
                std.log.err("Failed to link raytracing shader: {s}", .{error_log});
                return error.ShaderCompilationFailed;
            }

            break :cshader cshader;
        };

        const shader_pipeline = pipeline: {
            var pipeline: u32 = undefined;
            gl.genProgramPipelines(1, &pipeline);
            gl.useProgramStages(pipeline, gl.VERTEX_SHADER_BIT, vshader);
            gl.useProgramStages(pipeline, gl.FRAGMENT_SHADER_BIT, fshader);
            gl.useProgramStages(pipeline, gl.COMPUTE_SHADER_BIT, ray_tracing_shader);

            break :pipeline pipeline;
        };

        return .{
            .vao = vao,
            .vshader = vshader,
            .fshader = fshader,
            .pipeline = shader_pipeline,
            .render_texture = texture,
            .cshader = ray_tracing_shader,
            .render_texture_width = framebuffer_width,
            .render_texture_height = framebuffer_height,
            .buffer_handles = buffer_handles,
            .should_accumulate = false,
            .sample_index = 1,
            .should_reset_fb = false,
            .reset_shader = reset_shader,
        };
    }

    var tmp_bindless_handle: ?u64 = null;

    pub fn render(rt_state: *RTState, app_state: *const AppState, tmp_scene_data: *const SceneData) void {
        gl.bindVertexArray(rt_state.vao);
        gl.bindProgramPipeline(rt_state.pipeline);
        gl.bindTextureUnit(0, rt_state.render_texture);
        gl.bindTextureUnit(1, tmp_scene_data.texure_handles.items[0]);

        const buffer_bindings = struct {
            const vertex_buffer = 0;
            const index_buffer = 1;
            const blas_nodes_buffer = 2;
            const blas_index_buffer = 3;
            const tlas_node_buffer = 4;
            const mesh_transform_buffer = 5;
            const normal_matricies = 6;
            const primitive_buffer = 7;
            const material_buffer = 8;
        };

        gl.bindBufferBase(gl.SHADER_STORAGE_BUFFER, buffer_bindings.vertex_buffer, rt_state.buffer_handles.vertex_buffer);
        gl.bindBufferBase(gl.SHADER_STORAGE_BUFFER, buffer_bindings.index_buffer, rt_state.buffer_handles.index_buffer);
        gl.bindBufferBase(gl.SHADER_STORAGE_BUFFER, buffer_bindings.blas_nodes_buffer, rt_state.buffer_handles.blas_nodes_buffer);
        gl.bindBufferBase(gl.SHADER_STORAGE_BUFFER, buffer_bindings.blas_index_buffer, rt_state.buffer_handles.blas_index_buffer);
        gl.bindBufferBase(gl.SHADER_STORAGE_BUFFER, buffer_bindings.tlas_node_buffer, rt_state.buffer_handles.tlas_node_buffer);
        gl.bindBufferBase(gl.SHADER_STORAGE_BUFFER, buffer_bindings.mesh_transform_buffer, rt_state.buffer_handles.mesh_transform_buffer);
        gl.bindBufferBase(gl.SHADER_STORAGE_BUFFER, buffer_bindings.normal_matricies, rt_state.buffer_handles.normal_matricies);
        gl.bindBufferBase(gl.SHADER_STORAGE_BUFFER, buffer_bindings.primitive_buffer, rt_state.buffer_handles.primitive_buffer);
        gl.bindBufferBase(gl.SHADER_STORAGE_BUFFER, buffer_bindings.material_buffer, rt_state.buffer_handles.material_buffer);

        if (rt_state.should_reset_fb) {
            gl.useProgramStages(rt_state.pipeline, gl.COMPUTE_SHADER_BIT, rt_state.reset_shader);
            gl.dispatchCompute(rt_state.render_texture_width / 8, rt_state.render_texture_height / 4, 1);
            gl.memoryBarrier(gl.ALL_BARRIER_BITS);
            gl.useProgramStages(rt_state.pipeline, gl.COMPUTE_SHADER_BIT, rt_state.cshader);
            rt_state.should_reset_fb = false;
            rt_state.sample_index = 1;
        }

        rt_state.computeViewportVectors(app_state);
        gl.programUniform3fv(rt_state.cshader, 0, 1, @ptrCast(&rt_state.pixel_delta_x));
        gl.programUniform3fv(rt_state.cshader, 1, 1, @ptrCast(&rt_state.pixel_delta_y));
        gl.programUniform3fv(rt_state.cshader, 2, 1, @ptrCast(&rt_state.pixel00_loc));
        gl.programUniform3fv(rt_state.cshader, 3, 1, @ptrCast(&app_state.camera.pos));

        gl.programUniform1i(rt_state.cshader, 4, @intCast(rt_state.sample_index));
        gl.programUniform1i(rt_state.cshader, 5, @intFromBool(rt_state.should_accumulate));
        gl.programUniform1i(rt_state.cshader, 6, @intFromBool(use_normal_map));

        // Fragment shader
        gl.programUniform1i(rt_state.fshader, 0, @intCast(rt_state.sample_index));

        gl.disable(gl.FRAMEBUFFER_SRGB);
        gl.disable(gl.CULL_FACE);

        // TODO: Make this resiliant to resolution changes.
        gl.dispatchCompute(rt_state.render_texture_width / 8, rt_state.render_texture_height / 4, 1);

        if (rt_state.should_accumulate) {
            rt_state.sample_index += 1;
        } else {
            rt_state.sample_index = 1;
        }

        gl.memoryBarrier(gl.ALL_BARRIER_BITS);

        gl.drawArrays(gl.TRIANGLES, 0, 6);
    }

    pub fn resize(rt_state: *RTState, new_width: u32, new_height: u32) void {
        gl.deleteTextures(1, &rt_state.render_texture);
        rt_state.render_texture = createRenderTexture(new_width, new_height);
        rt_state.render_texture_width = new_width;
        rt_state.render_texture_height = new_height;
    }

    fn createRenderTexture(width: u32, height: u32) u32 {
        var texture: u32 = undefined;
        gl.createTextures(gl.TEXTURE_2D, 1, &texture);
        gl.textureParameteri(texture, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.textureParameteri(texture, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        gl.textureParameteri(texture, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.textureParameteri(texture, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

        gl.textureStorage2D(texture, 1, gl.RGBA32F, @intCast(width), @intCast(height));
        gl.bindImageTexture(0, texture, 0, gl.FALSE, 0, gl.READ_WRITE, gl.RGBA32F);
        return texture;
    }

    fn computeViewportVectors(state: *RTState, app_state: *const AppState) void {
        const fwidth: f32 = @floatFromInt(state.render_texture_width);
        const fheight: f32 = @floatFromInt(state.render_texture_height);
        const aspect_ratio = fwidth / fheight;

        state.viewport_width = aspect_ratio * state.viewport_height;
        const viewport_x: zm.Vec = zm.f32x4s(state.viewport_width) * app_state.camera.right;
        const viewport_y: zm.Vec = zm.f32x4s(state.viewport_height) * app_state.camera.up;
        state.pixel_delta_x = viewport_x / zm.f32x4s(fwidth);
        state.pixel_delta_y = viewport_y / zm.f32x4s(fheight);

        const viewport_upper_left = app_state.camera.pos + app_state.camera.forward - viewport_x * zm.f32x4s(0.5) - viewport_y * zm.f32x4s(0.5);
        state.pixel00_loc = viewport_upper_left + zm.f32x4s(0.5) * (state.pixel_delta_x + state.pixel_delta_y);
    }
};

const RasterState = struct {
    vao: u32,
    vshader: u32,
    fshader: u32,

    pipeline: u32,

    pub fn init(vbo: u32, ibo: u32) !RasterState {
        const raster_vao = vao: {
            var vao: u32 = undefined;
            gl.createVertexArrays(1, &vao);

            const vertex_buffer_index = 0;
            gl.vertexArrayVertexBuffer(vao, vertex_buffer_index, vbo, 0, @sizeOf(Vertex));

            gl.vertexArrayElementBuffer(vao, ibo);

            const a_tangent = 0;
            gl.vertexArrayAttribFormat(vao, a_tangent, 4, gl.FLOAT, gl.FALSE, @offsetOf(Vertex, "tangent"));
            gl.vertexArrayAttribBinding(vao, a_tangent, vertex_buffer_index);
            gl.enableVertexArrayAttrib(vao, a_tangent);

            const a_pos = 1;
            gl.vertexArrayAttribFormat(vao, a_pos, 3, gl.FLOAT, gl.FALSE, @offsetOf(Vertex, "position"));
            gl.vertexArrayAttribBinding(vao, a_pos, vertex_buffer_index);
            gl.enableVertexArrayAttrib(vao, a_pos);

            const a_u = 2;
            gl.vertexArrayAttribFormat(vao, a_u, 1, gl.FLOAT, gl.FALSE, @offsetOf(Vertex, "u"));
            gl.vertexArrayAttribBinding(vao, a_u, vertex_buffer_index);
            gl.enableVertexArrayAttrib(vao, a_u);

            const a_normal = 3;
            gl.vertexArrayAttribFormat(vao, a_normal, 3, gl.FLOAT, gl.FALSE, @offsetOf(Vertex, "normal"));
            gl.vertexArrayAttribBinding(vao, a_normal, vertex_buffer_index);
            gl.enableVertexArrayAttrib(vao, a_normal);

            const a_v = 4;
            gl.vertexArrayAttribFormat(vao, a_v, 1, gl.FLOAT, gl.FALSE, @offsetOf(Vertex, "v"));
            gl.vertexArrayAttribBinding(vao, a_v, vertex_buffer_index);
            gl.enableVertexArrayAttrib(vao, a_v);

            break :vao vao;
        };

        const vshader = try compileStaticShader("shaders/pbr.vert.glsl", gl.VERTEX_SHADER);
        const fshader = try compileStaticShader("shaders/pbr.frag.glsl", gl.FRAGMENT_SHADER);

        const shader_pipeline = pipeline: {
            var pipeline: u32 = undefined;
            gl.genProgramPipelines(1, &pipeline);
            gl.useProgramStages(pipeline, gl.VERTEX_SHADER_BIT, vshader);
            gl.useProgramStages(pipeline, gl.FRAGMENT_SHADER_BIT, fshader);

            break :pipeline pipeline;
        };

        gl.enable(gl.DEPTH_TEST);

        return .{
            .vao = raster_vao,
            .vshader = vshader,
            .fshader = fshader,
            .pipeline = shader_pipeline,
        };
    }

    pub fn render(self: *const RasterState, scene_data: *const SceneData, app_state: *const AppState) void {
        const fwidth: f32 = @floatFromInt(app_state.framebuffer_width);
        const fheight: f32 = @floatFromInt(app_state.framebuffer_height);
        const projection = zm.perspectiveFovRhGl(0.5 * std.math.pi, fwidth / fheight, 0.01, 100);
        const view = app_state.camera.getViewMatrix();

        gl.bindProgramPipeline(self.pipeline);

        // Fragment
        gl.programUniform3fv(self.fshader, 0, 1, zm.arr3Ptr(&app_state.camera.pos));

        // Vertex
        gl.programUniformMatrix4fv(self.vshader, 1, 1, gl.FALSE, zm.arrNPtr(&view));
        gl.programUniformMatrix4fv(self.vshader, 2, 1, gl.FALSE, zm.arrNPtr(&projection));

        gl.bindVertexArray(self.vao);

        // Turn on sRGB framebuffer for drawing scene.
        //gl.enable(gl.FRAMEBUFFER_SRGB);
        for (scene_data.meshes.items) |*mesh| {
            self.drawMesh(mesh, scene_data);
        }
    }

    fn drawMesh(self: *const RasterState, mesh: *const Mesh, scene_data: *const SceneData) void {
        gl.programUniformMatrix4fv(self.vshader, 0, 1, gl.FALSE, zm.arrNPtr(&mesh.transform));
        gl.programUniformMatrix3fv(self.vshader, 3, 1, gl.FALSE, &mesh.normal_matrix);

        gl.enable(gl.CULL_FACE);
        gl.cullFace(gl.BACK);

        for (scene_data.primitives.items[mesh.primitives_start..mesh.primitives_end]) |*prim| {
            const material = &scene_data.materials.items[prim.material_index];

            if (material.is_double_sided) {
                gl.disable(gl.CULL_FACE);
            } else {
                gl.enable(gl.CULL_FACE);
            }

            self.setMaterialUniformsPBR(material, scene_data);

            gl.drawElementsBaseVertex(
                gl.TRIANGLES,
                @intCast(prim.indices_end - prim.indices_start),
                gl.UNSIGNED_INT,
                @ptrFromInt(@sizeOf(u32) * prim.indices_start),
                @intCast(prim.vertices_start),
            );
        }
    }

    pub fn setMaterialUniformsPBR(self: *const RasterState, material: *const zgltf.Material, scene_data: *const SceneData) void {
        const Flags = packed struct(u32) {
            has_base_color: bool = false,
            has_metallic_roughness: bool = false,
            has_occlusion: bool = false,
            has_normal_map: bool = false,
            pad: u28 = undefined,
        };

        var flags: Flags = .{};

        // BaseColor
        if (material.metallic_roughness.base_color_texture) |*tex_info| {
            const handle_idx = tex_info.index;
            const texture_handle = scene_data.texure_handles.items[handle_idx];
            gl.bindTextureUnit(0, texture_handle);
            flags.has_base_color = true;
        }
        // MetallicRoughness
        if (material.metallic_roughness.metallic_roughness_texture) |*tex_info| {
            const handle_idx = tex_info.index;
            const texture_handle = scene_data.texure_handles.items[handle_idx];
            gl.bindTextureUnit(1, texture_handle);
            flags.has_metallic_roughness = true;
        }
        // AmbientOcculsion
        if (material.occlusion_texture) |*tex_info| {
            const handle_idx = tex_info.index;
            const texture_handle = scene_data.texure_handles.items[handle_idx];
            gl.bindTextureUnit(2, texture_handle);
            flags.has_occlusion = true;
        }
        // NormalMap
        if (material.normal_texture) |*tex_info| {
            const handle = scene_data.texure_handles.items[tex_info.index];
            gl.bindTextureUnit(3, handle);
            flags.has_normal_map = true;
        }

        flags.has_normal_map = flags.has_normal_map and use_normal_map;
        flags.has_occlusion = flags.has_occlusion and use_occlusion_map;

        gl.programUniform4fv(self.fshader, 1, 1, &material.metallic_roughness.base_color_factor); // 1
        gl.programUniform1f(self.fshader, 2, material.metallic_roughness.metallic_factor); // 2
        gl.programUniform1f(self.fshader, 3, material.metallic_roughness.roughness_factor); // 3
        gl.programUniform1ui(self.fshader, 4, @bitCast(flags)); // 4

    }
};

var use_normal_map: bool = true;
var use_occlusion_map: bool = true;

const Camera = struct {
    pos: zm.Vec = .{ 0, 0, 3, 0 },
    forward: zm.Vec = .{ 0, 0, -1, 0 },
    right: zm.Vec = .{ 1, 0, 0, 0 },
    up: zm.Vec = .{ 0, 1, 0, 0 },
    pitch: f32 = 0.0,
    yaw: f32 = -std.math.degreesToRadians(90.0),

    pub fn processInput(self: *Camera, mouse_delta: [2]f32, mouse_sens: f32) void {
        const x_offset = mouse_delta[0] * mouse_sens;
        const y_offset = mouse_delta[1] * mouse_sens;

        self.yaw -= x_offset;
        self.pitch -= y_offset;

        const max_angle = comptime std.math.degreesToRadians(89);
        self.pitch = std.math.clamp(self.pitch, -max_angle, max_angle);
        self.updateVectors();
    }

    pub fn updateVectors(self: *Camera) void {
        self.forward = zm.f32x4(
            @cos(self.yaw) * @cos(self.pitch),
            @sin(self.pitch),
            @sin(self.yaw) * @cos(self.pitch),
            0,
        );

        const world_up: zm.Vec = .{ 0, 1, 0, 0 };
        self.forward = zm.normalize3(self.forward);
        self.right = zm.normalize3(zm.cross3(self.forward, world_up));
        self.up = zm.normalize3(zm.cross3(self.right, self.forward));
    }

    pub fn getViewMatrix(self: Camera) zm.Mat {
        return zm.lookAtRh(self.pos, self.pos + self.forward, self.up);
    }
};

const Vertex = extern struct {
    tangent: [4]f32,
    position: [3]f32,
    u: f32, // Splitting the UV to avoid padding did somewhat hurt raster performance but improved raytracing perf.
    normal: [3]f32,
    v: f32,
};

const Primitive = struct {
    vertices_start: u32,
    vertices_end: u32,
    indices_start: u32,
    indices_end: u32,
    material_index: u32,
};

const Mesh = struct {
    transform: zm.Mat,
    normal_matrix: [9]f32,
    primitives_start: u32,
    primitives_end: u32,

    /// Needed so that the normal matrix gets properly adjusted.
    pub fn setTransform(self: *Mesh, transform: zm.Mat) void {
        self.transform = transform;
        self.normal_matrix = self.computeNormalMatrix();
    }

    pub fn computeNormalMatrix(self: *const Mesh) [9]f32 {
        var normal_matrix = zm.inverse(self.transform);
        normal_matrix = zm.transpose(normal_matrix);

        var normal: [9]f32 = undefined;
        normal[0..3].* = .{ normal_matrix[0][0], normal_matrix[0][1], normal_matrix[0][2] };
        normal[3..6].* = .{ normal_matrix[1][0], normal_matrix[1][1], normal_matrix[1][2] };
        normal[6..9].* = .{ normal_matrix[2][0], normal_matrix[2][1], normal_matrix[2][2] };
        return normal;
    }
};

const SceneData = struct {
    geom_arena: std.heap.ArenaAllocator,
    gpu_data: GPUData = .{},

    meshes: std.ArrayListUnmanaged(Mesh) = .{},
    primitives: std.ArrayListUnmanaged(Primitive) = .{},
    materials: std.ArrayListUnmanaged(zgltf.Material) = .{},
    texure_handles: std.ArrayListUnmanaged(u32) = .{},

    pub fn init(allocator: Allocator, file_directories: []const []const u8) !SceneData {
        var self: SceneData = .{
            .geom_arena = std.heap.ArenaAllocator.init(allocator),
        };

        // Memory handling
        const tmp_mem = try allocator.alignedAlloc(u8, std.mem.page_size, 1 << 29);
        defer allocator.free(tmp_mem);

        var tmp_linear_alloc = std.heap.FixedBufferAllocator.init(tmp_mem);
        const tmp_alloc = tmp_linear_alloc.allocator();
        const geom_arena = self.geom_arena.allocator();

        var fmt_buf: [1024]u8 = undefined;

        // Loading glTF files.
        for (file_directories) |dir| {
            const path = try std.fmt.bufPrint(&fmt_buf, "assets/{s}/{s}.gltf", .{ dir, dir });
            const gltf_file_contents = try std.fs.cwd().readFileAllocOptions(
                tmp_alloc,
                path,
                1 << 26,
                null,
                4,
                null,
            );
            defer tmp_alloc.free(gltf_file_contents);

            const gltf_bin = try std.fs.cwd().readFileAllocOptions(
                tmp_alloc,
                try std.fmt.bufPrint(&fmt_buf, "assets/{s}/{s}.bin", .{ dir, dir }),
                1 << 26,
                null,
                4,
                null,
            );
            defer tmp_alloc.free(gltf_bin);

            // Parse file.
            var gltf = zgltf.init(tmp_alloc);
            defer gltf.deinit();
            try gltf.parse(gltf_file_contents);

            // Parse meshes and their primitives
            const meshes = try self.meshes.addManyAsSlice(allocator, gltf.data.meshes.items.len);
            for (gltf.data.meshes.items, meshes) |*gltf_mesh, *our_mesh| {
                our_mesh.primitives_start = @intCast(self.primitives.items.len);

                const prims = try self.primitives.addManyAsSlice(allocator, gltf_mesh.primitives.items.len);
                for (gltf_mesh.primitives.items, prims) |*gltf_prim, *prim| {
                    prim.* = try parsePrimitive(geom_arena, &gltf, gltf_bin, gltf_prim, &self.gpu_data, self.materials.items.len);
                }

                our_mesh.primitives_end = @intCast(self.primitives.items.len);
            }

            // Set transforms
            setMeshTransforms(meshes, &gltf);

            // Save materials.
            try self.saveMaterialsAndTextures(allocator, tmp_alloc, &gltf, dir);
        }

        // TEMP.
        {
            var transform = zm.mul(zm.rotationY(0.5 * std.math.pi), zm.scalingV(@splat(0.75)));
            transform = zm.mul(transform, zm.translation(0, 1, 0));
            self.meshes.items[0].setTransform(transform);
        }
        //if (self.meshes.items.len > 1) {
        //    const transform = zm.mul(zm.scalingV(@splat(0.3)), zm.translation(1.5, 1, 0));
        //    self.meshes.items[1].setTransform(transform);
        //}

        try self.constructBVH(tmp_alloc);

        return self;
    }

    pub fn uploadDataToGPU(self: *const SceneData, tmp: Allocator) ![4]u32 {
        const GPUBLAS = extern struct {
            aabb_min: [3]f32,
            root_idx: u32,
            aabb_max: [3]f32,
            index_offset: u32,
        };

        const GPUPrimitive = extern struct {
            blas: GPUBLAS,
            base_index: u32,
            base_vertex: u32,
            mesh_index: u32,
            material_idx: u32,
        };

        const blas_instances = self.gpu_data.blas_instances.items;
        const primitives = self.primitives.items;
        const gpu_prims = try tmp.alloc(GPUPrimitive, self.gpu_data.blas_instances.items.len);
        defer tmp.free(gpu_prims);

        var prim_idx: u32 = 0;
        var node_sum: usize = 0;
        for (self.meshes.items, 0..) |*mesh, mesh_idx| {
            for (blas_instances[mesh.primitives_start..mesh.primitives_end], primitives[mesh.primitives_start..mesh.primitives_end]) |*cpu_blas, *cpu_prim| {
                std.debug.assert(cpu_prim.indices_start % 3 == 0);
                const gpu_blas: GPUBLAS = .{
                    .aabb_min = cpu_blas.bounds.min,
                    .aabb_max = cpu_blas.bounds.max,
                    .root_idx = @intCast(node_sum),
                    .index_offset = cpu_prim.indices_start / 3,
                };

                const gpu_prim: GPUPrimitive = .{
                    .blas = gpu_blas,
                    .base_index = cpu_prim.indices_start,
                    .base_vertex = cpu_prim.vertices_start,
                    .mesh_index = @intCast(mesh_idx),
                    .material_idx = cpu_prim.material_index,
                };

                gpu_prims[prim_idx] = gpu_prim;
                node_sum += cpu_blas.nodes.len;
                prim_idx += 1;
            }
        }

        var buffer_handles: [4]u32 = undefined;
        gl.createBuffers(buffer_handles.len, &buffer_handles);

        try self.uploadTransforms(tmp, buffer_handles[3], buffer_handles[1]);

        {
            const primitive_handle = buffer_handles[0];
            const buffer_size = @sizeOf(GPUPrimitive) * gpu_prims.len;
            gl.namedBufferStorage(primitive_handle, @intCast(buffer_size), gpu_prims.ptr, 0);
        }

        {
            const Flags = packed struct(u32) {
                double_sided: bool = false,
                has_normal_texture: bool = false,
                has_metallic_roughness_texture: bool = false,
                has_base_color_texture: bool = false,
                pad: u28 = undefined,
            };

            const GPUMaterial = extern struct {
                base_color_factor: [4]f32,
                emissive_factor: [3]f32,
                metallic_factor: f32,
                base_color_texture: u64 = std.math.maxInt(u64),
                metallic_roughness_texture: u64 = std.math.maxInt(u64),
                normal_map_texture: u64 = std.math.maxInt(u64),
                roughness_factor: f32,
                emissive_strength: f32,
                flags: Flags = .{},
                pad: [3]f32 = undefined,
            };

            const gpu_materials = try tmp.alloc(GPUMaterial, self.materials.items.len);
            defer tmp.free(gpu_materials);

            for (gpu_materials, self.materials.items) |*gpu_material, *cpu_material| {
                gpu_material.* = .{
                    .base_color_factor = cpu_material.metallic_roughness.base_color_factor,
                    .metallic_factor = cpu_material.metallic_roughness.metallic_factor,
                    .roughness_factor = cpu_material.metallic_roughness.roughness_factor,
                    .emissive_factor = cpu_material.emissive_factor,
                    .emissive_strength = cpu_material.emissive_strength,
                };

                // Make bindless handles.
                if (cpu_material.metallic_roughness.base_color_texture) |info| {
                    const tex = self.texure_handles.items[info.index];
                    gpu_material.base_color_texture = gl.GL_ARB_bindless_texture.getTextureHandleARB(tex);
                    gl.GL_ARB_bindless_texture.makeTextureHandleResidentARB(gpu_material.base_color_texture);
                    gpu_material.flags.has_base_color_texture = true;
                }
                if (cpu_material.metallic_roughness.metallic_roughness_texture) |info| {
                    const tex = self.texure_handles.items[info.index];
                    gpu_material.metallic_roughness_texture = gl.GL_ARB_bindless_texture.getTextureHandleARB(tex);
                    gl.GL_ARB_bindless_texture.makeTextureHandleResidentARB(gpu_material.metallic_roughness_texture);
                    gpu_material.flags.has_metallic_roughness_texture = true;
                }
                if (cpu_material.normal_texture) |info| {
                    const tex = self.texure_handles.items[info.index];
                    gpu_material.normal_map_texture = gl.GL_ARB_bindless_texture.getTextureHandleARB(tex);
                    gl.GL_ARB_bindless_texture.makeTextureHandleResidentARB(gpu_material.normal_map_texture);
                    gpu_material.flags.has_normal_texture = true;
                }

                gpu_material.flags.double_sided = cpu_material.is_double_sided;
            }

            const handle = buffer_handles[2];
            const buffer_size = @sizeOf(GPUMaterial) * gpu_materials.len;
            gl.namedBufferStorage(handle, @intCast(buffer_size), gpu_materials.ptr, 0);
        }

        return buffer_handles;
    }

    pub fn uploadTransforms(self: *const SceneData, tmp: Allocator, normal_matricies_handle: u32, mesh_transforms_handle: u32) !void {
        const mesh_transforms = try tmp.alloc(f32, self.meshes.items.len * 16);
        defer tmp.free(mesh_transforms);

        const normal_matricies = try tmp.alloc(f32, self.meshes.items.len * 12);
        defer tmp.free(normal_matricies);

        for (self.meshes.items, 0..) |*mesh, idx| {
            zm.storeMat(mesh_transforms[idx * 16 .. (idx + 1) * 16], zm.inverse(mesh.transform));
            // TODO: Investiage the difference between this and a straight memcpy.
            normal_matricies[idx * 12 .. (idx + 1) * 12][0..3].* = mesh.normal_matrix[0..3].*;
            normal_matricies[idx * 12 .. (idx + 1) * 12][4..7].* = mesh.normal_matrix[3..6].*;
            normal_matricies[idx * 12 .. (idx + 1) * 12][8..11].* = mesh.normal_matrix[6..9].*;
        }

        const buffer_size = @sizeOf(f32) * mesh_transforms.len;
        gl.namedBufferStorage(mesh_transforms_handle, @intCast(buffer_size), mesh_transforms.ptr, 0);
        gl.namedBufferStorage(normal_matricies_handle, @intCast(@sizeOf(f32) * normal_matricies.len), normal_matricies.ptr, 0);
    }

    pub fn constructBVH(self: *SceneData, tmp: Allocator) !void {
        var bvh_triangles = std.ArrayList(bvh.Tri).init(tmp);
        defer bvh_triangles.deinit();
        const indices = self.gpu_data.indices.items;
        const vertices = self.gpu_data.vertices.items;
        std.debug.assert(indices.len % 3 == 0);

        var nodes_used: usize = 0;
        const node_storage = try self.gpu_data.blas_nodes.addManyAsSlice(self.geom_arena.allocator(), 2 * (indices.len / 3 + 1));
        const blas_indices = try self.gpu_data.blas_index_buffer.addManyAsSlice(self.geom_arena.allocator(), indices.len);
        try self.gpu_data.blas_instances.ensureTotalCapacityPrecise(self.geom_arena.allocator(), self.primitives.items.len);

        for (self.meshes.items) |*mesh| {
            for (self.primitives.items[mesh.primitives_start..mesh.primitives_end]) |*prim| {
                // Convert triangles.
                {
                    bvh_triangles.shrinkRetainingCapacity(0);
                    const prim_indices = indices[prim.indices_start..prim.indices_end];
                    const prim_vertices = vertices[prim.vertices_start..prim.vertices_end];
                    try bvh_triangles.ensureTotalCapacityPrecise(prim_indices.len / 3);

                    var idx: u32 = 0;
                    while (idx < prim_indices.len) : (idx += 3) {
                        const v0 = prim_vertices[prim_indices[idx + 0]].position;
                        const v1 = prim_vertices[prim_indices[idx + 1]].position;
                        const v2 = prim_vertices[prim_indices[idx + 2]].position;
                        bvh_triangles.appendAssumeCapacity(.{
                            .v0 = v0,
                            .v1 = v1,
                            .v2 = v2,
                            .centroid = bvh.Tri.computeCentroid(v0, v1, v2),
                        });
                    }
                }

                const blas = try bvh.BLAS.init(
                    node_storage[nodes_used..],
                    bvh_triangles.items,
                    blas_indices[prim.indices_start / 3 ..],
                    mesh.transform,
                );

                try self.gpu_data.blas_instances.append(self.geom_arena.allocator(), blas);
                nodes_used += blas.nodes.len;
            }
        }

        self.gpu_data.blas_nodes.items = self.gpu_data.blas_nodes.items[0..nodes_used];

        const tlas = try bvh.TLAS.init(self.geom_arena.allocator(), tmp, self.gpu_data.blas_instances.items);
        self.gpu_data.tlas_nodes.items = tlas.nodes;
    }

    pub fn deinit(self: *SceneData, alloc: Allocator) void {
        self.primitives.deinit(alloc);
        self.materials.deinit(alloc);
        self.texure_handles.deinit(alloc);
        self.meshes.deinit(alloc);
    }

    fn setMeshTransforms(meshes_for_current_file: []Mesh, gltf: *const zgltf) void {
        for (gltf.data.nodes.items) |*node| {
            if (node.mesh) |mesh_idx| {
                var transform = zm.matFromQuat(zm.loadArr4(node.rotation));
                transform = zm.mul(transform, zm.scalingV(zm.loadArr3(node.scale)));
                transform = zm.mul(transform, zm.translationV(zm.loadArr3(node.translation)));
                meshes_for_current_file[mesh_idx].setTransform(transform);
            }
        }
    }

    fn saveMaterialsAndTextures(
        self: *SceneData,
        allocator: Allocator,
        tmp: Allocator,
        gltf: *const zgltf,
        gltf_dir: []const u8,
    ) !void {
        const our_materials = try self.materials.addManyAsSlice(allocator, gltf.data.materials.items.len);
        for (our_materials, gltf.data.materials.items) |*our_material, *gltf_material| {
            our_material.* = gltf_material.*;

            if (our_material.metallic_roughness.base_color_texture) |*col_texture| {
                try self.saveTexture(tmp, allocator, &col_texture.index, gltf, gltf_dir, true);
            }

            if (our_material.metallic_roughness.metallic_roughness_texture) |*texture_info| {
                try self.saveTexture(tmp, allocator, &texture_info.index, gltf, gltf_dir, false);
            }

            if (our_material.occlusion_texture) |*texture_info| {
                try self.saveTexture(tmp, allocator, &texture_info.index, gltf, gltf_dir, false);
            }

            if (our_material.normal_texture) |*texture_info| {
                try self.saveTexture(tmp, allocator, &texture_info.index, gltf, gltf_dir, false);
            }
        }
    }

    fn saveTexture(
        self: *SceneData,
        tmp: Allocator,
        allocator: Allocator,
        texture_index: *usize,
        gltf: *const zgltf,
        gltf_dir: []const u8,
        is_srgb: bool,
    ) !void {
        var fmt_buf: [1024]u8 = undefined;
        zstbi.init(tmp);
        defer zstbi.deinit();
        const texture_gltf = gltf.data.textures.items[texture_index.*];
        const image = gltf.data.images.items[texture_gltf.source.?];
        const texture_path = try std.fmt.bufPrintZ(&fmt_buf, "assets/{s}/{s}", .{ gltf_dir, image.uri.? });

        var texture = try zstbi.Image.loadFromFile(texture_path, 4);
        defer texture.deinit();

        const handle = try self.texure_handles.addOne(allocator);
        gl.createTextures(gl.TEXTURE_2D, 1, handle);

        const sampler = if (texture_gltf.sampler) |sampler_idx| gltf.data.samplers.items[sampler_idx] else zgltf.TextureSampler{};
        //const sampler = gltf.data.samplers.items[];
        handle.* = try createTexture(texture, sampler, is_srgb);

        // TODO: Make this the bindless handle later.
        texture_index.* = self.texure_handles.items.len - 1;
    }

    fn parsePrimitive(
        geom_arena: Allocator,
        gltf: *const zgltf,
        gltf_bin: []align(4) const u8,
        gltf_prim: *const zgltf.Primitive,
        gpu_data: *GPUData,
        total_material_count: usize,
    ) !Primitive {
        const vertices_start = gpu_data.vertices.items.len;
        const indices_start = gpu_data.indices.items.len;

        // Retrive indices.
        const index_accessor = gltf.data.accessors.items[gltf_prim.indices.?];
        const indices = try gpu_data.indices.addManyAsSlice(geom_arena, @intCast(index_accessor.count));

        // Store all indices as u32s (lazy.)
        switch (index_accessor.component_type) {
            inline else => |component| {
                const T = switch (component) {
                    .byte => i8,
                    .unsigned_byte => u8,
                    .short => i16,
                    .unsigned_short => u16,
                    .unsigned_integer => u32,
                    .float => @panic("Floating point indices are not supported"),
                };

                var it = index_accessor.iterator(T, gltf, gltf_bin);
                var idx: u32 = 0;
                while (it.next()) |val| : (idx += 1) {
                    std.debug.assert(val.len == 1);
                    indices[idx] = @intCast(val[0]);
                }
            },
        }

        // "All attribute accessors for a given primitive MUST have the same count."" [https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#meshes]
        // This mean that we can just look at the "count" of the first attribute and use that to preallocate vertices.
        const accesor_idx = switch (gltf_prim.attributes.items[0]) {
            inline else => |idx| idx,
        };
        const vertex_count = gltf.data.accessors.items[accesor_idx].count;
        const vertices = try gpu_data.vertices.addManyAsSlice(geom_arena, @intCast(vertex_count));

        // Retrive vertex attributes.
        for (gltf_prim.attributes.items) |attrib| {
            switch (attrib) {
                .tangent => |idx| {
                    const accessor = gltf.data.accessors.items[idx];
                    var it = accessor.iterator(f32, gltf, gltf_bin);
                    var i: u32 = 0;

                    while (it.next()) |v| : (i += 1) {
                        vertices[i].tangent = v[0..4].*;
                    }
                },
                .position => |idx| {
                    const accessor = gltf.data.accessors.items[idx];
                    var it = accessor.iterator(f32, gltf, gltf_bin);
                    var i: u32 = 0;

                    while (it.next()) |v| : (i += 1) {
                        vertices[i].position = v[0..3].*;
                    }
                },
                .normal => |idx| {
                    const accessor = gltf.data.accessors.items[idx];
                    var it = accessor.iterator(f32, gltf, gltf_bin);
                    var i: u32 = 0;
                    while (it.next()) |v| : (i += 1) {
                        vertices[i].normal = v[0..3].*;
                    }
                },
                .texcoord => |idx| {
                    const accessor = gltf.data.accessors.items[idx];
                    var it = accessor.iterator(f32, gltf, gltf_bin);
                    var i: u32 = 0;
                    while (it.next()) |v| : (i += 1) {
                        vertices[i].u = v[0];
                        vertices[i].v = v[1];
                    }
                },
                else => {},
            }
        }

        return .{
            .vertices_start = @intCast(vertices_start),
            .vertices_end = @intCast(gpu_data.vertices.items.len),
            .indices_start = @intCast(indices_start),
            .indices_end = @intCast(gpu_data.indices.items.len),
            .material_index = @intCast(total_material_count + gltf_prim.material.?),
        };
    }
};

/// Stores information that can be freed once uploaded to the GPU.
const GPUData = struct {
    /// Unified backing buffer for vertices.
    vertices: std.ArrayListUnmanaged(Vertex) = .{},

    /// Unified backing buffer for indices.
    indices: std.ArrayListUnmanaged(u32) = .{},

    blas_index_buffer: std.ArrayListUnmanaged(u32) = .{},
    blas_instances: std.ArrayListUnmanaged(bvh.BLAS) = .{},
    blas_nodes: std.ArrayListUnmanaged(bvh.BLAS.Node) = .{},
    tlas_nodes: std.ArrayListUnmanaged(bvh.TLAS.Node) = .{},

    pub fn upload(self: GPUData) struct {
        vbo: u32,
        ibo: u32,

        blas_nodes: u32,
        blas_index_buffer: u32,
        tlas_nodes: u32,
    } {
        var buffer_handles: [5]u32 = undefined;
        gl.createBuffers(buffer_handles.len, &buffer_handles);

        const vbo, const ibo, const blas_nodes, const blas_index_buffer, const tlas_nodes = buffer_handles;
        gl.namedBufferStorage(vbo, @intCast(@sizeOf(Vertex) * self.vertices.items.len), self.vertices.items.ptr, 0);
        gl.namedBufferStorage(ibo, @intCast(@sizeOf(u32) * self.indices.items.len), self.indices.items.ptr, 0);

        gl.namedBufferStorage(blas_nodes, @intCast(@sizeOf(bvh.BLAS.Node) * self.blas_nodes.items.len), self.blas_nodes.items.ptr, 0);
        gl.namedBufferStorage(blas_index_buffer, @intCast(@sizeOf(u32) * self.blas_index_buffer.items.len), self.blas_index_buffer.items.ptr, 0);

        gl.namedBufferStorage(tlas_nodes, @intCast(@sizeOf(bvh.TLAS.Node) * self.tlas_nodes.items.len), self.tlas_nodes.items.ptr, 0);

        return .{
            .vbo = vbo,
            .ibo = ibo,
            .blas_nodes = blas_nodes,
            .blas_index_buffer = blas_index_buffer,
            .tlas_nodes = tlas_nodes,
        };
    }
};

fn createTexture(texture: zstbi.Image, sampler: zgltf.TextureSampler, srgb: bool) !u32 {
    if (texture.num_components != 4 and srgb) {
        std.log.err("sRGB textures must be RGBA (4 components)", .{});
        return error.Non4ComponentsRGBImage;
    }

    var handle: u32 = undefined;
    gl.createTextures(gl.TEXTURE_2D, 1, &handle);

    var min_filter: u32 = if (sampler.min_filter) |min_filter| @intFromEnum(min_filter) else gl.LINEAR;
    const mag_fitler: u32 = if (sampler.mag_filter) |mag_fitler| @intFromEnum(mag_fitler) else gl.LINEAR;

    // If texture is color data.
    // FIXME: Ignoring specified sampler min filter here.
    if (srgb) {
        min_filter = gl.LINEAR_MIPMAP_LINEAR;
    }

    gl.textureParameteri(handle, gl.TEXTURE_MIN_FILTER, @intCast(min_filter));
    gl.textureParameteri(handle, gl.TEXTURE_MAG_FILTER, @intCast(mag_fitler));
    gl.textureParameteri(handle, gl.TEXTURE_WRAP_S, @intCast(@intFromEnum(sampler.wrap_s)));
    gl.textureParameteri(handle, gl.TEXTURE_WRAP_T, @intCast(@intFromEnum(sampler.wrap_t)));

    const max_dim = @max(texture.width, texture.height);
    const mip_levels: i32 = @intCast(std.math.log2(max_dim) + 1);

    gl.textureStorage2D(
        handle,
        if (srgb) mip_levels else 1,
        if (srgb) gl.SRGB8_ALPHA8 else gl.RGBA8,
        @intCast(texture.width),
        @intCast(texture.height),
    );

    gl.textureSubImage2D(
        handle,
        0,
        0,
        0,
        @intCast(texture.width),
        @intCast(texture.height),
        gl.RGBA,
        gl.UNSIGNED_BYTE,
        texture.data.ptr,
    );

    if (srgb) {
        gl.generateTextureMipmap(handle);
    }

    return handle;
}

pub fn reloadShader(
    temp: Allocator,
    pipeline: u32,
    shader: *u32,
    shader_type: enum(gl.GLenum) { compute, fragment, vertex },
    path: []const u8,
) !void {
    var timer = try std.time.Timer.start();

    const shader_gl_type: u32 = switch (shader_type) {
        .compute => gl.COMPUTE_SHADER,
        .fragment => gl.FRAGMENT_SHADER,
        .vertex => gl.VERTEX_SHADER,
    };

    const source = try std.fs.cwd().readFileAllocOptions(temp, path, 15 * 1000, null, @alignOf(u8), 0);
    defer temp.free(source);
    const new_shader = gl.createShaderProgramv(shader_gl_type, 1, &source.ptr);
    errdefer gl.deleteProgram(new_shader);

    var msg: [1024]u8 = undefined;
    if (try checkForCompileErrors(&msg, new_shader)) |error_log| {
        std.log.err("Failed to link shader {s}: {s}", .{ path, error_log });
        return error.ShaderCompilationFailed;
    }

    const shader_gl_bit: u32 = switch (shader_type) {
        .compute => gl.COMPUTE_SHADER_BIT,
        .fragment => gl.FRAGMENT_SHADER_BIT,
        .vertex => gl.VERTEX_SHADER_BIT,
    };

    gl.useProgramStages(pipeline, shader_gl_bit, 0);
    gl.deleteProgram(shader.*);

    shader.* = new_shader;
    gl.useProgramStages(pipeline, shader_gl_bit, shader.*);
    gl.validateProgramPipeline(pipeline);

    std.log.info("Sucesfully reloaded {s} shader in {d}ms!", .{ path, @as(f32, @floatFromInt(timer.read())) / std.time.ns_per_ms });
}

/// For shaders that should never be reloaded during runtime. Source code will be embedded in the binary.
fn compileStaticShader(comptime path: []const u8, shader_type: gl.GLenum) !u32 {
    const source: []const u8 = @embedFile(path);
    const shader = gl.createShaderProgramv(shader_type, 1, &source.ptr);

    var msg: [1024]u8 = undefined;
    if (try checkForCompileErrors(&msg, shader)) |error_log| {
        std.log.err("Failed to link shader {s}: {s}", .{ path, error_log });
        return error.ShaderCompilationFailed;
    }

    return shader;
}

fn checkForCompileErrors(buffer: []u8, shader: u32) !?[]const u8 {
    var linked: i32 = undefined;
    gl.getProgramiv(shader, gl.LINK_STATUS, &linked);
    if (linked != gl.TRUE) {
        gl.getProgramInfoLog(shader, @intCast(buffer.len), null, buffer.ptr);

        const end = std.mem.indexOfScalar(u8, buffer, 0) orelse return error.NotEnoughSpace;
        return buffer[0..end];
    }

    return null;
}

fn glLoadProc(_: void, procname: [:0]const u8) ?gl.FunctionPointer {
    return @ptrCast(glfw.getProcAddress(procname));
}

fn glfwErrorCallback(err: i32, msg: *?[:0]const u8) callconv(.C) void {
    std.log.err("GLFW error {d}: {?s}", .{ err, msg });
}

fn glDebugCallback(
    _: gl.GLenum,
    _: gl.GLenum,
    id: gl.GLuint,
    _: gl.GLenum,
    _: gl.GLsizei,
    message: ?[*:0]const u8,
    _: ?*const anyopaque,
) callconv(.C) void {
    if (id == 131169 or id == 131185 or id == 131218 or id == 131204) return;
    std.log.err("{?s}", .{message});
}

fn cursorPosCallback(
    window: *glfw.Window,
    x_pos_in: f64,
    y_pos_in: f64,
) callconv(.C) void {
    const app_state = window.getUserPointer(AppState).?;
    if (!app_state.lock_cursor) return;

    const xpos: f32 = @floatCast(x_pos_in);
    const ypos: f32 = @floatCast(y_pos_in);

    if (app_state.first_mouse) {
        app_state.mouse_pos = .{ xpos, ypos };
        app_state.first_mouse = false;
    }

    const x_offset = app_state.mouse_pos[0] - xpos;
    const y_offset = ypos - app_state.mouse_pos[1];

    app_state.mouse_pos = .{ xpos, ypos };
    app_state.mouse_delta = .{ x_offset, y_offset };
    app_state.camera.processInput(app_state.mouse_delta, 0.001);
    app_state.rt_state.should_reset_fb = true;
}

fn frameBufferSizeCallback(window: *glfw.Window, width: i32, height: i32) callconv(.C) void {
    const app_state = window.getUserPointer(AppState).?;
    gl.viewport(0, 0, width, height);
    app_state.framebuffer_width = @intCast(width);
    app_state.framebuffer_height = @intCast(height);
    app_state.rt_state.resize(@intCast(width), @intCast(height));

    const window_width, const window_height = window.getSize();
    app_state.window_width = @intCast(window_width);
    app_state.window_height = @intCast(window_height);
}
