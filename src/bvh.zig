const std = @import("std");
const zm = @import("zmath.zig");
const Allocator = std.mem.Allocator;

pub const Tri = struct {
    v0: [3]f32,
    v1: [3]f32,
    v2: [3]f32,
    centroid: [3]f32,

    pub fn computeCentroid(v_0: [3]f32, v_1: [3]f32, v_2: [3]f32) [3]f32 {
        const v0 = zm.loadArr3(v_0);
        const v1 = zm.loadArr3(v_1);
        const v2 = zm.loadArr3(v_2);
        var ret: [3]f32 = undefined;
        const c = (v0 + v1 + v2) * zm.f32x4s(0.3333);
        zm.storeArr3(&ret, c);
        return ret;
    }
};

pub const BLAS = struct {
    nodes: []Node,
    bounds: AABB,

    pub fn init(node_storage: []Node, triangles: []Tri, blas_index_buffer_storage: []u32, transform: zm.Mat) !BLAS {
        const nodes = try constructInPlace(node_storage, triangles, blas_index_buffer_storage);

        var blas: BLAS = .{
            .nodes = nodes,
            .bounds = .{},
        };
        blas.setTransform(transform);

        return blas;
    }

    pub fn setTransform(blas: *BLAS, transform: zm.Mat) void {
        var min: [3]f32 = undefined;
        var max: [3]f32 = undefined;
        zm.storeArr3(&min, zm.mul(transform, zm.loadArr3(blas.nodes[0].aabb_min)));
        zm.storeArr3(&max, zm.mul(transform, zm.loadArr3(blas.nodes[0].aabb_max)));

        const bmax = blas.nodes[0].aabb_max;
        const bmin = blas.nodes[0].aabb_min;
        blas.bounds = .{};
        for (0..8) |i| {
            const corner = zm.f32x4(
                if (i & 1 != 0) bmax[0] else bmin[0],
                if (i & 2 != 0) bmax[1] else bmin[1],
                if (i & 4 != 0) bmax[2] else bmin[2],
                1,
            );

            const new_corner = zm.mul(corner, transform);

            var tmp: [3]f32 = undefined;
            zm.storeArr3(&tmp, new_corner);
            blas.bounds.grow(tmp);
        }
    }

    pub const Node = extern struct {
        aabb_min: [3]f32,
        left_or_first: u32,
        aabb_max: [3]f32,
        tris_count: u32,

        fn updateBounds(self: *Node, all_triangles: []const Tri, indices: []const u32) void {
            var min = zm.f32x4s(std.math.inf(f32));
            var max = zm.f32x4s(-std.math.inf(f32));
            const node_indices = indices[self.left_or_first .. self.left_or_first + self.tris_count];
            for (node_indices) |i| {
                const leaf_tri = &all_triangles[i];
                const v0 = zm.loadArr3(leaf_tri.v0);
                const v1 = zm.loadArr3(leaf_tri.v1);
                const v2 = zm.loadArr3(leaf_tri.v2);

                min = @min(min, v0);
                min = @min(min, v1);
                min = @min(min, v2);
                max = @max(max, v0);
                max = @max(max, v1);
                max = @max(max, v2);
            }
            zm.storeArr3(&self.aabb_min, min);
            zm.storeArr3(&self.aabb_max, max);
        }

        fn evalSAH(self: *const Node, tri: []const Tri, indices: []const u32, axis: usize, pos: f32) f32 {
            var left_box: AABB = .{};
            var right_box: AABB = .{};
            var left_count: u32 = 0;
            var right_count: u32 = 0;
            for (0..self.tris_count) |i| {
                const triangle = &tri[indices[self.left_or_first + i]];
                if (triangle.centroid[axis] < pos) {
                    left_count += 1;
                    left_box.grow(triangle.v0);
                    left_box.grow(triangle.v1);
                    left_box.grow(triangle.v2);
                } else {
                    right_count += 1;
                    right_box.grow(triangle.v0);
                    right_box.grow(triangle.v1);
                    right_box.grow(triangle.v2);
                }
            }

            const cost = @as(f32, @floatFromInt(left_count)) * left_box.area() + @as(f32, @floatFromInt(right_count)) * right_box.area();
            return if (cost > 0) cost else std.math.inf(f32);
        }

        inline fn determineBestSplitPlane(self: *const Node, all_triangles: []const Tri, indices: []u32) struct { f32, u32, f32 } {
            var best_axis: u32 = 0;
            var best_pos: f32 = 0;
            var best_cost = std.math.inf(f32);
            for (0..3) |axis| {
                var bounds_min: f32 = std.math.inf(f32);
                var bounds_max: f32 = -std.math.inf(f32);

                for (0..self.tris_count) |i| {
                    const tri = &all_triangles[indices[self.left_or_first + i]];
                    bounds_min = @min(bounds_min, tri.centroid[axis]);
                    bounds_max = @max(bounds_max, tri.centroid[axis]);
                }
                if (bounds_min == bounds_max) continue;

                const num_split_planes = 8;
                const scale = (bounds_max - bounds_min) / num_split_planes;
                for (0..num_split_planes) |i| {
                    const fi: f32 = @floatFromInt(i);
                    const candidate_pos = bounds_min + fi * scale;
                    const cost = evalSAH(self, all_triangles, indices, axis, candidate_pos);
                    if (cost < best_cost) {
                        best_pos = candidate_pos;
                        best_axis = @intCast(axis);
                        best_cost = cost;
                    }
                }
            }

            return .{ best_cost, best_axis, best_pos };
        }

        inline fn computeCost(node: *const Node) f32 {
            const min = zm.loadArr3(node.aabb_min);
            const max = zm.loadArr3(node.aabb_max);
            const extent = max - min;

            const surface_area = extent[0] * extent[1] + extent[1] * extent[2] + extent[2] * extent[0];
            return @as(f32, @floatFromInt(node.tris_count)) * surface_area;
        }

        fn subdivide(self: *Node, all_triangles: []Tri, node_storage: []Node, indices: []u32, nodes_used: *u32) void {
            const split_cost, const split_axis, const split_pos = self.determineBestSplitPlane(all_triangles, indices);
            const no_split_cost = self.computeCost();

            if (split_cost >= no_split_cost) return;

            var i = self.left_or_first;
            var j = i + self.tris_count - 1;
            while (i <= j) {
                if (all_triangles[indices[i]].centroid[split_axis] < split_pos) {
                    i += 1;
                } else {
                    std.mem.swap(u32, &indices[i], &indices[j]);
                    j -= 1;
                }
            }

            const left_count = i - self.left_or_first;
            if (left_count == 0 or left_count == self.tris_count) return;

            // create child nodes
            const left_child_idx = nodes_used.*;
            nodes_used.* += 1;
            const right_child_idx = nodes_used.*;
            nodes_used.* += 1;

            node_storage[left_child_idx].left_or_first = self.left_or_first;
            node_storage[left_child_idx].tris_count = left_count;
            node_storage[right_child_idx].left_or_first = i;
            node_storage[right_child_idx].tris_count = self.tris_count - left_count;
            self.left_or_first = left_child_idx;
            self.tris_count = 0;

            node_storage[left_child_idx].updateBounds(all_triangles, indices);
            node_storage[right_child_idx].updateBounds(all_triangles, indices);
            node_storage[left_child_idx].subdivide(all_triangles, node_storage, indices, nodes_used);
            node_storage[right_child_idx].subdivide(all_triangles, node_storage, indices, nodes_used);
        }
    };
};

pub const TLAS = struct {
    blas: []BLAS,
    nodes: []Node,

    pub const Node = extern struct {
        aabb_min: [3]f32,
        left_right: u32,
        aabb_max: [3]f32,
        blas_idx: u32,
    };

    pub fn init(allocator: Allocator, temp: Allocator, blas_instances: []BLAS) !TLAS {
        const node_allocation_size = (blas_instances.len + 64) * 2;
        const tlas_nodes = try allocator.alignedAlloc(Node, 64, node_allocation_size);
        const node_indices = try temp.alloc(u32, node_allocation_size);
        defer temp.free(node_indices);
        var nodes_used: u32 = 1;

        // assign leaf nodes.
        for (tlas_nodes[1 .. blas_instances.len + 1], blas_instances, 0..) |*node, *blas, idx| {
            node_indices[idx] = @intCast(1 + idx);
            node.aabb_min = blas.bounds.min;
            node.aabb_max = blas.bounds.max;
            node.blas_idx = @intCast(idx);
            node.left_right = 0;
            nodes_used += 1;
        }

        var node_index: u32 = @intCast(blas_instances.len);
        var a: u32 = 0;
        var b: u32 = findBestMatch(tlas_nodes, node_indices, node_index, a);

        while (node_index > 1) {
            const c = findBestMatch(tlas_nodes, node_indices, node_index, b);
            if (a == c) {
                const idx_a = node_indices[a];
                const idx_b = node_indices[b];
                const node_a = &tlas_nodes[idx_a];
                const node_b = &tlas_nodes[idx_b];
                const new_node = &tlas_nodes[nodes_used];
                new_node.left_right = idx_a + (idx_b << 16);
                const new_max = @max(zm.loadArr3(node_a.aabb_max), zm.loadArr3(node_b.aabb_max));
                const new_min = @min(zm.loadArr3(node_a.aabb_min), zm.loadArr3(node_b.aabb_min));
                zm.storeArr3(&new_node.aabb_max, new_max);
                zm.storeArr3(&new_node.aabb_min, new_min);
                node_indices[a] = nodes_used;
                node_indices[b] = node_indices[node_index - 1];
                b = findBestMatch(tlas_nodes, node_indices, node_index - 1, a);
                node_index -= 1;
                nodes_used += 1;
            } else {
                a = b;
                b = c;
            }
        }
        tlas_nodes[0] = tlas_nodes[node_indices[a]];

        return .{
            .nodes = tlas_nodes,
            .blas = blas_instances,
        };
    }

    fn findBestMatch(nodes: []const Node, indices: []u32, n: u32, a: u32) u32 {
        var smallest = std.math.inf(f32);
        var best_b: u32 = std.math.maxInt(u32);

        for (0..n) |b| if (b != a) {
            const bmax = @max(zm.loadArr3(nodes[indices[a]].aabb_max), zm.loadArr3(nodes[indices[b]].aabb_max));
            const bmin = @min(zm.loadArr3(nodes[indices[a]].aabb_min), zm.loadArr3(nodes[indices[b]].aabb_min));
            const e = bmax - bmin;
            const surface_area = e[0] * e[1] + e[1] * e[2] + e[2] * e[0];
            if (surface_area < smallest) {
                smallest = surface_area;
                best_b = @intCast(b);
            }
        };

        return best_b;
    }
};

pub fn construct(allocator: Allocator, triangles: []Tri) ![]BLAS.Node {
    const highest_possible_node_count = 2 * triangles.len + 1;

    var nodes_used: u32 = 1;
    const nodes = try allocator.alignedAlloc(BLAS.Node, 32, highest_possible_node_count);

    const root = &nodes[0];
    root.left_or_first = 0;
    root.tris_count = @intCast(triangles.len);
    root.updateBounds(triangles);
    root.subdivide(triangles, nodes, &nodes_used);

    return nodes[0..nodes_used];
}

pub fn constructInPlace(node_storage: []BLAS.Node, triangles: []Tri, indices: []u32) ![]BLAS.Node {
    const highest_possible_node_count = 2 * triangles.len + 1;
    if (node_storage.len < highest_possible_node_count) return error.BufferTooSmall;

    for (indices, 0..) |*i, k| i.* = @intCast(k);

    var nodes_used: u32 = 1;
    const root = &node_storage[0];
    root.left_or_first = 0;
    root.tris_count = @intCast(triangles.len);
    root.updateBounds(triangles, indices);
    root.subdivide(triangles, node_storage, indices, &nodes_used);

    return node_storage[0..nodes_used];
}

const AABB = struct {
    min: [3]f32 = [_]f32{std.math.inf(f32)} ** 3,
    max: [3]f32 = [_]f32{-std.math.inf(f32)} ** 3,

    pub fn grow(self: *AABB, p: [3]f32) void {
        zm.storeArr3(&self.min, @min(zm.loadArr3(p), zm.loadArr3(self.min)));
        zm.storeArr3(&self.max, @max(zm.loadArr3(p), zm.loadArr3(self.max)));
    }

    pub fn growAABB(self: *AABB, other: AABB) void {
        zm.storeArr3(&self.min, @min(zm.loadArr3(other.min), zm.loadArr3(self.min)));
        zm.storeArr3(&self.max, @max(zm.loadArr3(other.max), zm.loadArr3(self.max)));
    }

    pub fn area(self: AABB) f32 {
        const extent = zm.loadArr3(self.max) - zm.loadArr3(self.min);
        return extent[0] * extent[1] + extent[1] * extent[2] + extent[2] * extent[0];
    }
};
