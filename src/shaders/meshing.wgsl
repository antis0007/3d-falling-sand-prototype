const CHUNK_SIDE: u32 = 32u;
const CHUNK_VOLUME: u32 = CHUNK_SIDE * CHUNK_SIDE * CHUNK_SIDE;
const VOXEL_SIZE: f32 = 0.5;

const FACE_MASK_POS_X: u32 = 1u << 0u;
const FACE_MASK_NEG_X: u32 = 1u << 1u;
const FACE_MASK_POS_Y: u32 = 1u << 2u;
const FACE_MASK_NEG_Y: u32 = 1u << 3u;
const FACE_MASK_POS_Z: u32 = 1u << 4u;
const FACE_MASK_NEG_Z: u32 = 1u << 5u;

const MAX_FACES_PER_PAGE: u32 = GPU_MAX_FACES_PER_PAGE;
const DEBUG_COLOR_OVERRIDE_ENABLED: bool = false;
const DEBUG_COLOR_OVERRIDE_PAGE: u32 = 0u;

struct FrameParams {
    page_index: u32,
    voxel_count: u32,
    state_index: u32,
    _pad0: u32,
    neighbor_pages: vec4<u32>,
    neighbor_pages_tail: vec2<u32>,
    _pad1: vec2<u32>,
};

struct GpuVertex {
    position: vec3<f32>,
    color: u32,
};

struct DrawIndexedIndirectArgs {
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
};

struct ChunkMeshMeta {
    slot_index: u32,
    vertex_offset: u32,
    index_offset: u32,
    _pad: u32,
};

// Meshing bind-group contract (must match `GpuComputeRuntime::create_meshing_bind_group`):
//  0 atlas_voxels        (read)
//  1 face_mask           (read_write)
//  2 face_offset         (read_write)
//  3 chunk_vertex_buffer (read_write)
//  4 chunk_index_buffer  (read_write)
//  5 draw_indirect       (read_write)
//  6 frame/page params   (read)
//  7 face_count          (read_write)
//  8 mesh_meta           (read_write)
//  9 chunk_origin        (read)
// NOTE: legacy `page_indirect` is intentionally NOT part of this meshing WGSL contract.
@group(0) @binding(0) var<storage, read> atlas_voxels: array<u32>;
@group(0) @binding(1) var<storage, read_write> face_mask: array<u32>;
@group(0) @binding(2) var<storage, read_write> face_offset: array<u32>;
@group(0) @binding(3) var<storage, read_write> chunk_vertex_buffer: array<GpuVertex>;
@group(0) @binding(4) var<storage, read_write> chunk_index_buffer: array<u32>;
@group(0) @binding(5) var<storage, read_write> draw_indirect_buffer: array<DrawIndexedIndirectArgs>;
@group(0) @binding(6) var<storage, read> frame_params: array<FrameParams>;
@group(0) @binding(7) var<storage, read_write> face_count: array<u32>;
@group(0) @binding(8) var<storage, read_write> mesh_meta_buffer: array<ChunkMeshMeta>;
@group(0) @binding(9) var<storage, read> chunk_origin_buffer: array<vec4<f32>>;

fn atlas_state_offset(page: u32, state: u32) -> u32 {
    return page * (CHUNK_VOLUME * 2u) + state * CHUNK_VOLUME;
}

fn unpack(idx: u32) -> vec3<u32> {
    let z = idx / (CHUNK_SIDE * CHUNK_SIDE);
    let rem = idx - z * (CHUNK_SIDE * CHUNK_SIDE);
    let y = rem / CHUNK_SIDE;
    let x = rem - y * CHUNK_SIDE;
    return vec3<u32>(x, y, z);
}

fn pack(p: vec3<u32>) -> u32 {
    return p.x + p.y * CHUNK_SIDE + p.z * CHUNK_SIDE * CHUNK_SIDE;
}

fn in_bounds(p: vec3<i32>) -> bool {
    return all(p >= vec3<i32>(0)) && all(p < vec3<i32>(i32(CHUNK_SIDE)));
}

fn neighbor_dir(i: u32) -> vec3<i32> {
    switch i {
        case 0u: { return vec3<i32>(1,0,0); }
        case 1u: { return vec3<i32>(-1,0,0); }
        case 2u: { return vec3<i32>(0,1,0); }
        case 3u: { return vec3<i32>(0,-1,0); }
        case 4u: { return vec3<i32>(0,0,1); }
        default: { return vec3<i32>(0,0,-1); }
    }
}

fn neighbor_page_for_dir(params: FrameParams, dir: u32) -> u32 {
    // Host packs neighbor_pages as [-X, +X, -Y, +Y, -Z, +Z].
    switch dir {
        case 0u: { return params.neighbor_pages.y; } // +X
        case 1u: { return params.neighbor_pages.x; } // -X
        case 2u: { return params.neighbor_pages.w; } // +Y
        case 3u: { return params.neighbor_pages.z; } // -Y
        case 4u: { return params.neighbor_pages_tail.y; } // +Z
        default: { return params.neighbor_pages_tail.x; } // -Z
    }
}

fn voxel_at(params: FrameParams, base_off: u32, p: vec3<i32>, dir: u32) -> u32 {
    if (in_bounds(p)) {
        return atlas_voxels[base_off + pack(vec3<u32>(p))];
    }

    let neighbor_page = neighbor_page_for_dir(params, dir);
    if (neighbor_page == 0xffffffffu) {
        // Missing neighbors are treated as empty to keep border faces visible.
        return EMPTY;
    }

    var wrapped = p;
    if (wrapped.x < 0) { wrapped.x = i32(CHUNK_SIDE) - 1; }
    if (wrapped.x >= i32(CHUNK_SIDE)) { wrapped.x = 0; }
    if (wrapped.y < 0) { wrapped.y = i32(CHUNK_SIDE) - 1; }
    if (wrapped.y >= i32(CHUNK_SIDE)) { wrapped.y = 0; }
    if (wrapped.z < 0) { wrapped.z = i32(CHUNK_SIDE) - 1; }
    if (wrapped.z >= i32(CHUNK_SIDE)) { wrapped.z = 0; }

    let neighbor_off = atlas_state_offset(neighbor_page, params.state_index);
    return atlas_voxels[neighbor_off + pack(vec3<u32>(wrapped))];
}

fn write_face_quad(
    dir: u32,
    base: vec3<f32>,
    chunk_origin: vec3<f32>,
    color: u32,
    local_vertex_offset: u32,
    global_index_offset: u32,
) {
    var c0: vec3<f32>;
    var c1: vec3<f32>;
    var c2: vec3<f32>;
    var c3: vec3<f32>;

    switch dir {
        case 0u: { c0 = base + vec3<f32>(1,0,0); c1 = base + vec3<f32>(1,1,0); c2 = base + vec3<f32>(1,1,1); c3 = base + vec3<f32>(1,0,1); }
        case 1u: { c0 = base + vec3<f32>(0,0,1); c1 = base + vec3<f32>(0,1,1); c2 = base + vec3<f32>(0,1,0); c3 = base + vec3<f32>(0,0,0); }
        case 2u: { c0 = base + vec3<f32>(0,1,0); c1 = base + vec3<f32>(0,1,1); c2 = base + vec3<f32>(1,1,1); c3 = base + vec3<f32>(1,1,0); }
        case 3u: { c0 = base + vec3<f32>(0,0,1); c1 = base + vec3<f32>(0,0,0); c2 = base + vec3<f32>(1,0,0); c3 = base + vec3<f32>(1,0,1); }
        case 4u: { c0 = base + vec3<f32>(1,0,1); c1 = base + vec3<f32>(1,1,1); c2 = base + vec3<f32>(0,1,1); c3 = base + vec3<f32>(0,0,1); }
        default:{ c0 = base + vec3<f32>(0,0,0); c1 = base + vec3<f32>(0,1,0); c2 = base + vec3<f32>(1,1,0); c3 = base + vec3<f32>(1,0,0); }
    }

    let v0 = local_vertex_offset;
    let v1 = local_vertex_offset + 1u;
    let v2 = local_vertex_offset + 2u;
    let v3 = local_vertex_offset + 3u;

    chunk_vertex_buffer[v0] = GpuVertex(chunk_origin + c0 * VOXEL_SIZE, color);
    chunk_vertex_buffer[v1] = GpuVertex(chunk_origin + c1 * VOXEL_SIZE, color);
    chunk_vertex_buffer[v2] = GpuVertex(chunk_origin + c2 * VOXEL_SIZE, color);
    chunk_vertex_buffer[v3] = GpuVertex(chunk_origin + c3 * VOXEL_SIZE, color);

    chunk_index_buffer[global_index_offset + 0u] = local_vertex_offset + 0u;
    chunk_index_buffer[global_index_offset + 1u] = local_vertex_offset + 1u;
    chunk_index_buffer[global_index_offset + 2u] = local_vertex_offset + 2u;
    chunk_index_buffer[global_index_offset + 3u] = local_vertex_offset + 0u;
    chunk_index_buffer[global_index_offset + 4u] = local_vertex_offset + 2u;
    chunk_index_buffer[global_index_offset + 5u] = local_vertex_offset + 3u;
}

fn pack_rgba8(r: u32, g: u32, b: u32, a: u32) -> u32 {
    return (r & 0xffu) | ((g & 0xffu) << 8u) | ((b & 0xffu) << 16u) | ((a & 0xffu) << 24u);
}

fn material_color(material_id: u32) -> u32 {
    // Match renderer::Vertex::desc location(1) = Unorm8x4 with visible alpha for solid voxels.
    switch material_id {
        case 1u: { return pack_rgba8(120u, 120u, 120u, 255u); } // Stone
        case 2u: { return pack_rgba8(122u, 81u, 46u, 255u); } // Wood
        case 3u: { return pack_rgba8(194u, 178u, 128u, 255u); } // Sand
        case 4u: { return pack_rgba8(230u, 235u, 240u, 255u); } // Snow
        case WATER: { return pack_rgba8(64u, 120u, 220u, 255u); }
        case LAVA: { return pack_rgba8(230u, 100u, 30u, 255u); }
        case ACID: { return pack_rgba8(60u, 220u, 90u, 255u); }
        case SMOKE: { return pack_rgba8(120u, 120u, 120u, 255u); }
        case STEAM: { return pack_rgba8(190u, 190u, 210u, 255u); }
        case 10u: { return pack_rgba8(112u, 128u, 140u, 255u); } // Steel
        case FIRE_GAS: { return pack_rgba8(255u, 155u, 72u, 255u); }
        case TORCH: { return pack_rgba8(255u, 184u, 96u, 255u); }
        case EMBER_HOT: { return pack_rgba8(255u, 105u, 38u, 255u); }
        case EMBER_WARM: { return pack_rgba8(168u, 76u, 52u, 255u); }
        case EMBER_ASH: { return pack_rgba8(90u, 84u, 84u, 255u); }
        case DIRT: { return pack_rgba8(121u, 88u, 56u, 255u); }
        case TURF: { return pack_rgba8(96u, 186u, 88u, 255u); }
        case BUSH: { return pack_rgba8(74u, 156u, 70u, 255u); }
        case GRASS: { return pack_rgba8(96u, 186u, 88u, 255u); }
        case PLANT: { return pack_rgba8(94u, 186u, 72u, 255u); }
        case WEED: { return pack_rgba8(78u, 146u, 62u, 255u); }
        case TREE_SEED: { return pack_rgba8(142u, 92u, 52u, 255u); }
        case LEAVES: { return pack_rgba8(70u, 156u, 66u, 255u); }
        case DEAD_LEAF: { return pack_rgba8(170u, 114u, 60u, 255u); }
        default: { return pack_rgba8(255u, 0u, 255u, 255u); }
    }
}

fn debug_override_color(page: u32, dir: u32, material_id: u32) -> u32 {
    if (page == DEBUG_COLOR_OVERRIDE_PAGE) {
        if ((dir & 1u) == 0u) {
            return pack_rgba8(255u, 0u, 255u, 255u);
        }
        return pack_rgba8(0u, 255u, 0u, 255u);
    }
    return material_color(material_id);
}


fn is_billboard_material(material_id: u32) -> bool {
    return material_id == BUSH || material_id == GRASS;
}

fn material_occludes(self_id: u32, neighbor_id: u32) -> bool {
    if (neighbor_id == EMPTY) { return false; }
    if (neighbor_id == self_id) { return true; }
    if (is_billboard_material(neighbor_id)) { return false; }
    // Match CPU meshing intent for non-solid/transient media.
    if (
        neighbor_id == WATER ||
        neighbor_id == LAVA ||
        neighbor_id == ACID ||
        neighbor_id == SMOKE ||
        neighbor_id == STEAM ||
        neighbor_id == FIRE_GAS
    ) {
        return false;
    }
    return true;
}

@compute @workgroup_size(128)
fn detect_faces(@builtin(global_invocation_id) gid: vec3<u32>) {

    let voxel_idx = gid.x;
    if (voxel_idx >= CHUNK_VOLUME) { return; }

    let params = frame_params[0];
    let page = params.page_index;

    let mesh_meta = mesh_meta_buffer[page];
    let chunk_slot = mesh_meta.slot_index;

    let mask_base = chunk_slot * CHUNK_VOLUME;

    let src_off = atlas_state_offset(page, params.state_index);

    if (voxel_idx >= params.voxel_count) {
        face_mask[mask_base + voxel_idx] = 0u;
        return;
    }

    let id = atlas_voxels[src_off + voxel_idx];
    if (id == EMPTY || is_billboard_material(id)) {
        face_mask[mask_base + voxel_idx] = 0u;
        return;
    }

    let p = vec3<i32>(unpack(voxel_idx));

    var mask = 0u;

    if (!material_occludes(id, voxel_at(params, src_off, p + neighbor_dir(0u), 0u))) { mask |= FACE_MASK_POS_X; }
    if (!material_occludes(id, voxel_at(params, src_off, p + neighbor_dir(1u), 1u))) { mask |= FACE_MASK_NEG_X; }
    if (!material_occludes(id, voxel_at(params, src_off, p + neighbor_dir(2u), 2u))) { mask |= FACE_MASK_POS_Y; }
    if (!material_occludes(id, voxel_at(params, src_off, p + neighbor_dir(3u), 3u))) { mask |= FACE_MASK_NEG_Y; }
    if (!material_occludes(id, voxel_at(params, src_off, p + neighbor_dir(4u), 4u))) { mask |= FACE_MASK_POS_Z; }
    if (!material_occludes(id, voxel_at(params, src_off, p + neighbor_dir(5u), 5u))) { mask |= FACE_MASK_NEG_Z; }

    face_mask[mask_base + voxel_idx] = mask;
}

@compute @workgroup_size(1)
fn prefix_scan() {

    let params = frame_params[0];
    let page = params.page_index;

    let mesh_meta = mesh_meta_buffer[page];
    let chunk_slot = mesh_meta.slot_index;

    let mask_base = chunk_slot * CHUNK_VOLUME;
    let offset_base = chunk_slot * CHUNK_VOLUME;

    var running: u32 = 0u;

    for (var i: u32 = 0u; i < CHUNK_VOLUME; i = i + 1u) {
        face_offset[offset_base + i] = running;
        running = running + countOneBits(face_mask[mask_base + i]);
    }

    face_count[page] = min(running, MAX_FACES_PER_PAGE);
}

@compute @workgroup_size(128)
fn emit_mesh(
    @builtin(global_invocation_id) gid: vec3<u32>
) {

    let params = frame_params[0];
    let page = params.page_index;

    let mesh_meta = mesh_meta_buffer[page];
    let chunk_slot = mesh_meta.slot_index;

    let mask_base = chunk_slot * CHUNK_VOLUME;
    let offset_base = chunk_slot * CHUNK_VOLUME;

    let vertex_base = mesh_meta.vertex_offset;
    let index_base = mesh_meta.index_offset;

    let total_faces = face_count[page];
    let chunk_origin = chunk_origin_buffer[page].xyz;

    // --- always write draw command ---
    if (gid.x == 0u) {

        draw_indirect_buffer[chunk_slot].index_count = total_faces * 6u;
        draw_indirect_buffer[chunk_slot].instance_count = 1u;
        draw_indirect_buffer[chunk_slot].first_index = index_base;
        draw_indirect_buffer[chunk_slot].base_vertex = i32(vertex_base);
        draw_indirect_buffer[chunk_slot].first_instance = 0u;
    }

    let voxel_idx = gid.x;
    if (voxel_idx >= CHUNK_VOLUME) { return; }

    let src_off = atlas_state_offset(page, params.state_index);

    let id = atlas_voxels[src_off + voxel_idx];
    if (id == EMPTY) { return; }

    let mask = face_mask[mask_base + voxel_idx];
    if (mask == 0u) { return; }

    var write_face = face_offset[offset_base + voxel_idx];
    if (write_face >= total_faces) { return; }

    let base = vec3<f32>(vec3<u32>(unpack(voxel_idx)));

    for (var dir: u32 = 0u; dir < 6u; dir = dir + 1u) {

        if ((mask & (1u << dir)) == 0u) { continue; }
        if (write_face >= total_faces) { break; }

        let vo = write_face * 4u;
        let io = write_face * 6u;

        var color = material_color(id);
        if (DEBUG_COLOR_OVERRIDE_ENABLED) {
            color = debug_override_color(page, dir, id);
        }
        write_face_quad(dir, base, chunk_origin, color, vo, index_base + io);

        write_face = write_face + 1u;
    }
}
