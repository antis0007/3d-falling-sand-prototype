use anyhow::Context;
use lz4_flex::{compress_prepend_size, decompress_size_prepended};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs;
use std::path::Path;

pub type MaterialId = u16;
pub const EMPTY: MaterialId = 0;
pub const CHUNK_SIZE: usize = 16;
const CHUNK_VOLUME: usize = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BrushShape {
    Sphere,
    Cube,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BrushMode {
    Place,
    Erase,
}

#[derive(Clone, Copy, Debug)]
pub struct BrushSettings {
    pub radius: i32,
    pub shape: BrushShape,
    pub mode: BrushMode,
    pub max_distance: f32,
}

impl Default for BrushSettings {
    fn default() -> Self {
        Self {
            radius: 1,
            shape: BrushShape::Sphere,
            mode: BrushMode::Place,
            max_distance: 16.0,
        }
    }
}

#[derive(Clone)]
pub struct Chunk {
    voxels: [MaterialId; CHUNK_VOLUME],
    pub dirty_mesh: bool,
    pub active: HashSet<u16>,
    pub settled: Vec<u8>,
}

impl Chunk {
    pub fn new() -> Self {
        Self {
            voxels: [EMPTY; CHUNK_VOLUME],
            dirty_mesh: true,
            active: HashSet::new(),
            settled: vec![0; CHUNK_VOLUME],
        }
    }

    #[inline]
    pub fn index(x: usize, y: usize, z: usize) -> usize {
        x + y * CHUNK_SIZE + z * CHUNK_SIZE * CHUNK_SIZE
    }

    pub fn get(&self, x: usize, y: usize, z: usize) -> MaterialId {
        self.voxels[Self::index(x, y, z)]
    }

    pub fn set(&mut self, x: usize, y: usize, z: usize, id: MaterialId) {
        let idx = Self::index(x, y, z);
        self.voxels[idx] = id;
        self.settled[idx] = 0;
        self.active.insert(idx as u16);
        self.dirty_mesh = true;
    }

    pub fn iter_raw(&self) -> &[MaterialId] {
        &self.voxels
    }

    pub fn iter_raw_mut(&mut self) -> &mut [MaterialId] {
        &mut self.voxels
    }
}

#[derive(Clone)]
pub struct World {
    pub dims: [usize; 3],
    pub chunks_dims: [usize; 3],
    pub chunks: Vec<Chunk>,
}

impl World {
    pub fn new(dims: [usize; 3]) -> Self {
        let chunks_dims = [
            dims[0] / CHUNK_SIZE,
            dims[1] / CHUNK_SIZE,
            dims[2] / CHUNK_SIZE,
        ];
        let mut world = Self {
            dims,
            chunks_dims,
            chunks: vec![Chunk::new(); chunks_dims[0] * chunks_dims[1] * chunks_dims[2]],
        };
        world.fill_floor(2, 1);
        world
    }

    pub fn chunk_index(&self, cx: usize, cy: usize, cz: usize) -> usize {
        cx + cy * self.chunks_dims[0] + cz * self.chunks_dims[0] * self.chunks_dims[1]
    }

    fn split_coord(v: usize) -> (usize, usize) {
        (v / CHUNK_SIZE, v % CHUNK_SIZE)
    }

    fn world_to_chunk(
        &self,
        x: usize,
        y: usize,
        z: usize,
    ) -> Option<(usize, usize, usize, usize, usize, usize)> {
        if x >= self.dims[0] || y >= self.dims[1] || z >= self.dims[2] {
            return None;
        }
        let (cx, lx) = Self::split_coord(x);
        let (cy, ly) = Self::split_coord(y);
        let (cz, lz) = Self::split_coord(z);
        Some((cx, cy, cz, lx, ly, lz))
    }

    pub fn get(&self, x: i32, y: i32, z: i32) -> MaterialId {
        if x < 0 || y < 0 || z < 0 {
            return EMPTY;
        }
        let (x, y, z) = (x as usize, y as usize, z as usize);
        if let Some((cx, cy, cz, lx, ly, lz)) = self.world_to_chunk(x, y, z) {
            self.chunks[self.chunk_index(cx, cy, cz)].get(lx, ly, lz)
        } else {
            EMPTY
        }
    }

    pub fn set(&mut self, x: i32, y: i32, z: i32, id: MaterialId) -> bool {
        if x < 0 || y < 0 || z < 0 {
            return false;
        }
        let (x, y, z) = (x as usize, y as usize, z as usize);
        let Some((cx, cy, cz, lx, ly, lz)) = self.world_to_chunk(x, y, z) else {
            return false;
        };
        let chunk_idx = self.chunk_index(cx, cy, cz);
        if self.chunks[chunk_idx].get(lx, ly, lz) == id {
            return false;
        }
        self.chunks[chunk_idx].set(lx, ly, lz, id);
        self.activate_neighbors(x as i32, y as i32, z as i32);
        true
    }

    pub fn activate_neighbors(&mut self, x: i32, y: i32, z: i32) {
        for dz in -1..=1 {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let nx = x + dx;
                    let ny = y + dy;
                    let nz = z + dz;
                    if nx < 0 || ny < 0 || nz < 0 {
                        continue;
                    }
                    let (ux, uy, uz) = (nx as usize, ny as usize, nz as usize);
                    let Some((cx, cy, cz, lx, ly, lz)) = self.world_to_chunk(ux, uy, uz) else {
                        continue;
                    };
                    let chunk_idx = self.chunk_index(cx, cy, cz);
                    let local = Chunk::index(lx, ly, lz);
                    self.chunks[chunk_idx].active.insert(local as u16);
                    self.chunks[chunk_idx].settled[local] = 0;
                }
            }
        }
    }

    pub fn fill_floor(&mut self, height: usize, mat: MaterialId) {
        for z in 0..self.dims[2] as i32 {
            for y in 0..height as i32 {
                for x in 0..self.dims[0] as i32 {
                    self.set(x, y, z, mat);
                }
            }
        }
    }

    pub fn clear(&mut self) {
        for c in &mut self.chunks {
            c.iter_raw_mut().fill(EMPTY);
            c.active.clear();
            c.settled.fill(0);
            c.dirty_mesh = true;
        }
        self.fill_floor(2, 1);
    }

    pub fn apply_brush(
        &mut self,
        center: [i32; 3],
        brush: BrushSettings,
        place_id: MaterialId,
        force_mode: Option<BrushMode>,
    ) {
        let mode = force_mode.unwrap_or(brush.mode);
        let rad = brush.radius.max(0);
        for dz in -rad..=rad {
            for dy in -rad..=rad {
                for dx in -rad..=rad {
                    let include = match brush.shape {
                        BrushShape::Sphere => (dx * dx + dy * dy + dz * dz) <= rad * rad,
                        BrushShape::Cube => dx.abs().max(dy.abs()).max(dz.abs()) <= rad,
                    };
                    if !include {
                        continue;
                    }
                    let p = [center[0] + dx, center[1] + dy, center[2] + dz];
                    let target = match mode {
                        BrushMode::Place => place_id,
                        BrushMode::Erase => EMPTY,
                    };
                    self.set(p[0], p[1], p[2], target);
                }
            }
        }
    }
}

#[derive(Serialize, Deserialize)]
struct SaveData {
    magic: [u8; 4],
    version: u16,
    dims: [usize; 3],
    chunks: Vec<Vec<(u16, u16)>>,
}

fn rle_encode(data: &[u16]) -> Vec<(u16, u16)> {
    if data.is_empty() {
        return vec![];
    }
    let mut out = Vec::new();
    let mut cur = data[0];
    let mut run: u16 = 1;
    for &v in &data[1..] {
        if v == cur && run < u16::MAX {
            run += 1;
        } else {
            out.push((cur, run));
            cur = v;
            run = 1;
        }
    }
    out.push((cur, run));
    out
}

fn rle_decode(encoded: &[(u16, u16)], out: &mut [u16]) {
    let mut i = 0;
    for &(value, run) in encoded {
        for _ in 0..run {
            if i >= out.len() {
                return;
            }
            out[i] = value;
            i += 1;
        }
    }
}

pub fn save_world(path: &Path, world: &World) -> anyhow::Result<()> {
    let chunks = world
        .chunks
        .iter()
        .map(|c| rle_encode(c.iter_raw()))
        .collect();
    let save = SaveData {
        magic: *b"VXL3",
        version: 1,
        dims: world.dims,
        chunks,
    };
    let bytes = bincode::serialize(&save)?;
    let packed = compress_prepend_size(&bytes);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, packed)?;
    Ok(())
}

pub fn load_world(path: &Path) -> anyhow::Result<World> {
    let bytes = fs::read(path)?;
    let unpacked = decompress_size_prepended(&bytes)?;
    let save: SaveData = bincode::deserialize(&unpacked)?;
    anyhow::ensure!(save.magic == *b"VXL3", "bad save magic");
    anyhow::ensure!(
        save.version == 1,
        "unsupported save version {}",
        save.version
    );
    let mut world = World::new(save.dims);
    anyhow::ensure!(
        save.chunks.len() == world.chunks.len(),
        "chunk count mismatch"
    );
    for (chunk, encoded) in world.chunks.iter_mut().zip(save.chunks.iter()) {
        chunk.iter_raw_mut().fill(EMPTY);
        rle_decode(encoded, chunk.iter_raw_mut());
        chunk.dirty_mesh = true;
        chunk.active.clear();
        chunk.settled.fill(0);
        let occupied = chunk.iter_raw().to_vec();
        for (i, id) in occupied.into_iter().enumerate() {
            if id != EMPTY {
                chunk.active.insert(i as u16);
            }
        }
    }
    Ok(world)
}

pub fn default_save_path() -> anyhow::Result<std::path::PathBuf> {
    let mut path = std::env::current_dir().context("cwd")?;
    path.push("saves");
    path.push("world.vxl");
    Ok(path)
}
