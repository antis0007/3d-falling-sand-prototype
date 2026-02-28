#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProceduralWorldBounds {
    pub min_macro_y: i32,
    pub max_macro_y: i32,
    pub min_world_y: i32,
    pub max_world_y: i32,
}

impl ProceduralWorldBounds {
    pub fn new(min_macro_y: i32, max_macro_y: i32, chunk_size_y: i32) -> Self {
        let chunk_size_y = chunk_size_y.max(1);
        let (min_macro_y, max_macro_y) = if min_macro_y <= max_macro_y {
            (min_macro_y, max_macro_y)
        } else {
            (max_macro_y, min_macro_y)
        };
        let min_world_y = min_macro_y.saturating_mul(chunk_size_y);
        let max_world_y = (max_macro_y.saturating_add(1))
            .saturating_mul(chunk_size_y)
            .saturating_sub(1);
        Self {
            min_macro_y,
            max_macro_y,
            min_world_y,
            max_world_y,
        }
    }

    pub fn clamp_macro_coord(self, coord: [i32; 3]) -> [i32; 3] {
        [
            coord[0],
            coord[1].clamp(self.min_macro_y, self.max_macro_y),
            coord[2],
        ]
    }

    pub fn contains_macro_coord(self, coord: [i32; 3]) -> bool {
        self.contains_macro_y(coord[1])
    }

    pub fn contains_macro_y(self, y: i32) -> bool {
        y >= self.min_macro_y && y <= self.max_macro_y
    }

    pub fn contains_global_y(self, y: i32) -> bool {
        y >= self.min_world_y && y <= self.max_world_y
    }
}
