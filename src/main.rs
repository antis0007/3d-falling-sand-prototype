mod app;
mod chunk_store;
mod edits;
mod floating_origin;
mod input;
mod meshing;
mod player;
mod procgen;
mod renderer;
mod sim;
mod sim_world;
mod streaming;
mod streaming_manager;
mod types;
mod ui;
mod world;
mod world_bounds;
mod world_manager;
mod world_stream;

fn main() -> anyhow::Result<()> {
    env_logger::init();
    pollster::block_on(app::run())
}
