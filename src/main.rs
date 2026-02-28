mod app;
mod input;
mod player;
mod procgen;
mod renderer;
mod sim;
mod ui;
mod world;
mod world_manager;
mod world_stream;

fn main() -> anyhow::Result<()> {
    env_logger::init();
    pollster::block_on(app::run())
}
