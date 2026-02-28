mod app;
mod input;
mod player;
mod procgen;
mod renderer;
mod sim;
mod streaming_manager;
mod ui;
mod world;
mod world_manager;

fn main() -> anyhow::Result<()> {
    env_logger::init();
    pollster::block_on(app::run())
}
