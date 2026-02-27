mod app;
mod input;
mod renderer;
mod sim;
mod ui;
mod world;

fn main() -> anyhow::Result<()> {
    env_logger::init();
    pollster::block_on(app::run())
}
