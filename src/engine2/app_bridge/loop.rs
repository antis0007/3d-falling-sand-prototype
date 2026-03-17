//! Engine2-owned app loop bridge.

use std::sync::Arc;

use anyhow::Context;
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, Event, WindowEvent};
use winit::event_loop::EventLoop;
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::WindowBuilder;

use crate::engine2::commands::{EditSphereCommand, EngineCommand};
use crate::engine2::gpu::buffers::BufferPoolConfig;
use crate::engine2::phases::SimInput;
use crate::engine2::render::camera::CameraState;
use crate::engine2::Engine2State;

#[derive(Debug)]
struct Engine2Presenter {
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
}

impl Engine2Presenter {
    fn new(
        instance: &wgpu::Instance,
        window: &'static winit::window::Window,
        adapter: &wgpu::Adapter,
        device: &wgpu::Device,
    ) -> anyhow::Result<Self> {
        let surface = instance.create_surface(window)?;
        let size = window.inner_size();
        let caps = surface.get_capabilities(adapter);
        let format = *caps
            .formats
            .first()
            .context("engine2 surface has no supported formats")?;
        let present_mode = if caps.present_modes.contains(&wgpu::PresentMode::Fifo) {
            wgpu::PresentMode::Fifo
        } else {
            *caps
                .present_modes
                .first()
                .context("engine2 surface has no present modes")?
        };
        let alpha_mode = *caps
            .alpha_modes
            .first()
            .context("engine2 surface has no alpha mode")?;

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode,
            alpha_mode,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(device, &config);

        Ok(Self { surface, config })
    }

    fn resize(&mut self, device: &wgpu::Device, size: PhysicalSize<u32>) {
        self.config.width = size.width.max(1);
        self.config.height = size.height.max(1);
        self.surface.configure(device, &self.config);
    }

    fn render(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        resident_count: usize,
        edited: bool,
    ) {
        let frame = match self.surface.get_current_texture() {
            Ok(frame) => frame,
            Err(err) => {
                log::warn!("[engine2] surface acquire failed: {err}");
                return;
            }
        };

        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let density = (resident_count as f64 / 16.0).clamp(0.0, 1.0) as f64;
        let clear = if edited {
            wgpu::Color {
                r: 0.15,
                g: 0.45 + density * 0.4,
                b: 0.2,
                a: 1.0,
            }
        } else {
            wgpu::Color {
                r: 0.05,
                g: 0.08 + density * 0.3,
                b: 0.25 + density * 0.5,
                a: 1.0,
            }
        };

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("engine2.present.encoder"),
        });
        {
            let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("engine2.present.clear"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(clear),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
        }

        queue.submit(std::iter::once(encoder.finish()));
        frame.present();
    }
}

#[derive(Debug, Default)]
pub struct EngineLoop {
    pub state: Engine2State,
}

impl EngineLoop {
    /// Orchestrates one engine2 frame with explicit phase ordering.
    pub fn tick_frame(&mut self, camera: CameraState) -> bool {
        self.state.request_minimum_world(camera);
        self.state.materialization_step(8);
        self.state.residency_update_step();
        let upload_output = self.state.upload_step();
        let edit_output = self.state.command_application_step();
        let sim_output = self.state.active_scheduling_step(SimInput {
            upload: upload_output,
            edit: edit_output.clone(),
        });
        self.state.queue_upload_step(&sim_output);
        let extract = self.state.render_extraction_step(camera, sim_output);
        let _draw = self.state.draw_step(extract);
        !edit_output.dirty_pages.is_empty()
    }
}

pub async fn run() -> anyhow::Result<()> {
    let event_loop = EventLoop::new()?;
    let window: &'static winit::window::Window = Box::leak(Box::new(
        WindowBuilder::new()
            .with_title("3D Falling Sand Prototype - engine2")
            .build(&event_loop)?,
    ));

    let instance = wgpu::Instance::default();
    let surface = instance.create_surface(window)?;
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        })
        .await
        .context("engine2 failed to acquire GPU adapter")?;

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor::default(), None)
        .await
        .context("engine2 failed to create device/queue")?;

    let mut presenter = Engine2Presenter::new(&instance, window, &adapter, &device)?;

    let device = Arc::new(device);
    let queue = Arc::new(queue);
    let mut loop_state = EngineLoop::default();
    loop_state
        .state
        .initialize_gpu(device.clone(), queue.clone(), BufferPoolConfig::default());

    let mut camera = CameraState::from_world_position([0.0, 8.0, 0.0], 128.0);

    event_loop.run(move |event, elwt| match event {
        Event::WindowEvent { event, .. } => match event {
            WindowEvent::CloseRequested => elwt.exit(),
            WindowEvent::Resized(size) => presenter.resize(device.as_ref(), size),
            WindowEvent::KeyboardInput { event, .. } => {
                if event.state == ElementState::Pressed {
                    match event.physical_key {
                        PhysicalKey::Code(KeyCode::Escape) => elwt.exit(),
                        PhysicalKey::Code(KeyCode::KeyE) => {
                            loop_state.state.commands.push(EngineCommand::EditSphere(
                                EditSphereCommand {
                                    center: crate::engine2::types::VoxelCoord { x: 0, y: 0, z: 0 },
                                    radius_voxels: 8.0,
                                    material: 3,
                                },
                            ));
                        }
                        PhysicalKey::Code(KeyCode::ArrowUp) => camera.world_y += 1.0,
                        PhysicalKey::Code(KeyCode::ArrowDown) => camera.world_y -= 1.0,
                        _ => {}
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                let edited = loop_state.tick_frame(camera);
                let resident = loop_state.state.residency.desired_count();
                presenter.render(device.as_ref(), queue.as_ref(), resident, edited);
                window.request_redraw();
            }
            _ => {}
        },
        Event::AboutToWait => window.request_redraw(),
        _ => {}
    })?;

    Ok(())
}
