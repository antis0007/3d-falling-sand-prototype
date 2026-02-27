use glam::{vec3, Vec2, Vec3};
use std::collections::HashSet;
use winit::event::{DeviceEvent, ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::keyboard::{KeyCode, PhysicalKey};

#[derive(Default, Clone)]
pub struct InputState {
    pub pressed: HashSet<KeyCode>,
    pub mouse_delta: Vec2,
    pub wheel: f32,
    pub lmb: bool,
    pub rmb: bool,
    pub just_lmb: bool,
    pub just_rmb: bool,
}

impl InputState {
    pub fn end_frame(&mut self) {
        self.mouse_delta = Vec2::ZERO;
        self.wheel = 0.0;
        self.just_lmb = false;
        self.just_rmb = false;
    }

    pub fn key(&self, key: KeyCode) -> bool {
        self.pressed.contains(&key)
    }

    pub fn on_window_event(&mut self, event: &WindowEvent) {
        match event {
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(code) = event.physical_key {
                    match event.state {
                        ElementState::Pressed => {
                            self.pressed.insert(code);
                        }
                        ElementState::Released => {
                            self.pressed.remove(&code);
                        }
                    }
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let down = *state == ElementState::Pressed;
                match button {
                    MouseButton::Left => {
                        self.just_lmb = down && !self.lmb;
                        self.lmb = down;
                    }
                    MouseButton::Right => {
                        self.just_rmb = down && !self.rmb;
                        self.rmb = down;
                    }
                    _ => {}
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                self.wheel += match delta {
                    MouseScrollDelta::LineDelta(_, y) => *y,
                    MouseScrollDelta::PixelDelta(pos) => pos.y as f32 * 0.05,
                };
            }
            _ => {}
        }
    }

    pub fn on_device_event(&mut self, event: &DeviceEvent) {
        if let DeviceEvent::MouseMotion { delta } = event {
            self.mouse_delta += Vec2::new(delta.0 as f32, delta.1 as f32);
        }
    }
}

#[derive(Clone)]
pub struct FpsController {
    pub position: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub vel_y: f32,
    pub flying: bool,
    pub last_space_t: f32,
    pub move_speed: f32,
    pub fly_speed: f32,
    pub jump_speed: f32,
    pub sensitivity: f32,
    pub double_jump_threshold: f32,
}

impl Default for FpsController {
    fn default() -> Self {
        Self {
            position: vec3(8.0, 6.0, 8.0),
            yaw: -90.0f32.to_radians(),
            pitch: 0.0,
            vel_y: 0.0,
            flying: false,
            last_space_t: -10.0,
            move_speed: 8.0,
            fly_speed: 10.0,
            jump_speed: 6.5,
            sensitivity: 0.001,
            double_jump_threshold: 0.28,
        }
    }
}

impl FpsController {
    pub fn look_dir(&self) -> Vec3 {
        Vec3::new(
            self.yaw.cos() * self.pitch.cos(),
            self.pitch.sin(),
            self.yaw.sin() * self.pitch.cos(),
        )
        .normalize_or_zero()
    }

    pub fn step(&mut self, input: &InputState, dt: f32, lock_mouse: bool, now_s: f32) {
        if lock_mouse {
            self.yaw += input.mouse_delta.x * self.sensitivity;
            self.pitch += input.mouse_delta.y * self.sensitivity;
            self.pitch = self.pitch.clamp(-1.54, 1.54);
        }

        let forward = Vec3::new(self.yaw.cos(), 0.0, self.yaw.sin()).normalize_or_zero();
        let right = Vec3::Y.cross(forward).normalize_or_zero();
        let mut move_dir = Vec3::ZERO;
        if input.key(KeyCode::KeyW) {
            move_dir += forward;
        }
        if input.key(KeyCode::KeyS) {
            move_dir -= forward;
        }
        if input.key(KeyCode::KeyA) {
            move_dir -= right;
        }
        if input.key(KeyCode::KeyD) {
            move_dir += right;
        }
        if move_dir.length_squared() > 0.0 {
            move_dir = move_dir.normalize();
        }

        if self.flying {
            if input.key(KeyCode::Space) {
                move_dir += Vec3::Y;
            }
            if input.key(KeyCode::ShiftLeft) {
                move_dir -= Vec3::Y;
            }
            self.position += move_dir * self.fly_speed * dt;
        } else {
            self.position += move_dir * self.move_speed * dt;
            self.vel_y -= 19.6 * dt;
            self.position.y += self.vel_y * dt;
            if self.position.y < 2.4 {
                self.position.y = 2.4;
                self.vel_y = 0.0;
            }
            if input.key(KeyCode::Space) && self.position.y <= 2.41 {
                self.vel_y = self.jump_speed;
            }
        }

        if input.key(KeyCode::Space) && (now_s - self.last_space_t) < self.double_jump_threshold {
            self.flying = !self.flying;
            self.vel_y = 0.0;
            self.last_space_t = -10.0;
        } else if input.key(KeyCode::Space) {
            self.last_space_t = now_s;
        }
    }
}
