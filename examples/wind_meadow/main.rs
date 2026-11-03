//! wind-meadow: a small meadow swaying in a gusting wind.
//!
//! Grass, flowers, ferns, bushes and two trees from the Quaternius Stylized
//! Nature MegaKit (CC0) are scattered over a ground plane. Every plant gets a
//! height-falloff sway mask, so its base stays planted while its top moves with
//! the wind. Gusts roll across the meadow as a travelling wave. Trees and bushes
//! take a smaller share of the wind than the grass. The rocks have no mask and
//! stay still.
//!
//! The app is a plain winit loop around a `ViewportInstance`. It steps the wind
//! plugin's CPU half with `step_runtime` and runs its GPU half with
//! `pre_prepare` before rendering, so the deformer sees this frame's wind.
//!
//! Controls:
//!   Left drag    : orbit
//!   Right drag   : pan
//!   Scroll       : zoom
//!   Space        : turn the wind on and off
//!
//! Run with `cargo run --release --example wind-meadow`.

mod models;

use std::f32::consts::{FRAC_PI_2, TAU};
use std::sync::Arc;
use std::time::Instant;

use glam::{Mat4, Quat, Vec2, Vec3};
use viewport_lib::input::adapters::{AdapterCtx, winit as vpl_winit};
use viewport_lib::runtime::{GpuFrameContext, ViewportRuntime};
use viewport_lib::{
    ButtonState, Camera, Colour, InputOwnership, Material, OrbitCameraController, ViewportContext,
    ViewportEvent, ViewportInstance, ViewportRenderer, primitives, viewer_bindings, wgpu,
};
use viewport_lib_wind::{WindAuthoring, WindMaterialParams, WindPlugin};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowAttributes, WindowId};

use models::{Model, ModelLoader};

/// A moderate breeze from the west-south-west with gusts that roll across the
/// meadow every few seconds. Displacement is in world units at the top of a
/// plant; the tall grass is about 1.8 units high.
fn breeze() -> WindAuthoring {
    WindAuthoring {
        direction: Vec3::new(1.0, 0.35, 0.0),
        base_strength: 0.3,
        gust_strength: 0.25,
        gust_frequency: 1.6,
        spatial_density: 0.3,
    }
}

/// No wind at all: every plant stands at rest.
fn calm() -> WindAuthoring {
    WindAuthoring {
        base_strength: 0.0,
        gust_strength: 0.0,
        ..breeze()
    }
}

/// A small deterministic random source, so the meadow is the same every run.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 40) as f32 / (1u64 << 24) as f32
    }

    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.next()
    }

    /// A placement on the ground: random spot in the rectangle, random turn
    /// about Z, and a scale between `s0` and `s1`.
    fn placement(&mut self, x: [f32; 2], y: [f32; 2], s0: f32, s1: f32) -> Mat4 {
        let pos = Vec3::new(self.range(x[0], x[1]), self.range(y[0], y[1]), 0.0);
        let turn = Quat::from_rotation_z(self.range(0.0, TAU));
        Mat4::from_scale_rotation_translation(Vec3::splat(self.range(s0, s1)), turn, pos)
    }
}

/// Load the models and scatter them over the meadow.
fn build_meadow(
    viewport: &mut ViewportInstance,
    wind: &WindPlugin,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) {
    let mut loader = ModelLoader {
        dir: concat!(env!("CARGO_MANIFEST_DIR"), "/examples/assets/megakit").into(),
        res: viewport.resources_mut(),
        device,
        queue,
        textures: Vec::new(),
    };
    let sway = |strength| {
        Some((
            wind,
            WindMaterialParams {
                strength,
                ..Default::default()
            },
        ))
    };
    let grass = [
        loader.load("Grass_Common_Tall", sway(1.0)),
        loader.load("Grass_Wispy_Tall", sway(1.0)),
    ];
    let flowers = [
        loader.load("Flower_3_Group", sway(0.9)),
        loader.load("Flower_4_Group", sway(0.9)),
    ];
    let fern = loader.load("Fern_1", sway(0.7));
    let bush = loader.load("Bush_Common_Flowers", sway(0.35));
    let tree = loader.load("CommonTree_1", sway(0.5));
    let rock = loader.load("Rock_Medium_1", None);

    let ground_material = Material::pbr(Colour::srgb_rgb(0.30, 0.42, 0.18), 0.0, 0.95);
    let res = viewport.resources_mut();
    let ground_mesh = res
        .upload_mesh_data(device, &primitives::grid_plane(400.0, 400.0, 1, 1))
        .expect("upload ground");
    let ground_material = res
        .upload_material(ground_material)
        .expect("ground material");

    let scene = viewport.scene_mut();
    scene.add(Some(ground_mesh), Mat4::IDENTITY, ground_material);
    let mut place = |model: &Model, at: Mat4| {
        for (mesh, material, transform) in model.instances(at) {
            scene.add(Some(mesh), transform, material);
        }
    };

    let mut rng = Lcg(0x5eed_1234);
    let (xs, ys) = ([-16.0, 16.0], [-9.0, 11.0]);
    for i in 0..420 {
        let at = rng.placement(xs, ys, 0.7, 1.15);
        place(&grass[i % 2], at);
    }
    for i in 0..60 {
        let at = rng.placement(xs, ys, 0.6, 0.9);
        place(&flowers[i % 2], at);
    }
    for _ in 0..24 {
        let at = rng.placement(xs, ys, 0.7, 1.0);
        place(&fern, at);
    }
    for (x, y) in [(-9.0, 6.0), (8.5, 4.0), (2.0, 9.0)] {
        let turn = Quat::from_rotation_z(rng.range(0.0, TAU));
        place(
            &bush,
            Mat4::from_rotation_translation(turn, Vec3::new(x, y, 0.0)),
        );
    }
    for (x, y, s) in [(-5.0, 13.0, 1.2), (7.0, 15.0, 1.0)] {
        let turn = Quat::from_rotation_z(rng.range(0.0, TAU));
        place(
            &tree,
            Mat4::from_scale_rotation_translation(Vec3::splat(s), turn, Vec3::new(x, y, 0.0)),
        );
    }
    for (x, y) in [(-3.0, -2.0), (11.0, -4.0), (-12.0, 2.0)] {
        let turn = Quat::from_rotation_z(rng.range(0.0, TAU));
        place(
            &rock,
            Mat4::from_rotation_translation(turn, Vec3::new(x, y, 0.0)),
        );
    }
}

struct State {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    viewport: ViewportInstance,
    orbit: OrbitCameraController,
    wind: WindPlugin,
    wind_on: bool,
    last_frame: Instant,
    frame_index: u64,
}

impl State {
    fn new(event_loop: &ActiveEventLoop) -> Self {
        let window = Arc::new(
            event_loop
                .create_window(
                    WindowAttributes::default()
                        .with_title("viewport-lib-wind : meadow")
                        .with_inner_size(winit::dpi::LogicalSize::new(1280u32, 720u32)),
                )
                .expect("window"),
        );
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let surface = instance.create_surface(window.clone()).expect("surface");
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            ..Default::default()
        }))
        .expect("adapter");
        // Per-vertex deformers need the raised limits (more bind groups and
        // storage buffers than the wgpu defaults).
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_features: ViewportRenderer::recommended_device_features(&adapter),
            required_limits: ViewportRenderer::recommended_device_limits(&adapter),
            ..Default::default()
        }))
        .expect("device");

        let size = window.inner_size();
        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let wind = WindPlugin::new(breeze());
        let runtime = ViewportRuntime::new()
            .with_plugin(wind.cpu_plugin())
            .with_gpu_plugin(wind.gpu_plugin());
        let mut viewport = ViewportInstance::new(&device, format)
            .with_bindings(viewer_bindings())
            .with_runtime(runtime);
        wind.install(viewport.resources_mut(), &device)
            .expect("register wind deformer");
        build_meadow(&mut viewport, &wind, &device, &queue);

        let frame = viewport.viewport_frame_mut();
        frame.background_colour = Some(Colour::srgb_rgb(0.62, 0.78, 0.92));
        frame.show_axes_indicator = false;
        let lighting = &mut viewport.effects_mut().lighting;
        lighting.sky_colour = Colour::srgb_rgb(0.80, 0.88, 1.0);
        lighting.ground_colour = Colour::srgb_rgb(0.45, 0.50, 0.35);

        // A low view across the meadow from the south, trees at the back.
        // The orbit eye sits along the orientation's local +Z.
        let pitch = 14f32.to_radians();
        let orientation = Quat::from_rotation_z(-0.35) * Quat::from_rotation_x(FRAC_PI_2 - pitch);
        *viewport.camera_mut() = Camera::orbit(glam::DVec3::new(0.0, 2.0, 1.2), 20.0, orientation);

        let [w, h] = [config.width as f32, config.height as f32];
        viewport.begin_frame(ViewportContext::new(InputOwnership::full_window(), [w, h]));

        Self {
            window,
            surface,
            device,
            queue,
            config,
            viewport,
            orbit: OrbitCameraController::new_stateless(),
            wind,
            wind_on: true,
            last_frame: Instant::now(),
            frame_index: 0,
        }
    }

    fn redraw(&mut self) {
        let frame = match self.surface.get_current_texture() {
            Ok(f) => f,
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                self.surface.configure(&self.device, &self.config);
                self.window.request_redraw();
                return;
            }
            Err(e) => {
                eprintln!("surface error: {e:?}");
                return;
            }
        };
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let [w, h] = [self.config.width as f32, self.config.height as f32];

        let now = Instant::now();
        let dt = (now - self.last_frame).as_secs_f32().min(0.1);
        self.last_frame = now;

        // Advance the wind clock, move the camera, and assemble the frame.
        self.viewport.step_runtime(dt);
        self.viewport.update_orbit(&mut self.orbit, dt);

        // Push this frame's wind into the deformer, then render.
        let camera = self.viewport.camera().clone();
        let ctx = GpuFrameContext::new(&camera, Vec2::new(w, h), dt, self.frame_index);
        let runtime = self.viewport.runtime_mut().expect("runtime attached");
        let plugin_cmds = runtime.pre_prepare(&self.device, &self.queue, &ctx);
        let cmd = self.viewport.render(&self.device, &self.queue, &view);
        self.queue
            .submit(plugin_cmds.into_iter().chain(std::iter::once(cmd)));
        frame.present();
        self.frame_index += 1;

        self.viewport
            .begin_frame(ViewportContext::new(InputOwnership::full_window(), [w, h]));
        self.window.request_redraw();
    }

    fn input(&mut self, event: &WindowEvent) {
        let mut events = Vec::new();
        vpl_winit::translate(event, &AdapterCtx::default(), &mut events);
        for ev in events {
            if let ViewportEvent::Key {
                state: ButtonState::Pressed,
                logical: Some(' '),
                ..
            } = ev
            {
                self.wind_on = !self.wind_on;
                self.wind
                    .set_authoring(if self.wind_on { breeze() } else { calm() });
            }
            self.viewport.handle_event(ev);
        }
    }
}

#[derive(Default)]
struct App {
    state: Option<State>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_none() {
            self.state = Some(State::new(event_loop));
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(state) = self.state.as_mut() else {
            return;
        };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) if size.width > 0 && size.height > 0 => {
                state.config.width = size.width;
                state.config.height = size.height;
                state.surface.configure(&state.device, &state.config);
                let size = [size.width as f32, size.height as f32];
                state.viewport.set_viewport_size(size);
            }
            WindowEvent::RedrawRequested => state.redraw(),
            other => state.input(&other),
        }
    }
}

fn main() {
    let event_loop = EventLoop::new().expect("event loop");
    event_loop.run_app(&mut App::default()).expect("run");
}
