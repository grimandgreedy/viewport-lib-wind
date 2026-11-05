//! wind-meadow: a small meadow swaying in a gusting wind.
//!
//! Grass, flowers, ferns, bushes and two trees from the Quaternius Stylized
//! Nature MegaKit (CC0) are scattered over a ground plane. Every plant gets a
//! height-falloff sway mask, so its base stays planted while its top moves with
//! the wind. Gusts roll across the meadow as a travelling wave. Trees and bushes
//! take a smaller share of the wind than the grass. The rocks have no mask and
//! stay still.
//!
//! The app runs on `ViewportApp`. The window's setup callback attaches a
//! runtime holding the wind plugin's CPU and GPU halves; the runner steps the
//! CPU half and runs the GPU half before each frame, so the deformer sees this
//! frame's wind.
//!
//! Controls:
//!   Left drag    : orbit
//!   Right drag   : pan
//!   Scroll       : zoom
//!   Space        : turn the wind on and off
//!
//! Run with `cargo run --release --example wind-meadow`.

mod models;

use std::cell::{Cell, RefCell};
use std::f32::consts::{FRAC_PI_2, TAU};
use std::rc::Rc;

use glam::{Mat4, Quat, Vec3};
use viewport_lib::runners::viewport_app::RedrawMode;
use viewport_lib::runtime::ViewportRuntime;
use viewport_lib::{
    AppConfig, ButtonState, Camera, Colour, Material, ViewportApp, ViewportEvent, ViewportInstance,
    WindowConfig, primitives, wgpu,
};
use viewport_lib_wind::{WindAuthoring, WindMaterialParams, WindPlugin};

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

fn main() {
    // The setup callback runs again on a new instance after a device loss, so
    // it builds the wind plugin each time; the frame callback toggles whichever
    // one is current.
    let wind: Rc<RefCell<Option<WindPlugin>>> = Rc::new(RefCell::new(None));
    let wind_on = Rc::new(Cell::new(true));
    let setup_wind = wind.clone();
    let setup_wind_on = wind_on.clone();

    ViewportApp::new(AppConfig::default())
        .window(
            WindowConfig::default()
                .with_title("viewport-lib-wind : meadow")
                .with_window_size(1280, 720)
                .with_redraw_mode(RedrawMode::Continuous),
            move |setup| {
                let device = setup.device().clone();
                let queue = setup.queue().clone();

                let plugin = WindPlugin::new(if setup_wind_on.get() {
                    breeze()
                } else {
                    calm()
                });
                setup.set_runtime(
                    ViewportRuntime::new()
                        .with_plugin(plugin.cpu_plugin())
                        .with_gpu_plugin(plugin.gpu_plugin()),
                );
                plugin
                    .install(setup.resources_mut(), &device)
                    .expect("register wind deformer");
                build_meadow(setup, &plugin, &device, &queue);

                let frame = setup.viewport_frame_mut();
                frame.background_colour = Some(Colour::srgb_rgb(0.62, 0.78, 0.92));
                frame.show_axes_indicator = false;
                let lighting = &mut setup.effects_mut().lighting;
                lighting.sky_colour = Colour::srgb_rgb(0.80, 0.88, 1.0);
                lighting.ground_colour = Colour::srgb_rgb(0.45, 0.50, 0.35);

                // A low view across the meadow from the south, trees at the
                // back. The orbit eye sits along the orientation's local +Z.
                let pitch = 14f32.to_radians();
                let orientation =
                    Quat::from_rotation_z(-0.35) * Quat::from_rotation_x(FRAC_PI_2 - pitch);
                *setup.camera_mut() =
                    Camera::orbit(glam::DVec3::new(0.0, 2.0, 1.2), 20.0, orientation);

                *setup_wind.borrow_mut() = Some(plugin);
            },
            move |ctx| {
                let toggled = ctx.events().iter().any(|event| {
                    matches!(
                        event,
                        ViewportEvent::Key {
                            state: ButtonState::Pressed,
                            logical: Some(' '),
                            ..
                        }
                    )
                });
                if toggled {
                    wind_on.set(!wind_on.get());
                    if let Some(plugin) = wind.borrow().as_ref() {
                        plugin.set_authoring(if wind_on.get() { breeze() } else { calm() });
                    }
                }
            },
        )
        .run();
}
