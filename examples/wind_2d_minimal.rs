//! Step a runtime with the wind plugin and print how far a row of 2D sprites
//! would sway.
//!
//! This example does not draw anything. The sprite and tile deformer bends each
//! quad from the midpoint of its bottom edge: the top edge moves along the
//! quad's right axis by the wind sampled there, times the sprite's weight. On a
//! side view (the XZ plane, Z up) the right axis is world +X, so the offset is
//! the x component of `WindField::sample` at the base. Neighbouring sprites sit
//! at different gust phases, so they sway a moment apart.
//!
//! Run with `cargo run --example wind-2d-minimal --features 2d`.

use glam::Vec3;
use viewport_lib::interaction::select::selection::Selection;
use viewport_lib::runtime::{RuntimeFrameContext, ViewportRuntime};
use viewport_lib::scene::scene::Scene;
use viewport_lib_wind::{WindAuthoring, WindPlugin, WindSway2d};

fn main() {
    let wind = WindPlugin::new(WindAuthoring {
        direction: Vec3::X,
        base_strength: 3.0,
        gust_strength: 3.0,
        gust_frequency: 1.4,
        spatial_density: 0.02,
    });
    let mut runtime = ViewportRuntime::new().with_plugin(wind.cpu_plugin());
    let mut scene = Scene::new();
    let mut selection = Selection::new();
    let mut frame = RuntimeFrameContext::default();
    frame.dt = 0.25;

    // Sprite bases along the ground, and how far each one sways.
    let bases: Vec<Vec3> = (0..6)
        .map(|i| Vec3::new(i as f32 * 24.0, 0.0, 0.0))
        .collect();
    let sways = [1.0, 1.0, 0.5, 0.0, 1.0, 1.4].map(WindSway2d::new);

    for _ in 0..4 {
        runtime.step(&mut scene, &mut selection, &frame);
        let field = wind.field();
        let offsets: Vec<String> = bases
            .iter()
            .zip(&sways)
            .map(|(base, sway)| format!("{:6.2}", field.sample(*base).x * sway.weight))
            .collect();
        println!(
            "t = {:4.2} s  top-edge offsets: {}",
            field.time(),
            offsets.join(" ")
        );
    }

    // What a host writes into `SpriteItem::user_values` for those sprites.
    let values: Vec<String> = sways
        .iter()
        .map(|s| format!("{:#06x}", s.to_user_value()))
        .collect();
    println!("user values: {}", values.join(" "));
}
