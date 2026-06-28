//! Spin up a runtime with the wind plugin and sample the field on the CPU.
//!
//! This example does not draw anything. It exists to confirm the field is
//! advancing and the helper formula round-trips through both sides.

use glam::Vec3;
use viewport_lib::runtime::ViewportRuntime;
use viewport_lib_wind::{WindAuthoring, WindPlugin};

fn main() {
    let wind = WindPlugin::new(WindAuthoring {
        direction: Vec3::new(1.0, 0.0, 0.0),
        base_strength: 1.2,
        gust_strength: 0.4,
        gust_frequency: 0.6,
        spatial_density: 0.3,
    });

    let _runtime = ViewportRuntime::new()
        .with_plugin(wind.cpu_plugin())
        .with_gpu_plugin(wind.gpu_plugin());

    let probe = Vec3::new(2.5, 0.0, 1.0);
    let field = wind.field();
    let sample = field.sample(probe);

    println!("authoring direction: {:?}", field.authoring().direction);
    println!("wind at {probe:?}: {sample:?}");
    println!("clock: {} s", field.time());
}
