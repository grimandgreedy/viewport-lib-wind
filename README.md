# viewport-lib-wind

Wind field for `viewport-lib`. A single global wind vector with gust
modulation, mirrored to the GPU as a small uniform buffer and to a CPU
sampler that consumers can use directly.

## Use

```rust
use viewport_lib::runtime::ViewportRuntime;
use viewport_lib_wind::{WindAuthoring, WindPlugin};

let wind = WindPlugin::new(WindAuthoring::default());

let runtime = ViewportRuntime::new()
    .with_plugin(wind.cpu_plugin())
    .with_gpu_plugin(wind.gpu_plugin());

// Sample the field from anywhere on the CPU:
let v = wind.field().sample(glam::Vec3::new(2.0, 0.0, 1.0));
```

The CPU sampler and the WGSL helper (`SHARED_WIND_WGSL`) use the same
formula, so vertex shaders and any CPU consumer (spring bones, particles,
cloth) see the same field at the same world position.

## Shader integration

`SHARED_WIND_WGSL` defines the `viewport_apply_vertex_displacement(world_pos,
sway_mask)` function declared by viewport-lib's displacement contract.
Combine the upstream declaration with this body when building a pipeline
that wants wind-driven vertex sway.

## Status

Early. The plugin is usable for CPU sampling and for hand-rolled pipelines
that bind the wind UBO. Routing through viewport-lib's standard mesh
pipeline requires the upstream vertex-displacement sidecar; that piece is
tracked separately.
