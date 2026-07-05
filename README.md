# viewport-lib-wind

Smooth, cyclical wind-like motion for meshes in `viewport-lib`. Geometry sways and ripples the way grass, foliage, cloth, and banners move in a breeze, driven by repeating vertex displacement rather than a physics sim. Handy for animation, rendering, and games.

Under the hood it is a single global wind vector with gust modulation, mirrored to the GPU as a small uniform buffer and to a CPU sampler that consumers can use directly.

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

The CPU sampler and the WGSL helper (`SHARED_WIND_WGSL`) use the same formula, so vertex shaders and any CPU consumer (spring bones, particles, cloth) see the same field at the same world position.


## License

`viewport-lib-wind` is a part of the viewport-lib ecosystem.

GPL-3.0
