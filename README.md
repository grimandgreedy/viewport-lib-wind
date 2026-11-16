# viewport-lib-wind

Wind sway for [viewport-lib](https://github.com/grimandgreedy/viewport-lib) meshes: grass, foliage and banners ripple in a gusting breeze, driven by per-vertex displacement in the mesh shaders rather than a physics sim. The same field is sampled on the CPU, so spring bones, particles or cloth can follow the wind as well.

With the `2d` feature, sprites and tile layers sway in the same field: each quad bends from its base, with a weight per sprite or per tile and an optional pixel-art mode that shears in whole texels.

<p align="center">
  <img src="https://raw.githubusercontent.com/grimandgreedy/viewport-lib-wind/refs/heads/master/assets/wind.gif" alt="viewport-lib-wind" width="50%" />
</p>

## Meshes

Register the wind deformer, add the CPU and GPU halves to the runtime, then give each mesh that should sway a per-vertex mask:

```rust
use viewport_lib::runtime::ViewportRuntime;
use viewport_lib_wind::{WindAuthoring, WindPlugin, WindSwayWeights};

let wind = WindPlugin::new(WindAuthoring::default());
wind.install(renderer.resources_mut(), &device)?;

let runtime = ViewportRuntime::new()
    .with_plugin(wind.cpu_plugin())
    .with_gpu_plugin(wind.gpu_plugin());

// Still at z = 0, full sway at the tip.
let mask = WindSwayWeights::height_falloff(&positions, 0.0, top_z);
wind.attach_sway_mask(renderer.resources_mut(), &device, mesh_id, &mask);
```

`WindSwayWeights::uniform` sways every vertex by the same amount, for flags and banners. `wind.field().sample(position)` returns the same wind vector on the CPU.

## Sprites and tiles

Enable the `2d` feature. Register the sprite and tile deformer after viewport-lib's item types, then opt batches and layers in and give each sprite or tile a `WindSway2d`:

```rust
use viewport_lib_wind::{WindAuthoring, WindPlugin, WindSway2d};

let wind = WindPlugin::new(WindAuthoring::default());
wind.install(renderer.resources_mut(), &device)?;
viewport_lib_plugins::item_types::install(&mut renderer, &device);
wind.install_2d(&mut renderer, &device)?;

// A batch of grass sprites, all swaying fully.
grass.deformers = wind.sprite_deformers();
grass.user_values = WindSway2d::new(1.0).user_values(grass.positions.len());

// A tile layer where only the grass tile sways.
tileset.user_values = WindSway2d::tile_values(&[(GRASS_TILE, WindSway2d::new(0.8))]);
layer.deformers = wind.sprite_deformers();

// Each frame, so gusts stay put when the renderer's world origin moves.
wind.set_world_origin(renderer.world_origin());
```

A sprite or tile with no user value stays still. `WindSway2d::new(w).with_pixel_art()` rounds the offset to whole texels as the art is drawn.

## Examples

- **wind-meadow**: a meadow of grass, plants and trees swaying in the wind.
- **wind-minimal**: no window; samples the field on the CPU.
- **wind-2d**: a side view of grass, reeds and tiles swaying; Space toggles the wind, P toggles pixel-art mode.
- **wind-2d-minimal**: no window; prints the 2D offsets for a row of sprite bases.

```sh
cargo run --release --example wind-meadow
cargo run --release --example wind-2d --features 2d
```

## Versions

| viewport-lib-wind | viewport-lib |
| --- | --- |
| 0.3.0 | 0.24 |
| 0.2.0 | 0.23 |
| 0.1.0 | 0.19 |

## License

GPL-3.0-only. See [LICENSE](LICENSE).
