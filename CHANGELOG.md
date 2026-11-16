# Changelog

## [0.3.0]

- **viewport-lib 0.24** - Builds against viewport-lib 0.24.
- **Sprite and tile sway** - With the `2d` feature, sprites and tile layers sway, weighted per sprite and per tile, with a pixel-art mode that steps by the drawn texel on sliced and tiled sprites. See the `wind-2d` examples.

## [0.2.0]

- **viewport-lib 0.23** - Builds against viewport-lib 0.23.
- **Z-up sway** - `WindSwayWeights::height_falloff` measures height along Z, and gusts travel across the XY ground plane instead of along X and Z.
- **World origin** - Gusts stay put when the renderer moves its world origin: call `set_world_origin` each frame if you use one.
- **wind-meadow example** - Grass, plants and trees swaying in the wind.

## [0.1.0]

First release, for viewport-lib 0.19.

- **Mesh sway** - Grass, foliage and banners sway in a gusting wind, applied in the mesh shaders.
- **Sway masks** - `WindSwayWeights` sets how much each vertex moves: evenly, or more towards the top.
- **Per-mesh settings** - `WindMaterialParams` scales strength, gust size and speed per mesh.
- **CPU sampling** - `WindField::sample` returns the same wind on the CPU, for anything else that should follow it.
