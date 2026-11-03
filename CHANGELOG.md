# Changelog

## [0.2.0]

- Works with viewport-lib 0.23.
- Fixes to orientation to match the canonical z-up behaviour expected in viewport-lib. Height for the sway falloff is measured along Z, and gusts roll across the ground instead of only along X.
- Gusts stay put when the renderer moves its world origin: call `set_world_origin` each frame if you use one.
- New `wind-meadow` example: grass, plants and trees swaying in the wind.

## [0.1.0]

First release, for viewport-lib 0.19.

- Grass, foliage and banners sway in a gusting wind, done in the mesh shaders.
- Choose how much each part of a mesh moves: evenly, or more towards the top.
- Turn the strength, gust size and speed up or down per mesh.
- Read the same wind on the CPU, for anything else that should follow it.
