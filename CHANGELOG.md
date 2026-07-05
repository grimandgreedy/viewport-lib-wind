# Changelog

## [0.1.0]

First release. A single global wind field for viewport-lib:

- `WindPlugin` with a CPU runtime plugin and a GPU plugin.
- Global wind vector with gust modulation, mirrored to a GPU uniform buffer.
- CPU sampler (`field().sample(pos)`) matching the WGSL helper so shaders and
  CPU consumers see the same field at the same world position.
- `SHARED_WIND_WGSL` implementing viewport-lib's vertex-displacement contract.
