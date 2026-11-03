# Example asset credits

Everything here is CC0, and none of it requires attribution. These notes record where it came from.

## Models (megakit/)

The Stylized Nature MegaKit by Quaternius, standard (free) version (https://quaternius.com/packs/stylizednaturemegakit.html), CC0. Its licence note is kept as `megakit/License_Standard.txt`.

Only the glTF models the meadow example uses are kept, each with its `.bin` buffer:

- `Grass_Common_Tall`, `Grass_Wispy_Tall`
- `Flower_3_Group`, `Flower_4_Group`
- `Fern_1`
- `Bush_Common_Flowers`
- `CommonTree_1`
- `Rock_Medium_1`

Only the base colour textures those models reference are kept: `Grass.png`, `Flowers.png`, `Leaves.png`, `Leaves_NormalTree_C.png`, `Bark_NormalTree.png` and `Rocks_Diffuse.png`. The bark normal map is left out, and the example does not read it. To keep the repository small, `Leaves.png` was downscaled from 2048 to 1024 pixels, and `Bark_NormalTree.png` and `Rocks_Diffuse.png` from 2048 to 512 pixels.
