//! Wind sway for viewport-lib's sprites and tile layers, behind the `2d`
//! feature.
//!
//! [`WindPlugin::install_2d`] registers [`WIND_SPRITE_DEFORMER_BODY`] as a
//! sprite and tile deformer that reads the mesh deformer's slot params, so the
//! GPU half's per-frame write drives meshes, sprites and tiles together. A
//! batch or layer sways once its `deformers` set holds the wind deformer,
//! from [`WindPlugin::sprite_deformers`].
//!
//! Each quad bends from its base: the wind is sampled at the midpoint of the
//! quad's bottom edge, projected onto the quad's right axis, and added to each
//! corner in proportion to its height across the quad. The bottom corners stay
//! put and the offset stays in the item's plane, on XY and XZ alike.
//!
//! How far each quad sways comes from its user value, encoded by
//! [`WindSway2d`]: per sprite through `SpriteItem::user_values` (or a ranged
//! write of `Sprite::user`), and per tile through `Tileset::user_values`,
//! keyed by tile index. A user value of zero does not sway, so a sprite or
//! tile with none stays still in a batch or layer that opts in.

use viewport_lib::DeformerId;
use viewport_lib::error::ViewportError;
use viewport_lib::renderer::ViewportRenderer;
use viewport_lib_plugins::item_types::sprite::{
    SpriteDeformerDesc, SpriteDeformerId, SpriteDeformerSet, SpriteDeformers,
};

use crate::plugin::WindPlugin;

/// Name the sprite and tile deformer registers under.
pub const WIND_SPRITE_DEFORMER_NAME: &str = "viewport_lib_wind_2d";

/// WGSL body for the sprite and tile deformer, registered by
/// [`WindPlugin::install_2d`].
///
/// It reads the same slot params as [`crate::WIND_DEFORMER_BODY`] and uses the
/// same phase and magnitude, so the offset at a quad's top edge is
/// `dot(WindField::sample(base), right) * weight` along `right`, with `base`
/// the midpoint of the quad's bottom edge in world space and `weight` from
/// the quad's [`WindSway2d`] user value. In pixel-art mode the offset is
/// rounded to whole texels of the quad's texture along `right`. The mesh path's
/// strength, density and speed multipliers have no sprite equivalent and are
/// 1.0 here.
pub const WIND_SPRITE_DEFORMER_BODY: &str = r#"
fn deform_sprite(v: SpriteVertex, slot: u32) -> vec3<f32> {
    // The weight is the low half of the user value, as a 16-bit float.
    let weight = unpack2x16float(v.user).x;
    if weight == 0.0 {
        return v.world;
    }
    let p0 = deform_header.slot_params[slot * 4u + 0u];
    let p1 = deform_header.slot_params[slot * 4u + 1u];
    let dir = vec3<f32>(p0.x, p0.y, p0.z);
    let dlen_sq = dot(dir, dir);
    if dlen_sq < 1e-8 {
        return v.world;
    }
    let ndir = dir * inverseSqrt(dlen_sq);
    let base = p0.w;
    let gust = p1.x;
    let freq = p1.y;
    let density = p1.z;
    let time = p1.w;
    // Sprite positions are relative to the renderer's world origin, so the
    // phase adds it back to stay keyed on the absolute position.
    let origin = deform_header.slot_params[slot * 4u + 2u].xyz;
    let world_base = v.base + origin;
    let phase = time * freq + (world_base.x + world_base.y) * density;
    let wind = ndir * (base + sin(phase) * gust);
    var shift = dot(wind, v.right) * weight * v.local_uv.y;
    // Pixel-art mode: whole texels only, so the quad shears in steps.
    if (v.user & 0x10000u) != 0u && v.texel.x > 0.0 {
        shift = round(shift / v.texel.x) * v.texel.x;
    }
    return v.world + v.right * shift;
}
"#;

/// How far one sprite or tile sways, packed into the `u32` user value the
/// sprite and tile deformer reads.
///
/// The low 16 bits hold `weight` as a 16-bit float and bit 16 is
/// `pixel_art`; the rest are reserved and zero. The user value is shared by
/// every vertex hook on a batch, so a host whose own material reads it has to
/// leave the low 17 bits to the wind.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct WindSway2d {
    /// Multiplier on the offset at the quad's top edge. `0.0` does not sway,
    /// `1.0` sways by the full field. Stored at 16-bit float precision.
    pub weight: f32,
    /// Round the offset to whole texels of the quad's texture, so pixel art
    /// shears in one-texel steps instead of smearing. Assumes the simple draw
    /// mode: sliced and tiled sprites have no single texel size. An
    /// untextured quad is not rounded.
    pub pixel_art: bool,
}

impl WindSway2d {
    /// Sways by `weight`.
    pub fn new(weight: f32) -> Self {
        Self {
            weight,
            pixel_art: false,
        }
    }

    /// This sway in pixel-art mode.
    pub fn with_pixel_art(mut self) -> Self {
        self.pixel_art = true;
        self
    }

    /// The user value for a sprite or tile.
    pub fn to_user_value(self) -> u32 {
        f16_bits(self.weight) as u32 | (self.pixel_art as u32) << PIXEL_ART_BIT
    }

    /// The sway a user value encodes. The reverse of [`Self::to_user_value`]
    /// up to 16-bit float precision.
    pub fn from_user_value(value: u32) -> Self {
        Self {
            weight: f16_to_f32(value as u16),
            pixel_art: value & (1 << PIXEL_ART_BIT) != 0,
        }
    }

    /// `count` copies of this sway's user value, for `SpriteItem::user_values`
    /// when every sprite in a batch sways alike.
    pub fn user_values(self, count: usize) -> Vec<u32> {
        vec![self.to_user_value(); count]
    }

    /// A `Tileset::user_values` table from `(tile index, sway)` pairs, so a
    /// tile sways wherever it is painted. Tiles not named read zero and stay
    /// still. A later pair for the same tile replaces an earlier one.
    pub fn tile_values(swaying: &[(u32, WindSway2d)]) -> Vec<u32> {
        let len = swaying
            .iter()
            .map(|&(t, _)| t as usize + 1)
            .max()
            .unwrap_or(0);
        let mut out = vec![0; len];
        for &(tile, sway) in swaying {
            out[tile as usize] = sway.to_user_value();
        }
        out
    }
}

/// The user-value bit that turns on pixel-art mode.
const PIXEL_ART_BIT: u32 = 16;

/// `x` as IEEE 754 half-precision bits, rounded to nearest even, matching
/// WGSL's `pack2x16float` for finite values.
fn f16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let man = b & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if man != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        // Subnormal or zero.
        if e < -10 {
            return sign;
        }
        let m = man | 0x80_0000;
        let shift = (14 - e) as u32;
        let mut h = m >> shift;
        let rem = m & ((1 << shift) - 1);
        let halfway = 1 << (shift - 1);
        if rem > halfway || (rem == halfway && h & 1 == 1) {
            h += 1;
        }
        return sign | h as u16;
    }
    let mut h = ((e as u32) << 10) | (man >> 13);
    let rem = man & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && h & 1 == 1) {
        // A carry out of the mantissa steps the exponent, as it should.
        h += 1;
    }
    sign | h as u16
}

/// Half-precision bits as an `f32`.
fn f16_to_f32(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let e = ((h >> 10) & 0x1f) as i32;
    let m = (h & 0x3ff) as f32;
    match e {
        0 => sign * m * 2f32.powi(-24),
        0x1f if m == 0.0 => sign * f32::INFINITY,
        0x1f => f32::NAN,
        _ => sign * (1.0 + m / 1024.0) * 2f32.powi(e - 15),
    }
}

impl WindPlugin {
    /// Register the wind's sprite and tile deformer with `renderer`.
    ///
    /// Call once, after [`Self::install`] and after the sprite and tilemap
    /// item types are installed on `renderer`: a tilemap type installed later
    /// does not see the deformer. The deformer reads the mesh deformer's slot
    /// params, so it needs no GPU half of its own.
    ///
    /// # Errors
    ///
    /// [`ViewportError::PluginInstallMissing`] when [`Self::install`] has not
    /// run, and any error from `register_sprite_deformer`.
    pub fn install_2d(
        &self,
        renderer: &mut ViewportRenderer,
        device: &wgpu::Device,
    ) -> Result<SpriteDeformerId, ViewportError> {
        let mesh: DeformerId = self
            .inner
            .lock()
            .expect("wind plugin poisoned")
            .deformer
            .ok_or(ViewportError::PluginInstallMissing {
                needed: "the mesh deformer from WindPlugin::install",
            })?;
        let desc = SpriteDeformerDesc::new(WIND_SPRITE_DEFORMER_NAME, WIND_SPRITE_DEFORMER_BODY)
            .with_params(mesh);
        let id = renderer.register_sprite_deformer(device, &desc)?;
        self.inner
            .lock()
            .expect("wind plugin poisoned")
            .sprite_deformer = Some(id);
        Ok(id)
    }

    /// The deformer set holding the wind deformer, for a batch's or layer's
    /// `deformers` field. [`SpriteDeformerSet::NONE`] before
    /// [`Self::install_2d`]. Combine with other deformers through
    /// [`SpriteDeformerSet::with`].
    pub fn sprite_deformers(&self) -> SpriteDeformerSet {
        match self
            .inner
            .lock()
            .expect("wind plugin poisoned")
            .sprite_deformer
        {
            Some(id) => SpriteDeformerSet::NONE.with(id),
            None => SpriteDeformerSet::NONE,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::WindAuthoring;
    use crate::plugin::tests::try_make_device;

    fn renderer(device: &wgpu::Device) -> ViewportRenderer {
        let config = viewport_lib::RendererConfig::new(wgpu::TextureFormat::Rgba8UnormSrgb)
            .with_pipeline_compilation(viewport_lib::PipelineCompilation::Blocking);
        let mut renderer = ViewportRenderer::with_config(device, &config);
        viewport_lib_plugins::item_types::install(&mut renderer, device);
        renderer
    }

    #[test]
    fn sway_round_trips_through_the_user_value() {
        for w in [0.0, 1.0, 0.5, 0.25, 2.0, -1.0] {
            let v = WindSway2d::new(w).to_user_value();
            assert_eq!(WindSway2d::from_user_value(v).weight, w);
            assert_eq!(v >> 16, 0);
        }
        let w = WindSway2d::from_user_value(WindSway2d::new(0.3).to_user_value()).weight;
        assert!((w - 0.3).abs() < 1e-3);
    }

    #[test]
    fn pixel_art_is_bit_16() {
        let v = WindSway2d::new(1.0).with_pixel_art().to_user_value();
        assert_eq!(v, 0x1_3c00);
        let back = WindSway2d::from_user_value(v);
        assert!(back.pixel_art);
        assert_eq!(back.weight, 1.0);
        assert!(!WindSway2d::from_user_value(0x3c00).pixel_art);
    }

    #[test]
    fn zero_sway_is_a_zero_user_value() {
        assert_eq!(WindSway2d::new(0.0).to_user_value(), 0);
        assert_eq!(WindSway2d::default().to_user_value(), 0);
    }

    #[test]
    fn half_bits_match_known_values() {
        assert_eq!(f16_bits(1.0), 0x3c00);
        assert_eq!(f16_bits(0.5), 0x3800);
        assert_eq!(f16_bits(-2.0), 0xc000);
        assert_eq!(f16_bits(65504.0), 0x7bff);
        assert_eq!(f16_bits(1e6), 0x7c00);
        assert_eq!(f16_bits(2f32.powi(-24)), 0x0001);
        assert_eq!(f16_to_f32(0x0001), 2f32.powi(-24));
    }

    #[test]
    fn tile_values_set_only_the_named_tiles() {
        let t = WindSway2d::tile_values(&[(3, WindSway2d::new(1.0)), (1, WindSway2d::new(0.5))]);
        assert_eq!(t.len(), 4);
        assert_eq!(t[0], 0);
        assert_eq!(t[2], 0);
        assert_eq!(WindSway2d::from_user_value(t[1]).weight, 0.5);
        assert_eq!(WindSway2d::from_user_value(t[3]).weight, 1.0);
        assert!(WindSway2d::tile_values(&[]).is_empty());
    }

    #[test]
    fn install_2d_registers_against_the_sprite_and_tile_shaders() {
        let Some((device, _queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut renderer = renderer(&device);
        let wind = WindPlugin::new(WindAuthoring::default());
        assert!(wind.sprite_deformers().is_empty());
        wind.install(renderer.resources_mut(), &device)
            .expect("install");
        let id = wind.install_2d(&mut renderer, &device).expect("install_2d");
        assert_eq!(
            renderer.sprite_deformer_id(WIND_SPRITE_DEFORMER_NAME),
            Some(id)
        );
        assert!(wind.sprite_deformers().contains(id));
    }

    #[test]
    fn install_2d_needs_install_first() {
        let Some((device, _queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut renderer = renderer(&device);
        let wind = WindPlugin::new(WindAuthoring::default());
        assert!(matches!(
            wind.install_2d(&mut renderer, &device),
            Err(ViewportError::PluginInstallMissing { .. })
        ));
        assert!(wind.sprite_deformers().is_empty());
    }
}

/// Renders sprites and tiles through the wind deformer and reads back where
/// their edges land. One world unit is one pixel, with the world origin in the
/// middle of the view.
#[cfg(test)]
mod render_tests {
    use super::*;
    use crate::field::WindAuthoring;
    use crate::plugin::tests::try_make_device;
    use glam::{Vec2, Vec3};
    use viewport_lib::plugin_api::Uploads;
    use viewport_lib::{
        Camera2d, CameraFrame, Colour, FrameData, Plane2d, SceneFrame, TextureData,
    };
    use viewport_lib_plugins::item_types::sprite::{SpriteItem, SpriteSizeMode};
    use viewport_lib_plugins::item_types::tilemap::{Tile, Tilemap, TilemapItem, Tileset};

    const W: u32 = 240;
    const H: u32 = 120;
    const RED: Colour = Colour::linear(1.0, 0.0, 0.0, 1.0);
    /// Side of every test sprite and tile, in world units.
    const SIDE: f32 = 40.0;

    struct Rig {
        device: wgpu::Device,
        queue: wgpu::Queue,
        renderer: ViewportRenderer,
        wind: WindPlugin,
    }

    /// A renderer with the wind installed for meshes and sprites, the field
    /// set to `authoring` and advanced to time `t`. `None` without a device.
    fn rig(authoring: WindAuthoring, t: f32) -> Option<Rig> {
        let Some((device, queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return None;
        };
        let config = viewport_lib::RendererConfig::new(wgpu::TextureFormat::Rgba8UnormSrgb)
            .with_pipeline_compilation(viewport_lib::PipelineCompilation::Blocking);
        let mut renderer = ViewportRenderer::with_config(&device, &config);
        viewport_lib_plugins::item_types::install(&mut renderer, &device);
        let wind = WindPlugin::new(authoring);
        wind.install(renderer.resources_mut(), &device)
            .expect("install");
        wind.install_2d(&mut renderer, &device).expect("install_2d");
        wind.step_and_write(&queue, t);
        Some(Rig {
            device,
            queue,
            renderer,
            wind,
        })
    }

    fn steady(strength: f32) -> WindAuthoring {
        WindAuthoring {
            direction: Vec3::X,
            base_strength: strength,
            gust_strength: 0.0,
            gust_frequency: 0.0,
            spatial_density: 0.0,
        }
    }

    impl Rig {
        fn render(&mut self, plane: Plane2d, scene: SceneFrame) -> Vec<u8> {
            let camera = Camera2d::new(plane)
                .with_units_per_pixel(1.0)
                .with_viewport_size(Vec2::new(W as f32, H as f32))
                .camera();
            let frame = FrameData::new(
                CameraFrame::from_camera(&camera, [W as f32, H as f32]),
                scene,
            );
            // The default world origin follows the camera, and sprite
            // positions are relative to it. Pass it to the wind the way a host
            // does each frame, then draw again with it in the slot params.
            self.renderer
                .render_offscreen(&self.device, &self.queue, &frame, W, H);
            self.wind.set_world_origin(self.renderer.world_origin());
            self.wind.step_and_write(&self.queue, 0.0);
            self.renderer
                .render_offscreen(&self.device, &self.queue, &frame, W, H)
        }

        fn sprites(&mut self, plane: Plane2d, item: SpriteItem) -> Vec<u8> {
            let mut scene = SceneFrame::default();
            scene.items_mut::<SpriteItem>().push(item);
            self.render(plane, scene)
        }

        /// Red world-space squares of side `SIDE` centred at `positions`,
        /// swaying with the wind by `sways`.
        fn squares(&self, positions: Vec<[f32; 3]>, sways: &[WindSway2d]) -> SpriteItem {
            let mut item = SpriteItem::default();
            item.positions = positions;
            item.default_size = SIDE;
            item.default_colour = RED;
            item.size_mode = SpriteSizeMode::WorldSpace;
            item.user_values = sways.iter().map(|s| s.to_user_value()).collect();
            item.deformers = self.wind.sprite_deformers();
            item
        }
    }

    fn red_at(pixels: &[u8], col: usize, row: usize) -> bool {
        let i = (row * W as usize + col) * 4;
        pixels[i] > 180 && pixels[i + 1] < 80 && pixels[i + 2] < 80
    }

    fn red(pixels: &[u8], x: f32, y: f32) -> bool {
        let col = (W as f32 * 0.5 + x).floor() as usize;
        let row = (H as f32 * 0.5 - y).floor() as usize;
        red_at(pixels, col, row)
    }

    /// The world x of the first red pixel centre at or right of `from` on the
    /// pixel row whose centre is at height `y`.
    fn left_edge(pixels: &[u8], from: f32, y: f32) -> f32 {
        let row = (H as f32 * 0.5 - y).floor() as usize;
        let start = (W as f32 * 0.5 + from).floor().max(0.0) as usize;
        let col = (start..W as usize)
            .find(|&c| red_at(pixels, c, row))
            .expect("a red pixel on the row");
        col as f32 + 0.5 - W as f32 * 0.5
    }

    /// The height of the topmost and bottommost pixel row centres inside a
    /// square centred at height 0.
    const TOP_ROW: f32 = SIDE * 0.5 - 0.5;
    const BOTTOM_ROW: f32 = -SIDE * 0.5 + 0.5;

    /// The left edge of a square centred at `x`, displaced at its top by
    /// `shift`, at height `y`. A red pixel centre lies at or right of it.
    fn expected_left(x: f32, shift: f32, y: f32) -> f32 {
        x - SIDE * 0.5 + shift * (y + SIDE * 0.5) / SIDE
    }

    fn assert_left_edge(pixels: &[u8], x: f32, shift: f32, y: f32) {
        let want = expected_left(x, shift, y);
        let got = left_edge(pixels, want - 6.0, y);
        assert!(
            got >= want - 0.25 && got <= want + 1.25,
            "square at {x}, row {y}: left edge pixel {got}, edge expected at {want}"
        );
    }

    /// Two squares on `plane`, centred at x = -60 and 60 at height 0, sway by
    /// `dot(WindField::sample(base), right)` at their top edge, with right
    /// world +X on both planes.
    fn top_edge_follows_the_field(plane: Plane2d, at: fn(f32, f32) -> Vec3) {
        let authoring = WindAuthoring {
            direction: Vec3::new(1.0, 0.3, 0.0),
            base_strength: 4.0,
            gust_strength: 8.0,
            gust_frequency: 1.3,
            spatial_density: 0.02,
        };
        let Some(mut rig) = rig(authoring, 0.7) else {
            return;
        };
        let field = rig.wind.field();
        let xs = [-60.0, 60.0];
        let item = rig.squares(
            xs.iter().map(|&x| at(x, 0.0).to_array()).collect(),
            &[WindSway2d::new(1.0); 2],
        );
        let pixels = rig.sprites(plane, item);
        let shifts: Vec<f32> = xs
            .iter()
            .map(|&x| field.sample(at(x, -SIDE * 0.5)).dot(Vec3::X))
            .collect();
        assert!(
            (shifts[0] - shifts[1]).abs() > 2.0,
            "the two squares should sit at different gust phases: {shifts:?}"
        );
        for (&x, &shift) in xs.iter().zip(&shifts) {
            assert_left_edge(&pixels, x, shift, TOP_ROW);
            assert_left_edge(&pixels, x, shift, BOTTOM_ROW);
        }
    }

    #[test]
    fn the_top_edge_follows_the_field_on_xy() {
        top_edge_follows_the_field(Plane2d::Xy, |x, h| Vec3::new(x, h, 0.0));
    }

    #[test]
    fn the_top_edge_follows_the_field_on_xz() {
        top_edge_follows_the_field(Plane2d::Xz, |x, h| Vec3::new(x, 0.0, h));
    }

    #[test]
    fn zero_weight_and_zero_wind_leave_the_quad_still() {
        let Some(mut rig) = rig(steady(10.0), 0.0) else {
            return;
        };
        // The second square has no user value at all.
        let item = rig.squares(
            vec![[-60.0, 0.0, 0.0], [60.0, 0.0, 0.0]],
            &[WindSway2d::new(0.0)],
        );
        let pixels = rig.sprites(Plane2d::Xy, item);
        assert_left_edge(&pixels, -60.0, 0.0, TOP_ROW);
        assert_left_edge(&pixels, 60.0, 0.0, TOP_ROW);

        rig.wind.set_authoring(steady(0.0));
        rig.wind.step_and_write(&rig.queue, 0.0);
        let item = rig.squares(vec![[0.0, 0.0, 0.0]], &[WindSway2d::new(1.0)]);
        let pixels = rig.sprites(Plane2d::Xy, item);
        assert_left_edge(&pixels, 0.0, 0.0, TOP_ROW);
    }

    #[test]
    fn only_the_tiles_in_the_table_sway() {
        let Some(mut rig) = rig(steady(10.0), 0.0) else {
            return;
        };
        let mut tileset = Tileset::new(None, [8, 8], 1);
        tileset.user_values = WindSway2d::tile_values(&[(0, WindSway2d::new(1.0))]);
        let mut map = Tilemap::new(tileset, [2, 1]);
        map.set(0, 0, Tile::new(0));
        map.set(1, 0, Tile::new(1));
        let map_id = rig
            .renderer
            .upload(&rig.device, &rig.queue, &map)
            .expect("upload");
        let mut layer = TilemapItem::new(map_id);
        layer.origin = [-SIDE, -SIDE * 0.5];
        layer.cell_size = [SIDE, SIDE];
        layer.tint = RED;
        layer.deformers = rig.wind.sprite_deformers();
        let mut scene = SceneFrame::default();
        scene.items_mut::<TilemapItem>().push(layer);
        let pixels = rig.render(Plane2d::Xy, scene);
        // Tile 0 spans -40..0 and leans 10 right at its top; tile 1 spans
        // 0..40 and stands still.
        assert_left_edge(&pixels, -SIDE * 0.5, 10.0, TOP_ROW);
        assert_left_edge(&pixels, -SIDE * 0.5, 10.0, BOTTOM_ROW);
        assert!(
            red(&pixels, SIDE - 0.5, TOP_ROW),
            "tile 1 keeps its top right"
        );
        assert!(!red(&pixels, SIDE + 0.5, TOP_ROW), "and does not lean");
    }

    #[test]
    fn pixel_art_steps_by_whole_texels() {
        let Some(mut rig) = rig(steady(7.0), 0.0) else {
            return;
        };
        // An 8 texel texture over 40 units: one texel is 5 units, so a 7 unit
        // offset rounds to 5.
        let texture = rig
            .renderer
            .resources_mut()
            .upload_texture(
                &rig.device,
                &rig.queue,
                TextureData::srgb(8, 8, vec![255; 8 * 8 * 4]),
            )
            .expect("texture upload");
        let mut item = rig.squares(
            vec![[-60.0, 0.0, 0.0], [60.0, 0.0, 0.0]],
            &[WindSway2d::new(1.0).with_pixel_art(), WindSway2d::new(1.0)],
        );
        item.texture_id = Some(texture);
        let pixels = rig.sprites(Plane2d::Xy, item);
        assert_left_edge(&pixels, -60.0, 5.0, TOP_ROW);
        assert_left_edge(&pixels, 60.0, 7.0, TOP_ROW);
    }
}
