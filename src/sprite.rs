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
/// the quad's [`WindSway2d`] user value. The mesh path's
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
    return v.world + v.right * (dot(wind, v.right) * weight * v.local_uv.y);
}
"#;

/// How far one sprite or tile sways, packed into the `u32` user value the
/// sprite and tile deformer reads.
///
/// The low 16 bits hold `weight` as a 16-bit float; the rest are reserved
/// and zero. The user value is shared by every vertex hook on a batch, so a
/// host whose own material reads it has to leave the low 16 bits to the wind.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct WindSway2d {
    /// Multiplier on the offset at the quad's top edge. `0.0` does not sway,
    /// `1.0` sways by the full field. Stored at 16-bit float precision.
    pub weight: f32,
}

impl WindSway2d {
    /// Sways by `weight`.
    pub fn new(weight: f32) -> Self {
        Self { weight }
    }

    /// The user value for a sprite or tile.
    pub fn to_user_value(self) -> u32 {
        f16_bits(self.weight) as u32
    }

    /// The sway a user value encodes. The reverse of [`Self::to_user_value`]
    /// up to 16-bit float precision.
    pub fn from_user_value(value: u32) -> Self {
        Self {
            weight: f16_to_f32(value as u16),
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
