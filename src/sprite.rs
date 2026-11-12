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
/// `dot(WindField::sample(base), right)` along `right`, with `base` the
/// midpoint of the quad's bottom edge in world space. The mesh path's
/// strength, density and speed multipliers have no sprite equivalent and are
/// 1.0 here.
pub const WIND_SPRITE_DEFORMER_BODY: &str = r#"
fn deform_sprite(v: SpriteVertex, slot: u32) -> vec3<f32> {
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
    return v.world + v.right * (dot(wind, v.right) * v.local_uv.y);
}
"#;

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
