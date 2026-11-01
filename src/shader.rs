//! WGSL deformer body that implements the wind sway formula against the
//! viewport-lib deformer registry.
//!
//! The body reads three inputs:
//!
//! - Per-vertex sway mask from the per-mesh deformer slot data
//!   (4 bytes / one `f32` per vertex), attached via
//!   [`crate::WindPlugin::attach_sway_mask`].
//! - Optional per-mesh material multipliers (strength, density, speed) from
//!   the per-mesh-instance deformer slot data, attached via
//!   [`crate::WindPlugin::attach_material_params`]. Defaults to 1.0 when
//!   absent.
//! - Global wind state from
//!   `deform_header.slot_params[ctx.slot * 4 .. + 4]`, written each frame
//!   by [`crate::WindPlugin`]'s GPU half. Word 2 carries the renderer's world
//!   origin: world-space deformer positions are relative to it, so the gust
//!   phase adds it back to stay put when the origin moves.
//!
//! The formula matches [`crate::WindField::sample`] modulo the per-material
//! multipliers, so a vertex drawn through the deformer and a CPU spring
//! bone tracking the same world position see the same wind to within float
//! precision when the material defaults are in play.

/// Catalogue version for [`WIND_DEFORMER_BODY`]. Bump on any formula or
/// slot-params layout change.
pub const WGSL_VERSION: u32 = 4;

/// Name the wind deformer registers under.
pub const WIND_DEFORMER_NAME: &str = "viewport_lib_wind";

/// Per-vertex sway-mask stride: one `f32` per vertex (4 bytes).
pub const WIND_SWAY_STRIDE_BYTES: u32 = 4;

/// Per-mesh-instance material-params stride: three `f32` (strength,
/// density, speed) = 12 bytes.
pub const WIND_MATERIAL_PARAMS_STRIDE_BYTES: u32 = 12;

/// WGSL body for the wind deformer. Composed into the mesh-family shaders
/// by `register_deformer`.
pub const WIND_DEFORMER_BODY: &str = r#"
fn deform(v: DeformVertex, ctx: DeformContext) -> DeformVertex {
    var out = v;
    let slot = ctx.slot;
    let stride = deform_slot_stride(slot);
    if stride == 0u {
        return out;
    }
    let mask = deform_read_f32(slot, v.vertex_index, 0u);
    if mask <= 0.0 {
        return out;
    }
    // Per-mesh material multipliers (strength, density, speed). Live in the
    // per-mesh-instance slot data when the host has attached them; default
    // to 1.0 otherwise.
    var mat_strength: f32 = 1.0;
    var mat_density:  f32 = 1.0;
    var mat_speed:    f32 = 1.0;
    // Instance 0 carries the per-mesh material params. `stride` here is in
    // u32 words (3 = 12 bytes); when no params have been attached the slot
    // reads as zeroed and defaults kick in above.
    if deform_instance_slot_stride(slot) >= 3u {
        mat_strength = deform_read_instance_f32(slot, 0u, 0u);
        mat_density  = deform_read_instance_f32(slot, 0u, 1u);
        mat_speed    = deform_read_instance_f32(slot, 0u, 2u);
    }
    let p0 = deform_header.slot_params[slot * 4u + 0u];
    let p1 = deform_header.slot_params[slot * 4u + 1u];
    let dir = vec3<f32>(p0.x, p0.y, p0.z);
    let base = p0.w;
    let gust = p1.x;
    let freq = p1.y * mat_speed;
    let density = p1.z * mat_density;
    let time = p1.w;
    let origin = deform_header.slot_params[slot * 4u + 2u].xyz;
    let dlen_sq = dot(dir, dir);
    if dlen_sq < 1e-8 {
        return out;
    }
    let ndir = dir * inverseSqrt(dlen_sq);
    // The phase is keyed on the absolute position, so it does not shift when
    // the renderer's world origin moves.
    let world = v.position + origin;
    let phase = time * freq + (world.x + world.z) * density;
    let g = sin(phase) * gust;
    out.position = v.position + ndir * ((base + g) * mask * mat_strength);
    return out;
}
"#;

#[deprecated(note = "use WIND_DEFORMER_BODY")]
pub const SHARED_WIND_WGSL: &str = WIND_DEFORMER_BODY;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn body_defines_deform_function() {
        assert!(WIND_DEFORMER_BODY.contains("fn deform("));
    }

    #[test]
    fn body_reads_slot_data_and_params() {
        assert!(WIND_DEFORMER_BODY.contains("deform_read_f32"));
        assert!(WIND_DEFORMER_BODY.contains("deform_header.slot_params"));
    }

    #[test]
    fn body_reads_per_mesh_material_multipliers() {
        assert!(WIND_DEFORMER_BODY.contains("deform_read_instance_f32"));
        assert!(WIND_DEFORMER_BODY.contains("mat_strength"));
        assert!(WIND_DEFORMER_BODY.contains("mat_density"));
        assert!(WIND_DEFORMER_BODY.contains("mat_speed"));
    }
}
