//! Wind field for `viewport-lib`.
//!
//! A single global wind vector with gust modulation, driven once per frame on
//! the CPU and pushed into a deformer slot's `slot_params` on the GPU.
//! Consumers read the field from two places:
//!
//! - The standard mesh shaders apply per-vertex wind through the deformer
//!   registry. [`WindPlugin::install`] registers the deformer; the GPU half
//!   keeps its slot params in sync with the CPU clock.
//! - Rust code calls [`WindField::sample`] for spring bones, particles, or
//!   any other CPU consumer that needs the same vector the renderer sees.
//! - With the `2d` feature, sprites and tile layers sway through a sprite
//!   and tile deformer that reads the mesh deformer's slot params.
//!   `WindPlugin::install_2d` registers it; a batch or layer opts in with
//!   `WindPlugin::sprite_deformers`.
//!
//! Typical setup:
//!
//! ```ignore
//! use viewport_lib::runtime::ViewportRuntime;
//! use viewport_lib_wind::{WindAuthoring, WindPlugin, WindSwayWeights};
//!
//! let wind = WindPlugin::new(WindAuthoring::default());
//! wind.install(&mut resources, &device).expect("register wind deformer");
//!
//! let runtime = ViewportRuntime::new()
//!     .with_plugin(wind.cpu_plugin())
//!     .with_gpu_plugin(wind.gpu_plugin());
//!
//! // Per wind-affected mesh:
//! let mask = WindSwayWeights::height_falloff(&positions, 0.0, top_z);
//! wind.attach_sway_mask(&mut resources, &device, mesh_id, &mask);
//! ```
//!
//! The CPU sampler and the WGSL deformer body use the same formula, so a
//! vertex drawn through the deformer and a spring bone tracking the same
//! world position see the same wind to within float precision.

mod field;
mod globals;
mod plugin;
mod shader;
#[cfg(feature = "2d")]
mod sprite;

pub use field::{WindAuthoring, WindField};
pub use globals::WindGlobals;
pub use plugin::{WindMaterialParams, WindPlugin, WindSwayWeights};
#[allow(deprecated)]
pub use shader::SHARED_WIND_WGSL;
pub use shader::{
    WGSL_VERSION, WIND_DEFORMER_BODY, WIND_DEFORMER_NAME, WIND_MATERIAL_PARAMS_STRIDE_BYTES,
    WIND_SWAY_STRIDE_BYTES,
};
#[cfg(feature = "2d")]
pub use sprite::{WIND_SPRITE_DEFORMER_BODY, WIND_SPRITE_DEFORMER_NAME};

/// Catalogue version of `viewport-lib` this crate was built against.
///
/// Bump in lockstep with `viewport_lib::plugin_api::shared_wgsl::WGSL_VERSION`
/// whenever the shared-WGSL contract changes. The compile-time assertion
/// below catches accidental drift.
pub const VIEWPORT_LIB_WGSL_VERSION: u32 = 12;

const _: () = assert!(
    viewport_lib::plugin_api::shared_wgsl::WGSL_VERSION == VIEWPORT_LIB_WGSL_VERSION,
    "viewport-lib catalogue version drifted; update VIEWPORT_LIB_WGSL_VERSION and review WIND_DEFORMER_BODY",
);
