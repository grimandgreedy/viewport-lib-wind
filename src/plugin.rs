//! Runtime + GPU plugin that registers the wind deformer and pushes the
//! current wind state into its slot params every frame.

use std::sync::{Arc, Mutex};

use viewport_lib::MeshId;
use viewport_lib::resources::DeviceResources;
use viewport_lib::runtime::{
    GpuFrameContext, GpuPlugin, RuntimePlugin, RuntimeStepContext, gpu_phase, phase,
};
use viewport_lib::{DeformSlotHandle, DeformStage, DeformerDesc, DeformerId};

use crate::field::{WindAuthoring, WindField};
use crate::shader::{
    WIND_DEFORMER_BODY, WIND_DEFORMER_NAME, WIND_MATERIAL_PARAMS_STRIDE_BYTES,
    WIND_SWAY_STRIDE_BYTES,
};

/// Per-mesh material parameters for the wind deformer.
///
/// Attach via [`WindPlugin::attach_material_params`]. Each multiplier
/// applies on top of the corresponding global field knob: `strength`
/// scales the total displacement, `density` scales the spatial wavelength
/// of the gust noise, `speed` scales the time-rate of the gust phase.
/// Meshes without attached params see `strength = density = speed = 1.0`
/// in the shader.
#[derive(Clone, Copy, Debug)]
pub struct WindMaterialParams {
    /// Multiplier on the field magnitude.
    pub strength: f32,
    /// Spatial wavelength multiplier on top of the field's `spatial_density`.
    pub density: f32,
    /// Time-rate multiplier on top of the field's `gust_frequency`.
    pub speed: f32,
}

impl Default for WindMaterialParams {
    fn default() -> Self {
        Self {
            strength: 1.0,
            density: 1.0,
            speed: 1.0,
        }
    }
}

impl WindMaterialParams {
    /// Pack into the 12-byte layout the deformer body expects.
    pub fn to_bytes(self) -> [u8; 12] {
        let words: [f32; 3] = [self.strength, self.density, self.speed];
        let mut out = [0u8; 12];
        out.copy_from_slice(bytemuck::cast_slice(&words));
        out
    }
}

/// Per-vertex sway-mask weights.
///
/// One `f32` per vertex, in the same order as the mesh's position
/// attribute. `0.0` is "this vertex never moves", `1.0` is "this vertex
/// moves at full strength". The `uniform` and `height_falloff` helpers
/// cover the common cases.
#[derive(Clone, Debug)]
pub struct WindSwayWeights {
    /// One sway mask per vertex.
    pub sway_mask: Vec<f32>,
}

impl WindSwayWeights {
    /// All vertices get the same mask value. Use `1.0` for a uniformly
    /// swaying flag or banner.
    pub fn uniform(vertex_count: usize, mask: f32) -> Self {
        Self {
            sway_mask: vec![mask; vertex_count],
        }
    }

    /// Sway scales with vertex height: vertices at `base_y` get `0.0`,
    /// vertices at `top_y` get `1.0`, anything outside the range is
    /// clamped. Use this for grass blades and tree foliage, where the
    /// root stays put and the tip catches the wind.
    pub fn height_falloff(positions: &[[f32; 3]], base_y: f32, top_y: f32) -> Self {
        let range = top_y - base_y;
        let sway_mask = if range.abs() < f32::EPSILON {
            vec![0.0; positions.len()]
        } else {
            positions
                .iter()
                .map(|p| ((p[1] - base_y) / range).clamp(0.0, 1.0))
                .collect()
        };
        Self { sway_mask }
    }
}

/// GPU-side state owned by the plugin. Held behind an `Arc<Mutex<_>>` so
/// the `RuntimePlugin` and `GpuPlugin` halves can share access without
/// taking ownership of each other.
struct WindInner {
    field: WindField,
    /// `Some` once [`WindPlugin::install`] has registered the deformer.
    slot: Option<DeformSlotHandle>,
}

/// Wind plugin handle.
///
/// Typical setup:
///
/// ```ignore
/// let wind = WindPlugin::new(WindAuthoring::default());
/// wind.install(&mut resources, &device)?;
///
/// let runtime = ViewportRuntime::new()
///     .with_plugin(wind.cpu_plugin())
///     .with_gpu_plugin(wind.gpu_plugin());
///
/// // For each grass / banner / branch mesh:
/// let mask = WindSwayWeights::height_falloff(&positions, 0.0, top_y);
/// wind.attach_sway_mask(&mut resources, &device, mesh_id, &mask)?;
/// ```
///
/// The CPU half advances the wind clock during `runtime.step()`. The GPU
/// half writes the current field state into the deformer's slot params
/// during `runtime.pre_prepare()` so every wind-affected mesh sees the
/// same global field that frame.
pub struct WindPlugin {
    inner: Arc<Mutex<WindInner>>,
}

impl WindPlugin {
    /// Create a new plugin with the given authoring values. Call
    /// [`Self::install`] before either trait-object half runs.
    pub fn new(authoring: WindAuthoring) -> Self {
        Self {
            inner: Arc::new(Mutex::new(WindInner {
                field: WindField::new(authoring),
                slot: None,
            })),
        }
    }

    /// Register the wind deformer against viewport-lib's mesh shader family.
    ///
    /// Composes the deformer body into every mesh-family shader, allocates
    /// a registry slot, and stashes a handle the GPU half uses each frame
    /// to push the wind state into the slot's `slot_params`.
    ///
    /// Returns the assigned [`DeformerId`] in case the host wants to
    /// reference the slot directly.
    pub fn install(
        &self,
        resources: &mut DeviceResources,
        device: &wgpu::Device,
    ) -> Result<DeformerId, viewport_lib::ViewportError> {
        let desc = DeformerDesc {
            name: WIND_DEFORMER_NAME,
            stage: DeformStage::WorldSpace,
            priority: 0,
            wgsl_body: WIND_DEFORMER_BODY.to_string(),
            per_vertex_stride: WIND_SWAY_STRIDE_BYTES,
        };
        let id = resources.register_deformer(device, desc)?;
        let handle = resources.deform_slot_handle(id);
        self.inner.lock().expect("wind plugin poisoned").slot = Some(handle);
        Ok(id)
    }

    /// Attach a sway mask to one mesh. The mask's length must equal the
    /// mesh's vertex count.
    ///
    /// Returns `false` and does nothing if [`Self::install`] has not run
    /// yet.
    pub fn attach_sway_mask(
        &self,
        resources: &mut DeviceResources,
        device: &wgpu::Device,
        mesh_id: MeshId,
        weights: &WindSwayWeights,
    ) -> bool {
        let inner = self.inner.lock().expect("wind plugin poisoned");
        let Some(handle) = inner.slot.as_ref() else {
            return false;
        };
        let slot = handle.slot();
        drop(inner);
        resources.attach_deform_slot(
            device,
            mesh_id,
            slot,
            WIND_SWAY_STRIDE_BYTES,
            bytemuck::cast_slice(&weights.sway_mask),
        );
        true
    }

    /// Detach the sway mask from one mesh.
    pub fn detach_sway_mask(
        &self,
        resources: &mut DeviceResources,
        device: &wgpu::Device,
        mesh_id: MeshId,
    ) -> bool {
        let slot = match self
            .inner
            .lock()
            .expect("wind plugin poisoned")
            .slot
            .as_ref()
        {
            Some(h) => h.slot(),
            None => return false,
        };
        resources.detach_deform_slot(device, mesh_id, slot)
    }

    /// Attach per-mesh material multipliers (strength / density / speed).
    ///
    /// Optional. Meshes that only have a sway mask attached use defaults of
    /// 1.0 across the board, matching the shipped global formula. Hosts
    /// that import per-material wind params from their content pipeline
    /// (e.g. Unity `_WindStrength`, `_WindDensity`, `_WindSpeed`) call this
    /// once after [`Self::attach_sway_mask`] for each mesh.
    ///
    /// Per-(mesh, instance) under the hood; for hosts that don't care about
    /// multiple instances per mesh, `instance_id = 0` is the convention
    /// and is what the shader body reads.
    ///
    /// Returns `false` and does nothing if [`Self::install`] has not run.
    pub fn attach_material_params(
        &self,
        resources: &mut DeviceResources,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        mesh_id: MeshId,
        params: &WindMaterialParams,
    ) -> bool {
        let inner = self.inner.lock().expect("wind plugin poisoned");
        let Some(handle) = inner.slot.as_ref() else {
            return false;
        };
        let slot = handle.slot();
        drop(inner);
        let bytes = params.to_bytes();
        resources.attach_deform_slot_instance(
            device,
            queue,
            mesh_id,
            0,
            slot,
            WIND_MATERIAL_PARAMS_STRIDE_BYTES,
            &bytes,
        );
        true
    }

    /// Trait object for the CPU half. Runs at [`phase::PREPARE`].
    pub fn cpu_plugin(&self) -> impl RuntimePlugin {
        WindCpuHalf {
            inner: self.inner.clone(),
        }
    }

    /// Trait object for the GPU half. Runs at [`gpu_phase::PRE_PREPARE`].
    pub fn gpu_plugin(&self) -> impl GpuPlugin {
        WindGpuHalf {
            inner: self.inner.clone(),
        }
    }

    /// Replace the authoring values. The wind clock is preserved.
    pub fn set_authoring(&self, authoring: WindAuthoring) {
        self.inner
            .lock()
            .expect("wind plugin poisoned")
            .field
            .set_authoring(authoring);
    }

    /// Snapshot the field for CPU consumers. Returns a clone so the caller
    /// can sample without holding the lock for the read.
    pub fn field(&self) -> WindField {
        self.inner
            .lock()
            .expect("wind plugin poisoned")
            .field
            .clone()
    }
}

struct WindCpuHalf {
    inner: Arc<Mutex<WindInner>>,
}

impl RuntimePlugin for WindCpuHalf {
    fn priority(&self) -> i32 {
        phase::PREPARE
    }

    fn step(&mut self, ctx: &mut RuntimeStepContext<'_>) {
        self.inner
            .lock()
            .expect("wind plugin poisoned")
            .field
            .advance(ctx.dt);
    }
}

struct WindGpuHalf {
    inner: Arc<Mutex<WindInner>>,
}

impl GpuPlugin for WindGpuHalf {
    fn priority(&self) -> i32 {
        gpu_phase::PRE_PREPARE
    }

    fn pre_prepare(
        &mut self,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
        _ctx: &GpuFrameContext<'_>,
    ) -> Vec<wgpu::CommandBuffer> {
        let inner = self.inner.lock().expect("wind plugin poisoned");
        if let Some(handle) = inner.slot.as_ref() {
            let params = inner.field.to_slot_params();
            handle.write(queue, &params);
        }
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    #[test]
    fn uniform_weights_have_right_length_and_value() {
        let w = WindSwayWeights::uniform(5, 0.7);
        assert_eq!(w.sway_mask.len(), 5);
        assert!(w.sway_mask.iter().all(|&m| (m - 0.7).abs() < 1e-6));
    }

    #[test]
    fn height_falloff_is_zero_at_base_and_one_at_top() {
        let positions = vec![
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 5.0, 0.0],
        ];
        let w = WindSwayWeights::height_falloff(&positions, 0.0, 2.0);
        assert_eq!(w.sway_mask[0], 0.0);
        assert_eq!(w.sway_mask[1], 0.5);
        assert_eq!(w.sway_mask[2], 1.0);
        assert_eq!(w.sway_mask[3], 0.0);
        assert_eq!(w.sway_mask[4], 1.0);
    }

    #[test]
    fn height_falloff_degenerate_range_is_zero() {
        let positions = vec![[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]];
        let w = WindSwayWeights::height_falloff(&positions, 1.0, 1.0);
        assert!(w.sway_mask.iter().all(|&m| m == 0.0));
    }

    fn try_make_device() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::LowPower,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok()?;
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).ok()
    }

    #[test]
    fn install_registers_deformer_on_a_host_slot() {
        let Some((device, _queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources = DeviceResources::new(&device, wgpu::TextureFormat::Bgra8UnormSrgb, 1);
        // One internal registration (skinning) is present at construction.
        let baseline = resources.registered_deformer_count();

        let wind = WindPlugin::new(WindAuthoring::default());
        let id = wind.install(&mut resources, &device).expect("install");

        assert!(id.slot() < viewport_lib::DEFORM_SLOT_COUNT);
        assert_eq!(resources.registered_deformer_count(), baseline + 1);
    }

    #[test]
    fn attach_material_params_writes_per_instance_slot_data() {
        let Some((device, queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources = DeviceResources::new(&device, wgpu::TextureFormat::Bgra8UnormSrgb, 1);
        let plane = viewport_lib::geometry::primitives::grid_plane(1.0, 1.0, 4, 4);
        let mesh_id = resources.upload_mesh_data(&device, &plane).unwrap();

        let wind = WindPlugin::new(WindAuthoring::default());
        let id = wind.install(&mut resources, &device).expect("install");

        let params = WindMaterialParams {
            strength: 0.2,
            density: 3.0,
            speed: 0.5,
        };
        assert!(wind.attach_material_params(&mut resources, &device, &queue, mesh_id, &params));
        assert!(resources.has_deform_slot_instance(mesh_id, 0, id.slot()));
    }

    #[test]
    fn attach_sway_mask_marks_slot_active_for_mesh() {
        let Some((device, _queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources = DeviceResources::new(&device, wgpu::TextureFormat::Bgra8UnormSrgb, 1);
        let plane = viewport_lib::geometry::primitives::grid_plane(1.0, 1.0, 4, 4);
        let mesh_id = resources.upload_mesh_data(&device, &plane).unwrap();

        let wind = WindPlugin::new(WindAuthoring::default());
        let id = wind.install(&mut resources, &device).expect("install");

        let mask = WindSwayWeights::uniform(plane.positions.len(), 1.0);
        assert!(wind.attach_sway_mask(&mut resources, &device, mesh_id, &mask));
        assert!(resources.has_deform_slot(mesh_id, id.slot()));

        assert!(wind.detach_sway_mask(&mut resources, &device, mesh_id));
        assert!(!resources.has_deform_slot(mesh_id, id.slot()));
    }

    #[test]
    fn plugin_field_round_trips_authoring() {
        let auth = WindAuthoring {
            direction: Vec3::Z,
            base_strength: 0.8,
            gust_strength: 0.0,
            gust_frequency: 0.0,
            spatial_density: 0.0,
        };
        let p = WindPlugin::new(auth);
        let sample = p.field().sample(Vec3::ZERO);
        assert!((sample - Vec3::new(0.0, 0.0, 0.8)).length() < 1e-6);
    }
}
