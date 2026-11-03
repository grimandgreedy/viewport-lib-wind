//! Loading the meadow's glTF models and uploading them with their wind masks.

use std::path::PathBuf;

use glam::Mat4;
use viewport_lib::resources::DeviceResources;
use viewport_lib::{MaterialId, MeshId, TextureData, TextureId, TextureSlot, wgpu};
use viewport_lib_io::TextureSource;
use viewport_lib_io::loaders::{gltf, png};
use viewport_lib_wind::{WindMaterialParams, WindPlugin, WindSwayWeights};

/// One uploaded mesh part of a model: a glTF primitive with its material.
pub struct Part {
    pub mesh: MeshId,
    pub material: MaterialId,
    /// The primitive's transform inside the model.
    pub local: Mat4,
}

/// A model ready to place: its parts, all sharing one sway mask range.
pub struct Model {
    pub parts: Vec<Part>,
}

impl Model {
    /// Every (mesh, material, transform) to add for one copy placed at `placement`.
    pub fn instances(&self, placement: Mat4) -> impl Iterator<Item = (MeshId, MaterialId, Mat4)> {
        self.parts
            .iter()
            .map(move |p| (p.mesh, p.material, placement * p.local))
    }
}

/// Uploads models from one folder, sharing textures between them.
pub struct ModelLoader<'a> {
    pub dir: PathBuf,
    pub res: &'a mut DeviceResources,
    pub device: &'a wgpu::Device,
    pub queue: &'a wgpu::Queue,
    /// Base colour textures already uploaded. Several models share an image,
    /// and the loader hands each one out decoded, so they are matched by content.
    pub textures: Vec<(TextureData, TextureId)>,
}

impl ModelLoader<'_> {
    /// Load `name.gltf`. With `sway` set, every part gets a height-falloff sway
    /// mask over the whole model's height (still at the base, full at the top)
    /// and the given wind multipliers.
    pub fn load(&mut self, name: &str, sway: Option<(&WindPlugin, WindMaterialParams)>) -> Model {
        let path = self.dir.join(format!("{name}.gltf"));
        let scene =
            gltf::scene_from_path(&path).unwrap_or_else(|e| panic!("load {}: {e}", path.display()));

        // Only the base colour texture is uploaded: the leaf cards need it for
        // their alpha cut-out, and the meadow reads fine without normal maps.
        let materials: Vec<MaterialId> = scene
            .materials
            .iter()
            .map(|m| {
                let material = m.to_material(|slot, source| {
                    if slot != TextureSlot::Albedo {
                        return None;
                    }
                    let data = match source {
                        TextureSource::Decoded(data) => data.clone(),
                        TextureSource::File(file) => png::texture_from_path_for_slot(file, slot)
                            .unwrap_or_else(|e| panic!("load {}: {e}", file.display())),
                    };
                    Some(self.texture(data))
                });
                self.res.upload_material(material).expect("upload material")
            })
            .collect();
        let fallback = materials[0];

        // The loader hands out Z-up data, so the model's height is its Z range.
        let (mut base_z, mut top_z) = (f32::MAX, f32::MIN);
        for mesh in &scene.meshes {
            for p in &mesh.mesh.positions {
                let z = mesh.transform.transform_point3((*p).into()).z;
                base_z = base_z.min(z);
                top_z = top_z.max(z);
            }
        }

        let parts = scene
            .meshes
            .iter()
            .map(|m| {
                let mesh = self
                    .res
                    .upload_mesh_data(self.device, &m.mesh)
                    .expect("upload mesh");
                if let Some((wind, params)) = sway {
                    // The mask is read against the mesh's own positions, so
                    // bring them into the model's frame first.
                    let positions: Vec<[f32; 3]> = m
                        .mesh
                        .positions
                        .iter()
                        .map(|p| m.transform.transform_point3((*p).into()).into())
                        .collect();
                    let mask = WindSwayWeights::height_falloff(&positions, base_z, top_z);
                    wind.attach_sway_mask(self.res, self.device, mesh, &mask);
                    wind.attach_material_params(self.res, self.device, self.queue, mesh, &params);
                }
                Part {
                    mesh,
                    material: m.material_index.map_or(fallback, |i| materials[i]),
                    local: m.transform,
                }
            })
            .collect();
        Model { parts }
    }

    /// Upload a base colour texture once and reuse it.
    fn texture(&mut self, data: TextureData) -> TextureId {
        if let Some((_, id)) = self.textures.iter().find(|(seen, _)| *seen == data) {
            return *id;
        }
        let id = self
            .res
            .upload_texture(self.device, self.queue, data.clone())
            .expect("upload texture");
        self.textures.push((data, id));
        id
    }
}
