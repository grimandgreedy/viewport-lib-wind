//! POD uniform layout uploaded to the GPU each frame.

use bytemuck::{Pod, Zeroable};

/// Per-frame wind state, mirroring the WGSL `WindGlobals` struct.
///
/// Total size is 64 bytes. Padding fields keep the Rust struct aligned with
/// the std140 layout the shader expects.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct WindGlobals {
    /// World-space wind direction. Need not be unit length: the shader
    /// normalises before use, and the magnitude contributes to base strength.
    pub direction: [f32; 3],
    /// Steady-state wind strength applied alongside `direction`.
    pub base_strength: f32,
    /// Amplitude of the gust envelope.
    pub gust_strength: f32,
    /// Cycles per second of the gust envelope.
    pub gust_frequency: f32,
    /// Spatial wavelength factor: larger values mean smaller gusts.
    pub spatial_density: f32,
    /// Wind clock, advanced by the runtime plugin each frame.
    pub time: f32,
    _pad: [f32; 8],
}

impl WindGlobals {
    /// Build a value from individual field values. The trailing padding is
    /// zeroed automatically so the std140 layout stays correct.
    pub fn new(
        direction: [f32; 3],
        base_strength: f32,
        gust_strength: f32,
        gust_frequency: f32,
        spatial_density: f32,
        time: f32,
    ) -> Self {
        Self {
            direction,
            base_strength,
            gust_strength,
            gust_frequency,
            spatial_density,
            time,
            _pad: [0.0; 8],
        }
    }

    /// A still scene: zero direction, zero strength, zero gusts.
    pub const STILL: Self = Self {
        direction: [0.0, 0.0, 0.0],
        base_strength: 0.0,
        gust_strength: 0.0,
        gust_frequency: 0.0,
        spatial_density: 1.0,
        time: 0.0,
        _pad: [0.0; 8],
    };

    /// Size in bytes of the uniform buffer this struct maps to.
    pub const SIZE: u64 = std::mem::size_of::<Self>() as u64;
}

impl Default for WindGlobals {
    fn default() -> Self {
        Self::STILL
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_is_64_bytes() {
        assert_eq!(WindGlobals::SIZE, 64);
        assert_eq!(std::mem::align_of::<WindGlobals>(), 4);
    }

    #[test]
    fn still_is_zero_except_density() {
        let s = WindGlobals::STILL;
        assert_eq!(s.direction, [0.0; 3]);
        assert_eq!(s.base_strength, 0.0);
        assert_eq!(s.gust_strength, 0.0);
        assert_eq!(s.gust_frequency, 0.0);
        assert_eq!(s.time, 0.0);
        assert_eq!(s.spatial_density, 1.0);
    }
}
