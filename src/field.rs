//! CPU-side wind authoring and sampling.

use glam::Vec3;

use crate::globals::WindGlobals;

/// Authoring knobs for a [`WindField`]. These describe what the field looks
/// like at rest; the field itself owns the time clock that ticks forward.
#[derive(Clone, Copy, Debug)]
pub struct WindAuthoring {
    /// Prevailing wind direction. Need not be unit length.
    pub direction: Vec3,
    /// Steady-state magnitude applied along `direction`.
    pub base_strength: f32,
    /// Amplitude of the gust envelope.
    pub gust_strength: f32,
    /// Gust cycles per second.
    pub gust_frequency: f32,
    /// Spatial wavelength factor for gust noise. Larger means smaller gusts.
    pub spatial_density: f32,
}

impl Default for WindAuthoring {
    fn default() -> Self {
        Self {
            direction: Vec3::new(1.0, 0.0, 0.0),
            base_strength: 1.0,
            gust_strength: 0.3,
            gust_frequency: 0.5,
            spatial_density: 0.25,
        }
    }
}

/// The CPU-side wind field. Holds the authoring values plus a running clock.
///
/// `sample(world_pos)` returns the same vector the GPU helper produces for a
/// vertex at the same position, so spring bones, particles, and cloth see a
/// consistent field.
#[derive(Clone, Debug)]
pub struct WindField {
    authoring: WindAuthoring,
    time: f32,
}

impl WindField {
    /// Build a field from authoring values, starting the clock at zero.
    pub fn new(authoring: WindAuthoring) -> Self {
        Self {
            authoring,
            time: 0.0,
        }
    }

    /// Replace the authoring values. The clock is preserved so a runtime
    /// tweak does not snap the field discontinuously.
    pub fn set_authoring(&mut self, authoring: WindAuthoring) {
        self.authoring = authoring;
    }

    /// Current authoring values.
    pub fn authoring(&self) -> WindAuthoring {
        self.authoring
    }

    /// Advance the clock by `dt` seconds.
    pub fn advance(&mut self, dt: f32) {
        self.time += dt;
    }

    /// Reset the clock to zero. Authoring values are preserved.
    pub fn reset(&mut self) {
        self.time = 0.0;
    }

    /// Current clock value.
    pub fn time(&self) -> f32 {
        self.time
    }

    /// Sample the wind at a world-space position, in world coordinates rather
    /// than relative to the renderer's world origin. Matches the deformer
    /// formula in [`crate::WIND_DEFORMER_BODY`].
    pub fn sample(&self, world_pos: Vec3) -> Vec3 {
        let dir = match self.authoring.direction.try_normalize() {
            Some(d) => d,
            None => return Vec3::ZERO,
        };
        let phase = self.time * self.authoring.gust_frequency
            + (world_pos.x + world_pos.z) * self.authoring.spatial_density;
        let gust = phase.sin() * self.authoring.gust_strength;
        dir * (self.authoring.base_strength + gust)
    }

    /// Pack the field into the GPU uniform layout.
    pub fn to_globals(&self) -> WindGlobals {
        WindGlobals::new(
            self.authoring.direction.to_array(),
            self.authoring.base_strength,
            self.authoring.gust_strength,
            self.authoring.gust_frequency,
            self.authoring.spatial_density,
            self.time,
        )
    }

    /// Pack the field into the four `vec4<f32>` slot-params words that the
    /// deformer body reads from `deform_header.slot_params[slot * 4 .. + 4]`.
    ///
    /// Layout:
    /// - `[0]` = `(direction.x, direction.y, direction.z, base_strength)`
    /// - `[1]` = `(gust_strength, gust_frequency, spatial_density, time)`
    /// - `[2]` = the renderer's world origin in xyz, which this leaves at zero;
    ///   [`crate::WindPlugin`] fills it in from
    ///   [`set_world_origin`](crate::WindPlugin::set_world_origin)
    /// - `[3]` is reserved for future use.
    pub fn to_slot_params(&self) -> [[f32; 4]; 4] {
        let d = self.authoring.direction;
        [
            [d.x, d.y, d.z, self.authoring.base_strength],
            [
                self.authoring.gust_strength,
                self.authoring.gust_frequency,
                self.authoring.spatial_density,
                self.time,
            ],
            [0.0; 4],
            [0.0; 4],
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: Vec3, b: Vec3, eps: f32) -> bool {
        (a - b).length() < eps
    }

    #[test]
    fn advance_accumulates_time() {
        let mut f = WindField::new(WindAuthoring::default());
        f.advance(0.5);
        f.advance(0.25);
        assert!((f.time() - 0.75).abs() < 1e-6);
    }

    #[test]
    fn reset_clears_time_only() {
        let auth = WindAuthoring {
            base_strength: 2.0,
            ..WindAuthoring::default()
        };
        let mut f = WindField::new(auth);
        f.advance(10.0);
        f.reset();
        assert_eq!(f.time(), 0.0);
        assert_eq!(f.authoring().base_strength, 2.0);
    }

    #[test]
    fn zero_direction_samples_zero() {
        let auth = WindAuthoring {
            direction: Vec3::ZERO,
            ..WindAuthoring::default()
        };
        let f = WindField::new(auth);
        assert_eq!(f.sample(Vec3::new(1.0, 2.0, 3.0)), Vec3::ZERO);
    }

    #[test]
    fn steady_field_returns_direction_times_strength() {
        let auth = WindAuthoring {
            direction: Vec3::X,
            base_strength: 2.0,
            gust_strength: 0.0,
            gust_frequency: 0.0,
            spatial_density: 0.0,
        };
        let f = WindField::new(auth);
        assert!(approx(
            f.sample(Vec3::new(7.0, 0.0, 4.0)),
            Vec3::new(2.0, 0.0, 0.0),
            1e-6,
        ));
    }

    #[test]
    fn to_globals_round_trips_fields() {
        let auth = WindAuthoring {
            direction: Vec3::new(1.0, 0.0, 2.0),
            base_strength: 1.5,
            gust_strength: 0.4,
            gust_frequency: 0.3,
            spatial_density: 0.5,
        };
        let mut f = WindField::new(auth);
        f.advance(1.25);
        let g = f.to_globals();
        assert_eq!(g.direction, [1.0, 0.0, 2.0]);
        assert_eq!(g.base_strength, 1.5);
        assert_eq!(g.gust_strength, 0.4);
        assert_eq!(g.gust_frequency, 0.3);
        assert_eq!(g.spatial_density, 0.5);
        assert!((g.time - 1.25).abs() < 1e-6);
    }
}
