//! wind-2d: grass, reeds and a strip of grass tiles swaying in a side view.
//!
//! A side view under `Camera2d` on the XZ plane (Z up), one world unit to a
//! texel. Along the ground a tile layer holds a row of dirt with a row of grass
//! tiles on top, and grass tufts, reeds and rocks stand on it as sprites. Every
//! texture is drawn in code as pixel art.
//!
//! The wind's sprite and tile deformer bends each quad from its base. The
//! sprites and the layer opt in through `WindPlugin::sprite_deformers`, and
//! each sprite's `WindSway2d` user value sets how far it sways: tufts by
//! slightly different amounts, reeds by more, rocks not at all. The tileset's
//! user-value table marks the grass tile as swaying and the dirt tile as still,
//! so the same layer holds both. Gusts roll along the ground, so neighbouring
//! plants move a moment apart.
//!
//! Controls:
//!   Left drag    : pan
//!   Scroll       : zoom
//!   Space        : turn the wind on and off
//!   P            : turn pixel-art mode on and off (offsets in whole texels)
//!
//! Run with `cargo run --release --example wind-2d --features 2d`.

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use glam::{Vec2, Vec3};
use viewport_lib::plugin_api::Uploads;
use viewport_lib::runners::viewport_app::RedrawMode;
use viewport_lib::runtime::ViewportRuntime;
use viewport_lib::{
    AppConfig, ButtonState, Camera2d, Camera2dControl, CameraControl, Colour, ExtraId, Plane2d,
    TextureData, TextureFilter, TextureId, ViewportApp, ViewportEvent, WindowConfig,
    planar_bindings,
};
use viewport_lib_plugins::item_types::sprite::{SpriteItem, SpriteSizeMode};
use viewport_lib_plugins::item_types::tilemap::{
    Tile, Tilemap, TilemapId, TilemapItem, TilemapPlane, Tileset,
};
use viewport_lib_wind::{WindAuthoring, WindPlugin, WindSway2d};

/// Ground tiles across the view, and the tile size in texels.
const COLUMNS: u32 = 24;
const TILE: u32 = 16;
/// The ground's left end and the height of its top surface.
const LEFT: f32 = -((COLUMNS * TILE) as f32) * 0.5;
const GROUND: f32 = -40.0;

/// A gusting breeze along +X. Offsets are in world units (texels) at the top of
/// a sprite; the reeds are 40 texels tall.
fn breeze() -> WindAuthoring {
    WindAuthoring {
        direction: Vec3::X,
        base_strength: 2.5,
        gust_strength: 3.5,
        gust_frequency: 1.6,
        spatial_density: 0.025,
    }
}

fn calm() -> WindAuthoring {
    WindAuthoring {
        base_strength: 0.0,
        gust_strength: 0.0,
        ..breeze()
    }
}

/// A small deterministic random source, so the field is the same every run.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 40) as f32 / (1u64 << 24) as f32
    }

    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.next()
    }
}

/// An RGBA image addressed from its bottom-left texel, turned over into the
/// top-row-first order a texture upload takes.
struct Canvas {
    w: u32,
    h: u32,
    rgba: Vec<u8>,
}

impl Canvas {
    fn new(w: u32, h: u32) -> Self {
        Self {
            w,
            h,
            rgba: vec![0; (w * h * 4) as usize],
        }
    }

    fn set(&mut self, x: i32, y: i32, c: [u8; 3]) {
        if x < 0 || y < 0 || x >= self.w as i32 || y >= self.h as i32 {
            return;
        }
        let row = self.h - 1 - y as u32;
        let i = ((row * self.w + x as u32) * 4) as usize;
        self.rgba[i..i + 4].copy_from_slice(&[c[0], c[1], c[2], 255]);
    }

    /// A blade from `(x, 0)` up to height `h`, leaning by `lean` texels at the
    /// tip.
    fn blade(&mut self, x: i32, h: i32, lean: f32, c: [u8; 3]) {
        for y in 0..h {
            let dx = (lean * y as f32 / h as f32).round() as i32;
            self.set(x + dx, y, c);
        }
    }

    fn texture(self) -> TextureData {
        TextureData::srgb(self.w, self.h, self.rgba)
    }
}

const GREENS: [[u8; 3]; 3] = [[86, 160, 62], [62, 128, 50], [118, 186, 74]];

fn tuft(rng: &mut Lcg) -> TextureData {
    let mut c = Canvas::new(16, 16);
    for i in 0..7 {
        let x = 2 + i * 2;
        let h = rng.range(8.0, 16.0) as i32;
        c.blade(x, h, rng.range(-2.0, 2.0), GREENS[i as usize % 3]);
    }
    c.texture()
}

fn reed() -> TextureData {
    let mut c = Canvas::new(8, 40);
    c.blade(3, 40, 0.0, [104, 132, 58]);
    c.blade(4, 28, 0.0, [86, 112, 48]);
    // The seed head.
    for y in 30..38 {
        c.set(3, y, [120, 76, 40]);
        c.set(4, y, [104, 64, 34]);
    }
    // A leaf off each side.
    c.blade(2, 18, -2.0, GREENS[1]);
    c.blade(5, 14, 2.0, GREENS[0]);
    c.texture()
}

fn rock() -> TextureData {
    let mut c = Canvas::new(16, 10);
    for y in 0..10 {
        for x in 0..16 {
            let (dx, dy) = ((x as f32 - 7.5) / 8.0, y as f32 / 10.0);
            if dx * dx + dy * dy < 1.0 {
                let shade = if y > 6 || dx < -0.4 { 150 } else { 120 };
                c.set(x, y, [shade, shade, shade + 8]);
            }
        }
    }
    c.texture()
}

/// Two tiles side by side: 0 is tall grass, 1 is dirt.
fn tiles(rng: &mut Lcg) -> TextureData {
    let mut c = Canvas::new(TILE * 2, TILE);
    for x in 0..TILE as i32 {
        let h = rng.range(6.0, 15.0) as i32;
        c.blade(x, h, rng.range(-1.0, 1.0), GREENS[x as usize % 3]);
    }
    for y in 0..TILE as i32 {
        for x in 0..TILE as i32 {
            let v = rng.next();
            let shade = if v > 0.85 {
                [96, 70, 46]
            } else {
                [120, 88, 56]
            };
            c.set(TILE as i32 + x, y, shade);
        }
    }
    c.texture()
}

/// Everything the frame callback needs, rebuilt by the setup callback after a
/// device loss.
struct Meadow {
    wind: WindPlugin,
    tufts: TextureId,
    reed: TextureId,
    rock: TextureId,
    ground: TilemapId,
}

/// One sprite of a batch: where its base stands, its size and its sway.
struct Plant {
    at: Vec2,
    size: [f32; 2],
    sway: f32,
}

fn plants(rng: &mut Lcg, count: usize, size: [f32; 2], sway: [f32; 2]) -> Vec<Plant> {
    (0..count)
        .map(|_| {
            let s = rng.range(0.8, 1.2);
            Plant {
                // Rooted a little way down into the grass row.
                at: Vec2::new(
                    rng.range(LEFT + 8.0, -LEFT - 8.0),
                    GROUND - rng.range(2.0, 8.0),
                ),
                size: [size[0] * s, size[1] * s],
                sway: rng.range(sway[0], sway[1]),
            }
        })
        .collect()
}

/// A batch of `plants` drawn with `texture`, standing on their bases slightly
/// in front of the tile layer.
fn batch(meadow: &Meadow, texture: TextureId, plants: &[Plant], pixel_art: bool) -> SpriteItem {
    let mut item = SpriteItem::default();
    item.texture_id = Some(texture);
    item.positions = plants.iter().map(|p| [p.at.x, -1.0, p.at.y]).collect();
    item.sizes = plants.iter().map(|p| p.size).collect();
    item.default_pivot = [0.5, 0.0];
    item.size_mode = SpriteSizeMode::WorldSpace;
    item.filter = TextureFilter::Nearest;
    item.user_values = plants
        .iter()
        .map(|p| {
            let sway = WindSway2d::new(p.sway);
            if pixel_art {
                sway.with_pixel_art()
            } else {
                sway
            }
            .to_user_value()
        })
        .collect();
    item.deformers = meadow.wind.sprite_deformers();
    item
}

fn pressed(events: &[ViewportEvent], key: char) -> bool {
    events.iter().any(|event| {
        matches!(
            event,
            ViewportEvent::Key {
                state: ButtonState::Pressed,
                logical: Some(k),
                ..
            } if *k == key
        )
    })
}

fn main() {
    let meadow: Rc<RefCell<Option<Meadow>>> = Rc::new(RefCell::new(None));
    let setup_meadow = meadow.clone();
    let wind_on = Rc::new(Cell::new(true));
    let setup_wind_on = wind_on.clone();
    let mut pixel_art = false;

    let mut rng = Lcg(0x2d_5eed);
    let tufts = plants(&mut rng, 40, [16.0, 16.0], [0.6, 1.0]);
    let reeds = plants(&mut rng, 9, [8.0, 40.0], [1.0, 1.3]);
    let rocks = plants(&mut rng, 5, [16.0, 10.0], [0.0, 0.0]);
    let mut extras: Vec<ExtraId> = Vec::new();

    let camera = Camera2d::new(Plane2d::Xz).with_units_per_pixel(0.25);

    ViewportApp::new(AppConfig::default())
        .window(
            WindowConfig::default()
                .with_title("viewport-lib-wind : 2D")
                .with_window_size(1280, 720)
                .with_redraw_mode(RedrawMode::Continuous)
                .with_camera_control(CameraControl::Controller(Box::new(Camera2dControl::new(
                    camera,
                )))),
            move |setup| {
                let device = setup.device().clone();
                let queue = setup.queue().clone();
                for (name, reason) in
                    viewport_lib_plugins::item_types::install(setup.renderer_mut(), &device)
                {
                    eprintln!("item type {name} unavailable: {reason}");
                }

                let wind = WindPlugin::new(if setup_wind_on.get() {
                    breeze()
                } else {
                    calm()
                });
                setup.set_runtime(
                    ViewportRuntime::new()
                        .with_plugin(wind.cpu_plugin())
                        .with_gpu_plugin(wind.gpu_plugin()),
                );
                wind.install(setup.resources_mut(), &device)
                    .expect("register wind deformer");
                // After the item types, so the tilemap type takes the deformer.
                wind.install_2d(setup.renderer_mut(), &device)
                    .expect("register wind sprite deformer");

                let mut rng = Lcg(0x7e_c5);
                let resources = setup.resources_mut();
                let mut upload = |data| {
                    resources
                        .upload_texture(&device, &queue, data)
                        .expect("texture upload")
                };
                let tufts = upload(tuft(&mut rng));
                let reed = upload(reed());
                let rock = upload(rock());
                let atlas = upload(tiles(&mut rng));

                let mut tileset = Tileset::new(Some(atlas), [TILE, TILE], 2);
                tileset.filter = TextureFilter::Nearest;
                tileset.user_values = WindSway2d::tile_values(&[(0, WindSway2d::new(0.8))]);
                let mut map = Tilemap::new(tileset, [COLUMNS, 2]);
                for x in 0..COLUMNS as i32 {
                    map.set(x, 0, Tile::new(1));
                    map.set(x, 1, Tile::new(0));
                }
                let ground = setup
                    .renderer_mut()
                    .upload(&device, &queue, &map)
                    .expect("tilemap upload");

                let frame = setup.viewport_frame_mut();
                frame.background_colour = Some(Colour::srgb_rgb(0.62, 0.78, 0.92));
                frame.show_axes_indicator = false;
                setup.set_bindings(planar_bindings());

                *setup_meadow.borrow_mut() = Some(Meadow {
                    wind,
                    tufts,
                    reed,
                    rock,
                    ground,
                });
            },
            move |ctx| {
                let events = ctx.events().to_vec();
                let meadow = meadow.borrow();
                let Some(meadow) = meadow.as_ref() else {
                    return;
                };
                if pressed(&events, ' ') {
                    wind_on.set(!wind_on.get());
                    meadow
                        .wind
                        .set_authoring(if wind_on.get() { breeze() } else { calm() });
                }
                if pressed(&events, 'p') {
                    pixel_art = !pixel_art;
                }
                // Sprite positions are relative to the renderer's world
                // origin, which follows the camera; the gust phase adds it back.
                let origin = ctx.renderer_mut().world_origin();
                meadow.wind.set_world_origin(origin);

                for id in extras.drain(..) {
                    ctx.remove_extra(id);
                }
                // The grass row of the layer sits under the sprites' bases, so
                // the ground's top surface is at `GROUND`.
                let mut layer = TilemapItem::new(meadow.ground);
                layer.plane = TilemapPlane::Xz;
                layer.origin = [LEFT, GROUND - (TILE * 2) as f32];
                layer.cell_size = [TILE as f32, TILE as f32];
                layer.deformers = meadow.wind.sprite_deformers();
                extras.push(ctx.add_item(layer));
                extras.push(ctx.add_item(batch(meadow, meadow.reed, &reeds, pixel_art)));
                extras.push(ctx.add_item(batch(meadow, meadow.tufts, &tufts, pixel_art)));
                extras.push(ctx.add_item(batch(meadow, meadow.rock, &rocks, pixel_art)));
            },
        )
        .run();
}
