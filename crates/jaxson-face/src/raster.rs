//! A tiny pure software rasterizer: turn a [`Face`] into a square black-and-white
//! [`Bitmap`] — two eyes, a mouth, and reactive ears. No GUI, so the look of the face can
//! be validated headlessly (printed as ASCII, property-tested), and the egui app and a
//! future hardware display can both draw from the very same pixels.

use crate::Face;

/// A square 1-bit image. `true` = ink (drawn), `false` = background.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Bitmap {
    size: usize,
    pixels: Vec<bool>,
}

impl Bitmap {
    /// A blank `size`×`size` bitmap.
    pub fn new(size: usize) -> Self {
        Bitmap {
            size,
            pixels: vec![false; size * size],
        }
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn get(&self, x: usize, y: usize) -> bool {
        self.pixels[y * self.size + x]
    }

    fn set(&mut self, x: usize, y: usize) {
        self.pixels[y * self.size + x] = true;
    }

    /// Number of inked pixels.
    pub fn ink(&self) -> usize {
        self.pixels.iter().filter(|p| **p).count()
    }

    /// Render as ASCII (`#` = ink). Rows are sampled at half resolution so the result
    /// looks roughly square in a monospace terminal (characters are ~2× tall).
    pub fn to_ascii(&self) -> String {
        let mut out = String::with_capacity(self.size * self.size / 2 + self.size);
        let mut y = 0;
        while y < self.size {
            for x in 0..self.size {
                out.push(if self.get(x, y) { '#' } else { ' ' });
            }
            out.push('\n');
            y += 2;
        }
        out
    }
}

/// Render `face` into a `size`×`size` bitmap. Coordinates are normalized to `[0, 1]`.
pub fn rasterize(face: &Face, size: usize) -> Bitmap {
    let mut bmp = Bitmap::new(size);

    // Ears on top, then eyes (mirrored around the vertical centerline), then mouth.
    draw_ears(&mut bmp, &face.ears);
    draw_eye(&mut bmp, 0.36, &face.left_eye);
    draw_eye(&mut bmp, 0.64, &face.right_eye);
    draw_mouth(&mut bmp, &face.mouth);

    bmp
}

fn draw_ears(bmp: &mut Bitmap, ears: &crate::Ears) {
    // Ears sit on the sides of the head, attached by a short vertical base. The tip
    // swings from pointing up (perked, +1) to hanging down the side (drooped, -1).
    let t = (ears.perk.clamp(-1.0, 1.0) + 1.0) / 2.0;
    let lerp = |drooped: (f64, f64), perked: (f64, f64)| {
        (
            drooped.0 + (perked.0 - drooped.0) * t,
            drooped.1 + (perked.1 - drooped.1) * t,
        )
    };
    // Left ear, on the left side of the head.
    let left_tip = lerp((0.07, 0.60), (0.15, 0.06));
    fill_triangle(bmp, (0.20, 0.30), (0.20, 0.47), left_tip);
    // Right ear mirrors the left around x = 0.5.
    let right_tip = lerp((0.93, 0.60), (0.85, 0.06));
    fill_triangle(bmp, (0.80, 0.30), (0.80, 0.47), right_tip);
}

/// Fill the triangle with normalized vertices `a`, `b`, `c` (winding-independent).
fn fill_triangle(bmp: &mut Bitmap, a: (f64, f64), b: (f64, f64), c: (f64, f64)) {
    let size = bmp.size as f64;
    let p = |v: (f64, f64)| (v.0 * size, v.1 * size);
    let (a, b, c) = (p(a), p(b), p(c));

    let min_x = a.0.min(b.0).min(c.0).floor().max(0.0) as usize;
    let max_x = ((a.0.max(b.0).max(c.0).ceil()) as usize).min(bmp.size - 1);
    let min_y = a.1.min(b.1).min(c.1).floor().max(0.0) as usize;
    let max_y = ((a.1.max(b.1).max(c.1).ceil()) as usize).min(bmp.size - 1);

    let edge = |s: (f64, f64), e: (f64, f64), x: f64, y: f64| {
        (e.0 - s.0) * (y - s.1) - (e.1 - s.1) * (x - s.0)
    };

    for py in min_y..=max_y {
        for px in min_x..=max_x {
            let (x, y) = (px as f64 + 0.5, py as f64 + 0.5);
            let d1 = edge(a, b, x, y);
            let d2 = edge(b, c, x, y);
            let d3 = edge(c, a, x, y);
            let has_neg = d1 < 0.0 || d2 < 0.0 || d3 < 0.0;
            let has_pos = d1 > 0.0 || d2 > 0.0 || d3 > 0.0;
            if !(has_neg && has_pos) {
                bmp.set(px, py);
            }
        }
    }
}

const EYE_CENTER_Y: f64 = 0.42;
const EYE_HALF_W: f64 = 0.05;
const EYE_HALF_H: f64 = 0.09;
const GAZE_SHIFT: f64 = 0.04;

fn draw_eye(bmp: &mut Bitmap, center_x: f64, eye: &crate::Eye) {
    // Ron's eyes are tall rounded rectangles (vertical capsules). Gaze drifts the whole
    // eye a little; blinking shrinks its height — a tall eye stays a capsule, a nearly
    // shut one collapses to a flat line (drawn as an ellipse).
    let cx = center_x + eye.pupil_dx * GAZE_SHIFT;
    let cy = EYE_CENTER_Y + eye.pupil_dy * GAZE_SHIFT;
    let half_h = (EYE_HALF_H * eye.openness).max(EYE_HALF_W * 0.15);
    if half_h > EYE_HALF_W {
        fill_capsule_v(bmp, cx, cy, EYE_HALF_W, half_h);
    } else {
        fill_ellipse(bmp, cx, cy, EYE_HALF_W, half_h);
    }
}

const MOUTH_CENTER_X: f64 = 0.5;
const MOUTH_BASE_Y: f64 = 0.60;
const MOUTH_HALF_WIDTH: f64 = 0.11;
const MOUTH_AMPLITUDE: f64 = 0.06;
/// Thin stroke of the closed lip line (half-thickness, normalized).
const MOUTH_LINE_HALF: f64 = 0.009;
/// How far the middle of the mouth drops open at full `openness`, normalized.
const MOUTH_OPEN_DEPTH: f64 = 0.12;

fn draw_mouth(bmp: &mut Bitmap, mouth: &crate::Mouth) {
    let size = bmp.size as f64;
    // The mouth opens *downward* into a crescent: the gap below the lip line is widest in
    // the middle and tapers to nothing at the corners (`1 - t²`), so the bottom edge is a
    // clean arc bulging down rather than two straight sides growing taller.
    let open_depth = mouth.openness * MOUTH_OPEN_DEPTH;
    let x_start = ((MOUTH_CENTER_X - MOUTH_HALF_WIDTH) * size)
        .floor()
        .max(0.0) as usize;
    let x_end = (((MOUTH_CENTER_X + MOUTH_HALF_WIDTH) * size).ceil() as usize).min(bmp.size - 1);

    for px in x_start..=x_end {
        let xn = (px as f64 + 0.5) / size;
        let t = (xn - MOUTH_CENTER_X) / MOUTH_HALF_WIDTH;
        if t.abs() > 1.0 {
            continue;
        }
        let taper = 1.0 - t * t;
        // The lip line: smile (curve > 0) dips the middle lower than the ends (a ∪); frown
        // inverts it. The top edge is the lip line; the bottom edge drops by `open_depth`
        // in the middle, pinching back to the lip line at the corners.
        let lip = MOUTH_BASE_Y + mouth.curve * MOUTH_AMPLITUDE * taper;
        let top = lip - MOUTH_LINE_HALF;
        let bottom = lip + MOUTH_LINE_HALF + open_depth * taper;
        let y0 = ((top * size).floor()).max(0.0) as usize;
        let y1 = (((bottom * size).ceil()) as usize).min(bmp.size - 1);
        for py in y0..=y1 {
            bmp.set(px, py);
        }
    }
}

/// Fill a vertical capsule (a "stadium": a rectangle capped by semicircles top and
/// bottom) centered at normalized `(cxn, cyn)`, half-width `hwn`, half-height `hhn`.
/// Expects `hhn >= hwn`.
fn fill_capsule_v(bmp: &mut Bitmap, cxn: f64, cyn: f64, hwn: f64, hhn: f64) {
    let size = bmp.size as f64;
    let cx = cxn * size;
    let cy = cyn * size;
    let hw = (hwn * size).max(0.5);
    let hh = (hhn * size).max(hw);
    let cap_offset = hh - hw; // distance from center to each semicircle's center

    let x0 = ((cx - hw).floor()).max(0.0) as usize;
    let x1 = (((cx + hw).ceil()) as usize).min(bmp.size - 1);
    let y0 = ((cy - hh).floor()).max(0.0) as usize;
    let y1 = (((cy + hh).ceil()) as usize).min(bmp.size - 1);

    for py in y0..=y1 {
        for px in x0..=x1 {
            let dx = px as f64 + 0.5 - cx;
            let dy = py as f64 + 0.5 - cy;
            let in_rect = dx.abs() <= hw && dy.abs() <= cap_offset;
            let in_top = dx * dx + (dy + cap_offset) * (dy + cap_offset) <= hw * hw;
            let in_bottom = dx * dx + (dy - cap_offset) * (dy - cap_offset) <= hw * hw;
            if in_rect || in_top || in_bottom {
                bmp.set(px, py);
            }
        }
    }
}

/// Fill an axis-aligned ellipse centered at normalized `(cxn, cyn)` with normalized
/// radii `(rxn, ryn)`. Radii are floored at half a pixel so thin shapes still draw.
fn fill_ellipse(bmp: &mut Bitmap, cxn: f64, cyn: f64, rxn: f64, ryn: f64) {
    let size = bmp.size as f64;
    let cx = cxn * size;
    let cy = cyn * size;
    let rx = (rxn * size).max(0.5);
    let ry = (ryn * size).max(0.5);

    let x0 = ((cx - rx).floor()).max(0.0) as usize;
    let x1 = (((cx + rx).ceil()) as usize).min(bmp.size - 1);
    let y0 = ((cy - ry).floor()).max(0.0) as usize;
    let y1 = (((cy + ry).ceil()) as usize).min(bmp.size - 1);

    for py in y0..=y1 {
        for px in x0..=x1 {
            let dx = (px as f64 + 0.5 - cx) / rx;
            let dy = (py as f64 + 0.5 - cy) / ry;
            if dx * dx + dy * dy <= 1.0 {
                bmp.set(px, py);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{face, Ears, Eye, Mouth};

    const NEUTRAL_EARS: Ears = Ears { perk: 0.0 };
    use jaxson_core::MoodVector;

    fn symmetric_eye() -> Eye {
        Eye {
            openness: 1.0,
            pupil_dx: 0.0,
            pupil_dy: 0.0,
        }
    }

    #[test]
    fn renders_some_ink() {
        let bmp = rasterize(&face(MoodVector::NEUTRAL, 1.0), 64);
        assert!(bmp.ink() > 0);
        assert_eq!(bmp.size(), 64);
    }

    #[test]
    fn is_left_right_symmetric_without_gaze() {
        let f = Face {
            left_eye: symmetric_eye(),
            right_eye: symmetric_eye(),
            mouth: Mouth {
                curve: 0.6,
                openness: 0.2,
            },
            ears: NEUTRAL_EARS,
        };
        let bmp = rasterize(&f, 64);
        for y in 0..bmp.size() {
            for x in 0..bmp.size() {
                assert_eq!(
                    bmp.get(x, y),
                    bmp.get(bmp.size() - 1 - x, y),
                    "asymmetry at ({x}, {y})"
                );
            }
        }
    }

    #[test]
    fn blinking_eyes_have_far_less_ink_than_open_eyes() {
        let open = Face {
            left_eye: symmetric_eye(),
            right_eye: symmetric_eye(),
            mouth: Mouth {
                curve: 0.0,
                openness: 0.0,
            },
            ears: NEUTRAL_EARS,
        };
        let mut shut = open;
        shut.left_eye.openness = 0.0;
        shut.right_eye.openness = 0.0;
        assert!(rasterize(&shut, 64).ink() < rasterize(&open, 64).ink());
    }

    #[test]
    fn smile_and_frown_produce_different_images() {
        let happy = rasterize(&face(MoodVector::new(0.9, 0.2), 1.0), 64);
        let sad = rasterize(&face(MoodVector::new(-0.9, 0.2), 1.0), 64);
        assert_ne!(happy, sad);
    }

    #[test]
    fn smile_curves_below_its_corners() {
        // For a smile, the mouth's lowest inked row at the center is below (greater y)
        // than at the corners.
        let f = Face {
            left_eye: symmetric_eye(),
            right_eye: symmetric_eye(),
            mouth: Mouth {
                curve: 1.0,
                openness: 0.0,
            },
            ears: NEUTRAL_EARS,
        };
        let bmp = rasterize(&f, 80);
        let lowest = |x: usize| (0..bmp.size()).filter(|&y| bmp.get(x, y)).max();
        let center = lowest(bmp.size() / 2).unwrap();
        // Sample near the mouth's corner (within its narrower width).
        let corner = lowest((0.44 * bmp.size() as f64) as usize).unwrap();
        assert!(center > corner);
    }

    #[test]
    fn open_mouth_is_a_downward_crescent() {
        // A speaking (open) mouth with a neutral curve: it should open *downward* into a
        // crescent — the bottom edge dips lowest in the middle and pinches back up at the
        // corners — and ink more than a closed mouth.
        let open = Face {
            left_eye: symmetric_eye(),
            right_eye: symmetric_eye(),
            mouth: Mouth {
                curve: 0.0,
                openness: 0.6,
            },
            ears: NEUTRAL_EARS,
        };
        let mut closed = open;
        closed.mouth.openness = 0.0;

        let bmp = rasterize(&open, 80);
        let lowest = |x: usize| (0..bmp.size()).filter(|&y| bmp.get(x, y)).max();
        let center = lowest(bmp.size() / 2).unwrap();
        let corner = lowest((0.43 * bmp.size() as f64) as usize).unwrap();
        // Bottom edge bulges down (convex): center is below the corners.
        assert!(
            center > corner,
            "center {center} should be below corner {corner}"
        );
        // Opening the mouth adds ink versus the closed line.
        assert!(rasterize(&open, 80).ink() > rasterize(&closed, 80).ink());
    }

    #[test]
    fn ascii_has_half_height_rows_and_full_width() {
        let bmp = rasterize(&face(MoodVector::NEUTRAL, 1.0), 64);
        let ascii = bmp.to_ascii();
        let lines: Vec<&str> = ascii.lines().collect();
        assert_eq!(lines.len(), 32); // 64 rows sampled every 2
        assert!(lines.iter().all(|l| l.chars().count() == 64));
    }
}
