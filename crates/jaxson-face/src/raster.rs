//! A tiny pure software rasterizer: turn a [`Face`] into a square black-and-white
//! [`Bitmap`] — two eyes and a mouth, nothing else. No GUI, so the look of the face can
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

    // Eyes (mirrored around the vertical centerline).
    draw_eye(&mut bmp, 0.34, &face.left_eye);
    draw_eye(&mut bmp, 0.66, &face.right_eye);

    // Mouth.
    draw_mouth(&mut bmp, &face.mouth);

    bmp
}

const EYE_CENTER_Y: f64 = 0.40;
const EYE_RX: f64 = 0.12;
const EYE_RY: f64 = 0.12;
const GAZE_SHIFT: f64 = 0.05;

fn draw_eye(bmp: &mut Bitmap, center_x: f64, eye: &crate::Eye) {
    // Gaze drifts the whole eye a little; blinking squashes it vertically. A minimum
    // height keeps a closed eye as a visible line rather than vanishing.
    let cx = center_x + eye.pupil_dx * GAZE_SHIFT;
    let cy = EYE_CENTER_Y + eye.pupil_dy * GAZE_SHIFT;
    let ry = EYE_RY * eye.openness.max(0.06);
    fill_ellipse(bmp, cx, cy, EYE_RX, ry);
}

const MOUTH_CENTER_X: f64 = 0.5;
const MOUTH_BASE_Y: f64 = 0.62;
const MOUTH_HALF_WIDTH: f64 = 0.22;
const MOUTH_AMPLITUDE: f64 = 0.08;

fn draw_mouth(bmp: &mut Bitmap, mouth: &crate::Mouth) {
    let size = bmp.size as f64;
    let half_thickness = 0.011 + mouth.openness * 0.035;
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
        // Smile (curve > 0) dips the middle lower than the ends (a ∪); frown inverts it.
        let yn = MOUTH_BASE_Y + mouth.curve * MOUTH_AMPLITUDE * (1.0 - t * t);
        let y0 = (((yn - half_thickness) * size).floor()).max(0.0) as usize;
        let y1 = ((((yn + half_thickness) * size).ceil()) as usize).min(bmp.size - 1);
        for py in y0..=y1 {
            bmp.set(px, py);
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
    use crate::{face, Eye, Mouth};
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
        };
        let bmp = rasterize(&f, 80);
        let lowest = |x: usize| (0..bmp.size()).filter(|&y| bmp.get(x, y)).max();
        let center = lowest(bmp.size() / 2).unwrap();
        let corner = lowest((0.33 * bmp.size() as f64) as usize).unwrap();
        assert!(center > corner);
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
