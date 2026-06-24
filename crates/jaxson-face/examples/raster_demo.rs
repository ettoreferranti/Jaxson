//! Renders the face for a few moods as ASCII, so the look can be checked without a GUI.
//!
//! ```text
//! cargo run --example raster_demo -p jaxson-face
//! ```

use jaxson_core::MoodVector;
use jaxson_face::{face, rasterize, Bitmap};
use jaxson_face::{Ears, Eye, Face, Mouth};

fn show(label: &str, bmp: &Bitmap) {
    println!("== {label} ==");
    print!("{}", bmp.to_ascii());
    println!();
}

fn main() {
    let size = 72;
    // t = 1.0 is between blinks, so eyes are open.
    show(
        "happy (v+0.9, a+0.4)",
        &rasterize(&face(MoodVector::new(0.9, 0.4), 1.0), size),
    );
    show(
        "content (v+0.6, a-0.4)",
        &rasterize(&face(MoodVector::new(0.6, -0.4), 1.0), size),
    );
    show("neutral", &rasterize(&face(MoodVector::NEUTRAL, 1.0), size));
    show(
        "sad (v-0.8, a-0.5)",
        &rasterize(&face(MoodVector::new(-0.8, -0.5), 1.0), size),
    );
    show(
        "upset (v-0.7, a+0.7)",
        &rasterize(&face(MoodVector::new(-0.7, 0.7), 1.0), size),
    );

    // A blink: same happy mood, sampled at the middle of a blink.
    let blinking = Face {
        left_eye: Eye {
            openness: 0.0,
            pupil_dx: 0.0,
            pupil_dy: 0.0,
        },
        right_eye: Eye {
            openness: 0.0,
            pupil_dx: 0.0,
            pupil_dy: 0.0,
        },
        mouth: Mouth {
            curve: 0.9,
            openness: 0.2,
        },
        ears: Ears { perk: 0.8 },
    };
    show("happy mid-blink", &rasterize(&blinking, size));
}
