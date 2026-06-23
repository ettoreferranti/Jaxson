//! Jaxson's desktop shell: an egui window showing the animated face above a chat box.
//!
//! The face is the `jaxson-face` rasterizer's bitmap, refreshed every frame so it
//! blinks and drifts, and its expression follows the agent's mood. The chat is wired to
//! a mock-backed [`Agent`]: replies are canned (no model yet — that's `--features
//! llama`), but the *face reacts live* to the sentiment of whatever you type.
//!
//! Run: `cargo run --manifest-path crates/jaxson-app/Cargo.toml`

use std::time::Instant;

use eframe::egui;
use egui::Color32;

use jaxson_agent::{Agent, HashEmbedder};
use jaxson_core::MoodVector;
use jaxson_face::{face, rasterize, Bitmap};
use jaxson_llm::{GenerationConfig, LlmError, TextGenerator};

const PERSONA: &str = "You are Jaxson, a warm, curious companion getting to know its owner.";
const FACE_PIXELS: usize = 250;

/// A stand-in "model" for the UI: returns empty extractions and a canned chat reply,
/// chosen by sniffing the prompt. Lets the loop run without a real model.
struct DemoModel;

impl TextGenerator for DemoModel {
    fn generate(
        &mut self,
        prompt: &str,
        _config: &GenerationConfig,
        on_token: &mut dyn FnMut(&str),
    ) -> Result<String, LlmError> {
        let reply = if prompt.contains("Extract memories as JSON") {
            r#"{"memories":[],"relations":[]}"#.to_string()
        } else {
            "I'm a demo brain for now (no model wired yet) — but watch my face react to \
             how you say things!"
                .to_string()
        };
        on_token(&reply);
        Ok(reply)
    }
}

struct JaxsonApp {
    agent: Agent,
    model: DemoModel,
    embedder: HashEmbedder,
    transcript: Vec<(&'static str, String)>,
    input: String,
    start: Instant,
    turn: i64,
    face_tex: egui::TextureHandle,
}

impl JaxsonApp {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let neutral = to_image(&rasterize(&face(MoodVector::NEUTRAL, 0.0), FACE_PIXELS));
        let face_tex =
            cc.egui_ctx
                .load_texture("jaxson-face", neutral, egui::TextureOptions::NEAREST);
        JaxsonApp {
            agent: Agent::new(PERSONA),
            model: DemoModel,
            embedder: HashEmbedder::default(),
            transcript: vec![("Jaxson", "Hi! I'm Jaxson. What's your name?".to_string())],
            input: String::new(),
            start: Instant::now(),
            turn: 0,
            face_tex,
        }
    }

    fn send(&mut self) {
        let input = std::mem::take(&mut self.input);
        let input = input.trim().to_string();
        if input.is_empty() {
            return;
        }
        self.transcript.push(("You", input.clone()));
        self.turn += 1;
        match self
            .agent
            .respond(&mut self.model, &self.embedder, self.turn, &input)
        {
            Ok(turn) => self.transcript.push(("Jaxson", turn.reply)),
            Err(e) => self.transcript.push(("Jaxson", format!("(error: {e})"))),
        }
    }
}

impl eframe::App for JaxsonApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Refresh the face every frame so it animates (blink/gaze) and reflects mood.
        let t = self.start.elapsed().as_secs_f64();
        let bitmap = rasterize(&face(self.agent.mood(), t), FACE_PIXELS);
        self.face_tex
            .set(to_image(&bitmap), egui::TextureOptions::NEAREST);

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.add_space(8.0);
            ui.vertical_centered(|ui| {
                ui.image(egui::load::SizedTexture::new(
                    self.face_tex.id(),
                    egui::vec2(250.0, 250.0),
                ));
                ui.label(format!("mood: {:?}", self.agent.mood().dominant_emotion()));
            });

            ui.separator();

            egui::ScrollArea::vertical()
                .auto_shrink([false, false])
                .max_height(180.0)
                .stick_to_bottom(true)
                .show(ui, |ui| {
                    for (who, text) in &self.transcript {
                        ui.label(format!("{who}: {text}"));
                    }
                });

            ui.separator();

            ui.horizontal(|ui| {
                let response = ui.text_edit_singleline(&mut self.input);
                let entered =
                    response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
                if ui.button("Send").clicked() || entered {
                    self.send();
                    ui.memory_mut(|m| m.request_focus(response.id));
                }
            });
        });

        // Keep animating between interactions.
        ctx.request_repaint();
    }
}

/// Convert a 1-bit [`Bitmap`] into an egui image: ink = black, background = white.
fn to_image(bitmap: &Bitmap) -> egui::ColorImage {
    let n = bitmap.size();
    let mut pixels = Vec::with_capacity(n * n);
    for y in 0..n {
        for x in 0..n {
            pixels.push(if bitmap.get(x, y) {
                Color32::BLACK
            } else {
                Color32::WHITE
            });
        }
    }
    egui::ColorImage {
        size: [n, n],
        pixels,
    }
}

fn main() -> eframe::Result<()> {
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([360.0, 560.0])
            .with_title("Jaxson"),
        ..Default::default()
    };
    eframe::run_native(
        "Jaxson",
        native_options,
        Box::new(|cc| Ok(Box::new(JaxsonApp::new(cc)))),
    )
}
