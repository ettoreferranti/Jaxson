//! A scripted walk-through of Jaxson's relationship state machine.
//!
//! This isn't a test — it's a way to *see* the core behave. It feeds a fake
//! conversation (the kind of events `jaxson-memory` and `jaxson-affect` will emit at
//! runtime) into [`RelationshipState`] and prints how trust, familiarity, mood, and
//! the gated behaviors evolve step by step.
//!
//! Run it with:
//!
//! ```text
//! cargo run --example state_machine_demo -p jaxson-core
//! ```

use jaxson_core::{RelationshipEvent, RelationshipState};

/// One scripted beat: a human-readable note plus the event it triggers.
struct Beat {
    note: &'static str,
    event: RelationshipEvent,
}

fn main() {
    let script = [
        Beat {
            note: "Owner says hi warmly — a gentle positive exchange.",
            event: RelationshipEvent::PositiveInteraction { strength: 0.4 },
        },
        Beat {
            note: "Learns a fact: \"My name is Ettore.\"",
            event: RelationshipEvent::LearnedFact,
        },
        Beat {
            note: "Learns a fact: \"I have a dog named Pixel.\"",
            event: RelationshipEvent::LearnedFact,
        },
        Beat {
            note: "A genuinely happy moment — strong positive exchange.",
            event: RelationshipEvent::PositiveInteraction { strength: 0.9 },
        },
        Beat {
            note: "Owner snaps at Jaxson — a negative exchange.",
            event: RelationshipEvent::NegativeInteraction { strength: 0.6 },
        },
        Beat {
            note: "Several more facts learned over a week of chatting.",
            event: RelationshipEvent::LearnedFact,
        },
        Beat {
            note: "Owner deletes a memory in the inspector — it's really forgotten.",
            event: RelationshipEvent::MemoryForgotten,
        },
    ];

    println!("Jaxson — relationship state machine demo\n");

    let mut state = RelationshipState::INITIAL;
    print_state(
        "start",
        "A fresh companion (the \"Ron's defect\" premise).",
        &state,
    );

    for (i, beat) in script.iter().enumerate() {
        state = state.apply(beat.event);
        print_state(&format!("step {}", i + 1), beat.note, &state);
    }

    // To make familiarity climb realistically, simulate many small facts learned.
    state = state.apply_all(std::iter::repeat_n(RelationshipEvent::LearnedFact, 40));
    print_state("later", "After ~40 more learned facts over time.", &state);
}

fn print_state(label: &str, note: &str, s: &RelationshipState) {
    let mood = s.mood();
    println!("[{label}] {note}");
    println!(
        "        trust {:.2}  familiarity {:.2}  mood (v {:+.2}, a {:+.2}) -> {:?}",
        s.trust(),
        s.familiarity(),
        mood.valence(),
        mood.arousal(),
        mood.dominant_emotion(),
    );
    println!(
        "        leads with questions? {:<3}   sensitive topics unlocked? {}",
        yes_no(s.should_prioritize_onboarding()),
        yes_no(s.allows_sensitive_topics()),
    );
    println!();
}

fn yes_no(b: bool) -> &'static str {
    if b {
        "yes"
    } else {
        "no"
    }
}
