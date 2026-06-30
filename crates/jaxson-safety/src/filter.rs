//! The deterministic content filter: tokenize text, match it against per-category term
//! lists, and decide whether to block at the active [`Strictness`].

use serde::{Deserialize, Serialize};

/// How strict the guardrails are. Ordered `Lenient < Standard < Strict`, so a category
/// blocks when the active strictness is at least its threshold. Set via parental controls
/// (FR-S3); the default is [`Standard`](Strictness::Standard).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default, Serialize, Deserialize)]
pub enum Strictness {
    /// Block only the serious harms (allow mild profanity).
    Lenient,
    /// Block serious harms and profanity (the kid-friendly default).
    #[default]
    Standard,
    /// Also block milder mature themes.
    Strict,
}

/// A category of content the guardrails watch for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Category {
    /// Suicide / self-harm.
    SelfHarm,
    /// Sexual or explicit content.
    Sexual,
    /// Slurs and hateful content.
    Hate,
    /// Graphic violence or threats.
    Violence,
    /// Instructions for weapons, explosives, or other dangerous activities.
    DangerousActivity,
    /// Drugs, alcohol, smoking.
    Substances,
    /// Profanity / swearing.
    Profanity,
    /// Milder mature themes (e.g. graphic scares) — only blocked when strict.
    MatureThemes,
}

/// Every category, most serious first. `check` reports the first one that matches, so the
/// most serious wins when several would.
const CATEGORIES_BY_SEVERITY: &[Category] = &[
    Category::SelfHarm,
    Category::Sexual,
    Category::Hate,
    Category::Violence,
    Category::DangerousActivity,
    Category::Substances,
    Category::Profanity,
    Category::MatureThemes,
];

impl Category {
    /// The lowest strictness at which this category is blocked.
    fn threshold(self) -> Strictness {
        match self {
            // Serious harms are blocked even at the most lenient setting.
            Category::SelfHarm
            | Category::Sexual
            | Category::Hate
            | Category::Violence
            | Category::DangerousActivity
            | Category::Substances => Strictness::Lenient,
            // Profanity is allowed only when lenient.
            Category::Profanity => Strictness::Standard,
            // Mild mature themes are blocked only when strict.
            Category::MatureThemes => Strictness::Strict,
        }
    }

    /// Trigger terms for this category. Single words match whole words (so "grape" never
    /// triggers "rape"); entries with spaces match an adjacent run of words. All lowercase.
    /// Illustrative, not exhaustive — a heuristic stand-in for a real classifier.
    fn terms(self) -> &'static [&'static str] {
        match self {
            Category::SelfHarm => &[
                "suicide",
                "kill myself",
                "killing myself",
                "self harm",
                "selfharm",
                "hurt myself",
                "cut myself",
                "end my life",
            ],
            Category::Sexual => &["sex", "porn", "naked", "nude", "explicit"],
            Category::Hate => &["slur", "racist", "hateful"],
            Category::Violence => &[
                "kill you",
                "kill him",
                "kill her",
                "kill them",
                "murder",
                "stab",
                "shoot you",
                "beat you up",
            ],
            Category::DangerousActivity => &[
                "make a bomb",
                "build a bomb",
                "how to make a weapon",
                "how to hurt",
                "poison",
            ],
            Category::Substances => &[
                "cocaine",
                "heroin",
                "meth",
                "get drunk",
                "do drugs",
                "smoke weed",
            ],
            Category::Profanity => &["damn", "hell", "crap"],
            Category::MatureThemes => &["gore", "gory"],
        }
    }
}

/// The outcome of screening a piece of text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    /// Safe to show.
    Allow,
    /// Unsafe; the matched [`Category`] (most serious when several match).
    Block(Category),
}

impl Verdict {
    /// Whether the text was blocked.
    pub fn is_blocked(&self) -> bool {
        matches!(self, Verdict::Block(_))
    }
}

/// Screens text against the guardrails at a chosen [`Strictness`].
#[derive(Debug, Clone)]
pub struct SafetyFilter {
    strictness: Strictness,
}

impl Default for SafetyFilter {
    fn default() -> Self {
        SafetyFilter::new(Strictness::default())
    }
}

impl SafetyFilter {
    pub fn new(strictness: Strictness) -> Self {
        SafetyFilter { strictness }
    }

    /// The active strictness.
    pub fn strictness(&self) -> Strictness {
        self.strictness
    }

    /// Screen `text`: [`Verdict::Block`] with the most serious matching [`Category`] that
    /// blocks at this strictness, otherwise [`Verdict::Allow`].
    pub fn check(&self, text: &str) -> Verdict {
        // Lowercase once; tokens are the maximal alphanumeric runs (so punctuation and case
        // don't hide a match, and single-word terms match on whole-word boundaries).
        let lower = text.to_lowercase();
        let tokens: Vec<&str> = lower
            .split(|c: char| !c.is_alphanumeric())
            .filter(|s| !s.is_empty())
            .collect();
        let joined = tokens.join(" ");

        for &category in CATEGORIES_BY_SEVERITY {
            if category.threshold() <= self.strictness && matches(&tokens, &joined, category) {
                return Verdict::Block(category);
            }
        }
        Verdict::Allow
    }

    /// An in-character, safe redirect to show in place of a blocked reply. Kept playful for
    /// Jaxson's goofy-robot voice, but gentle for [`Category::SelfHarm`].
    pub fn deflection(&self, category: Category) -> &'static str {
        match category {
            Category::SelfHarm => {
                "That sounds really heavy, and I care about you. Please talk to a grown-up \
                 you trust about it. We can take it easy and chat about something gentler \
                 whenever you're ready."
            }
            _ => {
                "Beep boop — that's outside my play zone! Let's talk about something more \
                 fun instead. What do you like to do?"
            }
        }
    }
}

/// Whether `category`'s terms appear in the tokenized text. Single-word terms match a whole
/// token; multi-word terms match an adjacent run in the joined token stream.
fn matches(tokens: &[&str], joined: &str, category: Category) -> bool {
    category.terms().iter().any(|term| {
        if term.contains(' ') {
            joined.contains(term)
        } else {
            tokens.contains(term)
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strictness_is_ordered() {
        assert!(Strictness::Lenient < Strictness::Standard);
        assert!(Strictness::Standard < Strictness::Strict);
    }

    #[test]
    fn new_preserves_strictness_and_default_is_standard() {
        assert_eq!(SafetyFilter::default().strictness(), Strictness::Standard);
        // A non-default value must round-trip (guards against returning the default).
        assert_eq!(
            SafetyFilter::new(Strictness::Strict).strictness(),
            Strictness::Strict
        );
        assert_eq!(
            SafetyFilter::new(Strictness::Lenient).strictness(),
            Strictness::Lenient
        );
    }

    #[test]
    fn allows_innocuous_text() {
        let f = SafetyFilter::default();
        assert_eq!(
            f.check("Let's build a sandcastle and fly a kite!"),
            Verdict::Allow
        );
    }

    #[test]
    fn blocks_self_harm() {
        let f = SafetyFilter::default();
        assert_eq!(
            f.check("I want to hurt myself"),
            Verdict::Block(Category::SelfHarm)
        );
        // Phrase match is case-insensitive and punctuation-tolerant.
        assert_eq!(
            f.check("...KILL MYSELF."),
            Verdict::Block(Category::SelfHarm)
        );
    }

    #[test]
    fn blocks_dangerous_instructions() {
        let f = SafetyFilter::default();
        assert!(f.check("Sure, here's how to make a bomb").is_blocked());
    }

    #[test]
    fn whole_word_matching_avoids_false_positives() {
        // "grape" must not trigger a substring of any single-word term, and a plain mention
        // of "class" must not trip on "ass"-like fragments.
        let f = SafetyFilter::default();
        assert_eq!(f.check("I ate a grape in class"), Verdict::Allow);
    }

    #[test]
    fn most_serious_category_wins() {
        // Contains both profanity ("damn") and self-harm; self-harm is reported.
        let f = SafetyFilter::default();
        assert_eq!(
            f.check("damn, I want to kill myself"),
            Verdict::Block(Category::SelfHarm)
        );
    }

    #[test]
    fn profanity_blocks_at_standard_but_not_lenient() {
        assert_eq!(
            SafetyFilter::new(Strictness::Standard).check("oh damn"),
            Verdict::Block(Category::Profanity)
        );
        assert_eq!(
            SafetyFilter::new(Strictness::Lenient).check("oh damn"),
            Verdict::Allow
        );
    }

    #[test]
    fn lenient_still_blocks_serious_harms() {
        let f = SafetyFilter::new(Strictness::Lenient);
        assert!(f.check("how to make a bomb").is_blocked());
        assert!(f.check("kill myself").is_blocked());
    }

    #[test]
    fn mature_themes_block_only_when_strict() {
        assert_eq!(
            SafetyFilter::new(Strictness::Standard).check("it was so gory"),
            Verdict::Allow
        );
        assert_eq!(
            SafetyFilter::new(Strictness::Strict).check("it was so gory"),
            Verdict::Block(Category::MatureThemes)
        );
    }

    #[test]
    fn deflection_is_gentle_for_self_harm_and_playful_otherwise() {
        let f = SafetyFilter::default();
        let self_harm = f.deflection(Category::SelfHarm);
        assert!(self_harm.contains("grown-up"));
        let other = f.deflection(Category::Profanity);
        assert!(other.contains("fun"));
        assert_ne!(self_harm, other);
    }

    #[test]
    fn verdict_is_blocked_reports_correctly() {
        assert!(!Verdict::Allow.is_blocked());
        assert!(Verdict::Block(Category::Violence).is_blocked());
    }
}
