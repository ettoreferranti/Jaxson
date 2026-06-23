use serde::{Deserialize, Serialize};

/// Decoding parameters for a single generation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenerationConfig {
    /// Maximum number of tokens to generate.
    pub max_tokens: usize,
    /// Sampling temperature; `0.0` is greedy. Higher is more random.
    pub temperature: f32,
    /// Nucleus-sampling probability mass, in `[0.0, 1.0]`.
    pub top_p: f32,
    /// Optional RNG seed for reproducible output.
    pub seed: Option<u32>,
    /// Strings that, once produced, stop generation.
    pub stop: Vec<String>,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        GenerationConfig {
            max_tokens: 512,
            temperature: 0.7,
            top_p: 0.95,
            seed: None,
            stop: Vec::new(),
        }
    }
}

impl GenerationConfig {
    /// Return a copy with values forced into valid ranges: at least one token,
    /// non-negative temperature, and `top_p` clamped to `[0.0, 1.0]`.
    pub fn validated(mut self) -> Self {
        self.max_tokens = self.max_tokens.max(1);
        self.temperature = self.temperature.max(0.0);
        self.top_p = self.top_p.clamp(0.0, 1.0);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_sane() {
        let c = GenerationConfig::default();
        assert_eq!(c.max_tokens, 512);
        assert!(c.temperature > 0.0);
        assert!((0.0..=1.0).contains(&c.top_p));
        assert!(c.seed.is_none());
        assert!(c.stop.is_empty());
    }

    #[test]
    fn validated_enforces_minimum_one_token() {
        let c = GenerationConfig {
            max_tokens: 0,
            ..Default::default()
        }
        .validated();
        assert_eq!(c.max_tokens, 1);
    }

    #[test]
    fn validated_clamps_negative_temperature_to_zero() {
        let c = GenerationConfig {
            temperature: -3.0,
            ..Default::default()
        }
        .validated();
        assert_eq!(c.temperature, 0.0);
    }

    #[test]
    fn validated_clamps_top_p_into_unit_range() {
        let high = GenerationConfig {
            top_p: 5.0,
            ..Default::default()
        }
        .validated();
        assert_eq!(high.top_p, 1.0);
        let low = GenerationConfig {
            top_p: -1.0,
            ..Default::default()
        }
        .validated();
        assert_eq!(low.top_p, 0.0);
    }

    #[test]
    fn validated_leaves_valid_values_untouched() {
        let c = GenerationConfig {
            max_tokens: 64,
            temperature: 0.5,
            top_p: 0.9,
            seed: Some(42),
            stop: vec!["<|im_end|>".to_string()],
        };
        assert_eq!(c.clone().validated(), c);
    }
}
