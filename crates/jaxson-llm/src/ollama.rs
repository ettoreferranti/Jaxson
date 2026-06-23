//! Discover models already installed via Ollama, so it can serve as a model manager.
//!
//! Ollama stores GGUF weights as content-addressed blobs plus small JSON manifests.
//! We read the manifests to list models and resolve each one's GGUF blob path. The
//! manifest parsing and name derivation are pure (and tested); discovery is a thin
//! filesystem walk.

use std::path::{Path, PathBuf};

use serde::Deserialize;

/// An Ollama-managed model and the GGUF file backing it.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct OllamaModel {
    /// Display name, e.g. `"llama3.1:8b"`.
    pub name: String,
    /// Path to the GGUF blob (loadable by the `llama` backend).
    pub path: PathBuf,
}

const MODEL_MEDIA_TYPE: &str = "application/vnd.ollama.image.model";

#[derive(Deserialize)]
struct Manifest {
    #[serde(default)]
    layers: Vec<Layer>,
}

#[derive(Deserialize)]
struct Layer {
    #[serde(rename = "mediaType")]
    media_type: String,
    digest: String,
}

/// Extract the GGUF weights-blob digest (e.g. `"sha256:abc…"`) from a manifest's JSON,
/// or `None` if it isn't a model manifest.
pub fn model_digest(manifest_json: &str) -> Option<String> {
    let manifest: Manifest = serde_json::from_str(manifest_json).ok()?;
    manifest
        .layers
        .into_iter()
        .find(|layer| layer.media_type == MODEL_MEDIA_TYPE)
        .map(|layer| layer.digest)
}

/// Turn a manifest path's components (relative to `manifests/`) into the model name
/// Ollama displays — e.g. `["registry.ollama.ai","library","llama3.1","8b"]` →
/// `"llama3.1:8b"`, and `["registry.ollama.ai","ns","model","tag"]` → `"ns/model:tag"`.
pub fn model_name(components: &[String]) -> Option<String> {
    // Drop the registry host.
    let parts = match components.split_first() {
        Some((_host, rest)) => rest,
        None => return None,
    };
    // Drop a leading `library` namespace (Ollama hides it).
    let parts: &[String] = match parts.split_first() {
        Some((first, rest)) if first == "library" => rest,
        _ => parts,
    };
    let (tag, name_parts) = parts.split_last()?;
    if name_parts.is_empty() {
        return None;
    }
    Some(format!("{}:{}", name_parts.join("/"), tag))
}

fn blob_path(root: &Path, digest: &str) -> PathBuf {
    root.join("blobs").join(digest.replace(':', "-"))
}

/// Discover models under an Ollama models `root` (the directory containing `manifests/`
/// and `blobs/`). Only models whose blob actually exists are returned, sorted by name.
pub fn discover_in(root: &Path) -> Vec<OllamaModel> {
    let manifests_root = root.join("manifests");
    let mut found = Vec::new();
    let mut stack = vec![manifests_root.clone()];

    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
                continue;
            }
            let Ok(json) = std::fs::read_to_string(&path) else {
                continue;
            };
            let Some(digest) = model_digest(&json) else {
                continue;
            };
            let blob = blob_path(root, &digest);
            if !blob.is_file() {
                continue;
            }
            let relative = path.strip_prefix(&manifests_root).unwrap_or(&path);
            let components: Vec<String> = relative
                .components()
                .filter_map(|c| c.as_os_str().to_str().map(str::to_string))
                .collect();
            if let Some(name) = model_name(&components) {
                found.push(OllamaModel { name, path: blob });
            }
        }
    }

    found.sort();
    found.dedup();
    found
}

/// Discover from the default location: `$OLLAMA_MODELS`, else `~/.ollama/models`.
pub fn discover() -> Vec<OllamaModel> {
    let root = root_from(
        std::env::var("OLLAMA_MODELS").ok(),
        std::env::var("HOME").ok(),
    );
    discover_in(&root)
}

/// Resolve the Ollama models root from the relevant env values (pure).
fn root_from(ollama_models: Option<String>, home: Option<String>) -> PathBuf {
    if let Some(custom) = ollama_models {
        return PathBuf::from(custom);
    }
    PathBuf::from(home.unwrap_or_default())
        .join(".ollama")
        .join("models")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    const MANIFEST: &str = r#"{
        "schemaVersion": 2,
        "layers": [
            {"mediaType": "application/vnd.ollama.image.template", "digest": "sha256:tpl", "size": 1},
            {"mediaType": "application/vnd.ollama.image.model", "digest": "sha256:weights", "size": 42}
        ]
    }"#;

    #[test]
    fn model_digest_finds_the_weights_layer() {
        assert_eq!(model_digest(MANIFEST), Some("sha256:weights".to_string()));
    }

    #[test]
    fn model_digest_none_without_a_model_layer() {
        let json = r#"{"layers":[{"mediaType":"application/vnd.ollama.image.template","digest":"sha256:x"}]}"#;
        assert_eq!(model_digest(json), None);
    }

    #[test]
    fn model_digest_none_on_garbage() {
        assert_eq!(model_digest("not json"), None);
    }

    #[test]
    fn model_name_drops_registry_and_library() {
        let c = vec![
            "registry.ollama.ai".to_string(),
            "library".to_string(),
            "llama3.1".to_string(),
            "8b".to_string(),
        ];
        assert_eq!(model_name(&c), Some("llama3.1:8b".to_string()));
    }

    #[test]
    fn model_name_keeps_custom_namespaces() {
        let c = vec![
            "registry.ollama.ai".to_string(),
            "MichelRosselli".to_string(),
            "apertus".to_string(),
            "8b-instruct".to_string(),
        ];
        assert_eq!(
            model_name(&c),
            Some("MichelRosselli/apertus:8b-instruct".to_string())
        );
    }

    #[test]
    fn model_name_rejects_too_few_components() {
        assert_eq!(model_name(&["registry".to_string()]), None);
        assert_eq!(model_name(&[]), None);
    }

    #[test]
    fn discover_lists_models_with_existing_blobs() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        // Manifest for library/llama3.1/8b -> blob sha256:weights.
        let man_dir = root.join("manifests/registry.ollama.ai/library/llama3.1");
        fs::create_dir_all(&man_dir).unwrap();
        fs::write(man_dir.join("8b"), MANIFEST).unwrap();
        // The referenced blob exists.
        let blobs = root.join("blobs");
        fs::create_dir_all(&blobs).unwrap();
        fs::write(blobs.join("sha256-weights"), b"GGUF...").unwrap();

        let models = discover_in(root);
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].name, "llama3.1:8b");
        assert_eq!(models[0].path, blobs.join("sha256-weights"));
    }

    #[test]
    fn discover_skips_models_whose_blob_is_missing() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        let man_dir = root.join("manifests/registry.ollama.ai/library/ghost");
        fs::create_dir_all(&man_dir).unwrap();
        fs::write(man_dir.join("latest"), MANIFEST).unwrap();
        // No blob written.
        assert!(discover_in(root).is_empty());
    }

    #[test]
    fn discover_on_missing_root_is_empty() {
        let dir = tempfile::tempdir().unwrap();
        assert!(discover_in(&dir.path().join("nope")).is_empty());
    }

    #[test]
    fn root_from_prefers_ollama_models_then_home() {
        assert_eq!(
            root_from(Some("/models".to_string()), Some("/home".to_string())),
            PathBuf::from("/models")
        );
        assert_eq!(
            root_from(None, Some("/home/me".to_string())),
            PathBuf::from("/home/me/.ollama/models")
        );
        assert_eq!(root_from(None, None), PathBuf::from(".ollama/models"));
    }

    #[test]
    fn discover_reads_the_root_from_env() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        let man_dir = root.join("manifests/registry.ollama.ai/library/llama3.1");
        fs::create_dir_all(&man_dir).unwrap();
        fs::write(man_dir.join("8b"), MANIFEST).unwrap();
        let blobs = root.join("blobs");
        fs::create_dir_all(&blobs).unwrap();
        fs::write(blobs.join("sha256-weights"), b"GGUF...").unwrap();

        // Point the default-location resolver at our fixture. (Only this test touches
        // OLLAMA_MODELS, so there's no cross-test interference.)
        std::env::set_var("OLLAMA_MODELS", root);
        let models = discover();
        std::env::remove_var("OLLAMA_MODELS");

        assert!(models.iter().any(|m| m.name == "llama3.1:8b"));
    }
}
