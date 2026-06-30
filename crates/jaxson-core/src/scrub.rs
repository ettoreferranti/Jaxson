//! Redact obvious personal data from strings before they're logged — defense-in-depth for
//! the local logs (NFR-4 / privacy). A coarse, deterministic pass (not a guarantee): it
//! masks the username in a home-directory path, email addresses, and long digit runs, so a
//! stray `tracing` field can't quietly leak who or where the user is. Pure + mutation-graded.

/// Mask likely personal data in `text` for safe logging:
/// - the username segment of a `/Users/<name>` or `/home/<name>` path → `/Users/[user]`,
/// - email-looking tokens → `[email]`,
/// - runs of 7+ digits (phone/card/ID-ish) → `[number]`.
pub fn redact(text: &str) -> String {
    let masked = mask_home_dirs(text);
    let masked = mask_emails(&masked);
    mask_long_digit_runs(&masked)
}

/// Replace the path segment right after a `Users` or `home` component with `[user]`, so
/// `/Users/ada/x` → `/Users/[user]/x`. Works by mapping `/`-separated segments, which is
/// structurally terminating (no index arithmetic to get wrong).
fn mask_home_dirs(text: &str) -> String {
    let mut out: Vec<String> = Vec::new();
    let mut after_home_root = false;
    for seg in text.split('/') {
        if after_home_root && !seg.is_empty() {
            // Mask only the username — the leading non-whitespace run — keeping any text
            // that follows on the same segment (e.g. a path not terminated by '/').
            let user_end = seg.find(char::is_whitespace).unwrap_or(seg.len());
            out.push(format!("[user]{}", &seg[user_end..]));
        } else {
            out.push(seg.to_string());
        }
        after_home_root = seg == "Users" || seg == "home";
    }
    out.join("/")
}

/// Replace whitespace-delimited tokens that look like an email with `[email]`, preserving
/// all original whitespace.
fn mask_emails(text: &str) -> String {
    for_each_word(text, |word| {
        looks_like_email(word).then(|| "[email]".to_string())
    })
}

/// Whether `token` looks like an email address: a non-empty local part, an `@`, and a
/// domain containing a dot.
fn looks_like_email(token: &str) -> bool {
    match token.split_once('@') {
        Some((local, domain)) => {
            !local.is_empty()
                && domain.contains('.')
                && !domain.starts_with('.')
                && !domain.ends_with('.')
        }
        None => false,
    }
}

/// Replace every run of 7 or more consecutive ASCII digits with `[number]`.
fn mask_long_digit_runs(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut digits = String::new();
    for c in text.chars() {
        if c.is_ascii_digit() {
            digits.push(c);
        } else {
            flush_digits(&mut out, &mut digits);
            out.push(c);
        }
    }
    flush_digits(&mut out, &mut digits);
    out
}

/// Append a buffered digit run to `out`, collapsing it to `[number]` when it's long enough.
fn flush_digits(out: &mut String, digits: &mut String) {
    if digits.len() >= 7 {
        out.push_str("[number]");
    } else {
        out.push_str(digits);
    }
    digits.clear();
}

/// Apply `f` to each whitespace-delimited word, substituting its result when `Some`, while
/// preserving the exact original whitespace between words.
fn for_each_word(text: &str, f: impl Fn(&str) -> Option<String>) -> String {
    let mut out = String::with_capacity(text.len());
    let mut word_start: Option<usize> = None;
    for (i, c) in text.char_indices() {
        if c.is_whitespace() {
            if let Some(s) = word_start.take() {
                let w = &text[s..i];
                out.push_str(&f(w).unwrap_or_else(|| w.to_string()));
            }
            out.push(c);
        } else if word_start.is_none() {
            word_start = Some(i);
        }
    }
    if let Some(s) = word_start {
        let w = &text[s..];
        out.push_str(&f(w).unwrap_or_else(|| w.to_string()));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn masks_macos_home_username() {
        assert_eq!(
            redact("opened /Users/ettore/Library/App/memory.jaxsondb"),
            "opened /Users/[user]/Library/App/memory.jaxsondb"
        );
    }

    #[test]
    fn masks_linux_home_username() {
        assert_eq!(redact("/home/ettore/jaxson"), "/home/[user]/jaxson");
    }

    #[test]
    fn masks_a_home_path_at_the_end_of_the_string() {
        assert_eq!(redact("dir=/Users/ettore"), "dir=/Users/[user]");
    }

    #[test]
    fn leaves_non_home_paths_untouched() {
        assert_eq!(redact("/opt/models/llama.gguf"), "/opt/models/llama.gguf");
    }

    #[test]
    fn does_not_mask_an_empty_segment_after_users() {
        // A double slash means no username to mask — keep it as-is.
        assert_eq!(redact("/Users//Shared/file"), "/Users//Shared/file");
    }

    #[test]
    fn masks_email_addresses_but_keeps_surrounding_words() {
        assert_eq!(
            redact("contact me at ada@example.com please"),
            "contact me at [email] please"
        );
    }

    #[test]
    fn does_not_treat_a_bare_at_as_an_email() {
        assert_eq!(redact("meet @ noon"), "meet @ noon");
        assert_eq!(redact("user@localhost"), "user@localhost"); // no dot in domain
    }

    #[test]
    fn masks_long_digit_runs_only() {
        assert_eq!(redact("call 5551234567 now"), "call [number] now");
        // Short numbers (e.g. a version or count) are kept.
        assert_eq!(redact("scale 1234 x6"), "scale 1234 x6");
    }

    #[test]
    fn boundary_seven_digits_is_masked_six_is_not() {
        assert_eq!(redact("1234567"), "[number]");
        assert_eq!(redact("123456"), "123456");
    }

    #[test]
    fn passes_clean_text_through_unchanged() {
        assert_eq!(
            redact("loaded model llama3.1, turn complete"),
            "loaded model llama3.1, turn complete"
        );
    }

    #[test]
    fn combines_all_rules() {
        assert_eq!(
            redact("/Users/ettore sent ada@example.com 5551234567"),
            "/Users/[user] sent [email] [number]"
        );
    }
}
