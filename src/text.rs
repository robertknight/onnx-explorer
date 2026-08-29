//! Small text helpers shared by the canvas and the side panels.

/// Format a large count compactly, eg. `1.2M`.
pub fn format_count(count: u64) -> String {
    const K: u64 = 1_000;
    const M: u64 = 1_000_000;
    const G: u64 = 1_000_000_000;
    match count {
        0..K => count.to_string(),
        K..M => format!("{:.1}K", count as f64 / K as f64),
        M..G => format!("{:.1}M", count as f64 / M as f64),
        _ => format!("{:.2}G", count as f64 / G as f64),
    }
}

/// Truncate `text` to `max` characters, appending an ellipsis if shortened.
pub fn elide(text: &str, max: usize) -> String {
    if max == 0 {
        return String::new();
    }
    if text.chars().count() <= max {
        return text.to_string();
    }
    let head: String = text.chars().take(max.saturating_sub(1)).collect();
    format!("{head}…")
}

#[cfg(test)]
mod tests {
    use super::{elide, format_count};

    #[test]
    fn test_format_count() {
        assert_eq!(format_count(0), "0");
        assert_eq!(format_count(999), "999");
        assert_eq!(format_count(1_500), "1.5K");
        assert_eq!(format_count(2_500_000), "2.5M");
        assert_eq!(format_count(7_000_000_000), "7.00G");
    }

    #[test]
    fn test_elide() {
        assert_eq!(elide("abc", 5), "abc");
        assert_eq!(elide("abcdef", 4), "abc…");
        assert_eq!(elide("abc", 0), "");
        // Character, not byte, boundaries.
        assert_eq!(elide("αβγδε", 3), "αβ…");
    }
}
