# Run every check: compilation, lints, tests and formatting.
check:
    cargo check --all-targets
    cargo clippy --all-targets -- -D warnings
    cargo test
    cargo fmt --check
