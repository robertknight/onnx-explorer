# onnx-explorer

A native app for exploring ONNX model graphs. It is like [Netron] but is much
faster at loading large and complex models.

## Building

Install the current version of [Rust](https://rust-lang.org/) and cargo, then
build the project with:

```sh
cargo build --release
```

This writes the binary to `target/release/onnx-explorer`. To install it onto
your path instead:

```sh
cargo install --path .
```

## Usage

```sh
onnx-explorer model.onnx
```

Print a summary to the terminal instead of opening a window:

```sh
onnx-explorer --summary model.onnx
```

## AI development disclaimer

This project was vibe-coded with Claude Code. The author has not read most of
the code.

[Netron]: https://github.com/lutzroeder/netron
