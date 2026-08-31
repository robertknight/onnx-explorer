//! Native viewer for ONNX model graphs.

mod canvas;
mod fonts;
mod hierarchy;
mod layout;
mod model;
mod text;
mod ui;
mod values;

use std::fs::File;
use std::path::Path;
use std::process::ExitCode;
use std::time::Instant;

use rten_onnx::onnx::ModelProto;

use model::Model;

/// Explore the structure of an ONNX model.
#[derive(argh::FromArgs)]
struct Args {
    /// print a summary to the terminal instead of opening a window
    #[argh(switch, short = 's')]
    summary: bool,

    /// lay out every graph and print timings, without opening a window
    #[argh(switch)]
    layout: bool,

    /// draw the UI with the fonts bundled with the app rather than the system
    /// UI font
    #[argh(switch)]
    no_system_font: bool,

    /// display the version
    #[argh(switch, short = 'V')]
    version: bool,

    /// path to the '.onnx' model to explore
    #[argh(positional)]
    model: Option<String>,
}

fn main() -> ExitCode {
    let args: Args = argh::from_env();

    if args.version {
        println!("onnx-explorer {}", env!("CARGO_PKG_VERSION"));
        return ExitCode::SUCCESS;
    }

    // The model is optional only so that `--version` can be used on its own.
    let Some(path) = args.model else {
        eprintln!("error: expected a path to an ONNX model");
        eprintln!("Run \"onnx-explorer --help\" for usage.");
        return ExitCode::FAILURE;
    };

    let start = Instant::now();
    let file = match File::open(&path) {
        Ok(file) => file,
        Err(err) => {
            eprintln!("error: could not open \"{path}\": {err}");
            return ExitCode::FAILURE;
        }
    };
    let proto = match ModelProto::parse_file(file) {
        Ok(proto) => proto,
        Err(err) => {
            eprintln!("error: could not parse \"{path}\": {err}");
            return ExitCode::FAILURE;
        }
    };
    let mut model = Model::from_proto(proto);
    // Weights kept outside the .onnx are named relative to it.
    model.source_dir = Path::new(&path)
        .parent()
        .unwrap_or(Path::new(""))
        .to_path_buf();
    let load_time = start.elapsed();

    let file_name = Path::new(&path)
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.clone());

    if args.summary {
        print_summary(&model, &file_name, load_time);
        return ExitCode::SUCCESS;
    }

    if args.layout {
        print_layout_timings(&model, &file_name, load_time);
        return ExitCode::SUCCESS;
    }

    if let Err(err) = ui::run(model, file_name, !args.no_system_font) {
        eprintln!("error: could not start the user interface: {err}");
        return ExitCode::FAILURE;
    }

    ExitCode::SUCCESS
}

/// Lay out every graph in the model and report how long each took.
fn print_layout_timings(model: &Model, file_name: &str, load_time: std::time::Duration) {
    use crate::layout::{LayoutOptions, layout_graph};

    println!(
        "{file_name}  (parsed in {:.1} ms)",
        load_time.as_secs_f64() * 1000.0
    );
    let opts = LayoutOptions::default();
    let mut total = std::time::Duration::ZERO;
    // Total drawn edge length, which is what suffers when a box is ranked far
    // from the nodes it connects to.
    let mut edge_length = 0.0;
    let mut elided = 0;

    for graph in model.graphs() {
        let start = Instant::now();
        let layout = layout_graph(graph, None, &opts);
        let elapsed = start.elapsed();
        total += elapsed;
        edge_length += layout
            .edges
            .iter()
            .flat_map(|edge| edge.points.windows(2))
            .map(|segment| (segment[1] - segment[0]).length())
            .sum::<f32>();
        elided += layout.edges.iter().filter(|edge| edge.elided).count();

        if graph.nodes().len() >= 100 {
            println!(
                "  {:<20} {:>6} nodes -> {:>6} boxes, {:>6} edges, {:>6} ranks, {:>8} dummies, {:>8.1} ms",
                graph.label,
                graph.nodes().len(),
                layout.nodes.len(),
                layout.edges.len(),
                layout.rank_count,
                layout.dummy_count,
                elapsed.as_secs_f64() * 1000.0,
            );
        }
    }

    println!("  total layout: {:.1} ms", total.as_secs_f64() * 1000.0);
    println!("  total edge length: {edge_length:.0}");
    println!("  elided edges: {elided}");

    let root = model.root();
    let shape_derived = root.shape_derived_nodes().iter().filter(|n| **n).count();
    println!(
        "  shape nodes: {shape_derived} of {} operators",
        root.nodes().len()
    );

    // The grouped view of the main graph, which is what opens by default when
    // node names carry structure.
    if let Some(hierarchy) = crate::hierarchy::Hierarchy::build(model.root()) {
        let scope = crate::layout::Scope {
            hierarchy: &hierarchy,
            group: hierarchy.root(),
        };
        let start = Instant::now();
        let layout = layout_graph(model.root(), Some(scope), &opts);
        println!(
            "  grouped top level:   {} blocks -> {} boxes, {} edges, {:.1} ms",
            hierarchy.group(hierarchy.root()).children.len(),
            layout.nodes.len(),
            layout.edges.len(),
            start.elapsed().as_secs_f64() * 1000.0,
        );
    }
}

/// Print an overview of the model, for use from the terminal and as a way to
/// sanity check parsing without opening a window.
fn print_summary(model: &Model, file_name: &str, load_time: std::time::Duration) {
    println!(
        "{file_name}  (loaded in {:.1} ms)",
        load_time.as_secs_f64() * 1000.0
    );

    if let Some(producer) = &model.producer_name {
        let version = model.producer_version.as_deref().unwrap_or("");
        println!(
            "  producer:   {}",
            format!("{producer} {version}").trim_end()
        );
    }
    if let Some(ir_version) = model.ir_version {
        println!("  ir version: {ir_version}");
    }
    for opset in &model.opset_imports {
        match opset.version {
            Some(version) => println!("  opset:      {} v{version}", opset.display_domain()),
            None => println!("  opset:      {}", opset.display_domain()),
        }
    }

    let root = model.root();
    println!("  graphs:     {}", model.graph_count());
    match crate::hierarchy::Hierarchy::build(root) {
        Some(hierarchy) => println!(
            "  blocks:     {} from node names",
            hierarchy.group_count() - 1
        ),
        None => println!("  blocks:     none, node names are not hierarchical"),
    }
    println!(
        "  nodes:      {} in the main graph, {} in total",
        root.nodes().len(),
        model.graphs().map(|g| g.nodes().len()).sum::<usize>()
    );
    println!(
        "  parameters: {}",
        model.graphs().map(|g| g.parameter_count()).sum::<u64>()
    );

    println!("  inputs:");
    for value_id in &root.inputs {
        let value = root.value(*value_id);
        println!("    {} {}", value.name, value.type_summary());
    }
    println!("  outputs:");
    for value_id in &root.outputs {
        let value = root.value(*value_id);
        println!("    {} {}", value.name, value.type_summary());
    }

    let ops = root.op_type_counts();
    println!("  operators:  {} distinct", ops.len());
    for (op_type, count) in ops.iter().take(8) {
        println!("    {count:>6}  {op_type}");
    }

    for graph in model.graphs() {
        if let Some(parent) = graph.parent {
            println!(
                "  subgraph {} of {}: {} nodes",
                graph.label,
                model.graph(parent).label,
                graph.nodes().len()
            );
        }
    }
}
