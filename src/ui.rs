//! Application shell: a node list on the left, the graph canvas in the middle
//! and details of the selection on the right.

use std::collections::HashMap;
use std::path::Path;

use egui::{Color32, RichText, TextStyle, Ui};

use crate::canvas::{Canvas, CanvasEvent};
use crate::fonts;
use crate::hierarchy::{GroupId, Hierarchy};
use crate::layout::{ItemKind, Layout, LayoutOptions, Scope, layout_graph};
use crate::model::{
    AttrValue, GraphId, Model, NodeId, Tensor, TensorData, Value, ValueId, ValueKind,
};
use crate::op_schema::{self, Arity};
use crate::text::{elide, format_count};
use crate::values;

pub fn run(model: Model, file_name: String, system_font: bool) -> eframe::Result<()> {
    let title = format!("{file_name} — ONNX Explorer");
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1440.0, 900.0])
            .with_min_inner_size([720.0, 480.0])
            .with_title(&title),
        ..Default::default()
    };

    let app = App::new(model);
    eframe::run_native(
        &title,
        options,
        Box::new(move |cc| {
            fonts::install(&cc.egui_ctx, system_font);
            Ok(Box::new(app))
        }),
    )
}

/// Identifies one drawing: a graph, and the block within it that is open.
/// Layouts are cached against this, since a graph laid out at two different
/// levels of its hierarchy is two different drawings.
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
struct ViewKey {
    graph: GraphId,
    group: Option<GroupId>,
    hide_shape_nodes: bool,
}

/// What the details pane is describing.
#[derive(Copy, Clone, PartialEq, Eq)]
enum Selection {
    Node(NodeId),
    Value(ValueId),
}

struct App {
    model: Model,

    /// Graph currently being viewed. Changes when entering a subgraph.
    graph: GraphId,
    /// Group tree for each graph, absent where names carry no structure.
    hierarchies: HashMap<GraphId, Option<Hierarchy>>,
    /// Whether to draw blocks rather than individual operators, where the
    /// current graph's names support it.
    grouped: bool,
    /// The block currently open, within the current graph's hierarchy.
    scope: GroupId,
    /// Layouts are computed on first view and kept, since they are expensive
    /// relative to a frame and never change for a given view.
    layouts: HashMap<ViewKey, Layout>,
    layout_options: LayoutOptions,

    canvas: Canvas,
    selection: Option<Selection>,
    /// What was being looked at before the current selection, most recent last.
    ///
    /// Following a link to a constant leads away from the operator that reads
    /// it, and the constant has no box on the canvas to click back from.
    history: Vec<Selection>,
    query: String,
    /// Whether the filter's matches are listed in place of the details.
    ///
    /// This cannot simply follow the filter box's focus. egui resolves clicks
    /// and focus at the start of a frame using the previous frame's layout, so
    /// by the time the box is drawn on the frame a match is clicked, focus has
    /// already been surrendered. Following focus would hide the list on that
    /// very frame and the click would never reach it.
    filtering: bool,
    /// Nodes of the current graph matching `query`, in graph order.
    matches: Vec<NodeId>,
}

impl App {
    fn new(model: Model) -> App {
        let graph = model.root_id();
        let mut app = App {
            model,
            graph,
            hierarchies: HashMap::new(),
            // Enabled by default wherever the model supports it; this is the
            // more useful way to read a large model.
            grouped: true,
            scope: GroupId(0),
            layouts: HashMap::new(),
            layout_options: LayoutOptions::default(),
            canvas: Canvas::new(),
            selection: None,
            history: Vec::new(),
            query: String::new(),
            filtering: false,
            matches: Vec::new(),
        };
        app.refresh_matches();
        app
    }

    /// Whether the current graph's node names form a usable hierarchy.
    fn has_hierarchy(&self) -> bool {
        self.hierarchies
            .get(&self.graph)
            .is_some_and(|hierarchy| hierarchy.is_some())
    }

    /// The hierarchy in use for the current graph, if grouping is on and the
    /// graph supports it.
    fn active_hierarchy(&self) -> Option<&Hierarchy> {
        if !self.grouped {
            return None;
        }
        self.hierarchies.get(&self.graph)?.as_ref()
    }

    fn view_key(&self) -> ViewKey {
        ViewKey {
            graph: self.graph,
            group: self.active_hierarchy().map(|_| self.scope),
            hide_shape_nodes: self.layout_options.hide_shape_nodes,
        }
    }

    fn ensure_hierarchy(&mut self) {
        if self.hierarchies.contains_key(&self.graph) {
            return;
        }
        let hierarchy = Hierarchy::build(self.model.graph(self.graph));
        self.hierarchies.insert(self.graph, hierarchy);
    }

    fn ensure_layout(&mut self) {
        let key = self.view_key();
        if self.layouts.contains_key(&key) {
            return;
        }
        // Disjoint field borrows: the hierarchy is read while the cache is
        // written.
        let hierarchy = if self.grouped {
            self.hierarchies
                .get(&self.graph)
                .and_then(|hierarchy| hierarchy.as_ref())
        } else {
            None
        };
        let scope = hierarchy.map(|hierarchy| Scope {
            hierarchy,
            group: self.scope,
        });
        let layout = layout_graph(self.model.graph(self.graph), scope, &self.layout_options);
        self.layouts.insert(key, layout);
    }

    /// Open a block, so its contents are drawn in place of its box.
    fn enter_group(&mut self, group: GroupId) {
        self.scope = group;
        self.clear_selection();
        self.canvas.request_home();
    }

    /// Whether the view can move up to an enclosing block.
    fn parent_group(&self) -> Option<GroupId> {
        self.active_hierarchy()?.group(self.scope).parent
    }

    fn refresh_matches(&mut self) {
        let graph = self.model.graph(self.graph);
        let query = self.query.trim().to_lowercase();
        self.matches = graph
            .nodes()
            .iter()
            .filter(|node| {
                query.is_empty()
                    || node.name.to_lowercase().contains(&query)
                    || node.op_type.to_lowercase().contains(&query)
            })
            .map(|node| node.id)
            .collect();
    }

    fn go_to_graph(&mut self, graph: GraphId) {
        self.graph = graph;
        self.scope = GroupId(0);
        self.clear_selection();
        self.canvas.request_home();
        self.ensure_hierarchy();
        self.refresh_matches();
    }

    /// Stop describing anything, discarding the trail with it.
    fn clear_selection(&mut self) {
        self.selection = None;
        self.history.clear();
    }

    /// Move to a selection, remembering where it was reached from.
    fn go_to(&mut self, selection: Selection) {
        match selection {
            Selection::Node(id) => self.select_node(id, true),
            Selection::Value(id) => {
                self.remember();
                self.selection = Some(Selection::Value(id));
            }
        }
    }

    /// Push the current selection onto the trail, if there is one to return to.
    fn remember(&mut self) {
        if let Some(current) = self.selection
            && Some(&current) != self.history.last()
        {
            self.history.push(current);
        }
    }

    /// Return to whatever was being looked at before the current selection.
    fn go_back(&mut self) {
        if let Some(previous) = self.history.pop() {
            match previous {
                // Reveal without recording, or going back would push the
                // selection being left onto the trail and trap the two.
                Selection::Node(id) => {
                    self.selection = Some(Selection::Node(id));
                    self.reveal_node(id);
                }
                Selection::Value(_) => self.selection = Some(previous),
            }
        }
    }

    /// Select a node, optionally bringing it into view on the canvas.
    ///
    /// With grouping on, the node may sit inside a block that is not open, so
    /// the view moves to the block that contains it first.
    fn select_node(&mut self, id: NodeId, reveal: bool) {
        self.remember();
        self.selection = Some(Selection::Node(id));
        if !reveal {
            return;
        }
        self.reveal_node(id);
    }

    /// Bring a node into view, opening the block holding it if need be.
    fn reveal_node(&mut self, id: NodeId) {
        // Read the group before mutating, so the hierarchy borrow ends first.
        let group = self.active_hierarchy().map(|h| h.group_of(id));
        if let Some(group) = group
            && group != self.scope
        {
            self.scope = group;
            self.canvas.request_home();
        }

        // The block may have just changed, so the drawing to focus within may
        // not have been built yet.
        self.ensure_layout();
        let key = self.view_key();
        let rect = self
            .layouts
            .get(&key)
            .and_then(|layout| layout.node_index(id).map(|index| layout.nodes[index].rect));
        if let Some(rect) = rect {
            self.canvas.focus_on(rect);
        }
    }
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.ensure_hierarchy();
        self.ensure_layout();

        egui::Panel::right("side")
            .resizable(true)
            .default_size(380.0)
            .size_range(280.0..=680.0)
            .show(ui, |ui| self.side_panel(ui));

        egui::CentralPanel::default().show(ui, |ui| self.graph_panel(ui));
    }
}

// Side panel.
impl App {
    fn side_panel(&mut self, ui: &mut Ui) {
        ui.add_space(4.0);
        ui.collapsing("Model info", |ui| self.model_info(ui));

        ui.add_space(4.0);
        self.breadcrumb(ui);

        ui.add_space(4.0);
        let filtering = self.filter_box(ui);

        ui.separator();

        // The match list stands in for the details while the filter is in
        // use, so neither the graph nor the details give up room to it.
        // Choosing a match moves focus off the box, which brings the details
        // for the chosen node straight back.
        if filtering {
            self.node_list(ui);
        } else {
            self.details_panel(ui);
        }
    }

    /// Draw the node filter, returning whether its matches should be listed.
    ///
    /// Taking focus opens the list; it stays open until a match is chosen, the
    /// graph is clicked, or Escape dismisses it.
    fn filter_box(&mut self, ui: &mut Ui) -> bool {
        let response = ui.add(
            egui::TextEdit::singleline(&mut self.query)
                .hint_text("Find node by name or type  (/)")
                .desired_width(f32::INFINITY),
        );
        if response.changed() {
            self.refresh_matches();
        }
        if response.gained_focus() {
            self.filtering = true;
        }

        // "/" jumps to the filter, as in a pager, and Cmd+F or Ctrl+F does the
        // same for anyone who reaches for the usual find shortcut. The keys are
        // consumed so they are not typed into the box, and ignored while the
        // box already has focus so that a slash can still be searched for.
        //
        // `COMMAND` is Cmd on macOS and Ctrl elsewhere, so one binding covers
        // both.
        if !response.has_focus() {
            let pressed = ui.input_mut(|input| {
                input.consume_key(egui::Modifiers::NONE, egui::Key::Slash)
                    || input.consume_key(egui::Modifiers::SHIFT, egui::Key::Slash)
                    || input.consume_key(egui::Modifiers::COMMAND, egui::Key::F)
            });
            if pressed {
                response.request_focus();
                self.filtering = true;
            }
        }

        if self.filtering && ui.input(|input| input.key_pressed(egui::Key::Escape)) {
            self.filtering = false;
            response.surrender_focus();
        }

        self.filtering
    }

    fn model_info(&mut self, ui: &mut Ui) {
        let model = &self.model;
        egui::Grid::new("model_info")
            .num_columns(2)
            .spacing([12.0, 4.0])
            .show(ui, |ui| {
                if let Some(producer) = &model.producer_name {
                    ui.label("Producer");
                    let version = model.producer_version.as_deref().unwrap_or("");
                    ui.label(format!("{producer} {version}").trim_end().to_string());
                    ui.end_row();
                }
                if let Some(ir_version) = model.ir_version {
                    ui.label("IR version");
                    ui.label(ir_version.to_string());
                    ui.end_row();
                }
                for opset in &model.opset_imports {
                    ui.label("Opset");
                    ui.label(match opset.version {
                        Some(version) => format!("{} v{}", opset.display_domain(), version),
                        None => opset.display_domain().to_string(),
                    });
                    ui.end_row();
                }
                ui.label("Graphs");
                ui.label(model.graph_count().to_string());
                ui.end_row();

                let params = model.graphs().map(|g| g.parameter_count()).sum::<u64>();
                ui.label("Parameters");
                ui.label(format_count(params));
                ui.end_row();

                for (key, value) in &model.metadata {
                    ui.label(key);
                    ui.label(elide(value, 64));
                    ui.end_row();
                }
            });
    }

    /// Path from the root graph to the current one, as clickable links.
    fn breadcrumb(&mut self, ui: &mut Ui) {
        // The trail runs through any enclosing subgraphs, then down through
        // the blocks opened within the current graph.
        let mut trail: Vec<(String, Step)> = self
            .model
            .path_to(self.graph)
            .into_iter()
            .map(|id| (self.model.graph(id).label.clone(), Step::Graph(id)))
            .collect();

        if let Some(hierarchy) = self.active_hierarchy() {
            for group_id in hierarchy.path_to(self.scope) {
                let group = hierarchy.group(group_id);
                // The root group stands for the whole graph, which the graph
                // trail already names.
                if group.parent.is_some() {
                    trail.push((group.name.clone(), Step::Group(group_id)));
                }
            }
        }

        if trail.len() < 2 {
            return;
        }

        let mut clicked = None;
        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().item_spacing.x = 4.0;
            let last = trail.len() - 1;
            for (index, (label, step)) in trail.iter().enumerate() {
                if index > 0 {
                    ui.label(RichText::new("›").weak());
                }
                if index == last {
                    ui.label(RichText::new(label).strong());
                } else if ui.link(label).clicked() {
                    clicked = Some(*step);
                }
            }
        });

        match clicked {
            Some(Step::Graph(id)) => self.go_to_graph(id),
            Some(Step::Group(id)) => self.enter_group(id),
            None => {}
        }
    }

    fn node_list(&mut self, ui: &mut Ui) {
        let row_height = ui.text_style_height(&TextStyle::Body) + ui.spacing().item_spacing.y;
        let mut clicked = None;

        // Models routinely have thousands of nodes, so only the visible rows
        // are laid out.
        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .show_rows(ui, row_height, self.matches.len(), |ui, range| {
                let graph = self.model.graph(self.graph);
                for index in range {
                    let node_id = self.matches[index];
                    let node = graph.node(node_id);
                    let selected = self.selection == Some(Selection::Node(node_id));
                    let label = format!("{}  {}", node.op_type, node.name);
                    if ui.selectable_label(selected, elide(&label, 56)).clicked() {
                        clicked = Some(node_id);
                    }
                }
            });

        if let Some(node_id) = clicked {
            self.select_node(node_id, true);
            // The chosen node's details replace the list.
            self.filtering = false;
        }
    }
}

// Graph canvas.
impl App {
    fn graph_panel(&mut self, ui: &mut Ui) {
        let mut home = false;
        let mut up = false;
        let mut toggled = false;
        let parent = self.parent_group();
        let has_hierarchy = self.has_hierarchy();

        ui.horizontal(|ui| {
            ui.strong(&self.model.graph(self.graph).label);
            ui.separator();
            home = ui
                .button("Home")
                .on_hover_text("Go to the first input, at a legible zoom")
                .clicked();

            if has_hierarchy && parent.is_some() {
                up = ui.button("Up").on_hover_text("Leave this block").clicked();
            }

            if has_hierarchy {
                toggled = ui
                    .checkbox(&mut self.grouped, "Auto-group nodes")
                    .on_hover_text("Group operators by the structure in their names")
                    .changed();
            }

            toggled |= ui
                .checkbox(
                    &mut self.layout_options.hide_shape_nodes,
                    "Hide shape nodes",
                )
                .on_hover_text("Hide Shape operators and subgraphs that process shapes")
                .changed();

            if self.canvas.is_simplified() {
                ui.label(
                    RichText::new("simplified view — zoom in for detail")
                        .small()
                        .color(Color32::from_rgb(200, 150, 60)),
                );
            }
        });
        if home {
            self.canvas.request_home();
        }
        if toggled {
            // The drawing changes completely, so the previous view is no
            // longer meaningful.
            self.scope = GroupId(0);
            self.clear_selection();
            self.canvas.request_home();
        }
        if up && let Some(parent) = parent {
            let left = self.scope;
            self.enter_group(parent);
            // Come back out looking at the block just left, rather than at the
            // top of an unfamiliar drawing.
            self.ensure_layout();
            let key = self.view_key();
            let rect = self.layouts.get(&key).and_then(|layout| {
                layout
                    .group_index(left)
                    .map(|index| layout.nodes[index].rect)
            });
            if let Some(rect) = rect {
                self.canvas.focus_on(rect);
            }
        }
        // Toggling or moving up may have selected a view with no layout yet.
        self.ensure_layout();
        ui.separator();

        // Disjoint field borrows: the canvas is mutated while the layout it
        // draws is borrowed from the cache.
        let key = self.view_key();
        let layout = &self.layouts[&key];
        let selected_index = match self.selection {
            Some(Selection::Node(id)) => layout.node_index(id),
            Some(Selection::Value(id)) => layout.nodes.iter().position(
                |node| matches!(node.kind, ItemKind::Input(v) | ItemKind::Output(v) if v == id),
            ),
            None => None,
        };
        let event = self.canvas.show(ui, layout, selected_index);

        match event {
            CanvasEvent::Selected(index) => match self.layouts[&key].nodes[index].kind {
                // Clicking a block opens it, which is the point of grouping.
                ItemKind::Group(group) => self.enter_group(group),
                // Selected by clicking it, so it is already on screen: moving
                // the view now would only shift the graph under the pointer.
                ItemKind::Op(id) => self.select_node(id, false),
                ItemKind::Input(id) | ItemKind::Output(id) => self.go_to(Selection::Value(id)),
            },
            CanvasEvent::Cleared => self.clear_selection(),
            CanvasEvent::None => {}
        }
        if !matches!(event, CanvasEvent::None) {
            self.filtering = false;
        }
    }
}

/// One step in the breadcrumb trail.
#[derive(Copy, Clone)]
enum Step {
    Graph(GraphId),
    Group(GroupId),
}

// Details panel.
impl App {
    fn details_panel(&mut self, ui: &mut Ui) {
        let inside_block = self
            .active_hierarchy()
            .is_some_and(|hierarchy| self.scope != hierarchy.root());

        match self.selection {
            Some(Selection::Node(id)) => self.node_details(ui, id),
            Some(Selection::Value(id)) => self.value_details(ui, id),
            None if inside_block => self.block_overview(ui),
            None => self.graph_overview(ui),
        }
    }

    /// Summarise the block currently open, when nothing else is selected.
    fn block_overview(&mut self, ui: &mut Ui) {
        let Some(hierarchy) = self.active_hierarchy() else {
            return;
        };
        let group = hierarchy.group(self.scope);
        let name = group.name.clone();
        let path = group.path.clone();
        let total = group.total_nodes;
        let children: Vec<(GroupId, String, usize)> = group
            .children
            .iter()
            .map(|id| {
                let child = hierarchy.group(*id);
                (*id, child.name.clone(), child.total_nodes)
            })
            .collect();

        ui.heading(&name);
        ui.label(RichText::new(&path).weak());
        ui.separator();

        let mut enter = None;
        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                egui::Grid::new("block_info")
                    .num_columns(2)
                    .spacing([12.0, 4.0])
                    .show(ui, |ui| {
                        ui.label("Operators");
                        ui.label(format_count(total as u64));
                        ui.end_row();
                        if !children.is_empty() {
                            ui.label("Blocks");
                            ui.label(children.len().to_string());
                            ui.end_row();
                        }
                    });

                if !children.is_empty() {
                    section_heading(ui, "Blocks");
                    for (id, name, count) in &children {
                        if ui.link(format!("{name} ({count} operators)")).clicked() {
                            enter = Some(*id);
                        }
                    }
                }
            });

        if let Some(id) = enter {
            self.enter_group(id);
        }
    }

    fn graph_overview(&mut self, ui: &mut Ui) {
        let mut goto = None;
        let mut select = None;
        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                let graph = self.model.graph(self.graph);

                section_heading(ui, "Inputs");
                for value_id in &graph.inputs {
                    let value = graph.value(*value_id);
                    ui.horizontal_wrapped(|ui| {
                        if ui.link(elide(&value.name, 32)).clicked() {
                            select = Some(*value_id);
                        }
                        ui.label(RichText::new(value.type_summary()).color(type_color(ui)));
                    });
                }

                section_heading(ui, "Outputs");
                for value_id in &graph.outputs {
                    let value = graph.value(*value_id);
                    ui.horizontal_wrapped(|ui| {
                        if ui.link(elide(&value.name, 32)).clicked() {
                            select = Some(*value_id);
                        }
                        ui.label(RichText::new(value.type_summary()).color(type_color(ui)));
                    });
                }

                let subgraphs: Vec<_> = graph.subgraphs().collect();
                if !subgraphs.is_empty() {
                    section_heading(ui, "Subgraphs");
                    for (node_id, attr_name, subgraph_id) in subgraphs {
                        let node = graph.node(node_id);
                        let label = format!(
                            "{}.{} ({} nodes)",
                            node.name,
                            attr_name,
                            self.model.graph(subgraph_id).nodes().len()
                        );
                        if ui.link(label).clicked() {
                            goto = Some(subgraph_id);
                        }
                    }
                }
            });

        if let Some(subgraph_id) = goto {
            self.go_to_graph(subgraph_id);
        }
        if let Some(value_id) = select {
            self.go_to(Selection::Value(value_id));
        }
    }

    fn node_details(&mut self, ui: &mut Ui, node_id: NodeId) {
        let mut goto = None;
        let mut goto_graph = None;
        let mut clear = false;
        let mut back = false;

        {
            let graph = self.model.graph(self.graph);
            let node = graph.node(node_id);

            ui.horizontal(|ui| {
                ui.heading(node.qualified_op_type());
                back = back_button(ui, &self.history);
                if ui.button("Clear").clicked() {
                    clear = true;
                }
            });
            ui.label(
                RichText::new(if node.named {
                    node.name.clone()
                } else {
                    format!("{} (unnamed)", node.name)
                })
                .weak(),
            );
            ui.separator();

            egui::ScrollArea::vertical()
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    // What the operator calls each position, where its
                    // signature is one the ONNX spec publishes.
                    let schema = op_schema::lookup(&node.domain, &node.op_type);

                    section_heading(ui, "Inputs");
                    for (index, value_id) in node.inputs.iter().enumerate() {
                        let label = parameter_label(schema.map(|s| s.inputs), index);
                        match value_id {
                            Some(value_id) => {
                                let value = graph.value(*value_id);
                                let constant = graph.constant_tensor(value);
                                if let Some(target) = value_row(
                                    ui,
                                    &label,
                                    value,
                                    constant,
                                    &self.model.source_dir,
                                    true,
                                ) {
                                    goto = Some(target);
                                }
                            }
                            None => {
                                ui.label(RichText::new(format!("{label}  (omitted)")).weak());
                            }
                        }
                    }

                    section_heading(ui, "Outputs");
                    for (index, value_id) in node.outputs.iter().enumerate() {
                        let label = parameter_label(schema.map(|s| s.outputs), index);
                        match value_id {
                            Some(value_id) => {
                                let value = graph.value(*value_id);
                                let constant = graph.constant_tensor(value);
                                if let Some(target) = value_row(
                                    ui,
                                    &label,
                                    value,
                                    constant,
                                    &self.model.source_dir,
                                    false,
                                ) {
                                    goto = Some(target);
                                }
                            }
                            None => {
                                ui.label(RichText::new(format!("{label}  (omitted)")).weak());
                            }
                        }
                    }

                    if !node.attrs.is_empty() {
                        section_heading(ui, "Attributes");
                        egui::Grid::new("attrs")
                            .num_columns(3)
                            .spacing([12.0, 3.0])
                            .striped(true)
                            .show(ui, |ui| {
                                for attr in &node.attrs {
                                    ui.monospace(&attr.name);
                                    ui.label(RichText::new(attr.value.type_name()).weak());
                                    match attr.value {
                                        AttrValue::Graph(subgraph_id) => {
                                            let count = self.model.graph(subgraph_id).nodes().len();
                                            if ui.link(format!("open ({count} nodes)")).clicked() {
                                                goto_graph = Some(subgraph_id);
                                            }
                                        }
                                        _ => {
                                            ui.label(elide(&attr.value.to_string(), 72));
                                        }
                                    }
                                    ui.end_row();
                                }
                            });
                    }
                });
        }

        if back {
            self.go_back();
        } else if clear {
            self.clear_selection();
        }
        if let Some(target) = goto {
            self.go_to(target);
        }
        if let Some(graph_id) = goto_graph {
            self.go_to_graph(graph_id);
        }
    }

    fn value_details(&mut self, ui: &mut Ui, value_id: ValueId) {
        let mut goto_node = None;
        let mut clear = false;
        let mut back = false;

        {
            let graph = self.model.graph(self.graph);
            let value = graph.value(value_id);

            ui.horizontal(|ui| {
                ui.heading(elide(&value.name, 28));
                back = back_button(ui, &self.history);
                if ui.button("Clear").clicked() {
                    clear = true;
                }
            });
            ui.label(RichText::new(value.kind.label()).weak());
            ui.separator();

            egui::ScrollArea::vertical()
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    egui::Grid::new("value_info")
                        .num_columns(2)
                        .spacing([12.0, 4.0])
                        .show(ui, |ui| {
                            ui.label("Name");
                            ui.monospace(&value.name);
                            ui.end_row();
                            ui.label("Type");
                            ui.label(value.type_summary());
                            ui.end_row();
                            if let Some(tensor) = graph.constant_tensor(value) {
                                ui.label("Elements");
                                ui.label(format_count(tensor.elem_count()));
                                ui.end_row();
                                if let Some(bytes) = tensor.byte_len() {
                                    ui.label("Size");
                                    ui.label(format!("{} bytes", format_count(bytes as u64)));
                                    ui.end_row();
                                }
                                // Weights kept outside the model file describe
                                // where to find themselves, which is worth
                                // showing since the data is not in the .onnx.
                                if let TensorData::External { entries } = &tensor.data {
                                    for (key, value) in entries {
                                        ui.label(format!("External {key}"));
                                        ui.label(value);
                                        ui.end_row();
                                    }
                                }
                            }
                            if let Some(outer) = value.outer {
                                ui.label("Defined in");
                                ui.label(&self.model.graph(outer.graph).label);
                                ui.end_row();
                            }
                        });

                    if let Some(tensor) = graph.constant_tensor(value) {
                        section_heading(ui, "Values");
                        tensor_values(ui, tensor, &self.model.source_dir, value_id);
                    }

                    section_heading(ui, "Producer");
                    match value.producer {
                        Some(producer) => {
                            let node = graph.node(producer);
                            if ui
                                .link(format!("{}  {}", node.op_type, elide(&node.name, 28)))
                                .clicked()
                            {
                                goto_node = Some(producer);
                            }
                        }
                        None => {
                            ui.label(RichText::new("none").weak());
                        }
                    }

                    section_heading(ui, format!("Consumers ({})", value.consumers.len()));
                    for consumer in &value.consumers {
                        let node = graph.node(*consumer);
                        if ui
                            .link(format!("{}  {}", node.op_type, elide(&node.name, 28)))
                            .clicked()
                        {
                            goto_node = Some(*consumer);
                        }
                    }
                });
        }

        if back {
            self.go_back();
        } else if clear {
            self.clear_selection();
        }
        if let Some(target) = goto_node {
            self.select_node(target, true);
        }
    }
}

/// Label for one position in an operator's inputs or outputs.
///
/// Operators outside the domains the specification covers fall back to the
/// position itself, which is all the model says about them.
fn parameter_label(params: Option<&'static [(&'static str, Arity)]>, index: usize) -> String {
    params
        .and_then(|params| op_schema::parameter(params, index))
        .unwrap_or_else(|| format!("{index}."))
}

/// Draw the button returning to the previously selected item, if there is one.
///
/// The label names what it goes back to, since following a chain of links can
/// leave it far from obvious.
fn back_button(ui: &mut Ui, history: &[Selection]) -> bool {
    let Some(previous) = history.last() else {
        return false;
    };
    let hint = match previous {
        Selection::Node(_) => "Back to the operator that led here",
        Selection::Value(_) => "Back to the value that led here",
    };
    ui.button("Back").on_hover_text(hint).clicked()
}

/// Largest constant written out on an input or output row.
///
/// Enough for a shape or a per-axis parameter, and short enough to leave the
/// row readable. Longer vectors are counted instead, and read in the value's
/// own pane.
const ROW_ELEMENTS: u64 = 8;

/// Format a constant small enough to show on its input row, if it is one.
///
/// Only scalars and short vectors qualify: a matrix has no reading that fits on
/// a line. Tensors stored outside the model file are skipped whatever their
/// size, so that listing a node's inputs never waits on the disk.
fn inline_value(tensor: &Tensor, base_dir: &Path) -> Option<String> {
    let count = tensor.elem_count();
    if count > ROW_ELEMENTS
        || tensor.dims.len() > 1
        || matches!(tensor.data, TensorData::External { .. })
    {
        return None;
    }

    let elements = values::read_elements(tensor, base_dir, 0, count as usize).ok()?;
    if elements.is_empty() {
        return None;
    }

    let text = elements.join(", ");
    // A scalar is written as the bare number; ONNX gives it no dimensions.
    Some(if tensor.dims.is_empty() {
        text
    } else {
        format!("[{text}]")
    })
}

/// Largest tensor shown in full, rather than behind an expander.
///
/// A vector of a few dozen numbers is worth reading at a glance; anything
/// bigger is a wall of digits that pushes the rest of the pane off screen.
const INLINE_ELEMENTS: u64 = 32;

/// Show the elements of a constant tensor.
///
/// Only the rows on screen are read, so a tensor of a hundred million weights
/// costs the same to scroll through as a small one. Weights held in a separate
/// file are read on demand and only once expanded, which keeps a model whose
/// data file has not been downloaded fully browsable.
fn tensor_values(ui: &mut Ui, tensor: &Tensor, base_dir: &Path, value_id: ValueId) {
    let count = tensor.elem_count();
    if count == 0 {
        ui.label(RichText::new("empty").weak());
        return;
    }

    if let Err(err) = values::readable(tensor) {
        ui.label(RichText::new(err.to_string()).weak());
        return;
    }

    // Small vectors read fine on one line. Anything external stays behind the
    // expander however small, so that opening the pane never goes to disk.
    let external = matches!(tensor.data, TensorData::External { .. });
    if count <= INLINE_ELEMENTS && tensor.dims.len() <= 1 && !external {
        match values::read_elements(tensor, base_dir, 0, count as usize) {
            Ok(elements) => {
                ui.horizontal_wrapped(|ui| {
                    ui.spacing_mut().item_spacing.x = 4.0;
                    ui.monospace(format!("[{}]", elements.join(", ")));
                });
            }
            Err(err) => {
                ui.label(RichText::new(err.to_string()).weak());
            }
        }
        return;
    }

    let rows = count.min(usize::MAX as u64) as usize;
    egui::CollapsingHeader::new(format!("{} elements", format_count(count)))
        .id_salt(("values", value_id.0))
        .show(ui, |ui| {
            let row_height = ui.text_style_height(&TextStyle::Monospace);
            egui::ScrollArea::vertical()
                .max_height(row_height * 12.0)
                .auto_shrink([false, false])
                .show_rows(ui, row_height, rows, |ui, range| {
                    let start = range.start as u64;
                    let elements = match values::read_elements(tensor, base_dir, start, range.len())
                    {
                        Ok(elements) => elements,
                        Err(err) => {
                            ui.label(RichText::new(err.to_string()).weak());
                            return;
                        }
                    };
                    for (offset, element) in elements.iter().enumerate() {
                        ui.horizontal(|ui| {
                            ui.spacing_mut().item_spacing.x = 8.0;
                            ui.label(
                                RichText::new(values::index_label(
                                    &tensor.dims,
                                    start + offset as u64,
                                ))
                                .weak()
                                .monospace(),
                            );
                            ui.monospace(element);
                        });
                    }
                });
        });
}

/// Draw a heading for a section of the details panel.
///
/// The sections all hold lists of similar-looking rows, so headings have to be
/// visible at a glance while scanning down the panel: they are set in capitals,
/// a size up from the body text and emboldened.
///
/// Where the platform provides no bold face (see [`fonts::has_real_bold`]) the
/// text is drawn twice a fraction of a pixel apart, which thickens the strokes
/// enough to read as bold.
fn section_heading(ui: &mut Ui, text: impl Into<String>) {
    ui.add_space(14.0);

    let font = fonts::bold(TextStyle::Body.resolve(ui.style()).size * 1.1);
    let color = ui.visuals().strong_text_color();
    let galley = ui
        .painter()
        .layout_no_wrap(text.into().to_uppercase(), font, color);

    let offset = if fonts::has_real_bold() {
        egui::Vec2::ZERO
    } else {
        egui::vec2(0.6, 0.0)
    };
    let (rect, _) = ui.allocate_exact_size(galley.size() + offset, egui::Sense::hover());
    ui.painter().galley(rect.min, galley.clone(), color);
    ui.painter().galley(rect.min + offset, galley, color);

    ui.add_space(8.0);
}

/// Colour for tensor types and shapes in the details pane.
///
/// egui's `weak` colour is too low-contrast to read comfortably at this size.
/// Using the normal text colour keeps these legible, and does the right thing
/// in both themes: darker against a light background, brighter against a dark
/// one.
fn type_color(ui: &Ui) -> Color32 {
    ui.visuals().text_color()
}

/// Render one input or output row. Returns what to select if the user clicked
/// the value's name, or a link to its producer or consumer.
fn value_row(
    ui: &mut Ui,
    label: &str,
    value: &Value,
    constant: Option<&Tensor>,
    base_dir: &Path,
    is_input: bool,
) -> Option<Selection> {
    let mut target = None;
    ui.horizontal_wrapped(|ui| {
        ui.spacing_mut().item_spacing.x = 6.0;
        ui.label(RichText::new(label).weak().monospace());

        // A constant written out in full says everything its name and type
        // would, so give the space to the value and leave a link for the rest.
        if let Some(text) = constant.and_then(|tensor| inline_value(tensor, base_dir)) {
            ui.monospace(text);
            if ui.link("(details)").clicked() {
                target = Some(Selection::Value(value.id));
            }
            return;
        }

        // The name opens the value itself, which is the way to its stored
        // values for a constant: those have no box on the canvas to click.
        if ui
            .link(RichText::new(elide(&value.name, 36)).monospace())
            .clicked()
        {
            target = Some(Selection::Value(value.id));
        }
        ui.label(RichText::new(value.type_summary()).color(type_color(ui)));

        match value.kind {
            ValueKind::Initializer | ValueKind::Constant => {
                let params = constant
                    .map(|t| format_count(t.elem_count()))
                    .unwrap_or_default();
                ui.label(RichText::new(format!("const · {params}")).color(Color32::GRAY));
            }
            ValueKind::OuterScope => {
                ui.label(RichText::new("outer scope").color(Color32::GRAY));
            }
            _ => {
                // Offer a jump to the node on the other end of this edge.
                if is_input {
                    if let Some(producer) = value.producer {
                        if ui.link("← producer").clicked() {
                            target = Some(Selection::Node(producer));
                        }
                    } else {
                        ui.label(RichText::new(value.kind.label()).color(Color32::GRAY));
                    }
                } else {
                    match value.consumers.len() {
                        0 => {
                            ui.label(RichText::new(value.kind.label()).color(Color32::GRAY));
                        }
                        1 => {
                            if ui.link("consumer →").clicked() {
                                target = Some(Selection::Node(value.consumers[0]));
                            }
                        }
                        n => {
                            ui.label(RichText::new(format!("{n} consumers")).weak());
                            for consumer in &value.consumers {
                                if ui.link("→").clicked() {
                                    target = Some(Selection::Node(*consumer));
                                }
                            }
                        }
                    }
                }
            }
        }
    });
    target
}
