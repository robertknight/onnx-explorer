//! Layered graph layout.
//!
//! This is a Sugiyama-style layout, the same family of algorithm that Netron
//! uses via dagre. Nodes are assigned to horizontal layers ("ranks") by
//! dependency depth, ordered within each layer to reduce edge crossings, then
//! given x coordinates that pull each node towards its neighbours.
//!
//! Edges spanning more than one rank are routed through invisible "dummy"
//! nodes, one per intervening rank, so that long edges take part in ordering
//! and are drawn as polylines rather than crossing arbitrarily through the
//! drawing.

use std::collections::{HashMap, HashSet};

use egui::{Pos2, Rect, Vec2, pos2, vec2};

use crate::hierarchy::{GroupId, Hierarchy, Placement};
use crate::model::{Graph, NodeId, OpCategory, Value, ValueId};

pub struct LayoutOptions {
    pub node_height: f32,
    pub min_node_width: f32,
    pub max_node_width: f32,
    /// Estimated width of one character of a box's title, used to size the
    /// box. Labels are elided to fit when drawn, so this only needs to be
    /// close.
    pub char_width: f32,
    /// The same for the smaller text of a box's subtitle.
    pub subtitle_char_width: f32,
    /// Vertical gap between ranks.
    pub layer_gap: f32,
    /// Minimum horizontal gap between nodes in the same rank.
    pub node_gap: f32,
    pub ordering_iterations: usize,
    pub position_iterations: usize,
    /// Longest edge, in ranks, that is routed through dummy nodes.
    ///
    /// A dummy is created for every rank an edge crosses, so in a deep graph a
    /// single edge skipping most of the model would create thousands of them.
    /// Edges longer than this are drawn as a direct line instead and take no
    /// part in ordering, which bounds the cost at the price of a few long
    /// edges cutting across the drawing.
    pub max_edge_span: usize,
}

impl Default for LayoutOptions {
    fn default() -> Self {
        LayoutOptions {
            node_height: 38.0,
            min_node_width: 96.0,
            max_node_width: 280.0,
            char_width: 7.6,
            subtitle_char_width: 5.9,
            layer_gap: 46.0,
            node_gap: 18.0,
            ordering_iterations: 8,
            position_iterations: 8,
            max_edge_span: 32,
        }
    }
}

/// What a drawn box represents.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum ItemKind {
    Op(NodeId),
    /// A block of nodes, standing in for everything inside it. Entering the
    /// group replaces the box with its contents.
    Group(GroupId),
    /// A value entering the view: a declared graph input, or a value produced
    /// outside the current scope.
    Input(ValueId),
    /// A value leaving the view: a declared graph output, or a value read
    /// outside the current scope.
    Output(ValueId),
}

/// The part of the model being drawn, when nodes are grouped by name.
#[derive(Copy, Clone)]
pub struct Scope<'a> {
    pub hierarchy: &'a Hierarchy,
    pub group: GroupId,
}

pub struct LayoutNode {
    pub kind: ItemKind,
    pub rect: Rect,
    /// Primary label: the op type, or the value name for an input or output.
    pub title: String,
    /// Secondary label, shown when zoomed in far enough to read it. Empty for
    /// operators, which are labelled by their type alone.
    pub subtitle: String,
    /// What kind of work the operator does, for colouring. `None` for boxes
    /// that are not operators.
    pub category: Option<OpCategory>,
}

pub struct LayoutEdge {
    /// Index into [`Layout::nodes`].
    pub from: usize,
    pub to: usize,
    /// Polyline from the source node's bottom edge to the target's top edge.
    pub points: Vec<Pos2>,
    /// Bounding box of `points`, for culling.
    pub bounds: Rect,
    pub label: String,
}

pub struct Layout {
    pub nodes: Vec<LayoutNode>,
    pub edges: Vec<LayoutEdge>,
    pub bounds: Rect,
    /// Number of ranks the drawing is divided into.
    pub rank_count: usize,
    /// Dummy nodes created for edge routing, reported for diagnostics.
    pub dummy_count: usize,
    node_by_id: HashMap<NodeId, usize>,
    group_by_id: HashMap<GroupId, usize>,
    /// Edge indices incident to each node, for highlighting a selection.
    incident: Vec<Vec<usize>>,
}

impl Layout {
    /// Index of the drawn box for `id`, absent if the node was folded away or
    /// lies outside the current scope.
    pub fn node_index(&self, id: NodeId) -> Option<usize> {
        self.node_by_id.get(&id).copied()
    }

    /// Index of the drawn box standing in for a group.
    pub fn group_index(&self, id: GroupId) -> Option<usize> {
        self.group_by_id.get(&id).copied()
    }

    pub fn incident_edges(&self, node: usize) -> &[usize] {
        &self.incident[node]
    }

    /// The box to open the graph at: its first declared input, or failing
    /// that whichever box sits highest in the drawing.
    pub fn entry_node(&self) -> Option<usize> {
        self.nodes
            .iter()
            .position(|node| matches!(node.kind, ItemKind::Input(_)))
            .or_else(|| {
                self.nodes
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| {
                        a.rect
                            .min
                            .y
                            .partial_cmp(&b.rect.min.y)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(index, _)| index)
            })
    }
}

/// A node in the internal layout graph: either one of the drawn boxes, or a
/// dummy standing in for a long edge as it passes through a rank.
struct Cell {
    rank: usize,
    order: usize,
    x: f32,
    width: f32,
    height: f32,
    /// Index into `Layout::nodes`, or `None` for a dummy.
    item: Option<usize>,
}

pub fn layout_graph(graph: &Graph, scope: Option<Scope>, opts: &LayoutOptions) -> Layout {
    let collected = collect(graph, scope, opts);
    let Collected {
        mut nodes,
        node_by_id,
        group_by_id,
        links,
    } = collected;

    let mut ranks = assign_ranks(nodes.len(), &links);
    sink_constant_fed(&mut ranks, &links, &constant_fed(graph, scope, &nodes));

    // Build the cell graph, inserting dummies for edges spanning several ranks.
    let mut cells: Vec<Cell> = nodes
        .iter()
        .enumerate()
        .map(|(index, node)| Cell {
            rank: ranks[index],
            order: 0,
            x: 0.0,
            width: node.rect.width(),
            height: node.rect.height(),
            item: Some(index),
        })
        .collect();

    let mut routes: Vec<Vec<usize>> = Vec::with_capacity(links.len());
    for link in &links {
        routes.push(route(&mut cells, link, &ranks, opts));
    }

    let (preds, succs) = adjacency(&cells, &routes);
    let rank_members = order_cells(&mut cells, &preds, &succs, opts);
    assign_x(&mut cells, &preds, &succs, &rank_members, opts);
    let rank_y = assign_y(&cells, &rank_members, opts);

    // Write the computed geometry back onto the drawn boxes.
    for cell in &cells {
        if let Some(index) = cell.item {
            let center = pos2(cell.x, rank_y[cell.rank]);
            nodes[index].rect = Rect::from_center_size(center, vec2(cell.width, cell.height));
        }
    }

    let edges = build_edges(&nodes, &cells, &rank_y, &links, &routes, graph);
    let bounds = bounds_of(&nodes, &edges);

    let mut incident = vec![Vec::new(); nodes.len()];
    for (index, edge) in edges.iter().enumerate() {
        incident[edge.from].push(index);
        incident[edge.to].push(index);
    }

    Layout {
        nodes,
        edges,
        bounds,
        rank_count: rank_y.len(),
        dummy_count: cells.iter().filter(|c| c.item.is_none()).count(),
        node_by_id,
        group_by_id,
        incident,
    }
}

/// A dependency between two drawn boxes.
struct Link {
    from: usize,
    to: usize,
    value: ValueId,
}

struct Collected {
    nodes: Vec<LayoutNode>,
    node_by_id: HashMap<NodeId, usize>,
    group_by_id: HashMap<GroupId, usize>,
    links: Vec<Link>,
}

/// Choose which boxes to draw and how they connect.
///
/// Without a scope this is every operator in the graph, plus a box for each
/// declared input and output. With a scope, the operators directly inside it
/// are drawn alongside one box per child group, and values crossing the
/// scope's boundary get input and output boxes so the view still shows what
/// flows in and out.
///
/// Constants are never drawn. A large model has hundreds of them and they
/// carry no structure; they appear in the details pane of the node that reads
/// them instead.
fn collect<'a>(graph: &'a Graph, scope: Option<Scope<'a>>, opts: &LayoutOptions) -> Collected {
    let mut builder = Collector {
        graph,
        scope,
        opts,
        nodes: Vec::new(),
        node_by_id: HashMap::new(),
        group_by_id: HashMap::new(),
        input_by_value: HashMap::new(),
        output_by_value: HashMap::new(),
        links: Vec::new(),
        seen_group_links: HashSet::new(),
    };
    builder.collect_boxes();
    builder.collect_links();

    Collected {
        nodes: builder.nodes,
        node_by_id: builder.node_by_id,
        group_by_id: builder.group_by_id,
        links: builder.links,
    }
}

struct Collector<'a> {
    graph: &'a Graph,
    scope: Option<Scope<'a>>,
    opts: &'a LayoutOptions,
    nodes: Vec<LayoutNode>,
    node_by_id: HashMap<NodeId, usize>,
    group_by_id: HashMap<GroupId, usize>,
    input_by_value: HashMap<ValueId, usize>,
    output_by_value: HashMap<ValueId, usize>,
    links: Vec<Link>,
    /// Links already drawn between a pair of boxes where one is a group.
    /// Aggregating a block's edges otherwise produces a bundle of identical
    /// lines between the same two boxes.
    seen_group_links: HashSet<(usize, usize)>,
}

impl<'a> Collector<'a> {
    /// Whether the view covers the whole graph, so declared inputs and outputs
    /// are shown whether or not anything uses them.
    fn at_top_level(&self) -> bool {
        match self.scope {
            Some(scope) => scope.group == scope.hierarchy.root(),
            None => true,
        }
    }

    fn push(&mut self, kind: ItemKind, title: String, subtitle: String) -> usize {
        // Both lines are drawn inside the box, so the wider of the two decides
        // its width. Sizing from the title alone leaves a long subtitle, such
        // as the type of a graph input, elided in a box with room to spare.
        let title_width = title.chars().count() as f32 * self.opts.char_width;
        let subtitle_width = subtitle.chars().count() as f32 * self.opts.subtitle_char_width;
        let width = (title_width.max(subtitle_width) + 28.0)
            .clamp(self.opts.min_node_width, self.opts.max_node_width);
        let index = self.nodes.len();
        self.nodes.push(LayoutNode {
            kind,
            rect: Rect::from_min_size(Pos2::ZERO, vec2(width, self.opts.node_height)),
            title,
            subtitle,
            category: None,
        });
        index
    }

    fn collect_boxes(&mut self) {
        if self.at_top_level() {
            for value_id in &self.graph.inputs {
                let value = self.graph.value(*value_id);
                if value.is_constant() {
                    continue;
                }
                let index = self.push(
                    ItemKind::Input(*value_id),
                    value.name.clone(),
                    value.type_summary(),
                );
                self.input_by_value.insert(*value_id, index);
            }
        }

        match self.scope {
            Some(scope) => {
                let group = scope.hierarchy.group(scope.group);
                for child in &group.children {
                    let child_group = scope.hierarchy.group(*child);
                    let count = child_group.total_nodes;
                    let index = self.push(
                        ItemKind::Group(*child),
                        child_group.name.clone(),
                        format!("{count} operators"),
                    );
                    self.group_by_id.insert(*child, index);
                }
                for node_id in &group.nodes {
                    self.push_op(*node_id);
                }
            }
            None => {
                for node in self.graph.nodes() {
                    self.push_op(node.id);
                }
            }
        }

        if self.at_top_level() {
            for value_id in &self.graph.outputs {
                let value = self.graph.value(*value_id);
                let index = self.push(
                    ItemKind::Output(*value_id),
                    value.name.clone(),
                    value.type_summary(),
                );
                self.output_by_value.insert(*value_id, index);
            }
        }
    }

    fn push_op(&mut self, node_id: NodeId) {
        let node = self.graph.node(node_id);
        if node.is_constant() {
            return;
        }
        let index = self.push(ItemKind::Op(node_id), node.op_type.clone(), String::new());
        self.nodes[index].category = Some(OpCategory::of(&node.op_type));
        self.node_by_id.insert(node_id, index);
    }

    /// Locate a node relative to the current scope.
    fn placement(&self, node: NodeId) -> Placement {
        match self.scope {
            Some(scope) => scope.hierarchy.placement(scope.group, node),
            None => Placement::Direct,
        }
    }

    /// The visible box standing in for a node, if it has one.
    fn box_for(&self, node: NodeId) -> Option<usize> {
        match self.placement(node) {
            Placement::Direct => self.node_by_id.get(&node).copied(),
            Placement::Within(group) => self.group_by_id.get(&group).copied(),
            Placement::Outside => None,
        }
    }

    fn collect_links(&mut self) {
        let graph_outputs: HashSet<ValueId> = self.graph.outputs.iter().copied().collect();

        for value in self.graph.values() {
            if value.is_constant() {
                continue;
            }

            // Consumers visible here, and whether the value is also read
            // somewhere outside this scope.
            let mut targets = Vec::new();
            let mut escapes = false;
            for consumer in &value.consumers {
                match self.placement(*consumer) {
                    Placement::Outside => escapes = true,
                    _ => {
                        if let Some(index) = self.box_for(*consumer) {
                            targets.push(index);
                        }
                    }
                }
            }
            targets.sort_unstable();
            targets.dedup();

            let source = match value.producer {
                Some(producer) => self.box_for(producer),
                // No producer: a declared graph input, or a value inherited
                // from an enclosing graph.
                None => self.input_by_value.get(&value.id).copied(),
            };

            let leaves = escapes || graph_outputs.contains(&value.id);
            let source = match source {
                Some(index) => index,
                None => {
                    if targets.is_empty() {
                        continue;
                    }
                    self.boundary_input(value)
                }
            };

            for target in targets {
                self.link(source, target, value.id);
            }

            if leaves {
                let target = self.boundary_output(value);
                self.link(source, target, value.id);
            }
        }
    }

    /// Box for a value arriving from outside the current scope.
    fn boundary_input(&mut self, value: &Value) -> usize {
        if let Some(index) = self.input_by_value.get(&value.id) {
            return *index;
        }
        let index = self.push(
            ItemKind::Input(value.id),
            value.name.clone(),
            value.type_summary(),
        );
        self.input_by_value.insert(value.id, index);
        index
    }

    /// Box for a value leaving the current scope.
    fn boundary_output(&mut self, value: &Value) -> usize {
        if let Some(index) = self.output_by_value.get(&value.id) {
            return *index;
        }
        let index = self.push(
            ItemKind::Output(value.id),
            value.name.clone(),
            value.type_summary(),
        );
        self.output_by_value.insert(value.id, index);
        index
    }

    fn link(&mut self, from: usize, to: usize, value: ValueId) {
        // An edge wholly inside one group is not visible at this level.
        if from == to {
            return;
        }
        let involves_group = matches!(self.nodes[from].kind, ItemKind::Group(_))
            || matches!(self.nodes[to].kind, ItemKind::Group(_));
        if involves_group && !self.seen_group_links.insert((from, to)) {
            return;
        }
        self.links.push(Link { from, to, value });
    }
}

/// Assign each box to a rank, so that every edge points downwards.
///
/// This is longest-path layering over a topological order. ONNX graphs are
/// required to be acyclic, but a malformed model could contain a cycle, so any
/// nodes left unranked are released in rank order rather than being dropped.
fn assign_ranks(count: usize, links: &[Link]) -> Vec<usize> {
    let mut succs: Vec<Vec<usize>> = vec![Vec::new(); count];
    let mut in_degree = vec![0usize; count];
    for link in links {
        if link.from == link.to {
            continue;
        }
        succs[link.from].push(link.to);
        in_degree[link.to] += 1;
    }

    let mut rank = vec![0usize; count];
    let mut ready: Vec<usize> = (0..count).filter(|i| in_degree[*i] == 0).collect();
    let mut done = 0;

    loop {
        while let Some(index) = ready.pop() {
            done += 1;
            for &succ in &succs[index] {
                rank[succ] = rank[succ].max(rank[index] + 1);
                in_degree[succ] -= 1;
                if in_degree[succ] == 0 {
                    ready.push(succ);
                }
            }
        }
        if done == count {
            break;
        }
        // A cycle remains. Break it at the node closest to the top.
        let stuck = (0..count)
            .filter(|i| in_degree[*i] > 0)
            .min_by_key(|i| rank[*i]);
        match stuck {
            Some(index) => {
                in_degree[index] = 0;
                ready.push(index);
            }
            None => break,
        }
    }

    rank
}

/// Mark the drawn boxes that stand for work on constants alone.
///
/// A block counts only if everything inside it does, since the block is drawn
/// as one box and moves as one.
fn constant_fed(graph: &Graph, scope: Option<Scope>, nodes: &[LayoutNode]) -> Vec<bool> {
    let by_node = graph.constant_fed_nodes();
    nodes
        .iter()
        .map(|node| match node.kind {
            ItemKind::Op(id) => by_node[id.0 as usize],
            ItemKind::Group(group) => {
                scope.is_some_and(|scope| group_is_constant_fed(scope.hierarchy, group, &by_node))
            }
            ItemKind::Input(_) | ItemKind::Output(_) => false,
        })
        .collect()
}

fn group_is_constant_fed(hierarchy: &Hierarchy, group: GroupId, by_node: &[bool]) -> bool {
    let group = hierarchy.group(group);
    group.nodes.iter().all(|id| by_node[id.0 as usize])
        && group
            .children
            .iter()
            .all(|child| group_is_constant_fed(hierarchy, *child, by_node))
}

/// Move boxes that only work on constants down to just above their first
/// consumer.
///
/// Longest-path layering puts everything without a predecessor in the top rank.
/// A shape computation or a generated mask therefore starts at the top of the
/// drawing, however far down the node that uses it sits, leaving an edge
/// stretched across the whole graph. These boxes have no path back to a graph
/// input, so nothing is holding them up there.
///
/// Boxes are lowered in reverse rank order so that a chain of them collapses in
/// one pass: each is placed against a successor that has already moved.
fn sink_constant_fed(ranks: &mut [usize], links: &[Link], constant_fed: &[bool]) {
    if !constant_fed.iter().any(|fed| *fed) {
        return;
    }

    let mut order: Vec<usize> = (0..ranks.len())
        .filter(|index| constant_fed[*index])
        .collect();
    order.sort_by_key(|index| std::cmp::Reverse(ranks[*index]));

    let mut consumers: Vec<Vec<usize>> = vec![Vec::new(); ranks.len()];
    for link in links {
        if link.from != link.to {
            consumers[link.from].push(link.to);
        }
    }

    for index in order {
        let lowest = consumers[index]
            .iter()
            .map(|consumer| ranks[*consumer])
            .min();
        // A consumer is always at least one rank down, but a broken cycle could
        // leave one level with or above this box, so clamp.
        if let Some(consumer) = lowest {
            ranks[index] = consumer.saturating_sub(1);
        }
    }

    compact_ranks(ranks);
}

/// Close up ranks left empty by [`sink_sources`], which would otherwise be
/// drawn as a band of blank space.
fn compact_ranks(ranks: &mut [usize]) {
    let Some(&highest) = ranks.iter().max() else {
        return;
    };
    let mut occupied = vec![false; highest + 1];
    for &rank in ranks.iter() {
        occupied[rank] = true;
    }
    let mut renumbered = vec![0usize; highest + 1];
    let mut next = 0;
    for (rank, used) in occupied.iter().enumerate() {
        renumbered[rank] = next;
        next += usize::from(*used);
    }
    for rank in ranks.iter_mut() {
        *rank = renumbered[*rank];
    }
}

/// Create the chain of cells an edge passes through, adding a dummy for each
/// rank strictly between its endpoints.
fn route(cells: &mut Vec<Cell>, link: &Link, ranks: &[usize], opts: &LayoutOptions) -> Vec<usize> {
    let (from_rank, to_rank) = (ranks[link.from], ranks[link.to]);
    let mut route = vec![link.from];

    if to_rank > from_rank + 1 && to_rank - from_rank <= opts.max_edge_span {
        for rank in (from_rank + 1)..to_rank {
            let index = cells.len();
            cells.push(Cell {
                rank,
                order: 0,
                x: 0.0,
                // Dummies are narrow: they only need to reserve enough room for
                // the edge to pass between neighbouring nodes.
                width: 1.0,
                height: opts.node_height,
                item: None,
            });
            route.push(index);
        }
    }

    route.push(link.to);
    route
}

fn adjacency(cells: &[Cell], routes: &[Vec<usize>]) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
    let mut preds = vec![Vec::new(); cells.len()];
    let mut succs = vec![Vec::new(); cells.len()];
    for route in routes {
        for pair in route.windows(2) {
            let (from, to) = (pair[0], pair[1]);
            // Skip links that do not descend a rank; they are drawn, but taking
            // part in ordering would be meaningless.
            if cells[to].rank <= cells[from].rank {
                continue;
            }
            succs[from].push(to);
            preds[to].push(from);
        }
    }
    (preds, succs)
}

/// Order cells within each rank to reduce edge crossings.
fn order_cells(
    cells: &mut [Cell],
    preds: &[Vec<usize>],
    succs: &[Vec<usize>],
    opts: &LayoutOptions,
) -> Vec<Vec<usize>> {
    let rank_count = cells.iter().map(|c| c.rank + 1).max().unwrap_or(0);
    let mut members: Vec<Vec<usize>> = vec![Vec::new(); rank_count];
    // ONNX requires nodes to be listed in topological order, so insertion order
    // is already a reasonable starting point.
    for (index, cell) in cells.iter().enumerate() {
        members[cell.rank].push(index);
    }
    write_orders(cells, &members);

    for iteration in 0..opts.ordering_iterations {
        let downward = iteration % 2 == 0;
        let sweep: Vec<usize> = if downward {
            (1..rank_count).collect()
        } else {
            (0..rank_count.saturating_sub(1)).rev().collect()
        };

        for rank in sweep {
            let neighbours = if downward { preds } else { succs };
            let keys: Vec<(f32, usize)> = members[rank]
                .iter()
                .enumerate()
                .map(|(position, &cell)| {
                    (median_order(cells, &neighbours[cell], position), position)
                })
                .collect();

            let mut ordered: Vec<usize> = (0..members[rank].len()).collect();
            ordered.sort_by(|&a, &b| {
                keys[a]
                    .0
                    .partial_cmp(&keys[b].0)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(keys[a].1.cmp(&keys[b].1))
            });
            members[rank] = ordered.into_iter().map(|i| members[rank][i]).collect();
            // Only this rank moved, so only its cells need renumbering.
            // Rewriting every rank here would make ordering quadratic in the
            // number of ranks.
            write_rank_orders(cells, &members[rank]);
        }

        transpose(cells, preds, succs, &mut members);
    }

    members
}

/// Median position of a cell's neighbours in the adjacent rank. Cells with no
/// neighbours keep their current position so they do not drift.
fn median_order(cells: &[Cell], neighbours: &[usize], position: usize) -> f32 {
    if neighbours.is_empty() {
        return position as f32;
    }
    let mut orders: Vec<usize> = neighbours.iter().map(|&n| cells[n].order).collect();
    orders.sort_unstable();
    let middle = orders.len() / 2;
    if orders.len() % 2 == 1 {
        orders[middle] as f32
    } else {
        (orders[middle - 1] + orders[middle]) as f32 / 2.0
    }
}

fn write_orders(cells: &mut [Cell], members: &[Vec<usize>]) {
    for rank in members {
        write_rank_orders(cells, rank);
    }
}

fn write_rank_orders(cells: &mut [Cell], rank: &[usize]) {
    for (position, &cell) in rank.iter().enumerate() {
        cells[cell].order = position;
    }
}

/// Swap adjacent pairs within a rank where doing so reduces crossings.
///
/// This catches the cases the median heuristic cannot, such as two nodes whose
/// neighbours give them the same median.
fn transpose(
    cells: &mut [Cell],
    preds: &[Vec<usize>],
    succs: &[Vec<usize>],
    members: &mut [Vec<usize>],
) {
    let mut improved = true;
    let mut rounds = 0;
    while improved && rounds < 4 {
        improved = false;
        rounds += 1;
        for rank in members.iter_mut() {
            for position in 0..rank.len().saturating_sub(1) {
                let left = rank[position];
                let right = rank[position + 1];
                let before = crossings(cells, &preds[left], &preds[right])
                    + crossings(cells, &succs[left], &succs[right]);
                let after = crossings(cells, &preds[right], &preds[left])
                    + crossings(cells, &succs[right], &succs[left]);
                if after < before {
                    rank.swap(position, position + 1);
                    cells[left].order = position + 1;
                    cells[right].order = position;
                    improved = true;
                }
            }
        }
    }
}

/// Number of crossings between the edges of a left cell and a right cell.
fn crossings(cells: &[Cell], left: &[usize], right: &[usize]) -> usize {
    let mut count = 0;
    for &l in left {
        for &r in right {
            if cells[l].order > cells[r].order {
                count += 1;
            }
        }
    }
    count
}

/// Assign x coordinates, alternating downward and upward passes.
///
/// Each pass pulls every cell towards the average position of its neighbours in
/// the adjacent rank, then resolves the rank so that no two nodes overlap while
/// staying as close to those targets as possible.
fn assign_x(
    cells: &mut [Cell],
    preds: &[Vec<usize>],
    succs: &[Vec<usize>],
    members: &[Vec<usize>],
    opts: &LayoutOptions,
) {
    // Start with each rank packed left to right in its chosen order.
    for rank in members {
        let mut x = 0.0;
        for (position, &cell) in rank.iter().enumerate() {
            if position > 0 {
                x += opts.node_gap / 2.0;
            }
            x += cells[cell].width / 2.0;
            cells[cell].x = x;
            x += cells[cell].width / 2.0 + opts.node_gap / 2.0;
        }
    }

    let mut targets: Vec<f32> = Vec::new();
    let mut weights: Vec<f32> = Vec::new();

    for iteration in 0..opts.position_iterations {
        let downward = iteration % 2 == 0;
        let sweep: Vec<usize> = if downward {
            (1..members.len()).collect()
        } else {
            (0..members.len().saturating_sub(1)).rev().collect()
        };

        for rank in sweep {
            let neighbours = if downward { preds } else { succs };
            targets.clear();
            weights.clear();

            for &cell in &members[rank] {
                let adjacent = &neighbours[cell];
                if adjacent.is_empty() {
                    targets.push(cells[cell].x);
                    // Unconstrained cells should yield to those with edges.
                    weights.push(0.4);
                } else {
                    let sum: f32 = adjacent.iter().map(|&n| cells[n].x).sum();
                    targets.push(sum / adjacent.len() as f32);
                    // Keeping long edges straight matters more than nudging any
                    // single node, so dummies pull harder.
                    weights.push(if cells[cell].item.is_none() { 6.0 } else { 1.0 });
                }
            }

            resolve_rank(cells, &members[rank], &mut targets, &weights, opts.node_gap);
        }
    }
}

/// Place one rank's cells as close to `targets` as the ordering and minimum
/// separation allow.
///
/// Subtracting each cell's minimum offset from its target turns the separation
/// constraints into a requirement that the result be non-decreasing, which is
/// isotonic regression. Pool-adjacent-violators solves it exactly in linear
/// time, so the rank lands at the least-squares optimum rather than at whatever
/// a sequence of local nudges happens to produce.
fn resolve_rank(
    cells: &mut [Cell],
    members: &[usize],
    targets: &mut [f32],
    weights: &[f32],
    gap: f32,
) {
    if members.is_empty() {
        return;
    }

    let mut offset = 0.0;
    let mut offsets = Vec::with_capacity(members.len());
    for (position, &cell) in members.iter().enumerate() {
        if position > 0 {
            offset += cells[members[position - 1]].width / 2.0 + gap + cells[cell].width / 2.0;
        }
        offsets.push(offset);
        targets[position] -= offset;
    }

    isotonic(targets, weights);

    for (position, &cell) in members.iter().enumerate() {
        cells[cell].x = targets[position] + offsets[position];
    }
}

/// Pool adjacent violators: the weighted least-squares fit to `values` subject
/// to the result being non-decreasing.
fn isotonic(values: &mut [f32], weights: &[f32]) {
    // Each stack entry is a block of merged values sharing one fitted value.
    let mut block_value: Vec<f32> = Vec::with_capacity(values.len());
    let mut block_weight: Vec<f32> = Vec::with_capacity(values.len());
    let mut block_len: Vec<usize> = Vec::with_capacity(values.len());

    for (index, &value) in values.iter().enumerate() {
        let mut value = value;
        let mut weight = weights[index].max(f32::EPSILON);
        let mut len = 1;

        // Merge back into any preceding block that this value would undercut.
        while let Some(&previous) = block_value.last() {
            if previous <= value {
                break;
            }
            let previous_weight = block_weight.pop().unwrap();
            block_value.pop();
            len += block_len.pop().unwrap();
            value = (value * weight + previous * previous_weight) / (weight + previous_weight);
            weight += previous_weight;
        }

        block_value.push(value);
        block_weight.push(weight);
        block_len.push(len);
    }

    let mut index = 0;
    for (block, &value) in block_value.iter().enumerate() {
        for _ in 0..block_len[block] {
            values[index] = value;
            index += 1;
        }
    }
}

/// Vertical centre of each rank, stacking ranks by their tallest cell.
fn assign_y(cells: &[Cell], members: &[Vec<usize>], opts: &LayoutOptions) -> Vec<f32> {
    let mut y = 0.0;
    let mut rank_y = Vec::with_capacity(members.len());
    for rank in members {
        let height = rank
            .iter()
            .map(|&cell| cells[cell].height)
            .fold(opts.node_height, f32::max);
        rank_y.push(y + height / 2.0);
        y += height + opts.layer_gap;
    }
    rank_y
}

fn build_edges(
    nodes: &[LayoutNode],
    cells: &[Cell],
    rank_y: &[f32],
    links: &[Link],
    routes: &[Vec<usize>],
    graph: &Graph,
) -> Vec<LayoutEdge> {
    links
        .iter()
        .zip(routes)
        .map(|(link, route)| {
            let mut points = Vec::with_capacity(route.len());
            points.push(bottom_centre(&nodes[link.from].rect));
            for &cell in &route[1..route.len() - 1] {
                points.push(pos2(cells[cell].x, rank_y[cells[cell].rank]));
            }
            points.push(top_centre(&nodes[link.to].rect));

            let bounds = points.iter().fold(Rect::NOTHING, |bounds, point| {
                bounds.union(Rect::from_center_size(*point, Vec2::ZERO))
            });

            LayoutEdge {
                from: link.from,
                to: link.to,
                points,
                bounds,
                label: graph.value(link.value).type_summary(),
            }
        })
        .collect()
}

fn bottom_centre(rect: &Rect) -> Pos2 {
    pos2(rect.center().x, rect.max.y)
}

fn top_centre(rect: &Rect) -> Pos2 {
    pos2(rect.center().x, rect.min.y)
}

fn bounds_of(nodes: &[LayoutNode], edges: &[LayoutEdge]) -> Rect {
    let mut bounds = Rect::NOTHING;
    for node in nodes {
        bounds = bounds.union(node.rect);
    }
    for edge in edges {
        bounds = bounds.union(edge.bounds);
    }
    if !bounds.is_finite() {
        return Rect::from_min_size(Pos2::ZERO, Vec2::ZERO);
    }
    bounds
}

#[cfg(test)]
mod tests {
    use super::{ItemKind, LayoutOptions, Scope, isotonic, layout_graph};
    use crate::hierarchy::Hierarchy;
    use crate::model::{Graph, Model};
    use rten_onnx::onnx::{
        DataType, Dimension, GraphProto, ModelProto, NodeProto, TensorShapeProto, TypeProto,
        TypeProtoTensor, ValueInfoProto,
    };

    fn node(op_type: &str, name: &str, inputs: &[&str], outputs: &[&str]) -> NodeProto {
        NodeProto {
            op_type: Some(op_type.to_string()),
            name: Some(name.to_string()),
            input: inputs.iter().map(|s| s.to_string()).collect(),
            output: outputs.iter().map(|s| s.to_string()).collect(),
            ..Default::default()
        }
    }

    fn value_info(name: &str) -> ValueInfoProto {
        ValueInfoProto {
            name: Some(name.to_string()),
            r#type: None,
        }
    }

    /// Find the box drawn for the operator named `name`.
    fn box_for<'a>(layout: &'a super::Layout, graph: &Graph, name: &str) -> &'a super::LayoutNode {
        let node = graph
            .nodes()
            .iter()
            .find(|node| node.name == name)
            .expect("no such node");
        layout
            .nodes
            .iter()
            .find(|drawn| drawn.kind == ItemKind::Op(node.id))
            .expect("node was not drawn")
    }

    /// Like [`value_info`], but with an element type and shape, which the box
    /// for the value uses as its subtitle.
    fn typed_value_info(name: &str, dims: &[i64]) -> ValueInfoProto {
        ValueInfoProto {
            name: Some(name.to_string()),
            r#type: Some(TypeProto {
                tensor_type: Some(TypeProtoTensor {
                    elem_type: Some(DataType::FLOAT),
                    shape: Some(TensorShapeProto {
                        dim: dims
                            .iter()
                            .map(|dim| Dimension {
                                dim_value: Some(*dim),
                                dim_param: None,
                            })
                            .collect(),
                    }),
                }),
                sequence: None,
            }),
        }
    }

    fn model(graph: GraphProto) -> Model {
        Model::from_proto(ModelProto {
            graph: Some(graph),
            ..Default::default()
        })
    }

    #[test]
    fn test_isotonic_is_monotonic_and_exact() {
        // Already sorted input is returned unchanged.
        let mut values = [1.0, 2.0, 3.0];
        isotonic(&mut values, &[1.0; 3]);
        assert_eq!(values, [1.0, 2.0, 3.0]);

        // A violating pair is replaced by its weighted mean.
        let mut values = [3.0, 1.0];
        isotonic(&mut values, &[1.0, 1.0]);
        assert_eq!(values, [2.0, 2.0]);

        // Weights bias the merged value towards the heavier element.
        let mut values = [3.0, 1.0];
        isotonic(&mut values, &[3.0, 1.0]);
        assert_eq!(values, [2.5, 2.5]);

        // The result is non-decreasing for an arbitrary input.
        let mut values = [5.0, 1.0, 4.0, 2.0, 9.0, 0.0];
        isotonic(&mut values, &[1.0; 6]);
        assert!(values.windows(2).all(|w| w[0] <= w[1]), "{values:?}");
    }

    #[test]
    fn test_chain_is_laid_out_top_to_bottom() {
        let model = model(GraphProto {
            node: vec![
                node("Relu", "a", &["x"], &["h1"]),
                node("Conv", "b", &["h1"], &["h2"]),
                node("Sigmoid", "c", &["h2"], &["y"]),
            ],
            input: vec![value_info("x")],
            output: vec![value_info("y")],
            ..Default::default()
        });

        let layout = layout_graph(model.root(), None, &LayoutOptions::default());

        // One box per operator, plus the graph input and output.
        assert_eq!(layout.nodes.len(), 5);
        assert_eq!(layout.edges.len(), 4);

        // Each box sits strictly below the one that feeds it.
        for edge in &layout.edges {
            let from = layout.nodes[edge.from].rect;
            let to = layout.nodes[edge.to].rect;
            assert!(from.max.y <= to.min.y, "edge should point downwards");
        }

        let input = layout
            .nodes
            .iter()
            .find(|n| matches!(n.kind, ItemKind::Input(_)))
            .unwrap();
        let output = layout
            .nodes
            .iter()
            .find(|n| matches!(n.kind, ItemKind::Output(_)))
            .unwrap();
        assert!(input.rect.min.y < output.rect.min.y);
    }

    #[test]
    fn test_nodes_in_a_rank_do_not_overlap() {
        // A fan-out to many nodes, all of which share a rank.
        let mut nodes = vec![node("Relu", "root", &["x"], &["h"])];
        for i in 0..24 {
            nodes.push(node(
                "Mul",
                &format!("leaf{i}"),
                &["h"],
                &[&format!("o{i}")],
            ));
        }

        let model = model(GraphProto {
            node: nodes,
            input: vec![value_info("x")],
            ..Default::default()
        });

        let opts = LayoutOptions::default();
        let layout = layout_graph(model.root(), None, &opts);

        let mut leaves: Vec<_> = (0..24)
            .map(|i| box_for(&layout, model.root(), &format!("leaf{i}")).rect)
            .collect();
        assert_eq!(leaves.len(), 24);
        leaves.sort_by(|a, b| a.min.x.partial_cmp(&b.min.x).unwrap());

        for pair in leaves.windows(2) {
            let separation = pair[1].min.x - pair[0].max.x;
            assert!(
                separation >= opts.node_gap - 0.01,
                "boxes overlap: gap was {separation}"
            );
        }
    }

    #[test]
    fn test_constant_fed_nodes_sink_to_their_consumer() {
        // A chain of four, with a pair of nodes working on a constant alone and
        // feeding the last of them. Ranked by longest path those two sit at the
        // top, a whole graph away from their only consumer.
        let constant = NodeProto {
            op_type: Some("Constant".to_string()),
            name: Some("k".to_string()),
            output: vec!["c".to_string()],
            attribute: vec![rten_onnx::onnx::AttributeProto {
                name: Some("value".to_string()),
                t: Some(rten_onnx::onnx::TensorProto {
                    name: Some("c".to_string()),
                    dims: vec![2],
                    data_type: Some(DataType::FLOAT),
                    ..Default::default()
                }),
                ..Default::default()
            }],
            ..Default::default()
        };
        let model = model(GraphProto {
            node: vec![
                constant,
                node("Mul", "const_chain", &["c", "c"], &["m"]),
                node("Reshape", "fed_by_const", &["m"], &["r"]),
                node("Relu", "a", &["x"], &["h1"]),
                node("Relu", "b", &["h1"], &["h2"]),
                node("Relu", "c2", &["h2"], &["h3"]),
                node("Add", "last", &["h3", "r"], &["y"]),
            ],
            input: vec![value_info("x")],
            ..Default::default()
        });

        let layout = layout_graph(model.root(), None, &LayoutOptions::default());
        let head = box_for(&layout, model.root(), "const_chain");
        let fed = box_for(&layout, model.root(), "fed_by_const");
        let last = box_for(&layout, model.root(), "last");
        let first = box_for(&layout, model.root(), "a");

        assert!(
            fed.rect.min.y > first.rect.min.y,
            "constant-fed node should not stay in the top rank"
        );
        assert!(
            fed.rect.max.y < last.rect.min.y,
            "constant-fed node should stay above its consumer"
        );
        // The whole chain moves, not just the box nearest the consumer.
        assert!(
            head.rect.min.y > first.rect.min.y && head.rect.max.y < fed.rect.min.y,
            "the rest of the constant-fed chain should follow it down"
        );
    }

    #[test]
    fn test_long_edges_are_routed_through_dummies() {
        // A skip connection from the first node to the last, spanning 3 ranks.
        let model = model(GraphProto {
            node: vec![
                node("Relu", "a", &["x"], &["h1"]),
                node("Conv", "b", &["h1"], &["h2"]),
                node("Conv", "c", &["h2"], &["h3"]),
                node("Add", "d", &["h3", "h1"], &["y"]),
            ],
            input: vec![value_info("x")],
            ..Default::default()
        });

        let layout = layout_graph(model.root(), None, &LayoutOptions::default());
        let is = |index: usize, name: &str| {
            std::ptr::eq(&layout.nodes[index], box_for(&layout, model.root(), name))
        };
        let skip = layout
            .edges
            .iter()
            .find(|e| is(e.from, "a") && is(e.to, "d"))
            .expect("skip connection should be present");

        // Endpoints plus one bend per intervening rank.
        assert!(skip.points.len() > 2, "long edge should have bend points");
        assert!(skip.points.windows(2).all(|w| w[0].y <= w[1].y));
    }

    #[test]
    fn test_constants_are_not_drawn() {
        let constant = NodeProto {
            op_type: Some("Constant".to_string()),
            name: Some("k".to_string()),
            output: vec!["k".to_string()],
            attribute: vec![rten_onnx::onnx::AttributeProto {
                name: Some("value".to_string()),
                t: Some(rten_onnx::onnx::TensorProto {
                    name: Some("k".to_string()),
                    dims: vec![4],
                    data_type: Some(rten_onnx::onnx::DataType::FLOAT),
                    ..Default::default()
                }),
                ..Default::default()
            }],
            ..Default::default()
        };

        let model = model(GraphProto {
            node: vec![constant, node("Mul", "mul", &["x", "k"], &["y"])],
            input: vec![value_info("x")],
            ..Default::default()
        });

        let layout = layout_graph(model.root(), None, &LayoutOptions::default());
        assert!(
            layout.nodes.iter().all(|n| n.title != "Constant"),
            "constant nodes should be folded away"
        );
        // The input and the Mul remain, connected by one edge.
        assert_eq!(layout.nodes.len(), 2);
        assert_eq!(layout.edges.len(), 1);
    }

    /// A small model named the way exporters name nodes: an embedding and two
    /// layers, the first with attention and MLP submodules.
    ///
    /// Every block holds at least two nodes, since a block standing for a
    /// single node is not created.
    fn hierarchical_model() -> Model {
        model(GraphProto {
            node: vec![
                node("MatMul", "/embed/MatMul", &["x"], &["e0"]),
                node("Add", "/embed/Add", &["e0"], &["h0"]),
                node("MatMul", "/layers.0/attn/MatMul", &["h0"], &["h1"]),
                node("Add", "/layers.0/attn/Add", &["h1"], &["h2"]),
                node("Mul", "/layers.0/mlp/Mul", &["h2"], &["m0"]),
                node("Add", "/layers.0/mlp/Add", &["m0"], &["h3"]),
                node("MatMul", "/layers.1/attn/MatMul", &["h3"], &["l0"]),
                node("Add", "/layers.1/attn/Add", &["l0"], &["y"]),
            ],
            input: vec![value_info("x")],
            output: vec![value_info("y")],
            ..Default::default()
        })
    }

    #[test]
    fn test_top_level_draws_blocks_not_operators() {
        let model = hierarchical_model();
        let graph = model.root();
        let hierarchy = Hierarchy::build(graph).unwrap();
        let scope = Scope {
            hierarchy: &hierarchy,
            group: hierarchy.root(),
        };

        let layout = layout_graph(graph, Some(scope), &LayoutOptions::default());

        // The graph input and output, plus one box per top-level block.
        let groups: Vec<&str> = layout
            .nodes
            .iter()
            .filter(|n| matches!(n.kind, ItemKind::Group(_)))
            .map(|n| n.title.as_str())
            .collect();
        assert_eq!(groups, ["embed", "layers.0", "layers.1"]);
        assert_eq!(layout.nodes.len(), 5);

        // No operator is drawn directly: they are all inside a block.
        assert!(
            !layout
                .nodes
                .iter()
                .any(|n| matches!(n.kind, ItemKind::Op(_)))
        );

        // A block's box reports everything nested inside it.
        let layer0 = layout.nodes.iter().find(|n| n.title == "layers.0").unwrap();
        assert_eq!(layer0.subtitle, "4 operators");

        // x -> embed -> layers.0 -> layers.1 -> y. The two edges inside
        // layers.0 are not visible at this level.
        assert_eq!(layout.edges.len(), 4);
    }

    #[test]
    fn test_blocks_aggregate_parallel_edges_into_one() {
        // Two separate values crossing the same block boundary.
        let model = model(GraphProto {
            node: vec![
                node("Identity", "/a/Identity", &["x"], &["x2"]),
                node("Split", "/a/Split", &["x2"], &["p", "q"]),
                node("Add", "/b/Add", &["p"], &["r"]),
                node("Mul", "/b/Mul", &["q"], &["s"]),
            ],
            input: vec![value_info("x")],
            ..Default::default()
        });
        let graph = model.root();
        let hierarchy = Hierarchy::build(graph).unwrap();
        let scope = Scope {
            hierarchy: &hierarchy,
            group: hierarchy.root(),
        };

        let layout = layout_graph(graph, Some(scope), &LayoutOptions::default());

        // `p` and `q` both run from block `a` to block `b`, but a bundle of
        // identical lines between the same two boxes says nothing extra.
        let a_to_b = layout
            .edges
            .iter()
            .filter(|e| layout.nodes[e.from].title == "a" && layout.nodes[e.to].title == "b")
            .count();
        assert_eq!(a_to_b, 1);
    }

    #[test]
    fn test_entering_a_block_shows_its_contents_and_boundary() {
        let model = hierarchical_model();
        let graph = model.root();
        let hierarchy = Hierarchy::build(graph).unwrap();

        let root = hierarchy.group(hierarchy.root());
        let layer0 = *root
            .children
            .iter()
            .find(|id| hierarchy.group(**id).name == "layers.0")
            .unwrap();

        let layout = layout_graph(
            graph,
            Some(Scope {
                hierarchy: &hierarchy,
                group: layer0,
            }),
            &LayoutOptions::default(),
        );

        // Its two submodules, drawn as blocks.
        let groups: Vec<&str> = layout
            .nodes
            .iter()
            .filter(|n| matches!(n.kind, ItemKind::Group(_)))
            .map(|n| n.title.as_str())
            .collect();
        assert_eq!(groups, ["attn", "mlp"]);

        // The value arriving from the embedding, and the one leaving for the
        // next layer, are drawn so the view shows what crosses the boundary.
        let inputs: Vec<&str> = layout
            .nodes
            .iter()
            .filter(|n| matches!(n.kind, ItemKind::Input(_)))
            .map(|n| n.title.as_str())
            .collect();
        let outputs: Vec<&str> = layout
            .nodes
            .iter()
            .filter(|n| matches!(n.kind, ItemKind::Output(_)))
            .map(|n| n.title.as_str())
            .collect();
        assert_eq!(inputs, ["h0"]);
        assert_eq!(outputs, ["h3"]);

        // h0 -> attn, attn -> mlp, mlp -> h3. The edge wholly inside attn is
        // not visible here.
        assert_eq!(layout.edges.len(), 3);
    }

    #[test]
    fn test_deepest_block_shows_operators() {
        let model = hierarchical_model();
        let graph = model.root();
        let hierarchy = Hierarchy::build(graph).unwrap();
        let attn = hierarchy.group_of(graph.nodes()[2].id);

        let layout = layout_graph(
            graph,
            Some(Scope {
                hierarchy: &hierarchy,
                group: attn,
            }),
            &LayoutOptions::default(),
        );

        let ops: Vec<&str> = layout
            .nodes
            .iter()
            .filter(|n| matches!(n.kind, ItemKind::Op(_)))
            .map(|n| n.title.as_str())
            .collect();
        assert_eq!(ops, ["MatMul", "Add"]);
        assert!(
            !layout
                .nodes
                .iter()
                .any(|n| matches!(n.kind, ItemKind::Group(_)))
        );
    }

    #[test]
    fn test_box_is_wide_enough_for_its_subtitle() {
        // A short input name over a long type summary. Sizing from the title
        // alone would leave the type elided in a box at minimum width.
        let model = model(GraphProto {
            node: vec![node("Mul", "m", &["x"], &["y"])],
            input: vec![typed_value_info("x", &[1, 128, 768, 4])],
            ..Default::default()
        });

        let opts = LayoutOptions::default();
        let layout = layout_graph(model.root(), None, &opts);
        let input = layout
            .nodes
            .iter()
            .find(|n| matches!(n.kind, ItemKind::Input(_)))
            .unwrap();

        assert!(
            input.rect.width() > opts.min_node_width,
            "box should grow past its minimum, was {}",
            input.rect.width()
        );
        assert!(
            input.rect.width() >= input.subtitle.chars().count() as f32 * opts.subtitle_char_width,
            "box should fit the subtitle {:?}, was {}",
            input.subtitle,
            input.rect.width()
        );
    }

    #[test]
    fn test_compact_ranks_closes_gaps() {
        let mut ranks = vec![0, 3, 3, 5];
        super::compact_ranks(&mut ranks);
        assert_eq!(ranks, [0, 1, 1, 2]);

        // Already contiguous ranks are left alone.
        let mut ranks = vec![2, 0, 1, 2];
        super::compact_ranks(&mut ranks);
        assert_eq!(ranks, [2, 0, 1, 2]);

        let mut ranks: Vec<usize> = Vec::new();
        super::compact_ranks(&mut ranks);
        assert!(ranks.is_empty());
    }

    #[test]
    fn test_empty_graph() {
        let layout = layout_graph(
            model(GraphProto::default()).root(),
            None,
            &LayoutOptions::default(),
        );
        assert!(layout.nodes.is_empty());
        assert!(layout.edges.is_empty());
    }
}
