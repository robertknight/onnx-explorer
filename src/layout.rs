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

use std::collections::HashMap;

use egui::{Pos2, Rect, Vec2, pos2, vec2};

use crate::model::{Graph, NodeId, ValueId};

pub struct LayoutOptions {
    pub node_height: f32,
    pub min_node_width: f32,
    pub max_node_width: f32,
    /// Estimated width of one character, used to size a node from its label.
    /// Labels are elided to fit when drawn, so this only needs to be close.
    pub char_width: f32,
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
    /// A declared input of the graph.
    Input(ValueId),
    /// A declared output of the graph.
    Output(ValueId),
}

pub struct LayoutNode {
    pub kind: ItemKind,
    pub rect: Rect,
    /// Primary label: the op type, or the value name for an input or output.
    pub title: String,
    /// Secondary label, shown when zoomed in far enough to read it.
    pub subtitle: String,
}

pub struct LayoutEdge {
    /// Index into [`Layout::nodes`].
    pub from: usize,
    pub to: usize,
    pub value: ValueId,
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
    /// Edge indices incident to each node, for highlighting a selection.
    incident: Vec<Vec<usize>>,
}

impl Layout {
    /// Index of the drawn box for `id`, absent if the node was folded away.
    pub fn node_index(&self, id: NodeId) -> Option<usize> {
        self.node_by_id.get(&id).copied()
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

pub fn layout_graph(graph: &Graph, opts: &LayoutOptions) -> Layout {
    let (mut nodes, node_by_id, input_by_value, output_by_value) = collect_items(graph, opts);
    let links = collect_links(graph, &node_by_id, &input_by_value, &output_by_value);

    let ranks = assign_ranks(nodes.len(), &links);

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
        incident,
    }
}

/// A dependency between two drawn boxes.
struct Link {
    from: usize,
    to: usize,
    value: ValueId,
}

type ItemMaps = (
    Vec<LayoutNode>,
    HashMap<NodeId, usize>,
    HashMap<ValueId, usize>,
    HashMap<ValueId, usize>,
);

/// Choose which boxes to draw: graph inputs, operators, and graph outputs.
///
/// Constants are not drawn. A large model has hundreds of them and they carry
/// no structure; they are shown in the details pane of the node that reads
/// them instead.
fn collect_items(graph: &Graph, opts: &LayoutOptions) -> ItemMaps {
    let mut nodes = Vec::new();
    let mut node_by_id = HashMap::new();
    let mut input_by_value = HashMap::new();
    let mut output_by_value = HashMap::new();

    let push = |nodes: &mut Vec<LayoutNode>, kind, title: String, subtitle: String| {
        let width = (title.chars().count() as f32 * opts.char_width + 28.0)
            .clamp(opts.min_node_width, opts.max_node_width);
        let index = nodes.len();
        nodes.push(LayoutNode {
            kind,
            rect: Rect::from_min_size(Pos2::ZERO, vec2(width, opts.node_height)),
            title,
            subtitle,
        });
        index
    };

    for value_id in &graph.inputs {
        let value = graph.value(*value_id);
        if value.is_constant() {
            continue;
        }
        let index = push(
            &mut nodes,
            ItemKind::Input(*value_id),
            value.name.clone(),
            value.type_summary(),
        );
        input_by_value.insert(*value_id, index);
    }

    for node in graph.nodes() {
        if node.is_constant() {
            continue;
        }
        let index = push(
            &mut nodes,
            ItemKind::Op(node.id),
            node.op_type.clone(),
            node.name.clone(),
        );
        node_by_id.insert(node.id, index);
    }

    for value_id in &graph.outputs {
        let value = graph.value(*value_id);
        let index = push(
            &mut nodes,
            ItemKind::Output(*value_id),
            value.name.clone(),
            value.type_summary(),
        );
        output_by_value.insert(*value_id, index);
    }

    (nodes, node_by_id, input_by_value, output_by_value)
}

fn collect_links(
    graph: &Graph,
    node_by_id: &HashMap<NodeId, usize>,
    input_by_value: &HashMap<ValueId, usize>,
    output_by_value: &HashMap<ValueId, usize>,
) -> Vec<Link> {
    let mut links = Vec::new();

    for value in graph.values() {
        if value.is_constant() {
            continue;
        }

        // A value flows from whichever box defines it: the node that produces
        // it, or the box drawn for a graph input.
        let source = value
            .producer
            .and_then(|id| node_by_id.get(&id).copied())
            .or_else(|| input_by_value.get(&value.id).copied());
        let Some(source) = source else {
            continue;
        };

        for consumer in &value.consumers {
            if let Some(target) = node_by_id.get(consumer).copied() {
                links.push(Link {
                    from: source,
                    to: target,
                    value: value.id,
                });
            }
        }

        if let Some(target) = output_by_value.get(&value.id).copied() {
            if target != source {
                links.push(Link {
                    from: source,
                    to: target,
                    value: value.id,
                });
            }
        }
    }

    links
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
        for rank in 0..members.len() {
            for position in 0..members[rank].len().saturating_sub(1) {
                let left = members[rank][position];
                let right = members[rank][position + 1];
                let before = crossings(cells, &preds[left], &preds[right])
                    + crossings(cells, &succs[left], &succs[right]);
                let after = crossings(cells, &preds[right], &preds[left])
                    + crossings(cells, &succs[right], &succs[left]);
                if after < before {
                    members[rank].swap(position, position + 1);
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

            let bounds = points
                .iter()
                .fold(Rect::NOTHING, |bounds, point| bounds.union(Rect::from_center_size(*point, Vec2::ZERO)));

            LayoutEdge {
                from: link.from,
                to: link.to,
                value: link.value,
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
    use super::{isotonic, LayoutOptions, ItemKind, layout_graph};
    use crate::model::Model;
    use rten_onnx::onnx::{GraphProto, ModelProto, NodeProto, ValueInfoProto};

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

        let layout = layout_graph(model.root(), &LayoutOptions::default());

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
            nodes.push(node("Mul", &format!("leaf{i}"), &["h"], &[&format!("o{i}")]));
        }

        let model = model(GraphProto {
            node: nodes,
            input: vec![value_info("x")],
            ..Default::default()
        });

        let opts = LayoutOptions::default();
        let layout = layout_graph(model.root(), &opts);

        let mut leaves: Vec<_> = layout
            .nodes
            .iter()
            .filter(|n| n.subtitle.starts_with("leaf"))
            .map(|n| n.rect)
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

        let layout = layout_graph(model.root(), &LayoutOptions::default());
        let skip = layout
            .edges
            .iter()
            .find(|e| {
                layout.nodes[e.from].subtitle == "a" && layout.nodes[e.to].subtitle == "d"
            })
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

        let layout = layout_graph(model.root(), &LayoutOptions::default());
        assert!(
            layout
                .nodes
                .iter()
                .all(|n| n.title != "Constant"),
            "constant nodes should be folded away"
        );
        // The input and the Mul remain, connected by one edge.
        assert_eq!(layout.nodes.len(), 2);
        assert_eq!(layout.edges.len(), 1);
    }

    #[test]
    fn test_empty_graph() {
        let layout = layout_graph(model(GraphProto::default()).root(), &LayoutOptions::default());
        assert!(layout.nodes.is_empty());
        assert!(layout.edges.is_empty());
    }
}
