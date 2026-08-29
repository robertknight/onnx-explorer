//! Grouping of graph nodes by the structure implied by their names.
//!
//! Exporters commonly name nodes after the module that produced them, written
//! like a path: `/layers.1/attn/MatMul` comes from the `attn` submodule of
//! `layers.1`. The last segment names the operator itself; everything before it
//! describes where the operator sits in the model.
//!
//! Recovering that tree gives a way to read a large model at the level of
//! blocks rather than individual operators. Not every model is named this way,
//! so [`Hierarchy::build`] returns `None` when the names carry no usable
//! structure and the graph should be shown as plain operators instead.
//!
//! Blocks that would hold a single node are not created; see
//! [`MIN_GROUP_NODES`].

use std::collections::HashMap;

use crate::model::{Graph, NodeId};

/// Index of a [`Group`] within a [`Hierarchy`].
#[derive(Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct GroupId(pub u32);

/// Smallest share of named nodes that must carry a module path before the
/// hierarchy is considered real rather than incidental.
const MIN_NAMED_FRACTION: f64 = 0.3;

/// Fewest operators a block may hold.
///
/// A block standing for a single operator hides nothing and costs a level of
/// navigation to see through, so such a node is left in the enclosing block
/// instead. This cascades: a block holding nothing but a one-operator block is
/// itself down to one operator, and is dropped in turn.
const MIN_GROUP_NODES: usize = 2;

pub struct Group {
    pub parent: Option<GroupId>,
    /// This group's own path segment, eg. `attn`. Empty for the root.
    pub name: String,
    /// Full path, eg. `/layers.1/attn`. Empty for the root.
    pub path: String,
    pub children: Vec<GroupId>,
    /// Operators belonging to this group but not to any of its children.
    ///
    /// Constants are excluded. They are folded into the nodes that read them
    /// rather than drawn, so counting them would describe blocks by work that
    /// is never shown.
    pub nodes: Vec<NodeId>,
    /// Operators in this group and all of its descendants.
    pub total_nodes: usize,
    pub depth: usize,
}

pub struct Hierarchy {
    groups: Vec<Group>,
    /// The group directly containing each node, indexed by node index.
    node_group: Vec<GroupId>,
}

/// Where a node sits relative to some scope.
pub enum Placement {
    /// Directly inside the scope, so it is drawn as an operator.
    Direct,
    /// Inside this child group of the scope, which stands in for it.
    Within(GroupId),
    /// Not underneath the scope at all.
    Outside,
}

impl Hierarchy {
    /// Build the group tree for a graph.
    ///
    /// Returns `None` when node names carry no usable structure, either
    /// because too few are path-like or because they all fall into one group.
    pub fn build(graph: &Graph) -> Option<Hierarchy> {
        let mut groups = vec![Group {
            parent: None,
            name: String::new(),
            path: String::new(),
            children: Vec::new(),
            nodes: Vec::new(),
            total_nodes: 0,
            depth: 0,
        }];
        let mut by_path: HashMap<(GroupId, String), GroupId> = HashMap::new();
        let mut node_group = vec![GroupId(0); graph.nodes().len()];

        let mut named = 0usize;
        let mut with_path = 0usize;

        // How many nodes fall under each module path, counted before any
        // group is created so that the ones not worth having can be skipped.
        let mut population: HashMap<String, usize> = HashMap::new();
        let mut path = String::new();

        for node in graph.nodes() {
            let segments = module_path(&node.name);

            // Only names the model actually supplied say anything about
            // structure. Synthesized names never contain a path.
            if node.named {
                named += 1;
                if !segments.is_empty() {
                    with_path += 1;
                }
            }

            // Constants are folded into their consumers rather than drawn,
            // so they do not count towards a block being worth having.
            if node.is_constant() {
                continue;
            }

            path.clear();
            for segment in segments {
                path.push('/');
                path.push_str(segment);
                match population.get_mut(&path) {
                    Some(count) => *count += 1,
                    None => {
                        population.insert(path.clone(), 1);
                    }
                }
            }
        }

        for node in graph.nodes() {
            let mut current = GroupId(0);
            path.clear();

            for segment in module_path(&node.name) {
                path.push('/');
                path.push_str(segment);
                // A path can only lose operators as it lengthens, so once one
                // is too sparse to be worth a block, everything below it is
                // too. A path with no operators at all is absent entirely,
                // which a constant's own path may well be.
                if population.get(&path).copied().unwrap_or(0) < MIN_GROUP_NODES {
                    break;
                }
                current = child_group(&mut groups, &mut by_path, current, segment);
            }

            if !node.is_constant() {
                groups[current.0 as usize].nodes.push(node.id);
            }
            // Recorded for every node, so that the group of any node can be
            // asked for even when it is not one that gets drawn.
            node_group[node.id.0 as usize] = current;
        }

        if named == 0 || groups.len() < 2 {
            return None;
        }
        if (with_path as f64) / (named as f64) < MIN_NAMED_FRACTION {
            return None;
        }

        // Groups are always created before their children, so a reverse pass
        // sees every child's total before it needs it.
        for index in (0..groups.len()).rev() {
            let descendants: usize = groups[index]
                .children
                .iter()
                .map(|child| groups[child.0 as usize].total_nodes)
                .sum();
            groups[index].total_nodes = groups[index].nodes.len() + descendants;
        }

        Some(Hierarchy { groups, node_group })
    }

    pub fn root(&self) -> GroupId {
        GroupId(0)
    }

    pub fn group(&self, id: GroupId) -> &Group {
        &self.groups[id.0 as usize]
    }

    pub fn group_count(&self) -> usize {
        self.groups.len()
    }

    /// The group that directly contains `node`.
    pub fn group_of(&self, node: NodeId) -> GroupId {
        self.node_group[node.0 as usize]
    }

    /// Chain of groups from the root down to `id`, for breadcrumbs.
    pub fn path_to(&self, id: GroupId) -> Vec<GroupId> {
        let mut path = vec![id];
        let mut current = id;
        while let Some(parent) = self.group(current).parent {
            path.push(parent);
            current = parent;
        }
        path.reverse();
        path
    }

    /// Locate `node` relative to `scope`: drawn directly, represented by one of
    /// the scope's child groups, or outside the scope entirely.
    pub fn placement(&self, scope: GroupId, node: NodeId) -> Placement {
        let mut current = self.group_of(node);
        if current == scope {
            return Placement::Direct;
        }
        loop {
            match self.group(current).parent {
                Some(parent) if parent == scope => return Placement::Within(current),
                Some(parent) => current = parent,
                None => return Placement::Outside,
            }
        }
    }
}

/// Split a node name into the module path leading up to it.
///
/// The final segment names the operator rather than a module, so it is
/// dropped: `/layers.1/attn/MatMul` describes the module path `layers.1/attn`.
fn module_path(name: &str) -> Vec<&str> {
    let trimmed = name.trim_start_matches('/');
    if trimmed.is_empty() {
        return Vec::new();
    }
    let mut segments: Vec<&str> = trimmed.split('/').collect();
    segments.pop();
    segments.retain(|segment| !segment.is_empty());
    segments
}

fn child_group(
    groups: &mut Vec<Group>,
    by_path: &mut HashMap<(GroupId, String), GroupId>,
    parent: GroupId,
    name: &str,
) -> GroupId {
    let key = (parent, name.to_string());
    if let Some(existing) = by_path.get(&key) {
        return *existing;
    }

    let id = GroupId(groups.len() as u32);
    let parent_group = &groups[parent.0 as usize];
    let path = format!("{}/{}", parent_group.path, name);
    let depth = parent_group.depth + 1;

    groups.push(Group {
        parent: Some(parent),
        name: name.to_string(),
        path,
        children: Vec::new(),
        nodes: Vec::new(),
        total_nodes: 0,
        depth,
    });
    groups[parent.0 as usize].children.push(id);
    by_path.insert(key, id);
    id
}

#[cfg(test)]
mod tests {
    use super::{Hierarchy, Placement, module_path};
    use crate::model::Model;
    use rten_onnx::onnx::{
        AttributeProto, DataType, GraphProto, ModelProto, NodeProto, TensorProto,
    };

    fn node(op_type: &str, name: &str) -> NodeProto {
        NodeProto {
            op_type: Some(op_type.to_string()),
            name: if name.is_empty() {
                None
            } else {
                Some(name.to_string())
            },
            ..Default::default()
        }
    }

    /// A `Constant` node, which is folded into its consumers rather than
    /// drawn.
    fn constant(name: &str) -> NodeProto {
        NodeProto {
            op_type: Some("Constant".to_string()),
            name: Some(name.to_string()),
            output: vec![name.to_string()],
            attribute: vec![AttributeProto {
                name: Some("value".to_string()),
                t: Some(TensorProto {
                    name: Some(name.to_string()),
                    dims: vec![1],
                    data_type: Some(DataType::FLOAT),
                    ..Default::default()
                }),
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    fn model(nodes: Vec<NodeProto>) -> Model {
        Model::from_proto(ModelProto {
            graph: Some(GraphProto {
                node: nodes,
                ..Default::default()
            }),
            ..Default::default()
        })
    }

    #[test]
    fn test_module_path_drops_the_operator_segment() {
        assert_eq!(module_path("/layers.1/attn/MatMul"), ["layers.1", "attn"]);
        // A name with no path describes a node at the top level.
        assert_eq!(module_path("/Shape"), Vec::<&str>::new());
        assert_eq!(module_path("Relu_0"), Vec::<&str>::new());
        assert_eq!(module_path(""), Vec::<&str>::new());
    }

    #[test]
    fn test_builds_tree_from_names() {
        let model = model(vec![
            node("Shape", "/Shape"),
            node("MatMul", "/layers.0/attn/MatMul"),
            node("Add", "/layers.0/attn/Add"),
            node("Mul", "/layers.0/mlp/Mul"),
            node("Add", "/layers.0/mlp/Add"),
            node("MatMul", "/layers.1/attn/MatMul"),
            node("Add", "/layers.1/attn/Add"),
        ]);
        let hierarchy = Hierarchy::build(model.root()).expect("names are hierarchical");

        let root = hierarchy.group(hierarchy.root());
        // `/Shape` belongs to the root; the rest are nested.
        assert_eq!(root.nodes.len(), 1);
        assert_eq!(root.total_nodes, 7);
        assert_eq!(root.children.len(), 2);

        let layers0 = hierarchy.group(root.children[0]);
        assert_eq!(layers0.name, "layers.0");
        assert_eq!(layers0.path, "/layers.0");
        assert_eq!(layers0.total_nodes, 4);
        // Its own nodes all sit in children, not directly in it.
        assert!(layers0.nodes.is_empty());
        assert_eq!(layers0.children.len(), 2);

        let attn = hierarchy.group(layers0.children[0]);
        assert_eq!(attn.path, "/layers.0/attn");
        assert_eq!(attn.nodes.len(), 2);
        assert_eq!(attn.depth, 2);
        // Nothing nested inside, so entering it shows operators.
        assert!(attn.children.is_empty());
    }

    #[test]
    fn test_placement_relative_to_a_scope() {
        let model = model(vec![
            node("Shape", "/Shape"),
            node("MatMul", "/layers.0/attn/MatMul"),
            node("Add", "/layers.0/attn/Add"),
            node("Mul", "/layers.1/mlp/Mul"),
            node("Add", "/layers.1/mlp/Add"),
        ]);
        let graph = model.root();
        let hierarchy = Hierarchy::build(graph).unwrap();
        let root = hierarchy.root();

        let top = graph.nodes()[0].id;
        let attn_node = graph.nodes()[1].id;
        let other_layer = graph.nodes()[3].id;

        // From the root, `/Shape` is drawn directly and the rest are folded
        // into their top-level blocks.
        assert!(matches!(hierarchy.placement(root, top), Placement::Direct));
        let layers0 = hierarchy.group(root).children[0];
        assert!(matches!(
            hierarchy.placement(root, attn_node),
            Placement::Within(id) if id == layers0
        ));

        // From inside `/layers.0`, a node in `/layers.1` is out of scope.
        assert!(matches!(
            hierarchy.placement(layers0, other_layer),
            Placement::Outside
        ));
        // And one level further down it is drawn directly.
        let attn = hierarchy.group(layers0).children[0];
        assert!(matches!(
            hierarchy.placement(attn, attn_node),
            Placement::Direct
        ));
    }

    #[test]
    fn test_rejects_models_without_hierarchical_names() {
        // Names with no path at all.
        let flat = model(vec![node("Relu", "relu"), node("Conv", "conv")]);
        assert!(Hierarchy::build(flat.root()).is_none());

        // Unnamed nodes, which are given synthesized names.
        let unnamed = model(vec![node("Relu", ""), node("Conv", "")]);
        assert!(Hierarchy::build(unnamed.root()).is_none());

        // Too few path-like names to be meaningful.
        let mut nodes = vec![node("MatMul", "/block/MatMul")];
        for i in 0..20 {
            nodes.push(node("Relu", &format!("relu_{i}")));
        }
        assert!(Hierarchy::build(model(nodes).root()).is_none());

        // Path-like names, but every block would hold a single node, so there
        // is no grouping left to offer.
        let singles = model(vec![node("Relu", "/a/Relu"), node("Conv", "/b/Conv")]);
        assert!(Hierarchy::build(singles.root()).is_none());
    }

    #[test]
    fn test_single_node_blocks_are_not_created() {
        let model = model(vec![
            node("MatMul", "/layers.0/attn/MatMul"),
            node("Add", "/layers.0/attn/Add"),
            node("Mul", "/layers.0/mlp/Mul"),
        ]);
        let hierarchy = Hierarchy::build(model.root()).unwrap();

        let layers0 = hierarchy.group(hierarchy.root()).children[0];
        let children: Vec<&str> = hierarchy
            .group(layers0)
            .children
            .iter()
            .map(|id| hierarchy.group(*id).name.as_str())
            .collect();

        // `mlp` holds one node, so it is not worth a block of its own.
        assert_eq!(children, ["attn"]);
        // That node is drawn in the enclosing block instead.
        assert_eq!(hierarchy.group(layers0).nodes.len(), 1);
        assert_eq!(hierarchy.group_of(model.root().nodes()[2].id), layers0);
    }

    #[test]
    fn test_constants_do_not_count_towards_a_block() {
        let model = model(vec![
            node("MatMul", "/layers.0/attn/MatMul"),
            node("Add", "/layers.0/attn/Add"),
            // One operator and two constants. The constants are folded into
            // whatever reads them, so `mlp` is still a one-operator block.
            node("Mul", "/layers.0/mlp/Mul"),
            constant("/layers.0/mlp/Constant"),
            constant("/layers.0/mlp/Constant_1"),
        ]);
        let hierarchy = Hierarchy::build(model.root()).unwrap();

        let layers0 = hierarchy.group(hierarchy.root()).children[0];
        let children: Vec<&str> = hierarchy
            .group(layers0)
            .children
            .iter()
            .map(|id| hierarchy.group(*id).name.as_str())
            .collect();
        assert_eq!(children, ["attn"]);

        // Three operators, not five nodes.
        assert_eq!(hierarchy.group(layers0).total_nodes, 3);
        assert_eq!(hierarchy.group(layers0).nodes.len(), 1);
    }

    #[test]
    fn test_dropping_single_node_blocks_cascades() {
        let model = model(vec![
            // The only node under `embed`, so neither `embed` nor `norm` is
            // worth a block and it rises to the top level.
            node("Cast", "/embed/norm/Cast"),
            node("MatMul", "/layers.0/attn/MatMul"),
            node("Add", "/layers.0/attn/Add"),
        ]);
        let hierarchy = Hierarchy::build(model.root()).unwrap();
        let root = hierarchy.root();

        assert_eq!(hierarchy.group_of(model.root().nodes()[0].id), root);
        let children: Vec<&str> = hierarchy
            .group(root)
            .children
            .iter()
            .map(|id| hierarchy.group(*id).name.as_str())
            .collect();
        assert_eq!(children, ["layers.0"]);
    }

    #[test]
    fn test_path_to_walks_back_to_the_root() {
        let model = model(vec![
            node("MatMul", "/layers.0/attn/MatMul"),
            node("Add", "/layers.0/attn/Add"),
        ]);
        let hierarchy = Hierarchy::build(model.root()).unwrap();
        let attn = hierarchy.group_of(model.root().nodes()[0].id);

        let path = hierarchy.path_to(attn);
        let names: Vec<&str> = path
            .iter()
            .map(|id| hierarchy.group(*id).name.as_str())
            .collect();
        assert_eq!(names, ["", "layers.0", "attn"]);
    }
}
