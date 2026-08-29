//! Intermediate representation of an ONNX model.
//!
//! This layer converts the protobuf types from [`rten_onnx`] into a graph that
//! is convenient for a viewer: values are interned so that edges are cheap
//! index lookups, each value knows its producer and consumers, and subgraphs
//! (the bodies of `If`, `Loop` and `Scan` nodes) are flattened into an arena so
//! they can be navigated by ID.
//!
//! The UI binds to these types and never sees a protobuf message.

use std::collections::HashMap;
use std::fmt;

use rten_onnx::onnx::{
    self, AttributeProto, DataType, GraphProto, ModelProto, NodeProto, TensorProto, TypeProto,
    ValueInfoProto,
};

/// Index of a [`Node`] within its owning [`Graph`].
#[derive(Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct NodeId(pub u32);

/// Index of a [`Value`] within its owning [`Graph`].
#[derive(Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct ValueId(pub u32);

/// Index of a [`Graph`] within a [`Model`]'s arena.
#[derive(Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct GraphId(pub u32);

/// Reference to a value in a specific graph, used for cross-graph links.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct ValueRef {
    pub graph: GraphId,
    pub value: ValueId,
}

/// A parsed ONNX model.
pub struct Model {
    pub ir_version: Option<i64>,
    pub producer_name: Option<String>,
    pub producer_version: Option<String>,
    pub opset_imports: Vec<OpsetImport>,
    pub metadata: Vec<(String, String)>,

    graphs: Vec<Graph>,
    root: GraphId,
}

/// An entry from the model's `opset_import` field.
#[derive(Clone, Debug)]
pub struct OpsetImport {
    /// Operator domain. The empty string is the default ONNX domain.
    pub domain: String,
    pub version: Option<i64>,
}

impl OpsetImport {
    /// Domain name for display. The default domain is reported as `ai.onnx`.
    pub fn display_domain(&self) -> &str {
        if self.domain.is_empty() {
            "ai.onnx"
        } else {
            &self.domain
        }
    }
}

impl Model {
    /// Convert a parsed protobuf message into the viewer's representation.
    pub fn from_proto(proto: ModelProto) -> Model {
        let ModelProto {
            ir_version,
            graph,
            opset_import,
            metadata_props,
            producer_name,
            producer_version,
        } = proto;

        let mut builder = Builder { graphs: Vec::new() };
        let root = builder.reserve();
        builder.build_graph(graph.unwrap_or_default(), root, None, "main".to_string());

        let mut model = Model {
            ir_version,
            producer_name,
            producer_version,
            opset_imports: opset_import
                .into_iter()
                .map(|o| OpsetImport {
                    domain: o.domain.unwrap_or_default(),
                    version: o.version,
                })
                .collect(),
            metadata: metadata_props
                .into_iter()
                .filter_map(|e| Some((e.key?, e.value.unwrap_or_default())))
                .collect(),
            // Every reserved slot is filled by `build_graph` before it returns.
            graphs: builder.graphs.into_iter().map(|g| g.unwrap()).collect(),
            root,
        };
        model.resolve_outer_scope_refs();
        model
    }

    pub fn root_id(&self) -> GraphId {
        self.root
    }

    pub fn root(&self) -> &Graph {
        self.graph(self.root)
    }

    pub fn graph(&self, id: GraphId) -> &Graph {
        &self.graphs[id.0 as usize]
    }

    pub fn graphs(&self) -> impl Iterator<Item = &Graph> {
        self.graphs.iter()
    }

    pub fn graph_count(&self) -> usize {
        self.graphs.len()
    }

    /// Return the chain of graphs from the root down to `id`, for breadcrumbs.
    pub fn path_to(&self, id: GraphId) -> Vec<GraphId> {
        let mut path = vec![id];
        let mut current = id;
        while let Some(parent) = self.graph(current).parent {
            path.push(parent);
            current = parent;
        }
        path.reverse();
        path
    }

    /// Link values that a subgraph reads from an enclosing scope back to the
    /// graph that defines them.
    ///
    /// ONNX subgraphs may reference values from any enclosing graph by name.
    /// This can only be resolved once every graph has been built, so it runs as
    /// a separate pass.
    fn resolve_outer_scope_refs(&mut self) {
        let mut resolved: Vec<(GraphId, ValueId, ValueRef)> = Vec::new();

        for graph in &self.graphs {
            for value in graph.values() {
                if value.kind != ValueKind::OuterScope {
                    continue;
                }
                let mut scope = graph.parent;
                while let Some(parent_id) = scope {
                    let parent = self.graph(parent_id);
                    if let Some(target) = parent.value_by_name(&value.name) {
                        resolved.push((
                            graph.id,
                            value.id,
                            ValueRef {
                                graph: parent_id,
                                value: target.id,
                            },
                        ));
                        break;
                    }
                    scope = parent.parent;
                }
            }
        }

        for (graph_id, value_id, target) in resolved {
            self.graphs[graph_id.0 as usize].values[value_id.0 as usize].outer = Some(target);
        }
    }
}

/// A single ONNX graph: the model's main graph, or the body of a control-flow
/// operator.
pub struct Graph {
    pub id: GraphId,
    /// Enclosing graph, for subgraphs.
    pub parent: Option<GraphId>,
    /// Human-readable label, eg. `main` or `If.then_branch`.
    pub label: String,

    nodes: Vec<Node>,
    values: Vec<Value>,
    value_by_name: HashMap<String, ValueId>,

    pub inputs: Vec<ValueId>,
    pub outputs: Vec<ValueId>,
}

impl Graph {
    pub fn node(&self, id: NodeId) -> &Node {
        &self.nodes[id.0 as usize]
    }

    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }

    pub fn value(&self, id: ValueId) -> &Value {
        &self.values[id.0 as usize]
    }

    pub fn values(&self) -> &[Value] {
        &self.values
    }

    pub fn value_by_name(&self, name: &str) -> Option<&Value> {
        self.value_by_name.get(name).map(|id| self.value(*id))
    }

    /// Count of each operator type, sorted by descending frequency.
    pub fn op_type_counts(&self) -> Vec<(&str, usize)> {
        let mut counts: HashMap<&str, usize> = HashMap::new();
        for node in &self.nodes {
            *counts.entry(node.op_type.as_str()).or_default() += 1;
        }
        let mut counts: Vec<_> = counts.into_iter().collect();
        counts.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(b.0)));
        counts
    }

    /// Total number of elements stored as constants in this graph.
    ///
    /// This counts both initializers and tensor-valued attributes. Some
    /// exporters (`spox`, for example) store all weights as `Constant` node
    /// attributes and declare no initializers at all.
    pub fn parameter_count(&self) -> u64 {
        let initializers: u64 = self
            .values
            .iter()
            .filter_map(|v| v.tensor.as_ref())
            .map(|t| t.elem_count())
            .sum();
        let attributes: u64 = self
            .nodes
            .iter()
            .flat_map(|node| node.attrs.iter())
            .filter_map(|attr| match &attr.value {
                AttrValue::Tensor(tensor) => Some(tensor.elem_count()),
                _ => None,
            })
            .sum();
        initializers + attributes
    }

    /// Constant data backing `value`, whether it comes from an initializer or
    /// from the attribute of a `Constant` node that produces it.
    pub fn constant_tensor<'a>(&'a self, value: &'a Value) -> Option<&'a Tensor> {
        if let Some(tensor) = &value.tensor {
            return Some(tensor);
        }
        self.node(value.producer?).constant_tensor()
    }

    /// Subgraphs directly contained by nodes in this graph.
    pub fn subgraphs(&self) -> impl Iterator<Item = (NodeId, &str, GraphId)> {
        self.nodes.iter().flat_map(|node| {
            node.attrs.iter().filter_map(move |attr| match attr.value {
                AttrValue::Graph(id) => Some((node.id, attr.name.as_str(), id)),
                _ => None,
            })
        })
    }
}

/// An operator instance.
pub struct Node {
    pub id: NodeId,
    /// Node name. Synthesized from the op type and index if the model does not
    /// name the node, which is common in exported models.
    pub name: String,
    /// Whether `name` came from the model rather than being synthesized.
    pub named: bool,
    pub op_type: String,
    /// Operator domain. Empty for the default ONNX domain.
    pub domain: String,
    /// Inputs, in operator argument order. `None` marks an omitted optional
    /// input, which ONNX encodes as an empty name.
    pub inputs: Vec<Option<ValueId>>,
    pub outputs: Vec<Option<ValueId>>,
    pub attrs: Vec<Attribute>,
}

impl Node {
    /// Whether this node just supplies a constant, in which case a graph
    /// drawing is clearer if it is folded into its consumers rather than drawn.
    pub fn is_constant(&self) -> bool {
        self.op_type == "Constant" && self.constant_tensor().is_some()
    }

    /// The tensor produced by a `Constant` node.
    pub fn constant_tensor(&self) -> Option<&Tensor> {
        if self.op_type != "Constant" {
            return None;
        }
        self.attrs.iter().find_map(|attr| match &attr.value {
            AttrValue::Tensor(tensor) => Some(tensor),
            _ => None,
        })
    }

    /// Op type qualified by domain, eg. `com.microsoft.Attention`.
    pub fn qualified_op_type(&self) -> String {
        if self.domain.is_empty() || self.domain == "ai.onnx" {
            self.op_type.clone()
        } else {
            format!("{}.{}", self.domain, self.op_type)
        }
    }
}

/// A named tensor flowing through a graph: a graph input or output, a weight,
/// or an intermediate result.
pub struct Value {
    pub id: ValueId,
    pub name: String,
    pub kind: ValueKind,
    pub dtype: Option<DataType>,
    pub shape: Option<Shape>,
    /// Node that produces this value, if any.
    pub producer: Option<NodeId>,
    /// Nodes that read this value.
    pub consumers: Vec<NodeId>,
    /// Constant data, for initializers.
    pub tensor: Option<Tensor>,
    /// Definition in an enclosing graph, for values read from an outer scope.
    pub outer: Option<ValueRef>,
}

impl Value {
    /// Whether this value is a constant folded into its consumers rather than
    /// something to draw as a graph edge.
    pub fn is_constant(&self) -> bool {
        matches!(self.kind, ValueKind::Initializer | ValueKind::Constant)
    }

    /// Description of the value's type and shape, eg. `FLOAT[1, 3, 224, 224]`.
    pub fn type_summary(&self) -> String {
        match (&self.dtype, &self.shape) {
            (Some(dtype), Some(shape)) => format!("{dtype}{shape}"),
            (Some(dtype), None) => format!("{dtype}[?]"),
            (None, Some(shape)) => format!("?{shape}"),
            (None, None) => "?".to_string(),
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum ValueKind {
    /// Declared input of the graph.
    Input,
    /// Declared output of the graph.
    Output,
    /// Constant weight stored in the model as an initializer.
    Initializer,
    /// Output of a `Constant` node, which is also a constant weight.
    Constant,
    /// Produced and consumed within the graph.
    Intermediate,
    /// Read from an enclosing graph's scope.
    OuterScope,
}

impl ValueKind {
    pub fn label(&self) -> &'static str {
        match self {
            ValueKind::Input => "input",
            ValueKind::Output => "output",
            ValueKind::Initializer => "initializer",
            ValueKind::Constant => "constant",
            ValueKind::Intermediate => "intermediate",
            ValueKind::OuterScope => "outer scope",
        }
    }
}

/// A tensor shape, which may contain symbolic dimensions.
#[derive(Clone, Debug, PartialEq)]
pub struct Shape(pub Vec<Dim>);

#[derive(Clone, Debug, PartialEq)]
pub enum Dim {
    Fixed(i64),
    /// Symbolic dimension, eg. `batch_size`.
    Param(String),
    Unknown,
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, dim) in self.0.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            match dim {
                Dim::Fixed(n) => write!(f, "{n}")?,
                Dim::Param(name) => write!(f, "{name}")?,
                Dim::Unknown => write!(f, "?")?,
            }
        }
        write!(f, "]")
    }
}

/// Constant tensor data from an initializer or a tensor-valued attribute.
pub struct Tensor {
    pub dtype: DataType,
    pub dims: Vec<i64>,
    pub data: TensorData,
}

impl Tensor {
    pub fn elem_count(&self) -> u64 {
        self.dims.iter().map(|d| (*d).max(0) as u64).product()
    }

    pub fn shape(&self) -> Shape {
        Shape(self.dims.iter().map(|d| Dim::Fixed(*d)).collect())
    }

    /// Size of the tensor's data in bytes, where known.
    pub fn byte_len(&self) -> Option<usize> {
        match &self.data {
            TensorData::Raw(bytes) => Some(bytes.len()),
            TensorData::Floats(v) => Some(v.len() * 4),
            TensorData::Int32s(v) => Some(v.len() * 4),
            TensorData::Int64s(v) => Some(v.len() * 8),
            TensorData::Doubles(v) => Some(v.len() * 8),
            TensorData::External { .. } | TensorData::Missing => None,
        }
    }
}

/// Storage for a constant tensor.
///
/// ONNX stores tensor data either as packed little-endian bytes in `raw_data`
/// or in one of several typed repeated fields, and may instead point at an
/// external file.
pub enum TensorData {
    Raw(Vec<u8>),
    Floats(Vec<f32>),
    Int32s(Vec<i32>),
    Int64s(Vec<i64>),
    Doubles(Vec<f64>),
    /// Data lives in a separate file, described by these key/value entries.
    External { entries: Vec<(String, String)> },
    Missing,
}

/// An operator attribute.
pub struct Attribute {
    pub name: String,
    pub value: AttrValue,
}

pub enum AttrValue {
    Float(f32),
    Int(i64),
    String(String),
    Floats(Vec<f32>),
    Ints(Vec<i64>),
    Strings(Vec<String>),
    Tensor(Tensor),
    Graph(GraphId),
    /// An attribute whose type `rten_onnx` does not decode.
    Unsupported(i32),
}

impl AttrValue {
    /// Name of the attribute's type, for display.
    pub fn type_name(&self) -> &'static str {
        match self {
            AttrValue::Float(_) => "float",
            AttrValue::Int(_) => "int",
            AttrValue::String(_) => "string",
            AttrValue::Floats(_) => "float[]",
            AttrValue::Ints(_) => "int[]",
            AttrValue::Strings(_) => "string[]",
            AttrValue::Tensor(_) => "tensor",
            AttrValue::Graph(_) => "graph",
            AttrValue::Unsupported(_) => "unsupported",
        }
    }
}

impl fmt::Display for AttrValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        /// Long lists are elided; the details pane has limited width and the
        /// full contents are rarely useful at a glance.
        const MAX_ITEMS: usize = 16;

        fn write_list<T: fmt::Display>(f: &mut fmt::Formatter<'_>, items: &[T]) -> fmt::Result {
            write!(f, "[")?;
            for (i, item) in items.iter().take(MAX_ITEMS).enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{item}")?;
            }
            if items.len() > MAX_ITEMS {
                write!(f, ", … ({} total)", items.len())?;
            }
            write!(f, "]")
        }

        match self {
            AttrValue::Float(v) => write!(f, "{v}"),
            AttrValue::Int(v) => write!(f, "{v}"),
            AttrValue::String(v) => write!(f, "{v:?}"),
            AttrValue::Floats(v) => write_list(f, v),
            AttrValue::Ints(v) => write_list(f, v),
            AttrValue::Strings(v) => write_list(f, v),
            AttrValue::Tensor(t) => write!(f, "{}{}", t.dtype, t.shape()),
            AttrValue::Graph(_) => write!(f, "<subgraph>"),
            AttrValue::Unsupported(ty) => write!(f, "<unsupported attribute type {ty}>"),
        }
    }
}

/// Builds the graph arena, recursing into subgraphs.
struct Builder {
    /// Slots are reserved before a graph is built so that a subgraph can be
    /// assigned an ID while its parent is still under construction.
    graphs: Vec<Option<Graph>>,
}

impl Builder {
    fn reserve(&mut self) -> GraphId {
        self.graphs.push(None);
        GraphId((self.graphs.len() - 1) as u32)
    }

    fn build_graph(
        &mut self,
        proto: GraphProto,
        id: GraphId,
        parent: Option<GraphId>,
        label: String,
    ) {
        let mut graph = Graph {
            id,
            parent,
            label,
            nodes: Vec::with_capacity(proto.node.len()),
            values: Vec::new(),
            value_by_name: HashMap::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
        };

        for info in proto.input {
            if let Some(value_id) = intern_value_info(&mut graph, info) {
                graph.inputs.push(value_id);
            }
        }
        for info in proto.output {
            if let Some(value_id) = intern_value_info(&mut graph, info) {
                graph.outputs.push(value_id);
            }
        }
        for init in proto.initializer {
            let Some(name) = init.name.clone() else {
                continue;
            };
            let value_id = intern(&mut graph, &name);
            let tensor = tensor_from_proto(init);
            let value = &mut graph.values[value_id.0 as usize];
            value.dtype.get_or_insert(tensor.dtype);
            value.shape.get_or_insert_with(|| tensor.shape());
            value.tensor = Some(tensor);
        }
        // `value_info` carries types for intermediate values. It is optional
        // and frequently absent, in which case shapes stay unknown until we
        // run shape inference.
        for info in proto.value_info {
            intern_value_info(&mut graph, info);
        }

        for (index, node_proto) in proto.node.into_iter().enumerate() {
            let node_id = NodeId(index as u32);
            let node = self.build_node(&mut graph, node_proto, node_id, id);
            graph.nodes.push(node);
        }

        assign_value_kinds(&mut graph, parent.is_some());
        self.graphs[id.0 as usize] = Some(graph);
    }

    fn build_node(
        &mut self,
        graph: &mut Graph,
        proto: NodeProto,
        id: NodeId,
        graph_id: GraphId,
    ) -> Node {
        let NodeProto {
            domain,
            name,
            input,
            output,
            op_type,
            attribute,
        } = proto;

        let op_type = op_type.unwrap_or_else(|| "Unknown".to_string());
        let named = name.as_ref().is_some_and(|n| !n.is_empty());
        let name = match name {
            Some(name) if !name.is_empty() => name,
            // Exporters often leave nodes unnamed. A synthesized name gives the
            // UI something stable to display and search on.
            _ => format!("{}_{}", op_type, id.0),
        };

        let inputs: Vec<Option<ValueId>> = input
            .iter()
            .map(|value_name| intern_optional(graph, value_name))
            .collect();
        let outputs: Vec<Option<ValueId>> = output
            .iter()
            .map(|value_name| intern_optional(graph, value_name))
            .collect();

        for value_id in inputs.iter().flatten() {
            let consumers = &mut graph.values[value_id.0 as usize].consumers;
            // A node may read the same value more than once, but should appear
            // once in the consumer list.
            if !consumers.contains(&id) {
                consumers.push(id);
            }
        }
        for value_id in outputs.iter().flatten() {
            graph.values[value_id.0 as usize].producer = Some(id);
        }

        let attrs = attribute
            .into_iter()
            .map(|attr| self.build_attr(attr, graph_id, &op_type))
            .collect();

        Node {
            id,
            name,
            named,
            op_type,
            domain: domain.unwrap_or_default(),
            inputs,
            outputs,
            attrs,
        }
    }

    fn build_attr(
        &mut self,
        proto: AttributeProto,
        graph_id: GraphId,
        op_type: &str,
    ) -> Attribute {
        let AttributeProto {
            name,
            f,
            s,
            i,
            g,
            t,
            floats,
            ints,
            strings,
            r#type,
        } = proto;

        let name = name.unwrap_or_default();

        // Prefer the declared type, since it disambiguates empty lists. Fall
        // back to whichever field is populated, as some exporters omit it.
        let declared = r#type.map(|t| t.0).unwrap_or(0);
        let value = match declared {
            x if x == onnx::AttributeType::FLOAT.0 && f.is_some() => AttrValue::Float(f.unwrap()),
            x if x == onnx::AttributeType::INT.0 && i.is_some() => AttrValue::Int(i.unwrap()),
            x if x == onnx::AttributeType::STRING.0 && s.is_some() => AttrValue::String(s.unwrap()),
            x if x == onnx::AttributeType::FLOATS.0 => AttrValue::Floats(floats),
            x if x == onnx::AttributeType::INTS.0 => AttrValue::Ints(ints),
            x if x == onnx::AttributeType::GRAPH.0 && g.is_some() => {
                AttrValue::Graph(self.build_subgraph(g.unwrap(), graph_id, op_type, &name))
            }
            _ => {
                if let Some(g) = g {
                    AttrValue::Graph(self.build_subgraph(g, graph_id, op_type, &name))
                } else if let Some(t) = t {
                    AttrValue::Tensor(tensor_from_proto(t))
                } else if let Some(f) = f {
                    AttrValue::Float(f)
                } else if let Some(i) = i {
                    AttrValue::Int(i)
                } else if let Some(s) = s {
                    AttrValue::String(s)
                } else if !floats.is_empty() {
                    AttrValue::Floats(floats)
                } else if !ints.is_empty() {
                    AttrValue::Ints(ints)
                } else if !strings.is_empty() {
                    AttrValue::Strings(strings)
                } else {
                    AttrValue::Unsupported(declared)
                }
            }
        };

        Attribute { name, value }
    }

    fn build_subgraph(
        &mut self,
        proto: GraphProto,
        parent: GraphId,
        op_type: &str,
        attr_name: &str,
    ) -> GraphId {
        let id = self.reserve();
        self.build_graph(proto, id, Some(parent), format!("{op_type}.{attr_name}"));
        id
    }
}

/// Get or create the value with `name`, returning `None` for the empty name
/// that ONNX uses to mark an omitted optional input or output.
fn intern_optional(graph: &mut Graph, name: &str) -> Option<ValueId> {
    if name.is_empty() {
        return None;
    }
    Some(intern(graph, name))
}

fn intern(graph: &mut Graph, name: &str) -> ValueId {
    if let Some(id) = graph.value_by_name.get(name) {
        return *id;
    }
    let id = ValueId(graph.values.len() as u32);
    graph.values.push(Value {
        id,
        name: name.to_string(),
        // Replaced by `assign_value_kinds` once the graph is complete.
        kind: ValueKind::Intermediate,
        dtype: None,
        shape: None,
        producer: None,
        consumers: Vec::new(),
        tensor: None,
        outer: None,
    });
    graph.value_by_name.insert(name.to_string(), id);
    id
}

fn intern_value_info(graph: &mut Graph, info: ValueInfoProto) -> Option<ValueId> {
    let name = info.name?;
    if name.is_empty() {
        return None;
    }
    let id = intern(graph, &name);
    if let Some(ty) = info.r#type {
        let (dtype, shape) = decompose_type(ty);
        let value = &mut graph.values[id.0 as usize];
        if let Some(dtype) = dtype {
            value.dtype.get_or_insert(dtype);
        }
        if let Some(shape) = shape {
            value.shape.get_or_insert(shape);
        }
    }
    Some(id)
}

/// Extract the element type and shape from a `TypeProto`, unwrapping sequence
/// types to describe their element.
fn decompose_type(ty: TypeProto) -> (Option<DataType>, Option<Shape>) {
    if let Some(tensor) = ty.tensor_type {
        let shape = tensor.shape.map(|s| {
            Shape(
                s.dim
                    .into_iter()
                    .map(|d| match (d.dim_value, d.dim_param) {
                        (Some(n), _) => Dim::Fixed(n),
                        (None, Some(name)) if !name.is_empty() => Dim::Param(name),
                        _ => Dim::Unknown,
                    })
                    .collect(),
            )
        });
        return (tensor.elem_type, shape);
    }
    if let Some(seq) = ty.sequence {
        if let Some(elem) = seq.elem_type {
            return decompose_type(elem);
        }
    }
    (None, None)
}

fn tensor_from_proto(proto: TensorProto) -> Tensor {
    let external = proto
        .data_location
        .is_some_and(|loc| loc == onnx::DataLocation::EXTERNAL);

    let data = if external {
        TensorData::External {
            entries: proto
                .external_data
                .into_iter()
                .map(|e| (e.key.unwrap_or_default(), e.value.unwrap_or_default()))
                .collect(),
        }
    } else if let Some(raw) = proto.raw_data {
        TensorData::Raw(raw.into_inner())
    } else if !proto.float_data.is_empty() {
        TensorData::Floats(proto.float_data)
    } else if !proto.int32_data.is_empty() {
        TensorData::Int32s(proto.int32_data)
    } else if !proto.int64_data.is_empty() {
        TensorData::Int64s(proto.int64_data)
    } else if !proto.double_data.is_empty() {
        TensorData::Doubles(proto.double_data)
    } else {
        TensorData::Missing
    };

    Tensor {
        dtype: proto.data_type.unwrap_or(DataType(0)),
        dims: proto.dims,
        data,
    }
}

/// Classify every value once the graph's nodes and initializers are known.
fn assign_value_kinds(graph: &mut Graph, is_subgraph: bool) {
    let inputs: Vec<ValueId> = graph.inputs.clone();
    let outputs: Vec<ValueId> = graph.outputs.clone();

    // Disjoint field borrows: kinds are written to `values` while `nodes` is
    // read to identify constants.
    let nodes = &graph.nodes;
    for value in &mut graph.values {
        // An initializer takes precedence over an input declaration: ONNX
        // allows a value to be both, which means an input with a default, and
        // such values are best shown as constants.
        value.kind = if value.tensor.is_some() {
            ValueKind::Initializer
        } else if value
            .producer
            .is_some_and(|id| nodes[id.0 as usize].is_constant())
        {
            ValueKind::Constant
        } else if inputs.contains(&value.id) {
            ValueKind::Input
        } else if outputs.contains(&value.id) {
            ValueKind::Output
        } else if value.producer.is_none() && is_subgraph {
            // Undefined within this graph, so it must come from an enclosing
            // scope. `Model::resolve_outer_scope_refs` links it up.
            ValueKind::OuterScope
        } else {
            ValueKind::Intermediate
        };
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use rten_onnx::onnx::{
        AttributeProto, AttributeType, DataType, Dimension, GraphProto, ModelProto, NodeProto,
        TensorProto, TensorShapeProto, TypeProto, TypeProtoTensor, ValueInfoProto,
    };

    use super::{AttrValue, Dim, Model, Shape, ValueKind};

    fn node(op_type: &str, name: &str, inputs: &[&str], outputs: &[&str]) -> NodeProto {
        NodeProto {
            op_type: Some(op_type.to_string()),
            name: if name.is_empty() {
                None
            } else {
                Some(name.to_string())
            },
            input: inputs.iter().map(|s| s.to_string()).collect(),
            output: outputs.iter().map(|s| s.to_string()).collect(),
            ..Default::default()
        }
    }

    fn value_info(name: &str, dims: &[i64]) -> ValueInfoProto {
        ValueInfoProto {
            name: Some(name.to_string()),
            r#type: Some(TypeProto {
                tensor_type: Some(TypeProtoTensor {
                    elem_type: Some(DataType::FLOAT),
                    shape: Some(TensorShapeProto {
                        dim: dims
                            .iter()
                            .map(|d| Dimension {
                                dim_value: Some(*d),
                                dim_param: None,
                            })
                            .collect(),
                    }),
                }),
                sequence: None,
            }),
        }
    }

    fn initializer(name: &str, dims: &[i64]) -> TensorProto {
        let elems: i64 = dims.iter().product();
        TensorProto {
            name: Some(name.to_string()),
            dims: dims.to_vec(),
            data_type: Some(DataType::FLOAT),
            raw_data: Some(RefCell::new(vec![0u8; elems as usize * 4])),
            ..Default::default()
        }
    }

    fn model(graph: GraphProto) -> Model {
        Model::from_proto(ModelProto {
            graph: Some(graph),
            ..Default::default()
        })
    }

    #[test]
    fn test_edges_are_connected() {
        let model = model(GraphProto {
            node: vec![
                node("Relu", "relu", &["x"], &["h"]),
                node("Sigmoid", "sigmoid", &["h"], &["y"]),
            ],
            input: vec![value_info("x", &[1, 4])],
            output: vec![value_info("y", &[1, 4])],
            ..Default::default()
        });

        let graph = model.root();
        let h = graph.value_by_name("h").unwrap();
        assert_eq!(h.kind, ValueKind::Intermediate);
        assert_eq!(h.producer, Some(graph.nodes()[0].id));
        assert_eq!(h.consumers, vec![graph.nodes()[1].id]);

        let x = graph.value_by_name("x").unwrap();
        assert_eq!(x.kind, ValueKind::Input);
        assert_eq!(x.producer, None);
        assert_eq!(x.shape, Some(Shape(vec![Dim::Fixed(1), Dim::Fixed(4)])));

        assert_eq!(graph.value_by_name("y").unwrap().kind, ValueKind::Output);
    }

    #[test]
    fn test_initializers_become_constants() {
        let model = model(GraphProto {
            node: vec![node("MatMul", "matmul", &["x", "w"], &["y"])],
            initializer: vec![initializer("w", &[4, 8])],
            input: vec![value_info("x", &[1, 4])],
            output: vec![value_info("y", &[1, 8])],
            ..Default::default()
        });

        let w = model.root().value_by_name("w").unwrap();
        assert_eq!(w.kind, ValueKind::Initializer);
        assert!(w.is_constant());
        let tensor = w.tensor.as_ref().unwrap();
        assert_eq!(tensor.dims, [4, 8]);
        assert_eq!(tensor.elem_count(), 32);
        assert_eq!(tensor.byte_len(), Some(128));
        // Shape and dtype are taken from the initializer when `value_info`
        // does not describe it.
        assert_eq!(w.dtype, Some(DataType::FLOAT));
        assert_eq!(model.root().parameter_count(), 32);
    }

    #[test]
    fn test_initializer_wins_over_input_declaration() {
        // ONNX allows a value to be both a graph input and an initializer,
        // which means an input with a default value.
        let model = model(GraphProto {
            node: vec![node("Relu", "relu", &["w"], &["y"])],
            initializer: vec![initializer("w", &[2])],
            input: vec![value_info("w", &[2])],
            output: vec![value_info("y", &[2])],
            ..Default::default()
        });

        assert_eq!(
            model.root().value_by_name("w").unwrap().kind,
            ValueKind::Initializer
        );
    }

    #[test]
    fn test_omitted_optional_inputs() {
        // ONNX marks a skipped optional input with an empty name.
        let model = model(GraphProto {
            node: vec![node("Clip", "clip", &["x", "", "max"], &["y"])],
            input: vec![value_info("x", &[4])],
            output: vec![value_info("y", &[4])],
            ..Default::default()
        });

        let node = &model.root().nodes()[0];
        assert_eq!(node.inputs.len(), 3);
        assert!(node.inputs[0].is_some());
        assert!(node.inputs[1].is_none());
        assert!(node.inputs[2].is_some());
    }

    #[test]
    fn test_unnamed_nodes_get_synthesized_names() {
        let model = model(GraphProto {
            node: vec![node("Relu", "", &["x"], &["y"])],
            ..Default::default()
        });

        let node = &model.root().nodes()[0];
        assert_eq!(node.name, "Relu_0");
        assert!(!node.named);
    }

    #[test]
    fn test_repeated_input_lists_consumer_once() {
        let model = model(GraphProto {
            node: vec![node("Add", "add", &["x", "x"], &["y"])],
            input: vec![value_info("x", &[4])],
            ..Default::default()
        });

        let x = model.root().value_by_name("x").unwrap();
        assert_eq!(x.consumers.len(), 1);
    }

    #[test]
    fn test_subgraphs_are_navigable() {
        let branch = |name: &str, output: &str| GraphProto {
            node: vec![node("Identity", name, &["outer"], &[output])],
            output: vec![value_info(output, &[1])],
            ..Default::default()
        };

        let if_node = NodeProto {
            op_type: Some("If".to_string()),
            name: Some("if".to_string()),
            input: vec!["cond".to_string()],
            output: vec!["y".to_string()],
            attribute: vec![
                AttributeProto {
                    name: Some("then_branch".to_string()),
                    g: Some(branch("then_id", "t")),
                    r#type: Some(AttributeType::GRAPH),
                    ..Default::default()
                },
                AttributeProto {
                    name: Some("else_branch".to_string()),
                    g: Some(branch("else_id", "e")),
                    r#type: Some(AttributeType::GRAPH),
                    ..Default::default()
                },
            ],
            ..Default::default()
        };

        let model = model(GraphProto {
            node: vec![node("Relu", "relu", &["x"], &["outer"]), if_node],
            input: vec![value_info("x", &[1]), value_info("cond", &[1])],
            output: vec![value_info("y", &[1])],
            ..Default::default()
        });

        assert_eq!(model.graph_count(), 3);

        let subgraphs: Vec<_> = model
            .root()
            .subgraphs()
            .map(|(_node, attr, id)| (attr.to_string(), id))
            .collect();
        assert_eq!(subgraphs.len(), 2);
        assert_eq!(subgraphs[0].0, "then_branch");
        assert_eq!(subgraphs[1].0, "else_branch");

        let then_branch = model.graph(subgraphs[0].1);
        assert_eq!(then_branch.label, "If.then_branch");
        assert_eq!(then_branch.parent, Some(model.root_id()));
        assert_eq!(
            model.path_to(then_branch.id),
            vec![model.root_id(), then_branch.id]
        );

        // The attribute holds the subgraph's ID.
        let if_node = &model.root().nodes()[1];
        assert!(matches!(if_node.attrs[0].value, AttrValue::Graph(id) if id == subgraphs[0].1));
    }

    #[test]
    fn test_outer_scope_values_are_resolved() {
        let branch = GraphProto {
            node: vec![node("Identity", "id", &["outer"], &["t"])],
            output: vec![value_info("t", &[1])],
            ..Default::default()
        };
        let if_node = NodeProto {
            op_type: Some("If".to_string()),
            name: Some("if".to_string()),
            input: vec!["cond".to_string()],
            output: vec!["y".to_string()],
            attribute: vec![AttributeProto {
                name: Some("then_branch".to_string()),
                g: Some(branch),
                r#type: Some(AttributeType::GRAPH),
                ..Default::default()
            }],
            ..Default::default()
        };

        let model = model(GraphProto {
            // `outer` is produced in the parent graph and read in the subgraph.
            node: vec![node("Relu", "relu", &["x"], &["outer"]), if_node],
            input: vec![value_info("x", &[1]), value_info("cond", &[1])],
            output: vec![value_info("y", &[1])],
            ..Default::default()
        });

        let (_node, _attr, subgraph_id) = model.root().subgraphs().next().unwrap();
        let subgraph = model.graph(subgraph_id);
        let outer = subgraph.value_by_name("outer").unwrap();

        assert_eq!(outer.kind, ValueKind::OuterScope);
        let target = outer.outer.expect("outer scope value should be resolved");
        assert_eq!(target.graph, model.root_id());
        assert_eq!(
            model.graph(target.graph).value(target.value).name,
            "outer"
        );
    }

    #[test]
    fn test_symbolic_dimensions() {
        let mut info = value_info("x", &[1, 4]);
        // Replace the first dimension with a symbolic one.
        info.r#type.as_mut().unwrap().tensor_type.as_mut().unwrap()
            .shape.as_mut().unwrap().dim[0] = Dimension {
            dim_value: None,
            dim_param: Some("batch".to_string()),
        };

        let model = model(GraphProto {
            node: vec![node("Relu", "relu", &["x"], &["y"])],
            input: vec![info],
            ..Default::default()
        });

        let x = model.root().value_by_name("x").unwrap();
        let shape = x.shape.as_ref().unwrap();
        assert_eq!(
            shape.0,
            vec![Dim::Param("batch".to_string()), Dim::Fixed(4)]
        );
        assert_eq!(shape.to_string(), "[batch, 4]");
        assert_eq!(x.type_summary(), "FLOAT[batch, 4]");
    }

    #[test]
    fn test_op_type_counts_sorted_by_frequency() {
        let model = model(GraphProto {
            node: vec![
                node("Relu", "a", &["x"], &["h1"]),
                node("Conv", "b", &["h1"], &["h2"]),
                node("Relu", "c", &["h2"], &["h3"]),
                node("Relu", "d", &["h3"], &["y"]),
            ],
            ..Default::default()
        });

        assert_eq!(model.root().op_type_counts(), vec![("Relu", 3), ("Conv", 1)]);
    }

    #[test]
    fn test_constant_nodes_count_as_parameters() {
        // Some exporters store every weight as a `Constant` node attribute and
        // declare no initializers.
        let constant = NodeProto {
            op_type: Some("Constant".to_string()),
            name: Some("w".to_string()),
            output: vec!["w".to_string()],
            attribute: vec![AttributeProto {
                name: Some("value".to_string()),
                t: Some(initializer("w", &[4, 8])),
                ..Default::default()
            }],
            ..Default::default()
        };

        let model = model(GraphProto {
            node: vec![constant, node("MatMul", "matmul", &["x", "w"], &["y"])],
            input: vec![value_info("x", &[1, 4])],
            ..Default::default()
        });

        let graph = model.root();
        assert_eq!(graph.parameter_count(), 32);

        let w = graph.value_by_name("w").unwrap();
        assert_eq!(w.kind, ValueKind::Constant);
        assert!(w.is_constant());
        // The data lives on the producing node, not the value.
        assert!(w.tensor.is_none());
        assert_eq!(graph.constant_tensor(w).unwrap().elem_count(), 32);
        assert!(graph.nodes()[0].is_constant());
        assert!(!graph.nodes()[1].is_constant());
    }

    #[test]
    fn test_empty_model() {
        let model = Model::from_proto(ModelProto::default());
        assert_eq!(model.graph_count(), 1);
        assert!(model.root().nodes().is_empty());
        assert_eq!(model.root().label, "main");
    }
}
