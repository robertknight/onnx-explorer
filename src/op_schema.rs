//! What an operator calls its inputs and outputs.
//!
//! A model says which value is wired to each position of an operator, but never
//! what the operator calls that position: `NodeProto` holds an `input` and an
//! `output` list and nothing else. The names — `X`, `W` and `B` for a `Conv` —
//! are part of the operator's schema, so they are carried in [`table`], which
//! is generated from the ONNX specification.

mod table;

/// How many values a formal parameter stands for.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Arity {
    Required,
    /// May be left out. ONNX marks an omitted parameter with an empty name,
    /// keeping the ones after it in place, so positions still line up.
    Optional,
    /// Stands for any number of values, and so comes last.
    Variadic,
}

/// The signature of one operator.
pub struct OpSchema {
    /// Operator domain. The empty string is the default ONNX domain.
    pub domain: &'static str,
    pub op_type: &'static str,
    pub inputs: &'static [(&'static str, Arity)],
    pub outputs: &'static [(&'static str, Arity)],
}

/// Find the signature of an operator, if it is one this build knows.
///
/// Operators from other domains — the `com.microsoft` fusions, or anything a
/// runtime has added of its own — have no published signature here, and their
/// parameters stay unnamed.
pub fn lookup(domain: &str, op_type: &str) -> Option<&'static OpSchema> {
    let domain = normalize(domain);
    let start = table::SCHEMAS
        .binary_search_by(|schema| schema.op_type.cmp(op_type))
        .ok()?;

    // Binary search lands anywhere among operators sharing a type, so walk out
    // to find the one in the right domain.
    let same_type = |index: usize| table::SCHEMAS[index].op_type == op_type;
    let first = (0..=start).rev().take_while(|i| same_type(*i)).last()?;
    table::SCHEMAS[first..]
        .iter()
        .take_while(|schema| schema.op_type == op_type)
        .find(|schema| schema.domain == domain)
}

/// The name of the parameter at `index`, as the specification writes it.
///
/// A variadic parameter covers every position from its own onwards, so it is
/// numbered off its start: the third input of a `Concat` is `inputs[2]`.
pub fn parameter(params: &[(&'static str, Arity)], index: usize) -> Option<String> {
    if let Some((name, arity)) = params.get(index)
        && *arity != Arity::Variadic
    {
        return Some((*name).to_string());
    }

    match params.last() {
        Some((name, Arity::Variadic)) => {
            let first = params.len() - 1;
            Some(format!("{name}[{}]", index - first))
        }
        _ => None,
    }
}

/// Treat the explicit spelling of the default domain as the default domain.
fn normalize(domain: &str) -> &str {
    if domain == "ai.onnx" { "" } else { domain }
}

#[cfg(test)]
mod tests {
    use super::{Arity, lookup, parameter, table};

    #[test]
    fn test_table_is_sorted_for_binary_search() {
        assert!(
            table::SCHEMAS
                .windows(2)
                .all(|w| w[0].op_type <= w[1].op_type),
            "the generated table must be ordered by operator type"
        );
    }

    #[test]
    fn test_looks_up_operators() {
        let conv = lookup("", "Conv").expect("Conv is a standard operator");
        assert_eq!(conv.inputs[0], ("X", Arity::Required));
        assert_eq!(conv.inputs[2], ("B", Arity::Optional));
        assert_eq!(conv.outputs[0], ("Y", Arity::Required));

        // The default domain may be written out in full.
        assert!(lookup("ai.onnx", "Conv").is_some());
        // Operators of other domains have no published signature here.
        assert!(lookup("com.microsoft", "Attention").is_none());
        assert!(lookup("", "NoSuchOperator").is_none());
        // An operator that only exists in the ai.onnx.ml domain.
        assert!(lookup("ai.onnx.ml", "LinearClassifier").is_some());
        assert!(lookup("", "LinearClassifier").is_none());
    }

    #[test]
    fn test_names_parameters() {
        let conv = lookup("", "Conv").unwrap();
        assert_eq!(parameter(conv.inputs, 0).as_deref(), Some("X"));
        assert_eq!(parameter(conv.inputs, 2).as_deref(), Some("B"));
        // More values than the operator declares, which a malformed model can
        // do, leaves the extra ones unnamed.
        assert_eq!(parameter(conv.inputs, 3), None);

        // A variadic parameter is numbered from where it starts.
        let concat = lookup("", "Concat").unwrap();
        assert_eq!(parameter(concat.inputs, 0).as_deref(), Some("inputs[0]"));
        assert_eq!(parameter(concat.inputs, 7).as_deref(), Some("inputs[7]"));

        // `Loop` has two ordinary inputs before its variadic one.
        let loop_op = lookup("", "Loop").unwrap();
        assert_eq!(parameter(loop_op.inputs, 0).as_deref(), Some("M"));
        assert_eq!(parameter(loop_op.inputs, 1).as_deref(), Some("cond"));
        assert_eq!(
            parameter(loop_op.inputs, 3).as_deref(),
            Some("v_initial[1]")
        );

        assert_eq!(parameter(&[], 0), None);
    }
}
