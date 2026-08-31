#!/usr/bin/env python3
"""Generate the table of operator input and output names.

A model records only the values wired to each operator, never what the operator
calls them, so the names have to come from the ONNX operator schemas. They are
generated from the operator documentation in a checkout of the ONNX repository,
which is itself generated from those schemas.

    tools/generate-op-schemas.py ~/other/onnx > src/op_schema/table.rs
"""

import pathlib
import re
import sys


def parse(path, domain):
    """Read one operator document into {(domain, op): {"Inputs": [...], ...}}."""
    ops = {}
    # Each operator is introduced by an anchor naming it.
    sections = re.split(r'\n### <a name="([^"]+)">', path.read_text())
    for name, body in zip(sections[1::2], sections[2::2]):
        # The ai.onnx.ml document qualifies its anchors with the domain.
        op = name.split(".")[-1]
        entry = {}
        for kind in ("Inputs", "Outputs"):
            section = re.search(
                rf"#### {kind}[^\n]*\n(.*?)(?=\n####|\n### |\Z)", body, re.S
            )
            params = []
            if section:
                # eg. "<dt><tt>B</tt> (optional, differentiable) : T</dt>"
                for entry_tag in re.finditer(
                    r"<dt><tt>([^<]+)</tt>([^:<]*):", section.group(1)
                ):
                    param, markers = entry_tag.group(1), entry_tag.group(2)
                    if "variadic" in markers:
                        arity = "Variadic"
                    elif "optional" in markers:
                        arity = "Optional"
                    else:
                        arity = "Required"
                    params.append((param, arity))
            entry[kind] = params
        if entry["Inputs"] or entry["Outputs"]:
            ops[(domain, op)] = entry
    return ops


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    repo = pathlib.Path(sys.argv[1]).expanduser()
    version = (repo / "VERSION_NUMBER").read_text().strip()

    ops = parse(repo / "docs/Operators.md", "")
    ops.update(parse(repo / "docs/Operators-ml.md", "ai.onnx.ml"))

    def params(entries):
        if not entries:
            return "&[]"
        return "&[" + ", ".join(f'("{n}", {a})' for n, a in entries) + "]"

    out = [
        "//! Operator signatures from the ONNX specification.",
        "//!",
        f"//! Generated from the operator documentation of ONNX {version} by",
        "//! `tools/generate-op-schemas.py`. Do not edit.",
        "",
        "use super::Arity::{Optional, Required, Variadic};",
        "use super::OpSchema;",
        "",
        "/// Every operator of the default and `ai.onnx.ml` domains, ordered by type",
        "/// so that a lookup can binary search.",
        "pub const SCHEMAS: &[OpSchema] = &[",
    ]
    for domain, op in sorted(ops, key=lambda key: (key[1], key[0])):
        entry = ops[(domain, op)]
        out += [
            "    OpSchema {",
            f'        domain: "{domain}",',
            f'        op_type: "{op}",',
            f"        inputs: {params(entry['Inputs'])},",
            f"        outputs: {params(entry['Outputs'])},",
            "    },",
        ]
    out += ["];", ""]

    print("\n".join(out))
    print(f"{len(ops)} operators from ONNX {version}", file=sys.stderr)


main()
