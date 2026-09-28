"""Generate API reference pages for key mifrost modules/classes."""

from __future__ import annotations

import mkdocs_gen_files

PAGES = {
    "reference/api/index.md": """# API Reference

This section documents the main Python-facing API surfaces.

## Core Encoders

- [HGraphEncoder](hgraph.md)
- [Flat Encoders](flat.md)
- [HorizonEncoder](horizon.md)
- [Transition Encoders](transition.md)
- [ColorEncoder](color.md)
- [ILGEncoder](ilg.md)

## Derived and Pure-Python Encoders

- [Derived-Graph Encoders](derived.md)
- [LiftedTaskEncoder](lifted.md)
- [ObjectFeatureEncoder](object-feature.md)
- [Sparse Atom-Composition Encoder](sparse-atom.md)
- [Custom Encoder Toolkit](custom.md)
- [Cross-Stack Adapters](cross-stack.md)

## Types and Graph Fields

- [Graph Fields](graph-fields.md)
- [Input and Adapter Types](types.md)
""",
    "reference/api/hgraph.md": """# HGraphEncoder

::: mifrost.encoders.hgraph.HGraphEncoder
""",
    "reference/api/flat.md": """# Flat Encoders

::: mifrost.encoders.flat.FlatRelationEncoder

::: mifrost.encoders.flat_horizon.FlatHorizonEncoder

::: mifrost.encoders.flat_transition.FlatTransitionEncoder

::: mifrost.encoders.flat_transition.FlatTransitionEffectsEncoder

::: mifrost.encoders.flat_data.FlatRelationData
""",
    "reference/api/horizon.md": """# HorizonEncoder

::: mifrost.encoders.horizon.HorizonEncoder
""",
    "reference/api/transition.md": """# Transition Encoders

::: mifrost.encoders.transition.TransitionHGraphEncoder

::: mifrost.encoders.transition.TransitionEffectsHGraphEncoder
""",
    "reference/api/color.md": """# ColorEncoder

::: mifrost.encoders.color.ColorEncoder
""",
    "reference/api/ilg.md": """# ILGEncoder

::: mifrost.encoders.ilg.ILGEncoder
""",
    "reference/api/derived.md": """# Derived-Graph Encoders

::: mifrost.encoders.derived.StarGraphEncoder

::: mifrost.encoders.derived.ObjectGraphEncoder

::: mifrost.encoders.derived.AtomLineGraphEncoder

::: mifrost.encoders.derived.HypergraphIncidenceEncoder

::: mifrost.encoders.derived.TupleTensorEncoder

::: mifrost.encoders.derived.TransformerBiasEncoder
""",
    "reference/api/lifted.md": """# LiftedTaskEncoder

::: mifrost.encoders.lifted.LiftedTaskEncoder
""",
    "reference/api/object-feature.md": """# ObjectFeatureEncoder

::: mifrost.encoders.object_feature.ObjectFeatureEncoder
""",
    "reference/api/sparse-atom.md": """# Sparse Atom-Composition Encoder

::: mifrost.encoders.sparse_atom.SparseAtomCompositionEncoder

::: mifrost.encoders.sparse_atom.SparseAtomCompositionEncoding

::: mifrost.encoders.sparse_atom.SparseAtomPredicateSchema

::: mifrost.encoders.sparse_atom.SparseAtomTypeSchema

::: mifrost.encoders.sparse_atom.build_predicate_schema

::: mifrost.encoders.sparse_atom.build_type_schema

::: mifrost.encoders.sparse_atom.encode_sparse_atom_facts

::: mifrost.encoders.sparse_atom.batch_sparse_atom_encodings

::: mifrost.encoders.sparse_atom.validate_sparse_atom_composition
""",
    "reference/api/custom.md": """# Custom Encoder Toolkit

::: mifrost.encoders.custom.state_view.StateView

::: mifrost.encoders.custom.writer.GraphWriter

::: mifrost.encoders.custom.base.CustomGraphEncoder

::: mifrost.encoders.custom.base.CustomStream

::: mifrost.encoders.custom.tables.Vocabulary

::: mifrost.encoders.custom.tables.NodeTable

::: mifrost.encoders.custom.tables.EdgeSink

::: mifrost.encoders.custom.harness.assert_backend_parity

::: mifrost.encoders.custom.harness.channel_summary

::: mifrost.encoders.custom.harness.conformance_smoke
""",
    "reference/api/cross-stack.md": """# Cross-Stack Adapters

::: mifrost.encoders.cross_stack.to_dgl

::: mifrost.encoders.cross_stack.to_jraph
""",
    "reference/api/graph-fields.md": """# Graph Fields

::: mifrost.graph_fields.Mode

::: mifrost.graph_fields.DType

::: mifrost.graph_fields.Inc

::: mifrost.graph_fields.GraphFieldSpec
""",
    "reference/api/types.md": """# Input and Adapter Types

::: mifrost.encoders.types
""",
}

for path, content in PAGES.items():
    with mkdocs_gen_files.open(path, "w") as file_obj:
        file_obj.write(content)
