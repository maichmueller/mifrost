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
