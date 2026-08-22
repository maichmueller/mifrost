"""Shared encoder export policy for the package root and encoder namespace."""

from __future__ import annotations

from dataclasses import dataclass

ENCODER_OPTIONAL_DEPENDENCY_MESSAGE = (
    "mifrost encoder wrappers require optional dependencies. "
    "Install with `pip install mifrost[test]` (for tests) or "
    "`pip install mifrost[torch]` / `pip install mifrost[perf]`."
)


@dataclass(frozen=True)
class LazyEncoderExport:
    name: str
    module: str
    attr: str
    top_level: bool = True


@dataclass(frozen=True)
class DirectEncoderExport:
    name: str
    top_level: bool = False


@dataclass(frozen=True)
class EncoderExportManifest:
    lazy_exports: tuple[LazyEncoderExport, ...]
    direct_exports: tuple[DirectEncoderExport, ...]
    top_level_order: tuple[str, ...]
    optional_dependency_message: str = ENCODER_OPTIONAL_DEPENDENCY_MESSAGE

    def __post_init__(self) -> None:
        namespace_names = set(self.namespace_export_names)
        missing = set(self.top_level_order) - namespace_names
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise RuntimeError(
                f"top-level encoder exports missing from encoder namespace: {missing_text}"
            )

        top_level_names = {
            export.name for export in self.lazy_exports if export.top_level
        }
        top_level_names.update(
            export.name for export in self.direct_exports if export.top_level
        )
        unlisted = top_level_names - set(self.top_level_order)
        if unlisted:
            unlisted_text = ", ".join(sorted(unlisted))
            raise RuntimeError(
                "top-level encoder exports missing from explicit order: "
                f"{unlisted_text}"
            )
        extra = set(self.top_level_order) - top_level_names
        if extra:
            extra_text = ", ".join(sorted(extra))
            raise RuntimeError(
                "top-level encoder export order contains non-top-level names: "
                f"{extra_text}"
            )

    @property
    def lazy_imports(self) -> dict[str, tuple[str, str]]:
        return {
            export.name: (export.module, export.attr) for export in self.lazy_exports
        }

    @property
    def direct_export_names(self) -> tuple[str, ...]:
        return tuple(export.name for export in self.direct_exports)

    @property
    def namespace_export_names(self) -> tuple[str, ...]:
        return (
            *(export.name for export in self.lazy_exports),
            *self.direct_export_names,
        )

    @property
    def top_level_export_names(self) -> tuple[str, ...]:
        return self.top_level_order


ENCODER_EXPORT_MANIFEST = EncoderExportManifest(
    lazy_exports=(
        LazyEncoderExport("HGraphEncoder", ".hgraph", "HGraphEncoder"),
        LazyEncoderExport("HGraphEncoderStream", ".hgraph", "HGraphEncoderStream"),
        LazyEncoderExport(
            "HGraphMutableEncoderStream",
            ".hgraph",
            "HGraphMutableEncoderStream",
        ),
        LazyEncoderExport("HorizonEncoder", ".horizon", "HorizonEncoder"),
        LazyEncoderExport("HorizonEncoderStream", ".horizon", "HorizonEncoderStream"),
        LazyEncoderExport("ColorEncoder", ".color", "ColorEncoder"),
        LazyEncoderExport("ColorEncoderStream", ".color", "ColorEncoderStream"),
        LazyEncoderExport("StarGraphEncoder", ".derived", "StarGraphEncoder"),
        LazyEncoderExport(
            "StarGraphEncoderStream", ".derived", "StarGraphEncoderStream"
        ),
        LazyEncoderExport("ObjectGraphEncoder", ".derived", "ObjectGraphEncoder"),
        LazyEncoderExport(
            "ObjectGraphEncoderStream",
            ".derived",
            "ObjectGraphEncoderStream",
        ),
        LazyEncoderExport("AtomLineGraphEncoder", ".derived", "AtomLineGraphEncoder"),
        LazyEncoderExport(
            "AtomLineGraphEncoderStream",
            ".derived",
            "AtomLineGraphEncoderStream",
        ),
        LazyEncoderExport("FlatRelationEncoder", ".flat", "FlatRelationEncoder"),
        LazyEncoderExport(
            "FlatRelationEncoderStream",
            ".flat",
            "FlatRelationEncoderStream",
        ),
        LazyEncoderExport(
            "FlatRelationMutableEncoderStream",
            ".flat",
            "FlatRelationMutableEncoderStream",
        ),
        LazyEncoderExport("FlatHorizonEncoder", ".flat_horizon", "FlatHorizonEncoder"),
        LazyEncoderExport(
            "FlatRootedHorizonEncoder",
            ".flat_rooted_horizon",
            "FlatRootedHorizonEncoder",
        ),
        LazyEncoderExport(
            "FlatHorizonEncoderStream",
            ".flat_horizon",
            "FlatHorizonEncoderStream",
        ),
        LazyEncoderExport(
            "FlatHorizonMutableEncoderStream",
            ".flat_horizon",
            "FlatHorizonMutableEncoderStream",
        ),
        LazyEncoderExport(
            "FlatTransitionEncoder",
            ".flat_transition",
            "FlatTransitionEncoder",
        ),
        LazyEncoderExport(
            "FlatTransitionEffectsEncoder",
            ".flat_transition",
            "FlatTransitionEffectsEncoder",
        ),
        LazyEncoderExport(
            "FlatTransitionEncoderStream",
            ".flat_transition",
            "FlatTransitionEncoderStream",
        ),
        LazyEncoderExport(
            "FlatTransitionEffectsEncoderStream",
            ".flat_transition",
            "FlatTransitionEffectsEncoderStream",
        ),
        LazyEncoderExport("FlatRelationData", ".flat_data", "FlatRelationData"),
        LazyEncoderExport("FlatRelationSchema", ".flat_data", "FlatRelationSchema"),
        LazyEncoderExport(
            "TransitionHGraphEncoder",
            ".transition",
            "TransitionHGraphEncoder",
        ),
        LazyEncoderExport(
            "TransitionEffectsHGraphEncoder",
            ".transition",
            "TransitionEffectsHGraphEncoder",
        ),
        LazyEncoderExport(
            "TransitionHGraphEncoderStream",
            ".transition",
            "TransitionHGraphEncoderStream",
        ),
        LazyEncoderExport(
            "TransitionEffectsHGraphEncoderStream",
            ".transition",
            "TransitionEffectsHGraphEncoderStream",
        ),
        LazyEncoderExport("ILGEncoder", ".ilg", "ILGEncoder"),
        LazyEncoderExport("ILGEncoderStream", ".ilg", "ILGEncoderStream"),
        LazyEncoderExport("AtomStatus", ".ilg", "AtomStatus", top_level=False),
        LazyEncoderExport(
            "ExampleConstantEncoder",
            ".custom_example",
            "ExampleConstantEncoder",
            top_level=False,
        ),
        LazyEncoderExport(
            "ExampleConstantStreamEncoder",
            ".custom_example",
            "ExampleConstantStreamEncoder",
            top_level=False,
        ),
    ),
    direct_exports=(
        DirectEncoderExport("EncoderBase", top_level=True),
        DirectEncoderExport("StreamEncoderBase", top_level=True),
        DirectEncoderExport("transition_dag_from_rustworkx", top_level=True),
        DirectEncoderExport("_encoding_dict_to_pyg"),
        DirectEncoderExport("_split_goals"),
        DirectEncoderExport("encoding_to_tensors", top_level=True),
        DirectEncoderExport("to_pyg"),
        DirectEncoderExport("to_tensor_payload"),
        DirectEncoderExport("CollateSpec"),
        DirectEncoderExport("TYPE_ATTR_SEPARATOR"),
        DirectEncoderExport("EDGE_TYPE_SEPARATOR"),
        DirectEncoderExport("EDGE_INDEX_ATTR_PREFIX"),
        DirectEncoderExport("EDGE_INDEX_KEY_PREFIX"),
        DirectEncoderExport("EDGE_INDEX_SRC_COMPONENT"),
        DirectEncoderExport("EDGE_INDEX_DST_COMPONENT"),
        DirectEncoderExport("PTR_ATTR"),
        DirectEncoderExport("BATCH_ATTR"),
        DirectEncoderExport("make_type_attr_key"),
        DirectEncoderExport("make_edge_type_key"),
        DirectEncoderExport("BatchEncodingLike", top_level=True),
        DirectEncoderExport("BatchEncodingInput", top_level=True),
        DirectEncoderExport("BatchParam"),
        DirectEncoderExport("FlatEncoding", top_level=True),
        DirectEncoderExport("register_state_adapter", top_level=True),
        DirectEncoderExport("unregister_state_adapter", top_level=True),
        DirectEncoderExport("register_domain_adapter", top_level=True),
        DirectEncoderExport("unregister_domain_adapter", top_level=True),
        DirectEncoderExport("register_literal_adapter", top_level=True),
        DirectEncoderExport("unregister_literal_adapter", top_level=True),
        DirectEncoderExport("register_action_adapter", top_level=True),
        DirectEncoderExport("unregister_action_adapter", top_level=True),
    ),
    top_level_order=(
        "HGraphEncoder",
        "HGraphEncoderStream",
        "HGraphMutableEncoderStream",
        "ColorEncoder",
        "ColorEncoderStream",
        "StarGraphEncoder",
        "StarGraphEncoderStream",
        "ObjectGraphEncoder",
        "ObjectGraphEncoderStream",
        "AtomLineGraphEncoder",
        "AtomLineGraphEncoderStream",
        "FlatRelationEncoder",
        "FlatRelationEncoderStream",
        "FlatRelationMutableEncoderStream",
        "FlatHorizonEncoder",
        "FlatRootedHorizonEncoder",
        "FlatHorizonEncoderStream",
        "FlatHorizonMutableEncoderStream",
        "FlatTransitionEncoder",
        "FlatTransitionEffectsEncoder",
        "FlatTransitionEncoderStream",
        "FlatTransitionEffectsEncoderStream",
        "FlatRelationData",
        "FlatRelationSchema",
        "HorizonEncoder",
        "HorizonEncoderStream",
        "TransitionHGraphEncoder",
        "TransitionEffectsHGraphEncoder",
        "TransitionHGraphEncoderStream",
        "TransitionEffectsHGraphEncoderStream",
        "ILGEncoder",
        "ILGEncoderStream",
        "EncoderBase",
        "StreamEncoderBase",
        "transition_dag_from_rustworkx",
        "encoding_to_tensors",
        "BatchEncodingLike",
        "BatchEncodingInput",
        "FlatEncoding",
        "register_state_adapter",
        "unregister_state_adapter",
        "register_domain_adapter",
        "unregister_domain_adapter",
        "register_literal_adapter",
        "unregister_literal_adapter",
        "register_action_adapter",
        "unregister_action_adapter",
    ),
)


def encoder_lazy_imports() -> dict[str, tuple[str, str]]:
    return ENCODER_EXPORT_MANIFEST.lazy_imports


def encoder_direct_export_names() -> tuple[str, ...]:
    return ENCODER_EXPORT_MANIFEST.direct_export_names


def encoder_namespace_export_names() -> tuple[str, ...]:
    return ENCODER_EXPORT_MANIFEST.namespace_export_names


def top_level_encoder_export_names() -> tuple[str, ...]:
    return ENCODER_EXPORT_MANIFEST.top_level_export_names


ENCODER_LAZY_EXPORTS = encoder_lazy_imports()
ENCODER_DIRECT_EXPORTS = encoder_direct_export_names()
ENCODER_NAMESPACE_EXPORTS = encoder_namespace_export_names()
TOP_LEVEL_ENCODER_EXPORTS = top_level_encoder_export_names()

__all__ = [
    "ENCODER_DIRECT_EXPORTS",
    "ENCODER_EXPORT_MANIFEST",
    "ENCODER_LAZY_EXPORTS",
    "ENCODER_NAMESPACE_EXPORTS",
    "ENCODER_OPTIONAL_DEPENDENCY_MESSAGE",
    "TOP_LEVEL_ENCODER_EXPORTS",
    "DirectEncoderExport",
    "EncoderExportManifest",
    "LazyEncoderExport",
    "encoder_direct_export_names",
    "encoder_lazy_imports",
    "encoder_namespace_export_names",
    "top_level_encoder_export_names",
]
