#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

#include <stdexcept>
#include <string>
#include <string_view>

#include "mifrost/bindings.hpp"
#include "mifrost/capsule_bridge.hpp"
#include "mifrost/core/encoders/homo/semantic_derived_graph_encoder.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace mifrost {
namespace {

std::optional< DerivedNodeUniverse > parse_node_universe(std::string_view value)
{
   if(value == "objects_and_atoms")
      return DerivedNodeUniverse::objects_and_atoms;
   if(value == "objects_only")
      return DerivedNodeUniverse::objects_only;
   return std::nullopt;
}

std::optional< DerivedAtomExpansion > parse_atom_expansion(std::string_view value)
{
   if(value == "star")
      return DerivedAtomExpansion::star;
   if(value == "clique")
      return DerivedAtomExpansion::clique;
   if(value == "chain")
      return DerivedAtomExpansion::chain;
   if(value == "star_first")
      return DerivedAtomExpansion::star_first;
   return std::nullopt;
}

void apply_derived_config_kwargs(
   SemanticDerivedGraphEncoderConfig& config,
   const nb::kwargs& kwargs
)
{
   for(const auto& [key_handle, value_handle] : kwargs) {
      const auto key = nb::str(key_handle);
      const auto key_view = std::string_view{key.c_str()};
      if(key_view == "node_universe") {
         if(nb::isinstance< nb::str >(value_handle)) {
            const auto parsed = parse_node_universe(nb::str(value_handle).c_str());
            if(not parsed) {
               throw std::invalid_argument(
                  "Unknown SemanticDerivedGraphEncoderConfig node_universe '"
                  + std::string(nb::str(value_handle).c_str()) + "'"
               );
            }
            config.node_universe = *parsed;
         } else {
            config.node_universe = nb::cast< DerivedNodeUniverse >(value_handle);
         }
      } else if(key_view == "atom_expansion") {
         if(nb::isinstance< nb::str >(value_handle)) {
            const auto parsed = parse_atom_expansion(nb::str(value_handle).c_str());
            if(not parsed) {
               throw std::invalid_argument(
                  "Unknown SemanticDerivedGraphEncoderConfig atom_expansion '"
                  + std::string(nb::str(value_handle).c_str()) + "'"
               );
            }
            config.atom_expansion = *parsed;
         } else {
            config.atom_expansion = nb::cast< DerivedAtomExpansion >(value_handle);
         }
      } else if(key_view == "include_reverse_edges") {
         config.include_reverse_edges = nb::cast< bool >(value_handle);
      } else if(key_view == "export_node_names") {
         config.export_node_names = nb::cast< bool >(value_handle);
      } else if(key_view == "include_line_graph") {
         config.include_line_graph = nb::cast< bool >(value_handle);
      } else if(key_view == "line_graph_max_degree") {
         config.line_graph_max_degree = nb::cast< int64_t >(value_handle);
      } else if(key_view == "include_hyperedge_incidence") {
         config.include_hyperedge_incidence = nb::cast< bool >(value_handle);
      } else if(key_view == "include_tuple_tensors") {
         config.include_tuple_tensors = nb::cast< bool >(value_handle);
      } else if(key_view == "include_spd") {
         config.include_spd = nb::cast< bool >(value_handle);
      } else if(key_view == "spd_max_hops") {
         config.spd_max_hops = nb::cast< int64_t >(value_handle);
      } else {
         throw std::invalid_argument(
            "Unknown SemanticDerivedGraphEncoderConfig kwarg '" + std::string(key_view) + "'"
         );
      }
   }
}

}  // namespace

void init_semantic_derived_graph_encoder(nb::module_& m)
{
   nb::enum_< DerivedNodeUniverse >(m, "DerivedNodeUniverse", nb::is_arithmetic())
      .value("objects_and_atoms", DerivedNodeUniverse::objects_and_atoms)
      .value("objects_only", DerivedNodeUniverse::objects_only);
   nb::enum_< DerivedAtomExpansion >(m, "DerivedAtomExpansion", nb::is_arithmetic())
      .value("star", DerivedAtomExpansion::star)
      .value("clique", DerivedAtomExpansion::clique)
      .value("chain", DerivedAtomExpansion::chain)
      .value("star_first", DerivedAtomExpansion::star_first);

   nb::class_< SemanticDerivedGraphEncoderConfig >(m, "SemanticDerivedGraphEncoderConfig")
      .def(nb::init<>())
      .def(
         "__init__",
         [](SemanticDerivedGraphEncoderConfig* self, const nb::kwargs& kwargs) {
            new(self) SemanticDerivedGraphEncoderConfig();
            apply_derived_config_kwargs(*self, kwargs);
         }
      )
      .def_rw("node_universe", &SemanticDerivedGraphEncoderConfig::node_universe)
      .def_rw("atom_expansion", &SemanticDerivedGraphEncoderConfig::atom_expansion)
      .def_rw("include_reverse_edges", &SemanticDerivedGraphEncoderConfig::include_reverse_edges)
      .def_rw("export_node_names", &SemanticDerivedGraphEncoderConfig::export_node_names)
      .def_rw("include_line_graph", &SemanticDerivedGraphEncoderConfig::include_line_graph)
      .def_rw("line_graph_max_degree", &SemanticDerivedGraphEncoderConfig::line_graph_max_degree)
      .def_rw(
         "include_hyperedge_incidence",
         &SemanticDerivedGraphEncoderConfig::include_hyperedge_incidence
      )
      .def_rw("include_tuple_tensors", &SemanticDerivedGraphEncoderConfig::include_tuple_tensors)
      .def_rw("include_spd", &SemanticDerivedGraphEncoderConfig::include_spd)
      .def_rw("spd_max_hops", &SemanticDerivedGraphEncoderConfig::spd_max_hops);

   nb::class_< SemanticDerivedGraphEncoderEngine >(m, "SemanticDerivedGraphEncoderEngine")
      .def(
         nb::init< std::vector< SemanticPredicateSpec >, SemanticDerivedGraphEncoderConfig >(),
         "predicates"_a,
         "config"_a = SemanticDerivedGraphEncoderConfig{}
      )
      .def_prop_ro(
         "config", &SemanticDerivedGraphEncoderEngine::get_config, nb::rv_policy::reference_internal
      )
      .def_prop_ro("predicates", &SemanticDerivedGraphEncoderEngine::get_predicates)
      .def(
         "encode",
         static_cast< BatchBuilder::BatchEncoding (SemanticDerivedGraphEncoderEngine::*)(
            const SemanticFlatRelationInput&
         ) const >(&SemanticDerivedGraphEncoderEngine::encode),
         "input"_a
      )
      .def(
         "encode",
         static_cast< void (SemanticDerivedGraphEncoderEngine::*)(
            const SemanticFlatRelationInput&, BatchBuilder&
         ) const >(&SemanticDerivedGraphEncoderEngine::encode),
         "input"_a,
         "builder"_a
      )
      .def(
         "encode_batch",
         static_cast< BatchBuilder::BatchEncoding (SemanticDerivedGraphEncoderEngine::*)(
            const std::vector< SemanticFlatRelationInput >&
         ) const >(&SemanticDerivedGraphEncoderEngine::encode_batch),
         "inputs"_a
      );

   m.def(
      "_semantic_derived_config_capsule",
      [](const SemanticDerivedGraphEncoderConfig& config) {
         auto* capsule = capsule_bridge::make_owned(
            SemanticDerivedGraphEncoderConfig(config), capsule_bridge::derived_config_name
         );
         if(capsule == nullptr) {
            throw nb::python_error();
         }
         return nb::steal< nb::object >(capsule);
      },
      "config"_a
   );
   m.def(
      "_consume_semantic_derived_engine_capsule",
      [](nb::handle capsule) {
         return capsule_bridge::take< SemanticDerivedGraphEncoderEngine >(
            capsule.ptr(), capsule_bridge::derived_engine_name
         );
      },
      "capsule"_a
   );
}

}  // namespace mifrost
