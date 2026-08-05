#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

#include <stdexcept>
#include <string>
#include <string_view>

#include "mifrost/bindings.hpp"
#include "mifrost/capsule_bridge.hpp"
#include "mifrost/core/encoders/homo/semantic_color_encoder.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace mifrost {
namespace {

void apply_color_config_kwargs(SemanticColorEncoderConfig& config, const nb::kwargs& kwargs)
{
   for(const auto& [key_handle, value_handle] : kwargs) {
      const auto key = nb::str(key_handle);
      const auto key_view = std::string_view{key.c_str()};
      if(key_view == "edge_features") {
         config.edge_features = nb::cast< bool >(value_handle);
      } else if(key_view == "enable_global_predicate_nodes") {
         config.enable_global_predicate_nodes = nb::cast< bool >(value_handle);
      } else if(key_view == "export_node_names") {
         config.export_node_names = nb::cast< bool >(value_handle);
      } else {
         throw std::invalid_argument(
            "Unknown SemanticColorEncoderConfig kwarg '" + std::string(key_view) + "'"
         );
      }
   }
}

}  // namespace

void init_semantic_color_encoder(nb::module_& m)
{
   nb::class_< SemanticColorEncoderConfig >(m, "SemanticColorEncoderConfig")
      .def(nb::init<>())
      .def(
         "__init__",
         [](SemanticColorEncoderConfig* self, const nb::kwargs& kwargs) {
            new(self) SemanticColorEncoderConfig();
            apply_color_config_kwargs(*self, kwargs);
         }
      )
      .def_rw("edge_features", &SemanticColorEncoderConfig::edge_features)
      .def_rw(
         "enable_global_predicate_nodes", &SemanticColorEncoderConfig::enable_global_predicate_nodes
      )
      .def_rw("export_node_names", &SemanticColorEncoderConfig::export_node_names);

   nb::class_< SemanticColorEncoderEngine >(m, "SemanticColorEncoderEngine")
      .def(
         nb::init< std::vector< SemanticPredicateSpec >, SemanticColorEncoderConfig >(),
         "predicates"_a,
         "config"_a = SemanticColorEncoderConfig{}
      )
      .def_prop_ro(
         "config", &SemanticColorEncoderEngine::get_config, nb::rv_policy::reference_internal
      )
      .def_prop_ro("predicates", &SemanticColorEncoderEngine::get_predicates)
      .def(
         "encode",
         static_cast< BatchBuilder::BatchEncoding (SemanticColorEncoderEngine::*)(
            const SemanticFlatRelationInput&
         ) const >(&SemanticColorEncoderEngine::encode),
         "input"_a
      )
      .def(
         "encode",
         static_cast< void (SemanticColorEncoderEngine::*)(
            const SemanticFlatRelationInput&, BatchBuilder&
         ) const >(&SemanticColorEncoderEngine::encode),
         "input"_a,
         "builder"_a
      )
      .def("encode_batch", &SemanticColorEncoderEngine::encode_batch, "inputs"_a);

   m.def(
      "_semantic_color_config_capsule",
      [](const SemanticColorEncoderConfig& config) {
         auto* capsule = capsule_bridge::make_owned(
            SemanticColorEncoderConfig(config), capsule_bridge::color_config_name
         );
         if(capsule == nullptr) {
            throw nb::python_error();
         }
         return nb::steal< nb::object >(capsule);
      },
      "config"_a
   );
   m.def(
      "_consume_semantic_color_engine_capsule",
      [](nb::handle capsule) {
         return capsule_bridge::take< SemanticColorEncoderEngine >(
            capsule.ptr(), capsule_bridge::color_engine_name
         );
      },
      "capsule"_a
   );
}

}  // namespace mifrost
