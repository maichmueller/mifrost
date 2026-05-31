#include <nanobind/nanobind.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "mifrost/core/map_view.hpp"
#include "mifrost/core/nb_instance.hpp"
#include "mifrost/core/schema.hpp"
#include "mifrost/schema_python.hpp"

namespace nb = nanobind;

namespace mifrost {

void init_schema(nb::module_& m)
{
   register_mapview_maybe< absl::btree_map< std::string, bool > >(m);

   nb::enum_< GraphFieldDType >(m, "GraphFieldDType")
      .value("F32", GraphFieldDType::F32)
      .value("I64", GraphFieldDType::I64);

   nb::enum_< GraphFieldMode >(m, "GraphFieldMode")
      .value("STACK", GraphFieldMode::STACK)
      .value("CAT", GraphFieldMode::CAT)
      .value("RAGGED_CAT", GraphFieldMode::RAGGED_CAT)
      .value("CONST", GraphFieldMode::CONST);

   nb::enum_< GraphFieldInc::Kind >(m, "GraphFieldIncKind")
      .value("NONE", GraphFieldInc::Kind::NONE)
      .value("NODE_OFFSET", GraphFieldInc::Kind::NODE_OFFSET)
      .value("FIELD_OFFSET", GraphFieldInc::Kind::FIELD_OFFSET);

   nb::class_< GraphFieldInc >(m, "GraphFieldInc")
      .def(nb::init<>())
      .def_rw("kind", &GraphFieldInc::kind)
      .def_rw("node_type", &GraphFieldInc::node_type)
      .def_rw("field_key", &GraphFieldInc::field_key);

   nb::class_< EdgeType >(m, "EdgeType")
      .def(nb::init<>())
      .def_rw("src", &EdgeType::src)
      .def_rw("rel", &EdgeType::rel)
      .def_rw("dst", &EdgeType::dst);

   nb::class_< NodeTensorSpec >(m, "NodeTensorSpec")
      .def(nb::init<>())
      .def_rw("node_type", &NodeTensorSpec::node_type)
      .def_rw("attr", &NodeTensorSpec::attr)
      .def_rw("key", &NodeTensorSpec::key);

   nb::class_< EdgeTensorSpec >(m, "EdgeTensorSpec")
      .def(nb::init<>())
      .def_rw("edge_type", &EdgeTensorSpec::edge_type)
      .def_rw("attr", &EdgeTensorSpec::attr)
      .def_rw("key", &EdgeTensorSpec::key)
      .def_rw("part", &EdgeTensorSpec::part);

   nb::class_< GraphTensorSpec >(m, "GraphTensorSpec")
      .def(nb::init<>())
      .def_rw("attr", &GraphTensorSpec::attr)
      .def_rw("key", &GraphTensorSpec::key)
      .def_rw("ptr_key", &GraphTensorSpec::ptr_key)
      .def_rw("mode", &GraphTensorSpec::mode)
      .def_rw("dtype", &GraphTensorSpec::dtype)
      .def_rw("dim", &GraphTensorSpec::dim)
      .def_rw("cat_dim", &GraphTensorSpec::cat_dim)
      .def_rw("inc", &GraphTensorSpec::inc);

   auto schema_cls = nb::class_< Schema >(m, "Schema")
                        .def(nb::init<>())
                        .def_rw("version", &Schema::version)
                        .def_rw("graph_kind", &Schema::graph_kind)
                        .def_rw("node_types", &Schema::node_types)
                        .def_rw("edge_types", &Schema::edge_types)
                        .def_rw("node_tensors", &Schema::node_tensors)
                        .def_rw("edge_tensors", &Schema::edge_tensors)
                        .def_rw("graph_tensors", &Schema::graph_tensors)
                        .def_rw("flags", &Schema::flags)
                        .def(
                           "flags_view",
                           [](nb::handle self) {
                              auto* schema = require_instance_ptr< Schema >(
                                 self, "Schema.flags_view called with invalid instance"
                              );
                              return make_map_view(schema->flags, self);
                           },
                           nb::rv_policy::move
                        )
                        .def("validate", &Schema::validate)
                        .def("to_dict", [](const Schema& schema) { return schema_to_dict(schema); })
                        .def_static("from_dict", [](const nb::dict& schema) {
                           return schema_from_dict(schema);
                        });

   schema_cls.attr("__mifrost_map_view_methods__") = nb::make_tuple("flags_view");
}

}  // namespace mifrost
