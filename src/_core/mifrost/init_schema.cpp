#include <nanobind/nanobind.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "mifrost/core/schema.hpp"

namespace nb = nanobind;

namespace mifrost {

void init_schema(nb::module_& m)
{
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

   nb::class_< Schema >(m, "Schema")
      .def(nb::init<>())
      .def_rw("version", &Schema::version)
      .def_rw("graph_kind", &Schema::graph_kind)
      .def_rw("node_types", &Schema::node_types)
      .def_rw("edge_types", &Schema::edge_types)
      .def_rw("node_tensors", &Schema::node_tensors)
      .def_rw("edge_tensors", &Schema::edge_tensors)
      .def_rw("flags", &Schema::flags)
      .def("validate", &Schema::validate)
      .def("to_dict", &Schema::to_dict)
      .def_static("from_dict", &Schema::from_dict);
}

}  // namespace mifrost
