#pragma once

#include <nanobind/nanobind.h>

namespace mifrost {

void init_relation_formatter(nanobind::module_& m);
void init_color_encoder(nanobind::module_& m);
void init_hgraph_encoders(nanobind::module_& m);
void init_transition_dag(nanobind::module_& m);
void init_schema(nanobind::module_& m);

}  // namespace mifrost
