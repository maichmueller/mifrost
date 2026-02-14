#pragma once

#include <nanobind/nanobind.h>

namespace mifrost {

void init_common(nanobind::module_& m);
void init_map_view(nanobind::module_& m);
void init_relation_formatter(nanobind::module_& m);
void init_schema(nanobind::module_& m);
void init_color_encoder(nanobind::module_& m);
void init_batch_encoding(nanobind::module_& m);
void init_hgraph_encoder(nanobind::module_& m);
void init_successor_encoders(nanobind::module_& m);
void init_transition_dag(nanobind::module_& m);
void init_horizon_encoder(nanobind::module_& m);

}  // namespace mifrost
