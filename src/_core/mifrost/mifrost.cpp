#include <nanobind/nanobind.h>

#include "mifrost/bindings.hpp"

namespace nb = nanobind;

namespace mifrost {

NB_MODULE(_core, m)
{
   init_common(m);
   init_relation_formatter(m);
   init_schema(m);
   init_color_encoder(m);
   init_hgraph_encoder(m);
   init_successor_encoders(m);
   init_transition_dag(m);
   init_horizon_encoder(m);
}

}  // namespace mifrost
