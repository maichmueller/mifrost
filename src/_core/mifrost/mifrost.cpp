#include <nanobind/nanobind.h>

#include "mifrost/bindings.hpp"

namespace nb = nanobind;

namespace mifrost {

NB_MODULE(_core, m)
{
   init_relation_formatter(m);
   init_color_encoder(m);
   init_schema(m);
   init_hgraph_encoders(m);
   init_transition_dag(m);
}

}  // namespace mifrost
