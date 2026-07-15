#include <nanobind/nanobind.h>

#include "mifrost/bindings.hpp"

namespace nb = nanobind;

namespace mifrost {

NB_MODULE(_neutral_core, m)
{
#ifdef NDEBUG
   nb::set_leak_warnings(false);
#endif
   init_map_view(m);
   init_schema(m);
   init_batch_encoding(m);
   init_semantic_flat_encoder(m);
   init_semantic_color_encoder(m);
   init_semantic_hgraph_encoder(m);
}

}  // namespace mifrost
