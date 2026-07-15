#include <nanobind/nanobind.h>

#include "mifrost/bindings.hpp"

namespace nb = nanobind;

namespace mifrost {

NB_MODULE(_pymimir_adapter, m)
{
#ifdef NDEBUG
   nb::set_leak_warnings(false);
#endif
   // Neutral types are registered in a planner-free extension. Import it even
   // for direct/stubgen loads so adapter bindings can reference those types.
   nb::module_::import_("mifrost._neutral_core");
   init_common(m);
   init_relation_formatter(m);
   init_color_encoder(m);
   init_hgraph_encoder(m);
   init_flat_encoder(m);
   init_successor_encoders(m);
   init_transition_dag(m);
   init_horizon_encoder(m);
}

}  // namespace mifrost
