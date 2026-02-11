#include <nanobind/nanobind.h>

#include "mifrost/bindings.hpp"
#include "mifrost/core/map_view.hpp"

namespace nb = nanobind;

namespace mifrost {

void init_map_view(nb::module_& m)
{
   register_mapview_base_maybe(m);
}

}  // namespace mifrost
