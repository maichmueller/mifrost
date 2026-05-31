#pragma once

#if defined(MIFROST_ENABLE_PYTHON_API)

   #include <nanobind/nanobind.h>

   #include "mifrost/core/schema.hpp"

namespace nb = nanobind;

namespace mifrost {

[[nodiscard]] nb::dict schema_to_dict(const Schema& schema);

Schema schema_from_dict(const nb::dict& schema);

}  // namespace mifrost

#endif
