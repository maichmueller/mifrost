#pragma once

#if defined(MIFROST_ENABLE_PYTHON_API)

   #include <nanobind/nanobind.h>

   #include "mifrost/core/batch_builder.hpp"

namespace nb = nanobind;

namespace mifrost {

nb::dict batch_builder_build_dict(BatchBuilder& builder);

nb::object batch_builder_build_pyg(BatchBuilder& builder);

}  // namespace mifrost

#endif
