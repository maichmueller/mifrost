#pragma once

#include <nanobind/nanobind.h>

#include <cstdint>
#include <optional>

#include "mifrost/core/batch_builder.hpp"

namespace nb = nanobind;

namespace mifrost {

nb::object owner_target_device(nb::handle owner);

std::optional< nb::dict > owner_tensor_cache_if_present(nb::handle owner);

void clear_owner_tensor_cache(nb::handle owner);

bool owner_target_device_matches(nb::handle owner, nb::handle device);

nb::object move_object_to_device(nb::handle value, nb::handle device);

void set_owner_target_device(nb::handle owner, nb::handle device);

void materialize_owner_tensor_cache(nb::handle owner, BatchBuilder::BatchEncoding& encoding);

nb::object zeros_f32_on_owner_device(nb::handle owner, int64_t rows, int64_t cols);

}  // namespace mifrost
