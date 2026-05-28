#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <cstdint>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "core/api.hpp"
#include "core/schema.hpp"
#include "core/utils/macro.hpp"
#include "core/utils/type_traits.hpp"

namespace mifrost {

MIFROST_API std::vector< int64_t > ptr_to_batch(const std::vector< int64_t >& ptr);

}  // namespace mifrost

namespace mifrost::py {

namespace nb = nanobind;

MIFROST_API std::string to_std_string(nanobind::handle value);

MIFROST_API nanobind::handle builtins_module();
MIFROST_API nanobind::handle pickle_module();
MIFROST_API nanobind::handle types_module();
MIFROST_API nanobind::handle torch_module();
MIFROST_API nanobind::handle torch_geometric_data_module();
MIFROST_API nanobind::handle mifrost_module();
MIFROST_API nanobind::handle mifrost_core_module();
MIFROST_API nanobind::handle mifrost_flat_data_module();
MIFROST_API nanobind::handle operator_module();
MIFROST_API nanobind::handle numpy_module();

MIFROST_API nanobind::handle builtins_object_setattr();
MIFROST_API nanobind::handle builtins_open();
MIFROST_API nanobind::handle builtins_tuple_ctor();
MIFROST_API nanobind::handle builtins_type_type();
MIFROST_API nanobind::handle builtins_str_type();
MIFROST_API nanobind::handle builtins_float_type();
MIFROST_API nanobind::handle builtins_int_type();

MIFROST_API nanobind::handle pickle_dumps();
MIFROST_API nanobind::handle pickle_loads();
MIFROST_API nanobind::handle mapping_proxy_type_ctor();
MIFROST_API nanobind::object mapping_proxy(const nanobind::dict& mapping);
MIFROST_API void
set_python_attribute(nanobind::handle self, const std::string& key, nanobind::handle value);
MIFROST_API void set_python_attribute(nanobind::handle self, nb::str key, nanobind::handle value);
MIFROST_API nanobind::object flatten_single_graph_metadata_list(nanobind::handle value);

MIFROST_API nanobind::handle torch_tensor_type();
MIFROST_API nanobind::handle torch_equal_fn();
MIFROST_API nanobind::handle torch_as_tensor_fn();
MIFROST_API nanobind::handle torch_from_dlpack_fn();
MIFROST_API nanobind::handle torch_utils_dlpack_from_dlpack_fn();
MIFROST_API nanobind::object to_torch_tensor(nanobind::handle value);
MIFROST_API nanobind::handle torch_device_ctor();
MIFROST_API nanobind::handle torch_stack_fn();
MIFROST_API nanobind::handle torch_zeros_fn();
MIFROST_API nanobind::handle torch_float32_dtype();

MIFROST_API nanobind::handle torch_geometric_batch_ctor();
MIFROST_API nanobind::handle torch_geometric_heterodata_ctor();
MIFROST_API nanobind::handle torch_geometric_data_ctor();

MIFROST_API nanobind::handle mifrost_core_batch_encoding_cls();
MIFROST_API nanobind::handle mifrost_batch_encoding_loader();
MIFROST_API nanobind::handle mifrost_flat_relation_data_from_pyg_fn();
MIFROST_API nanobind::handle operator_eq_fn();
MIFROST_API nanobind::handle numpy_array_type();
MIFROST_API nanobind::handle numpy_array_equal_fn();

template < typename T >
   requires requires(T t) {
      as_tuple(t);
      mifrost::detail::is_specialization_v< detail::raw_t< decltype(as_tuple(t)) >, std::tuple >;
   }
nb::tuple to_py_tuple(const T& t)
{
   return std::apply(AS_LAMBDA(nb::make_tuple), as_tuple(t));
}

}  // namespace mifrost::py
