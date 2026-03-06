#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <cstdint>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "core/schema.hpp"
#include "core/utils/macro.hpp"
#include "core/utils/type_traits.hpp"

namespace mifrost {

std::vector< int64_t > ptr_to_batch(const std::vector< int64_t >& ptr);

}  // namespace mifrost

namespace mifrost::py {

std::string to_std_string(nanobind::handle value);

nanobind::handle builtins_module();
nanobind::handle pickle_module();
nanobind::handle types_module();
nanobind::handle torch_module();
nanobind::handle torch_geometric_data_module();
nanobind::handle mifrost_module();
nanobind::handle mifrost_core_module();
nanobind::handle mifrost_flat_data_module();
nanobind::handle operator_module();
nanobind::handle numpy_module();

nanobind::handle builtins_object_setattr();
nanobind::handle builtins_open();
nanobind::handle builtins_tuple_ctor();
nanobind::handle builtins_type_type();
nanobind::handle builtins_str_type();
nanobind::handle builtins_float_type();
nanobind::handle builtins_int_type();

nanobind::handle pickle_dumps();
nanobind::handle pickle_loads();
nanobind::handle mapping_proxy_type_ctor();
nanobind::object mapping_proxy(const nanobind::dict& mapping);
void set_python_attribute(nanobind::handle self, const std::string& key, nanobind::handle value);
void set_python_attribute(nanobind::handle self, nb::str key, nanobind::handle value);
nanobind::object flatten_single_graph_metadata_list(nanobind::handle value);

nanobind::handle torch_tensor_type();
nanobind::handle torch_equal_fn();
nanobind::handle torch_as_tensor_fn();
nanobind::handle torch_from_dlpack_fn();
nanobind::handle torch_utils_dlpack_from_dlpack_fn();
nanobind::object to_torch_tensor(nanobind::handle value);
nanobind::handle torch_device_ctor();
nanobind::handle torch_stack_fn();
nanobind::handle torch_zeros_fn();
nanobind::handle torch_float32_dtype();

nanobind::handle torch_geometric_batch_ctor();
nanobind::handle torch_geometric_heterodata_ctor();
nanobind::handle torch_geometric_data_ctor();

nanobind::handle mifrost_core_batch_encoding_cls();
nanobind::handle mifrost_batch_encoding_loader();
nanobind::handle mifrost_flat_relation_data_from_pyg_fn();
nanobind::handle operator_eq_fn();
nanobind::handle numpy_array_type();
nanobind::handle numpy_array_equal_fn();

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
