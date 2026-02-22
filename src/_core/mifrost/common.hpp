#pragma once

#include <nanobind/nanobind.h>

#include <cstdint>
#include <string>
#include <vector>

namespace mifrost {

std::vector< int64_t > ptr_to_batch(const std::vector< int64_t >& ptr);

}  // namespace mifrost

namespace mifrost::py {

std::string to_std_string(nanobind::handle value);

nanobind::object try_import_module(const char* module_name);

nanobind::handle builtins_module();
nanobind::handle pickle_module();
nanobind::handle types_module();
nanobind::handle torch_module();
nanobind::handle torch_geometric_data_module();
nanobind::handle mifrost_module();
nanobind::handle mifrost_core_module();
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
nanobind::handle operator_eq_fn();
nanobind::handle numpy_array_type();
nanobind::handle numpy_array_equal_fn();

}  // namespace mifrost::py
