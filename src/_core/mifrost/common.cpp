#include "mifrost/common.hpp"

#include <algorithm>

namespace nb = nanobind;

namespace mifrost {

std::vector< int64_t > ptr_to_batch(const std::vector< int64_t >& ptr)
{
   std::vector< int64_t > batch;
   if(ptr.size() < 2) {
      return batch;
   }
   batch.reserve(std::max< int64_t >(0, ptr.back()));
   for(size_t idx = 0; idx + 1 < ptr.size(); ++idx) {
      const int64_t count = std::max< int64_t >(0, ptr[idx + 1] - ptr[idx]);
      batch.insert(batch.end(), count, static_cast< int64_t >(idx));
   }
   return batch;
}

}  // namespace mifrost

namespace mifrost::py {

std::string to_std_string(nb::handle value)
{
   return {nb::str(value).c_str()};
}

nb::handle builtins_module()
{
   static nb::object* module = [] { return new nb::object(nb::module_::import_("builtins")); }();
   return *module;
}

nb::handle pickle_module()
{
   static nb::object* module = [] { return new nb::object(nb::module_::import_("pickle")); }();
   return *module;
}

nb::handle types_module()
{
   static nb::object* module = [] { return new nb::object(nb::module_::import_("types")); }();
   return *module;
}

nb::handle torch_module()
{
   static nb::object* module = [] { return new nb::object(nb::module_::import_("torch")); }();
   return *module;
}

nb::handle torch_geometric_data_module()
{
   static nb::object* module = [] {
      return new nb::object(nb::module_::import_("torch_geometric.data"));
   }();
   return *module;
}

nb::handle mifrost_module()
{
   static nb::object* module = [] { return new nb::object(nb::module_::import_("mifrost")); }();
   return *module;
}

nb::handle mifrost_core_module()
{
   static nb::object* module = [] {
      return new nb::object(nb::module_::import_("mifrost._core"));
   }();
   return *module;
}

nb::handle mifrost_flat_data_module()
{
   static nb::object* module = [] {
      return new nb::object(nb::module_::import_("mifrost.encoders.flat_data"));
   }();
   return *module;
}

nb::handle operator_module()
{
   static nb::object* module = [] { return new nb::object(nb::module_::import_("operator")); }();
   return *module;
}

nb::handle numpy_module()
{
   static nb::object* module = [] { return new nb::object(nb::module_::import_("numpy")); }();
   return *module;
}

nb::handle builtins_object_setattr()
{
   static nb::object* fn = [] {
      return new nb::object(builtins_module().attr("object").attr("__setattr__"));
   }();
   return *fn;
}

nb::handle builtins_open()
{
   static nb::object* fn = [] { return new nb::object(builtins_module().attr("open")); }();
   return *fn;
}

nb::handle builtins_tuple_ctor()
{
   static nb::object* fn = [] { return new nb::object(builtins_module().attr("tuple")); }();
   return *fn;
}

nb::handle builtins_type_type()
{
   static nb::object* type = [] { return new nb::object(builtins_module().attr("type")); }();
   return *type;
}

nb::handle builtins_str_type()
{
   static nb::object* type = [] { return new nb::object(builtins_module().attr("str")); }();
   return *type;
}

nb::handle builtins_float_type()
{
   static nb::object* type = [] { return new nb::object(builtins_module().attr("float")); }();
   return *type;
}

nb::handle builtins_int_type()
{
   static nb::object* type = [] { return new nb::object(builtins_module().attr("int")); }();
   return *type;
}

nb::handle pickle_dumps()
{
   static nb::object* fn = [] { return new nb::object(pickle_module().attr("dumps")); }();
   return *fn;
}

nb::handle pickle_loads()
{
   static nb::object* fn = [] { return new nb::object(pickle_module().attr("loads")); }();
   return *fn;
}

nb::handle mapping_proxy_type_ctor()
{
   static nb::object* ctor = [] {
      return new nb::object(types_module().attr("MappingProxyType"));
   }();
   return *ctor;
}

nb::object mapping_proxy(const nb::dict& mapping)
{
   return mapping_proxy_type_ctor()(mapping);
}

void set_python_attribute(nb::handle self, nb::str key, nb::handle value)
{
   builtins_object_setattr()(self, std::move(key), value);
}
void set_python_attribute(nb::handle self, const std::string& key, nb::handle value)
{
   set_python_attribute(self, nb::str(key.c_str()), value);
}

nb::object flatten_single_graph_metadata_list(nb::handle value)
{
   if(not nb::isinstance< nb::list >(value)) {
      return nb::borrow< nb::object >(value);
   }
   const nb::list outer = nb::borrow< nb::list >(value);
   if(nb::len(outer) == 1 and nb::isinstance< nb::list >(outer[0])) {
      return nb::borrow< nb::object >(outer[0]);
   }
   return nb::borrow< nb::object >(value);
}

nb::handle torch_tensor_type()
{
   static nb::object* type = [] { return new nb::object(torch_module().attr("Tensor")); }();
   return *type;
}

nb::handle torch_equal_fn()
{
   static nb::object* fn = [] { return new nb::object(torch_module().attr("equal")); }();
   return *fn;
}

nb::handle torch_as_tensor_fn()
{
   static nb::object* fn = [] { return new nb::object(torch_module().attr("as_tensor")); }();
   return *fn;
}

nb::handle torch_from_dlpack_fn()
{
   static nb::object* fn = [] { return new nb::object(torch_module().attr("from_dlpack")); }();
   return *fn;
}

nb::handle torch_utils_dlpack_from_dlpack_fn()
{
   static nb::object* fn = [] {
      return new nb::object(torch_module().attr("utils").attr("dlpack").attr("from_dlpack"));
   }();
   return *fn;
}

nb::object to_torch_tensor_impl(PyObject* value_ptr)
{
   nb::handle value(value_ptr);
   if(nb::isinstance(value, torch_tensor_type())) {
      return nb::borrow< nb::object >(value);
   }
   if(nb::hasattr(value, "__dlpack__")) {
      return torch_from_dlpack_fn()(nb::borrow< nb::object >(value));
   }
   return torch_as_tensor_fn()(value);
}

nb::handle torch_device_ctor()
{
   static nb::object* ctor = [] { return new nb::object(torch_module().attr("device")); }();
   return *ctor;
}

nb::handle torch_stack_fn()
{
   static nb::object* fn = [] { return new nb::object(torch_module().attr("stack")); }();
   return *fn;
}

nb::handle torch_zeros_fn()
{
   static nb::object* fn = [] { return new nb::object(torch_module().attr("zeros")); }();
   return *fn;
}

nb::handle torch_float32_dtype()
{
   static nb::object* dtype = [] { return new nb::object(torch_module().attr("float32")); }();
   return *dtype;
}

nb::handle torch_geometric_batch_ctor()
{
   static nb::object* ctor = [] {
      return new nb::object(torch_geometric_data_module().attr("Batch"));
   }();
   return *ctor;
}

nb::handle torch_geometric_heterodata_ctor()
{
   static nb::object* ctor = [] {
      return new nb::object(torch_geometric_data_module().attr("HeteroData"));
   }();
   return *ctor;
}

nb::handle torch_geometric_data_ctor()
{
   static nb::object* ctor = [] {
      return new nb::object(torch_geometric_data_module().attr("Data"));
   }();
   return *ctor;
}

nb::handle mifrost_core_batch_encoding_cls()
{
   static nb::object* cls = [] {
      return new nb::object(mifrost_core_module().attr("BatchEncoding"));
   }();
   return *cls;
}

nb::handle mifrost_batch_encoding_loader()
{
   static nb::object* loader = [] {
      return new nb::object(mifrost_module().attr("_batch_encoding_from_payload"));
   }();
   return *loader;
}

nb::handle mifrost_flat_relation_data_from_pyg_fn()
{
   static nb::object* fn = [] {
      return new nb::object(mifrost_flat_data_module().attr("flat_relation_data_from_pyg"));
   }();
   return *fn;
}

nb::handle operator_eq_fn()
{
   static nb::object* fn = [] { return new nb::object(operator_module().attr("eq")); }();
   return *fn;
}

nb::handle numpy_array_type()
{
   static nb::object* type = [] {
      const nb::handle np = numpy_module();
      if(np.is_none()) {
         return new nb::object(nb::none());
      }
      return new nb::object(np.attr("ndarray"));
   }();
   return *type;
}

nb::handle numpy_array_equal_fn()
{
   static nb::object* fn = [] {
      const nb::handle np = numpy_module();
      if(np.is_none()) {
         return new nb::object(nb::none());
      }
      return new nb::object(np.attr("array_equal"));
   }();
   return *fn;
}

}  // namespace mifrost::py
