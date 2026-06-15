#include "mifrost/batch_encoding_tensor_cache.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <string_view>

#include "mifrost/batch_encoding_graph_field_access.hpp"
#include "mifrost/batch_encoding_python_collation.hpp"
#include "mifrost/common.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace mifrost {

namespace {

constexpr std::string_view kPythonTensorDeviceAttr = "__mifrost_tensor_device__";
constexpr std::string_view kPythonTensorCacheAttr = "__mifrost_tensor_cache__";

}  // namespace

nb::object owner_target_device(nb::handle owner)
{
   auto attrs = nb::cast< nb::dict >(owner.attr("__dict__"));
   if(not attrs.contains(kPythonTensorDeviceAttr.data())) {
      return nb::none();
   }
   return nb::borrow< nb::object >(attrs[kPythonTensorDeviceAttr.data()]);
}

std::optional< nb::dict > owner_tensor_cache_if_present(nb::handle owner)
{
   auto attrs = nb::cast< nb::dict >(owner.attr("__dict__"));
   if(not attrs.contains(kPythonTensorCacheAttr.data())) {
      return std::nullopt;
   }
   auto raw_cache = nb::borrow< nb::object >(attrs[kPythonTensorCacheAttr.data()]);
   if(not nb::isinstance< nb::dict >(raw_cache)) {
      throw std::invalid_argument("BatchEncoding internal tensor cache must be a dict");
   }
   return nb::cast< nb::dict >(raw_cache);
}

void clear_owner_tensor_cache(nb::handle owner)
{
   auto attrs = nb::cast< nb::dict >(owner.attr("__dict__"));
   if(attrs.contains(kPythonTensorCacheAttr.data())) {
      attrs.attr("pop")(kPythonTensorCacheAttr.data());
   }
}

bool owner_target_device_matches(nb::handle owner, nb::handle device)
{
   nb::object current = owner_target_device(owner);
   if(current.is_none() or device.is_none()) {
      return current.is_none() and device.is_none();
   }
   return py::to_std_string(nb::str(current)) == py::to_std_string(nb::str(device));
}

nb::object move_object_to_device(nb::handle value, nb::handle device)
{
   if(device.is_none()) {
      return nb::borrow< nb::object >(value);
   }
   if(is_torch_tensor(value)) {
      return nb::borrow< nb::object >(value).attr("to")(device);
   }
   if(nb::isinstance< nb::dict >(value)) {
      nb::dict out;
      for(auto [k, v] : nb::borrow< nb::dict >(value)) {
         out[k] = move_object_to_device(nb::borrow< nb::object >(v), device);
      }
      return out;
   }
   if(nb::isinstance< nb::list >(value)) {
      nb::list out;
      for(nb::handle item : nb::borrow< nb::list >(value)) {
         out.append(move_object_to_device(item, device));
      }
      return out;
   }
   if(nb::isinstance< nb::tuple >(value)) {
      nb::list tmp;
      for(nb::handle item : nb::borrow< nb::tuple >(value)) {
         tmp.append(move_object_to_device(item, device));
      }
      return py::builtins_tuple_ctor()(tmp);
   }
   return nb::borrow< nb::object >(value);
}

void set_owner_target_device(nb::handle owner, nb::handle device)
{
   nb::dict attrs = nb::cast< nb::dict >(owner.attr("__dict__"));
   if(device.is_none()) {
      if(attrs.contains(kPythonTensorDeviceAttr.data())) {
         attrs.attr("pop")(kPythonTensorDeviceAttr.data());
      }
      return;
   }
   attrs[kPythonTensorDeviceAttr.data()] = py::torch_device_ctor()(device);
}

void materialize_owner_tensor_cache(nb::handle owner, BatchBuilder::BatchEncoding& encoding)
{
   clear_owner_tensor_cache(owner);

   nb::dict cache;

   const auto native_tensor_keys = batch_encoding_native_tensor_keys(encoding);

   for(const auto& key : native_tensor_keys) {
      cache[key.c_str()] = batch_encoding_get_native_tensor(encoding, key, owner);
   }

   nb::dict attrs = nb::cast< nb::dict >(owner.attr("__dict__"));
   attrs[kPythonTensorCacheAttr.data()] = std::move(cache);
}

nb::object zeros_f32_on_owner_device(nb::handle owner, int64_t rows, int64_t cols)
{
   nb::object device = owner_target_device(owner);
   if(device.is_none()) {
      return py::torch_zeros_fn()(
         nb::make_tuple(rows, cols), "dtype"_a = py::torch_float32_dtype()
      );
   }
   return py::torch_zeros_fn()(
      nb::make_tuple(rows, cols), "dtype"_a = py::torch_float32_dtype(), "device"_a = device
   );
}

}  // namespace mifrost
