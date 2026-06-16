#include "mifrost/batch_encoding_attributes.hpp"

#include <nanobind/nanobind.h>

#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include "mifrost/batch_encoding_conversion.hpp"
#include "mifrost/batch_encoding_graph_field_access.hpp"
#include "mifrost/batch_encoding_graph_field_mutation.hpp"
#include "mifrost/batch_encoding_python_collation.hpp"
#include "mifrost/batch_encoding_tensor_cache.hpp"
#include "mifrost/common.hpp"
#include "mifrost/core/batch_builder.hpp"
#include "mifrost/core/nb_instance.hpp"

namespace nb = nanobind;

namespace mifrost {

namespace {

BatchBuilder::BatchEncoding*
require_batch_encoding(nb::handle self, std::string_view context)
{
   return require_instance_ptr< BatchBuilder::BatchEncoding >(self, context);
}

std::set< std::string >
batch_encoding_visible_keys(const BatchBuilder::BatchEncoding& encoding, nb::handle self)
{
   auto key_set = batch_encoding_native_graph_field_keys(encoding);
   nb::dict attrs = batch_encoding_python_attrs(self);
   for(auto [key_obj, value_obj] : attrs) {
      (void) value_obj;
      const std::string key = py::to_std_string(key_obj);
      if(is_forbidden_dynamic_attr_key(encoding, key) or key_set.contains(key)) {
         continue;
      }
      key_set.insert(key);
   }
   return key_set;
}

}  // namespace

nb::object batch_encoding_to_device(nb::handle self, nb::handle device)
{
   auto* encoding =
      require_batch_encoding(self, "BatchEncoding.to called with invalid instance");
   if(device.is_none()) {
      return nb::borrow< nb::object >(self);
   }
   nb::object normalized = py::torch_device_ctor()(device);
   const bool same_device = owner_target_device_matches(self, normalized);
   set_owner_target_device(self, normalized);
   nb::dict attrs = batch_encoding_python_attrs(self);
   for(auto [key_obj, value_obj] : attrs) {
      const std::string key = py::to_std_string(key_obj);
      if(is_forbidden_dynamic_attr_key(*encoding, key)) {
         continue;
      }
      attrs[key_obj] = move_object_to_device(nb::borrow< nb::object >(value_obj), normalized);
   }
   if(not same_device) {
      clear_owner_tensor_cache(self);
      materialize_owner_tensor_cache(self, *encoding);
   }
   return nb::borrow< nb::object >(self);
}

nb::object batch_encoding_getattr(nb::handle self, const std::string& key)
{
   auto* encoding =
      require_batch_encoding(self, "BatchEncoding.__getattr__ called with invalid instance");
   if(batch_encoding_has_graph_field(*encoding, key)) {
      return batch_encoding_get_graph_field(*encoding, key, self);
   }
   if(auto value = batch_encoding_graph_attr_if_present(*encoding, key); value.has_value()) {
      return std::move(*value);
   }
   const std::string message = "'BatchEncoding' object has no attribute '" + key + "'";
   PyErr_SetString(PyExc_AttributeError, message.c_str());
   throw nb::python_error();
}

void batch_encoding_setattr(nb::handle self, const std::string& key, nb::handle value)
{
   auto* encoding =
      require_batch_encoding(self, "BatchEncoding.__setattr__ called with invalid instance");
   if(batch_encoding_has_graph_field(*encoding, key)) {
      if(is_native_graph_field_ptr_key(*encoding, key)) {
         throw std::invalid_argument(
            "Direct assignment to ragged ptr key '" + key
            + "' is not supported; assign the base field as (values, ptr)"
         );
      }
      set_batch_encoding_graph_field(*encoding, key, value);
      clear_owner_tensor_cache(self);
      return;
   }
   if(is_forbidden_dynamic_attr_key(*encoding, key)) {
      throw std::invalid_argument(
         "Dynamic attribute key '" + key + "' collides with reserved/native key"
      );
   }
   py::set_python_attribute(self, key, value);
}

nb::list batch_encoding_keys(nb::handle self)
{
   auto* encoding =
      require_batch_encoding(self, "BatchEncoding.keys called with invalid instance");
   nb::list out;
   for(const auto& key : batch_encoding_visible_keys(*encoding, self)) {
      out.append(key);
   }
   return out;
}

nb::list batch_encoding_items(nb::handle self)
{
   auto* encoding =
      require_batch_encoding(self, "BatchEncoding.items called with invalid instance");
   const auto key_set = batch_encoding_visible_keys(*encoding, self);
   nb::dict attrs = batch_encoding_python_attrs(self);

   nb::list out;
   for(const auto& key : key_set) {
      nb::object value;
      if(batch_encoding_has_graph_field(*encoding, key)) {
         value = batch_encoding_get_graph_field(*encoding, key, self);
      } else {
         value = nb::borrow< nb::object >(attrs[key.c_str()]);
      }
      out.append(nb::make_tuple(key, std::move(value)));
   }
   return out;
}

}  // namespace mifrost
