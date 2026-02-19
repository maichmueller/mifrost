#include "mifrost/batch_encoding_graph_field_access.hpp"

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "mifrost/core/dlpack_utils.hpp"

namespace nb = nanobind;

namespace mifrost {

namespace {

constexpr std::string_view kPythonTensorDeviceAttr = "__mifrost_tensor_device__";
constexpr std::string_view kPythonTensorCacheAttr = "__mifrost_tensor_cache__";

enum class GraphFieldLookupKind { VALUE, PTR, MISSING };

struct GraphFieldLookup {
   GraphFieldLookupKind kind = GraphFieldLookupKind::MISSING;
   GraphField* field = nullptr;
};

GraphFieldLookup
batch_encoding_lookup_graph_field(BatchBuilder::BatchEncoding& encoding, std::string_view key)
{
   if(const auto it = encoding.graph_fields.find(std::string(key));
      it != encoding.graph_fields.end()) {
      return {.kind = GraphFieldLookupKind::VALUE, .field = &it->second};
   }
   constexpr std::string_view kPtrSuffix = "_ptr";
   if(key.size() <= kPtrSuffix.size()
      or key.compare(key.size() - kPtrSuffix.size(), kPtrSuffix.size(), kPtrSuffix) != 0) {
      return {};
   }
   std::string base(key.substr(0, key.size() - kPtrSuffix.size()));
   if(const auto it = encoding.graph_fields.find(base); it != encoding.graph_fields.end()) {
      if(it->second.spec.mode == GraphFieldMode::RAGGED_CAT) {
         return {.kind = GraphFieldLookupKind::PTR, .field = &it->second};
      }
   }
   return {};
}

nb::handle torch_module_handle()
{
   static nb::object* module = []() { return new nb::object(nb::module_::import_("torch")); }();
   return *module;
}

nb::object to_torch_tensor(nb::handle array_like)
{
   nb::object torch = nb::borrow< nb::object >(torch_module_handle());
   if(nb::isinstance(array_like, torch.attr("Tensor"))) {
      return nb::borrow< nb::object >(array_like);
   }
   if(dlpack_utils::is_dlpack_capsule(array_like)) {
      return torch.attr("utils").attr("dlpack").attr("from_dlpack")(array_like);
   }
   if(nb::hasattr(array_like, "__dlpack__")) {
      return torch.attr("from_dlpack")(nb::borrow< nb::object >(array_like));
   }
   return torch.attr("as_tensor")(array_like);
}

nb::object target_device_from_owner(nb::handle owner)
{
   nb::dict attrs = nb::cast< nb::dict >(owner.attr("__dict__"));
   if(not attrs.contains(kPythonTensorDeviceAttr.data())) {
      return nb::none();
   }
   return nb::borrow< nb::object >(attrs[kPythonTensorDeviceAttr.data()]);
}

std::optional< nb::dict > owner_tensor_cache_if_present(nb::handle owner)
{
   nb::dict attrs = nb::cast< nb::dict >(owner.attr("__dict__"));
   if(not attrs.contains(kPythonTensorCacheAttr.data())) {
      return std::nullopt;
   }
   nb::object raw_cache = nb::borrow< nb::object >(attrs[kPythonTensorCacheAttr.data()]);
   if(not nb::isinstance< nb::dict >(raw_cache)) {
      throw std::invalid_argument("BatchEncoding internal tensor cache must be a dict");
   }
   return nb::cast< nb::dict >(raw_cache);
}

nb::object maybe_move_tensor_to_device(nb::object tensor, nb::handle owner)
{
   nb::object device = target_device_from_owner(owner);
   if(device.is_none()) {
      return tensor;
   }
   return tensor.attr("to")(device);
}

nb::object graph_field_values_to_tensor_view(GraphField& field, nb::handle owner)
{
   return std::visit(
      [&](auto& values) -> nb::object {
         using T = std::decay_t< decltype(values) >::value_type;
         if(field.spec.dim == 1) {
            return to_torch_tensor(dlpack_utils::vector_to_dlpack_view_1d(values, owner));
         }
         const bool cat_dim_one = (field.spec.mode == GraphFieldMode::CAT
                                   or field.spec.mode == GraphFieldMode::RAGGED_CAT)
                                  and graph_field_cat_dim_is_one(field.spec.cat_dim);
         const size_t rows = cat_dim_one ? static_cast< size_t >(field.spec.dim)
                                         : values.size() / static_cast< size_t >(field.spec.dim);
         const size_t cols = cat_dim_one ? values.size() / static_cast< size_t >(field.spec.dim)
                                         : static_cast< size_t >(field.spec.dim);
         return to_torch_tensor(dlpack_utils::vector_to_dlpack_view_2d(values, rows, cols, owner));
      },
      field.values
   );
}

nb::object graph_field_ptr_to_tensor(const GraphField& field)
{
   return to_torch_tensor(dlpack_utils::vector_to_dlpack_owned_copy_1d(field.ptr));
}

}  // namespace

std::set< std::string > batch_encoding_native_graph_field_keys(
   const BatchBuilder::BatchEncoding& encoding
)
{
   std::set< std::string > out;
   for(const auto& [key, field] : encoding.graph_fields) {
      out.insert(key);
      if(field.spec.mode == GraphFieldMode::RAGGED_CAT) {
         out.insert(key + "_ptr");
      }
   }
   return out;
}

std::set< std::string > batch_encoding_native_tensor_keys(
   const BatchBuilder::BatchEncoding& encoding
)
{
   std::set< std::string > out;

   for(const auto& [key, col] : encoding.columns) {
      (void) col;
      out.insert(key);
   }

   for(const auto& [node_type, ptr] : encoding.ptrs) {
      (void) ptr;
      out.insert(node_type + "/ptr");
      out.insert(node_type + "/batch");
   }

   const auto graph_keys = batch_encoding_native_graph_field_keys(encoding);
   out.insert(graph_keys.begin(), graph_keys.end());
   return out;
}

bool batch_encoding_has_graph_field(
   const BatchBuilder::BatchEncoding& encoding,
   std::string_view key
)
{
   if(encoding.graph_fields.contains(std::string(key))) {
      return true;
   }
   constexpr std::string_view kPtrSuffix = "_ptr";
   if(key.size() <= kPtrSuffix.size()
      or key.compare(key.size() - kPtrSuffix.size(), kPtrSuffix.size(), kPtrSuffix) != 0) {
      return false;
   }
   std::string base(key.substr(0, key.size() - kPtrSuffix.size()));
   if(const auto it = encoding.graph_fields.find(base); it != encoding.graph_fields.end()) {
      return it->second.spec.mode == GraphFieldMode::RAGGED_CAT;
   }
   return false;
}

nb::object batch_encoding_get_graph_field(
   BatchBuilder::BatchEncoding& encoding,
   std::string_view key,
   nb::handle owner
)
{
   const std::string key_string(key);
   if(auto cache = owner_tensor_cache_if_present(owner);
      cache.has_value() and cache->contains(key_string.c_str())) {
      return nb::borrow< nb::object >((*cache)[key_string.c_str()]);
   }

   const auto lookup = batch_encoding_lookup_graph_field(encoding, key);
   if(lookup.kind == GraphFieldLookupKind::VALUE) {
      nb::object value = maybe_move_tensor_to_device(
         graph_field_values_to_tensor_view(*lookup.field, owner), owner
      );
      if(auto cache = owner_tensor_cache_if_present(owner); cache.has_value()) {
         (*cache)[key_string.c_str()] = value;
      }
      return value;
   }
   if(lookup.kind == GraphFieldLookupKind::PTR) {
      nb::object value = maybe_move_tensor_to_device(
         graph_field_ptr_to_tensor(*lookup.field), owner
      );
      if(auto cache = owner_tensor_cache_if_present(owner); cache.has_value()) {
         (*cache)[key_string.c_str()] = value;
      }
      return value;
   }
   throw std::invalid_argument("BatchEncoding.get_field unknown key '" + std::string(key) + "'");
}

void validate_batch_encoding_graph_fields(
   const BatchBuilder::BatchEncoding& encoding,
   std::string_view context
)
{
   for(const auto& [key, field] : encoding.graph_fields) {
      try {
         validate_graph_field_storage(key, field, encoding.num_graphs);
      } catch(const std::exception& ex) {
         throw std::invalid_argument(
            "Invalid graph field '" + key + "' in " + std::string(context) + ": " + ex.what()
         );
      }
   }
}

}  // namespace mifrost
