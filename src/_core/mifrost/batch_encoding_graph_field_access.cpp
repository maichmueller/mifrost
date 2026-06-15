#include "mifrost/batch_encoding_graph_field_access.hpp"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "mifrost/batch_encoding_tensor_cache.hpp"
#include "mifrost/common.hpp"
#include "mifrost/core/dlpack_utils.hpp"

namespace nb = nanobind;

namespace mifrost {

namespace {

constexpr std::string_view kPtrKeySuffix = "/ptr";
constexpr std::string_view kBatchKeySuffix = "/batch";

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

nb::object maybe_move_tensor_to_device(nb::object tensor, nb::handle owner)
{
   nb::object device = owner_target_device(owner);
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
            nb::object dlpack = dlpack_utils::vector_to_dlpack_view_1d(values, owner);
            return py::to_torch_tensor(dlpack);
         }
         const bool cat_dim_one = (field.spec.mode == GraphFieldMode::CAT
                                   or field.spec.mode == GraphFieldMode::RAGGED_CAT)
                                  and graph_field_cat_dim_is_one(field.spec.cat_dim);
         const size_t rows = cat_dim_one ? static_cast< size_t >(field.spec.dim)
                                         : values.size() / static_cast< size_t >(field.spec.dim);
         const size_t cols = cat_dim_one ? values.size() / static_cast< size_t >(field.spec.dim)
                                         : static_cast< size_t >(field.spec.dim);
         nb::object dlpack = dlpack_utils::vector_to_dlpack_view_2d(values, rows, cols, owner);
         return py::to_torch_tensor(dlpack);
      },
      field.values
   );
}

nb::object graph_field_ptr_to_tensor(const GraphField& field)
{
   nb::object dlpack = dlpack_utils::vector_to_dlpack_owned_copy_1d(field.ptr);
   return py::to_torch_tensor(dlpack);
}

bool has_suffix(std::string_view key, std::string_view suffix)
{
   return key.size() >= suffix.size()
          and key.compare(key.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::optional< std::string > node_type_for_suffix_key(std::string_view key, std::string_view suffix)
{
   if(not has_suffix(key, suffix) or key.size() == suffix.size()) {
      return std::nullopt;
   }
   return std::string(key.substr(0, key.size() - suffix.size()));
}

bool has_exported_ptrs(const BatchBuilder::BatchEncoding& encoding)
{
   for(const auto& [node_type, ptr] : encoding.ptrs) {
      (void) node_type;
      if(ptr.size() >= 2) {
         return true;
      }
   }
   return false;
}

const std::vector< int64_t >*
find_exported_ptr(const BatchBuilder::BatchEncoding& encoding, std::string_view node_type)
{
   const auto it = encoding.ptrs.find(std::string(node_type));
   if(it == encoding.ptrs.end() or it->second.size() < 2) {
      return nullptr;
   }
   return &it->second;
}

std::vector< int64_t >*
find_exported_ptr(BatchBuilder::BatchEncoding& encoding, std::string_view node_type)
{
   const auto it = encoding.ptrs.find(std::string(node_type));
   if(it == encoding.ptrs.end() or it->second.size() < 2) {
      return nullptr;
   }
   return &it->second;
}

std::optional< int64_t >
fallback_ptr_node_count(const BatchBuilder::BatchEncoding& encoding, std::string_view node_type)
{
   if(has_exported_ptrs(encoding)) {
      return std::nullopt;
   }
   const auto count_it = encoding.node_counts.find(std::string(node_type));
   if(count_it == encoding.node_counts.end() or count_it->second <= 0) {
      return std::nullopt;
   }
   return count_it->second;
}

bool is_edge_index_key(std::string_view key)
{
   return key.find("/edge_index_") != std::string_view::npos;
}

nb::object column_to_tensor(BatchBuilder::Column& column, std::string_view key, nb::handle owner)
{
   return std::visit(
      [&](auto& values) -> nb::object {
         if(is_edge_index_key(key)) {
            nb::object dlpack = dlpack_utils::vector_to_dlpack_view_1d(values, owner);
            return py::to_torch_tensor(dlpack);
         }
         const size_t rows = column.dim > 0 ? values.size() / static_cast< size_t >(column.dim) : 0;
         nb::object dlpack = dlpack_utils::vector_to_dlpack_view_2d(
            values, rows, static_cast< size_t >(column.dim), owner
         );
         return py::to_torch_tensor(dlpack);
      },
      column.data
   );
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
      if(ptr.size() < 2) {
         continue;
      }
      out.insert(node_type + "/ptr");
      out.insert(node_type + "/batch");
   }
   if(not has_exported_ptrs(encoding)) {
      for(const auto& [node_type, count] : encoding.node_counts) {
         if(count <= 0) {
            continue;
         }
         out.insert(node_type + "/ptr");
         out.insert(node_type + "/batch");
      }
   }

   const auto graph_keys = batch_encoding_native_graph_field_keys(encoding);
   out.insert(graph_keys.begin(), graph_keys.end());
   return out;
}

bool batch_encoding_has_native_tensor(
   const BatchBuilder::BatchEncoding& encoding,
   std::string_view key
)
{
   if(batch_encoding_has_graph_field(encoding, key)) {
      return true;
   }

   if(const auto node_type = node_type_for_suffix_key(key, kPtrKeySuffix); node_type.has_value()) {
      return find_exported_ptr(encoding, *node_type) != nullptr
             or fallback_ptr_node_count(encoding, *node_type).has_value();
   }
   if(const auto node_type = node_type_for_suffix_key(key, kBatchKeySuffix);
      node_type.has_value()) {
      return find_exported_ptr(encoding, *node_type) != nullptr
             or fallback_ptr_node_count(encoding, *node_type).has_value();
   }

   return encoding.columns.contains(std::string(key));
}

nb::object batch_encoding_get_native_tensor(
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

   if(batch_encoding_has_graph_field(encoding, key)) {
      return batch_encoding_get_graph_field(encoding, key, owner);
   }

   const auto cache_value = [&](nb::object value) {
      if(auto cache = owner_tensor_cache_if_present(owner); cache.has_value()) {
         (*cache)[key_string.c_str()] = value;
      }
      return value;
   };

   if(const auto node_type = node_type_for_suffix_key(key, kPtrKeySuffix); node_type.has_value()) {
      if(auto* ptr = find_exported_ptr(encoding, *node_type); ptr != nullptr) {
         nb::object dlpack = dlpack_utils::vector_to_dlpack_view_1d(*ptr, owner);
         return cache_value(maybe_move_tensor_to_device(py::to_torch_tensor(dlpack), owner));
      }
      if(const auto count = fallback_ptr_node_count(encoding, *node_type); count.has_value()) {
         std::vector< int64_t > ptr{0, *count};
         nb::object dlpack = dlpack_utils::vector_to_dlpack_owned_1d(std::move(ptr));
         return cache_value(maybe_move_tensor_to_device(py::to_torch_tensor(dlpack), owner));
      }
      throw std::invalid_argument(
         "BatchEncoding.get_native_tensor unknown key '" + std::string(key) + "'"
      );
   }

   if(const auto node_type = node_type_for_suffix_key(key, kBatchKeySuffix);
      node_type.has_value()) {
      if(auto* ptr = find_exported_ptr(encoding, *node_type); ptr != nullptr) {
         auto batch = mifrost::ptr_to_batch(*ptr);
         nb::object dlpack = dlpack_utils::vector_to_dlpack_owned_1d(std::move(batch));
         return cache_value(maybe_move_tensor_to_device(py::to_torch_tensor(dlpack), owner));
      }
      if(const auto count = fallback_ptr_node_count(encoding, *node_type); count.has_value()) {
         std::vector< int64_t > batch(static_cast< size_t >(*count), 0);
         nb::object dlpack = dlpack_utils::vector_to_dlpack_owned_1d(std::move(batch));
         return cache_value(maybe_move_tensor_to_device(py::to_torch_tensor(dlpack), owner));
      }
      throw std::invalid_argument(
         "BatchEncoding.get_native_tensor unknown key '" + std::string(key) + "'"
      );
   }

   if(const auto col_it = encoding.columns.find(key_string); col_it != encoding.columns.end()) {
      return cache_value(
         maybe_move_tensor_to_device(column_to_tensor(col_it->second, key, owner), owner)
      );
   }

   throw std::invalid_argument(
      "BatchEncoding.get_native_tensor unknown key '" + std::string(key) + "'"
   );
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
