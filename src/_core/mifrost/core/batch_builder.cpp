#include "batch_builder.hpp"

#include <absl/container/btree_map.h>
#include <fmt/format.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <array>
#include <numeric>
#include <optional>
#include <range/v3/view/enumerate.hpp>
#include <set>
#include <stdexcept>
#include <string_view>
#include <tuple>

#include "mifrost/common.hpp"
#include "mifrost/core/dlpack_utils.hpp"
#include "mifrost/core/schema_key_separators.hpp"
#include "schema.hpp"
#include "utils/macro.hpp"

namespace mifrost {

namespace {

template < typename T >
std::vector< T >& expect_column(NumericColumnData& data, const std::string& key)
{
   auto* ptr = std::get_if< std::vector< T > >(&data);
   if(ptr == nullptr) {
      throw std::invalid_argument("Graph field '" + key + "' dtype mismatch");
   }
   return *ptr;
}

template < typename T >
const std::vector< T >& expect_column(const NumericColumnData& data, const std::string& key)
{
   const auto* ptr = std::get_if< std::vector< T > >(&data);
   if(ptr == nullptr) {
      throw std::invalid_argument("Graph field '" + key + "' dtype mismatch");
   }
   return *ptr;
}

int64_t committed_field_offset(
   const hash_map< std::string, GraphField >& fields,
   const std::string& key,
   const GraphFieldSpec& spec
)
{
   const auto field_it = fields.find(spec.inc.field_key);
   if(field_it == fields.end()) {
      throw std::invalid_argument(
         "Graph field '" + key + "' FIELD_OFFSET references missing field '" + spec.inc.field_key
         + "'"
      );
   }

   const auto& offset_field = field_it->second;
   if(offset_field.spec.mode != GraphFieldMode::STACK
      or offset_field.spec.dtype != GraphFieldDType::I64 or offset_field.spec.dim != 1) {
      throw std::invalid_argument(
         "Graph field '" + key + "' FIELD_OFFSET requires referenced field '" + spec.inc.field_key
         + "' to be STACK/i64/dim=1"
      );
   }

   const auto& values = expect_column< int64_t >(offset_field.values, spec.inc.field_key);
   return std::accumulate(values.begin(), values.end(), int64_t{0});
}

bool is_reserved_pyg_graph_attr_key(std::string_view key)
{
   static constexpr std::array< std::string_view, 14 > kReserved{
      "x",
      "edge_index",
      "edge_attr",
      "batch",
      "ptr",
      "x_dict",
      "edge_index_dict",
      "edge_attr_dict",
      "batch_dict",
      "ptr_dict",
      "_num_graphs",
      "object_names",
      "node_names",
      "num_nodes",
   };
   return key.starts_with("__mifrost_") or std::ranges::find(kReserved, key) != kReserved.end();
}

bool pyg_global_store_contains_key(nb::object& batch, const std::string& key)
{
   nb::object global_store = batch.attr("_global_store");
   return nb::cast< bool >(global_store.attr("__contains__")(key.c_str()));
}

std::string make_type_attr_key(std::string_view type_key, std::string_view attr)
{
   std::string key;
   key.reserve(type_key.size() + attr.size() + 1);
   key.append(type_key);
   key.push_back(schema_key::kTypeAttrSeparator);
   key.append(attr);
   return key;
}

std::string make_edge_type_base_key(
   std::string_view src_type,
   std::string_view rel_type,
   std::string_view dst_type
)
{
   std::string key;
   key.reserve(src_type.size() + rel_type.size() + dst_type.size() + 2);
   key.append(src_type);
   key.push_back(schema_key::kEdgeTypeSeparator);
   key.append(rel_type);
   key.push_back(schema_key::kEdgeTypeSeparator);
   key.append(dst_type);
   return key;
}

std::string make_edge_index_component_key(std::string_view edge_key_base, char component)
{
   std::string key;
   key.reserve(edge_key_base.size() + schema_key::kEdgeIndexKeyPrefix.size() + 1);
   key.append(edge_key_base);
   key.push_back(schema_key::kTypeAttrSeparator);
   key.append(schema_key::kEdgeIndexAttrPrefix);
   key.push_back(component);
   return key;
}

bool key_has_edge_separator(std::string_view key)
{
   return key.find(schema_key::kEdgeTypeSeparator) != std::string_view::npos;
}

std::string_view::size_type find_type_attr_separator(std::string_view key)
{
   return key.find(schema_key::kTypeAttrSeparator);
}

bool key_has_edge_index_prefix(std::string_view key)
{
   return key.find(schema_key::kEdgeIndexKeyPrefix) != std::string_view::npos;
}

bool key_has_ptr_suffix(std::string_view key)
{
   if(key == schema_key::kPtrAttr) {
      return true;
   }
   if(key.size() <= schema_key::kPtrAttr.size()) {
      return false;
   }
   const auto suffix_pos = key.size() - schema_key::kPtrAttr.size();
   return key[suffix_pos - 1] == schema_key::kTypeAttrSeparator
          and key.substr(suffix_pos) == schema_key::kPtrAttr;
}

bool key_has_batch_suffix(std::string_view key)
{
   if(key == schema_key::kBatchAttr) {
      return true;
   }
   if(key.size() <= schema_key::kBatchAttr.size()) {
      return false;
   }
   const auto suffix_pos = key.size() - schema_key::kBatchAttr.size();
   return key[suffix_pos - 1] == schema_key::kTypeAttrSeparator
          and key.substr(suffix_pos) == schema_key::kBatchAttr;
}

void set_graph_attrs_on_pyg_batch(
   nb::object& batch,
   const hash_map< std::string, BatchBuilder::GraphAttrValue >& graph_attrs,
   const std::unique_ptr< hash_map< std::string, GraphField > >& graph_fields
)
{
   if(graph_attrs.empty()) {
      return;
   }

   auto collides_with_native_graph_field = [&](const std::string& key) {
      if(not graph_fields) {
         return false;
      }
      if(graph_fields->contains(key)) {
         return true;
      }
      if(key.size() > 4 and key.ends_with("_ptr")) {
         std::string base_key = key.substr(0, key.size() - 4);
         if(graph_fields->contains(base_key)) {
            return true;
         }
      }
      return false;
   };

   for(const auto& [key, value] : graph_attrs) {
      if(is_reserved_pyg_graph_attr_key(key) or pyg_global_store_contains_key(batch, key)
         or collides_with_native_graph_field(key)) {
         throw std::invalid_argument(
            "Graph attr key '" + key + "' collides with a reserved/existing PyG key"
         );
      }
      std::visit(
         [&](const auto& typed_value) {
            batch.attr("__setattr__")(key.c_str(), nb::cast(typed_value));
         },
         value
      );
   }
}

void append_with_inc(
   const std::string& key,
   const GraphFieldSpec& spec,
   NumericColumnData& dst_data,
   const NumericColumnData& src_data,
   int64_t inc
)
{
   if(spec.dtype == GraphFieldDType::F32) {
      auto& dst = expect_column< float >(dst_data, key);
      const auto& src = expect_column< float >(src_data, key);
      const bool cat_dim_one = (spec.mode == GraphFieldMode::CAT
                                or spec.mode == GraphFieldMode::RAGGED_CAT)
                               and graph_field_cat_dim_is_one(spec.cat_dim) and spec.dim > 1;
      if(not cat_dim_one) {
         dst.reserve(dst.size() + src.size());
         dst.insert(dst.end(), src.begin(), src.end());
         return;
      }

      const size_t dim = static_cast< size_t >(spec.dim);
      if(dst.size() % dim != 0 or src.size() % dim != 0) {
         throw std::invalid_argument(
            "Graph field '" + key + "' invalid size for cat_dim=1 concatenation"
         );
      }
      const size_t dst_cols = dst.size() / dim;
      const size_t src_cols = src.size() / dim;
      std::vector< float > merged(dim * (dst_cols + src_cols));
      for(size_t row = 0; row < dim; ++row) {
         const size_t dst_in = row * dst_cols;
         const size_t src_in = row * src_cols;
         const size_t out = row * (dst_cols + src_cols);
         std::copy_n(
            dst.begin() + static_cast< std::vector< float >::difference_type >(dst_in),
            dst_cols,
            merged.begin() + static_cast< std::vector< float >::difference_type >(out)
         );
         std::copy_n(
            src.begin() + static_cast< std::vector< float >::difference_type >(src_in),
            src_cols,
            merged.begin() + static_cast< std::vector< float >::difference_type >(out + dst_cols)
         );
      }
      dst.swap(merged);
      return;
   }

   auto& dst = expect_column< int64_t >(dst_data, key);
   const auto& src = expect_column< int64_t >(src_data, key);
   const bool cat_dim_one = (spec.mode == GraphFieldMode::CAT
                             or spec.mode == GraphFieldMode::RAGGED_CAT)
                            and graph_field_cat_dim_is_one(spec.cat_dim) and spec.dim > 1;
   if(not cat_dim_one) {
      dst.reserve(dst.size() + src.size());
      if(inc == 0) {
         dst.insert(dst.end(), src.begin(), src.end());
         return;
      }
      for(const auto value : src) {
         dst.push_back(value + inc);
      }
      return;
   }

   const size_t dim = static_cast< size_t >(spec.dim);
   if(dst.size() % dim != 0 or src.size() % dim != 0) {
      throw std::invalid_argument(
         "Graph field '" + key + "' invalid size for cat_dim=1 concatenation"
      );
   }
   const size_t dst_cols = dst.size() / dim;
   const size_t src_cols = src.size() / dim;
   std::vector< int64_t > merged(dim * (dst_cols + src_cols));
   for(size_t row = 0; row < dim; ++row) {
      const size_t dst_in = row * dst_cols;
      const size_t src_in = row * src_cols;
      const size_t out = row * (dst_cols + src_cols);
      std::copy_n(
         dst.begin() + static_cast< std::vector< int64_t >::difference_type >(dst_in),
         dst_cols,
         merged.begin() + static_cast< std::vector< int64_t >::difference_type >(out)
      );
      if(inc == 0) {
         std::copy_n(
            src.begin() + static_cast< std::vector< int64_t >::difference_type >(src_in),
            src_cols,
            merged.begin() + static_cast< std::vector< int64_t >::difference_type >(out + dst_cols)
         );
      } else {
         for(size_t col = 0; col < src_cols; ++col) {
            merged[out + dst_cols + col] = src[src_in + col] + inc;
         }
      }
   }
   dst.swap(merged);
}

int64_t rows_for_graph_field(const std::string& key, const GraphField& field)
{
   const auto total_size = std::visit(
      [](const auto& values) { return static_cast< int64_t >(values.size()); }, field.values
   );
   if(field.spec.dim <= 0) {
      throw std::invalid_argument("Graph field '" + key + "' invalid dim");
   }
   return total_size / static_cast< int64_t >(field.spec.dim);
}

int64_t rows_for_pending(const std::string& key, const GraphField& field)
{
   if(not field.pending.has_value()) {
      return 0;
   }
   const auto total_size = std::visit(
      [](const auto& values) { return static_cast< int64_t >(values.size()); }, *field.pending
   );
   if(field.spec.dim <= 0) {
      throw std::invalid_argument("Graph field '" + key + "' invalid dim");
   }
   return total_size / static_cast< int64_t >(field.spec.dim);
}

hash_map< std::string, std::string > build_edge_index_offset_node_types(
   const BatchBuilder::BatchEncoding& batch_encoding
)
{
   hash_map< std::string, std::string > node_type_by_key;
   node_type_by_key.reserve(batch_encoding.schema.edge_tensors.size());

   for(const auto& edge_spec : batch_encoding.schema.edge_tensors) {
      if(edge_spec.attr != "edge_index") {
         continue;
      }
      const int edge_type_idx = edge_spec.edge_type;
      if(edge_type_idx < 0
         or static_cast< size_t >(edge_type_idx) >= batch_encoding.schema.edge_types.size()) {
         throw std::invalid_argument(
            "append_batch_encoding encountered invalid schema edge_type index for edge_index key '"
            + edge_spec.key + "'"
         );
      }

      const auto& edge_type = batch_encoding.schema
                                 .edge_types[static_cast< size_t >(edge_type_idx)];
      std::string_view node_type;
      if(edge_spec.part == "0") {
         node_type = edge_type.src;
      } else if(edge_spec.part == "1") {
         node_type = edge_type.dst;
      } else {
         throw std::invalid_argument(
            "append_batch_encoding encountered unsupported edge_index part '" + edge_spec.part
            + "' in schema"
         );
      }

      auto [it, inserted] = node_type_by_key.try_emplace(edge_spec.key, std::string(node_type));
      if(not inserted and it->second != node_type) {
         throw std::invalid_argument(
            "append_batch_encoding schema maps edge_index key to conflicting node types for key '"
            + edge_spec.key + "'"
         );
      }
   }

   return node_type_by_key;
}

}  // namespace

BatchBuilder::BatchBuilder()
{
   constexpr size_t kSmallReserve = 32;
   constexpr size_t kColumnReserve = 64;
   current_node_counts.reserve(kSmallReserve);
   node_offsets.reserve(kSmallReserve);
   node_feature_dims.reserve(kSmallReserve);
   node_names.reserve(kSmallReserve);
   ptrs.reserve(kSmallReserve);
   columns.reserve(kColumnReserve);
   graph_kind = "hetero";
   graph_fields = nullptr;
}

void BatchBuilder::reset()
{
   constexpr size_t kSmallReserve = 32;
   constexpr size_t kColumnReserve = 64;

   current_node_counts.clear();
   current_node_counts.reserve(kSmallReserve);

   node_offsets.clear();
   node_offsets.reserve(kSmallReserve);

   node_feature_dims.clear();
   node_feature_dims.reserve(kSmallReserve);

   node_names.clear();
   node_names.reserve(kSmallReserve);

   object_names.clear();
   object_names.reserve(kSmallReserve);

   graph_kind = "hetero";
   schema_flags.clear();

   ptrs.clear();
   ptrs.reserve(kSmallReserve);

   batch_indices.clear();
   batch_indices.reserve(kSmallReserve);

   graph_attrs.clear();
   graph_attrs.reserve(kSmallReserve);

   if(graph_fields) {
      graph_fields->clear();
      graph_fields.reset();
   }

   columns.clear();
   columns.reserve(kColumnReserve);

   current_graph_idx = 0;
}

void BatchBuilder::add_node_features(
   const std::string& node_type,
   const std::string& attr_name,
   std::span< const float > data,
   int feature_dim
)
{
   set_node_feature_dim(node_type, feature_dim);

   const std::string key = make_type_attr_key(node_type, attr_name);
   auto& col = get_column< float >(key, feature_dim);
   col.insert(col.end(), data.begin(), data.end());

   const auto num_nodes = static_cast< int64_t >(data.size() / feature_dim);
   auto [it, inserted] = current_node_counts.try_emplace(node_type, num_nodes);
   if(not inserted and it->second < num_nodes) {
      it->second = num_nodes;
   }
}

void BatchBuilder::set_node_feature_dim(const std::string& node_type, int dim)
{
   auto [it, inserted] = node_feature_dims.try_emplace(node_type, dim);
   if(not inserted and it->second != dim) {
      throw std::invalid_argument(
         fmt::format("Node feature dim mismatch for node_type '{}'", node_type)
      );
   }
}

void BatchBuilder::add_nodes(const std::string& node_type, int64_t count)
{
   if(count < 0) {
      throw std::invalid_argument("Node count must be non-negative");
   }
   auto [it, inserted] = current_node_counts.try_emplace(node_type, count);
   if(not inserted and it->second < count) {
      it->second = count;
   }
}

void BatchBuilder::ensure_edge_type(
   const std::string& src_type,
   const std::string& rel_type,
   const std::string& dst_type
)
{
   const std::string edge_key_base = make_edge_type_base_key(src_type, rel_type, dst_type);
   const std::string src_key = make_edge_index_component_key(
      edge_key_base, schema_key::kEdgeIndexSrcComponent
   );
   const std::string dst_key = make_edge_index_component_key(
      edge_key_base, schema_key::kEdgeIndexDstComponent
   );
   get_column< int64_t >(src_key, 1);
   get_column< int64_t >(dst_key, 1);
}

void BatchBuilder::set_node_names(const std::string& node_type, std::vector< std::string > names)
{
   const auto graph_count = static_cast< int64_t >(names.size());
   auto [it, inserted] = node_names.try_emplace(node_type, std::vector< std::string >{});
   auto& existing = it->second;
   if(existing.empty()) {
      existing = std::move(names);
   } else {
      existing.reserve(existing.size() + names.size());
      existing.insert(existing.end(), names.begin(), names.end());
   }
   auto [count_it, count_inserted] = current_node_counts.try_emplace(node_type, graph_count);
   if(not count_inserted and count_it->second < graph_count) {
      count_it->second = graph_count;
   }
}

void BatchBuilder::set_object_names(std::vector< std::string > names)
{
   if(object_names.empty()) {
      object_names = std::move(names);
      return;
   }
   object_names.reserve(object_names.size() + names.size());
   object_names.insert(object_names.end(), names.begin(), names.end());
}

void BatchBuilder::set_graph_kind(std::string kind)
{
   graph_kind = std::move(kind);
}

void BatchBuilder::set_schema_flag(const std::string& key, bool value)
{
   schema_flags[key] = value;
}

void BatchBuilder::set_graph_attr(const std::string& key, std::vector< int64_t > values)
{
   graph_attrs[key] = std::move(values);
}

void BatchBuilder::set_graph_attr(const std::string& key, std::vector< std::string > values)
{
   graph_attrs[key] = std::move(values);
}

void BatchBuilder::set_graph_attr(const std::string& key, int64_t value)
{
   graph_attrs[key] = value;
}

void BatchBuilder::set_graph_attr(const std::string& key, std::string value)
{
   graph_attrs[key] = std::move(value);
}

void BatchBuilder::register_field(const std::string& key, const GraphFieldSpec& spec)
{
   GraphFieldSpec normalized_spec = spec;
   normalized_spec.cat_dim = normalize_graph_field_cat_dim(normalized_spec.cat_dim);
   validate_graph_field_spec(key, normalized_spec);

   if(not graph_fields) {
      graph_fields = std::make_unique< hash_map< std::string, GraphField > >();
      graph_fields->reserve(8);
   }

   auto [it, inserted] = graph_fields->try_emplace(key, GraphField{});
   auto& field = it->second;
   if(inserted) {
      field.spec = normalized_spec;
      field.values = make_numeric_column_data(normalized_spec.dtype);
      if(normalized_spec.mode == GraphFieldMode::RAGGED_CAT) {
         field.ptr = {0};
      }
      return;
   }

   if(field.spec != normalized_spec) {
      throw std::invalid_argument("Graph field '" + key + "' registered with different spec");
   }
}

std::vector< std::string > BatchBuilder::field_keys() const
{
   std::vector< std::string > out;
   if(not graph_fields) {
      return out;
   }
   out.reserve(graph_fields->size());
   for(const auto& [key, field] : *graph_fields) {
      (void) field;
      out.push_back(key);
   }
   std::ranges::sort(out);
   return out;
}

absl::btree_map< std::string, GraphFieldSpec > BatchBuilder::field_specs() const
{
   absl::btree_map< std::string, GraphFieldSpec > out;
   if(not graph_fields) {
      return out;
   }
   for(const auto& [key, field] : *graph_fields) {
      out.emplace(key, field.spec);
   }
   return out;
}

GraphFieldSpec BatchBuilder::get_graph_field_spec(const std::string& key) const
{
   if(not graph_fields) {
      throw std::invalid_argument("Graph field '" + key + "' is not registered");
   }
   const auto it = graph_fields->find(key);
   if(it == graph_fields->end()) {
      throw std::invalid_argument("Graph field '" + key + "' is not registered");
   }
   return it->second.spec;
}

void BatchBuilder::set_field(const std::string& key, std::span< const float > values)
{
   if(not graph_fields) {
      throw std::invalid_argument("Graph field '" + key + "' is not registered");
   }
   auto it = graph_fields->find(key);
   if(it == graph_fields->end()) {
      throw std::invalid_argument("Graph field '" + key + "' is not registered");
   }
   auto& field = it->second;
   if(field.spec.dtype != GraphFieldDType::F32) {
      throw std::invalid_argument("Graph field '" + key + "' expects dtype=i64");
   }
   if(field.pending.has_value()) {
      throw std::invalid_argument(
         "Graph field '" + key + "' can only be assigned once per graph before next_graph()"
      );
   }

   const int64_t total = static_cast< int64_t >(values.size());
   if(field.spec.mode == GraphFieldMode::STACK or field.spec.mode == GraphFieldMode::CONST) {
      if(total != field.spec.dim) {
         throw std::invalid_argument(
            "Graph field '" + key + "' STACK/CONST expects exactly dim values"
         );
      }
   } else if(field.spec.mode == GraphFieldMode::RAGGED_CAT
             or field.spec.mode == GraphFieldMode::CAT) {
      if(total % field.spec.dim != 0) {
         throw std::invalid_argument(
            "Graph field '" + key + "' values size must be divisible by dim"
         );
      }
   }

   std::vector< float > data(values.begin(), values.end());
   field.pending = NumericColumnData{std::move(data)};
}

void BatchBuilder::set_field(const std::string& key, std::span< const int64_t > values)
{
   if(not graph_fields) {
      throw std::invalid_argument("Graph field '" + key + "' is not registered");
   }
   auto it = graph_fields->find(key);
   if(it == graph_fields->end()) {
      throw std::invalid_argument("Graph field '" + key + "' is not registered");
   }
   auto& field = it->second;
   if(field.spec.dtype != GraphFieldDType::I64) {
      throw std::invalid_argument("Graph field '" + key + "' expects dtype=f32");
   }
   if(field.pending.has_value()) {
      throw std::invalid_argument(
         "Graph field '" + key + "' can only be assigned once per graph before next_graph()"
      );
   }

   const int64_t total = static_cast< int64_t >(values.size());
   if(field.spec.mode == GraphFieldMode::STACK or field.spec.mode == GraphFieldMode::CONST) {
      if(total != field.spec.dim) {
         throw std::invalid_argument(
            "Graph field '" + key + "' STACK/CONST expects exactly dim values"
         );
      }
   } else if(field.spec.mode == GraphFieldMode::RAGGED_CAT
             or field.spec.mode == GraphFieldMode::CAT) {
      if(total % field.spec.dim != 0) {
         throw std::invalid_argument(
            "Graph field '" + key + "' values size must be divisible by dim"
         );
      }
   }

   std::vector< int64_t > data(values.begin(), values.end());
   field.pending = NumericColumnData{std::move(data)};
}

void BatchBuilder::commit_graph_fields()
{
   if(not graph_fields) {
      return;
   }

   hash_map< std::string, int64_t > graph_field_incs;
   graph_field_incs.reserve(graph_fields->size());
   for(const auto& [key, field] : *graph_fields) {
      int64_t inc = 0;
      if(field.spec.inc.kind == GraphFieldInc::Kind::NODE_OFFSET) {
         auto it = node_offsets.find(field.spec.inc.node_type);
         if(it != node_offsets.end()) {
            inc = it->second;
         }
      } else if(field.spec.inc.kind == GraphFieldInc::Kind::FIELD_OFFSET) {
         inc = committed_field_offset(*graph_fields, key, field.spec);
      }
      graph_field_incs.emplace(key, inc);
   }

   for(auto& [key, field] : *graph_fields) {
      const bool has_value = field.pending.has_value();
      const int64_t inc = graph_field_incs.at(key);
      switch(field.spec.mode) {
         case GraphFieldMode::STACK: {
            if(not has_value) {
               throw std::invalid_argument(
                  "Graph field '" + key + "' (STACK) must be assigned for every graph"
               );
            }
            append_with_inc(key, field.spec, field.values, *field.pending, inc);
            field.pending.reset();
            break;
         }
         case GraphFieldMode::RAGGED_CAT: {
            if(field.ptr.empty()) {
               field.ptr.push_back(0);
            }
            int64_t rows = 0;
            if(has_value) {
               rows = rows_for_pending(key, field);
               append_with_inc(key, field.spec, field.values, *field.pending, inc);
            }
            field.ptr.push_back(field.ptr.back() + rows);
            field.pending.reset();
            break;
         }
         case GraphFieldMode::CAT: {
            if(has_value) {
               append_with_inc(key, field.spec, field.values, *field.pending, inc);
            }
            field.pending.reset();
            break;
         }
         case GraphFieldMode::CONST: {
            if(not has_value) {
               throw std::invalid_argument(
                  "Graph field '" + key + "' (CONST) must be assigned for every graph"
               );
            }
            if(current_graph_idx == 0) {
               field.values = *field.pending;
            } else if(field.values != *field.pending) {
               throw std::invalid_argument(
                  "Graph field '" + key + "' (CONST) value mismatch across graphs"
               );
            }
            field.pending.reset();
            break;
         }
      }
   }
}

void BatchBuilder::add_edges(
   const std::string& src_type,
   const std::string& rel_type,
   const std::string& dst_type,
   std::span< const int64_t > src_indices,
   std::span< const int64_t > dst_indices
)
{
   if(src_indices.size() != dst_indices.size()) {
      throw std::invalid_argument("src and dst indices must have same length");
   }

   // We stick to storing separate src and dst index columns for now as they are easier to build.
   // Construct keys: "src_type|rel_type|dst_type/edge_index_0"
   const std::string edge_key_base = make_edge_type_base_key(src_type, rel_type, dst_type);
   const std::string src_key = make_edge_index_component_key(
      edge_key_base, schema_key::kEdgeIndexSrcComponent
   );
   const std::string dst_key = make_edge_index_component_key(
      edge_key_base, schema_key::kEdgeIndexDstComponent
   );

   // Create both columns before taking references; inserting the second key may
   // rehash the underlying map.
   (void) get_column< int64_t >(src_key, 1);
   (void) get_column< int64_t >(dst_key, 1);
   auto& col_src = std::get< LongCol >(columns.at(src_key).data);
   auto& col_dst = std::get< LongCol >(columns.at(dst_key).data);

   int64_t src_offset = node_offsets.try_emplace(src_type, 0).first->second;
   int64_t dst_offset = node_offsets.try_emplace(dst_type, 0).first->second;

   col_src.reserve(col_src.size() + src_indices.size());
   col_dst.reserve(col_dst.size() + dst_indices.size());

   // Apply offsets and push
   for(auto idx : src_indices)
      col_src.emplace_back(idx + src_offset);
   for(auto idx : dst_indices)
      col_dst.emplace_back(idx + dst_offset);
}

void BatchBuilder::add_edge(
   const std::string& src_type,
   const std::string& rel_type,
   const std::string& dst_type,
   int64_t src_index,
   int64_t dst_index
)
{
   const std::string edge_key_base = make_edge_type_base_key(src_type, rel_type, dst_type);
   const std::string src_key = make_edge_index_component_key(
      edge_key_base, schema_key::kEdgeIndexSrcComponent
   );
   const std::string dst_key = make_edge_index_component_key(
      edge_key_base, schema_key::kEdgeIndexDstComponent
   );

   // Create both columns before taking references; inserting the second key may
   // rehash the underlying map.
   (void) get_column< int64_t >(src_key, 1);
   (void) get_column< int64_t >(dst_key, 1);
   auto& col_src = std::get< LongCol >(columns.at(src_key).data);
   auto& col_dst = std::get< LongCol >(columns.at(dst_key).data);

   int64_t src_offset = node_offsets.try_emplace(src_type, 0).first->second;
   int64_t dst_offset = node_offsets.try_emplace(dst_type, 0).first->second;

   col_src.emplace_back(src_index + src_offset);
   col_dst.emplace_back(dst_index + dst_offset);
}

void BatchBuilder::add_edge_features(
   const std::string& src_type,
   const std::string& rel_type,
   const std::string& dst_type,
   const std::string& attr_name,
   std::span< const float > data,
   int feature_dim
)
{
   const std::string key = make_type_attr_key(
      make_edge_type_base_key(src_type, rel_type, dst_type), attr_name
   );
   auto& col = get_column< float >(key, feature_dim);
   col.insert(col.end(), data.begin(), data.end());
}

void BatchBuilder::next_graph()
{
   commit_graph_fields();

   for(auto& [ntype, count] : current_node_counts) {
      auto& offset = node_offsets.try_emplace(ntype, 0).first->second;
      offset += count;

      auto& p = ptrs.try_emplace(ntype, std::vector< int64_t >{}).first->second;
      if(p.empty()) {
         p.emplace_back(0);
      }
      p.emplace_back(offset);

      count = 0;
   }
   // Batch indices tracking could go here if we want homogeneous batch vector
   current_graph_idx++;
}

// --- DLPack Owner Capsule ---

template < typename T >
nb::object vector_to_1d_dlpack(std::vector< T >&& vec)
{
   return dlpack_utils::vector_to_dlpack_owned_1d(std::move(vec));
}

template < typename T >
nb::object vector_to_2d_dlpack(std::vector< T >&& vec, size_t rows, size_t cols)
{
   return dlpack_utils::vector_to_dlpack_owned_2d(std::move(vec), rows, cols);
}

// --- Build / Export ---

nb::dict BatchBuilder::build_dict()
{
   // Destructive export: tensor backing vectors are moved into Python-owned
   // DLPack capsules to avoid copies.
   nb::dict out;

   for(auto& [key, col] : columns) {
      const bool is_edge_index = key_has_edge_index_prefix(key);
      std::visit(
         [&]< typename T >(std::vector< T >& vec) {
            size_t size = vec.size();
            if(is_edge_index) {
               out[key.c_str()] = vector_to_1d_dlpack(std::move(vec));
               return;
            }

            int dim = col.dim;
            size_t num_rows = dim > 0 ? size / dim : 0;
            out[key.c_str()] = vector_to_2d_dlpack(
               std::move(vec), num_rows, static_cast< size_t >(dim)
            );
         },
         col.data
      );
   }

   // Also export Ptr columns (converting them to tensor columns first
   // essentially)
   for(auto& [ntype, p_vec] : ptrs) {
      auto tensor = vector_to_1d_dlpack(std::move(p_vec));
      std::string key = make_type_attr_key(ntype, schema_key::kPtrAttr);
      out[key.c_str()] = tensor;
   }

   return out;
}

nb::object BatchBuilder::build_pyg()
{
   absl::btree_map< std::string, int64_t > node_counts;
   for(const auto& [key, col] : columns) {
      if(key_has_edge_separator(key)) {
         continue;
      }
      const auto slash = find_type_attr_separator(key);
      if(slash == std::string::npos) {
         continue;
      }
      const std::string node_type = key.substr(0, slash);
      std::visit(
         [&](const auto& items) {
            const size_t size = items.size();
            const int dim = col.dim;
            const int64_t rows = dim > 0 ? static_cast< int64_t >(size / dim) : 0;
            auto& count = node_counts[node_type];
            if(rows > count) {
               count = rows;
            }
         },
         col.data
      );
   }
   for(const auto& [node_type, ptr] : ptrs) {
      if(not ptr.empty()) {
         const int64_t count = ptr.back();
         auto& existing = node_counts[node_type];
         if(count > existing) {
            existing = count;
         }
      }
   }
   for(const auto& [node_type, count] : current_node_counts) {
      auto& existing = node_counts[node_type];
      if(count > existing) {
         existing = count;
      }
   }
   for(const auto& [node_type, names] : node_names) {
      auto& existing = node_counts[node_type];
      const int64_t count = static_cast< int64_t >(names.size());
      if(count > existing) {
         existing = count;
      }
   }
   for(const auto& [node_type, dim] : node_feature_dims) {
      (void) dim;
      if(not node_counts.contains(node_type)) {
         node_counts[node_type] = 0;
      }
   }

   absl::btree_map< std::string, std::vector< int64_t > > ptr_vectors;
   absl::btree_map< std::string, std::vector< int64_t > > batch_vectors;
   int64_t graph_count = 0;
   for(const auto& [node_type, ptr] : ptrs) {
      if(ptr.size() < 2) {
         continue;
      }
      ptr_vectors[node_type] = ptr;
      graph_count = std::max< int64_t >(graph_count, ptr.size() - 1);
      std::vector< int64_t > batch;
      batch.reserve(ptr.back());
      for(size_t idx = 0; idx + 1 < ptr.size(); ++idx) {
         const int64_t count = ptr[idx + 1] - ptr[idx];
         batch.insert(batch.end(), count, static_cast< int64_t >(idx));
      }
      batch_vectors[node_type] = std::move(batch);
   }
   if(ptr_vectors.empty()) {
      for(const auto& [node_type, count] : node_counts) {
         if(count <= 0) {
            continue;
         }
         ptr_vectors[node_type] = {0, count};
         batch_vectors[node_type] = std::vector< int64_t >(count, 0);
      }
      if(not node_counts.empty()) {
         graph_count = 1;
      }
   }

   nb::dict payload = build_dict();

   nb::object batch = py::torch_geometric_batch_ctor()(
      nb::arg("_base_cls") = py::torch_geometric_heterodata_ctor()
   );

   using EdgeKey = std::tuple< std::string, std::string, std::string >;
   struct EdgeComponents {
      nb::object src;
      nb::object dst;
   };
   absl::btree_map< EdgeKey, EdgeComponents > edge_components;

   for(auto [key_handle, value_handle] : payload) {
      const std::string key = nb::str(key_handle).c_str();
      if(key_has_ptr_suffix(key)) {
         continue;
      }

      const auto edge_pos = key.rfind(schema_key::kEdgeIndexKeyPrefix);
      if(edge_pos != std::string::npos) {
         const std::string base = key.substr(0, edge_pos);
         const std::string suffix = key.substr(edge_pos + schema_key::kEdgeIndexKeyPrefix.size());
         const auto first = base.find(schema_key::kEdgeTypeSeparator);
         const auto second = base.find(schema_key::kEdgeTypeSeparator, first + 1);
         if(first == std::string::npos or second == std::string::npos) {
            throw std::invalid_argument(fmt::format("Malformed edge key '{}'", key));
         }
         const std::string src = base.substr(0, first);
         const std::string rel = base.substr(first + 1, second - first - 1);
         const std::string dst = base.substr(second + 1);

         EdgeKey edge_key{src, rel, dst};
         auto& components = edge_components[edge_key];
         nb::object tensor = py::to_torch_tensor(nb::borrow< nb::object >(value_handle));
         if(suffix.size() == 1 and suffix.front() == schema_key::kEdgeIndexSrcComponent) {
            components.src = tensor;
         } else if(suffix.size() == 1 and suffix.front() == schema_key::kEdgeIndexDstComponent) {
            components.dst = tensor;
         } else {
            throw std::invalid_argument(fmt::format("Unexpected edge index suffix '{}'", key));
         }
         continue;
      }

      const auto slash = find_type_attr_separator(key);
      if(slash == std::string::npos) {
         continue;
      }
      const std::string type_key = key.substr(0, slash);
      const std::string attr = key.substr(slash + 1);

      const auto first = type_key.find(schema_key::kEdgeTypeSeparator);
      const auto second = first == std::string::npos
                             ? std::string::npos
                             : type_key.find(schema_key::kEdgeTypeSeparator, first + 1);
      const auto third = second == std::string::npos
                            ? std::string::npos
                            : type_key.find(schema_key::kEdgeTypeSeparator, second + 1);
      const bool is_edge_type = first != std::string::npos and second != std::string::npos
                                and third == std::string::npos;
      nb::object store;
      if(is_edge_type) {
         const std::string src = type_key.substr(0, first);
         const std::string rel = type_key.substr(first + 1, second - first - 1);
         const std::string dst = type_key.substr(second + 1);
         store = batch.attr("__getitem__")(nb::make_tuple(src, rel, dst));
      } else {
         store = batch.attr("__getitem__")(type_key);
      }
      nb::object tensor = py::to_torch_tensor(nb::borrow< nb::object >(value_handle));
      store.attr("__setitem__")(attr, tensor);
   }

   for(const auto& [edge_key, components] : edge_components) {
      if(not components.src.is_valid() or not components.dst.is_valid()) {
         throw std::invalid_argument("Incomplete edge_index components for edge type");
      }
      nb::object edge_index = py::torch_stack_fn()(
         nb::make_tuple(components.src, components.dst), nb::arg("dim") = 0
      );
      nb::object store = batch.attr("__getitem__")(
         nb::make_tuple(std::get< 0 >(edge_key), std::get< 1 >(edge_key), std::get< 2 >(edge_key))
      );
      store.attr("__setitem__")("edge_index", edge_index);
   }

   for(const auto& [node_type, ptr] : ptr_vectors) {
      nb::object ptr_tensor = py::to_torch_tensor(
         dlpack_utils::vector_to_dlpack_owned_copy_1d(ptr)
      );

      auto batch_it = batch_vectors.find(node_type);
      const std::vector< int64_t > batch_values = batch_it != batch_vectors.end()
                                                     ? batch_it->second
                                                     : std::vector< int64_t >{};
      nb::object batch_tensor = py::to_torch_tensor(
         dlpack_utils::vector_to_dlpack_owned_copy_1d(batch_values)
      );

      nb::object store = batch.attr("__getitem__")(node_type);
      store.attr("__setitem__")("ptr", ptr_tensor);
      store.attr("__setitem__")("batch", batch_tensor);
   }

   for(const auto& [node_type, count] : node_counts) {
      nb::object store = batch.attr("__getitem__")(node_type);
      bool has_x = nb::cast< bool >(store.attr("__contains__")("x"));
      if(not has_x) {
         int dim = 0;
         auto dim_it = node_feature_dims.find(node_type);
         if(dim_it != node_feature_dims.end()) {
            dim = dim_it->second;
         }
         nb::object zeros = py::torch_zeros_fn()(
            nb::make_tuple(count, dim), nb::arg("dtype") = py::torch_float32_dtype()
         );
         store.attr("__setitem__")("x", zeros);
      }
   }

   for(const auto& [node_type, names] : node_names) {
      nb::object store = batch.attr("__getitem__")(node_type);
      if(graph_count > 0) {
         std::vector< std::vector< std::string > > per_graph;
         auto ptr_it = ptr_vectors.find(node_type);
         if(ptr_it != ptr_vectors.end() and ptr_it->second.size() >= 2) {
            const auto& ptr = ptr_it->second;
            per_graph.reserve(ptr.size() - 1);
            for(size_t i = 0; i + 1 < ptr.size(); ++i) {
               const auto start = static_cast< size_t >(std::max< int64_t >(0, ptr[i]));
               const auto end = static_cast< size_t >(
                  std::min< int64_t >(ptr[i + 1], static_cast< int64_t >(names.size()))
               );
               if(start <= end and end <= names.size()) {
                  per_graph.emplace_back(names.begin() + start, names.begin() + end);
               } else {
                  per_graph.emplace_back();
               }
            }
         } else {
            per_graph.emplace_back(names);
         }
         store.attr("node_names") = nb::cast(per_graph);
      } else {
         store.attr("node_names") = nb::cast(names);
      }
   }
   if(not object_names.empty()) {
      if(graph_count > 0) {
         std::vector< std::vector< std::string > > per_graph;
         bool assigned = false;
         for(const auto& [node_type, names] : node_names) {
            if(names != object_names) {
               continue;
            }
            auto ptr_it = ptr_vectors.find(node_type);
            if(ptr_it == ptr_vectors.end() or ptr_it->second.size() < 2) {
               break;
            }
            const auto& ptr = ptr_it->second;
            per_graph.reserve(ptr.size() - 1);
            for(size_t i = 0; i + 1 < ptr.size(); ++i) {
               const auto start = static_cast< size_t >(std::max< int64_t >(0, ptr[i]));
               const auto end = static_cast< size_t >(
                  std::min< int64_t >(ptr[i + 1], static_cast< int64_t >(object_names.size()))
               );
               if(start <= end and end <= object_names.size()) {
                  per_graph.emplace_back(object_names.begin() + start, object_names.begin() + end);
               } else {
                  per_graph.emplace_back();
               }
            }
            assigned = true;
            break;
         }
         if(not assigned) {
            per_graph.emplace_back(object_names);
         }
         batch.attr("object_names") = nb::cast(per_graph);
      } else {
         batch.attr("object_names") = nb::cast(object_names);
      }
   }

   set_graph_attrs_on_pyg_batch(batch, graph_attrs, graph_fields);

   if(graph_fields) {
      for(auto& [attr, field] : *graph_fields) {
         nb::object value_tensor;
         std::visit(
            [&](auto& values) {
               using T = std::decay_t< decltype(values) >::value_type;
               const size_t size = values.size();
               if(field.spec.dim == 1) {
                  value_tensor = py::to_torch_tensor(vector_to_1d_dlpack< T >(std::move(values)));
               } else {
                  const bool cat_dim_one = (field.spec.mode == GraphFieldMode::CAT
                                            or field.spec.mode == GraphFieldMode::RAGGED_CAT)
                                           and graph_field_cat_dim_is_one(field.spec.cat_dim);
                  const size_t rows = cat_dim_one ? static_cast< size_t >(field.spec.dim)
                                                  : size / static_cast< size_t >(field.spec.dim);
                  const size_t cols = cat_dim_one ? size / static_cast< size_t >(field.spec.dim)
                                                  : static_cast< size_t >(field.spec.dim);
                  value_tensor = py::to_torch_tensor(
                     vector_to_2d_dlpack< T >(std::move(values), rows, cols)
                  );
               }
            },
            field.values
         );
         batch.attr("__setattr__")(attr.c_str(), value_tensor);

         if(field.spec.mode == GraphFieldMode::RAGGED_CAT) {
            std::string ptr_attr = attr + "_ptr";
            batch.attr("__setattr__")(
               ptr_attr.c_str(), py::to_torch_tensor(vector_to_1d_dlpack(std::move(field.ptr)))
            );
         }
      }
   }

   if(graph_count > 0) {
      batch.attr("_num_graphs") = graph_count;
   }

   reset();
   return batch;
}

BatchBuilder::BatchEncoding BatchBuilder::build()
{
   bool has_uncommitted_nodes = false;
   for(const auto& [node_type, count] : current_node_counts) {
      (void) node_type;
      if(count > 0) {
         has_uncommitted_nodes = true;
         break;
      }
   }
   bool has_pending_graph_fields = false;
   if(graph_fields) {
      for(const auto& [key, field] : *graph_fields) {
         (void) key;
         if(field.pending.has_value()) {
            has_pending_graph_fields = true;
            break;
         }
      }
   }
   if(has_uncommitted_nodes or has_pending_graph_fields) {
      next_graph();
   }

   absl::btree_map< std::string, int64_t > node_counts;
   for(const auto& [key, col] : columns) {
      if(key_has_edge_separator(key)) {
         continue;
      }
      const auto slash = find_type_attr_separator(key);
      if(slash == std::string::npos) {
         continue;
      }
      const std::string node_type = key.substr(0, slash);
      std::visit(
         [&](const auto& items) {
            const size_t size = items.size();
            const int dim = col.dim;
            const int64_t rows = dim > 0 ? static_cast< int64_t >(size / dim) : 0;
            auto& count = node_counts[node_type];
            if(rows > count) {
               count = rows;
            }
         },
         col.data
      );
   }
   for(const auto& [node_type, ptr] : ptrs) {
      if(not ptr.empty()) {
         const int64_t count = ptr.back();
         auto& existing = node_counts[node_type];
         if(count > existing) {
            existing = count;
         }
      }
   }
   for(const auto& [node_type, names] : node_names) {
      auto& existing = node_counts[node_type];
      const int64_t count = static_cast< int64_t >(names.size());
      if(count > existing) {
         existing = count;
      }
   }
   for(const auto& [node_type, dim] : node_feature_dims) {
      (void) dim;
      if(not node_counts.contains(node_type)) {
         node_counts[node_type] = 0;
      }
   }

   absl::btree_map< std::string, std::vector< int64_t > > ptr_vectors;
   int64_t graph_count = 0;
   for(const auto& [node_type, ptr] : ptrs) {
      if(ptr.size() < 2) {
         continue;
      }
      ptr_vectors[node_type] = ptr;
      graph_count = std::max< int64_t >(graph_count, ptr.size() - 1);
   }
   if(ptr_vectors.empty()) {
      for(const auto& [node_type, count] : node_counts) {
         if(count <= 0) {
            continue;
         }
         ptr_vectors[node_type] = {0, count};
      }
      if(not node_counts.empty()) {
         graph_count = 1;
      }
   }
   if(graph_count == 0 and current_graph_idx > 0) {
      graph_count = current_graph_idx;
   }

   std::vector< NodeTensorSpec > node_specs;
   struct EdgeTensorKeySpec {
      EdgeType edge_type;
      std::string attr;
      std::string part;
      std::string key;
   };
   std::vector< EdgeTensorKeySpec > edge_specs;
   std::vector< EdgeType > edge_types;

   for(const auto& [key, col] : columns) {
      (void) col;
      const auto slash = find_type_attr_separator(key);
      if(slash == std::string::npos) {
         continue;
      }
      const bool is_edge = key_has_edge_separator(key);
      if(not is_edge) {
         node_specs.push_back(
            NodeTensorSpec{
               key.substr(0, slash),
               key.substr(slash + 1),
               key,
            }
         );
         continue;
      }
      const std::string base = key.substr(0, slash);
      const std::string attr = key.substr(slash + 1);
      const auto first = base.find(schema_key::kEdgeTypeSeparator);
      if(first == std::string::npos) {
         continue;
      }
      const auto second = base.find(schema_key::kEdgeTypeSeparator, first + 1);
      if(second == std::string::npos) {
         continue;
      }
      const EdgeType& edge_key = edge_types.emplace_back(
         base.substr(0, first),  //
         base.substr(first + 1, second - first - 1),
         base.substr(second + 1)
      );

      std::string part;
      std::string attr_name = attr;
      if(attr.rfind(schema_key::kEdgeIndexAttrPrefix, 0) == 0) {
         attr_name = "edge_index";
         part = attr.substr(schema_key::kEdgeIndexAttrPrefix.size());
      }
      edge_specs.push_back(
         EdgeTensorKeySpec{
            edge_key,
            attr_name,
            part,
            key,
         }
      );
   }

   for(const auto& [node_type, ptr] : ptr_vectors) {
      (void) ptr;
      node_specs.push_back(
         NodeTensorSpec{
            node_type,
            std::string(schema_key::kPtrAttr),
            make_type_attr_key(node_type, schema_key::kPtrAttr),
         }
      );
      node_specs.push_back(
         NodeTensorSpec{
            node_type,
            std::string(schema_key::kBatchAttr),
            make_type_attr_key(node_type, schema_key::kBatchAttr),
         }
      );
   }

   std::ranges::sort(node_specs, [](const auto& lhs, const auto& rhs) {
      return lhs.key < rhs.key;
   });
   std::ranges::sort(edge_specs, [](const auto& lhs, const auto& rhs) {
      return lhs.key < rhs.key;
   });

   // deduplicate edge types
   std::ranges::sort(edge_types);
   auto uniq = std::ranges::unique(edge_types);
   edge_types.erase(uniq.begin(), edge_types.end());

   absl::btree_map< EdgeType, int > edge_type_ids;
   for(auto&& [idx, edge_type] : ranges::views::enumerate(edge_types)) {
      edge_type_ids.emplace(edge_type, static_cast< int >(idx));
   }

   std::vector< std::string > node_types;
   node_types.reserve(node_counts.size());
   for(const auto& [node_type, count] : node_counts) {
      (void) count;
      node_types.push_back(node_type);
   }

   std::vector< EdgeTensorSpec > edge_tensor_specs;
   edge_tensor_specs.reserve(edge_specs.size());
   for(const auto& spec : edge_specs) {
      const auto it = edge_type_ids.find(spec.edge_type);
      if(it == edge_type_ids.end()) {
         throw std::invalid_argument("Edge tensor spec references unknown edge type");
      }
      edge_tensor_specs.emplace_back(
         EdgeTensorSpec{
            .edge_type = it->second,
            .attr = spec.attr,
            .key = spec.key,
            .part = spec.part,
         }
      );
   }

   std::vector< GraphTensorSpec > graph_tensor_specs;
   if(graph_fields) {
      graph_tensor_specs.reserve(graph_fields->size());
      for(const auto& [attr, field] : *graph_fields) {
         std::string key = make_type_attr_key("__graph__", attr);
         std::string ptr_key;
         if(field.spec.mode == GraphFieldMode::RAGGED_CAT) {
            ptr_key = make_type_attr_key(key, schema_key::kPtrAttr);
         }
         graph_tensor_specs.push_back(
            GraphTensorSpec{
               .attr = attr,
               .key = std::move(key),
               .ptr_key = std::move(ptr_key),
               .mode = field.spec.mode,
               .dtype = field.spec.dtype,
               .dim = field.spec.dim,
               .cat_dim = field.spec.cat_dim,
               .inc = field.spec.inc,
            }
         );
      }
   }
   std::ranges::sort(graph_tensor_specs, [](const auto& lhs, const auto& rhs) {
      return lhs.key < rhs.key;
   });

   hash_map< std::string, GraphField > built_graph_fields;
   if(graph_fields) {
      built_graph_fields = std::move(*graph_fields);
      for(auto& [key, field] : built_graph_fields) {
         (void) key;
         field.pending.reset();
      }
   }

   Schema schema;
   schema.version = 1;
   schema.graph_kind = graph_kind;
   schema.node_types = std::move(node_types);
   schema.edge_types = std::move(edge_types);
   schema.node_tensors = std::move(node_specs);
   schema.edge_tensors = std::move(edge_tensor_specs);
   schema.graph_tensors = std::move(graph_tensor_specs);
   schema.flags = schema_flags;
   schema.validate();

   BatchEncoding out{
      .columns = std::move(columns),
      .node_names = std::move(node_names),
      .object_names = std::move(object_names),
      .node_feature_dims = std::move(node_feature_dims),
      .graph_attrs = std::move(graph_attrs),
      .graph_fields = std::move(built_graph_fields),
      .ptrs = std::move(ptrs),
      .schema_flags = std::move(schema_flags),
      .graph_kind = std::move(graph_kind),
      .num_graphs = graph_count,
      .node_counts = std::move(node_counts),
      .schema = std::move(schema)
   };
   reset();
   return out;
}

void BatchBuilder::append_batch_encoding(const BatchEncoding& batch_encoding)
{
   if(batch_encoding.num_graphs <= 0) {
      return;
   }
   if(batch_encoding.num_graphs != 1) {
      throw std::invalid_argument("append_batch_encoding expects num_graphs == 1");
   }
   for(const auto& [key, field] : batch_encoding.graph_fields) {
      try {
         validate_graph_field_storage(key, field, batch_encoding.num_graphs);
      } catch(const std::exception& ex) {
         throw std::invalid_argument(
            "append_batch_encoding invalid graph field '" + key + "': " + ex.what()
         );
      }
   }

   if(graph_kind.empty()) {
      graph_kind = batch_encoding.graph_kind;
   } else if(not batch_encoding.graph_kind.empty() and graph_kind != batch_encoding.graph_kind) {
      throw std::invalid_argument("append_batch_encoding graph_kind mismatch");
   }

   for(const auto& [key, value] : batch_encoding.schema_flags) {
      auto [it, inserted] = schema_flags.try_emplace(key, value);
      if(not inserted and it->second != value) {
         throw std::invalid_argument("append_batch_encoding schema flag mismatch");
      }
   }

   for(const auto& [node_type, dim] : batch_encoding.node_feature_dims) {
      set_node_feature_dim(node_type, dim);
   }
   for(const auto& [node_type, names] : batch_encoding.node_names) {
      set_node_names(node_type, names);
   }
   if(not batch_encoding.object_names.empty()) {
      set_object_names(batch_encoding.object_names);
   }
   for(const auto& [key, value] : batch_encoding.graph_attrs) {
      std::visit([&](const auto& v) { set_graph_attr(key, v); }, value);
   }

   absl::btree_map< std::string, int64_t > node_counts = batch_encoding.node_counts;
   if(node_counts.empty()) {
      for(const auto& [key, col] : batch_encoding.columns) {
         if(key_has_edge_separator(key)) {
            continue;
         }
         const auto slash = find_type_attr_separator(key);
         if(slash == std::string::npos) {
            continue;
         }
         const std::string node_type = key.substr(0, slash);
         std::visit(
            [&](const auto& items) {
               const size_t size = items.size();
               const int dim = col.dim;
               const int64_t rows = dim > 0 ? static_cast< int64_t >(size / dim) : 0;
               auto& count = node_counts[node_type];
               if(rows > count) {
                  count = rows;
               }
            },
            col.data
         );
      }
      for(const auto& [node_type, ptr] : batch_encoding.ptrs) {
         if(not ptr.empty()) {
            const int64_t count = ptr.back();
            auto& existing = node_counts[node_type];
            if(count > existing) {
               existing = count;
            }
         }
      }
      for(const auto& [node_type, names] : batch_encoding.node_names) {
         auto& existing = node_counts[node_type];
         const int64_t count = static_cast< int64_t >(names.size());
         if(count > existing) {
            existing = count;
         }
      }
      for(const auto& [node_type, dim] : batch_encoding.node_feature_dims) {
         (void) dim;
         if(not node_counts.contains(node_type)) {
            node_counts[node_type] = 0;
         }
      }
   }

   for(const auto& [node_type, count] : node_counts) {
      add_nodes(node_type, count);
   }

   auto offset_for = [&](std::string_view node_type) -> int64_t {
      auto it = node_offsets.find(node_type);
      if(it == node_offsets.end()) {
         return 0;
      }
      return it->second;
   };

   const auto edge_index_offset_node_types = build_edge_index_offset_node_types(batch_encoding);

   if(graph_fields or not batch_encoding.graph_fields.empty()) {
      if(not graph_fields) {
         graph_fields = std::make_unique< hash_map< std::string, GraphField > >();
         graph_fields->reserve(batch_encoding.graph_fields.size());
      }
      for(const auto& [key, field] : batch_encoding.graph_fields) {
         register_field(key, field.spec);
      }

      for(auto& [key, field] : *graph_fields) {
         const auto src_it = batch_encoding.graph_fields.find(key);
         if(src_it == batch_encoding.graph_fields.end()) {
            if(field.spec.mode == GraphFieldMode::STACK
               or field.spec.mode == GraphFieldMode::CONST) {
               throw std::invalid_argument(
                  "append_batch_encoding missing required graph field '" + key + "'"
               );
            }
            if(field.spec.mode == GraphFieldMode::RAGGED_CAT) {
               if(field.ptr.empty()) {
                  field.ptr.push_back(0);
               }
               field.ptr.push_back(field.ptr.back());
            }
            continue;
         }

         const auto& src_field = src_it->second;
         if(src_field.spec != field.spec) {
            throw std::invalid_argument(
               "append_batch_encoding field spec mismatch for '" + key + "'"
            );
         }

         int64_t inc = 0;
         if(field.spec.inc.kind == GraphFieldInc::Kind::NODE_OFFSET) {
            inc = offset_for(field.spec.inc.node_type);
         } else if(field.spec.inc.kind == GraphFieldInc::Kind::FIELD_OFFSET) {
            inc = committed_field_offset(*graph_fields, key, field.spec);
         }

         switch(field.spec.mode) {
            case GraphFieldMode::STACK:
               append_with_inc(key, field.spec, field.values, src_field.values, inc);
               break;
            case GraphFieldMode::RAGGED_CAT: {
               append_with_inc(key, field.spec, field.values, src_field.values, inc);
               int64_t rows = 0;
               if(src_field.ptr.size() >= 2) {
                  rows = src_field.ptr.back() - src_field.ptr.front();
               } else {
                  rows = rows_for_graph_field(key, src_field);
               }
               if(field.ptr.empty()) {
                  field.ptr.push_back(0);
               }
               field.ptr.push_back(field.ptr.back() + rows);
               break;
            }
            case GraphFieldMode::CONST:
               if(rows_for_graph_field(key, field) == 0) {
                  field.values = src_field.values;
               } else if(field.values != src_field.values) {
                  throw std::invalid_argument(
                     "append_batch_encoding CONST graph field mismatch for '" + key + "'"
                  );
               }
               break;
            case GraphFieldMode::CAT:
               append_with_inc(key, field.spec, field.values, src_field.values, inc);
               break;
         }
      }
   }

   for(const auto& [key, col] : batch_encoding.columns) {
      if(key_has_ptr_suffix(key)) {
         continue;
      }
      if(key_has_batch_suffix(key)) {
         continue;
      }

      int64_t edge_index_offset = 0;
      bool has_edge_index_offset = false;
      if(const auto it = edge_index_offset_node_types.find(key);
         it != edge_index_offset_node_types.end()) {
         edge_index_offset = offset_for(it->second);
         has_edge_index_offset = true;
      } else {
         // Compatibility fallback for legacy encodings that do not provide edge_tensors schema.
         const std::string_view key_view = key;
         const auto slash = key_view.find(schema_key::kTypeAttrSeparator);
         if(slash != std::string_view::npos) {
            const std::string_view attr = key_view.substr(slash + 1);
            if(attr.rfind(schema_key::kEdgeIndexAttrPrefix, 0) == 0) {
               const std::string_view base = key_view.substr(0, slash);
               const auto first = base.find(schema_key::kEdgeTypeSeparator);
               const auto second = base.find(schema_key::kEdgeTypeSeparator, first + 1);
               if(first == std::string_view::npos or second == std::string_view::npos) {
                  throw std::invalid_argument("Malformed edge key in append_batch_encoding");
               }
               const std::string_view src_type = base.substr(0, first);
               const std::string_view dst_type = base.substr(second + 1);
               const std::string_view part = attr.substr(schema_key::kEdgeIndexAttrPrefix.size());
               if(part.size() == 1 and part.front() == schema_key::kEdgeIndexSrcComponent) {
                  edge_index_offset = offset_for(src_type);
               } else if(part.size() == 1 and part.front() == schema_key::kEdgeIndexDstComponent) {
                  edge_index_offset = offset_for(dst_type);
               } else {
                  throw std::invalid_argument(
                     "Unexpected edge_index part in append_batch_encoding"
                  );
               }
               has_edge_index_offset = true;
            }
         }
      }

      if(has_edge_index_offset) {
         if(not std::holds_alternative< LongCol >(col.data)) {
            throw std::invalid_argument("edge_index column must be int64");
         }
         auto& dest = get_column< int64_t >(key, 1);
         const auto& src = std::get< LongCol >(col.data);
         dest.reserve(dest.size() + src.size());
         for(const auto value : src) {
            dest.push_back(value + edge_index_offset);
         }
         continue;
      }

      std::visit(
         [&]< typename T >(const std::vector< T >& items) {
            auto& dest = get_column< T >(key, col.dim);
            dest.reserve(dest.size() + items.size());
            dest.insert(dest.end(), items.begin(), items.end());
         },
         col.data
      );
   }

   for(auto& [ntype, count] : current_node_counts) {
      auto& offset = node_offsets.try_emplace(ntype, 0).first->second;
      offset += count;

      auto& p = ptrs.try_emplace(ntype, std::vector< int64_t >{}).first->second;
      if(p.empty()) {
         p.emplace_back(0);
      }
      p.emplace_back(offset);

      count = 0;
   }
   current_graph_idx++;
}

void BatchBuilder::load_from_batch_encoding(const BatchEncoding& batch_encoding)
{
   reset();
   columns = batch_encoding.columns;
   node_names = batch_encoding.node_names;
   object_names = batch_encoding.object_names;
   node_feature_dims = batch_encoding.node_feature_dims;
   graph_attrs = batch_encoding.graph_attrs;
   if(batch_encoding.graph_fields.empty()) {
      graph_fields.reset();
   } else {
      graph_fields = std::make_unique< hash_map< std::string, GraphField > >(
         batch_encoding.graph_fields
      );
   }
   ptrs = batch_encoding.ptrs;
   schema_flags = batch_encoding.schema_flags;
   graph_kind = batch_encoding.graph_kind;
   current_graph_idx = batch_encoding.num_graphs;

   current_node_counts.clear();
   node_offsets.clear();
   for(const auto& [node_type, ptr] : ptrs) {
      if(not ptr.empty()) {
         node_offsets[node_type] = ptr.back();
      }
      current_node_counts[node_type] = 0;
   }
   for(const auto& [node_type, count] : batch_encoding.node_counts) {
      if(not node_offsets.contains(node_type)) {
         node_offsets[node_type] = count;
      }
      if(not current_node_counts.contains(node_type)) {
         current_node_counts[node_type] = 0;
      }
   }
}

void BatchBuilder::load_from_batch_encoding(BatchEncoding&& batch_encoding)
{
   reset();
   columns = std::move(batch_encoding.columns);
   node_names = std::move(batch_encoding.node_names);
   object_names = std::move(batch_encoding.object_names);
   node_feature_dims = std::move(batch_encoding.node_feature_dims);
   graph_attrs = std::move(batch_encoding.graph_attrs);
   if(batch_encoding.graph_fields.empty()) {
      graph_fields.reset();
   } else {
      graph_fields = std::make_unique< hash_map< std::string, GraphField > >(
         std::move(batch_encoding.graph_fields)
      );
   }
   ptrs = std::move(batch_encoding.ptrs);
   schema_flags = std::move(batch_encoding.schema_flags);
   graph_kind = std::move(batch_encoding.graph_kind);
   current_graph_idx = batch_encoding.num_graphs;

   current_node_counts.clear();
   node_offsets.clear();
   for(const auto& [node_type, ptr] : ptrs) {
      if(not ptr.empty()) {
         node_offsets[node_type] = ptr.back();
      }
      current_node_counts[node_type] = 0;
   }
   for(const auto& [node_type, count] : batch_encoding.node_counts) {
      if(not node_offsets.contains(node_type)) {
         node_offsets[node_type] = count;
      }
      if(not current_node_counts.contains(node_type)) {
         current_node_counts[node_type] = 0;
      }
   }
}

}  // namespace mifrost
