#include "mifrost/batch_encoding_state.hpp"

#include <absl/container/btree_map.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include <cstdint>
#include <cstring>
#include <iterator>
#include <mimir/search/formatter.hpp>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "mifrost/batch_encoding_graph_field_access.hpp"
#include "mifrost/batch_encoding_graph_field_serialization.hpp"
#include "mifrost/batch_encoding_python_collation.hpp"
#include "mifrost/batch_encoding_schema.hpp"
#include "mifrost/batch_encoding_tensor_cache.hpp"
#include "mifrost/common.hpp"
#include "mifrost/core/encoders/common/target_metadata.hpp"
#include "mifrost/core/nanobind_unordered_dense.hpp"
#include "mifrost/core/nb_instance.hpp"
#include "mifrost/schema_python.hpp"

namespace nb = nanobind;

namespace mifrost {

namespace {

constexpr std::string_view kPythonTensorCacheAttr = "__mifrost_tensor_cache__";

std::vector< std::string > materialize_target_name_states(
   std::span< const mimir::search::State > states
)
{
   std::vector< std::string > names;
   names.reserve(states.size());
   for(const auto& state : states) {
      std::ostringstream stream;
      stream << state;
      names.push_back(stream.str());
   }
   return names;
}

nb::dict batch_encoding_to_state_dict(BatchBuilder::BatchEncoding& encoding, bool include_metadata)
{
   auto map_to_dict =
      []< typename value_t >(const absl::btree_map< std::string, value_t >& values) {
         nb::dict out;
         for(const auto& [key, value] : values) {
            out[key.c_str()] = value;
         }
         return out;
      };

   nb::dict state;
   materialize_batch_encoding_lazy_graph_attrs(encoding);
   state["format_version"] = 1;
   state["graph_kind"] = encoding.graph_kind;
   state["num_graphs"] = encoding.num_graphs;
   state["schema_flags"] = map_to_dict(encoding.schema_flags);
   state["node_feature_dims"] = encoding.node_feature_dims;
   state["graph_attrs"] = encoding.graph_attrs;
   state["graph_fields"] = graph_field_map_to_dict(encoding.graph_fields);
   state["ptrs"] = encoding.ptrs;
   state["node_counts"] = map_to_dict(encoding.node_counts);
   state["schema"] = schema_to_dict(encoding.schema);
   if(include_metadata) {
      state["node_names"] = encoding.node_names;
      state["object_names"] = encoding.object_names;
   } else {
      state["node_names"] = hash_map< std::string, std::vector< std::string > >{};
      state["object_names"] = std::vector< std::string >{};
   }

   nb::dict columns;
   for(const auto& [key, column] : encoding.columns) {
      nb::dict c;
      c["dim"] = column.dim;
      std::visit(
         [&]< typename T >(const std::vector< T >& data) {
            if constexpr(std::is_same_v< T, float >) {
               c["dtype"] = "f32";
            } else {
               c["dtype"] = "i64";
            }
            c["length"] = static_cast< int64_t >(data.size());
            const auto* ptr = reinterpret_cast< const char* >(data.data());
            c["raw"] = nb::bytes(ptr, data.size() * sizeof(T));
         },
         column.data
      );
      columns[key.c_str()] = std::move(c);
   }
   state["columns"] = std::move(columns);
   return state;
}

template < typename value_type >
auto map_from_dict(const nb::dict& source)
{
   absl::btree_map< std::string, value_type > out;
   for(auto [key_obj, value_obj] : source) {
      out.emplace(py::to_std_string(key_obj), nb::cast< value_type >(value_obj));
   }
   return out;
};

}  // namespace

void materialize_batch_encoding_lazy_graph_attrs(BatchBuilder::BatchEncoding& encoding)
{
   if(not encoding.lazy_target_name_strings.empty()) {
      const auto graph_attr_it = encoding.graph_attrs.find(std::string(kTargetNamesAttr));
      if(graph_attr_it == encoding.graph_attrs.end()) {
         encoding.graph_attrs.emplace(
            std::string(kTargetNamesAttr), std::move(encoding.lazy_target_name_strings)
         );
      } else {
         auto* existing = std::get_if< std::vector< std::string > >(&graph_attr_it->second);
         if(existing == nullptr) {
            throw std::invalid_argument(
               "BatchEncoding target_names graph attr must be a string vector"
            );
         }
         existing->insert(
            existing->end(),
            std::make_move_iterator(encoding.lazy_target_name_strings.begin()),
            std::make_move_iterator(encoding.lazy_target_name_strings.end())
         );
         encoding.lazy_target_name_strings.clear();
      }
   }
   if(not encoding.lazy_target_name_states.empty()) {
      auto names = materialize_target_name_states(std::span(encoding.lazy_target_name_states));
      const auto graph_attr_it = encoding.graph_attrs.find(std::string(kTargetNamesAttr));
      if(graph_attr_it == encoding.graph_attrs.end()) {
         encoding.graph_attrs.emplace(std::string(kTargetNamesAttr), std::move(names));
      } else {
         auto* existing = std::get_if< std::vector< std::string > >(&graph_attr_it->second);
         if(existing == nullptr) {
            throw std::invalid_argument(
               "BatchEncoding target_names graph attr must be a string vector"
            );
         }
         existing->insert(
            existing->end(),
            std::make_move_iterator(names.begin()),
            std::make_move_iterator(names.end())
         );
      }
      encoding.lazy_target_name_states.clear();
   }
}

BatchBuilder::BatchEncoding batch_encoding_from_state_dict(const nb::dict& state)
{
   const int version = nb::cast< int >(state["format_version"]);
   if(version != 1) {
      throw std::invalid_argument("Unsupported BatchEncoding format version");
   }

   BatchBuilder::BatchEncoding encoding;
   try {
      encoding.graph_kind = py::to_std_string(state["graph_kind"]);
   } catch(const std::exception& ex) {
      throw std::invalid_argument("Failed to parse state['graph_kind']: " + std::string(ex.what()));
   }
   try {
      encoding.num_graphs = nb::cast< int64_t >(state["num_graphs"]);
   } catch(const std::exception& ex) {
      throw std::invalid_argument("Failed to parse state['num_graphs']: " + std::string(ex.what()));
   }
   try {
      nb::dict schema_flags = nb::cast< nb::dict >(state["schema_flags"]);
      encoding.schema_flags = map_from_dict< bool >(schema_flags);
   } catch(const std::exception& ex) {
      throw std::invalid_argument(
         "Failed to parse state['schema_flags']: " + std::string(ex.what())
      );
   }
   {
      nb::dict node_feature_dims;
      try {
         node_feature_dims = nb::cast< nb::dict >(state["node_feature_dims"]);
      } catch(const std::exception& ex) {
         throw std::invalid_argument(
            "Failed to parse state['node_feature_dims']: " + std::string(ex.what())
         );
      }
      encoding.node_feature_dims.clear();
      encoding.node_feature_dims.reserve(node_feature_dims.size());
      for(auto [key_obj, value_obj] : node_feature_dims) {
         try {
            encoding.node_feature_dims.emplace(
               py::to_std_string(key_obj), nb::cast< int >(value_obj)
            );
         } catch(const std::exception& ex) {
            throw std::invalid_argument(
               "Failed to parse state['node_feature_dims'] entry: " + std::string(ex.what())
            );
         }
      }
   }
   {
      nb::dict graph_attrs;
      try {
         graph_attrs = nb::cast< nb::dict >(state["graph_attrs"]);
      } catch(const std::exception& ex) {
         throw std::invalid_argument(
            "Failed to parse state['graph_attrs']: " + std::string(ex.what())
         );
      }
      encoding.graph_attrs.clear();
      encoding.graph_attrs.reserve(graph_attrs.size());
      for(auto [key_obj, value_obj] : graph_attrs) {
         try {
            encoding.graph_attrs.emplace(
               py::to_std_string(key_obj), nb::cast< BatchBuilder::GraphAttrValue >(value_obj)
            );
         } catch(const std::exception& ex) {
            throw std::invalid_argument(
               "Failed to parse state['graph_attrs'] entry: " + std::string(ex.what())
            );
         }
      }
   }
   if(state.contains("graph_fields")) {
      try {
         encoding.graph_fields = graph_field_map_from_dict(
            nb::cast< nb::dict >(state["graph_fields"])
         );
      } catch(const std::exception& ex) {
         throw std::invalid_argument(
            "Failed to parse state['graph_fields']: " + std::string(ex.what())
         );
      }
   }
   {
      nb::dict ptrs;
      try {
         ptrs = nb::cast< nb::dict >(state["ptrs"]);
      } catch(const std::exception& ex) {
         throw std::invalid_argument("Failed to parse state['ptrs']: " + std::string(ex.what()));
      }
      encoding.ptrs.clear();
      encoding.ptrs.reserve(ptrs.size());
      for(auto [key_obj, value_obj] : ptrs) {
         try {
            encoding.ptrs.emplace(
               py::to_std_string(key_obj), nb::cast< std::vector< int64_t > >(value_obj)
            );
         } catch(const std::exception& ex) {
            throw std::invalid_argument(
               "Failed to parse state['ptrs'] entry: " + std::string(ex.what())
            );
         }
      }
   }
   {
      try {
         nb::dict node_counts = nb::cast< nb::dict >(state["node_counts"]);
         encoding.node_counts = map_from_dict< int64_t >(node_counts);
      } catch(const std::exception& ex) {
         throw std::invalid_argument(
            "Failed to parse state['node_counts']: " + std::string(ex.what())
         );
      }
   }
   {
      try {
         nb::dict schema = nb::cast< nb::dict >(state["schema"]);
         encoding.schema = schema_from_dict(schema);
      } catch(const std::exception& ex) {
         throw std::invalid_argument("Failed to parse state['schema']: " + std::string(ex.what()));
      }
   }
   {
      nb::dict node_names;
      try {
         node_names = nb::cast< nb::dict >(state["node_names"]);
      } catch(const std::exception& ex) {
         throw std::invalid_argument(
            "Failed to parse state['node_names']: " + std::string(ex.what())
         );
      }
      encoding.node_names.clear();
      encoding.node_names.reserve(node_names.size());
      for(auto [key_obj, value_obj] : node_names) {
         try {
            encoding.node_names.emplace(
               py::to_std_string(key_obj), nb::cast< std::vector< std::string > >(value_obj)
            );
         } catch(const std::exception& ex) {
            throw std::invalid_argument(
               "Failed to parse state['node_names'] entry: " + std::string(ex.what())
            );
         }
      }
   }
   try {
      encoding.object_names = nb::cast< std::vector< std::string > >(state["object_names"]);
   } catch(const std::exception& ex) {
      throw std::invalid_argument(
         "Failed to parse state['object_names']: " + std::string(ex.what())
      );
   }

   nb::dict columns;
   try {
      columns = nb::cast< nb::dict >(state["columns"]);
   } catch(const std::exception& ex) {
      throw std::invalid_argument("Failed to parse state['columns']: " + std::string(ex.what()));
   }
   for(auto [key_obj, col_obj] : columns) {
      auto col = nb::cast< nb::dict >(col_obj);
      const auto key = py::to_std_string(key_obj);
      const auto dim = nb::cast< int >(col["dim"]);
      const auto dtype = py::to_std_string(col["dtype"]);
      const auto length = nb::cast< int64_t >(col["length"]);
      const auto raw_bytes = nb::cast< nb::bytes >(col["raw"]);
      const std::string_view raw(raw_bytes.c_str(), raw_bytes.size());

      if(dtype == "f32") {
         if(raw.size() != length * sizeof(float)) {
            throw std::invalid_argument("Malformed f32 column payload");
         }
         std::vector< float > data(length);
         if(length > 0) {
            std::memcpy(data.data(), raw.data(), raw.size());
         }
         encoding.columns[key] = BatchBuilder::Column{std::move(data), dim};
      } else if(dtype == "i64") {
         if(raw.size() != length * sizeof(int64_t)) {
            throw std::invalid_argument("Malformed i64 column payload");
         }
         std::vector< int64_t > data(length);
         if(length > 0) {
            std::memcpy(data.data(), raw.data(), raw.size());
         }
         encoding.columns[key] = BatchBuilder::Column{std::move(data), dim};
      } else {
         throw std::invalid_argument("Unsupported BatchEncoding column dtype");
      }
   }

   return encoding;
}

nb::dict batch_encoding_state_from_instance(nb::handle self, bool include_metadata)
{
   auto* encoding = require_instance_ptr< BatchBuilder::BatchEncoding >(
      self, "BatchEncoding state extraction called with invalid instance"
   );
   validate_batch_encoding_graph_fields(*encoding, "BatchEncoding state extraction");
   nb::dict state = batch_encoding_to_state_dict(*encoding, include_metadata);
   nb::dict py_attrs = batch_encoding_python_attrs_copy(self);
   if(py_attrs.contains(kPythonTensorCacheAttr.data())) {
      py_attrs.attr("pop")(kPythonTensorCacheAttr.data());
   }
   if(nb::len(py_attrs) > 0) {
      state["python_attrs"] = std::move(py_attrs);
   }
   return state;
}

nb::object batch_encoding_object_from_state(const nb::dict& state)
{
   nb::object obj = py::mifrost_core_batch_encoding_cls()();
   auto* encoding = require_instance_ptr< BatchBuilder::BatchEncoding >(
      obj, "Failed to instantiate BatchEncoding during state load"
   );
   *encoding = batch_encoding_from_state_dict(state);
   auto attrs = nb::cast< nb::dict >(obj.attr("__dict__"));
   attrs.clear();
   batch_encoding_apply_python_attrs_from_state(obj, state, attrs);
   clear_owner_tensor_cache(obj);
   return obj;
}

void batch_encoding_save(nb::handle self, const std::string& path, bool include_metadata)
{
   nb::object file = py::builtins_open()(path, "wb");
   nb::bytes payload = batch_encoding_dumps(self, include_metadata);
   file.attr("write")(payload);
   file.attr("close")();
}

nb::object batch_encoding_load(const std::string& path)
{
   nb::object file = py::builtins_open()(path, "rb");
   nb::bytes payload = nb::cast< nb::bytes >(file.attr("read")());
   file.attr("close")();
   return batch_encoding_loads(payload);
}

nb::bytes batch_encoding_dumps(nb::handle self, bool include_metadata)
{
   nb::dict state = batch_encoding_state_from_instance(self, include_metadata);
   return nb::cast< nb::bytes >(py::pickle_dumps()(state, 5));
}

nb::object batch_encoding_loads(nb::bytes payload)
{
   nb::dict state = nb::cast< nb::dict >(py::pickle_loads()(payload));
   return batch_encoding_object_from_state(state);
}

nb::dict batch_encoding_getstate(nb::handle self)
{
   return batch_encoding_state_from_instance(self, true);
}

nb::tuple batch_encoding_reduce(nb::handle self)
{
   nb::bytes payload = batch_encoding_dumps(self, true);
   return nb::make_tuple(py::mifrost_batch_encoding_loader(), nb::make_tuple(std::move(payload)));
}

void batch_encoding_setstate(nb::handle self, const nb::dict& state)
{
   auto* encoding = require_instance_ptr< BatchBuilder::BatchEncoding >(
      self, "BatchEncoding.__setstate__ called with invalid instance"
   );
   *encoding = batch_encoding_from_state_dict(state);
   batch_encoding_clear_python_attrs(self);
   batch_encoding_apply_python_attrs_from_state(self, state);
   clear_owner_tensor_cache(self);
}

}  // namespace mifrost
