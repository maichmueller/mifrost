#include <nanobind/make_iterator.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/set.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>
#include <nanobind/trampoline.h>

#include <cstdint>
#include <cstring>
#include <mimir/formalism/problem.hpp>
#include <mimir/search/axiom_evaluators/grounded/grounded.hpp>
#include <mimir/search/axiom_evaluators/interface.hpp>
#include <mimir/search/grounders/lifted.hpp>
#include <mimir/search/state_repository.hpp>
#include <optional>
#include <type_traits>

#include "mifrost/binding_kwargs.hpp"
#include "mifrost/bindings.hpp"
#include "mifrost/core/batch_builder.hpp"
#include "mifrost/core/default_relations.hpp"
#include "mifrost/core/goal_inputs.hpp"
#include "mifrost/core/hgraph_stream_encoder.hpp"
#include "mifrost/core/horizon_hgraph_encoder.hpp"
#include "mifrost/core/nanobind_unordered_dense.hpp"
#include "mifrost/core/successor_hgraph_encoder.hpp"
#include "mifrost/core/transition_dag.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace mifrost {

namespace {

void apply_hgraph_config_kwargs(HGraphEncoderEngine::Config& config, const nb::kwargs& kwargs)
{
   apply_config_kwargs(config, kwargs, "HGraphEncoderConfig");
}

template < typename T >
void vector_owner_deleter(void* p) noexcept
{
   delete static_cast< std::vector< T >* >(p);
}

template < typename T >
auto vector_to_1d_ndarray_view(std::vector< T >& vec, nb::handle owner)
{
   size_t shape[1] = {vec.size()};
   return nb::ndarray< nb::numpy, T, nb::shape< -1 > >(vec.data(), 1, shape, owner);
}

template < typename T >
auto vector_to_2d_ndarray_view(std::vector< T >& vec, size_t rows, size_t cols, nb::handle owner)
{
   size_t shape[2] = {rows, cols};
   return nb::ndarray< nb::numpy, T, nb::shape< -1, -1 > >(vec.data(), 2, shape, owner);
}

template < typename T >
auto vector_to_1d_ndarray_owned(std::vector< T >&& vec)
{
   auto* heap_vec = new std::vector< T >(std::move(vec));
   heap_vec->shrink_to_fit();
   size_t shape[1] = {heap_vec->size()};
   nb::capsule owner(heap_vec, vector_owner_deleter< T >);
   return nb::ndarray< nb::numpy, T, nb::shape< -1 > >(heap_vec->data(), 1, shape, owner);
}

std::vector< int64_t > ptr_to_batch(const std::vector< int64_t >& ptr)
{
   std::vector< int64_t > batch;
   if(ptr.size() < 2) {
      return batch;
   }
   batch.reserve(static_cast< size_t >(std::max< int64_t >(0, ptr.back())));
   for(size_t idx = 0; idx + 1 < ptr.size(); ++idx) {
      const int64_t count = std::max< int64_t >(0, ptr[idx + 1] - ptr[idx]);
      batch.insert(batch.end(), static_cast< size_t >(count), static_cast< int64_t >(idx));
   }
   return batch;
}

nb::object flatten_single_graph_metadata_list(nb::handle value)
{
   if(! nb::isinstance< nb::list >(value)) {
      return nb::borrow< nb::object >(value);
   }
   nb::list outer = nb::borrow< nb::list >(value);
   if(nb::len(outer) == 1 && nb::isinstance< nb::list >(outer[0])) {
      return nb::borrow< nb::object >(outer[0]);
   }
   return nb::borrow< nb::object >(value);
}

void copy_store_attrs_without_batch(nb::object& dst_store, nb::object& src_store)
{
   for(auto key_obj : src_store.attr("keys")()) {
      const std::string key = nb::cast< std::string >(key_obj);
      if(key == "ptr" || key == "batch") {
         continue;
      }
      dst_store.attr("__setitem__")(key_obj, src_store.attr("__getitem__")(key_obj));
   }
}

void copy_global_attrs_for_single(nb::object& dst, nb::object& src)
{
   nb::object global_store = src.attr("_global_store");
   for(auto key_obj : global_store.attr("keys")()) {
      const std::string key = nb::cast< std::string >(key_obj);
      if(key == "_num_graphs") {
         continue;
      }
      nb::object value = global_store.attr("__getitem__")(key_obj);
      if(key == "object_names") {
         value = flatten_single_graph_metadata_list(value);
      }
      dst.attr(key.c_str()) = value;
   }
}

nb::object batch_to_single_hetero_data(nb::object& pyg_batch)
{
   nb::object tg_data = nb::module_::import_("torch_geometric.data");
   nb::object out = tg_data.attr("HeteroData")();

   for(auto node_type_obj : pyg_batch.attr("node_types")) {
      std::string node_type = nb::cast< std::string >(node_type_obj);
      nb::object src_store = pyg_batch.attr("__getitem__")(node_type);
      nb::object dst_store = out.attr("__getitem__")(node_type);
      copy_store_attrs_without_batch(dst_store, src_store);
      if(nb::cast< bool >(src_store.attr("__contains__")("node_names"))) {
         nb::object flat_names = flatten_single_graph_metadata_list(
            src_store.attr("__getitem__")("node_names")
         );
         dst_store.attr("node_names") = flat_names;
         dst_store.attr("num_nodes") = nb::len(flat_names);
      }
   }

   for(auto edge_type_obj : pyg_batch.attr("edge_types")) {
      nb::object src_store = pyg_batch.attr("__getitem__")(edge_type_obj);
      nb::object dst_store = out.attr("__getitem__")(edge_type_obj);
      for(auto key_obj : src_store.attr("keys")()) {
         dst_store.attr("__setitem__")(key_obj, src_store.attr("__getitem__")(key_obj));
      }
   }

   copy_global_attrs_for_single(out, pyg_batch);
   return out;
}

nb::object batch_to_single_homo_data(nb::object& pyg_batch)
{
   nb::object tg_data = nb::module_::import_("torch_geometric.data");
   nb::object out = tg_data.attr("Data")();

   nb::list node_types = nb::cast< nb::list >(pyg_batch.attr("node_types"));
   if(nb::len(node_types) > 1) {
      throw std::invalid_argument(
         "BatchEncoding.as_pyg(as_batch=False) for homo expects a single node type"
      );
   }

   if(nb::len(node_types) == 1) {
      std::string node_type = nb::cast< std::string >(node_types[0]);
      nb::object src_store = pyg_batch.attr("__getitem__")(node_type);
      for(auto key_obj : src_store.attr("keys")()) {
         const std::string key = nb::cast< std::string >(key_obj);
         if(key == "ptr" || key == "batch") {
            continue;
         }
         out.attr("__setitem__")(key_obj, src_store.attr("__getitem__")(key_obj));
      }
      if(nb::cast< bool >(src_store.attr("__contains__")("node_names"))) {
         nb::object flat_names = flatten_single_graph_metadata_list(
            src_store.attr("__getitem__")("node_names")
         );
         out.attr("node_names") = flat_names;
      }
   }

   nb::list edge_types = nb::cast< nb::list >(pyg_batch.attr("edge_types"));
   if(nb::len(edge_types) > 1) {
      throw std::invalid_argument(
         "BatchEncoding.as_pyg(as_batch=False) for homo expects a single edge type"
      );
   }
   if(nb::len(edge_types) == 1) {
      nb::object src_store = pyg_batch.attr("__getitem__")(edge_types[0]);
      for(auto key_obj : src_store.attr("keys")()) {
         out.attr("__setitem__")(key_obj, src_store.attr("__getitem__")(key_obj));
      }
   }

   copy_global_attrs_for_single(out, pyg_batch);
   return out;
}

nb::dict
batch_encoding_to_state_dict(const BatchBuilder::BatchEncoding& encoding, bool include_metadata)
{
   nb::dict state;
   state["format_version"] = 1;
   state["graph_kind"] = encoding.graph_kind;
   state["num_graphs"] = encoding.num_graphs;
   state["schema_flags"] = encoding.schema_flags;
   state["node_feature_dims"] = encoding.node_feature_dims;
   state["graph_attrs"] = encoding.graph_attrs;
   state["ptrs"] = encoding.ptrs;
   state["node_counts"] = encoding.node_counts;
   state["schema"] = encoding.schema.to_dict();
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
         [&](const auto& data) {
            using T = std::decay_t< decltype(data) >::value_type;
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

BatchBuilder::BatchEncoding batch_encoding_from_state_dict(const nb::dict& state)
{
   const int version = nb::cast< int >(state["format_version"]);
   if(version != 1) {
      throw std::invalid_argument("Unsupported BatchEncoding format version");
   }

   BatchBuilder::BatchEncoding encoding;
   encoding.graph_kind = nb::cast< std::string >(state["graph_kind"]);
   encoding.num_graphs = nb::cast< int64_t >(state["num_graphs"]);
   encoding.schema_flags = nb::cast< absl::btree_map< std::string, bool > >(state["schema_flags"]);
   encoding.node_feature_dims = nb::cast< hash_map< std::string, int > >(
      state["node_feature_dims"]
   );
   encoding.graph_attrs = nb::cast< hash_map< std::string, BatchBuilder::GraphAttrValue > >(
      state["graph_attrs"]
   );
   encoding.ptrs = nb::cast< hash_map< std::string, std::vector< int64_t > > >(state["ptrs"]);
   encoding.node_counts = nb::cast< absl::btree_map< std::string, int64_t > >(state["node_counts"]);
   encoding.schema = Schema::from_dict(nb::cast< nb::dict >(state["schema"]));
   encoding.node_names = nb::cast< hash_map< std::string, std::vector< std::string > > >(
      state["node_names"]
   );
   encoding.object_names = nb::cast< std::vector< std::string > >(state["object_names"]);

   nb::dict columns = nb::cast< nb::dict >(state["columns"]);
   for(auto [key_obj, col_obj] : columns) {
      const std::string key = nb::cast< std::string >(key_obj);
      nb::dict col = nb::cast< nb::dict >(col_obj);
      const int dim = nb::cast< int >(col["dim"]);
      const std::string dtype = nb::cast< std::string >(col["dtype"]);
      const size_t length = static_cast< size_t >(nb::cast< int64_t >(col["length"]));
      const std::string raw = nb::cast< std::string >(col["raw"]);

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

uint64_t schema_fingerprint(const BatchBuilder::BatchEncoding& encoding)
{
   nb::dict payload;
   payload["graph_kind"] = encoding.graph_kind;
   payload["schema_flags"] = encoding.schema_flags;
   payload["node_feature_dims"] = encoding.node_feature_dims;
   payload["schema"] = encoding.schema.to_dict();

   nb::object json = nb::module_::import_("json");
   const std::string text = nb::cast< std::string >(
      json.attr("dumps")(payload, "sort_keys"_a = true, "separators"_a = nb::make_tuple(",", ":"))
   );

   constexpr uint64_t kOffset = 1469598103934665603ULL;
   constexpr uint64_t kPrime = 1099511628211ULL;
   uint64_t h = kOffset;
   for(unsigned char c : text) {
      h ^= static_cast< uint64_t >(c);
      h *= kPrime;
   }
   return h;
}

nb::dict batch_encoding_as_dict(BatchBuilder::BatchEncoding& encoding, nb::handle owner)
{
   nb::dict tensors;

   for(auto& [key, col] : encoding.columns) {
      const bool is_edge_index = key.find("/edge_index_") != std::string::npos;
      std::visit(
         [&]< typename T >(std::vector< T >& data) {
            if(is_edge_index) {
               tensors[key.c_str()] = vector_to_1d_ndarray_view(data, owner);
               return;
            }
            const size_t rows = col.dim > 0 ? data.size() / static_cast< size_t >(col.dim) : 0;
            tensors[key.c_str()] = vector_to_2d_ndarray_view(
               data, rows, static_cast< size_t >(col.dim), owner
            );
         },
         col.data
      );
   }

   bool exported_ptr = false;
   for(auto& [node_type, ptr] : encoding.ptrs) {
      if(ptr.size() < 2) {
         continue;
      }
      exported_ptr = true;
      tensors[(node_type + "/ptr").c_str()] = vector_to_1d_ndarray_view(ptr, owner);
      tensors[(node_type + "/batch").c_str()] = vector_to_1d_ndarray_owned(ptr_to_batch(ptr));
   }
   if(! exported_ptr) {
      for(const auto& [node_type, count] : encoding.node_counts) {
         if(count <= 0) {
            continue;
         }
         std::vector< int64_t > ptr{0, count};
         tensors[(node_type + "/ptr").c_str()] = vector_to_1d_ndarray_owned(std::move(ptr));
         tensors[(node_type + "/batch").c_str()] = vector_to_1d_ndarray_owned(
            std::vector< int64_t >(count, 0)
         );
      }
   }

   nb::dict out;
   out["tensors"] = std::move(tensors);
   out["schema"] = encoding.schema.to_dict();

   nb::dict node_names_dict;
   for(const auto& [node_type, names] : encoding.node_names) {
      node_names_dict[node_type.c_str()] = nb::cast(names);
   }
   out["node_names"] = std::move(node_names_dict);

   nb::dict dims_dict;
   for(const auto& [node_type, dim] : encoding.node_feature_dims) {
      dims_dict[node_type.c_str()] = dim;
   }
   out["node_feature_dims"] = std::move(dims_dict);
   out["object_names"] = nb::cast(encoding.object_names);
   out["num_graphs"] = encoding.num_graphs;

   if(! encoding.graph_attrs.empty()) {
      nb::dict graph_attrs_dict;
      for(const auto& [key, value] : encoding.graph_attrs) {
         std::visit([&](const auto& v) { graph_attrs_dict[key.c_str()] = nb::cast(v); }, value);
      }
      out["graph_attrs"] = std::move(graph_attrs_dict);
   }

   return out;
}

nb::object
batch_encoding_as_pyg(BatchBuilder::BatchEncoding& encoding, std::optional< bool > as_batch)
{
   const bool want_batch = as_batch.value_or(encoding.num_graphs != 1);
   BatchBuilder builder;
   builder.set_graph_kind(encoding.graph_kind);
   builder.load_from_batch_encoding(encoding);
   nb::object pyg_batch = builder.build_pyg();

   if(! want_batch && encoding.num_graphs != 1) {
      throw std::invalid_argument("BatchEncoding.as_pyg(as_batch=False) requires num_graphs == 1");
   }

   if(! want_batch) {
      if(encoding.graph_kind == "homo") {
         return batch_to_single_homo_data(pyg_batch);
      }
      return batch_to_single_hetero_data(pyg_batch);
   }

   return pyg_batch;
}

void save_batch_encoding(
   const BatchBuilder::BatchEncoding& encoding,
   const std::string& path,
   bool include_metadata
)
{
   nb::object builtins = nb::module_::import_("builtins");
   nb::object pickle = nb::module_::import_("pickle");
   nb::object file = builtins.attr("open")(path, "wb");
   pickle.attr("dump")(batch_encoding_to_state_dict(encoding, include_metadata), file, 5);
   file.attr("close")();
}

BatchBuilder::BatchEncoding load_batch_encoding(const std::string& path)
{
   nb::object builtins = nb::module_::import_("builtins");
   nb::object pickle = nb::module_::import_("pickle");
   nb::object file = builtins.attr("open")(path, "rb");
   nb::dict state = nb::cast< nb::dict >(pickle.attr("load")(file));
   file.attr("close")();
   return batch_encoding_from_state_dict(state);
}

}  // namespace

void init_hgraph_encoder(nb::module_& m)
{
   nb::class_< BatchBuilder >(m, "BatchBuilder")
      .def(nb::init<>())
      .def(
         "add_node_features",
         [](BatchBuilder& builder,
            const std::string& node_type,
            const std::string& attr_name,
            nb::ndarray< nb::numpy, float > data) {
            if(data.ndim() != 1 and data.ndim() != 2) {
               throw std::invalid_argument("add_node_features expects a 1D/2D array");
            }
            const int feature_dim = data.ndim() == 2 ? static_cast< int >(data.shape(1)) : 1;
            const auto count = static_cast< size_t >(data.size());
            builder.add_node_features(
               node_type, attr_name, std::span< const float >(data.data(), count), feature_dim
            );
         }
      )
      .def(
         "add_edges",
         [](BatchBuilder& builder,
            const std::string& src_type,
            const std::string& rel_type,
            const std::string& dst_type,
            nb::ndarray< nb::numpy, int64_t > src,
            nb::ndarray< nb::numpy, int64_t > dst) {
            if(src.ndim() != 1 || dst.ndim() != 1) {
               throw std::invalid_argument("add_edges expects 1D arrays for src/dst indices");
            }
            if(src.size() != dst.size()) {
               throw std::invalid_argument("add_edges expects src/dst arrays of equal length");
            }
            builder.add_edges(
               src_type,
               rel_type,
               dst_type,
               std::span< const int64_t >(src.data(), src.size()),
               std::span< const int64_t >(dst.data(), dst.size())
            );
         }
      )
      .def(
         "add_edge_features",
         [](BatchBuilder& builder,
            const std::string& src_type,
            const std::string& rel_type,
            const std::string& dst_type,
            const std::string& attr_name,
            nb::ndarray< nb::numpy, float > data) {
            if(data.ndim() != 1 and data.ndim() != 2) {
               throw std::invalid_argument("add_edge_features expects a 1D/2D array");
            }
            const int feature_dim = data.ndim() == 2 ? static_cast< int >(data.shape(1)) : 1;
            const auto count = static_cast< size_t >(data.size());
            builder.add_edge_features(
               src_type,
               rel_type,
               dst_type,
               attr_name,
               std::span< const float >(data.data(), count),
               feature_dim
            );
         }
      )
      .def("add_nodes", &BatchBuilder::add_nodes)
      .def("add_edge", &BatchBuilder::add_edge)
      .def("set_node_names", &BatchBuilder::set_node_names)
      .def("set_object_names", &BatchBuilder::set_object_names)
      .def("build", &BatchBuilder::build)
      .def("build_pyg", &BatchBuilder::build_pyg)
      .def("append_batch_encoding", &BatchBuilder::append_batch_encoding)
      .def(
         "load_from_batch_encoding",
         nb::overload_cast< const BatchBuilder::BatchEncoding& >(
            &BatchBuilder::load_from_batch_encoding
         )
      )
      .def("next_graph", &BatchBuilder::next_graph)
      .def("set_graph_kind", &BatchBuilder::set_graph_kind, "kind"_a)
      .def("set_schema_flag", &BatchBuilder::set_schema_flag, "key"_a, "value"_a);

   nb::class_< BatchBuilder::BatchEncoding >(m, "BatchEncoding")
      .def(nb::init<>())
      .def_ro("num_graphs", &BatchBuilder::BatchEncoding::num_graphs)
      .def_ro("graph_kind", &BatchBuilder::BatchEncoding::graph_kind)
      .def_ro("schema", &BatchBuilder::BatchEncoding::schema)
      .def_ro("schema_flags", &BatchBuilder::BatchEncoding::schema_flags)
      .def_ro("node_feature_dims", &BatchBuilder::BatchEncoding::node_feature_dims)
      .def_ro("graph_attrs", &BatchBuilder::BatchEncoding::graph_attrs)
      .def(
         "as_dict",
         [](nb::handle self) {
            auto* encoding = nb::inst_ptr< BatchBuilder::BatchEncoding >(self);
            if(encoding == nullptr) {
               throw std::invalid_argument("BatchEncoding.as_dict called with invalid instance");
            }
            return batch_encoding_as_dict(*encoding, self);
         }
      )
      .def(
         "as_pyg",
         [](nb::handle self, std::optional< bool > as_batch) {
            auto* encoding = nb::inst_ptr< BatchBuilder::BatchEncoding >(self);
            if(encoding == nullptr) {
               throw std::invalid_argument("BatchEncoding.as_pyg called with invalid instance");
            }
            return batch_encoding_as_pyg(*encoding, as_batch);
         },
         "as_batch"_a = nb::none()
      )
      .def("schema_fingerprint", &schema_fingerprint)
      .def(
         "save",
         [](const BatchBuilder::BatchEncoding& encoding,
            const std::string& path,
            bool include_metadata) { save_batch_encoding(encoding, path, include_metadata); },
         "path"_a,
         "include_metadata"_a = false
      )
      .def_static("load", [](const std::string& path) { return load_batch_encoding(path); });

   m.def(
      "batch_encodings",
      [](std::vector< BatchBuilder::BatchEncoding > encodings) {
         if(encodings.empty()) {
            return BatchBuilder::BatchEncoding{};
         }
         const auto expected_fp = schema_fingerprint(encodings.front());
         BatchBuilder builder;
         builder.set_graph_kind(encodings.front().graph_kind);
         for(const auto& encoding : encodings) {
            if(encoding.num_graphs != 1) {
               throw std::invalid_argument("batch_encodings expects inputs with num_graphs == 1");
            }
            if(schema_fingerprint(encoding) != expected_fp) {
               throw std::invalid_argument("batch_encodings schema_fingerprint mismatch");
            }
            builder.append_batch_encoding(encoding);
         }
         return builder.build();
      },
      "encodings"_a
   );

   nb::class_< HGraphEncoderEngine::Config >(m, "HGraphEncoderConfig")
      .def(nb::init<>())
      .def(
         "__init__",
         [](HGraphEncoderEngine::Config* self, const nb::kwargs& kwargs) {
            new(self) HGraphEncoderEngine::Config();
            apply_hgraph_config_kwargs(*self, kwargs);
         }
      )
      .def_rw("symbol_type_id", &HGraphEncoderEngine::Config::symbol_type_id)
      .def_rw("nullary_object_name", &HGraphEncoderEngine::Config::nullary_object_name)
      .def_rw("max_goal_level", &HGraphEncoderEngine::Config::max_goal_level)
      .def_rw("support_literals", &HGraphEncoderEngine::Config::support_literals)
      .def_rw(
         "goal_satisfaction_derivations",
         &HGraphEncoderEngine::Config::goal_satisfaction_derivations
      )
      .def_rw("add_nullary_predicates", &HGraphEncoderEngine::Config::add_nullary_predicates)
      .def_rw("ignore_actions", &HGraphEncoderEngine::Config::ignore_actions)
      .def_rw("include_static", &HGraphEncoderEngine::Config::include_static)
      .def_rw("include_lgan_edges", &HGraphEncoderEngine::Config::include_lgan_edges)
      .def_rw("include_empty_edge_types", &HGraphEncoderEngine::Config::include_empty_edge_types)
      .def_rw("export_node_names", &HGraphEncoderEngine::Config::export_node_names)
      .def_rw("history_link_relation", &HGraphEncoderEngine::Config::history_link_relation)
      .def_rw(
         "lgan_nn_edge_pos",
         &HGraphEncoderEngine::Config::lgan_nn_edge_pos,
         "lgan_nn_edge_pos"_a = defaults::lgan_nn_edge_pos
      );

   nb::class_< HGraphEncoderEngine >(m, "HGraphEncoderEngine")
      .def(nb::init< const mimir::formalism::DomainImpl& >())
      .def(nb::init< const mimir::formalism::DomainImpl&, HGraphEncoderEngine::Config >())
      .def(nb::init< mimir::formalism::Domain >())
      .def(nb::init< mimir::formalism::Domain, HGraphEncoderEngine::Config >())
      .def_prop_ro("config", &HGraphEncoderEngine::get_config, nb::rv_policy::reference_internal)
      .def(
         "encode",
         [](HGraphEncoderEngine& encoder, const mimir::search::State& state) {
            BatchBuilder builder;
            builder.set_graph_kind("hetero");
            encoder.encode(state, builder);
            return builder.build();
         },
         "state"_a
      )
      .def(
         "encode",
         [](HGraphEncoderEngine& encoder,
            const mimir::search::State& state,
            const GoalInputs& goals,
            const std::vector< mimir::formalism::GroundAction >& actions) {
            BatchBuilder builder;
            builder.set_graph_kind("hetero");
            encoder.encode(state, goals, actions, builder);
            return builder.build();
         },
         "state"_a,
         "goals"_a,
         "actions"_a
      )
      .def(
         "encode",
         [](HGraphEncoderEngine& encoder,
            const mimir::search::State& state,
            const GoalInputs& goals,
            const std::vector< mimir::formalism::GroundAction >& actions,
            const std::vector< HGraphEncoderEngine::HistorySubgoal >& history_subgoals,
            std::optional< int > history_max_steps) {
            BatchBuilder builder;
            builder.set_graph_kind("hetero");
            encoder.encode(state, goals, actions, history_subgoals, history_max_steps, builder);
            return builder.build();
         },
         "state"_a,
         "goals"_a,
         "actions"_a,
         "history_subgoals"_a,
         "history_max_steps"_a = std::nullopt
      )
      .def(
         "encode",
         [](HGraphEncoderEngine& encoder,
            const mimir::search::State& state,
            BatchBuilder& builder) { encoder.encode(state, builder); },
         "state"_a,
         "builder"_a
      )
      .def(
         "encode",
         [](HGraphEncoderEngine& encoder,
            const mimir::search::State& state,
            const GoalInputs& goals,
            const std::vector< mimir::formalism::GroundAction >& actions,
            BatchBuilder& builder) { encoder.encode(state, goals, actions, builder); },
         "state"_a,
         "goals"_a,
         "actions"_a,
         "builder"_a
      )
      .def(
         "encode",
         [](HGraphEncoderEngine& encoder,
            const mimir::search::State& state,
            const GoalInputs& goals,
            const std::vector< mimir::formalism::GroundAction >& actions,
            const std::vector< HGraphEncoderEngine::HistorySubgoal >& history_subgoals,
            std::optional< int > history_max_steps,
            BatchBuilder& builder) {
            encoder.encode(state, goals, actions, history_subgoals, history_max_steps, builder);
         },
         "state"_a,
         "goals"_a,
         "actions"_a,
         "history_subgoals"_a,
         "history_max_steps"_a = std::nullopt,
         "builder"_a
      );

   nb::class_< HGraphMutableStreamEncoder >(m, "HGraphMutableStreamEncoder")
      .def(nb::init< HGraphEncoderEngine& >(), nb::keep_alive< 1, 2 >())
      .def(
         "append",
         nb::overload_cast< const mimir::search::State& >(&HGraphMutableStreamEncoder::append),
         "state"_a,
         nb::call_guard< nb::gil_scoped_release >()
      )
      .def(
         "append",
         nb::overload_cast<
            const mimir::search::State&,
            const GoalInputs&,
            const std::vector< mimir::formalism::GroundAction >& >(
            &HGraphMutableStreamEncoder::append
         ),
         "state"_a,
         "goals"_a,
         "actions"_a,
         nb::call_guard< nb::gil_scoped_release >()
      )
      .def(
         "append",
         nb::overload_cast<
            const mimir::search::State&,
            const GoalInputs&,
            const std::vector< mimir::formalism::GroundAction >&,
            const std::vector< HGraphEncoderEngine::HistorySubgoal >&,
            std::optional< int > >(&HGraphMutableStreamEncoder::append),
         "state"_a,
         "goals"_a,
         "actions"_a,
         "history_subgoals"_a,
         "history_max_steps"_a = std::nullopt,
         nb::call_guard< nb::gil_scoped_release >()
      )
      .def(
         "update",
         nb::overload_cast< int64_t, const mimir::search::State& >(
            &HGraphMutableStreamEncoder::update
         ),
         "id"_a,
         "state"_a,
         nb::call_guard< nb::gil_scoped_release >()
      )
      .def(
         "update",
         nb::overload_cast<
            int64_t,
            const mimir::search::State&,
            const GoalInputs&,
            const std::vector< mimir::formalism::GroundAction >& >(
            &HGraphMutableStreamEncoder::update
         ),
         "id"_a,
         "state"_a,
         "goals"_a,
         "actions"_a,
         nb::call_guard< nb::gil_scoped_release >()
      )
      .def(
         "update",
         nb::overload_cast<
            int64_t,
            const mimir::search::State&,
            const GoalInputs&,
            const std::vector< mimir::formalism::GroundAction >&,
            const std::vector< HGraphEncoderEngine::HistorySubgoal >&,
            std::optional< int > >(&HGraphMutableStreamEncoder::update),
         "id"_a,
         "state"_a,
         "goals"_a,
         "actions"_a,
         "history_subgoals"_a,
         "history_max_steps"_a = std::nullopt,
         nb::call_guard< nb::gil_scoped_release >()
      )
      .def("remove", &HGraphMutableStreamEncoder::remove, "id"_a)
      .def("set_reuse_removed", &HGraphMutableStreamEncoder::set_reuse_removed, "value"_a)
      .def("flush", &HGraphMutableStreamEncoder::flush)
      .def("flush_pyg", &HGraphMutableStreamEncoder::flush_pyg)
      .def("reset", &HGraphMutableStreamEncoder::reset);

   nb::class_< HGraphStreamEncoder >(m, "HGraphStreamEncoder")
      .def(nb::init< HGraphEncoderEngine& >(), nb::keep_alive< 1, 2 >())
      .def(
         "append",
         nb::overload_cast< const mimir::search::State& >(&HGraphStreamEncoder::append),
         "state"_a,
         nb::call_guard< nb::gil_scoped_release >()
      )
      .def(
         "append",
         nb::overload_cast<
            const mimir::search::State&,
            const GoalInputs&,
            const std::vector< mimir::formalism::GroundAction >& >(&HGraphStreamEncoder::append),
         "state"_a,
         "goals"_a,
         "actions"_a,
         nb::call_guard< nb::gil_scoped_release >()
      )
      .def(
         "append",
         nb::overload_cast<
            const mimir::search::State&,
            const GoalInputs&,
            const std::vector< mimir::formalism::GroundAction >&,
            const std::vector< HGraphEncoderEngine::HistorySubgoal >&,
            std::optional< int > >(&HGraphStreamEncoder::append),
         "state"_a,
         "goals"_a,
         "actions"_a,
         "history_subgoals"_a,
         "history_max_steps"_a = std::nullopt,
         nb::call_guard< nb::gil_scoped_release >()
      )
      .def("flush", &HGraphStreamEncoder::flush)
      .def("flush_pyg", &HGraphStreamEncoder::flush_pyg)
      .def("reset", &HGraphStreamEncoder::reset);
}

}  // namespace mifrost
