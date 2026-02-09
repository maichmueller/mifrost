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

nb::dict batch_encoding_to_parts(const BatchBuilder::BatchEncoding& encoding)
{
   BatchBuilder builder;
   builder.load_from_batch_encoding(encoding);
   return builder.build_batch_encoding_py();
}

nb::object batch_encoding_as_pyg(
   BatchBuilder::BatchEncoding& encoding,
   bool consume,
   std::optional< bool > as_batch
)
{
   BatchBuilder builder;
   const auto num_graphs = encoding.num_graphs;
   if(consume) {
      builder.load_from_batch_encoding(std::move(encoding));
      encoding = BatchBuilder::BatchEncoding{};
   } else {
      builder.load_from_batch_encoding(encoding);
   }

   const bool want_batch = as_batch.value_or(num_graphs != 1);
   if(want_batch) {
      return builder.build();
   }

   nb::dict parts = builder.build_batch_encoding_py();
   nb::object common = nb::module_::import_("mifrost.encoders.common");
   return common.attr("_parts_to_pyg")(parts, "as_batch"_a = false, "include_metadata"_a = true);
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
      .def("build_batch_encoding_py", &BatchBuilder::build_batch_encoding_py)
      .def("build_batch_encoding", &BatchBuilder::build_batch_encoding)
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
      .def("to_parts", &batch_encoding_to_parts)
      .def("as_pyg", &batch_encoding_as_pyg, "consume"_a = false, "as_batch"_a = nb::none())
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
         return builder.build_batch_encoding();
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
            return builder.build_batch_encoding();
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
            return builder.build_batch_encoding();
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
            return builder.build_batch_encoding();
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
      .def("flush_batch_encoding_py", &HGraphMutableStreamEncoder::flush_batch_encoding_py)
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
      .def("flush_batch_encoding_py", &HGraphStreamEncoder::flush_batch_encoding_py)
      .def("flush", &HGraphStreamEncoder::flush)
      .def("flush_pyg", &HGraphStreamEncoder::flush_pyg)
      .def("reset", &HGraphStreamEncoder::reset);
}

}  // namespace mifrost
