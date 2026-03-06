#pragma once

#include <fmt/format.h>

#include <array>
#include <boost/describe.hpp>
#include <cstdint>
#include <map>
#include <mimir/formalism/action.hpp>
#include <mimir/formalism/domain.hpp>
#include <mimir/formalism/ground_action.hpp>
#include <mimir/formalism/ground_literal.hpp>
#include <mimir/search/state.hpp>
#include <optional>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "batch_builder.hpp"
#include "common_types.hpp"
#include "default_relations.hpp"
#include "goal_inputs.hpp"
#include "relation_dict.hpp"
#include "relation_formatter.hpp"
#include "schema_key_separators.hpp"
#include "stream_encoder_base.hpp"
#include "target_metadata.hpp"
#include "target_source.hpp"

namespace mifrost {

namespace batch_input {
namespace parsed {
struct HGraphBatchInputs;
}
}  // namespace batch_input

/**
 * @brief Heterogeneous graph encoder engine.
 *
 * Builds a relation-typed node/edge schema from planning states/goals/actions and writes the
 * resulting tensors and metadata into a caller-provided `BatchBuilder`.
 *
 * Contract:
 *  - The caller owns `BatchBuilder` and may reuse it across steps.
 *  - Encoding mutates/extends the builder; it does not clear existing content.
 */
class HGraphEncoderEngine {
  public:
   using HistorySubgoal = std::pair< int, std::vector< LiteralVariant > >;
   using RelationRef = uint64_t;

   /**
    * @brief Per-encode scratch space for heterogeneous graph construction.
    *
    * This workspace holds temporary indices and name tables that are accumulated while encoding a
    * single state/goal/action step into the `BatchBuilder`.
    *
    * Lifetime:
    *  - Created once per encode call (see `init_hetero_workspace`).
    *  - Mutated by `encode_objects`, `encode_facts`, goal/action/history encoders.
    *  - Read by `finalize_hetero_encoding` and (optionally) `add_lgan_edges`.
    *
    * Invariants:
    *  - `node_indices` stores string-keyed dedup maps (used for literal/history-style keys).
    *  - `node_indices_i64` stores compact i64-keyed dedup maps.
    *  - `node_indices_u64` stores packed integer-keyed dedup maps.
    *  - `symbol_indices` maps compact symbol ids to node indices for the symbol node type.
    *  - `node_names[type][idx]` is the exported node name (when metadata export is enabled).
    */
   struct HeteroEncodingWorkspace {
      /// String-keyed node dedup map per node type.
      hash_map< std::string, hash_map< std::string, int64_t > > node_indices;

      /// Integer-keyed node dedup map per node type.
      hash_map< std::string, hash_map< int64_t, int64_t > > node_indices_i64;
      hash_map< std::string, hash_map< uint64_t, int64_t > > node_indices_u64;

      /// Compact symbol id -> symbol node index (for `Config::symbol_type_id`).
      hash_map< int64_t, int64_t > symbol_indices;
      hash_map< std::string, int64_t > symbol_key_to_id;

      /// Stable ids for non-object symbols (action targets, nullary pseudo-object, extra objects).
      hash_map< std::string, int64_t > special_symbol_ids;
      int64_t next_special_symbol_id = -1;

      /**
       * @brief Node index -> node name table per node type.
       *
       * Maintained in lockstep with node creation so names can be exported to the builder.
       */
      hash_map< std::string, std::vector< std::string > > node_names;

      /**
       * @brief Relation-ref -> set of symbol ids appearing in that relation.
       *
       * RelationRef packs `(type_id, relation_idx)` into one `uint64_t`.
       */
      hash_map< RelationRef, hash_set< int64_t > > relation_to_symbols;

      /**
       * @brief Symbol id -> set of relation refs in which the symbol appears.
       *
       * Reverse index of `relation_to_symbols`.
       * This enables fast lookup of all relations that a given symbol participates in, which is
       * used when computing LGAN-style edges.
       */
      hash_map< int64_t, hash_set< RelationRef > > symbol_to_relations;

      /// Per-encode node-type interning used by RelationRef.
      hash_map< std::string, uint32_t > relation_type_ids;
      std::vector< std::string > relation_type_names;

      /// Explicit LGAN target symbol ids (action targets / horizon target nodes).
      hash_set< int64_t > lgan_target_symbol_ids;

      /// Target metadata rows for configured `Config::target_sources`.
      TargetColumns targets;
      /// Stable target group labels, indexed by target_group_ids.
      std::vector< std::string > target_groups;
      std::map< TargetSource, int64_t > target_group_ids;
      int64_t next_target_index = 0;
   };

   /// Runtime configuration controlling which relations/nodes/edges are emitted.
   struct Config {
      std::string symbol_type_id = defaults::symbol_type_id;
      std::string target_symbol_prefix = std::string(kDefaultTargetSymbolPrefix);
      std::string nullary_object_name = "![nullary_symbol]!";
      std::string lgan_tn_edge_pos = defaults::lgan_tn_edge_pos;
      std::string lgan_nn_edge_pos = defaults::lgan_nn_edge_pos;
      std::string lgan_rr_edge_pos = defaults::lgan_rr_edge_pos;
      std::string history_link_relation = defaults::history_link_relation;
      size_t max_goal_level = 0;
      bool support_literals = false;
      bool add_nullary_predicates = false;
      bool ignore_actions = true;
      bool include_lgan_edges = false;
      bool include_static = true;
      bool include_empty_edge_types = true;
      bool export_node_names = true;
      std::set< TargetSource > target_sources = {};
      std::set< GoalSatisfaction > goal_satisfaction_derivations = {GoalSatisfaction::satisfied};
   };

   /// Construct from a borrowed domain implementation reference (caller must keep it alive).
   explicit HGraphEncoderEngine(const mimir::formalism::DomainImpl& domain);
   HGraphEncoderEngine(const mimir::formalism::DomainImpl& domain, Config config);

   /// Construct from an owning domain handle (engine stores the domain internally).
   explicit HGraphEncoderEngine(mimir::formalism::Domain domain);
   HGraphEncoderEngine(mimir::formalism::Domain domain, Config config);

   ~HGraphEncoderEngine() = default;

   /**
    * @brief Encode a state-only step.
    *
    * Writes symbol/object nodes and fact relations derived from `state`.
    *
    * @param state Planning state to encode.
    * @param builder Output builder to append into.
    */
   void encode_state(const mimir::search::State& state, BatchBuilder& builder)
   {
      encode_state_impl(state, builder);
   }

   /**
    * @brief Encode a state with typed goal literals and optional actions.
    *
    * Convenience overload that wraps typed goal literals into `GoalInputs` internally.
    *
    * @tparam GoalTag Tag of the provided goal literals (Static/Fluent/Derived).
    * @param state Planning state to encode.
    * @param goals Goal literals of the given tag.
    * @param actions Actions to encode (ignored when `Config::ignore_actions` is true).
    * @param builder Output builder to append into.
    */
   template < typename GoalTag >
   void encode_step(
      const mimir::search::State& state,
      std::span< const mimir::formalism::GroundLiteral< GoalTag > > goals,
      std::span< const mimir::formalism::GroundAction > actions,
      BatchBuilder& builder
   )
   {
      encode_step_impl(state, goals, actions, builder);
   }

   /**
    * @brief Encode a state with fully split `GoalInputs` and explicit actions.
    *
    * Encodes objects, facts, goal literal nodes, optional actions, goal-satisfaction nodes, and
    * optional LGAN edges (depending on `Config`).
    *
    * @param state Planning state to encode.
    * @param goals Goal input groups and (optional) goal levels.
    * @param actions Actions to encode.
    * @param builder Output builder to append into.
    */
   void encode(
      const mimir::search::State& state,
      const GoalInputs& goals,
      std::span< const mimir::formalism::GroundAction > actions,
      BatchBuilder& builder
   )
   {
      encode_impl(state, goals, actions, builder);
   }

   /**
    * @brief Encode a step with optional history subgoals.
    *
    * History entries are encoded into a dedicated "history" node type and linked to the referenced
    * relation nodes via `Config::history_link_relation`.
    *
    * @param state Planning state to encode.
    * @param goals Goal input groups and (optional) goal levels.
    * @param actions Actions to encode.
    * @param history_subgoals (dt, literals) pairs; dt must be negative (past steps).
    * @param history_max_steps Optional limit; entries with |dt| > max are skipped.
    * @param builder Output builder to append into.
    *
    * @throws std::invalid_argument if any dt value is non-negative.
    */
   void encode(
      const mimir::search::State& state,
      const GoalInputs& goals,
      std::span< const mimir::formalism::GroundAction > actions,
      const std::vector< HistorySubgoal >& history_subgoals,
      std::optional< int > history_max_steps,
      BatchBuilder& builder
   )
   {
      encode_impl_core(state, goals, actions, history_subgoals, history_max_steps, builder);
   }

   /// Encode a state-only step (goals inferred from the state's problem).
   void encode(const mimir::search::State& state, BatchBuilder& builder)
   {
      encode_state(state, builder);
   }

   /// Encode a parsed batch input plan into one batch encoding.
   BatchBuilder::BatchEncoding encode_batch(
      const batch_input::parsed::HGraphBatchInputs& inputs,
      std::optional< int > history_max_steps
   );

   /// Return the effective runtime configuration.
   const Config& get_config() const { return config_; }

   /// Return the effective built relation dictionary.
   const RelationDict& get_relation_dict() const { return relation_dict_; }

   /// Replace the effective relation dictionary used by subsequent encodes.
   void update_relations(RelationDict relation_dict);

  protected:
   /**
    * @brief Initialize per-step workspace and ensure feature dimension defaults.
    *
    * Called once per encode step. Ensures required node feature dims exist in `builder`.
    */
   HeteroEncodingWorkspace& init_hetero_workspace(BatchBuilder& builder);

   /**
    * @brief Encode goal literals for all goal tags.
    *
    * Appends literal relation nodes and their symbol-position edges.
    */
   void encode_goal_inputs(
      const GoalInputs& goals,
      BatchBuilder& builder,
      HeteroEncodingWorkspace& workspace,
      std::span< const std::string > extra_objects = {}
   );

   /**
    * @brief Encode goal-satisfaction nodes for all goal tags.
    *
    * Uses `fact_keys` to decide satisfied/unsatisfied variants and emits the configured
    * derivations.
    */
   void encode_goal_satisfaction_inputs(
      const GoalInputs& goals,
      const hash_set< uint64_t >& fact_keys,
      BatchBuilder& builder,
      HeteroEncodingWorkspace& workspace,
      std::string_view suffix = "",
      std::span< const std::string > extra_objects = {}
   );

   /**
    * @brief Conditionally add LGAN TN/NN/RR edges.
    *
    * Uses `workspace.relation_to_symbols` / `workspace.symbol_to_relations` collected during
    * encoding.
    */
   void maybe_add_lgan_edges(BatchBuilder& builder, const HeteroEncodingWorkspace& workspace);

   /**
    * @brief Finalize builder metadata after encoding.
    *
    * Exports node names per type, sets object names for the symbol type, and ensures empty edge
    * types if configured.
    *
    * @param object_names_override Optional object-name list to export instead of symbol node names.
    */
   void finalize_hetero_encoding(
      BatchBuilder& builder,
      const HeteroEncodingWorkspace& workspace,
      const std::vector< std::string >* object_names_override = nullptr
   ) const;

   /// Build the relation dictionary and precompute schema metadata from the configured domain.
   void initialize_from_domain();

   /// Recompute predeclared edge-type metadata from the current relation dictionary.
   void rebuild_all_edge_types();

   /// Internal implementation for typed goals.
   template < typename GoalTag >
   void encode_step_impl(
      const mimir::search::State& state,
      std::span< const mimir::formalism::GroundLiteral< GoalTag > > goals,
      std::span< const mimir::formalism::GroundAction > actions,
      BatchBuilder& builder
   );

   /// Internal implementation for state-only encoding.
   void encode_state_impl(const mimir::search::State& state, BatchBuilder& builder);

   /// Internal implementation for full encoding.
   void encode_impl(
      const mimir::search::State& state,
      const GoalInputs& goals,
      std::span< const mimir::formalism::GroundAction > actions,
      BatchBuilder& builder
   );

   /// Shared implementation for full encoding, with optional history support.
   void encode_impl_core(
      const mimir::search::State& state,
      const GoalInputs& goals,
      std::span< const mimir::formalism::GroundAction > actions,
      std::span< const HistorySubgoal > history_subgoals,
      std::optional< int > history_max_steps,
      BatchBuilder& builder
   );

   /**
    * @brief Encode symbol/object nodes.
    *
    * Emits the configured symbol node type and records indices/names in the workspace tables.
    * Extra objects are appended as additional symbols.
    */
   void encode_objects(
      const mimir::search::State& state,
      BatchBuilder& builder,
      hash_map< std::string, hash_map< std::string, int64_t > >& node_indices,
      hash_map< std::string, std::vector< std::string > >& node_names,
      std::span< const std::string > extra_objects = {}
   );

   /**
    * @brief Encode fact atoms and return formatted fact keys.
    *
    * Adds relation nodes for all included facts and returns the set of formatted fact keys used for
    * goal-satisfaction derivations.
    */
   hash_set< uint64_t > encode_facts(
      const mimir::search::State& state,
      BatchBuilder& builder,
      hash_map< std::string, hash_map< std::string, int64_t > >& node_indices,
      hash_map< std::string, std::vector< std::string > >& node_names,
      hash_map< RelationRef, hash_set< int64_t > >& relation_to_symbols,
      hash_map< int64_t, hash_set< RelationRef > >& symbol_to_relations,
      std::span< const std::string > extra_objects = {}
   );

   /**
    * @brief Encode goal literals as relation nodes.
    *
    * Adds literal relation nodes and symbol-position edges; also updates relation/symbol incidence
    * maps for optional LGAN edge derivation.
    */
   template < typename GoalTag >
   void encode_literals(
      std::span< const mimir::formalism::GroundLiteral< GoalTag > > goals,
      const hash_map< mimir::formalism::GroundLiteral< GoalTag >, size_t >& goal_levels,
      BatchBuilder& builder,
      hash_map< std::string, hash_map< std::string, int64_t > >& node_indices,
      hash_map< std::string, std::vector< std::string > >& node_names,
      hash_map< RelationRef, hash_set< int64_t > >& relation_to_symbols,
      hash_map< int64_t, hash_set< RelationRef > >& symbol_to_relations,
      std::span< const std::string > extra_objects = {}
   );

   /**
    * @brief Encode grounded actions as relation nodes.
    *
    * Adds action relation nodes and symbol-position edges; ignored when `Config::ignore_actions` is
    * true.
    */
   void encode_actions(
      std::span< const mimir::formalism::GroundAction > actions,
      BatchBuilder& builder,
      hash_map< std::string, hash_map< std::string, int64_t > >& node_indices,
      hash_map< std::string, std::vector< std::string > >& node_names,
      hash_map< RelationRef, hash_set< int64_t > >& relation_to_symbols,
      hash_map< int64_t, hash_set< RelationRef > >& symbol_to_relations,
      std::span< const std::string > extra_objects = {}
   );

   /**
    * @brief Encode goal-satisfaction derivations.
    *
    * Emits satisfied/unsatisfied variants as configured in `RelationDictConfig`.
    */
   template < typename GoalTag >
   void encode_goal_satisfaction(
      std::span< const mimir::formalism::GroundLiteral< GoalTag > > goals,
      const hash_map< mimir::formalism::GroundLiteral< GoalTag >, size_t >& goal_levels,
      const hash_set< uint64_t >& fact_keys,
      BatchBuilder& builder,
      hash_map< std::string, hash_map< std::string, int64_t > >& node_indices,
      hash_map< std::string, std::vector< std::string > >& node_names,
      hash_map< RelationRef, hash_set< int64_t > >& relation_to_symbols,
      hash_map< int64_t, hash_set< RelationRef > >& symbol_to_relations,
      std::string_view suffix = "",
      std::span< const std::string > extra_objects = {}
   );

   /**
    * @brief Add LGAN target-neighbor, neighbor-neighbor, and relation-relation edges.
    *
    * Targets are explicit target symbols only (action targets / horizon target nodes).
    */
   void add_lgan_edges(
      BatchBuilder& builder,
      const hash_set< int64_t >& lgan_target_symbol_ids,
      const hash_map< int64_t, int64_t >& symbol_indices,
      const hash_map< RelationRef, hash_set< int64_t > >& relation_to_symbols,
      const hash_map< int64_t, hash_set< RelationRef > >& symbol_to_relations,
      const std::vector< std::string >& relation_type_names
   );

   /**
    * @brief Encode history subgoals.
    *
    * Creates "history" nodes with a single feature (dt) and links each referenced literal relation
    * node bidirectionally via `Config::history_link_relation`.
    */
   void encode_history(
      std::span< const HistorySubgoal > history_subgoals,
      std::optional< int > history_max_steps,
      BatchBuilder& builder,
      hash_map< std::string, hash_map< std::string, int64_t > >& node_indices,
      hash_map< std::string, std::vector< std::string > >& node_names,
      hash_map< RelationRef, hash_set< int64_t > >& relation_to_symbols,
      hash_map< int64_t, hash_set< RelationRef > >& symbol_to_relations
   );

   /**
    * @brief Track relation-symbol incidence only when LGAN edges are enabled.
    */
   void track_relation_symbols_if_enabled(
      RelationRef rel_ref,
      std::span< const int64_t > object_symbol_ids,
      std::span< const int64_t > extra_symbol_ids,
      hash_map< RelationRef, hash_set< int64_t > >& relation_to_symbols,
      hash_map< int64_t, hash_set< RelationRef > >& symbol_to_relations
   );
   void track_relation_symbols_if_enabled(
      RelationRef rel_ref,
      std::span< const std::string > object_keys,
      std::span< const std::string > extra_objects,
      hash_map< RelationRef, hash_set< int64_t > >& relation_to_symbols,
      hash_map< int64_t, hash_set< RelationRef > >& symbol_to_relations
   );

   /// Ensure configured edge types exist in the schema even when no edges were emitted.
   void ensure_empty_edge_types(BatchBuilder& builder) const;
   /// Ensure node feature dimensions are set for all configured node types.
   void ensure_node_feature_dims(BatchBuilder& builder) const;

   /// Append one directed edge (single src/dst index) to the builder.
   static void append_edges(
      BatchBuilder& builder,
      const std::string& src_type,
      const std::string& rel_type,
      const std::string& dst_type,
      int64_t src,
      int64_t dst
   );
   static uint64_t pack_u32_u32(uint32_t a, uint32_t b);
   static uint64_t pack_i32_u32(int32_t a, uint32_t b);
   static RelationRef relation_ref_from_parts(uint32_t type_id, uint32_t relation_idx);
   uint32_t get_or_assign_relation_type_id(const std::string& node_type);
   RelationRef relation_ref_for(const std::string& node_type, int64_t relation_idx);

   std::string symbol_node_key(const mimir::formalism::Object& obj) const;
   [[nodiscard]] bool has_target_source(TargetSource source) const;
   [[nodiscard]] static std::string_view target_group_name(TargetSource source);
   int64_t get_or_assign_target_group_id(TargetSource source);

   int64_t get_or_assign_special_symbol_id(std::string_view symbol_name);
   int64_t get_or_add_symbol_object_node(
      const mimir::formalism::Object& obj,
      BatchBuilder& builder,
      hash_map< std::string, std::vector< std::string > >& node_names
   );
   int64_t get_or_add_symbol_special_node(
      std::string_view symbol_key,
      std::string_view symbol_name,
      BatchBuilder& builder,
      hash_map< std::string, std::vector< std::string > >& node_names
   );
   int64_t get_or_add_symbol_node(
      int64_t symbol_id,
      std::string_view symbol_key,
      std::string_view symbol_name,
      BatchBuilder& builder,
      hash_map< std::string, std::vector< std::string > >& node_names
   );
   int64_t get_or_add_relation_node_i64(
      const std::string& node_type,
      int64_t key,
      BatchBuilder& builder,
      hash_map< std::string, hash_map< std::string, int64_t > >& node_indices,
      hash_map< std::string, std::vector< std::string > >& node_names,
      std::string_view node_name
   );
   int64_t get_or_add_relation_node_u64(
      const std::string& node_type,
      uint64_t key,
      BatchBuilder& builder,
      hash_map< std::string, hash_map< std::string, int64_t > >& node_indices,
      hash_map< std::string, std::vector< std::string > >& node_names,
      std::string_view node_name
   );

   /**
    * @brief Get or create a node index for (type,key).
    *
    * Ensures `node_indices[type][key]` and `node_names[type]` stay aligned with builder node
    * counts.
    *
    * @return Node index in the given type.
    */
   int64_t get_or_add_node(
      const std::string& node_type,
      const std::string& node_key,
      BatchBuilder& builder,
      hash_map< std::string, hash_map< std::string, int64_t > >& node_indices,
      hash_map< std::string, std::vector< std::string > >& node_names,
      bool store_node_name
   );

   /// Optional owning domain storage (used when constructed from a domain handle).
   mimir::formalism::Domain domain_holder_;
   /// Active domain implementation reference (either borrowed or owned via `domain_holder_`).
   const mimir::formalism::DomainImpl& domain_;
   /// Effective runtime configuration.
   Config config_;
   /// Reused scratch workspace to avoid repeated per-step allocations.
   HeteroEncodingWorkspace workspace_;
   /// Derived relation and schema metadata.
   RelationDict relation_dict_;
   /// Precomputed edge types used when `Config::include_empty_edge_types` is enabled.
   std::vector< std::tuple< std::string, std::string, std::string > > all_edge_types_;
};

BOOST_DESCRIBE_STRUCT(
   HGraphEncoderEngine::Config,
   (),
   (symbol_type_id,
    target_symbol_prefix,
    nullary_object_name,
    lgan_tn_edge_pos,
    lgan_nn_edge_pos,
    lgan_rr_edge_pos,
    history_link_relation,
    max_goal_level,
    support_literals,
    add_nullary_predicates,
    ignore_actions,
    include_lgan_edges,
    include_static,
    include_empty_edge_types,
    export_node_names,
    target_sources,
    goal_satisfaction_derivations)
)

/**
 * @brief Input payload for one streaming encode step.
 *
 * Fields are optional; `state` must be set. When `goals` is null, the engine encodes state-only.
 */
struct HGraphStepInput {
   const mimir::search::State* state = nullptr;
   const GoalInputs* goals = nullptr;
   const std::vector< mimir::formalism::GroundAction >* actions = nullptr;
   const std::vector< HGraphEncoderEngine::HistorySubgoal >* history = nullptr;
   std::optional< int > history_max_steps;
};

/**
 * @brief Mutable streaming wrapper around `HGraphEncoderEngine`.
 *
 * Maintains a stream of step inputs and encodes them into builders via `StreamEncoderBase`.
 */
class HGraphMutableStreamEncoder:
    public StreamEncoderBase< HGraphMutableStreamEncoder, HGraphStepInput > {
  public:
   /// Identifier of the produced graph kind (used by the stream base).
   static constexpr std::string_view graph_kind() { return "hetero"; }

   explicit HGraphMutableStreamEncoder(HGraphEncoderEngine& engine) : engine_(&engine) { reset(); }

   int64_t append(const mimir::search::State& state)
   {
      HGraphStepInput step;
      step.state = &state;
      return StreamEncoderBase::append(step);
   }

   int64_t append(
      const mimir::search::State& state,
      const GoalInputs& goals,
      const std::vector< mimir::formalism::GroundAction >& actions
   )
   {
      HGraphStepInput step;
      step.state = &state;
      step.goals = &goals;
      step.actions = &actions;
      return StreamEncoderBase::append(step);
   }

   int64_t append(
      const mimir::search::State& state,
      const GoalInputs& goals,
      const std::vector< mimir::formalism::GroundAction >& actions,
      const std::vector< HGraphEncoderEngine::HistorySubgoal >& history,
      std::optional< int > history_max_steps
   )
   {
      HGraphStepInput step;
      step.state = &state;
      step.goals = &goals;
      step.actions = &actions;
      step.history = &history;
      step.history_max_steps = history_max_steps;
      return StreamEncoderBase::append(step);
   }

   void update(int64_t id, const mimir::search::State& state)
   {
      HGraphStepInput step;
      step.state = &state;
      StreamEncoderBase::update(id, step);
   }

   void update(
      int64_t id,
      const mimir::search::State& state,
      const GoalInputs& goals,
      const std::vector< mimir::formalism::GroundAction >& actions
   )
   {
      HGraphStepInput step;
      step.state = &state;
      step.goals = &goals;
      step.actions = &actions;
      StreamEncoderBase::update(id, step);
   }

   void update(
      int64_t id,
      const mimir::search::State& state,
      const GoalInputs& goals,
      const std::vector< mimir::formalism::GroundAction >& actions,
      const std::vector< HGraphEncoderEngine::HistorySubgoal >& history,
      std::optional< int > history_max_steps
   )
   {
      HGraphStepInput step;
      step.state = &state;
      step.goals = &goals;
      step.actions = &actions;
      step.history = &history;
      step.history_max_steps = history_max_steps;
      StreamEncoderBase::update(id, step);
   }

   void encode_step(const HGraphStepInput& step, BatchBuilder& builder)
   {
      if(engine_ == nullptr or step.state == nullptr) {
         throw std::invalid_argument("HGraphMutableStreamEncoder requires a valid engine/state");
      }

      const auto& state = *step.state;
      if(step.goals == nullptr) {
         if(step.history != nullptr and not step.history->empty()) {
            throw std::invalid_argument("History encoding requires explicit GoalInputs");
         }
         engine_->encode(state, builder);
         return;
      }

      std::span< const mimir::formalism::GroundAction > actions_span;
      if(step.actions != nullptr) {
         actions_span = std::span< const mimir::formalism::GroundAction >(*step.actions);
      }

      if(step.history != nullptr and not step.history->empty()) {
         engine_->encode(
            state, *step.goals, actions_span, *step.history, step.history_max_steps, builder
         );
      } else {
         engine_->encode(state, *step.goals, actions_span, builder);
      }
   }

  private:
   HGraphEncoderEngine* engine_ = nullptr;
};

/**
 * @brief Append-only streaming wrapper around `HGraphEncoderEngine`.
 *
 * Uses one persistent builder and commits one graph per append.
 */
class HGraphStreamEncoder {
  public:
   explicit HGraphStreamEncoder(HGraphEncoderEngine& engine) : engine_(&engine) { reset(); }

   int64_t append(const mimir::search::State& state)
   {
      ensure_valid();
      engine_->encode(state, builder_);
      builder_.next_graph();
      return next_id_++;
   }

   int64_t append(
      const mimir::search::State& state,
      const GoalInputs& goals,
      const std::vector< mimir::formalism::GroundAction >& actions
   )
   {
      ensure_valid();
      engine_->encode(state, goals, actions, builder_);
      builder_.next_graph();
      return next_id_++;
   }

   int64_t append(
      const mimir::search::State& state,
      const GoalInputs& goals,
      const std::vector< mimir::formalism::GroundAction >& actions,
      const std::vector< HGraphEncoderEngine::HistorySubgoal >& history,
      std::optional< int > history_max_steps
   )
   {
      ensure_valid();
      engine_->encode(state, goals, actions, history, history_max_steps, builder_);
      builder_.next_graph();
      return next_id_++;
   }

   BatchEncoding flush()
   {
      ensure_valid();
      auto out = builder_.build();
      reset();
      return out;
   }

   nb::object flush_pyg()
   {
      ensure_valid();
      auto out = builder_.build_pyg();
      reset();
      return out;
   }

   void reset()
   {
      builder_.reset();
      builder_.set_graph_kind("hetero");
      next_id_ = 0;
   }

  private:
   void ensure_valid() const
   {
      if(engine_ == nullptr) {
         throw std::invalid_argument("HGraphStreamEncoder requires a valid engine");
      }
   }

   HGraphEncoderEngine* engine_ = nullptr;
   BatchBuilder builder_;
   int64_t next_id_ = 0;
};

template < typename GoalTag >
void HGraphEncoderEngine::encode_step_impl(
   const mimir::search::State& state,
   std::span< const mimir::formalism::GroundLiteral< GoalTag > > goals,
   std::span< const mimir::formalism::GroundAction > actions,
   BatchBuilder& builder
)
{
   GoalInputs inputs;
   for(const auto& goal : goals) {
      if constexpr(std::is_same_v< GoalTag, mimir::formalism::StaticTag >) {
         inputs.static_goals.emplace_back(goal);
         inputs.static_goal_levels[goal] = 0;
      } else if constexpr(std::is_same_v< GoalTag, mimir::formalism::FluentTag >) {
         inputs.fluent_goals.emplace_back(goal);
         inputs.fluent_goal_levels[goal] = 0;
      } else {
         inputs.derived_goals.emplace_back(goal);
         inputs.derived_goal_levels[goal] = 0;
      }
   }
   encode_impl(state, inputs, actions, builder);
}

template < typename GoalTag >
void HGraphEncoderEngine::encode_literals(
   std::span< const mimir::formalism::GroundLiteral< GoalTag > > goals,
   const hash_map< mimir::formalism::GroundLiteral< GoalTag >, size_t >& goal_levels,
   BatchBuilder& builder,
   hash_map< std::string, hash_map< std::string, int64_t > >& node_indices,
   hash_map< std::string, std::vector< std::string > >& node_names,
   hash_map< RelationRef, hash_set< int64_t > >& relation_to_symbols,
   hash_map< int64_t, hash_set< RelationRef > >& symbol_to_relations,
   std::span< const std::string > extra_objects
)
{
   for(const auto& literal : goals) {
      const auto atom = literal->get_atom();
      const auto predicate = atom->get_predicate();
      const std::optional< int > goal_level = goal_levels.contains(literal)
                                                 ? std::optional< size_t >(goal_levels.at(literal))
                                                 : std::nullopt;
      const bool is_subgoal = goal_level.has_value() and (*goal_level > 0);
      std::optional< TargetSource > target_source = std::nullopt;
      if(is_subgoal and has_target_source(TargetSource::Subgoals)) {
         target_source = TargetSource::Subgoals;
      } else if((not is_subgoal) and has_target_source(TargetSource::Goals)) {
         target_source = TargetSource::Goals;
      }

      std::string node_type;
      std::string node_name;
      std::string formatted_literal;
      if(goal_level.has_value()) {
         const GoalLevel level(*goal_level);
         node_type = RelationFormatter::format_predicate(
            predicate, level, std::nullopt, literal->get_polarity()
         );
         formatted_literal = RelationFormatter::format_literal< GoalTag >(literal, level);
         node_name = config_.export_node_names ? formatted_literal : "";
      } else {
         node_type = RelationFormatter::format_predicate(
            predicate, std::nullopt, std::nullopt, literal->get_polarity()
         );
         formatted_literal = RelationFormatter::format_literal< GoalTag >(literal, std::nullopt);
         node_name = config_.export_node_names ? formatted_literal : "";
      }

      std::vector< int64_t > object_symbol_ids;
      if(predicate->get_arity() == 0) {
         if(not config_.add_nullary_predicates) {
            continue;
         }
         const auto nullary_idx = get_or_add_symbol_special_node(
            config_.nullary_object_name, config_.nullary_object_name, builder, node_names
         );
         (void) nullary_idx;
         object_symbol_ids.emplace_back(
            get_or_assign_special_symbol_id(config_.nullary_object_name)
         );
      } else {
         for(const auto& obj : atom->get_objects()) {
            const auto obj_idx = get_or_add_symbol_object_node(obj, builder, node_names);
            (void) obj_idx;
            object_symbol_ids.emplace_back(static_cast< int64_t >(obj->get_index()));
         }
      }
      const auto relation_idx = get_or_add_relation_node_i64(
         node_type,
         static_cast< int64_t >(atom->get_index()),
         builder,
         node_indices,
         node_names,
         node_name
      );

      std::vector< int64_t > extra_symbol_ids;
      extra_symbol_ids.reserve(
         extra_objects.size() + static_cast< size_t >(target_source.has_value() ? 1 : 0)
      );

      size_t pos = 0;
      if(target_source.has_value()) {
         const std::string target_symbol_key = fmt::format(
            "{}{}{}{}{}{}{}{}{}{}",
            config_.target_symbol_prefix,
            target_group_name(*target_source),
            schema_key::kEdgeTypeSeparator,
            node_type,
            schema_key::kEdgeTypeSeparator,
            literal->get_polarity() ? 1 : 0,
            schema_key::kEdgeTypeSeparator,
            atom->get_index(),
            schema_key::kEdgeTypeSeparator,
            goal_level.value_or(-1)
         );
         const std::string target_symbol_name = config_.export_node_names
                                                   ? fmt::format(
                                                        "{}{}{}",
                                                        target_symbol_key,
                                                        schema_key::kEdgeTypeSeparator,
                                                        formatted_literal
                                                     )
                                                   : target_symbol_key;
         const auto target_symbol_idx = get_or_add_symbol_special_node(
            target_symbol_key, target_symbol_name, builder, node_names
         );
         const auto target_symbol_id = get_or_assign_special_symbol_id(target_symbol_key);
         if(config_.include_lgan_edges) {
            workspace_.lgan_target_symbol_ids.insert(target_symbol_id);
         }
         workspace_.targets.append(
            TargetRecord{
               .position = target_symbol_idx,
               .index = workspace_.next_target_index,
               .candidate_id = workspace_.next_target_index,
               .depth = std::nullopt,
               .group_id = get_or_assign_target_group_id(*target_source),
               .name = formatted_literal,
            },
            /*include_depth=*/false,
            /*include_group=*/true
         );
         ++workspace_.next_target_index;
         extra_symbol_ids.emplace_back(target_symbol_id);
         const std::string pos_str = std::to_string(pos++);
         append_edges(
            builder, config_.symbol_type_id, pos_str, node_type, target_symbol_idx, relation_idx
         );
         append_edges(
            builder, node_type, pos_str, config_.symbol_type_id, relation_idx, target_symbol_idx
         );
      }

      for(const auto symbol_id : object_symbol_ids) {
         const auto obj_idx = workspace_.symbol_indices.at(symbol_id);
         const std::string pos_str = std::to_string(pos++);
         append_edges(builder, config_.symbol_type_id, pos_str, node_type, obj_idx, relation_idx);
         append_edges(builder, node_type, pos_str, config_.symbol_type_id, relation_idx, obj_idx);
      }

      for(size_t i = 0; i < extra_objects.size(); ++i) {
         const auto& symbol_name = extra_objects[i];
         const auto obj_idx = get_or_add_symbol_special_node(
            symbol_name, symbol_name, builder, node_names
         );
         const auto symbol_id = get_or_assign_special_symbol_id(symbol_name);
         extra_symbol_ids.emplace_back(symbol_id);
         const std::string pos_str = std::to_string(pos++);
         append_edges(builder, config_.symbol_type_id, pos_str, node_type, obj_idx, relation_idx);
         append_edges(builder, node_type, pos_str, config_.symbol_type_id, relation_idx, obj_idx);
      }

      const auto rel_ref = relation_ref_for(node_type, relation_idx);
      track_relation_symbols_if_enabled(
         rel_ref,
         std::span{object_symbol_ids},
         std::span{extra_symbol_ids},
         relation_to_symbols,
         symbol_to_relations
      );
   }
}

template < typename GoalTag >
void HGraphEncoderEngine::encode_goal_satisfaction(
   std::span< const mimir::formalism::GroundLiteral< GoalTag > > goals,
   const hash_map< mimir::formalism::GroundLiteral< GoalTag >, size_t >& goal_levels,
   const hash_set< uint64_t >& fact_keys,
   BatchBuilder& builder,
   hash_map< std::string, hash_map< std::string, int64_t > >& node_indices,
   hash_map< std::string, std::vector< std::string > >& node_names,
   hash_map< RelationRef, hash_set< int64_t > >& relation_to_symbols,
   hash_map< int64_t, hash_set< RelationRef > >& symbol_to_relations,
   std::string_view suffix,
   std::span< const std::string > extra_objects
)
{
   for(const auto& goal : goals) {
      const auto atom = goal->get_atom();
      const auto predicate = atom->get_predicate();
      const auto fact_key = pack_u32_u32(
         static_cast< uint32_t >(goal->get_atom()->get_index()),
         std::is_same_v< GoalTag, mimir::formalism::StaticTag >
            ? 1U
            : (std::is_same_v< GoalTag, mimir::formalism::FluentTag > ? 2U : 3U)
      );
      const bool satisfied = fact_keys.contains(fact_key) == goal->get_polarity();
      const GoalSatisfaction sat = satisfied ? GoalSatisfaction::satisfied
                                             : GoalSatisfaction::unsatisfied;
      if(not relation_dict_.goal_satisfaction_derivations.contains(sat)) {
         continue;
      }

      std::optional< size_t > goal_level = goal_levels.contains(goal)
                                              ? std::optional< size_t >(goal_levels.at(goal))
                                              : std::nullopt;
      std::optional< TargetSource > target_source = std::nullopt;

      std::string node_type;
      std::string node_name;
      std::string formatted_literal;
      if(goal_level.has_value()) {
         const GoalLevel level(*goal_level);
         node_type = RelationFormatter::format_predicate(
            predicate, level, sat, goal->get_polarity(), suffix
         );
         formatted_literal = RelationFormatter::format_literal< GoalTag >(
            goal, level, sat, std::nullopt, suffix
         );
         node_name = config_.export_node_names ? formatted_literal : "";
      } else {
         node_type = RelationFormatter::format_predicate(
            predicate, std::nullopt, sat, goal->get_polarity(), suffix
         );
         formatted_literal = RelationFormatter::format_literal< GoalTag >(
            goal, std::nullopt, sat, std::nullopt, suffix
         );
         node_name = config_.export_node_names ? formatted_literal : "";
      }

      std::vector< int64_t > object_symbol_ids;
      if(predicate->get_arity() == 0) {
         if(not config_.add_nullary_predicates) {
            continue;
         }
         const auto nullary_idx = get_or_add_symbol_special_node(
            config_.nullary_object_name, config_.nullary_object_name, builder, node_names
         );
         (void) nullary_idx;
         object_symbol_ids.emplace_back(
            get_or_assign_special_symbol_id(config_.nullary_object_name)
         );
      } else {
         for(const auto& obj : atom->get_objects()) {
            const auto obj_idx = get_or_add_symbol_object_node(obj, builder, node_names);
            (void) obj_idx;
            object_symbol_ids.emplace_back(static_cast< int64_t >(obj->get_index()));
         }
      }
      const auto relation_idx = get_or_add_relation_node_i64(
         node_type,
         static_cast< int64_t >(atom->get_index()),
         builder,
         node_indices,
         node_names,
         node_name
      );

      std::vector< int64_t > extra_symbol_ids;
      extra_symbol_ids.reserve(
         extra_objects.size() + static_cast< size_t >(target_source.has_value() ? 1 : 0)
      );

      size_t pos = 0;
      if(target_source.has_value()) {
         const std::string target_symbol_key = fmt::format(
            "{}{}{}{}{}{}{}{}{}{}",
            config_.target_symbol_prefix,
            target_group_name(*target_source),
            schema_key::kEdgeTypeSeparator,
            node_type,
            schema_key::kEdgeTypeSeparator,
            goal->get_polarity() ? 1 : 0,
            schema_key::kEdgeTypeSeparator,
            atom->get_index(),
            schema_key::kEdgeTypeSeparator,
            goal_level.has_value() ? static_cast< int >(*goal_level) : -1
         );
         const std::string target_symbol_name = config_.export_node_names
                                                   ? fmt::format(
                                                        "{}{}{}",
                                                        target_symbol_key,
                                                        schema_key::kEdgeTypeSeparator,
                                                        formatted_literal
                                                     )
                                                   : target_symbol_key;
         const auto target_symbol_idx = get_or_add_symbol_special_node(
            target_symbol_key, target_symbol_name, builder, node_names
         );
         const auto target_symbol_id = get_or_assign_special_symbol_id(target_symbol_key);
         if(config_.include_lgan_edges) {
            workspace_.lgan_target_symbol_ids.insert(target_symbol_id);
         }
         workspace_.targets.append(
            TargetRecord{
               .position = target_symbol_idx,
               .index = workspace_.next_target_index,
               .candidate_id = workspace_.next_target_index,
               .depth = std::nullopt,
               .group_id = get_or_assign_target_group_id(*target_source),
               .name = formatted_literal,
            },
            /*include_depth=*/false,
            /*include_group=*/true
         );
         ++workspace_.next_target_index;
         extra_symbol_ids.emplace_back(target_symbol_id);
         const std::string pos_str = std::to_string(pos++);
         append_edges(
            builder, config_.symbol_type_id, pos_str, node_type, target_symbol_idx, relation_idx
         );
         append_edges(
            builder, node_type, pos_str, config_.symbol_type_id, relation_idx, target_symbol_idx
         );
      }

      for(const auto symbol_id : object_symbol_ids) {
         const auto obj_idx = workspace_.symbol_indices.at(symbol_id);
         const std::string pos_str = std::to_string(pos++);
         append_edges(builder, config_.symbol_type_id, pos_str, node_type, obj_idx, relation_idx);
         append_edges(builder, node_type, pos_str, config_.symbol_type_id, relation_idx, obj_idx);
      }

      for(size_t i = 0; i < extra_objects.size(); ++i) {
         const auto& symbol_name = extra_objects[i];
         const auto obj_idx = get_or_add_symbol_special_node(
            symbol_name, symbol_name, builder, node_names
         );
         const auto symbol_id = get_or_assign_special_symbol_id(symbol_name);
         extra_symbol_ids.emplace_back(symbol_id);
         const std::string pos_str = std::to_string(pos++);
         append_edges(builder, config_.symbol_type_id, pos_str, node_type, obj_idx, relation_idx);
         append_edges(builder, node_type, pos_str, config_.symbol_type_id, relation_idx, obj_idx);
      }

      const auto rel_ref = relation_ref_for(node_type, relation_idx);
      track_relation_symbols_if_enabled(
         rel_ref,
         std::span{object_symbol_ids},
         std::span{extra_symbol_ids},
         relation_to_symbols,
         symbol_to_relations
      );
   }
}

}  // namespace mifrost
