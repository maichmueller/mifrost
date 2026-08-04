/**
 * @file flat_horizon_encoder.hpp
 * @brief Pymimir flat horizon compatibility encoder for transition DAGs.
 *
 * This encoder reuses the flat tuple, node-table, and schema helpers, but it
 * builds state rows, topology relations, and root handling for a transition DAG.
 */
#pragma once

#include <boost/describe.hpp>
#include <map>
#include <memory>
#include <mimir/formalism/domain.hpp>
#include <mimir/search/state.hpp>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "flat_entity_context.hpp"
#include "mifrost/backends/pymimir/encoders/common/goal_inputs.hpp"
#include "mifrost/backends/pymimir/encoders/common/relation_dict.hpp"
#include "mifrost/backends/pymimir/encoders/common/transition_dag.hpp"
#include "mifrost/core/batch_builder.hpp"
#include "mifrost/core/common_types.hpp"
#include "mifrost/core/encoders/common/default_relations.hpp"
#include "mifrost/core/encoders/common/root_policy.hpp"
#include "mifrost/core/encoders/common/stream_encoder_base.hpp"
#include "mifrost/core/encoders/common/target_metadata.hpp"
#include "mifrost/core/encoders/common/target_source.hpp"
#include "mifrost/core/encoders/flat/semantic_flat_horizon_encoder.hpp"

namespace mifrost {

namespace batch_input {
namespace parsed {
struct HorizonBatchInputs;
}
}  // namespace batch_input

/**
 * @brief Encodes a transition DAG into flat tuples.
 *
 * Predicate facts still use logical predicate arity, but encoded tuples may
 * add a leading state slot and an optional predicate virtual node slot.
 */
class MIFROST_API FlatHorizonEncoderEngine {
  public:
   /// How non-root transition nodes are represented.
   enum class Mode {
      full,
      delta,
      action,
   };

   /// Runtime configuration for the flat horizon encoder.
   struct Config {
      size_t max_goal_level = 0;
      bool support_literals = false;
      bool include_static = true;
      bool export_node_names = true;
      bool ignore_zero_arity_relations = true;
      bool ignore_actions = true;
      bool use_predicate_virtual_nodes = false;
      bool include_lgan_edges = false;
      Mode transition_mode = Mode::full;
      std::string target_symbol_prefix = std::string(kDefaultTargetSymbolPrefix);
      std::string parent_relation = defaults::parent_relation;
      std::string sibling_relation = defaults::sibling_relation;
      std::string cousin_relation = defaults::cousin_relation;
      std::string lgan_tn_edge_pos = defaults::lgan_tn_edge_pos;
      std::string lgan_nn_edge_pos = defaults::lgan_nn_edge_pos;
      std::string lgan_rr_edge_pos = defaults::lgan_rr_edge_pos;
      bool enable_parent_relation = false;
      bool enable_sibling_relation = false;
      bool enable_cousin_relation = false;
      RootPolicy root_policy = RootPolicy::exclude;
      bool pack_relation_args_relation_major = false;
      std::set< GoalDerivation > goal_derivations = {
         GoalDerivation::plain,
         GoalDerivation::satisfied,
      };
   };

   explicit FlatHorizonEncoderEngine(const mimir::formalism::DomainImpl& domain);
   FlatHorizonEncoderEngine(const mimir::formalism::DomainImpl& domain, Config config);
   explicit FlatHorizonEncoderEngine(mimir::formalism::Domain domain);
   FlatHorizonEncoderEngine(mimir::formalism::Domain domain, Config config);
   FlatHorizonEncoderEngine(const FlatHorizonEncoderEngine&) = delete;
   FlatHorizonEncoderEngine& operator=(const FlatHorizonEncoderEngine&) = delete;
   FlatHorizonEncoderEngine(FlatHorizonEncoderEngine&&) = delete;
   FlatHorizonEncoderEngine& operator=(FlatHorizonEncoderEngine&&) = delete;
   ~FlatHorizonEncoderEngine();

   /// Encode one root state plus transition DAG into a flat graph.
   void encode(
      const mimir::search::State& root,
      const TransitionDAG& dag,
      const GoalInputs& goals,
      BatchBuilder& builder
   );

   /// Encode a parsed batch of horizon inputs into one flat batch encoding.
   BatchBuilder::BatchEncoding encode_batch(const batch_input::parsed::HorizonBatchInputs& inputs);
   /// Apply configured batch-level post-processing to a built flat batch.
   void finalize_batch_encoding(BatchBuilder::BatchEncoding& encoding) const;

   [[nodiscard]] const Config& get_config() const { return config_; }
   [[nodiscard]] const RelationDict& get_relation_dict() const;
   [[nodiscard]] const std::vector< std::string >& get_relation_names() const;
   [[nodiscard]] const std::vector< int64_t >& get_relation_arities() const;
   [[nodiscard]] const std::vector< std::string >& get_relation_sources() const;
   [[nodiscard]] const std::vector< int64_t >& get_relation_logical_arities() const;
   [[nodiscard]] const std::vector< int64_t >& get_relation_encoded_arities() const;
   [[nodiscard]] const std::vector< int64_t >& get_relation_slot_roles() const;
   [[nodiscard]] const std::vector< int64_t >& get_relation_slot_role_offsets() const;
   [[nodiscard]] const std::vector< std::string >& get_slot_role_names() const;

   /**
    * @brief Per-graph flat entity state for horizon encoding.
    *
    * State rows and predicate virtual nodes are appended after objects and are
    * distinguished through `entity_role_ids`.
    */
   struct EncodingContext {
      hash_map< int64_t, int64_t > entity_index_by_object_id;
      hash_map< int64_t, int64_t > state_entity_index_by_node_index;
      hash_map< PredicateSymbolKey, int64_t, PredicateSymbolKeyHash > predicate_entity_index_by_key;
      std::vector< std::string > entity_names;
      std::vector< int64_t > entity_role_ids;
      std::vector< std::string > object_names;
      std::vector< int64_t > object_indices;
      std::vector< int64_t > target_entity_indices;
      std::vector< int64_t > target_entity_group_ids;
      std::vector< mimir::search::State > target_name_states;
      TargetColumns target_columns;
   };

  private:
   struct PredicateSpec {
      std::string name;
      int arity = 0;
   };

   struct SemanticImpl;

   mimir::formalism::Domain domain_holder_;
   const mimir::formalism::DomainImpl& domain_;
   Config config_;
   std::unique_ptr< SemanticImpl > semantic_;
};

BOOST_DESCRIBE_STRUCT(
   FlatHorizonEncoderEngine::Config,
   (),
   (max_goal_level,
    support_literals,
    include_static,
    export_node_names,
    ignore_zero_arity_relations,
    ignore_actions,
    use_predicate_virtual_nodes,
    include_lgan_edges,
    transition_mode,
    target_symbol_prefix,
    parent_relation,
    sibling_relation,
    cousin_relation,
    lgan_tn_edge_pos,
    lgan_nn_edge_pos,
    lgan_rr_edge_pos,
    enable_parent_relation,
    enable_sibling_relation,
    enable_cousin_relation,
    root_policy,
    pack_relation_args_relation_major,
    goal_derivations)
)

struct FlatHorizonStepInput {
   const mimir::search::State* root = nullptr;
   const TransitionDAG* dag = nullptr;
   const GoalInputs* goals = nullptr;
};

/// Streaming wrapper for the flat horizon encoder.
class FlatHorizonStreamEncoder:
    public StreamEncoderBase< FlatHorizonStreamEncoder, FlatHorizonStepInput > {
  public:
   static constexpr std::string_view graph_kind() { return "flat"; }

   explicit FlatHorizonStreamEncoder(FlatHorizonEncoderEngine& engine) : engine_(&engine)
   {
      reset();
   }

   int64_t
   append(const mimir::search::State& root, const TransitionDAG& dag, const GoalInputs& goals)
   {
      FlatHorizonStepInput step;
      step.root = &root;
      step.dag = &dag;
      step.goals = &goals;
      return StreamEncoderBase::append(step);
   }

   int64_t append(const mimir::search::State& root, const GoalInputs& goals)
   {
      TransitionDAG dag(root);
      FlatHorizonStepInput step;
      step.root = &root;
      step.dag = &dag;
      step.goals = &goals;
      return StreamEncoderBase::append(step);
   }

   void update(
      int64_t id,
      const mimir::search::State& root,
      const TransitionDAG& dag,
      const GoalInputs& goals
   )
   {
      FlatHorizonStepInput step;
      step.root = &root;
      step.dag = &dag;
      step.goals = &goals;
      StreamEncoderBase::update(id, step);
   }

   void update(int64_t id, const mimir::search::State& root, const GoalInputs& goals)
   {
      TransitionDAG dag(root);
      FlatHorizonStepInput step;
      step.root = &root;
      step.dag = &dag;
      step.goals = &goals;
      StreamEncoderBase::update(id, step);
   }

   void encode_step(const FlatHorizonStepInput& step, BatchBuilder& builder)
   {
      if(engine_ == nullptr or step.root == nullptr or step.dag == nullptr
         or step.goals == nullptr) {
         throw std::invalid_argument("FlatHorizonStreamEncoder requires root/dag/goals");
      }
      // Streaming mode already stores a complete step payload, so this path is a thin dispatch.
      engine_->encode(*step.root, *step.dag, *step.goals, builder);
   }

   BatchEncoding flush()
   {
      if(engine_ == nullptr) {
         throw std::invalid_argument("FlatHorizonStreamEncoder requires engine");
      }
      auto encoding = StreamEncoderBase::flush();
      engine_->finalize_batch_encoding(encoding);
      return encoding;
   }

  private:
   FlatHorizonEncoderEngine* engine_ = nullptr;
};

}  // namespace mifrost
