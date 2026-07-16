/**
 * @file flat_relation_encoder.hpp
 * @brief Pymimir flat compatibility encoder for states, goals, actions, and history.
 *
 * This is the main entry point for the flat relation encoder. It builds one
 * flat node table plus relation-instance tensors, while helper files handle
 * tuple layout, schema setup, and per-graph setup work.
 */
#pragma once

#include <ankerl/unordered_dense.h>

#include <map>
#include <memory>
#include <mimir/formalism/domain.hpp>
#include <mimir/formalism/ground_action.hpp>
#include <mimir/search/state.hpp>
#include <optional>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "flat_entity_context.hpp"
#include "flat_goal_helpers.hpp"
#include "mifrost/backends/pymimir/encoders/common/goal_inputs.hpp"
#include "mifrost/backends/pymimir/encoders/common/relation_dict.hpp"
#include "mifrost/core/batch_builder.hpp"
#include "mifrost/core/common_types.hpp"
#include "mifrost/core/encoders/common/default_relations.hpp"
#include "mifrost/core/encoders/common/stream_encoder_base.hpp"
#include "mifrost/core/encoders/common/target_metadata.hpp"
#include "mifrost/core/encoders/common/target_source.hpp"
#include "mifrost/core/encoders/flat/flat_relation_config.hpp"
#include "mifrost/core/encoders/flat/flat_relation_schema.hpp"

namespace mifrost {

namespace batch_input {
namespace parsed {
struct FlatBatchInputs;
}
}  // namespace batch_input

/**
 * @brief Encodes a single planning state into flat tuples.
 *
 * Responsibility:
 *  - register and export flat relation metadata
 *  - create the per-graph node table
 *  - emit relation tuples for state facts, goals, actions, and history
 *
 * Non-responsibility:
 *  - it does not invent relation names itself; naming lives in the
 *    shared formatter/schema helpers
 *  - it does not manage batch merging beyond writing into `BatchBuilder`
 *
 * Invariants:
 *  - `relation_arities_` stores encoded arities for backward compatibility
 *  - `relation_logical_arities_` stores the underlying predicate/action arity
 *  - `relation_slot_roles_` / `relation_slot_role_offsets_` stay aligned with
 *    `relation_names_` and `relation_encoded_arities_`
 */
class MIFROST_API FlatRelationEncoderEngine {
  public:
   using HistorySubgoal = std::pair< int, std::vector< LiteralVariant > >;

   /// Backward-compatible engine-local name for the shared semantic config.
   using Config = FlatRelationEncoderConfig;

   explicit FlatRelationEncoderEngine(const mimir::formalism::DomainImpl& domain);
   FlatRelationEncoderEngine(const mimir::formalism::DomainImpl& domain, Config config);
   explicit FlatRelationEncoderEngine(mimir::formalism::Domain domain);
   FlatRelationEncoderEngine(mimir::formalism::Domain domain, Config config);
   FlatRelationEncoderEngine(const FlatRelationEncoderEngine&) = delete;
   FlatRelationEncoderEngine& operator=(const FlatRelationEncoderEngine&) = delete;
   FlatRelationEncoderEngine(FlatRelationEncoderEngine&&) = delete;
   FlatRelationEncoderEngine& operator=(FlatRelationEncoderEngine&&) = delete;
   ~FlatRelationEncoderEngine();

   /// Encode a state with default goal handling.
   void encode(const mimir::search::State& state, BatchBuilder& builder);
   /// Encode a state plus explicit action candidates.
   void encode(
      const mimir::search::State& state,
      std::span< const mimir::formalism::GroundAction > actions,
      BatchBuilder& builder
   );
   /// Encode a state plus explicit goal inputs.
   void encode(const mimir::search::State& state, const GoalInputs& goals, BatchBuilder& builder);
   /// Encode a state plus goals and explicit action candidates.
   void encode(
      const mimir::search::State& state,
      const GoalInputs& goals,
      std::span< const mimir::formalism::GroundAction > actions,
      BatchBuilder& builder
   );
   /// Encode a full relation step including optional history targets.
   void encode(
      const mimir::search::State& state,
      const GoalInputs& goals,
      std::span< const mimir::formalism::GroundAction > actions,
      std::span< const HistorySubgoal > history_subgoals,
      std::optional< int > history_max_steps,
      BatchBuilder& builder
   );
   /// Encode a parsed batch of flat relation inputs.
   BatchBuilder::BatchEncoding encode_batch(
      const batch_input::parsed::FlatBatchInputs& inputs,
      std::optional< int > history_max_steps = std::nullopt
   );
   /// Apply configured batch-level post-processing to a built flat batch.
   void finalize_batch_encoding(BatchBuilder::BatchEncoding& encoding) const;

   /// Return the effective immutable encoder config.
   [[nodiscard]] const Config& get_config() const { return config_; }
   /// Return the exported relation dictionary keyed by relation name.
   [[nodiscard]] const RelationDict& get_relation_dict() const { return relation_dict_; }
   [[nodiscard]] const std::vector< std::string >& get_relation_names() const
   {
      return relation_names_;
   }
   [[nodiscard]] const std::vector< int64_t >& get_relation_arities() const
   {
      return relation_arities_;
   }
   [[nodiscard]] const std::vector< std::string >& get_relation_sources() const
   {
      return relation_sources_;
   }
   [[nodiscard]] const std::vector< int64_t >& get_relation_logical_arities() const
   {
      return relation_logical_arities_;
   }
   [[nodiscard]] const std::vector< int64_t >& get_relation_encoded_arities() const
   {
      return relation_encoded_arities_;
   }
   [[nodiscard]] const std::vector< int64_t >& get_relation_slot_roles() const
   {
      return relation_slot_roles_;
   }
   [[nodiscard]] const std::vector< int64_t >& get_relation_slot_role_offsets() const
   {
      return relation_slot_role_offsets_;
   }
   [[nodiscard]] const std::vector< std::string >& get_slot_role_names() const
   {
      return slot_role_names_;
   }

   /**
    * @brief Shared per-graph state assembled before relation emission.
    *
    * Objects, target rows, history rows, and predicate virtual nodes all live
    * in the same flat node table and are distinguished by `entity_role_ids`.
    */
   struct EncodingContext {
      struct HistoryEntry {
         int dt = 0;
         size_t entry_idx = 0;
         int64_t entity_index = -1;
         std::vector< LiteralVariant > literals;
      };

      hash_map< int64_t, int64_t > entity_index_by_object_id;
      hash_map< PredicateSymbolKey, int64_t, PredicateSymbolKeyHash > predicate_entity_index_by_key;
      hash_map< FlatTargetEntityKey, int64_t, FlatTargetEntityKeyHash > target_entity_index_by_key;
      std::vector< std::string > entity_names;
      std::vector< int64_t > entity_role_ids;
      std::vector< std::string > object_names;
      std::vector< int64_t > object_indices;
      std::vector< int64_t > history_entity_indices;
      std::vector< int64_t > history_entity_dt;
      std::vector< int64_t > target_entity_indices;
      std::vector< int64_t > target_entity_group_ids;
      std::vector< HistoryEntry > history_entries;
      std::vector< mimir::formalism::GroundAction > unique_actions;
      TargetColumns target_columns;
   };

  private:
   struct PredicateSpec {
      std::string name;
      int arity = 0;
   };

   class RelationComponent;
   class StateFactsComponent;
   class GoalFactsComponent;
   class GoalDerivationComponent;
   class GroundActionsComponent;
   class HistoryFactsComponent;

   void validate_config() const;
   void initialize_from_domain();
   void rebuild_schema();
   void prepare_builder(BatchBuilder& builder) const;
   void encode_default_goals(
      const mimir::search::State& state,
      std::span< const mimir::formalism::GroundAction > actions,
      std::span< const HistorySubgoal > history_subgoals,
      std::optional< int > history_max_steps,
      BatchBuilder& builder
   );
   void encode_impl(
      const mimir::search::State& state,
      const GoalInputs& goals,
      std::span< const mimir::formalism::GroundAction > actions,
      std::span< const HistorySubgoal > history_subgoals,
      std::optional< int > history_max_steps,
      BatchBuilder& builder,
      std::vector< std::string >* batch_target_names = nullptr,
      bool prepare_builder = true
   );
   EncodingContext make_context(
      const mimir::search::State& state,
      const GoalInputs& goals,
      std::span< const mimir::formalism::GroundAction > actions,
      std::span< const HistorySubgoal > history_subgoals,
      std::optional< int > history_max_steps
   ) const;
   int relation_id_for(const std::string& name) const;
   [[nodiscard]] bool has_target_source(TargetSource source) const;
   [[nodiscard]] bool has_lgan_anchor_source(TargetSource source) const;
   [[nodiscard]] bool has_anchor_entity_source(TargetSource source) const;
   [[nodiscard]] bool supports_target_metadata() const;
   [[nodiscard]] int64_t target_entity_group_id(TargetSource source) const;
   [[nodiscard]] int64_t target_metadata_group_id(TargetSource source) const;

   mimir::formalism::Domain domain_holder_;
   const mimir::formalism::DomainImpl& domain_;
   Config config_;
   RelationDict relation_dict_;
   std::vector< PredicateSpec > predicate_specs_;
   std::vector< PredicateSpec > regular_predicate_specs_;
   std::vector< PredicateSpec > action_specs_;
   std::vector< std::unique_ptr< RelationComponent > > components_;
   std::vector< std::string > relation_names_;
   std::vector< int64_t > relation_arities_;
   std::vector< std::string > relation_sources_;
   std::vector< int64_t > relation_logical_arities_;
   std::vector< int64_t > relation_encoded_arities_;
   std::vector< int64_t > relation_slot_roles_;
   std::vector< int64_t > relation_slot_role_offsets_;
   std::vector< std::string > slot_role_names_;
   hash_map< std::string, int > relation_name_to_id_;
   std::vector< std::string > target_entity_group_names_;
   std::map< TargetSource, int64_t > target_entity_group_ids_;
   std::vector< std::string > target_metadata_group_names_;
   std::map< TargetSource, int64_t > target_metadata_group_ids_;
};

struct FlatRelationStepInput {
   const mimir::search::State* state = nullptr;
   const GoalInputs* goals = nullptr;
   const std::vector< mimir::formalism::GroundAction >* actions = nullptr;
   const std::vector< FlatRelationEncoderEngine::HistorySubgoal >* history_subgoals = nullptr;
   std::optional< int > history_max_steps = std::nullopt;
};

/**
 * @brief Streaming wrapper for the flat relation encoder.
 *
 * Each appended step becomes one flat graph encoded through the owning engine.
 */
class FlatRelationMutableStreamEncoder:
    public StreamEncoderBase< FlatRelationMutableStreamEncoder, FlatRelationStepInput > {
  public:
   static constexpr std::string_view graph_kind() { return "flat"; }

   explicit FlatRelationMutableStreamEncoder(FlatRelationEncoderEngine& engine) : engine_(&engine)
   {
      reset();
   }

   int64_t append(const mimir::search::State& state)
   {
      FlatRelationStepInput step;
      step.state = &state;
      return StreamEncoderBase::append(step);
   }

   int64_t append(
      const mimir::search::State& state,
      const std::vector< mimir::formalism::GroundAction >& actions
   )
   {
      FlatRelationStepInput step;
      step.state = &state;
      step.actions = &actions;
      return StreamEncoderBase::append(step);
   }

   int64_t append(const mimir::search::State& state, const GoalInputs& goals)
   {
      FlatRelationStepInput step;
      step.state = &state;
      step.goals = &goals;
      return StreamEncoderBase::append(step);
   }

   int64_t append(
      const mimir::search::State& state,
      const GoalInputs& goals,
      const std::vector< mimir::formalism::GroundAction >& actions
   )
   {
      FlatRelationStepInput step;
      step.state = &state;
      step.goals = &goals;
      step.actions = &actions;
      return StreamEncoderBase::append(step);
   }

   int64_t append(
      const mimir::search::State& state,
      const GoalInputs& goals,
      const std::vector< mimir::formalism::GroundAction >& actions,
      const std::vector< FlatRelationEncoderEngine::HistorySubgoal >& history_subgoals,
      std::optional< int > history_max_steps
   )
   {
      FlatRelationStepInput step;
      step.state = &state;
      step.goals = &goals;
      step.actions = &actions;
      step.history_subgoals = &history_subgoals;
      step.history_max_steps = history_max_steps;
      return StreamEncoderBase::append(step);
   }

   void update(int64_t id, const mimir::search::State& state)
   {
      FlatRelationStepInput step;
      step.state = &state;
      StreamEncoderBase::update(id, step);
   }

   void update(
      int64_t id,
      const mimir::search::State& state,
      const std::vector< mimir::formalism::GroundAction >& actions
   )
   {
      FlatRelationStepInput step;
      step.state = &state;
      step.actions = &actions;
      StreamEncoderBase::update(id, step);
   }

   void update(int64_t id, const mimir::search::State& state, const GoalInputs& goals)
   {
      FlatRelationStepInput step;
      step.state = &state;
      step.goals = &goals;
      StreamEncoderBase::update(id, step);
   }

   void update(
      int64_t id,
      const mimir::search::State& state,
      const GoalInputs& goals,
      const std::vector< mimir::formalism::GroundAction >& actions
   )
   {
      FlatRelationStepInput step;
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
      const std::vector< FlatRelationEncoderEngine::HistorySubgoal >& history_subgoals,
      std::optional< int > history_max_steps
   )
   {
      FlatRelationStepInput step;
      step.state = &state;
      step.goals = &goals;
      step.actions = &actions;
      step.history_subgoals = &history_subgoals;
      step.history_max_steps = history_max_steps;
      StreamEncoderBase::update(id, step);
   }

   void encode_step(const FlatRelationStepInput& step, BatchBuilder& builder)
   {
      if(engine_ == nullptr or step.state == nullptr) {
         throw std::invalid_argument("FlatRelationMutableStreamEncoder requires state");
      }

      // Normalize the optional payloads once, then dispatch to the matching engine overload.
      const std::span< const mimir::formalism::GroundAction >
         actions = (step.actions == nullptr)
                      ? std::span< const mimir::formalism::GroundAction >{}
                      : std::span< const mimir::formalism::GroundAction >(*step.actions);
      const std::span< const FlatRelationEncoderEngine::HistorySubgoal >
         history_subgoals = (step.history_subgoals == nullptr)
                               ? std::span< const FlatRelationEncoderEngine::HistorySubgoal >{}
                               : std::span< const FlatRelationEncoderEngine::HistorySubgoal >(
                                    *step.history_subgoals
                                 );

      if(not history_subgoals.empty()) {
         if(step.goals == nullptr) {
            throw std::invalid_argument(
               "FlatRelationMutableStreamEncoder history encoding requires GoalInputs"
            );
         }
         engine_->encode(
            *step.state, *step.goals, actions, history_subgoals, step.history_max_steps, builder
         );
         return;
      }

      if(step.goals != nullptr) {
         engine_->encode(*step.state, *step.goals, actions, builder);
         return;
      }
      if(not actions.empty()) {
         engine_->encode(*step.state, actions, builder);
         return;
      }
      engine_->encode(*step.state, builder);
   }

   BatchEncoding flush()
   {
      if(engine_ == nullptr) {
         throw std::invalid_argument("FlatRelationMutableStreamEncoder requires engine");
      }
      auto encoding = StreamEncoderBase::flush();
      engine_->finalize_batch_encoding(encoding);
      return encoding;
   }

  private:
   FlatRelationEncoderEngine* engine_ = nullptr;
};

class FlatRelationStreamEncoder {
  public:
   static constexpr std::string_view graph_kind() { return "flat"; }

   explicit FlatRelationStreamEncoder(FlatRelationEncoderEngine& engine) : engine_(&engine)
   {
      reset();
   }

   int64_t append(const mimir::search::State& state)
   {
      if(engine_ == nullptr) {
         throw std::invalid_argument("FlatRelationStreamEncoder requires engine");
      }
      engine_->encode(state, builder_);
      builder_.next_graph();
      return next_id_++;
   }

   int64_t append(
      const mimir::search::State& state,
      const std::vector< mimir::formalism::GroundAction >& actions
   )
   {
      if(engine_ == nullptr) {
         throw std::invalid_argument("FlatRelationStreamEncoder requires engine");
      }
      engine_->encode(state, actions, builder_);
      builder_.next_graph();
      return next_id_++;
   }

   int64_t append(const mimir::search::State& state, const GoalInputs& goals)
   {
      if(engine_ == nullptr) {
         throw std::invalid_argument("FlatRelationStreamEncoder requires engine");
      }
      engine_->encode(state, goals, builder_);
      builder_.next_graph();
      return next_id_++;
   }

   int64_t append(
      const mimir::search::State& state,
      const GoalInputs& goals,
      const std::vector< mimir::formalism::GroundAction >& actions
   )
   {
      if(engine_ == nullptr) {
         throw std::invalid_argument("FlatRelationStreamEncoder requires engine");
      }
      engine_->encode(state, goals, actions, builder_);
      builder_.next_graph();
      return next_id_++;
   }

   int64_t append(
      const mimir::search::State& state,
      const GoalInputs& goals,
      const std::vector< mimir::formalism::GroundAction >& actions,
      const std::vector< FlatRelationEncoderEngine::HistorySubgoal >& history_subgoals,
      std::optional< int > history_max_steps
   )
   {
      if(engine_ == nullptr) {
         throw std::invalid_argument("FlatRelationStreamEncoder requires engine");
      }
      engine_->encode(state, goals, actions, history_subgoals, history_max_steps, builder_);
      builder_.next_graph();
      return next_id_++;
   }

   BatchEncoding flush()
   {
      if(engine_ == nullptr) {
         throw std::invalid_argument("FlatRelationStreamEncoder requires engine");
      }
      auto encoding = builder_.build();
      engine_->finalize_batch_encoding(encoding);
      return encoding;
   }

   void reset()
   {
      builder_ = BatchBuilder{};
      builder_.set_graph_kind(std::string(graph_kind()));
      next_id_ = 0;
   }

  private:
   FlatRelationEncoderEngine* engine_ = nullptr;
   BatchBuilder builder_;
   int64_t next_id_ = 0;
};

}  // namespace mifrost
