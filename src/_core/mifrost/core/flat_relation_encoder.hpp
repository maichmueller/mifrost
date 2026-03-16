#pragma once

#include <ankerl/unordered_dense.h>

#include <boost/describe.hpp>
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

#include "batch_builder.hpp"
#include "common_types.hpp"
#include "default_relations.hpp"
#include "goal_inputs.hpp"
#include "relation_dict.hpp"
#include "stream_encoder_base.hpp"
#include "target_metadata.hpp"
#include "target_source.hpp"

namespace mifrost {

namespace batch_input {
namespace parsed {
struct FlatBatchInputs;
}
}  // namespace batch_input

class FlatRelationEncoderEngine {
  public:
   using HistorySubgoal = std::pair< int, std::vector< LiteralVariant > >;

   struct Config {
      size_t max_goal_level = 0;
      bool support_literals = false;
      bool include_static = true;
      bool export_node_names = true;
      bool ignore_zero_arity_relations = true;
      bool include_lgan_edges = false;
      /// Extra rows that may anchor LGAN edges without becoming prediction targets.
      std::set< TargetSource > lgan_anchor_sources = {};
      /// Which semantic sources should produce target metadata rows.
      /// `Actions` = explicit grounded actions, `Goals` = root-goal literals,
      /// `Subgoals` = layered goal literals, `History` = history literals.
      /// `States` belongs to the horizon/transition flat lanes instead.
      std::set< TargetSource > target_sources = {};
      std::string target_symbol_prefix = std::string(kDefaultTargetSymbolPrefix);
      std::string lgan_tn_edge_pos = defaults::lgan_tn_edge_pos;
      std::string lgan_nn_edge_pos = defaults::lgan_nn_edge_pos;
      std::string lgan_rr_edge_pos = defaults::lgan_rr_edge_pos;
      std::set< GoalDerivation > goal_derivations = {
         GoalDerivation::plain,
         GoalDerivation::satisfied,
      };
   };

   struct TargetEntityKey {
      TargetSource source = TargetSource::actions;
      int64_t discriminator = 0;
      int64_t primary = 0;
      int64_t secondary = 0;
      int64_t tertiary = 0;
      int64_t quaternary = 0;

      auto operator==(const TargetEntityKey& other) const -> bool = default;
   };

   struct TargetEntityKeyHash {
      using is_avalanching = void;

      [[nodiscard]] auto operator()(const TargetEntityKey& key) const noexcept -> uint64_t;
   };

   struct EncodingContext {
      struct HistoryEntry {
         int dt = 0;
         size_t entry_idx = 0;
         int64_t entity_index = -1;
         std::vector< LiteralVariant > literals;
      };

      hash_map< int64_t, int64_t > entity_index_by_object_id;
      hash_map< TargetEntityKey, int64_t, TargetEntityKeyHash, std::equal_to< TargetEntityKey > >
         target_entity_index_by_key;
      std::vector< std::string > entity_names;
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

   explicit FlatRelationEncoderEngine(const mimir::formalism::DomainImpl& domain);
   FlatRelationEncoderEngine(const mimir::formalism::DomainImpl& domain, Config config);
   explicit FlatRelationEncoderEngine(mimir::formalism::Domain domain);
   FlatRelationEncoderEngine(mimir::formalism::Domain domain, Config config);
   FlatRelationEncoderEngine(const FlatRelationEncoderEngine&) = delete;
   FlatRelationEncoderEngine& operator=(const FlatRelationEncoderEngine&) = delete;
   FlatRelationEncoderEngine(FlatRelationEncoderEngine&&) = delete;
   FlatRelationEncoderEngine& operator=(FlatRelationEncoderEngine&&) = delete;
   ~FlatRelationEncoderEngine();

   void encode(const mimir::search::State& state, BatchBuilder& builder);
   void encode(
      const mimir::search::State& state,
      std::span< const mimir::formalism::GroundAction > actions,
      BatchBuilder& builder
   );
   void encode(const mimir::search::State& state, const GoalInputs& goals, BatchBuilder& builder);
   void encode(
      const mimir::search::State& state,
      const GoalInputs& goals,
      std::span< const mimir::formalism::GroundAction > actions,
      BatchBuilder& builder
   );
   void encode(
      const mimir::search::State& state,
      const GoalInputs& goals,
      std::span< const mimir::formalism::GroundAction > actions,
      std::span< const HistorySubgoal > history_subgoals,
      std::optional< int > history_max_steps,
      BatchBuilder& builder
   );
   BatchBuilder::BatchEncoding encode_batch(
      const batch_input::parsed::FlatBatchInputs& inputs,
      std::optional< int > history_max_steps = std::nullopt
   );

   [[nodiscard]] const Config& get_config() const { return config_; }
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
   hash_map< std::string, int > relation_name_to_id_;
   std::vector< std::string > target_entity_group_names_;
   std::map< TargetSource, int64_t > target_entity_group_ids_;
   std::vector< std::string > target_metadata_group_names_;
   std::map< TargetSource, int64_t > target_metadata_group_ids_;
};

BOOST_DESCRIBE_STRUCT(
   FlatRelationEncoderEngine::Config,
   (),
   (max_goal_level,
    support_literals,
    include_static,
    export_node_names,
    ignore_zero_arity_relations,
    include_lgan_edges,
    lgan_anchor_sources,
    target_sources,
    target_symbol_prefix,
    lgan_tn_edge_pos,
    lgan_nn_edge_pos,
    lgan_rr_edge_pos,
    goal_derivations)
)

struct FlatRelationStepInput {
   const mimir::search::State* state = nullptr;
   const GoalInputs* goals = nullptr;
   const std::vector< mimir::formalism::GroundAction >* actions = nullptr;
   const std::vector< FlatRelationEncoderEngine::HistorySubgoal >* history_subgoals = nullptr;
   std::optional< int > history_max_steps = std::nullopt;
};

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

   BatchEncoding flush() { return builder_.build(); }

   nb::object flush_pyg() { return builder_.build_pyg(); }

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
