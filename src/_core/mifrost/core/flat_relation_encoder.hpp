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
#include <string>
#include <vector>

#include "batch_builder.hpp"
#include "common_types.hpp"
#include "goal_inputs.hpp"
#include "relation_dict.hpp"
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
      std::set< TargetSource > target_sources = {};
      std::string target_symbol_prefix = std::string(kDefaultTargetSymbolPrefix);
      std::set< GoalSatisfaction > goal_satisfaction_derivations = {
         GoalSatisfaction::satisfied,
      };
   };

   struct TargetEntityKey {
      TargetSource source = TargetSource::Actions;
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
   class GoalSatisfactionComponent;
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
      std::vector< std::string >* batch_target_names = nullptr
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
    target_sources,
    target_symbol_prefix,
    goal_satisfaction_derivations)
)

}  // namespace mifrost
