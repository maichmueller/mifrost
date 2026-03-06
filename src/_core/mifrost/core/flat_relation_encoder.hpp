#pragma once

#include <boost/describe.hpp>
#include <memory>
#include <mimir/formalism/domain.hpp>
#include <mimir/search/state.hpp>
#include <set>
#include <string>
#include <vector>

#include "batch_builder.hpp"
#include "common_types.hpp"
#include "goal_inputs.hpp"
#include "relation_dict.hpp"

namespace mifrost {

namespace batch_input {
namespace parsed {
struct ColorBatchInputs;
}
}  // namespace batch_input

class FlatRelationEncoderEngine {
  public:
   struct Config {
      size_t max_goal_level = 0;
      bool support_literals = false;
      bool include_static = true;
      bool export_node_names = true;
      bool ignore_zero_arity_relations = true;
      std::set< GoalSatisfaction > goal_satisfaction_derivations = {
         GoalSatisfaction::satisfied,
      };
   };

   struct EncodingContext {
      hash_map< int64_t, int64_t > entity_index_by_object_id;
      std::vector< std::string > entity_names;
      std::vector< int64_t > object_indices;
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
   void encode(const mimir::search::State& state, const GoalInputs& goals, BatchBuilder& builder);
   BatchBuilder::BatchEncoding encode_batch(const batch_input::parsed::ColorBatchInputs& inputs);

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

   void initialize_from_domain();
   void rebuild_schema();
   void prepare_builder(BatchBuilder& builder) const;
   void encode_default_goals(const mimir::search::State& state, BatchBuilder& builder);
   void
   encode_impl(const mimir::search::State& state, const GoalInputs& goals, BatchBuilder& builder);
   EncodingContext make_context(const mimir::search::State& state) const;
   int relation_id_for(const std::string& name) const;

   mimir::formalism::Domain domain_holder_;
   const mimir::formalism::DomainImpl& domain_;
   Config config_;
   RelationDict relation_dict_;
   std::vector< PredicateSpec > predicate_specs_;
   std::vector< PredicateSpec > regular_predicate_specs_;
   std::vector< std::unique_ptr< RelationComponent > > components_;
   std::vector< std::string > relation_names_;
   std::vector< int64_t > relation_arities_;
   std::vector< std::string > relation_sources_;
   hash_map< std::string, int > relation_name_to_id_;
};

BOOST_DESCRIBE_STRUCT(
   FlatRelationEncoderEngine::Config,
   (),
   (max_goal_level,
    support_literals,
    include_static,
    export_node_names,
    ignore_zero_arity_relations,
    goal_satisfaction_derivations)
)

}  // namespace mifrost
