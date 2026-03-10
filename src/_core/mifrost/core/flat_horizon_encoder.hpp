#pragma once

#include <boost/describe.hpp>
#include <map>
#include <mimir/formalism/domain.hpp>
#include <mimir/search/state.hpp>
#include <optional>
#include <set>
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
#include "transition_dag.hpp"

namespace mifrost {

namespace batch_input {
namespace parsed {
struct HorizonBatchInputs;
}
}  // namespace batch_input

class FlatHorizonEncoderEngine {
  public:
   enum class Mode {
      Full,
      Delta,
      Action,
   };

   struct Config {
      size_t max_goal_level = 0;
      bool support_literals = false;
      bool include_static = true;
      bool export_node_names = true;
      bool ignore_zero_arity_relations = true;
      bool ignore_actions = true;
      bool include_lgan_edges = false;
      Mode transition_mode = Mode::Full;
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
      bool exclude_root_candidate = true;
      std::set< GoalSatisfaction > goal_satisfaction_derivations = {
         GoalSatisfaction::satisfied,
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

   void encode(
      const mimir::search::State& root,
      const TransitionDAG& dag,
      const GoalInputs& goals,
      BatchBuilder& builder
   );

   BatchBuilder::BatchEncoding encode_batch(const batch_input::parsed::HorizonBatchInputs& inputs);

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

   struct PredicateSpec {
      std::string name;
      int arity = 0;
   };

   struct EncodingContext {
      hash_map< int64_t, int64_t > entity_index_by_object_id;
      hash_map< int64_t, int64_t > state_entity_index_by_node_index;
      std::vector< std::string > entity_names;
      std::vector< std::string > object_names;
      std::vector< int64_t > object_indices;
      std::vector< int64_t > target_entity_indices;
      std::vector< int64_t > target_entity_group_ids;
      TargetColumns target_columns;
   };

  private:
   void initialize_from_domain();
   void prepare_builder(BatchBuilder& builder) const;
   void encode_impl(
      const mimir::search::State& root,
      const TransitionDAG& dag,
      const GoalInputs& goals,
      BatchBuilder& builder,
      std::vector< std::string >* batch_target_names = nullptr
   );
   [[nodiscard]] EncodingContext
   make_context(const mimir::search::State& root, const TransitionDAG& dag) const;
   [[nodiscard]] int relation_id_for(const std::string& name) const;
   [[nodiscard]] int64_t
   state_entity_index_for(const EncodingContext& context, int64_t node_index) const;
   [[nodiscard]] std::string target_node_name(int idx) const;

   mimir::formalism::Domain domain_holder_;
   const mimir::formalism::DomainImpl& domain_;
   Config config_;
   RelationDict relation_dict_;
   std::vector< PredicateSpec > predicate_specs_;
   std::vector< PredicateSpec > regular_predicate_specs_;
   std::vector< PredicateSpec > action_specs_;
   std::vector< std::string > relation_names_;
   std::vector< int64_t > relation_arities_;
   std::vector< std::string > relation_sources_;
   hash_map< std::string, int > relation_name_to_id_;
   std::vector< std::string > target_entity_group_names_;
   std::vector< std::string > target_metadata_group_names_;
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
    exclude_root_candidate,
    goal_satisfaction_derivations)
)

struct FlatHorizonStepInput {
   const mimir::search::State* root = nullptr;
   const TransitionDAG* dag = nullptr;
   const GoalInputs* goals = nullptr;
};

class FlatHorizonStreamEncoder:
    public StreamEncoderBase< FlatHorizonStreamEncoder, FlatHorizonStepInput > {
  public:
   static constexpr std::string_view graph_kind() { return "homo"; }

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
      engine_->encode(*step.root, *step.dag, *step.goals, builder);
   }

  private:
   FlatHorizonEncoderEngine* engine_ = nullptr;
};

}  // namespace mifrost
