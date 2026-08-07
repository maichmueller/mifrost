#include "successor_hgraph_encoder.hpp"

#include <stdexcept>
#include <utility>

#include "mifrost/input_handling/batch_input_parser.hpp"

namespace mifrost {
namespace {

HGraphEncoderEngine::Config base_config(const SuccessorHGraphEncoderEngine::Config& config)
{
   HGraphEncoderEngine::Config result = config;
   if(config.successor_mode == SuccessorHGraphEncoderEngine::Mode::delta) {
      result.support_literals = true;
   }
   return result;
}

}  // namespace

SemanticSuccessorHGraphEncoderConfig SuccessorHGraphEncoderEngine::semantic_config(
   const Config& config
)
{
   SemanticSuccessorHGraphEncoderConfig result;
   const auto base = base_config(config);
   result.symbol_type_id = base.symbol_type_id;
   result.target_symbol_prefix = base.target_symbol_prefix;
   result.nullary_object_name = base.nullary_object_name;
   result.lgan_tn_edge_pos = base.lgan_tn_edge_pos;
   result.lgan_nn_edge_pos = base.lgan_nn_edge_pos;
   result.lgan_rr_edge_pos = base.lgan_rr_edge_pos;
   result.history_link_relation = base.history_link_relation;
   result.max_goal_level = base.max_goal_level;
   result.support_literals = base.support_literals;
   result.add_nullary_predicates = base.add_nullary_predicates;
   result.ignore_actions = base.ignore_actions;
   result.include_lgan_edges = base.include_lgan_edges;
   result.include_static = base.include_static;
   result.include_empty_edge_types = base.include_empty_edge_types;
   result.export_node_names = base.export_node_names;
   result.allow_subgoal_layers_beyond_max_goal_level = true;
   result.lgan_anchor_sources = base.lgan_anchor_sources;
   result.target_sources = base.target_sources;
   result.goal_derivations = base.goal_derivations;
   result.successor_mode = config.successor_mode == Mode::delta ? SemanticSuccessorMode::delta
                                                                : SemanticSuccessorMode::full;
   result.successor_suffix = config.successor_suffix;
   result.include_successor_goal_satisfaction = config.include_successor_goal_satisfaction;
   return result;
}

SuccessorHGraphEncoderEngine::SuccessorHGraphEncoderEngine(
   const mimir::formalism::DomainImpl& domain
)
    : SuccessorHGraphEncoderEngine(domain, Config{})
{
}

SuccessorHGraphEncoderEngine::SuccessorHGraphEncoderEngine(
   const mimir::formalism::DomainImpl& domain,
   Config config
)
    : HGraphEncoderEngine(domain, base_config(config)),
      successor_config_(std::move(config)),
      semantic_successor_(
         std::make_unique< SemanticSuccessorHGraphEncoderEngine >(
            schema_.predicates,
            schema_.actions,
            semantic_config(successor_config_)
         )
      )
{
}

SuccessorHGraphEncoderEngine::SuccessorHGraphEncoderEngine(mimir::formalism::Domain domain)
    : SuccessorHGraphEncoderEngine(std::move(domain), Config{})
{
}

SuccessorHGraphEncoderEngine::SuccessorHGraphEncoderEngine(
   mimir::formalism::Domain domain,
   Config config
)
    : HGraphEncoderEngine(std::move(domain), base_config(config)),
      successor_config_(std::move(config)),
      semantic_successor_(
         std::make_unique< SemanticSuccessorHGraphEncoderEngine >(
            schema_.predicates,
            schema_.actions,
            semantic_config(successor_config_)
         )
      )
{
}

void SuccessorHGraphEncoderEngine::encode(
   const mimir::search::State& current,
   const mimir::search::State& successor,
   const GoalInputs& goals,
   BatchBuilder& builder
)
{
   if(&current.get_problem() != &successor.get_problem()) {
      throw std::invalid_argument(
         "SuccessorHGraphEncoder states must belong to the same planning problem"
      );
   }
   // The engine is built from the domain and stays put: it is the problem
   // context that changes per transition, and `update_relations()` therefore
   // can no longer be discarded by a rebuild.
   auto& adapter = problem_adapter(current);
   const auto problem_context_value = adapter.get_problem_context();
   const auto current_view = adapter.make_state_view(current);
   const auto successor_view = adapter.make_state_view(successor);
   const auto empty_actions = adapter.make_action_views(
      std::span< const mimir::formalism::GroundAction >{}
   );
   const auto goal_views = adapter.make_goal_views(goals);
   semantic_successor_->encode(
      problem_context_value,
      current_view,
      goal_views.goals_view(),
      goal_views.subgoal_layers_view(),
      empty_actions,
      successor_view,
      empty_actions,
      builder
   );
}

void SuccessorHGraphEncoderEngine::update_relations(RelationDict relation_dict)
{
   semantic_relation_arities_ = relation_dict.arity;
   semantic_successor_->update_relations(semantic_relation_arities_);
   HGraphEncoderEngine::update_relations(std::move(relation_dict));
}

BatchBuilder::BatchEncoding SuccessorHGraphEncoderEngine::encode_batch(
   const batch_input::parsed::SuccessorBatchInputs& inputs
)
{
   BatchBuilder builder;
   builder.set_graph_kind("hetero");
   for(size_t index = 0; index < inputs.states.states.size(); ++index) {
      const auto& current = inputs.states.states[index];
      const auto& successor = inputs.successors.at(index);
      const auto goals_payload = inputs.goals.at(index);
      const auto layers_payload = inputs.subgoal_layers.at(index);
      const GoalInputs goals = goals_payload
                                  ? batch_input::compose_goal_inputs(
                                       *goals_payload, layers_payload ? &*layers_payload : nullptr
                                    )
                                  : batch_input::default_goal_inputs_for_batch_state(current);
      encode(current.state, successor->state, goals, builder);
      builder.next_graph();
   }
   return builder.build();
}

int64_t TransitionStreamEncoder::append(
   const mimir::search::State& current,
   const mimir::search::State& successor,
   const GoalInputs& goals
)
{
   return StreamEncoderBase::append(
      TransitionStepInput{.current = &current, .successor = &successor, .goals = &goals}
   );
}

void TransitionStreamEncoder::update(
   int64_t id,
   const mimir::search::State& current,
   const mimir::search::State& successor,
   const GoalInputs& goals
)
{
   StreamEncoderBase::update(
      id, TransitionStepInput{.current = &current, .successor = &successor, .goals = &goals}
   );
}

void TransitionStreamEncoder::encode_step(const TransitionStepInput& step, BatchBuilder& builder)
{
   if(engine_ == nullptr or step.current == nullptr or step.successor == nullptr
      or step.goals == nullptr) {
      throw std::invalid_argument("TransitionStreamEncoder requires current/successor/goals");
   }
   engine_->encode(*step.current, *step.successor, *step.goals, builder);
}

}  // namespace mifrost
