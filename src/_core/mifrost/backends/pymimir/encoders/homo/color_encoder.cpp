/**
 * @file color_encoder.cpp
 * @brief Pymimir homogeneous color encoder implementation.
 */
#include "color_encoder.hpp"

#include <mimir/formalism/problem.hpp>
#include <stdexcept>

#include "mifrost/input_handling/batch_input_parser.hpp"

namespace mifrost {
namespace {

SemanticColorEncoderConfig semantic_config(const ColorEncoderEngine::Config& config)
{
   return SemanticColorEncoderConfig{
      .edge_features = config.edge_features,
      .enable_global_predicate_nodes = config.enable_global_predicate_nodes,
      .export_node_names = true,
   };
}

}  // namespace

ColorEncoderEngine::ColorEncoderEngine(const mimir::formalism::DomainImpl& domain)
    : ColorEncoderEngine(domain, Config{})
{
}

ColorEncoderEngine::ColorEncoderEngine(const mimir::formalism::DomainImpl& domain, Config config)
    : domain_(domain),
      config_(std::move(config)),
      semantic_engine_(
         std::make_unique< SemanticColorEncoderEngine >(
            pymimir::make_semantic_predicates(domain),
            semantic_config(config_)
         )
      )
{
}

ColorEncoderEngine::ColorEncoderEngine(mimir::formalism::Domain domain)
    : ColorEncoderEngine(std::move(domain), Config{})
{
}

ColorEncoderEngine::ColorEncoderEngine(mimir::formalism::Domain domain, Config config)
    : domain_holder_(std::move(domain)),
      domain_(**domain_holder_),
      config_(std::move(config)),
      semantic_engine_(
         std::make_unique< SemanticColorEncoderEngine >(
            pymimir::make_semantic_predicates(domain_),
            semantic_config(config_)
         )
      )
{
}

void ColorEncoderEngine::ensure_problem(const mimir::search::State& state)
{
   const auto* problem = &state.get_problem();
   if(problem_ == problem) {
      return;
   }
   problem_ = problem;
   problem_adapter_ = std::make_unique< pymimir::SemanticProblemAdapter >(*problem);
   semantic_engine_ = std::make_unique< SemanticColorEncoderEngine >(
      problem_adapter_->get_task_context(), semantic_config(config_)
   );
}

void ColorEncoderEngine::encode_state_impl(const mimir::search::State& state, BatchBuilder& builder)
{
   ensure_problem(state);
   const auto state_view = problem_adapter_->make_state_view(state);
   const auto actions = problem_adapter_->make_action_views(
      std::span< const mimir::formalism::GroundAction >{}
   );
   semantic_engine_->encode(state_view, actions, builder);
}

void ColorEncoderEngine::encode_impl(
   const mimir::search::State& state,
   const GoalInputs& goals,
   std::span< const mimir::formalism::GroundAction > actions,
   BatchBuilder& builder
)
{
   if(not actions.empty()) {
      throw std::invalid_argument("ColorEncoderEngine does not support action encoding");
   }
   ensure_problem(state);
   const auto state_view = problem_adapter_->make_state_view(state);
   const auto goal_views = problem_adapter_->make_goal_views(goals);
   const auto action_views = problem_adapter_->make_action_views(actions);
   semantic_engine_->encode(
      state_view, goal_views.goals_view(), goal_views.subgoal_layers_view(), action_views, builder
   );
}

BatchBuilder::BatchEncoding ColorEncoderEngine::encode_batch(
   const batch_input::parsed::ColorBatchInputs& inputs
)
{
   BatchBuilder builder;
   builder.set_graph_kind("homo");

   const size_t state_count = inputs.states.states.size();
   for(size_t idx = 0; idx < state_count; ++idx) {
      const auto& state_entry = inputs.states.states[idx];
      const auto& goals_entry = inputs.goals.at(idx);
      const auto& subgoal_layers_entry = inputs.subgoal_layers.at(idx);

      if(not goals_entry.has_value() and not subgoal_layers_entry.has_value()) {
         encode(state_entry.state, builder);
         builder.next_graph();
         continue;
      }

      GoalInputs goal_inputs;
      if(goals_entry.has_value()) {
         const auto* layers_ptr = subgoal_layers_entry.has_value() ? &(*subgoal_layers_entry)
                                                                   : nullptr;
         goal_inputs = batch_input::compose_goal_inputs(*goals_entry, layers_ptr);
      } else {
         goal_inputs = batch_input::default_goal_inputs_for_batch_state(state_entry);
         if(subgoal_layers_entry.has_value()) {
            size_t level = 1;
            for(const auto& layer : *subgoal_layers_entry) {
               goal_inputs.extend(layer, level);
               ++level;
            }
         }
      }

      encode(state_entry.state, goal_inputs, builder);
      builder.next_graph();
   }

   return builder.build();
}

}  // namespace mifrost
