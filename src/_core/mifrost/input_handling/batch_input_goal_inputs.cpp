#include <mimir/formalism/ground_literal.hpp>

#include "mifrost/input_handling/batch_input_parser.hpp"

namespace mifrost::batch_input {

GoalInputs compose_goal_inputs(
   const parsed::GoalPayload& goals,
   const parsed::SubgoalLayersPayload* subgoal_layers
)
{
   GoalInputs inputs;
   inputs.extend(goals, 0);
   if(subgoal_layers != nullptr) {
      size_t depth = 1;
      for(const auto& layer : *subgoal_layers) {
         inputs.extend(layer, depth);
         ++depth;
      }
   }
   return inputs;
}

GoalInputs default_goal_inputs_for_batch_state(const parsed::StateEntry& state_entry)
{
   GoalInputs inputs;
   const auto& problem = state_entry.state.get_problem();
   for(const auto& goal : problem.get_goal_literals< mimir::formalism::StaticTag >()) {
      inputs.append(goal, 0);
   }
   for(const auto& goal : problem.get_goal_literals< mimir::formalism::FluentTag >()) {
      inputs.append(goal, 0);
   }
   for(const auto& goal : problem.get_goal_literals< mimir::formalism::DerivedTag >()) {
      inputs.append(goal, 0);
   }
   return inputs;
}

}  // namespace mifrost::batch_input