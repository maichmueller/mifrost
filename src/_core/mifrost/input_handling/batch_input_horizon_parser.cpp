#include "batch_input_parser.hpp"

namespace mifrost::batch_input {

namespace {

template < typename T >
bool batch_param_has_non_empty_entries(const BatchParam< T >& param)
{
   if(param.is_shared()) {
      return param.shared().has_value() and not param.shared()->empty();
   }
   if(param.is_per_state()) {
      for(const auto& entry : param.per_state()) {
         if(entry.has_value() and not entry->empty()) {
            return true;
         }
      }
   }
   return false;
}

}  // namespace

parsed::HorizonBatchInputs parse_horizon_batch_inputs(
   nb::handle roots,
   nb::handle dags,
   nb::handle goals,
   nb::handle actions,
   nb::handle subgoal_layers,
   nb::handle history_subgoals,
   std::optional< int > history_max_steps
)
{
   parsed::HorizonBatchInputs out;
   out.roots = parse_states_batch_param(roots, "states");
   const size_t state_count = out.roots.states.size();
   const auto parsed_actions = parse_actions_batch_param(actions, state_count);
   if(batch_param_has_non_empty_entries(parsed_actions)) {
      throw std::invalid_argument(
         "Horizon batch encoding does not support explicit action payloads"
      );
   }
   const auto parsed_history = parse_history_subgoals_batch_param(history_subgoals, state_count);
   if(batch_param_has_non_empty_entries(parsed_history) or history_max_steps.has_value()) {
      throw std::invalid_argument(
         "Horizon batch encoding does not support history_subgoals payloads"
      );
   }
   out.dags = parse_dags_batch_param(dags, state_count);
   out.goals = parse_goals_batch_param(goals, state_count);
   out.subgoal_layers = parse_subgoal_layers_batch_param(subgoal_layers, state_count);
   return out;
}

}  // namespace mifrost::batch_input
