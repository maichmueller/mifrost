#pragma once

#include <nanobind/nanobind.h>

#include <mimir/formalism/ground_action.hpp>
#include <mimir/search/state.hpp>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include "mifrost/core/api.hpp"
#include "mifrost/core/common_types.hpp"
#include "mifrost/core/goal_inputs.hpp"
#include "mifrost/core/transition_dag.hpp"

namespace mifrost::batch_input {

namespace nb = nanobind;

template < typename T >
class BatchParam {
  public:
   enum class Kind { None, Shared, PerState };

   static BatchParam none() { return BatchParam(); }

   static BatchParam shared(T value)
   {
      BatchParam out;
      out.kind_ = Kind::Shared;
      out.shared_ = std::move(value);
      return out;
   }

   static BatchParam per_state(std::vector< std::optional< T > > values)
   {
      BatchParam out;
      out.kind_ = Kind::PerState;
      out.per_state_ = std::move(values);
      return out;
   }

   [[nodiscard]] Kind kind() const { return kind_; }

   [[nodiscard]] bool is_none() const { return kind_ == Kind::None; }

   [[nodiscard]] bool is_shared() const { return kind_ == Kind::Shared; }

   [[nodiscard]] bool is_per_state() const { return kind_ == Kind::PerState; }

   [[nodiscard]] const std::optional< T >& shared() const { return shared_; }

   [[nodiscard]] const std::vector< std::optional< T > >& per_state() const { return per_state_; }

   [[nodiscard]] const std::optional< T >& at(size_t idx) const
   {
      static const std::optional< T > kNone = std::nullopt;
      switch(kind_) {
         case Kind::Shared: return shared_;
         case Kind::PerState: return per_state_.at(idx);
         case Kind::None:
         default: return kNone;
      }
   }

  private:
   Kind kind_ = Kind::None;
   std::optional< T > shared_;
   std::vector< std::optional< T > > per_state_;
};

namespace parsed {

struct MIFROST_LOCAL StateEntry {
   nb::object source;
   mimir::search::State state;
};

struct MIFROST_LOCAL StateBatch {
   std::vector< StateEntry > states;
};

using GoalPayload = std::vector< LiteralVariant >;
using ActionPayload = std::vector< mimir::formalism::GroundAction >;
using SubgoalLayersPayload = std::vector< std::vector< LiteralVariant > >;
using HistorySubgoal = std::pair< int, std::vector< LiteralVariant > >;
using HistoryPayload = std::vector< HistorySubgoal >;

using GoalBatch = BatchParam< GoalPayload >;
using ActionBatch = BatchParam< ActionPayload >;
using SubgoalLayersBatch = BatchParam< SubgoalLayersPayload >;
using HistoryBatch = BatchParam< HistoryPayload >;
using DagBatch = BatchParam< TransitionDAG >;
using SuccessorBatch = BatchParam< StateEntry >;

struct MIFROST_LOCAL HGraphBatchInputs {
   StateBatch states;
   GoalBatch goals;
   ActionBatch actions;
   SubgoalLayersBatch subgoal_layers;
   HistoryBatch history_subgoals;
};

struct MIFROST_LOCAL ColorBatchInputs {
   StateBatch states;
   GoalBatch goals;
   SubgoalLayersBatch subgoal_layers;
};

struct MIFROST_LOCAL FlatBatchInputs {
   StateBatch states;
   GoalBatch goals;
   ActionBatch actions;
   SubgoalLayersBatch subgoal_layers;
};

struct MIFROST_LOCAL SuccessorBatchInputs {
   StateBatch states;
   SuccessorBatch successors;
   GoalBatch goals;
   SubgoalLayersBatch subgoal_layers;
};

struct MIFROST_LOCAL HorizonBatchInputs {
   StateBatch roots;
   DagBatch dags;
   GoalBatch goals;
   SubgoalLayersBatch subgoal_layers;
};

}  // namespace parsed

MIFROST_LOCAL parsed::StateBatch
parse_states_batch_param(nb::handle states, std::string_view field_name = "states");

MIFROST_LOCAL parsed::GoalBatch parse_goals_batch_param(nb::handle goals, size_t state_count);

MIFROST_LOCAL parsed::ActionBatch parse_actions_batch_param(nb::handle actions, size_t state_count);

MIFROST_LOCAL parsed::SubgoalLayersBatch
parse_subgoal_layers_batch_param(nb::handle subgoal_layers, size_t state_count);

MIFROST_LOCAL parsed::HistoryBatch
parse_history_subgoals_batch_param(nb::handle history_subgoals, size_t state_count);

MIFROST_LOCAL parsed::SuccessorBatch
parse_successors_batch_param(nb::handle successors, size_t state_count);

MIFROST_LOCAL parsed::DagBatch parse_dags_batch_param(nb::handle dags, size_t state_count);

MIFROST_LOCAL GoalInputs compose_goal_inputs(
   const parsed::GoalPayload& goals,
   const parsed::SubgoalLayersPayload* subgoal_layers = nullptr
);

MIFROST_LOCAL GoalInputs default_goal_inputs_for_batch_state(const parsed::StateEntry& state_entry);

MIFROST_LOCAL parsed::HGraphBatchInputs parse_hgraph_batch_inputs(
   nb::handle states,
   nb::handle goals,
   nb::handle actions,
   nb::handle subgoal_layers,
   nb::handle history_subgoals
);

MIFROST_LOCAL parsed::ColorBatchInputs parse_color_batch_inputs(
   nb::handle states,
   nb::handle goals,
   nb::handle actions,
   nb::handle subgoal_layers
);

MIFROST_LOCAL parsed::FlatBatchInputs parse_flat_batch_inputs(
   nb::handle states,
   nb::handle goals,
   nb::handle actions,
   nb::handle subgoal_layers
);

MIFROST_LOCAL parsed::SuccessorBatchInputs parse_successor_batch_inputs(
   nb::handle states,
   nb::handle successors,
   nb::handle goals,
   nb::handle actions,
   nb::handle subgoal_layers,
   nb::handle history_subgoals,
   std::optional< int > history_max_steps
);

MIFROST_LOCAL parsed::HorizonBatchInputs parse_horizon_batch_inputs(
   nb::handle roots,
   nb::handle dags,
   nb::handle goals,
   nb::handle actions,
   nb::handle subgoal_layers,
   nb::handle history_subgoals,
   std::optional< int > history_max_steps
);

MIFROST_LOCAL nb::list parse_states_batch_python(nb::handle states);
MIFROST_LOCAL nb::tuple parse_goals_batch_param_python(nb::handle goals, size_t state_count);
MIFROST_LOCAL nb::tuple parse_actions_batch_param_python(nb::handle actions, size_t state_count);
MIFROST_LOCAL nb::tuple
parse_subgoal_layers_batch_param_python(nb::handle subgoal_layers, size_t state_count);
MIFROST_LOCAL nb::tuple
parse_history_subgoals_batch_param_python(nb::handle history_subgoals, size_t state_count);
MIFROST_LOCAL nb::list
parse_successors_batch_param_python(nb::handle successors, size_t state_count);
MIFROST_LOCAL nb::list parse_dags_batch_param_python(nb::handle dags, size_t state_count);
MIFROST_LOCAL nb::tuple parse_ilg_batch_inputs_python(
   nb::handle states,
   nb::handle goals,
   nb::handle actions,
   nb::handle subgoal_layers
);

}  // namespace mifrost::batch_input
