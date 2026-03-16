#pragma once

#include <array>
#include <string_view>

namespace mifrost {

enum class TargetSource {
   /// Grounded action candidates from explicit `actions=...` inputs.
   /// Use this when the model should score or read out action choices.
   actions,
   /// Root-goal literals from the main `goals=...` input.
   /// Use this when goal literals themselves are prediction targets.
   goals,
   /// Layered goal literals from `subgoal_layers=...`.
   /// Use this when intermediate subgoals should become separate targets.
   subgoals,
   /// Candidate states from horizon or transition DAG encoders.
   /// This does not refer to the plain input state on the main state lanes.
   states,
   /// Time-indexed literals from `history_subgoals=...`.
   /// Use this when past literals should become separate targets.
   History,
};

/// Shared source order used for stable target-group ids and metadata layout.
inline constexpr std::array kCanonicalTargetSourceOrder = {
   TargetSource::goals,
   TargetSource::subgoals,
   TargetSource::actions,
   TargetSource::History,
   TargetSource::states,
};

inline std::string_view target_source_group_name(TargetSource source)
{
   switch(source) {
      case TargetSource::actions: return "action";
      case TargetSource::goals: return "goal";
      case TargetSource::subgoals: return "subgoal";
      case TargetSource::states: return "state";
      case TargetSource::History: return "history";
   }
   return "target";
}

}  // namespace mifrost
