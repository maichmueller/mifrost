#pragma once

#include <array>
#include <string_view>

namespace mifrost {

enum class TargetSource {
   /// Grounded action candidates from explicit `actions=...` inputs.
   /// Use this when the model should score or read out action choices.
   Actions,
   /// Root-goal literals from the main `goals=...` input.
   /// Use this when goal literals themselves are prediction targets.
   Goals,
   /// Layered goal literals from `subgoal_layers=...`.
   /// Use this when intermediate subgoals should become separate targets.
   Subgoals,
   /// Candidate states from horizon or transition DAG encoders.
   /// This does not refer to the plain input state on the main state lanes.
   States,
   /// Time-indexed literals from `history_subgoals=...`.
   /// Use this when past literals should become separate targets.
   History,
};

/// Shared source order used for stable target-group ids and metadata layout.
inline constexpr std::array kCanonicalTargetSourceOrder = {
   TargetSource::Goals,
   TargetSource::Subgoals,
   TargetSource::Actions,
   TargetSource::History,
   TargetSource::States,
};

inline std::string_view target_source_group_name(TargetSource source)
{
   switch(source) {
      case TargetSource::Actions: return "action";
      case TargetSource::Goals: return "goal";
      case TargetSource::Subgoals: return "subgoal";
      case TargetSource::States: return "state";
      case TargetSource::History: return "history";
   }
   return "target";
}

}  // namespace mifrost
