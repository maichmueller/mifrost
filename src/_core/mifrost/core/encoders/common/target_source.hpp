/**
 * @file target_source.hpp
 * @brief Names for where a target or extra leading node comes from.
 *
 * This header is shared across encoders. It keeps the target-source names and
 * order in one place so metadata and slot-role logic stay consistent.
 */
#pragma once

#include <array>
#include <string_view>

namespace mifrost {

/**
 * @brief Source of a target row or extra leading node.
 *
 * The same source is used in target metadata, flat tuple slots, and LGAN
 * anchor selection.
 */
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
   /// This does not refer to the plain input state itself.
   states,
   /// Time-indexed literals from `history_subgoals=...`.
   /// Use this when past literals should become separate targets.
   history,
};

/// Shared source order used for stable target-group ids and metadata layout.
inline constexpr std::array kCanonicalTargetSourceOrder = {
   TargetSource::goals,
   TargetSource::subgoals,
   TargetSource::actions,
   TargetSource::history,
   TargetSource::states,
};

/**
 * @brief Public group name for one target source.
 *
 * These names are exported on graph metadata and must remain consistent across
 * encoders and batches.
 */
inline std::string_view target_source_group_name(TargetSource source)
{
   switch(source) {
      case TargetSource::actions: return "action";
      case TargetSource::goals: return "goal";
      case TargetSource::subgoals: return "subgoal";
      case TargetSource::states: return "state";
      case TargetSource::history: return "history";
   }
   return "target";
}

}  // namespace mifrost
