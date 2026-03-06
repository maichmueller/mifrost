#pragma once

#include <string_view>

namespace mifrost {

enum class TargetSource {
   Actions,
   Goals,
   Subgoals,
   States,
   History,
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
