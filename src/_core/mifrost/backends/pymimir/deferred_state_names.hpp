/**
 * @file deferred_state_names.hpp
 * @brief Pymimir-only deferred formatting adapter for neutral batch metadata.
 */
#pragma once

#include <memory>
#include <mimir/search/formatter.hpp>
#include <mimir/search/state.hpp>
#include <span>
#include <sstream>
#include <utility>
#include <vector>

#include "mifrost/core/batch_builder.hpp"

namespace mifrost::pymimir_backend {

class DeferredStateNames final: public DeferredStringBatch {
  public:
   explicit DeferredStateNames(std::span< const mimir::search::State > states)
       : states_(states.begin(), states.end())
   {
   }

   [[nodiscard]] std::vector< std::string > materialize() const override
   {
      std::vector< std::string > names;
      names.reserve(states_.size());
      for(const auto& state : states_) {
         std::ostringstream stream;
         stream << state;
         names.push_back(stream.str());
      }
      return names;
   }

  private:
   std::vector< mimir::search::State > states_;
};

inline void
add_deferred_state_names(BatchBuilder& builder, std::span< const mimir::search::State > states)
{
   if(states.empty()) {
      return;
   }
   builder.add_lazy_target_name_batch(std::make_shared< DeferredStateNames >(states));
}

}  // namespace mifrost::pymimir_backend
