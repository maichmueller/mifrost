#pragma once

#include <string>

#include "flat_horizon_encoder.hpp"

namespace mifrost {

struct FlatHorizonContextBuildConfig {
   RootPolicy root_policy = RootPolicy::exclude;
   bool export_node_names = true;
   size_t predicate_symbol_capacity = 0;
   std::string target_symbol_prefix = std::string(kDefaultTargetSymbolPrefix);
   std::string hidden_root_carrier_name = "_root_state_";
};

FlatHorizonEncoderEngine::EncodingContext build_flat_horizon_encoding_context(
   const mimir::search::State& root,
   const TransitionDAG& dag,
   const FlatHorizonContextBuildConfig& config
);

}  // namespace mifrost
