/**
 * @file flat_horizon_context.cpp
 * @brief Per-graph setup for the Pymimir flat horizon encoder.
 *
 * This file prepares state rows, target metadata rows, and root-aware lookup
 * maps before flat horizon tuple emission starts.
 */
#include "flat_horizon_context.hpp"

#include <fmt/format.h>

#include <algorithm>
#include <string>
#include <vector>

#include "flat_entity_context.hpp"
#include "mifrost/backends/pymimir/transition_target_metadata.hpp"

namespace mifrost {

namespace {

std::string target_node_name(const FlatHorizonContextBuildConfig& config, int idx)
{
   return fmt::format("{}{}", config.target_symbol_prefix, idx);
}

}  // namespace

FlatHorizonEncoderEngine::EncodingContext build_flat_horizon_encoding_context(
   const mimir::search::State& root,
   const TransitionDAG& dag,
   const FlatHorizonContextBuildConfig& config
)
{
   FlatHorizonEncoderEngine::EncodingContext context;
   auto objects = root.get_problem().get_problem_and_domain_objects();
   std::ranges::sort(objects, [](const auto& lhs, const auto& rhs) {
      return lhs->get_index() < rhs->get_index();
   });

   const auto& nodes = dag.nodes();
   reserve_common_entity_context(
      context, objects.size(), nodes.size(), config.predicate_symbol_capacity
   );
   context.state_entity_index_by_node_index.reserve(nodes.size());
   context.target_entity_indices.reserve(
      (! root_in_target_metadata(config.root_policy) && ! nodes.empty()) ? nodes.size() - 1
                                                                         : nodes.size()
   );
   context.target_entity_group_ids.reserve(context.target_entity_indices.capacity());
   context.target_name_states.reserve(nodes.size());
   context.target_columns.reserve(
      (! root_in_target_metadata(config.root_policy) && ! nodes.empty()) ? nodes.size() - 1
                                                                         : nodes.size(),
      /*include_depth=*/true,
      /*include_group=*/true
   );

   append_object_entities(context, objects);

   hash_map< int64_t, int64_t > target_positions_by_index;
   target_positions_by_index.reserve(nodes.size());
   for(const auto& node : nodes) {
      const int64_t local_index = static_cast< int64_t >(context.entity_names.size());
      const bool include_in_target_metadata = ! (
         ! root_in_target_metadata(config.root_policy) && node.index == dag.root_index()
      );
      const bool include_in_public_carrier = root_in_public_carrier(config.root_policy)
                                             || node.index != dag.root_index();
      const std::string node_name = include_in_public_carrier ? target_node_name(config, node.index)
                                                              : config.hidden_root_carrier_name;
      context.entity_names.push_back(node_name);
      context.entity_role_ids.push_back(static_cast< int64_t >(FlatEntityRole::state));
      context.state_entity_index_by_node_index.emplace(node.index, local_index);
      if(include_in_target_metadata) {
         context.target_entity_indices.push_back(local_index);
         context.target_entity_group_ids.push_back(0);
         target_positions_by_index.emplace(node.index, local_index);
      }
   }

   const auto rows = pymimir_backend::collect_transition_dag_target_candidate_rows(
      dag,
      target_positions_by_index,
      config.root_policy,
      int64_t{0},
      /*include_names=*/false
   );
   append_target_candidate_rows(
      context.target_columns,
      rows,
      TargetCandidateAppendConfig{
         .include_depth = true,
         .include_group = true,
         .missing_candidate_id_prefix = "missing candidate_id for target node index ",
         .duplicate_candidate_id_prefix = "duplicate candidate_id ",
      }
   );
   if(config.export_node_names) {
      for(const auto& node : nodes) {
         if(! root_in_target_metadata(config.root_policy) && node.index == dag.root_index()) {
            continue;
         }
         if(! target_positions_by_index.contains(node.index)) {
            continue;
         }
         context.target_name_states.push_back(node.state);
      }
   }

   return context;
}

}  // namespace mifrost
