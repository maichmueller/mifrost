#include "mifrost/backends/pymimir/transition_target_metadata.hpp"

#include <mimir/search/formatter.hpp>
#include <sstream>

namespace mifrost::pymimir_backend {

std::vector< TargetCandidateRow > collect_transition_dag_target_candidate_rows(
   const TransitionDAG& dag,
   const hash_map< int64_t, int64_t >& positions_by_index,
   RootPolicy root_policy,
   std::optional< int64_t > group_id,
   bool include_names
)
{
   const auto& nodes = dag.nodes();
   const size_t reserved = (not root_in_target_metadata(root_policy) and not nodes.empty())
                              ? (nodes.size() - 1)
                              : nodes.size();
   std::vector< TargetCandidateRow > rows;
   rows.reserve(reserved);

   for(const auto& node : nodes) {
      if(not root_in_target_metadata(root_policy) and node.index == dag.root_index()) {
         continue;
      }
      const auto position_it = positions_by_index.find(node.index);
      if(position_it == positions_by_index.end()) {
         continue;
      }
      rows.push_back(
         TargetCandidateRow{
            .position = position_it->second,
            .index = node.index,
            .candidate_id = node.candidate_id,
            .depth = node.depth,
            .group_id = group_id,
            .name = [&]() {
               if(not include_names) {
                  return std::string{};
               }
               std::ostringstream stream;
               stream << node.state;
               return stream.str();
            }(),
         }
      );
   }

   return rows;
}

}  // namespace mifrost::pymimir_backend
