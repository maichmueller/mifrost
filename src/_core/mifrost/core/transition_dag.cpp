#include "transition_dag.hpp"

#include <algorithm>
#include <queue>
#include <stdexcept>

namespace mifrost {

TransitionDAG::TransitionDAG(mimir::search::State root) : root_(std::move(root))
{
   const auto index = next_index_++;
   state_to_index_.emplace(root_, index);
   nodes_ordered_.push_back(
      Node{.state = root_, .index = index, .depth = 0, .action = std::nullopt}
   );
}

int TransitionDAG::get_or_add_node(
   const mimir::search::State& state,
   const std::optional< mimir::formalism::GroundAction >& action_for_new_node
)
{
   auto it = state_to_index_.find(state);
   if(it != state_to_index_.end()) {
      const int existing_idx = it->second;
      if(action_for_new_node.has_value() and not nodes_ordered_[existing_idx].action.has_value()) {
         nodes_ordered_[existing_idx].action = action_for_new_node;
      }
      return existing_idx;
   }

   const int idx = next_index_++;
   Node node{
      .state = state,
      .index = idx,
      .depth = -1,
      .action = action_for_new_node,
   };
   state_to_index_.emplace(state, idx);
   nodes_ordered_.push_back(std::move(node));
   return idx;
}

std::pair< int, int > TransitionDAG::register_transition_impl(
   const mimir::search::State& parent,
   const mimir::search::State& child,
   const std::optional< mimir::formalism::GroundAction >& action,
   const bool recompute_depths,
   const bool require_parent_exists
)
{
   if(require_parent_exists and not contains(parent)) {
      throw std::invalid_argument("Parent state not in DAG");
   }

   const int parent_idx = require_parent_exists ? state_to_index_.at(parent)
                                                : get_or_add_node(parent, std::nullopt);
   const int child_idx = get_or_add_node(child, action);

   adjacency_[parent_idx].push_back(child_idx);

   if(recompute_depths) {
      finalize_depths();
   }

   return {parent_idx, child_idx};
}

std::pair< int, int > TransitionDAG::register_transition(
   const mimir::search::State& parent,
   const mimir::search::State& child,
   const std::optional< mimir::formalism::GroundAction > action
)
{
   return register_transition_impl(parent, child, action, true, true);
}

int TransitionDAG::index(const mimir::search::State& state) const
{
   auto it = state_to_index_.find(state);
   if(it == state_to_index_.end()) {
      throw std::out_of_range("State not found in DAG");
   }
   return it->second;
}

int TransitionDAG::depth(int idx) const
{
   if(idx < 0 or idx >= static_cast< int >(nodes_ordered_.size())) {
      throw std::out_of_range("Invalid node index");
   }
   return nodes_ordered_[idx].depth;
}

std::optional< mimir::formalism::GroundAction > TransitionDAG::action(int idx) const
{
   if(idx < 0 or idx >= static_cast< int >(nodes_ordered_.size())) {
      throw std::out_of_range("Invalid node index");
   }
   return nodes_ordered_[idx].action;
}

mimir::search::State TransitionDAG::state(int idx) const
{
   if(idx < 0 or idx >= static_cast< int >(nodes_ordered_.size())) {
      throw std::out_of_range("Invalid node index");
   }
   return nodes_ordered_[idx].state;
}

const std::vector< int >* TransitionDAG::children(int parent_idx) const
{
   auto it = adjacency_.find(parent_idx);
   if(it == adjacency_.end()) {
      return nullptr;
   }
   return &(it->second);
}

std::vector< TransitionDAG::Node > TransitionDAG::successors() const
{
   if(nodes_ordered_.empty()) {
      return {};
   }
   // Return all nodes except the first (root)
   return std::vector< Node >(nodes_ordered_.begin() + 1, nodes_ordered_.end());
}

std::vector< std::pair< int, int > > TransitionDAG::transitions() const
{
   std::vector< std::pair< int, int > > result;
   for(const auto& [parent_idx, children_vec] : adjacency_) {
      for(int child_idx : children_vec) {
         result.emplace_back(parent_idx, child_idx);
      }
   }
   // Sort for deterministic output
   std::ranges::sort(result);
   return result;
}

bool TransitionDAG::contains(const mimir::search::State& state) const
{
   return state_to_index_.contains(state);
}

void TransitionDAG::finalize_depths()
{
   if(nodes_ordered_.empty()) {
      return;
   }

   for(auto& node : nodes_ordered_) {
      node.depth = -1;
   }

   // BFS to compute shortest path from root.
   std::queue< int > queue;
   hash_set< int > visited;

   queue.push(0);
   visited.insert(0);
   nodes_ordered_[0].depth = 0;

   while(not queue.empty()) {
      int current_idx = queue.front();
      queue.pop();

      int current_depth = nodes_ordered_[current_idx].depth;

      auto it = adjacency_.find(current_idx);
      if(it != adjacency_.end()) {
         for(int child_idx : it->second) {
            if(visited.contains(child_idx)) {
               nodes_ordered_[child_idx].depth = std::min(
                  nodes_ordered_[child_idx].depth, current_depth + 1
               );
            } else {
               visited.insert(child_idx);
               nodes_ordered_[child_idx].depth = current_depth + 1;
               queue.push(child_idx);
            }
         }
      }
   }
}

}  // namespace mifrost
