/**
 * @file semantic_transition_dag.cpp
 * @brief Validation and accessors for the neutral semantic transition DAG.
 */
#include "semantic_transition_dag.hpp"

#include <algorithm>
#include <queue>
#include <set>
#include <stdexcept>
#include <string_view>

namespace mifrost {
namespace {

template < typename Spec >
void validate_schema_specs(const std::vector< Spec >& specs, std::string_view kind)
{
   std::set< std::string, std::less<> > names;
   for(const auto& spec : specs) {
      if(spec.name.empty()) {
         throw std::invalid_argument(std::string(kind) + " name must not be empty");
      }
      if(spec.arity < 0) {
         throw std::invalid_argument(std::string(kind) + " arity must be non-negative");
      }
      if(not names.emplace(spec.name).second) {
         throw std::invalid_argument(
            "SemanticTransitionDAG requires unique " + std::string(kind) + " names"
         );
      }
   }
}

void validate_atom(
   const SemanticAtom& atom,
   const std::vector< SemanticPredicateSpec >& predicates,
   size_t object_count,
   std::string_view lane
)
{
   if(atom.predicate < 0 or static_cast< size_t >(atom.predicate) >= predicates.size()) {
      throw std::invalid_argument(
         "SemanticTransitionDAG " + std::string(lane) + " predicate index out of range"
      );
   }
   if(static_cast< int64_t >(atom.arguments.size())
      != predicates[static_cast< size_t >(atom.predicate)].arity) {
      throw std::invalid_argument(
         "SemanticTransitionDAG " + std::string(lane)
         + " argument count does not match schema arity"
      );
   }
   for(const auto object : atom.arguments) {
      if(object < 0 or static_cast< size_t >(object) >= object_count) {
         throw std::invalid_argument(
            "SemanticTransitionDAG " + std::string(lane) + " object index out of range"
         );
      }
   }
}

void validate_action(
   const SemanticGroundAction& action,
   const std::vector< SemanticActionSpec >& actions,
   size_t object_count,
   std::string_view lane
)
{
   if(action.action < 0 or static_cast< size_t >(action.action) >= actions.size()) {
      throw std::invalid_argument(
         "SemanticTransitionDAG " + std::string(lane) + " action index out of range"
      );
   }
   if(static_cast< int64_t >(action.arguments.size())
      != actions[static_cast< size_t >(action.action)].arity) {
      throw std::invalid_argument(
         "SemanticTransitionDAG " + std::string(lane)
         + " argument count does not match schema arity"
      );
   }
   for(const auto object : action.arguments) {
      if(object < 0 or static_cast< size_t >(object) >= object_count) {
         throw std::invalid_argument(
            "SemanticTransitionDAG " + std::string(lane) + " object index out of range"
         );
      }
   }
}

void validate_state(
   const SemanticFlatRelationInput& state,
   const std::vector< SemanticPredicateSpec >& predicates,
   const std::vector< SemanticActionSpec >& actions
)
{
   const auto& objects = semantic_objects(state);
   const auto& goals = semantic_goals(state);
   const auto& static_facts = semantic_static_facts(state);
   std::set< std::string, std::less<> > object_names;
   for(const auto& object : objects) {
      if(object.empty()) {
         throw std::invalid_argument("SemanticTransitionDAG object name must not be empty");
      }
      if(not object_names.emplace(object).second) {
         throw std::invalid_argument("SemanticTransitionDAG object names must be unique");
      }
   }

   for(const auto& atom : state.state_facts) {
      validate_atom(atom, predicates, objects.size(), "state fact");
   }
   for(const auto& atom : static_facts) {
      validate_atom(atom, predicates, objects.size(), "static fact");
   }
   for(const auto& literal : goals) {
      validate_atom(literal.atom, predicates, objects.size(), "goal literal");
   }
   for(const auto& layer : state.subgoal_layers) {
      for(const auto& literal : layer) {
         validate_atom(literal.atom, predicates, objects.size(), "subgoal literal");
      }
   }
   for(const auto& entry : state.history) {
      if(entry.dt >= 0) {
         throw std::invalid_argument("SemanticTransitionDAG history requires negative dt values");
      }
      for(const auto& literal : entry.literals) {
         validate_atom(literal.atom, predicates, objects.size(), "history literal");
      }
   }
   if(state.history_max_steps.has_value() and *state.history_max_steps < 0) {
      throw std::invalid_argument("SemanticTransitionDAG history_max_steps must be non-negative");
   }
   for(const auto& action : state.actions) {
      validate_action(action, actions, objects.size(), "state action");
   }
}

void validate_node_index(int64_t index, size_t node_count)
{
   if(index < 0 or static_cast< size_t >(index) >= node_count) {
      throw std::out_of_range("SemanticTransitionDAG node index out of range");
   }
}

}  // namespace

SemanticTransitionDAG::SemanticTransitionDAG(
   std::vector< SemanticPredicateSpec > predicates,
   std::vector< SemanticActionSpec > actions,
   std::vector< Node > nodes,
   std::vector< Edge > edges
)
    : predicates_(std::move(predicates)),
      actions_(std::move(actions)),
      nodes_(std::move(nodes)),
      edges_(std::move(edges))
{
   validate_schema_specs(predicates_, "predicate");
   validate_schema_specs(actions_, "action");
   for(const auto& predicate : predicates_) {
      switch(predicate.category) {
         case SemanticPredicateCategory::static_predicate:
         case SemanticPredicateCategory::fluent:
         case SemanticPredicateCategory::derived: break;
         default:
            throw std::invalid_argument("SemanticTransitionDAG predicate category is invalid");
      }
   }
   if(nodes_.empty()) {
      throw std::invalid_argument("SemanticTransitionDAG requires a nonempty node list");
   }

   const auto& root_state = nodes_.front().state;
   const auto& object_table = semantic_objects(root_state);
   for(size_t index = 0; index < nodes_.size(); ++index) {
      const auto& node = nodes_[index];
      if(node.index != static_cast< int64_t >(index)) {
         throw std::invalid_argument(
            "SemanticTransitionDAG node indices must be stable and contiguous"
         );
      }
      if(node.depth < 0) {
         throw std::invalid_argument("SemanticTransitionDAG node depth must be non-negative");
      }
      if(semantic_objects(node.state) != object_table) {
         throw std::invalid_argument(
            "SemanticTransitionDAG nodes require identical ordered object tables"
         );
      }
      if(root_state.task_context and node.state.task_context != root_state.task_context) {
         throw std::invalid_argument(
            "SemanticTransitionDAG context-backed nodes must share one task context"
         );
      }
      if(not root_state.task_context and node.state.task_context) {
         throw std::invalid_argument(
            "SemanticTransitionDAG cannot mix legacy and context-backed state inputs"
         );
      }
      if(node.display_name.has_value() and node.display_name->empty()) {
         throw std::invalid_argument(
            "SemanticTransitionDAG display name must not be empty when provided"
         );
      }
      validate_state(node.state, predicates_, actions_);
      if(node.incoming_action.has_value()) {
         validate_action(*node.incoming_action, actions_, object_table.size(), "incoming action");
      }
      if(node.delta_literals.has_value()) {
         for(const auto& literal : *node.delta_literals) {
            validate_atom(literal.atom, predicates_, object_table.size(), "delta literal");
         }
      }
   }
   if(nodes_.front().index != 0 or nodes_.front().depth != 0) {
      throw std::invalid_argument("SemanticTransitionDAG root must have index and depth 0");
   }
   if(nodes_.front().incoming_action.has_value()) {
      throw std::invalid_argument("SemanticTransitionDAG root cannot have an incoming action");
   }
   if(nodes_.front().delta_literals.has_value()) {
      throw std::invalid_argument("SemanticTransitionDAG root cannot have delta literals");
   }

   std::set< Edge > unique_edges;
   for(const auto& [parent, child] : edges_) {
      if(parent < 0 or child < 0 or static_cast< size_t >(parent) >= nodes_.size()
         or static_cast< size_t >(child) >= nodes_.size()) {
         throw std::invalid_argument("SemanticTransitionDAG edge endpoint out of range");
      }
      if(parent == child) {
         throw std::invalid_argument("SemanticTransitionDAG self edges are not allowed");
      }
      if(not unique_edges.emplace(parent, child).second) {
         throw std::invalid_argument("SemanticTransitionDAG duplicate edge");
      }
   }
   std::ranges::sort(edges_);

   std::vector< std::vector< int64_t > > adjacency(nodes_.size());
   std::vector< size_t > indegree(nodes_.size(), 0);
   for(const auto& [parent, child] : edges_) {
      adjacency[static_cast< size_t >(parent)].push_back(child);
      ++indegree[static_cast< size_t >(child)];
   }

   auto remaining_indegree = indegree;
   std::queue< int64_t > ready;
   for(size_t index = 0; index < remaining_indegree.size(); ++index) {
      if(remaining_indegree[index] == 0) {
         ready.push(static_cast< int64_t >(index));
      }
   }
   size_t topological_count = 0;
   while(not ready.empty()) {
      const auto parent = ready.front();
      ready.pop();
      ++topological_count;
      for(const auto child : adjacency[static_cast< size_t >(parent)]) {
         auto& child_indegree = remaining_indegree[static_cast< size_t >(child)];
         --child_indegree;
         if(child_indegree == 0) {
            ready.push(child);
         }
      }
   }
   if(topological_count != nodes_.size()) {
      throw std::invalid_argument("SemanticTransitionDAG edges must be acyclic");
   }

   std::vector< int64_t > shortest_depth(nodes_.size(), -1);
   std::queue< int64_t > frontier;
   shortest_depth[0] = 0;
   frontier.push(0);
   while(not frontier.empty()) {
      const auto parent = frontier.front();
      frontier.pop();
      for(const auto child : adjacency[static_cast< size_t >(parent)]) {
         const auto candidate_depth = shortest_depth[static_cast< size_t >(parent)] + 1;
         auto& known_depth = shortest_depth[static_cast< size_t >(child)];
         if(known_depth < 0 or candidate_depth < known_depth) {
            known_depth = candidate_depth;
            frontier.push(child);
         }
      }
   }
   for(size_t index = 0; index < nodes_.size(); ++index) {
      if(shortest_depth[index] < 0) {
         throw std::invalid_argument("SemanticTransitionDAG nodes must be reachable from root");
      }
      if(nodes_[index].depth != shortest_depth[index]) {
         throw std::invalid_argument(
            "SemanticTransitionDAG node depth must equal shortest-path depth"
         );
      }
   }
}

std::vector< int64_t > SemanticTransitionDAG::children(int64_t node_index) const
{
   validate_node_index(node_index, nodes_.size());
   std::vector< int64_t > result;
   const auto first = std::ranges::lower_bound(edges_, Edge{node_index, 0});
   for(auto it = first; it != edges_.end() and it->first == node_index; ++it) {
      result.push_back(it->second);
   }
   return result;
}

std::vector< int64_t > SemanticTransitionDAG::parents(int64_t node_index) const
{
   validate_node_index(node_index, nodes_.size());
   std::vector< int64_t > result;
   for(const auto& [parent, child] : edges_) {
      if(child == node_index) {
         result.push_back(parent);
      }
   }
   return result;
}

SemanticCandidateIdCoverage SemanticTransitionDAG::candidate_id_coverage(bool include_root) const
{
   const size_t first = include_root ? 0 : 1;
   size_t present = 0;
   for(size_t index = first; index < nodes_.size(); ++index) {
      present += nodes_[index].candidate_id.has_value() ? 1 : 0;
   }
   const size_t selected = nodes_.size() - first;
   if(present == 0) {
      return SemanticCandidateIdCoverage::none;
   }
   if(present == selected) {
      return SemanticCandidateIdCoverage::complete;
   }
   return SemanticCandidateIdCoverage::partial;
}

void SemanticTransitionDAG::validate_candidate_ids(bool include_root) const
{
   const auto coverage = candidate_id_coverage(include_root);
   if(coverage == SemanticCandidateIdCoverage::none) {
      return;
   }
   const size_t first = include_root ? 0 : 1;
   for(size_t index = first; index < nodes_.size(); ++index) {
      if(not nodes_[index].candidate_id.has_value()) {
         throw std::invalid_argument(
            "missing candidate_id for node index " + std::to_string(index)
         );
      }
   }
   std::set< int64_t > seen;
   for(size_t index = first; index < nodes_.size(); ++index) {
      const auto candidate_id = *nodes_[index].candidate_id;
      if(not seen.insert(candidate_id).second) {
         throw std::invalid_argument("duplicate candidate_id " + std::to_string(candidate_id));
      }
   }
}

std::vector< int64_t > SemanticTransitionDAG::candidate_ids(bool include_root) const
{
   validate_candidate_ids(include_root);
   if(candidate_id_coverage(include_root) == SemanticCandidateIdCoverage::none) {
      return {};
   }
   const size_t first = include_root ? 0 : 1;
   std::vector< int64_t > result;
   result.reserve(nodes_.size() - first);
   for(size_t index = first; index < nodes_.size(); ++index) {
      result.push_back(*nodes_[index].candidate_id);
   }
   return result;
}

}  // namespace mifrost
