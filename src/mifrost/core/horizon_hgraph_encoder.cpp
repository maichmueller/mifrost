#include "horizon_hgraph_encoder.hpp"

#include <fmt/format.h>

#include <algorithm>

namespace mifrost {

HorizonHGraphEncoderEngine::HorizonHGraphEncoderEngine(const mimir::formalism::DomainImpl& domain)
    : HGraphEncoderEngine(domain)
{
}

HorizonHGraphEncoderEngine::HorizonHGraphEncoderEngine(
   const mimir::formalism::DomainImpl& domain,
   Config config
)
    : HGraphEncoderEngine(domain, config), horizon_config_(std::move(config))
{
}

HorizonHGraphEncoderEngine::HorizonHGraphEncoderEngine(mimir::formalism::Domain domain)
    : HGraphEncoderEngine(domain)
{
}

HorizonHGraphEncoderEngine::HorizonHGraphEncoderEngine(
   mimir::formalism::Domain domain,
   Config config
)
    : HGraphEncoderEngine(domain, config), horizon_config_(std::move(config))
{
}

void HorizonHGraphEncoderEngine::encode(
   const mimir::search::State& root,
   const TransitionDAG& dag,
   const GoalInputs& goals,
   BatchBuilder& builder
)
{
   encode_impl(root, dag, goals, builder);
}

void HorizonHGraphEncoderEngine::encode_impl(
   const mimir::search::State& root,
   const TransitionDAG& dag,
   const GoalInputs& goals,
   BatchBuilder& builder
)
{
   ensure_node_feature_dims(builder);

   hash_map< std::string, hash_map< std::string, int64_t > > node_indices;
   hash_map< std::string, std::vector< std::string > > node_names;
   hash_map< std::string, hash_set< std::string > > relation_to_symbols;
   hash_map< std::string, hash_set< std::string > > symbol_to_relations;

   // 1. Create target nodes
   std::vector< std::string > target_keys;
   for(const auto& node : dag.nodes()) {
      target_keys.push_back(target_node_key(node.index));
   }

   // Register target nodes in builder immediately or via encode_objects?
   // encode_objects will add them if they are in extra_objects.

   // 2. Encode root state
   std::vector< std::string > root_extra = {target_keys[0]};
   encode_objects(root, builder, node_indices, node_names, root_extra);

   const auto root_fact_keys = encode_facts(
      root, builder, node_indices, node_names, relation_to_symbols, symbol_to_relations, root_extra
   );

   // 3. Encode successors
   const auto& nodes = dag.nodes();
   for(size_t i = 1; i < nodes.size(); ++i) {
      const auto& node = nodes[i];
      std::vector< std::string > succ_extra = {target_keys[i]};

      if(horizon_config_.transition_mode == Mode::Full) {
         encode_facts(
            node.state,
            builder,
            node_indices,
            node_names,
            relation_to_symbols,
            symbol_to_relations,
            succ_extra
         );
      } else if(horizon_config_.transition_mode == Mode::Delta) {
         // TODO: Implement delta mode (diff vs root)
         // For now, let's just do full if delta is requested but not yet implemented
         encode_facts(
            node.state,
            builder,
            node_indices,
            node_names,
            relation_to_symbols,
            symbol_to_relations,
            succ_extra
         );
      }
   }

   // 4. Encode goals
   encode_literals(
      std::span{goals.static_goals},
      goals.static_goal_levels,
      builder,
      node_indices,
      node_names,
      relation_to_symbols,
      symbol_to_relations
   );
   encode_literals(
      std::span{goals.fluent_goals},
      goals.fluent_goal_levels,
      builder,
      node_indices,
      node_names,
      relation_to_symbols,
      symbol_to_relations
   );
   encode_literals(
      std::span{goals.derived_goals},
      goals.derived_goal_levels,
      builder,
      node_indices,
      node_names,
      relation_to_symbols,
      symbol_to_relations
   );

   // 5. Goal satisfaction for root (and maybe for others in future?)
   if(! goals.static_goals.empty()) {
      encode_goal_satisfaction(
         std::span{goals.static_goals},
         goals.static_goal_levels,
         root_fact_keys,
         builder,
         node_indices,
         node_names,
         relation_to_symbols,
         symbol_to_relations
      );
   }
   if(! goals.fluent_goals.empty()) {
      encode_goal_satisfaction(
         std::span{goals.fluent_goals},
         goals.fluent_goal_levels,
         root_fact_keys,
         builder,
         node_indices,
         node_names,
         relation_to_symbols,
         symbol_to_relations
      );
   }
   if(! goals.derived_goals.empty()) {
      encode_goal_satisfaction(
         std::span{goals.derived_goals},
         goals.derived_goal_levels,
         root_fact_keys,
         builder,
         node_indices,
         node_names,
         relation_to_symbols,
         symbol_to_relations
      );
   }

   // 6. Parent relations
   if(horizon_config_.enable_parent_relation) {
      for(const auto& pair : dag.transitions()) {
         int parent_idx = pair.first;
         int child_idx = pair.second;
         const std::string& parent_key = target_keys[parent_idx];
         const std::string& child_key = target_keys[child_idx];

         const auto p_node = get_or_add_node(
            config_.symbol_type_id, parent_key, builder, node_indices, node_names
         );
         const auto c_node = get_or_add_node(
            config_.symbol_type_id, child_key, builder, node_indices, node_names
         );

         // Add edge (child) --[parent]--> (parent)
         HGraphEncoderEngine::append_edges(
            builder,
            config_.symbol_type_id,
            horizon_config_.parent_relation,
            config_.symbol_type_id,
            c_node,
            p_node
         );
      }
   }

   // 7. LGAN edges
   if(config_.include_lgan_edges) {
      add_lgan_nn_edges(builder, node_indices, relation_to_symbols, symbol_to_relations);
   }

   // 8. Finalize node names
   for(const auto& [node_type, _] : relation_dict_.arity) {
      if(! node_names.contains(node_type)) {
         builder.set_node_names(node_type, {});
      }
   }
   if(! node_names.contains(config_.symbol_type_id)) {
      builder.set_node_names(config_.symbol_type_id, {});
   } else {
      builder.set_node_names(config_.symbol_type_id, node_names[config_.symbol_type_id]);
      builder.set_object_names(node_names[config_.symbol_type_id]);
   }

   for(const auto& [node_type, names] : node_names) {
      if(node_type == config_.symbol_type_id) {
         continue;
      }
      builder.set_node_names(node_type, names);
   }

   ensure_empty_edge_types(builder);
}

std::string HorizonHGraphEncoderEngine::target_node_key(int idx) const
{
   return fmt::format("{}{}", horizon_config_.target_symbol_prefix, idx);
}

}  // namespace mifrost
