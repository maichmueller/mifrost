#include "successor_hgraph_encoder.hpp"

#include <fmt/format.h>

#include <algorithm>

namespace mifrost {

SuccessorHGraphEncoderEngine::SuccessorHGraphEncoderEngine(
   const mimir::formalism::DomainImpl& domain
)
    : HGraphEncoderEngine(domain)
{
}

SuccessorHGraphEncoderEngine::SuccessorHGraphEncoderEngine(
   const mimir::formalism::DomainImpl& domain,
   Config config
)
    : HGraphEncoderEngine(domain, config), successor_config_(std::move(config))
{
}

SuccessorHGraphEncoderEngine::SuccessorHGraphEncoderEngine(mimir::formalism::Domain domain)
    : HGraphEncoderEngine(domain)
{
}

SuccessorHGraphEncoderEngine::SuccessorHGraphEncoderEngine(
   mimir::formalism::Domain domain,
   Config config
)
    : HGraphEncoderEngine(domain, config), successor_config_(std::move(config))
{
}

void SuccessorHGraphEncoderEngine::encode(
   const mimir::search::State& current,
   const mimir::search::State& successor,
   const GoalInputs& goals,
   BatchBuilder& builder
)
{
   encode_impl(current, successor, goals, builder);
}

void SuccessorHGraphEncoderEngine::encode_impl(
   const mimir::search::State& current,
   const mimir::search::State& successor,
   const GoalInputs& goals,
   BatchBuilder& builder
)
{
   ensure_node_feature_dims(builder);

   hash_map< std::string, hash_map< std::string, int64_t > > node_indices;
   hash_map< std::string, std::vector< std::string > > node_names;
   hash_map< std::string, hash_set< std::string > > relation_to_symbols;
   hash_map< std::string, hash_set< std::string > > symbol_to_relations;

   // 1. Encode objects
   encode_objects(current, builder, node_indices, node_names);

   // 2. Encode current facts
   const auto cur_fact_keys = encode_facts(
      current, builder, node_indices, node_names, relation_to_symbols, symbol_to_relations
   );

   // 3. Encode successor facts
   hash_set< std::string > suc_fact_keys;
   const auto& problem = successor.get_problem();
   const auto& repos = problem.get_repositories();

   auto handle_suc_atom = [&](auto atom, bool polarity = true) {
      using Tag = typename std::remove_pointer_t< decltype(atom) >::Type;
      const auto predicate = atom->get_predicate();
      if(predicate->get_arity() == 0 and not config_.add_nullary_predicates) {
         return;
      }
      const std::string node_type = RelationFormatter::format_predicate(
         predicate->get_name(),
         std::nullopt,
         std::nullopt,
         polarity,
         successor_config_.successor_suffix
      );
      const std::string node_key = RelationFormatter::format_atom< Tag >(
         atom, successor_config_.successor_suffix
      );
      const auto relation_idx = get_or_add_node(
         node_type, node_key, builder, node_indices, node_names
      );

      std::vector< std::string > object_keys;
      if(predicate->get_arity() == 0) {
         object_keys.emplace_back(config_.nullary_object_name);
      } else {
         for(const auto& obj : atom->get_objects()) {
            object_keys.emplace_back(RelationFormatter::format_object(*obj));
         }
      }

      for(size_t pos = 0; pos < object_keys.size(); ++pos) {
         const auto& obj_key = object_keys[pos];
         const auto obj_idx = get_or_add_node(
            config_.symbol_type_id, obj_key, builder, node_indices, node_names
         );
         const std::string pos_str = std::to_string(pos);
         append_edges(builder, config_.symbol_type_id, pos_str, node_type, obj_idx, relation_idx);
         append_edges(builder, node_type, pos_str, config_.symbol_type_id, relation_idx, obj_idx);
      }

      const std::string rel_key = relation_key(node_type, node_key);
      auto& symbols = relation_to_symbols[rel_key];
      for(const auto& obj_key : object_keys) {
         symbols.insert(obj_key);
         symbol_to_relations[obj_key].insert(rel_key);
      }

      if(polarity) {
         suc_fact_keys.insert(RelationFormatter::format_atom< Tag >(atom));
      }
   };

   if(successor_config_.successor_mode == Mode::Full) {
      const auto fluent_atoms = repos.get_ground_atoms_from_indices< mimir::formalism::FluentTag >(
         successor.get_atoms< mimir::formalism::FluentTag >()
      );
      for(const auto& atom : fluent_atoms) {
         handle_suc_atom(atom);
      }
      const auto derived_atoms = repos
                                    .get_ground_atoms_from_indices< mimir::formalism::DerivedTag >(
                                       successor.get_atoms< mimir::formalism::DerivedTag >()
                                    );
      for(const auto& atom : derived_atoms) {
         handle_suc_atom(atom);
      }
   } else {
      // Delta mode
      auto handle_delta = [&](auto cur_indices, auto suc_indices, auto tag_ptr) {
         using Tag = std::remove_pointer_t< decltype(tag_ptr) >;
         std::vector< uint32_t > added;
         std::vector< uint32_t > removed;
         std::set_difference(
            suc_indices.begin(),
            suc_indices.end(),
            cur_indices.begin(),
            cur_indices.end(),
            std::back_inserter(added)
         );
         std::set_difference(
            cur_indices.begin(),
            cur_indices.end(),
            suc_indices.begin(),
            suc_indices.end(),
            std::back_inserter(removed)
         );

         for(uint32_t idx : added) {
            handle_suc_atom(repos.get_ground_atom< Tag >(idx), true);
         }
         for(uint32_t idx : removed) {
            handle_suc_atom(repos.get_ground_atom< Tag >(idx), false);
         }
      };

      handle_delta(
         current.get_atoms< mimir::formalism::FluentTag >(),
         successor.get_atoms< mimir::formalism::FluentTag >(),
         (mimir::formalism::FluentTag*) nullptr
      );
      handle_delta(
         current.get_atoms< mimir::formalism::DerivedTag >(),
         successor.get_atoms< mimir::formalism::DerivedTag >(),
         (mimir::formalism::DerivedTag*) nullptr
      );
   }

   // 4. Encode goals for current
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

   // 5. Encode goal satisfaction for current
   if(! goals.static_goals.empty()) {
      encode_goal_satisfaction(
         std::span{goals.static_goals},
         goals.static_goal_levels,
         cur_fact_keys,
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
         cur_fact_keys,
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
         cur_fact_keys,
         builder,
         node_indices,
         node_names,
         relation_to_symbols,
         symbol_to_relations
      );
   }

   // 6. Encode goal satisfaction for successor
   auto encode_suc_goal_satisfaction = [&](auto goals_span, const auto& levels) {
      for(const auto& goal : goals_span) {
         const auto atom = goal->get_atom();
         const auto predicate = atom->get_predicate();
         const auto key = RelationFormatter::format_atom(atom);
         const bool satisfied = suc_fact_keys.contains(key) == goal->get_polarity();
         const GoalSatisfaction sat = satisfied ? GoalSatisfaction::satisfied
                                                : GoalSatisfaction::unsatisfied;

         if(! relation_dict_.goal_satisfaction_derivations.contains(sat)) {
            continue;
         }

         std::optional< int > goal_level = levels.contains(goal)
                                              ? std::optional< int >(levels.at(goal))
                                              : std::nullopt;

         std::string node_type;
         std::string node_key;
         if(goal_level.has_value()) {
            const GoalLevel level(*goal_level);
            node_type = RelationFormatter::format_predicate(
               predicate->get_name(),
               level,
               sat,
               goal->get_polarity(),
               successor_config_.successor_suffix
            );
            node_key = RelationFormatter::format_literal(
               goal, level, sat, goal->get_polarity(), successor_config_.successor_suffix
            );
         } else {
            node_type = RelationFormatter::format_predicate(
               predicate->get_name(),
               std::nullopt,
               sat,
               goal->get_polarity(),
               successor_config_.successor_suffix
            );
            node_key = RelationFormatter::format_literal(
               goal, std::nullopt, sat, goal->get_polarity(), successor_config_.successor_suffix
            );
         }

         const auto relation_idx = get_or_add_node(
            node_type, node_key, builder, node_indices, node_names
         );

         std::vector< std::string > object_keys;
         if(predicate->get_arity() == 0) {
            object_keys.emplace_back(config_.nullary_object_name);
         } else {
            for(const auto& obj : atom->get_objects()) {
               object_keys.emplace_back(RelationFormatter::format_object(*obj));
            }
         }

         for(size_t pos = 0; pos < object_keys.size(); ++pos) {
            const auto& obj_key = object_keys[pos];
            const auto obj_idx = get_or_add_node(
               config_.symbol_type_id, obj_key, builder, node_indices, node_names
            );
            const std::string pos_str = std::to_string(pos);
            append_edges(
               builder, config_.symbol_type_id, pos_str, node_type, obj_idx, relation_idx
            );
            append_edges(
               builder, node_type, pos_str, config_.symbol_type_id, relation_idx, obj_idx
            );
         }
      }
   };

   encode_suc_goal_satisfaction(std::span{goals.static_goals}, goals.static_goal_levels);
   encode_suc_goal_satisfaction(std::span{goals.fluent_goals}, goals.fluent_goal_levels);
   encode_suc_goal_satisfaction(std::span{goals.derived_goals}, goals.derived_goal_levels);

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

}  // namespace mifrost
