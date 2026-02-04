#include "horizon_hgraph_encoder.hpp"

#include <fmt/format.h>

#include <algorithm>
#include <array>
#include <mimir/search/formatter.hpp>
#include <set>
#include <sstream>

namespace mifrost {

HorizonHGraphEncoderEngine::HorizonHGraphEncoderEngine(const mimir::formalism::DomainImpl& domain)
    : HGraphEncoderEngine(domain)
{
   configure_relations();
}

HorizonHGraphEncoderEngine::HorizonHGraphEncoderEngine(
   const mimir::formalism::DomainImpl& domain,
   Config config
)
    : HGraphEncoderEngine(
         domain,
         [&]() {
            if(config.transition_mode == Mode::Delta and not config.support_literals) {
               config.support_literals = true;
            }
            return config;
         }()
      ),
      horizon_config_(std::move(config))
{
   configure_relations();
}

HorizonHGraphEncoderEngine::HorizonHGraphEncoderEngine(mimir::formalism::Domain domain)
    : HGraphEncoderEngine(domain)
{
   configure_relations();
}

HorizonHGraphEncoderEngine::HorizonHGraphEncoderEngine(
   mimir::formalism::Domain domain,
   Config config
)
    : HGraphEncoderEngine(
         domain,
         [&]() {
            if(config.transition_mode == Mode::Delta and not config.support_literals) {
               config.support_literals = true;
            }
            return config;
         }()
      ),
      horizon_config_(std::move(config))
{
   configure_relations();
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

   if(horizon_config_.transition_mode == Mode::Delta and not config_.support_literals) {
      throw std::invalid_argument("Delta horizon encoding requires support_literals=true.");
   }
   if(horizon_config_.transition_mode == Mode::Action and config_.ignore_actions) {
      throw std::invalid_argument("Action horizon encoding requires ignore_actions=false.");
   }

   hash_map< std::string, hash_map< std::string, int64_t > > node_indices;
   hash_map< std::string, std::vector< std::string > > node_names;
   hash_map< std::string, hash_set< std::string > > relation_to_symbols;
   hash_map< std::string, hash_set< std::string > > symbol_to_relations;

   auto make_prefix = [](const std::string& target_key) { return target_key + "|"; };

   // Helpers for horizon-specific encoding (extra_objects first, prefixed node keys)
   auto encode_atoms_with_prefix =
      [&](auto atoms, const std::string& prefix, std::span< const std::string > extra_objects) {
         hash_set< std::string > fact_keys;
         for(const auto& atom : atoms) {
            const auto predicate = atom->get_predicate();
            if(predicate->get_arity() == 0 and not config_.add_nullary_predicates) {
               continue;
            }
            const std::string node_type = RelationFormatter::format_predicate(predicate);
            const std::string node_key = prefix + RelationFormatter::format_atom(atom);
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

            size_t pos = 0;
            for(const auto& obj_key : extra_objects) {
               const auto obj_idx = get_or_add_node(
                  config_.symbol_type_id, obj_key, builder, node_indices, node_names
               );
               const std::string pos_str = std::to_string(pos++);
               append_edges(
                  builder, config_.symbol_type_id, pos_str, node_type, obj_idx, relation_idx
               );
               append_edges(
                  builder, node_type, pos_str, config_.symbol_type_id, relation_idx, obj_idx
               );
            }

            for(const auto& obj_key : object_keys) {
               const auto obj_idx = get_or_add_node(
                  config_.symbol_type_id, obj_key, builder, node_indices, node_names
               );
               const std::string pos_str = std::to_string(pos++);
               append_edges(
                  builder, config_.symbol_type_id, pos_str, node_type, obj_idx, relation_idx
               );
               append_edges(
                  builder, node_type, pos_str, config_.symbol_type_id, relation_idx, obj_idx
               );
            }

            const std::string rel_key = relation_key(node_type, node_key);
            auto& symbols = relation_to_symbols[rel_key];
            for(const auto& obj_key : extra_objects) {
               symbols.insert(obj_key);
               symbol_to_relations[obj_key].insert(rel_key);
            }
            for(const auto& obj_key : object_keys) {
               symbols.insert(obj_key);
               symbol_to_relations[obj_key].insert(rel_key);
            }
            fact_keys.insert(node_key);
         }
         return fact_keys;
      };

   auto encode_state_facts_with_prefix = [&](
                                            const mimir::search::State& state,
                                            const std::string& prefix,
                                            std::span< const std::string > extra_objects,
                                            bool include_static
                                         ) {
      hash_set< std::string > fact_keys;
      const auto& problem = state.get_problem();
      const auto& repos = problem.get_repositories();

      if(include_static) {
         const auto& literals = problem.get_initial_literals< mimir::formalism::StaticTag >();
         for(const auto& literal : literals) {
            if(not literal->get_polarity()) {
               continue;
            }
            auto atom = literal->get_atom();
            std::array< decltype(atom), 1 > atoms{atom};
            auto keys = encode_atoms_with_prefix(std::span{atoms}, prefix, extra_objects);
            for(const auto& key : keys) {
               fact_keys.insert(key);
            }
         }
      }

      const auto fluent_atoms = repos.get_ground_atoms_from_indices< mimir::formalism::FluentTag >(
         state.get_atoms< mimir::formalism::FluentTag >()
      );
      auto fluent_keys = encode_atoms_with_prefix(std::span{fluent_atoms}, prefix, extra_objects);
      for(const auto& key : fluent_keys) {
         fact_keys.insert(key);
      }

      const auto derived_atoms = repos
                                    .get_ground_atoms_from_indices< mimir::formalism::DerivedTag >(
                                       state.get_atoms< mimir::formalism::DerivedTag >()
                                    );
      auto derived_keys = encode_atoms_with_prefix(std::span{derived_atoms}, prefix, extra_objects);
      for(const auto& key : derived_keys) {
         fact_keys.insert(key);
      }

      return fact_keys;
   };

   auto encode_literals_with_prefix =
      [&]< typename GoalTag >(
         std::span< const mimir::formalism::GroundLiteral< GoalTag > > literals,
         const hash_map< mimir::formalism::GroundLiteral< GoalTag >, int >& goal_levels,
         const std::string& prefix,
         std::span< const std::string > extra_objects,
         std::optional< GoalSatisfaction > satisfaction_override = std::nullopt
      ) {
         for(const auto& literal : literals) {
            const auto atom = literal->get_atom();
            const auto predicate = atom->get_predicate();
            const std::optional< int > goal_level = goal_levels.contains(literal)
                                                       ? std::optional< int >(
                                                            goal_levels.at(literal)
                                                         )
                                                       : std::nullopt;

            const GoalSatisfaction satisfaction = satisfaction_override.has_value()
                                                     ? *satisfaction_override
                                                     : GoalSatisfaction::none;

            std::string node_type;
            std::string node_key;
            auto format_with = [&](auto level_arg, auto satisfaction_arg) {
               node_type = RelationFormatter::format_predicate(
                  predicate, level_arg, satisfaction_arg, literal->get_polarity()
               );
               node_key = prefix
                          + RelationFormatter::format_literal< GoalTag >(
                             literal, level_arg, satisfaction_arg
                          );
            };

            if(goal_level.has_value()) {
               const GoalLevel level(*goal_level);
               if(satisfaction_override.has_value()) {
                  format_with(level, satisfaction);
               } else {
                  format_with(level, std::nullopt);
               }
            } else {
               if(satisfaction_override.has_value()) {
                  format_with(std::nullopt, satisfaction);
               } else {
                  format_with(std::nullopt, std::nullopt);
               }
            }

            const auto relation_idx = get_or_add_node(
               node_type, node_key, builder, node_indices, node_names
            );

            std::vector< std::string > object_keys;
            if(predicate->get_arity() == 0) {
               if(not config_.add_nullary_predicates) {
                  continue;
               }
               object_keys.emplace_back(config_.nullary_object_name);
            } else {
               for(const auto& obj : atom->get_objects()) {
                  object_keys.emplace_back(RelationFormatter::format_object(*obj));
               }
            }

            size_t pos = 0;
            for(const auto& obj_key : extra_objects) {
               const auto obj_idx = get_or_add_node(
                  config_.symbol_type_id, obj_key, builder, node_indices, node_names
               );
               const std::string pos_str = std::to_string(pos++);
               append_edges(
                  builder, config_.symbol_type_id, pos_str, node_type, obj_idx, relation_idx
               );
               append_edges(
                  builder, node_type, pos_str, config_.symbol_type_id, relation_idx, obj_idx
               );
            }
            for(const auto& obj_key : object_keys) {
               const auto obj_idx = get_or_add_node(
                  config_.symbol_type_id, obj_key, builder, node_indices, node_names
               );
               const std::string pos_str = std::to_string(pos++);
               append_edges(
                  builder, config_.symbol_type_id, pos_str, node_type, obj_idx, relation_idx
               );
               append_edges(
                  builder, node_type, pos_str, config_.symbol_type_id, relation_idx, obj_idx
               );
            }

            const std::string rel_key = relation_key(node_type, node_key);
            auto& symbols = relation_to_symbols[rel_key];
            for(const auto& obj_key : extra_objects) {
               symbols.insert(obj_key);
               symbol_to_relations[obj_key].insert(rel_key);
            }
            for(const auto& obj_key : object_keys) {
               symbols.insert(obj_key);
               symbol_to_relations[obj_key].insert(rel_key);
            }
         }
      };

   auto encode_literal_atom_with_prefix = [&](
                                             auto atom,
                                             bool polarity,
                                             const std::string& prefix,
                                             std::span< const std::string > extra_objects
                                          ) {
      const auto predicate = atom->get_predicate();
      if(predicate->get_arity() == 0 and not config_.add_nullary_predicates) {
         return;
      }

      const std::string node_type = RelationFormatter::format_predicate(
         predicate, std::nullopt, std::nullopt, polarity
      );
      const std::string atom_str = RelationFormatter::format_atom(atom);
      const std::string literal_str = fmt::format(
         "{}{}", RelationFormatter::polarity_prefix(polarity), atom_str
      );
      const std::string node_key = prefix + literal_str;

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

      size_t pos = 0;
      for(const auto& obj_key : extra_objects) {
         const auto obj_idx = get_or_add_node(
            config_.symbol_type_id, obj_key, builder, node_indices, node_names
         );
         const std::string pos_str = std::to_string(pos++);
         append_edges(builder, config_.symbol_type_id, pos_str, node_type, obj_idx, relation_idx);
         append_edges(builder, node_type, pos_str, config_.symbol_type_id, relation_idx, obj_idx);
      }
      for(const auto& obj_key : object_keys) {
         const auto obj_idx = get_or_add_node(
            config_.symbol_type_id, obj_key, builder, node_indices, node_names
         );
         const std::string pos_str = std::to_string(pos++);
         append_edges(builder, config_.symbol_type_id, pos_str, node_type, obj_idx, relation_idx);
         append_edges(builder, node_type, pos_str, config_.symbol_type_id, relation_idx, obj_idx);
      }

      const std::string rel_key = relation_key(node_type, node_key);
      auto& symbols = relation_to_symbols[rel_key];
      for(const auto& obj_key : extra_objects) {
         symbols.insert(obj_key);
         symbol_to_relations[obj_key].insert(rel_key);
      }
      for(const auto& obj_key : object_keys) {
         symbols.insert(obj_key);
         symbol_to_relations[obj_key].insert(rel_key);
      }
   };

   auto encode_goal_satisfaction_with_prefix =
      [&]< typename GoalTag >(
         std::span< const mimir::formalism::GroundLiteral< GoalTag > > literals,
         const hash_map< mimir::formalism::GroundLiteral< GoalTag >, int >& goal_levels,
         const hash_set< std::string >& fact_keys,
         const std::string& prefix,
         std::span< const std::string > extra_objects
      ) {
         for(const auto& goal : literals) {
            const auto atom = goal->get_atom();
            const auto predicate = atom->get_predicate();
            const auto key = prefix + RelationFormatter::format_atom< GoalTag >(atom);
            const bool satisfied = fact_keys.contains(key) == goal->get_polarity();
            const GoalSatisfaction sat = satisfied ? GoalSatisfaction::satisfied
                                                   : GoalSatisfaction::unsatisfied;
            if(not relation_dict_.goal_satisfaction_derivations.contains(sat)) {
               continue;
            }

            std::optional< int > goal_level = goal_levels.contains(goal)
                                                 ? std::optional< int >(goal_levels.at(goal))
                                                 : std::nullopt;

            std::string node_type;
            std::string node_key;
            if(goal_level.has_value()) {
               const GoalLevel level(*goal_level);
               node_type = RelationFormatter::format_predicate(
                  predicate, level, sat, goal->get_polarity()
               );
               node_key = prefix + RelationFormatter::format_literal< GoalTag >(goal, level, sat);
            } else {
               node_type = RelationFormatter::format_predicate(
                  predicate, std::nullopt, sat, goal->get_polarity()
               );
               node_key = prefix
                          + RelationFormatter::format_literal< GoalTag >(goal, std::nullopt, sat);
            }

            const auto relation_idx = get_or_add_node(
               node_type, node_key, builder, node_indices, node_names
            );

            std::vector< std::string > object_keys;
            if(predicate->get_arity() == 0) {
               if(not config_.add_nullary_predicates) {
                  continue;
               }
               object_keys.emplace_back(config_.nullary_object_name);
            } else {
               for(const auto& obj : atom->get_objects()) {
                  object_keys.emplace_back(RelationFormatter::format_object(*obj));
               }
            }

            size_t pos = 0;
            for(const auto& obj_key : extra_objects) {
               const auto obj_idx = get_or_add_node(
                  config_.symbol_type_id, obj_key, builder, node_indices, node_names
               );
               const std::string pos_str = std::to_string(pos++);
               append_edges(
                  builder, config_.symbol_type_id, pos_str, node_type, obj_idx, relation_idx
               );
               append_edges(
                  builder, node_type, pos_str, config_.symbol_type_id, relation_idx, obj_idx
               );
            }
            for(const auto& obj_key : object_keys) {
               const auto obj_idx = get_or_add_node(
                  config_.symbol_type_id, obj_key, builder, node_indices, node_names
               );
               const std::string pos_str = std::to_string(pos++);
               append_edges(
                  builder, config_.symbol_type_id, pos_str, node_type, obj_idx, relation_idx
               );
               append_edges(
                  builder, node_type, pos_str, config_.symbol_type_id, relation_idx, obj_idx
               );
            }

            const std::string rel_key = relation_key(node_type, node_key);
            auto& symbols = relation_to_symbols[rel_key];
            for(const auto& obj_key : extra_objects) {
               symbols.insert(obj_key);
               symbol_to_relations[obj_key].insert(rel_key);
            }
            for(const auto& obj_key : object_keys) {
               symbols.insert(obj_key);
               symbol_to_relations[obj_key].insert(rel_key);
            }
         }
      };

   auto encode_action_with_prefix = [&](
                                       const mimir::formalism::GroundAction& action,
                                       const std::string& prefix,
                                       std::span< const std::string > extra_objects
                                    ) {
      const std::string node_type = RelationFormatter::format_action_schema(*action->get_action());
      const std::string node_key = prefix + RelationFormatter::format_action(action);
      const auto relation_idx = get_or_add_node(
         node_type, node_key, builder, node_indices, node_names
      );

      std::vector< std::string > object_keys;
      for(const auto& obj_key : extra_objects) {
         object_keys.emplace_back(obj_key);
      }
      for(const auto& obj : action->get_objects()) {
         object_keys.emplace_back(RelationFormatter::format_object(*obj));
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
   };

   // 1. Create target nodes first to keep them contiguous in symbol list.
   const auto& nodes = dag.nodes();
   std::vector< std::string > target_keys(nodes.size());
   for(const auto& node : nodes) {
      if(node.index >= static_cast< int >(target_keys.size())) {
         target_keys.resize(node.index + 1);
      }
      const std::string key = target_node_key(node.index);
      target_keys[node.index] = key;
      get_or_add_node(config_.symbol_type_id, key, builder, node_indices, node_names);
   }

   // 2. Encode root state (objects then facts/goals)
   encode_objects(root, builder, node_indices, node_names);
   const std::string root_prefix = make_prefix(target_keys[0]);
   const std::array< std::string, 1 > root_extra{target_keys[0]};

   const auto root_fact_keys = encode_state_facts_with_prefix(
      root, root_prefix, root_extra, config_.include_static
   );

   encode_literals_with_prefix.template operator()< mimir::formalism::StaticTag >(
      std::span{goals.static_goals}, goals.static_goal_levels, root_prefix, root_extra
   );
   encode_literals_with_prefix.template operator()< mimir::formalism::FluentTag >(
      std::span{goals.fluent_goals}, goals.fluent_goal_levels, root_prefix, root_extra
   );
   encode_literals_with_prefix.template operator()< mimir::formalism::DerivedTag >(
      std::span{goals.derived_goals}, goals.derived_goal_levels, root_prefix, root_extra
   );

   if(not goals.static_goals.empty()) {
      encode_goal_satisfaction_with_prefix.template operator()< mimir::formalism::StaticTag >(
         std::span{goals.static_goals},
         goals.static_goal_levels,
         root_fact_keys,
         root_prefix,
         root_extra
      );
   }
   if(not goals.fluent_goals.empty()) {
      encode_goal_satisfaction_with_prefix.template operator()< mimir::formalism::FluentTag >(
         std::span{goals.fluent_goals},
         goals.fluent_goal_levels,
         root_fact_keys,
         root_prefix,
         root_extra
      );
   }
   if(not goals.derived_goals.empty()) {
      encode_goal_satisfaction_with_prefix.template operator()< mimir::formalism::DerivedTag >(
         std::span{goals.derived_goals},
         goals.derived_goal_levels,
         root_fact_keys,
         root_prefix,
         root_extra
      );
   }

   const bool encode_actions = (not config_.ignore_actions)
                               || (horizon_config_.transition_mode == Mode::Action);

   // Precompute root atoms (no statics) for delta mode.
   hash_set< int > root_fluent_indices;
   hash_set< int > root_derived_indices;
   if(horizon_config_.transition_mode == Mode::Delta) {
      const auto& repos = root.get_problem().get_repositories();
      const auto root_fluents = repos.get_ground_atoms_from_indices< mimir::formalism::FluentTag >(
         root.get_atoms< mimir::formalism::FluentTag >()
      );
      for(const auto& atom : root_fluents) {
         root_fluent_indices.insert(atom->get_index());
      }
      const auto root_derived = repos.get_ground_atoms_from_indices< mimir::formalism::DerivedTag >(
         root.get_atoms< mimir::formalism::DerivedTag >()
      );
      for(const auto& atom : root_derived) {
         root_derived_indices.insert(atom->get_index());
      }
   }

   // 3. Encode successors
   for(const auto& node : nodes) {
      if(node.index == 0) {
         continue;
      }
      const std::string prefix = make_prefix(target_keys[node.index]);
      const std::array< std::string, 1 > succ_extra{target_keys[node.index]};

      if(horizon_config_.transition_mode == Mode::Full) {
         const auto succ_fact_keys = encode_state_facts_with_prefix(
            node.state, prefix, succ_extra, false
         );
         if(encode_actions and node.action.has_value()) {
            encode_action_with_prefix(*node.action, prefix, succ_extra);
         }
         if(not goals.static_goals.empty()) {
            encode_goal_satisfaction_with_prefix.template operator()< mimir::formalism::StaticTag >(
               std::span{goals.static_goals},
               goals.static_goal_levels,
               succ_fact_keys,
               prefix,
               succ_extra
            );
         }
         if(not goals.fluent_goals.empty()) {
            encode_goal_satisfaction_with_prefix.template operator()< mimir::formalism::FluentTag >(
               std::span{goals.fluent_goals},
               goals.fluent_goal_levels,
               succ_fact_keys,
               prefix,
               succ_extra
            );
         }
         if(not goals.derived_goals.empty()) {
            encode_goal_satisfaction_with_prefix
               .template operator()< mimir::formalism::DerivedTag >(
                  std::span{goals.derived_goals},
                  goals.derived_goal_levels,
                  succ_fact_keys,
                  prefix,
                  succ_extra
               );
         }
      } else if(horizon_config_.transition_mode == Mode::Delta) {
         const auto& repos = node.state.get_problem().get_repositories();
         const auto succ_fluents = repos
                                      .get_ground_atoms_from_indices< mimir::formalism::FluentTag >(
                                         node.state.get_atoms< mimir::formalism::FluentTag >()
                                      );
         const auto
            succ_derived = repos.get_ground_atoms_from_indices< mimir::formalism::DerivedTag >(
               node.state.get_atoms< mimir::formalism::DerivedTag >()
            );
         hash_set< int > added_fluents;
         hash_set< int > removed_fluents;
         hash_set< int > added_derived;
         hash_set< int > removed_derived;

         hash_set< int > succ_fluent_indices;
         for(const auto& atom : succ_fluents) {
            if(atom->get_predicate()->get_arity() == 0 and not config_.add_nullary_predicates) {
               continue;
            }
            succ_fluent_indices.insert(atom->get_index());
            if(not root_fluent_indices.contains(atom->get_index())) {
               added_fluents.insert(atom->get_index());
               encode_literal_atom_with_prefix(atom, true, prefix, succ_extra);
            }
         }
         for(const auto& idx : root_fluent_indices) {
            if(not succ_fluent_indices.contains(idx)) {
               removed_fluents.insert(idx);
               auto atom = repos.get_ground_atom< mimir::formalism::FluentTag >(idx);
               if(atom->get_predicate()->get_arity() == 0 and not config_.add_nullary_predicates) {
                  continue;
               }
               encode_literal_atom_with_prefix(atom, false, prefix, succ_extra);
            }
         }

         hash_set< int > succ_derived_indices;
         for(const auto& atom : succ_derived) {
            if(atom->get_predicate()->get_arity() == 0 and not config_.add_nullary_predicates) {
               continue;
            }
            succ_derived_indices.insert(atom->get_index());
            if(not root_derived_indices.contains(atom->get_index())) {
               added_derived.insert(atom->get_index());
               encode_literal_atom_with_prefix(atom, true, prefix, succ_extra);
            }
         }
         for(const auto& idx : root_derived_indices) {
            if(not succ_derived_indices.contains(idx)) {
               removed_derived.insert(idx);
               auto atom = repos.get_ground_atom< mimir::formalism::DerivedTag >(idx);
               if(atom->get_predicate()->get_arity() == 0 and not config_.add_nullary_predicates) {
                  continue;
               }
               encode_literal_atom_with_prefix(atom, false, prefix, succ_extra);
            }
         }

         if(encode_actions and node.action.has_value()) {
            encode_action_with_prefix(*node.action, prefix, succ_extra);
         }

         if(relation_dict_.goal_satisfaction_derivations.size() > 0) {
            auto encode_delta_satisfaction =
               [&]< typename GoalTag >(
                  std::span< const mimir::formalism::GroundLiteral< GoalTag > > goal_list,
                  const hash_map< mimir::formalism::GroundLiteral< GoalTag >, int >& goal_levels,
                  const hash_set< int >& added_set,
                  const hash_set< int >& removed_set
               ) {
                  for(const auto& goal : goal_list) {
                     const auto atom = goal->get_atom();
                     const auto idx = atom->get_index();
                     bool added_match = added_set.contains(idx);
                     bool removed_match = removed_set.contains(idx);
                     std::optional< GoalSatisfaction > sat;
                     if(added_match == goal->get_polarity()) {
                        sat = GoalSatisfaction::added_satisfied;
                     } else if(removed_match != goal->get_polarity()) {
                        sat = GoalSatisfaction::added_unsatisfied;
                     }
                     if(not sat.has_value()) {
                        continue;
                     }
                     if(not relation_dict_.goal_satisfaction_derivations.contains(*sat)) {
                        continue;
                     }
                     encode_literals_with_prefix.template operator()< GoalTag >(
                        std::span{&goal, 1}, goal_levels, prefix, succ_extra, *sat
                     );
                  }
               };

            encode_delta_satisfaction.template operator()< mimir::formalism::StaticTag >(
               std::span{goals.static_goals},
               goals.static_goal_levels,
               hash_set< int >{},
               hash_set< int >{}
            );
            encode_delta_satisfaction.template operator()< mimir::formalism::FluentTag >(
               std::span{goals.fluent_goals},
               goals.fluent_goal_levels,
               added_fluents,
               removed_fluents
            );
            encode_delta_satisfaction.template operator()< mimir::formalism::DerivedTag >(
               std::span{goals.derived_goals},
               goals.derived_goal_levels,
               added_derived,
               removed_derived
            );
         }
      } else if(horizon_config_.transition_mode == Mode::Action) {
         if(encode_actions and node.action.has_value()) {
            encode_action_with_prefix(*node.action, prefix, succ_extra);
         }
      }
   }

   // 4. Parent relations
   if(horizon_config_.enable_parent_relation) {
      for(const auto& pair : dag.transitions()) {
         const int parent_idx = pair.first;
         const int child_idx = pair.second;
         const std::string rel_key = fmt::format(
            "{}({}->{})", horizon_config_.parent_relation, parent_idx, child_idx
         );
         const auto rel_idx = get_or_add_node(
            horizon_config_.parent_relation, rel_key, builder, node_indices, node_names
         );
         const auto p_node = get_or_add_node(
            config_.symbol_type_id, target_keys[parent_idx], builder, node_indices, node_names
         );
         const auto c_node = get_or_add_node(
            config_.symbol_type_id, target_keys[child_idx], builder, node_indices, node_names
         );
         append_edges(
            builder, config_.symbol_type_id, "0", horizon_config_.parent_relation, p_node, rel_idx
         );
         append_edges(
            builder, horizon_config_.parent_relation, "0", config_.symbol_type_id, rel_idx, p_node
         );
         append_edges(
            builder, config_.symbol_type_id, "1", horizon_config_.parent_relation, c_node, rel_idx
         );
         append_edges(
            builder, horizon_config_.parent_relation, "1", config_.symbol_type_id, rel_idx, c_node
         );
      }
   }

   // 5. Sibling/Cousin relations
   if(horizon_config_.enable_sibling_relation || horizon_config_.enable_cousin_relation) {
      hash_map< int, std::vector< int > > parent_to_children;
      for(const auto& pair : dag.transitions()) {
         parent_to_children[pair.first].push_back(pair.second);
      }

      auto emplace_symmetric_relation = [&](const std::string& relation, int a, int b) {
         for(int dir = 0; dir < 2; ++dir) {
            int src = dir == 0 ? a : b;
            int dst = dir == 0 ? b : a;
            const std::string rel_key = fmt::format("{}({}->{})", relation, src, dst);
            const auto rel_idx = get_or_add_node(
               relation, rel_key, builder, node_indices, node_names
            );
            const auto a_node = get_or_add_node(
               config_.symbol_type_id, target_keys[src], builder, node_indices, node_names
            );
            const auto b_node = get_or_add_node(
               config_.symbol_type_id, target_keys[dst], builder, node_indices, node_names
            );
            append_edges(builder, config_.symbol_type_id, "0", relation, a_node, rel_idx);
            append_edges(builder, relation, "0", config_.symbol_type_id, rel_idx, a_node);
            append_edges(builder, config_.symbol_type_id, "1", relation, b_node, rel_idx);
            append_edges(builder, relation, "1", config_.symbol_type_id, rel_idx, b_node);
         }
      };

      std::set< std::pair< int, int > > siblings_seen;
      if(horizon_config_.enable_sibling_relation) {
         for(auto& [_, children] : parent_to_children) {
            std::sort(children.begin(), children.end());
            for(size_t i = 0; i < children.size(); ++i) {
               for(size_t j = i + 1; j < children.size(); ++j) {
                  const int a = children[i];
                  const int b = children[j];
                  const auto pair = std::pair{a, b};
                  if(siblings_seen.contains(pair)) {
                     continue;
                  }
                  siblings_seen.insert(pair);
                  emplace_symmetric_relation(horizon_config_.sibling_relation, a, b);
               }
            }
         }
      }

      if(horizon_config_.enable_cousin_relation) {
         std::set< std::pair< int, int > > cousins_seen;
         for(const auto& [g, parents] : parent_to_children) {
            std::vector< int > par = parents;
            std::sort(par.begin(), par.end());
            for(size_t i = 0; i < par.size(); ++i) {
               for(size_t j = i + 1; j < par.size(); ++j) {
                  const int pu = par[i];
                  const int pv = par[j];
                  const auto& cu = parent_to_children[pu];
                  const auto& cv = parent_to_children[pv];
                  for(int u : cu) {
                     for(int v : cv) {
                        if(u == v) {
                           continue;
                        }
                        const int a = std::min(u, v);
                        const int b = std::max(u, v);
                        const auto pair = std::pair{a, b};
                        if(cousins_seen.contains(pair) || siblings_seen.contains(pair)) {
                           continue;
                        }
                        cousins_seen.insert(pair);
                        emplace_symmetric_relation(horizon_config_.cousin_relation, a, b);
                     }
                  }
               }
            }
         }
      }
   }

   // 6. LGAN edges
   if(config_.include_lgan_edges) {
      add_lgan_nn_edges(builder, node_indices, relation_to_symbols, symbol_to_relations);
   }

   // 7. Finalize node names
   for(const auto& [node_type, _] : relation_dict_.arity) {
      if(not node_names.contains(node_type)) {
         builder.set_node_names(node_type, {});
      }
   }

   if(not nodes.empty()) {
      std::vector< int64_t > target_positions;
      std::vector< int64_t > target_depths;
      std::vector< std::string > target_names;
      target_positions.reserve(nodes.size());
      target_depths.reserve(nodes.size());
      target_names.reserve(nodes.size());

      const auto& symbol_indices = node_indices[config_.symbol_type_id];
      for(const auto& node : nodes) {
         const auto key = target_keys[node.index];
         const auto it = symbol_indices.find(key);
         if(it == symbol_indices.end()) {
            continue;
         }
         target_positions.push_back(it->second);
         target_depths.push_back(node.depth);

         std::ostringstream stream;
         stream << node.state;
         target_names.emplace_back(stream.str());
      }

      builder.set_graph_attr("target_positions", std::move(target_positions));
      builder.set_graph_attr("target_depths", std::move(target_depths));
      builder.set_graph_attr("target_names", std::move(target_names));
      builder.set_graph_attr("target_symbol_prefix", horizon_config_.target_symbol_prefix);
      builder.set_graph_attr("parent_relation", horizon_config_.parent_relation);
   }
   if(not node_names.contains(config_.symbol_type_id)) {
      builder.set_node_names(config_.symbol_type_id, {});
      builder.set_object_names({});
   } else {
      const auto& symbol_names = node_names[config_.symbol_type_id];
      builder.set_node_names(config_.symbol_type_id, symbol_names);

      if(target_keys.empty()) {
         builder.set_object_names(symbol_names);
      } else {
         hash_set< std::string > target_set;
         target_set.reserve(target_keys.size());
         for(const auto& key : target_keys) {
            if(not key.empty()) {
               target_set.insert(key);
            }
         }
         std::vector< std::string > object_names;
         object_names.reserve(symbol_names.size());
         for(const auto& name : symbol_names) {
            if(not target_set.contains(name)) {
               object_names.push_back(name);
            }
         }
         builder.set_object_names(std::move(object_names));
      }
   }

   for(const auto& [node_type, names] : node_names) {
      if(node_type == config_.symbol_type_id) {
         continue;
      }
      builder.set_node_names(node_type, names);
   }

   ensure_empty_edge_types(builder);
}

void HorizonHGraphEncoderEngine::configure_relations()
{
   if(horizon_config_.enable_parent_relation) {
      register_relation_type(horizon_config_.parent_relation);
   }
   if(horizon_config_.enable_sibling_relation) {
      register_relation_type(horizon_config_.sibling_relation);
   }
   if(horizon_config_.enable_cousin_relation) {
      register_relation_type(horizon_config_.cousin_relation);
   }
}

void HorizonHGraphEncoderEngine::register_relation_type(const std::string& relation)
{
   relation_dict_.arity[relation] = 2;
   for(int pos = 0; pos < 2; ++pos) {
      const std::string pos_str = std::to_string(pos);
      all_edge_types_.emplace_back(config_.symbol_type_id, pos_str, relation);
      all_edge_types_.emplace_back(relation, pos_str, config_.symbol_type_id);
   }
   std::ranges::sort(all_edge_types_);
   all_edge_types_.erase(std::ranges::unique(all_edge_types_).begin(), all_edge_types_.end());
}

std::string HorizonHGraphEncoderEngine::target_node_key(int idx) const
{
   return fmt::format("{}{}", horizon_config_.target_symbol_prefix, idx);
}

}  // namespace mifrost
