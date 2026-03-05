#include "horizon_hgraph_encoder.hpp"

#include <fmt/format.h>

#include <algorithm>
#include <array>
#include <mimir/search/formatter.hpp>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "mifrost/core/schema_key_separators.hpp"
#include "mifrost/input_handling/batch_input_parser.hpp"

namespace mifrost {

namespace {

HorizonHGraphEncoderEngine::Config normalize_horizon_config(
   HorizonHGraphEncoderEngine::Config config
)
{
   if(config.transition_mode == HorizonHGraphEncoderEngine::Mode::Delta
      and not config.support_literals) {
      config.support_literals = true;
   }
   return config;
}

RelationDict build_horizon_relation_dict(
   const mimir::formalism::DomainImpl& domain,
   const HGraphEncoderEngine::Config& config
)
{
   RelationDictConfig rel_config;
   rel_config.max_goal_level = static_cast< int >(config.max_goal_level);
   rel_config.support_literals = config.support_literals;
   rel_config.goal_satisfaction_derivations = config.goal_satisfaction_derivations;
   rel_config.top_type_predicates.insert(config.symbol_type_id);

   std::vector< mimir::formalism::Action > actions;
   if(not config.ignore_actions) {
      actions.assign(domain.get_actions().begin(), domain.get_actions().end());
   }

   return RelationDict(
      domain,
      actions,
      rel_config,
      /*predicate_arity_offset=*/1,
      /*action_arity_offset=*/1
   );
}

}  // namespace

HorizonHGraphEncoderEngine::HorizonHGraphEncoderEngine(const mimir::formalism::DomainImpl& domain)
    : HGraphEncoderEngine(domain, normalize_horizon_config(Config{})),
      horizon_config_(normalize_horizon_config(Config{}))
{
   relation_dict_ = build_horizon_relation_dict(domain_, config_);
   configure_relations();
}

HorizonHGraphEncoderEngine::HorizonHGraphEncoderEngine(
   const mimir::formalism::DomainImpl& domain,
   Config config
)
    : HGraphEncoderEngine(domain, normalize_horizon_config(config)),
      horizon_config_(normalize_horizon_config(std::move(config)))
{
   relation_dict_ = build_horizon_relation_dict(domain_, config_);
   configure_relations();
}

HorizonHGraphEncoderEngine::HorizonHGraphEncoderEngine(mimir::formalism::Domain domain)
    : HGraphEncoderEngine(std::move(domain), normalize_horizon_config(Config{})),
      horizon_config_(normalize_horizon_config(Config{}))
{
   relation_dict_ = build_horizon_relation_dict(domain_, config_);
   configure_relations();
}

HorizonHGraphEncoderEngine::HorizonHGraphEncoderEngine(
   mimir::formalism::Domain domain,
   Config config
)
    : HGraphEncoderEngine(domain, normalize_horizon_config(config)),
      horizon_config_(normalize_horizon_config(std::move(config)))
{
   relation_dict_ = build_horizon_relation_dict(domain_, config_);
   configure_relations();
}

void HorizonHGraphEncoderEngine::encode(
   const mimir::search::State& root,
   const TransitionDAG& dag,
   const GoalInputs& goals,
   BatchBuilder& builder
)
{
   if(dag.root() != root) {
      throw std::invalid_argument("dag root must match root state");
   }
   encode_impl(root, dag, goals, builder);
}

void HorizonHGraphEncoderEngine::update_relations(RelationDict relation_dict)
{
   HGraphEncoderEngine::update_relations(std::move(relation_dict));
   configure_relations();
}

void HorizonHGraphEncoderEngine::encode_impl(
   const mimir::search::State& root,
   const TransitionDAG& dag,
   const GoalInputs& goals,
   BatchBuilder& builder
)
{
   auto& workspace = init_hetero_workspace(builder);
   auto& node_indices = workspace.node_indices;
   auto& node_names = workspace.node_names;
   auto& relation_to_symbols = workspace.relation_to_symbols;
   auto& symbol_to_relations = workspace.symbol_to_relations;

   if(horizon_config_.transition_mode == Mode::Delta and not config_.support_literals) {
      throw std::invalid_argument("Delta horizon encoding requires support_literals=true.");
   }
   if(horizon_config_.transition_mode == Mode::Action and config_.ignore_actions) {
      throw std::invalid_argument("Action horizon encoding requires ignore_actions=false.");
   }

   auto make_prefix = [](const std::string& target_key) {
      std::string prefix = target_key;
      prefix.push_back(schema_key::kEdgeTypeSeparator);
      return prefix;
   };

   // Helpers for horizon-specific encoding (extra_objects first, prefixed node keys)
   auto encode_atoms_with_prefix = [&](
                                      auto atoms,
                                      int target_idx,
                                      const std::string& prefix,
                                      std::span< const std::string > extra_objects
                                   ) {
      hash_set< uint64_t > fact_keys;
      for(const auto& atom : atoms) {
         const auto predicate = atom->get_predicate();
         if(predicate->get_arity() == 0 and not config_.add_nullary_predicates) {
            continue;
         }
         const std::string node_type = RelationFormatter::format_predicate(predicate);
         const uint64_t node_key = pack_u32_u32(
            static_cast< uint32_t >(target_idx), static_cast< uint32_t >(atom->get_index())
         );
         const std::string node_name = config_.export_node_names
                                          ? (prefix + RelationFormatter::format_atom(atom))
                                          : "";
         const auto relation_idx = get_or_add_relation_node_u64(
            node_type, node_key, builder, node_indices, node_names, node_name
         );

         std::vector< int64_t > object_symbol_ids;
         if(predicate->get_arity() == 0) {
            const auto nullary_idx = get_or_add_symbol_special_node(
               config_.nullary_object_name, config_.nullary_object_name, builder, node_names
            );
            (void) nullary_idx;
            object_symbol_ids.emplace_back(
               get_or_assign_special_symbol_id(config_.nullary_object_name)
            );
         } else {
            for(const auto& obj : atom->get_objects()) {
               const auto obj_idx = get_or_add_symbol_object_node(obj, builder, node_names);
               (void) obj_idx;
               object_symbol_ids.emplace_back(static_cast< int64_t >(obj->get_index()));
            }
         }

         size_t pos = 0;
         std::vector< int64_t > extra_symbol_ids;
         extra_symbol_ids.reserve(extra_objects.size());
         for(const auto& obj_key : extra_objects) {
            const auto obj_idx = get_or_add_symbol_special_node(
               obj_key, obj_key, builder, node_names
            );
            const auto symbol_id = get_or_assign_special_symbol_id(obj_key);
            extra_symbol_ids.emplace_back(symbol_id);
            const std::string pos_str = std::to_string(pos++);
            append_edges(
               builder, config_.symbol_type_id, pos_str, node_type, obj_idx, relation_idx
            );
            append_edges(
               builder, node_type, pos_str, config_.symbol_type_id, relation_idx, obj_idx
            );
         }

         for(const auto symbol_id : object_symbol_ids) {
            const auto obj_idx = workspace_.symbol_indices.at(symbol_id);
            const std::string pos_str = std::to_string(pos++);
            append_edges(
               builder, config_.symbol_type_id, pos_str, node_type, obj_idx, relation_idx
            );
            append_edges(
               builder, node_type, pos_str, config_.symbol_type_id, relation_idx, obj_idx
            );
         }

         const auto rel_ref = relation_ref_for(node_type, relation_idx);
         track_relation_symbols_if_enabled(
            rel_ref,
            std::span{object_symbol_ids},
            std::span{extra_symbol_ids},
            relation_to_symbols,
            symbol_to_relations
         );
         fact_keys.insert(node_key);
      }
      return fact_keys;
   };

   auto encode_state_facts_with_prefix = [&](
                                            const mimir::search::State& state,
                                            int target_idx,
                                            const std::string& prefix,
                                            std::span< const std::string > extra_objects,
                                            bool include_static
                                         ) {
      hash_set< uint64_t > fact_keys;
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
            auto keys = encode_atoms_with_prefix(
               std::span{atoms}, target_idx, prefix, extra_objects
            );
            for(const auto& key : keys) {
               fact_keys.insert(key);
            }
         }
      }

      const auto fluent_atoms = repos.get_ground_atoms_from_indices< mimir::formalism::FluentTag >(
         state.get_atoms< mimir::formalism::FluentTag >()
      );
      auto fluent_keys = encode_atoms_with_prefix(
         std::span{fluent_atoms}, target_idx, prefix, extra_objects
      );
      for(const auto& key : fluent_keys) {
         fact_keys.insert(key);
      }

      const auto derived_atoms = repos
                                    .get_ground_atoms_from_indices< mimir::formalism::DerivedTag >(
                                       state.get_atoms< mimir::formalism::DerivedTag >()
                                    );
      auto derived_keys = encode_atoms_with_prefix(
         std::span{derived_atoms}, target_idx, prefix, extra_objects
      );
      for(const auto& key : derived_keys) {
         fact_keys.insert(key);
      }

      return fact_keys;
   };

   auto encode_literals_with_prefix =
      [&]< typename GoalTag >(
         std::span< const mimir::formalism::GroundLiteral< GoalTag > > literals,
         const hash_map< mimir::formalism::GroundLiteral< GoalTag >, size_t >& goal_levels,
         int target_idx,
         const std::string& prefix,
         std::span< const std::string > extra_objects,
         std::optional< GoalSatisfaction > satisfaction_override = std::nullopt
      ) {
         for(const auto& literal : literals) {
            const auto atom = literal->get_atom();
            const auto predicate = atom->get_predicate();
            const std::optional< size_t > goal_level = goal_levels.contains(literal)
                                                          ? std::optional< size_t >(
                                                               goal_levels.at(literal)
                                                            )
                                                          : std::nullopt;

            const GoalSatisfaction satisfaction = satisfaction_override.has_value()
                                                     ? *satisfaction_override
                                                     : GoalSatisfaction::none;

            std::string node_type;
            std::string literal_name;
            auto format_with = [&](auto level_arg, auto satisfaction_arg) {
               node_type = RelationFormatter::format_predicate(
                  predicate, level_arg, satisfaction_arg, literal->get_polarity()
               );
               if(config_.export_node_names) {
                  literal_name = prefix
                                 + RelationFormatter::format_literal< GoalTag >(
                                    literal, level_arg, satisfaction_arg
                                 );
               }
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

            const uint64_t relation_key = pack_u32_u32(
               static_cast< uint32_t >(target_idx), static_cast< uint32_t >(atom->get_index())
            );
            const auto relation_idx = get_or_add_relation_node_u64(
               node_type, relation_key, builder, node_indices, node_names, literal_name
            );

            std::vector< int64_t > object_symbol_ids;
            if(predicate->get_arity() == 0) {
               if(not config_.add_nullary_predicates) {
                  continue;
               }
               const auto nullary_idx = get_or_add_symbol_special_node(
                  config_.nullary_object_name, config_.nullary_object_name, builder, node_names
               );
               (void) nullary_idx;
               object_symbol_ids.emplace_back(
                  get_or_assign_special_symbol_id(config_.nullary_object_name)
               );
            } else {
               for(const auto& obj : atom->get_objects()) {
                  const auto obj_idx = get_or_add_symbol_object_node(obj, builder, node_names);
                  (void) obj_idx;
                  object_symbol_ids.emplace_back(static_cast< int64_t >(obj->get_index()));
               }
            }

            size_t pos = 0;
            std::vector< int64_t > extra_symbol_ids;
            extra_symbol_ids.reserve(extra_objects.size());
            for(const auto& obj_key : extra_objects) {
               const auto obj_idx = get_or_add_symbol_special_node(
                  obj_key, obj_key, builder, node_names
               );
               const auto symbol_id = get_or_assign_special_symbol_id(obj_key);
               extra_symbol_ids.emplace_back(symbol_id);
               const std::string pos_str = std::to_string(pos++);
               append_edges(
                  builder, config_.symbol_type_id, pos_str, node_type, obj_idx, relation_idx
               );
               append_edges(
                  builder, node_type, pos_str, config_.symbol_type_id, relation_idx, obj_idx
               );
            }
            for(const auto symbol_id : object_symbol_ids) {
               const auto obj_idx = workspace_.symbol_indices.at(symbol_id);
               const std::string pos_str = std::to_string(pos++);
               append_edges(
                  builder, config_.symbol_type_id, pos_str, node_type, obj_idx, relation_idx
               );
               append_edges(
                  builder, node_type, pos_str, config_.symbol_type_id, relation_idx, obj_idx
               );
            }

            const auto rel_ref = relation_ref_for(node_type, relation_idx);
            track_relation_symbols_if_enabled(
               rel_ref,
               std::span{object_symbol_ids},
               std::span{extra_symbol_ids},
               relation_to_symbols,
               symbol_to_relations
            );
         }
      };

   auto encode_literal_atom_with_prefix = [&](
                                             auto atom,
                                             bool polarity,
                                             int target_idx,
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
      const std::string node_name = config_.export_node_names ? (prefix + literal_str) : "";

      const uint64_t relation_key = pack_u32_u32(
         static_cast< uint32_t >(target_idx), static_cast< uint32_t >(atom->get_index())
      );
      const auto relation_idx = get_or_add_relation_node_u64(
         node_type, relation_key, builder, node_indices, node_names, node_name
      );

      std::vector< int64_t > object_symbol_ids;
      if(predicate->get_arity() == 0) {
         const auto nullary_idx = get_or_add_symbol_special_node(
            config_.nullary_object_name, config_.nullary_object_name, builder, node_names
         );
         (void) nullary_idx;
         object_symbol_ids.emplace_back(
            get_or_assign_special_symbol_id(config_.nullary_object_name)
         );
      } else {
         for(const auto& obj : atom->get_objects()) {
            const auto obj_idx = get_or_add_symbol_object_node(obj, builder, node_names);
            (void) obj_idx;
            object_symbol_ids.emplace_back(static_cast< int64_t >(obj->get_index()));
         }
      }

      size_t pos = 0;
      std::vector< int64_t > extra_symbol_ids;
      extra_symbol_ids.reserve(extra_objects.size());
      for(const auto& obj_key : extra_objects) {
         const auto obj_idx = get_or_add_symbol_special_node(obj_key, obj_key, builder, node_names);
         const auto symbol_id = get_or_assign_special_symbol_id(obj_key);
         extra_symbol_ids.emplace_back(symbol_id);
         const std::string pos_str = std::to_string(pos++);
         append_edges(builder, config_.symbol_type_id, pos_str, node_type, obj_idx, relation_idx);
         append_edges(builder, node_type, pos_str, config_.symbol_type_id, relation_idx, obj_idx);
      }
      for(const auto symbol_id : object_symbol_ids) {
         const auto obj_idx = workspace_.symbol_indices.at(symbol_id);
         const std::string pos_str = std::to_string(pos++);
         append_edges(builder, config_.symbol_type_id, pos_str, node_type, obj_idx, relation_idx);
         append_edges(builder, node_type, pos_str, config_.symbol_type_id, relation_idx, obj_idx);
      }

      const auto rel_ref = relation_ref_for(node_type, relation_idx);
      track_relation_symbols_if_enabled(
         rel_ref,
         std::span{object_symbol_ids},
         std::span{extra_symbol_ids},
         relation_to_symbols,
         symbol_to_relations
      );
   };

   auto encode_goal_satisfaction_with_prefix =
      [&]< typename GoalTag >(
         std::span< const mimir::formalism::GroundLiteral< GoalTag > > literals,
         const hash_map< mimir::formalism::GroundLiteral< GoalTag >, size_t >& goal_levels,
         const hash_set< uint64_t >& fact_keys,
         int target_idx,
         const std::string& prefix,
         std::span< const std::string > extra_objects
      ) {
         for(const auto& goal : literals) {
            const auto atom = goal->get_atom();
            const auto predicate = atom->get_predicate();
            const auto key = pack_u32_u32(
               static_cast< uint32_t >(target_idx), static_cast< uint32_t >(atom->get_index())
            );
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
            std::string node_name;
            if(goal_level.has_value()) {
               const GoalLevel level(*goal_level);
               node_type = RelationFormatter::format_predicate(
                  predicate, level, sat, goal->get_polarity()
               );
               if(config_.export_node_names) {
                  node_name = prefix
                              + RelationFormatter::format_literal< GoalTag >(goal, level, sat);
               }
            } else {
               node_type = RelationFormatter::format_predicate(
                  predicate, std::nullopt, sat, goal->get_polarity()
               );
               if(config_.export_node_names) {
                  node_name = prefix
                              + RelationFormatter::format_literal< GoalTag >(
                                 goal, std::nullopt, sat
                              );
               }
            }

            const uint64_t relation_key = pack_u32_u32(
               static_cast< uint32_t >(target_idx), static_cast< uint32_t >(atom->get_index())
            );
            const auto relation_idx = get_or_add_relation_node_u64(
               node_type, relation_key, builder, node_indices, node_names, node_name
            );

            std::vector< int64_t > object_symbol_ids;
            if(predicate->get_arity() == 0) {
               if(not config_.add_nullary_predicates) {
                  continue;
               }
               const auto nullary_idx = get_or_add_symbol_special_node(
                  config_.nullary_object_name, config_.nullary_object_name, builder, node_names
               );
               (void) nullary_idx;
               object_symbol_ids.emplace_back(
                  get_or_assign_special_symbol_id(config_.nullary_object_name)
               );
            } else {
               for(const auto& obj : atom->get_objects()) {
                  const auto obj_idx = get_or_add_symbol_object_node(obj, builder, node_names);
                  (void) obj_idx;
                  object_symbol_ids.emplace_back(static_cast< int64_t >(obj->get_index()));
               }
            }

            size_t pos = 0;
            std::vector< int64_t > extra_symbol_ids;
            extra_symbol_ids.reserve(extra_objects.size());
            for(const auto& obj_key : extra_objects) {
               const auto obj_idx = get_or_add_symbol_special_node(
                  obj_key, obj_key, builder, node_names
               );
               const auto symbol_id = get_or_assign_special_symbol_id(obj_key);
               extra_symbol_ids.emplace_back(symbol_id);
               const std::string pos_str = std::to_string(pos++);
               append_edges(
                  builder, config_.symbol_type_id, pos_str, node_type, obj_idx, relation_idx
               );
               append_edges(
                  builder, node_type, pos_str, config_.symbol_type_id, relation_idx, obj_idx
               );
            }
            for(const auto symbol_id : object_symbol_ids) {
               const auto obj_idx = workspace_.symbol_indices.at(symbol_id);
               const std::string pos_str = std::to_string(pos++);
               append_edges(
                  builder, config_.symbol_type_id, pos_str, node_type, obj_idx, relation_idx
               );
               append_edges(
                  builder, node_type, pos_str, config_.symbol_type_id, relation_idx, obj_idx
               );
            }

            const auto rel_ref = relation_ref_for(node_type, relation_idx);
            track_relation_symbols_if_enabled(
               rel_ref,
               std::span{object_symbol_ids},
               std::span{extra_symbol_ids},
               relation_to_symbols,
               symbol_to_relations
            );
         }
      };

   auto encode_action_with_prefix = [&](
                                       const mimir::formalism::GroundAction& action,
                                       int target_idx,
                                       const std::string& prefix,
                                       std::span< const std::string > extra_objects
                                    ) {
      const std::string node_type = RelationFormatter::format_action_schema(*action->get_action());
      const std::string node_name = config_.export_node_names
                                       ? (prefix + RelationFormatter::format_action(action))
                                       : "";
      const uint64_t relation_key = pack_u32_u32(
         static_cast< uint32_t >(target_idx), static_cast< uint32_t >(action->get_index())
      );
      const auto relation_idx = get_or_add_relation_node_u64(
         node_type, relation_key, builder, node_indices, node_names, node_name
      );

      std::vector< int64_t > object_symbol_ids;
      for(const auto& obj_key : extra_objects) {
         const auto obj_idx = get_or_add_symbol_special_node(obj_key, obj_key, builder, node_names);
         (void) obj_idx;
         object_symbol_ids.emplace_back(get_or_assign_special_symbol_id(obj_key));
      }
      for(const auto& obj : action->get_objects()) {
         const auto obj_idx = get_or_add_symbol_object_node(obj, builder, node_names);
         (void) obj_idx;
         object_symbol_ids.emplace_back(static_cast< int64_t >(obj->get_index()));
      }

      for(size_t pos = 0; pos < object_symbol_ids.size(); ++pos) {
         const auto obj_idx = workspace_.symbol_indices.at(object_symbol_ids[pos]);
         const std::string pos_str = std::to_string(pos);
         append_edges(builder, config_.symbol_type_id, pos_str, node_type, obj_idx, relation_idx);
         append_edges(builder, node_type, pos_str, config_.symbol_type_id, relation_idx, obj_idx);
      }

      const auto rel_ref = relation_ref_for(node_type, relation_idx);
      track_relation_symbols_if_enabled(
         rel_ref,
         std::span{object_symbol_ids},
         std::span< const int64_t >{},
         relation_to_symbols,
         symbol_to_relations
      );
   };

   // 1. Create target nodes first to keep them contiguous in symbol list.
   const auto& nodes = dag.nodes();
   const int root_index = dag.root_index();
   std::vector< std::string > target_keys(nodes.size());
   for(const auto& node : nodes) {
      if(node.index >= static_cast< int >(target_keys.size())) {
         target_keys.resize(node.index + 1);
      }
      const std::string key = target_node_key(node.index);
      target_keys[node.index] = key;
      get_or_add_symbol_special_node(key, key, builder, node_names);
      if(config_.include_lgan_edges
         and not(horizon_config_.exclude_root_candidate and node.index == root_index)) {
         const auto symbol_id_it = workspace.symbol_key_to_id.find(key);
         if(symbol_id_it != workspace.symbol_key_to_id.end()) {
            workspace.lgan_target_symbol_ids.insert(symbol_id_it->second);
         }
      }
   }

   // 2. Encode root state (objects then facts/goals)
   encode_objects(root, builder, node_indices, node_names);
   const std::string root_prefix = make_prefix(target_keys[0]);
   const std::array< std::string, 1 > root_extra{target_keys[0]};

   const auto root_fact_keys = encode_state_facts_with_prefix(
      root, 0, root_prefix, root_extra, config_.include_static
   );

   encode_literals_with_prefix.template operator()< mimir::formalism::StaticTag >(
      std::span{goals.static_goals}, goals.static_goal_levels, 0, root_prefix, root_extra
   );
   encode_literals_with_prefix.template operator()< mimir::formalism::FluentTag >(
      std::span{goals.fluent_goals}, goals.fluent_goal_levels, 0, root_prefix, root_extra
   );
   encode_literals_with_prefix.template operator()< mimir::formalism::DerivedTag >(
      std::span{goals.derived_goals}, goals.derived_goal_levels, 0, root_prefix, root_extra
   );

   if(not goals.static_goals.empty()) {
      encode_goal_satisfaction_with_prefix.template operator()< mimir::formalism::StaticTag >(
         std::span{goals.static_goals},
         goals.static_goal_levels,
         root_fact_keys,
         0,
         root_prefix,
         root_extra
      );
   }
   if(not goals.fluent_goals.empty()) {
      encode_goal_satisfaction_with_prefix.template operator()< mimir::formalism::FluentTag >(
         std::span{goals.fluent_goals},
         goals.fluent_goal_levels,
         root_fact_keys,
         0,
         root_prefix,
         root_extra
      );
   }
   if(not goals.derived_goals.empty()) {
      encode_goal_satisfaction_with_prefix.template operator()< mimir::formalism::DerivedTag >(
         std::span{goals.derived_goals},
         goals.derived_goal_levels,
         root_fact_keys,
         0,
         root_prefix,
         root_extra
      );
   }

   const bool encode_actions = (not config_.ignore_actions)
                               or (horizon_config_.transition_mode == Mode::Action);

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
            node.state, node.index, prefix, succ_extra, false
         );
         if(encode_actions and node.action.has_value()) {
            encode_action_with_prefix(*node.action, node.index, prefix, succ_extra);
         }
         if(not goals.static_goals.empty()) {
            encode_goal_satisfaction_with_prefix.template operator()< mimir::formalism::StaticTag >(
               std::span{goals.static_goals},
               goals.static_goal_levels,
               succ_fact_keys,
               node.index,
               prefix,
               succ_extra
            );
         }
         if(not goals.fluent_goals.empty()) {
            encode_goal_satisfaction_with_prefix.template operator()< mimir::formalism::FluentTag >(
               std::span{goals.fluent_goals},
               goals.fluent_goal_levels,
               succ_fact_keys,
               node.index,
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
                  node.index,
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
               encode_literal_atom_with_prefix(atom, true, node.index, prefix, succ_extra);
            }
         }
         for(const auto& idx : root_fluent_indices) {
            if(not succ_fluent_indices.contains(idx)) {
               removed_fluents.insert(idx);
               auto atom = repos.get_ground_atom< mimir::formalism::FluentTag >(idx);
               if(atom->get_predicate()->get_arity() == 0 and not config_.add_nullary_predicates) {
                  continue;
               }
               encode_literal_atom_with_prefix(atom, false, node.index, prefix, succ_extra);
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
               encode_literal_atom_with_prefix(atom, true, node.index, prefix, succ_extra);
            }
         }
         for(const auto& idx : root_derived_indices) {
            if(not succ_derived_indices.contains(idx)) {
               removed_derived.insert(idx);
               auto atom = repos.get_ground_atom< mimir::formalism::DerivedTag >(idx);
               if(atom->get_predicate()->get_arity() == 0 and not config_.add_nullary_predicates) {
                  continue;
               }
               encode_literal_atom_with_prefix(atom, false, node.index, prefix, succ_extra);
            }
         }

         if(encode_actions and node.action.has_value()) {
            encode_action_with_prefix(*node.action, node.index, prefix, succ_extra);
         }

         if(relation_dict_.goal_satisfaction_derivations.size() > 0) {
            auto encode_delta_satisfaction =
               [&]< typename GoalTag >(
                  std::span< const mimir::formalism::GroundLiteral< GoalTag > > goal_list,
                  const hash_map< mimir::formalism::GroundLiteral< GoalTag >, size_t >& goal_levels,
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
                        std::span{&goal, 1}, goal_levels, node.index, prefix, succ_extra, *sat
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
            encode_action_with_prefix(*node.action, node.index, prefix, succ_extra);
         }
      }
   }

   // 4. Parent relations
   if(horizon_config_.enable_parent_relation) {
      for(const auto& pair : dag.transitions()) {
         const int parent_idx = pair.first;
         const int child_idx = pair.second;
         const uint64_t rel_key = pack_u32_u32(
            static_cast< uint32_t >(parent_idx), static_cast< uint32_t >(child_idx)
         );
         const std::string rel_name = config_.export_node_names
                                         ? fmt::format(
                                              "{}({}->{})",
                                              horizon_config_.parent_relation,
                                              parent_idx,
                                              child_idx
                                           )
                                         : "";
         const auto rel_idx = get_or_add_relation_node_u64(
            horizon_config_.parent_relation, rel_key, builder, node_indices, node_names, rel_name
         );
         const auto p_node = get_or_add_symbol_special_node(
            target_keys[parent_idx], target_keys[parent_idx], builder, node_names
         );
         const auto c_node = get_or_add_symbol_special_node(
            target_keys[child_idx], target_keys[child_idx], builder, node_names
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
         if(config_.include_lgan_edges) {
            const auto parent_symbol_id_it = workspace.symbol_key_to_id.find(
               target_keys[parent_idx]
            );
            const auto child_symbol_id_it = workspace.symbol_key_to_id.find(target_keys[child_idx]);
            if(parent_symbol_id_it != workspace.symbol_key_to_id.end()
               and child_symbol_id_it != workspace.symbol_key_to_id.end()) {
               const std::array< int64_t, 2 > object_symbol_ids = {
                  parent_symbol_id_it->second,
                  child_symbol_id_it->second,
               };
               track_relation_symbols_if_enabled(
                  relation_ref_for(horizon_config_.parent_relation, rel_idx),
                  std::span{object_symbol_ids},
                  std::span< const int64_t >{},
                  relation_to_symbols,
                  symbol_to_relations
               );
            }
         }
      }
   }

   // 5. Sibling/Cousin relations
   if(horizon_config_.enable_sibling_relation or horizon_config_.enable_cousin_relation) {
      hash_map< int, std::vector< int > > parent_to_children;
      for(const auto& pair : dag.transitions()) {
         parent_to_children[pair.first].push_back(pair.second);
      }

      auto emplace_symmetric_relation = [&](const std::string& relation, int a, int b) {
         for(int dir = 0; dir < 2; ++dir) {
            int src = dir == 0 ? a : b;
            int dst = dir == 0 ? b : a;
            const uint64_t rel_key = pack_u32_u32(
               static_cast< uint32_t >(src), static_cast< uint32_t >(dst)
            );
            const std::string rel_name = config_.export_node_names
                                            ? fmt::format("{}({}->{})", relation, src, dst)
                                            : "";
            const auto rel_idx = get_or_add_relation_node_u64(
               relation, rel_key, builder, node_indices, node_names, rel_name
            );
            const auto a_node = get_or_add_symbol_special_node(
               target_keys[src], target_keys[src], builder, node_names
            );
            const auto b_node = get_or_add_symbol_special_node(
               target_keys[dst], target_keys[dst], builder, node_names
            );
            append_edges(builder, config_.symbol_type_id, "0", relation, a_node, rel_idx);
            append_edges(builder, relation, "0", config_.symbol_type_id, rel_idx, a_node);
            append_edges(builder, config_.symbol_type_id, "1", relation, b_node, rel_idx);
            append_edges(builder, relation, "1", config_.symbol_type_id, rel_idx, b_node);
            if(config_.include_lgan_edges) {
               const auto src_symbol_id_it = workspace.symbol_key_to_id.find(target_keys[src]);
               const auto dst_symbol_id_it = workspace.symbol_key_to_id.find(target_keys[dst]);
               if(src_symbol_id_it != workspace.symbol_key_to_id.end()
                  and dst_symbol_id_it != workspace.symbol_key_to_id.end()) {
                  const std::array< int64_t, 2 > object_symbol_ids = {
                     src_symbol_id_it->second,
                     dst_symbol_id_it->second,
                  };
                  track_relation_symbols_if_enabled(
                     relation_ref_for(relation, rel_idx),
                     std::span{object_symbol_ids},
                     std::span< const int64_t >{},
                     relation_to_symbols,
                     symbol_to_relations
                  );
               }
            }
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
                  const auto cu_it = parent_to_children.find(pu);
                  const auto cv_it = parent_to_children.find(pv);
                  if(cu_it == parent_to_children.end() or cv_it == parent_to_children.end()) {
                     continue;
                  }
                  const auto& cu = cu_it->second;
                  const auto& cv = cv_it->second;
                  for(int u : cu) {
                     for(int v : cv) {
                        if(u == v) {
                           continue;
                        }
                        const int a = std::min(u, v);
                        const int b = std::max(u, v);
                        const auto pair = std::pair{a, b};
                        if(cousins_seen.contains(pair) or siblings_seen.contains(pair)) {
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
   maybe_add_lgan_edges(builder, workspace);

   TargetColumns target_columns;
   const bool export_state_targets = has_target_source(TargetSource::States);
   if(not nodes.empty()) {
      const size_t candidate_count = (horizon_config_.exclude_root_candidate and not nodes.empty())
                                        ? (nodes.size() - 1)
                                        : nodes.size();
      struct CandidateRow {
         int64_t position = 0;
         int64_t index = 0;
         int64_t depth = 0;
         std::optional< int64_t > explicit_candidate_id = std::nullopt;
         std::string name;
      };
      std::vector< CandidateRow > candidate_rows;
      candidate_rows.reserve(candidate_count);

      for(const auto& node : nodes) {
         if(horizon_config_.exclude_root_candidate and node.index == root_index) {
            continue;
         }
         const auto key = target_keys[node.index];
         const auto id_it = workspace.symbol_key_to_id.find(key);
         if(id_it == workspace.symbol_key_to_id.end()) {
            continue;
         }
         const auto it = workspace.symbol_indices.find(id_it->second);
         if(it == workspace.symbol_indices.end()) {
            continue;
         }

         std::ostringstream stream;
         stream << node.state;
         candidate_rows.push_back(
            CandidateRow{
               .position = it->second,
               .index = node.index,
               .depth = node.depth,
               .explicit_candidate_id = node.candidate_id,
               .name = stream.str(),
            }
         );
      }

      bool has_explicit_candidate_ids = false;
      std::optional< int64_t > first_missing_candidate_id_node = std::nullopt;
      for(const auto& row : candidate_rows) {
         if(row.explicit_candidate_id.has_value()) {
            has_explicit_candidate_ids = true;
         } else if(not first_missing_candidate_id_node.has_value()) {
            first_missing_candidate_id_node = row.index;
         }
      }
      if(has_explicit_candidate_ids and first_missing_candidate_id_node.has_value()) {
         throw std::invalid_argument(
            "missing candidate_id for target node index "
            + std::to_string(*first_missing_candidate_id_node)
         );
      }

      if(export_state_targets) {
         target_columns.reserve(
            candidate_rows.size(), /*include_depth=*/true, /*include_group=*/true
         );
      }
      hash_set< int64_t > seen_candidate_ids;
      seen_candidate_ids.reserve(candidate_rows.size());
      for(const auto& row : candidate_rows) {
         const int64_t candidate_id = has_explicit_candidate_ids ? *row.explicit_candidate_id
                                                                 : row.index;
         if(not seen_candidate_ids.emplace(candidate_id).second) {
            throw std::invalid_argument("duplicate candidate_id " + std::to_string(candidate_id));
         }
         if(export_state_targets) {
            target_columns.append(
               TargetRecord{
                  .position = row.position,
                  .index = row.index,
                  .candidate_id = candidate_id,
                  .depth = row.depth,
                  .group_id = get_or_assign_target_group_id(TargetSource::States),
                  .name = row.name,
               },
               /*include_depth=*/true,
               /*include_group=*/true
            );
         }
      }
   }

   if(export_state_targets) {
      const TargetMetadataEmitConfig target_emit_config{
         .symbol_type_id = config_.symbol_type_id,
         .symbol_prefix = config_.target_symbol_prefix,
         .include_depth = true,
         .include_group = true,
         .groups = workspace.target_groups,
         .parent_relation = horizon_config_.parent_relation,
      };
      emit_target_metadata(builder, target_columns, target_emit_config);
   }

   std::vector< std::string > object_names_override;
   const std::vector< std::string >* object_names_override_ptr = nullptr;
   if(node_names.contains(config_.symbol_type_id)) {
      const auto& symbol_names = node_names[config_.symbol_type_id];
      if(target_keys.empty()) {
         object_names_override = symbol_names;
      } else {
         hash_set< std::string > target_set;
         target_set.reserve(target_keys.size());
         for(const auto& key : target_keys) {
            if(not key.empty()) {
               target_set.insert(key);
            }
         }
         object_names_override.reserve(symbol_names.size());
         for(const auto& name : symbol_names) {
            if(not target_set.contains(name)) {
               object_names_override.push_back(name);
            }
         }
      }
      object_names_override_ptr = &object_names_override;
   }

   finalize_hetero_encoding(builder, workspace, object_names_override_ptr);
}

BatchBuilder::BatchEncoding HorizonHGraphEncoderEngine::encode_batch(
   const batch_input::parsed::HorizonBatchInputs& inputs
)
{
   BatchBuilder builder;
   builder.set_graph_kind("hetero");

   const size_t state_count = inputs.roots.states.size();
   for(size_t idx = 0; idx < state_count; ++idx) {
      const auto& root_entry = inputs.roots.states[idx];
      const auto& dag_entry = inputs.dags.at(idx);
      const auto& goals_entry = inputs.goals.at(idx);
      const auto& subgoal_layers_entry = inputs.subgoal_layers.at(idx);

      const TransitionDAG default_dag(root_entry.state);
      const TransitionDAG& dag_ref = dag_entry.has_value() ? *dag_entry : default_dag;

      GoalInputs goal_inputs;
      if(goals_entry.has_value()) {
         const auto* layers_ptr = subgoal_layers_entry.has_value() ? &(*subgoal_layers_entry)
                                                                   : nullptr;
         goal_inputs = batch_input::compose_goal_inputs(*goals_entry, layers_ptr);
      } else {
         goal_inputs = batch_input::default_goal_inputs_for_batch_state(root_entry);
         if(subgoal_layers_entry.has_value()) {
            size_t level = 1;
            for(const auto& layer : *subgoal_layers_entry) {
               goal_inputs.extend(layer, level);
               ++level;
            }
         }
      }

      encode(root_entry.state, dag_ref, goal_inputs, builder);
      builder.next_graph();
   }

   return builder.build();
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

   rebuild_all_edge_types();
}

void HorizonHGraphEncoderEngine::register_relation_type(const std::string& relation)
{
   relation_dict_.arity[relation] = 2;
}

std::string HorizonHGraphEncoderEngine::target_node_key(int idx) const
{
   return fmt::format("{}{}", config_.target_symbol_prefix, idx);
}

}  // namespace mifrost
