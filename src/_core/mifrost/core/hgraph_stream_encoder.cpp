#include "hgraph_stream_encoder.hpp"

#include <fmt/format.h>

#include <algorithm>
#include <mimir/formalism/action.hpp>
#include <mimir/formalism/problem.hpp>
#include <numeric>
#include <span>
#include <string>
#include <type_traits>

namespace mifrost {
void HGraphEncoderEngine::append_edges(
   BatchBuilder& builder,
   const std::string& src_type,
   const std::string& rel_type,
   const std::string& dst_type,
   int64_t src,
   int64_t dst
)
{
   const std::array< int64_t, 1 > src_arr{src};
   const std::array< int64_t, 1 > dst_arr{dst};
   builder.add_edges(src_type, rel_type, dst_type, src_arr, dst_arr);
}

HGraphEncoderEngine::HGraphEncoderEngine(const mimir::formalism::DomainImpl& domain)
    : HGraphEncoderEngine(domain, Config{})
{
}

HGraphEncoderEngine::HGraphEncoderEngine(const mimir::formalism::DomainImpl& domain, Config config)
    : domain_(domain), config_(std::move(config))
{
   initialize_from_domain();
}

HGraphEncoderEngine::HGraphEncoderEngine(mimir::formalism::Domain domain)
    : HGraphEncoderEngine(std::move(domain), Config{})
{
}

HGraphEncoderEngine::HGraphEncoderEngine(mimir::formalism::Domain domain, Config config)
    : domain_holder_(std::move(domain)), domain_(*domain_holder_), config_(std::move(config))
{
   initialize_from_domain();
}

void HGraphEncoderEngine::initialize_from_domain()
{
   RelationDictConfig rel_config;
   rel_config.max_goal_level = config_.max_goal_level;
   rel_config.support_literals = config_.support_literals;
   rel_config.goal_satisfaction_derivations = config_.goal_satisfaction_derivations;
   rel_config.top_type_predicates.insert(config_.symbol_type_id);

   std::vector< mimir::formalism::Action > actions;
   if(! config_.ignore_actions) {
      actions.assign(domain_.get_actions().begin(), domain_.get_actions().end());
   }
   relation_dict_ = RelationDict::build(domain_, actions, rel_config);

   for(const auto& [node_type, arity] : relation_dict_.arity) {
      const int effective_arity = (config_.add_nullary_predicates and arity == 0) ? 1 : arity;
      for(int pos = 0; pos < effective_arity; ++pos) {
         const std::string pos_str = std::to_string(pos);
         all_edge_types_.emplace_back(config_.symbol_type_id, pos_str, node_type);
         all_edge_types_.emplace_back(node_type, pos_str, config_.symbol_type_id);
      }
   }
   if(config_.include_lgan_edges) {
      all_edge_types_.emplace_back(
         config_.lgan_nn_edge_pos, config_.lgan_nn_edge_pos, config_.symbol_type_id
      );
   }
   std::ranges::sort(all_edge_types_);
   all_edge_types_.erase(std::ranges::unique(all_edge_types_).begin(), all_edge_types_.end());
}

template < typename GoalTag >
void HGraphEncoderEngine::encode_step_impl(
   const mimir::search::State& state,
   std::span< const mimir::formalism::GroundLiteral< GoalTag > > goals,
   std::span< const mimir::formalism::GroundAction > actions,
   BatchBuilder& builder
)
{
   GoalInputs inputs;
   for(const auto& goal : goals) {
      if constexpr(std::is_same_v< GoalTag, mimir::formalism::StaticTag >) {
         inputs.static_goals.emplace_back(goal);
         inputs.static_goal_levels[goal] = 0;
      } else if constexpr(std::is_same_v< GoalTag, mimir::formalism::FluentTag >) {
         inputs.fluent_goals.emplace_back(goal);
         inputs.fluent_goal_levels[goal] = 0;
      } else {
         inputs.derived_goals.emplace_back(goal);
         inputs.derived_goal_levels[goal] = 0;
      }
   }
   encode_impl(state, inputs, actions, builder);
}

void HGraphEncoderEngine::encode_state_impl(
   const mimir::search::State& state,
   BatchBuilder& builder
)
{
   GoalInputs inputs;
   const auto& problem = state.get_problem();
   for(const auto& goal : problem.get_goal_literals< mimir::formalism::StaticTag >()) {
      inputs.static_goals.emplace_back(goal);
      inputs.static_goal_levels[goal] = 0;
   }
   for(const auto& goal : problem.get_goal_literals< mimir::formalism::FluentTag >()) {
      inputs.fluent_goals.emplace_back(goal);
      inputs.fluent_goal_levels[goal] = 0;
   }
   for(const auto& goal : problem.get_goal_literals< mimir::formalism::DerivedTag >()) {
      inputs.derived_goals.emplace_back(goal);
      inputs.derived_goal_levels[goal] = 0;
   }
   encode_impl(state, inputs, {}, builder);
}

void HGraphEncoderEngine::encode_impl(
   const mimir::search::State& state,
   const GoalInputs& goals,
   std::span< const mimir::formalism::GroundAction > actions,
   BatchBuilder& builder
)
{
   ensure_node_feature_dims(builder);

   hash_map< std::string, hash_map< std::string, int64_t > > node_indices;
   hash_map< std::string, std::vector< std::string > > node_names;
   hash_map< std::string, hash_set< std::string > > relation_to_symbols;
   hash_map< std::string, hash_set< std::string > > symbol_to_relations;

   encode_objects(state, builder, node_indices, node_names);
   const auto fact_keys = encode_facts(
      state, builder, node_indices, node_names, relation_to_symbols, symbol_to_relations
   );
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
   if(! config_.ignore_actions) {
      encode_actions(
         actions, builder, node_indices, node_names, relation_to_symbols, symbol_to_relations
      );
   }
   if(! goals.static_goals.empty()) {
      encode_goal_satisfaction(
         std::span{goals.static_goals},
         goals.static_goal_levels,
         fact_keys,
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
         fact_keys,
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
         fact_keys,
         builder,
         node_indices,
         node_names,
         relation_to_symbols,
         symbol_to_relations
      );
   }
   if(config_.include_lgan_edges) {
      add_lgan_nn_edges(builder, node_indices, relation_to_symbols, symbol_to_relations);
   }

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

void HGraphEncoderEngine::encode_objects(
   const mimir::search::State& state,
   BatchBuilder& builder,
   hash_map< std::string, hash_map< std::string, int64_t > >& node_indices,
   hash_map< std::string, std::vector< std::string > >& node_names,
   std::span< const std::string > extra_objects
)
{
   const auto& problem = state.get_problem();
   const auto& objects = problem.get_problem_and_domain_objects();

   std::vector< mimir::formalism::Object > ordered(objects.begin(), objects.end());
   std::sort(ordered.begin(), ordered.end(), [](auto lhs, auto rhs) {
      return lhs->get_index() < rhs->get_index();
   });

   for(const auto& obj : ordered) {
      const std::string key = RelationFormatter::format_object(*obj);
      get_or_add_node(config_.symbol_type_id, key, builder, node_indices, node_names);
   }
   for(const auto& key : extra_objects) {
      get_or_add_node(config_.symbol_type_id, key, builder, node_indices, node_names);
   }
   if(config_.add_nullary_predicates) {
      get_or_add_node(
         config_.symbol_type_id, config_.nullary_object_name, builder, node_indices, node_names
      );
   }
}

hash_set< std::string > HGraphEncoderEngine::encode_facts(
   const mimir::search::State& state,
   BatchBuilder& builder,
   hash_map< std::string, hash_map< std::string, int64_t > >& node_indices,
   hash_map< std::string, std::vector< std::string > >& node_names,
   hash_map< std::string, hash_set< std::string > >& relation_to_symbols,
   hash_map< std::string, hash_set< std::string > >& symbol_to_relations,
   std::span< const std::string > extra_objects
)
{
   hash_set< std::string > fact_keys;
   const auto& problem = state.get_problem();
   const auto& repos = problem.get_repositories();

   auto handle_atom = [&](auto atom) {
      using Tag = typename std::remove_pointer_t< decltype(atom) >::Type;
      const auto predicate = atom->get_predicate();
      if(predicate->get_arity() == 0 and not config_.add_nullary_predicates) {
         return;
      }
      const std::string node_type = RelationFormatter::format_predicate(predicate->get_name());
      const std::string node_key = RelationFormatter::format_atom< Tag >(atom);
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

      for(size_t i = 0; i < extra_objects.size(); ++i) {
         const auto& obj_key = extra_objects[i];
         const auto obj_idx = get_or_add_node(
            config_.symbol_type_id, obj_key, builder, node_indices, node_names
         );
         const std::string pos_str = std::to_string(object_keys.size() + i);
         append_edges(builder, config_.symbol_type_id, pos_str, node_type, obj_idx, relation_idx);
         append_edges(builder, node_type, pos_str, config_.symbol_type_id, relation_idx, obj_idx);
      }

      const std::string rel_key = relation_key(node_type, node_key);
      auto& symbols = relation_to_symbols[rel_key];
      for(const auto& obj_key : object_keys) {
         symbols.insert(obj_key);
         symbol_to_relations[obj_key].insert(rel_key);
      }
      for(const auto& obj_key : extra_objects) {
         symbols.insert(obj_key);
         symbol_to_relations[obj_key].insert(rel_key);
      }

      fact_keys.insert(node_key);
   };

   if(config_.include_static) {
      const auto& literals = problem.get_initial_literals< mimir::formalism::StaticTag >();
      for(const auto& literal : literals) {
         if(! literal->get_polarity()) {
            continue;
         }
         handle_atom(literal->get_atom());
      }
   }

   const auto fluent_atoms = repos.get_ground_atoms_from_indices< mimir::formalism::FluentTag >(
      state.get_atoms< mimir::formalism::FluentTag >()
   );
   for(const auto& atom : fluent_atoms) {
      handle_atom(atom);
   }

   const auto derived_atoms = repos.get_ground_atoms_from_indices< mimir::formalism::DerivedTag >(
      state.get_atoms< mimir::formalism::DerivedTag >()
   );
   for(const auto& atom : derived_atoms) {
      handle_atom(atom);
   }

   return fact_keys;
}

template < typename GoalTag >
void HGraphEncoderEngine::encode_literals(
   std::span< const mimir::formalism::GroundLiteral< GoalTag > > goals,
   const hash_map< mimir::formalism::GroundLiteral< GoalTag >, int >& goal_levels,
   BatchBuilder& builder,
   hash_map< std::string, hash_map< std::string, int64_t > >& node_indices,
   hash_map< std::string, std::vector< std::string > >& node_names,
   hash_map< std::string, hash_set< std::string > >& relation_to_symbols,
   hash_map< std::string, hash_set< std::string > >& symbol_to_relations,
   std::span< const std::string > extra_objects
)
{
   for(const auto& literal : goals) {
      const auto atom = literal->get_atom();
      const auto predicate = atom->get_predicate();
      const std::optional< int > goal_level = goal_levels.contains(literal)
                                                 ? std::optional< int >(goal_levels.at(literal))
                                                 : std::nullopt;

      std::string node_type;
      std::string node_key;
      if(goal_level.has_value()) {
         const GoalLevel level(*goal_level);
         node_type = RelationFormatter::format_predicate(
            predicate->get_name(), level, std::nullopt, literal->get_polarity()
         );
         node_key = RelationFormatter::format_literal< GoalTag >(literal, level);
      } else {
         node_type = RelationFormatter::format_predicate(
            predicate->get_name(), std::nullopt, std::nullopt, literal->get_polarity()
         );
         node_key = RelationFormatter::format_literal< GoalTag >(literal, std::nullopt);
      }

      std::vector< std::string > object_keys;
      if(predicate->get_arity() == 0) {
         if(! config_.add_nullary_predicates) {
            continue;
         }
         object_keys.emplace_back(config_.nullary_object_name);
      } else {
         for(const auto& obj : atom->get_objects()) {
            object_keys.emplace_back(RelationFormatter::format_object(*obj));
         }
      }
      const auto relation_idx = get_or_add_node(
         node_type, node_key, builder, node_indices, node_names
      );

      for(size_t pos = 0; pos < object_keys.size(); ++pos) {
         const auto& obj_key = object_keys[pos];
         const auto obj_idx = get_or_add_node(
            config_.symbol_type_id, obj_key, builder, node_indices, node_names
         );
         const std::string pos_str = std::to_string(pos);
         append_edges(builder, config_.symbol_type_id, pos_str, node_type, obj_idx, relation_idx);
         append_edges(builder, node_type, pos_str, config_.symbol_type_id, relation_idx, obj_idx);
      }

      for(size_t i = 0; i < extra_objects.size(); ++i) {
         const auto& obj_key = extra_objects[i];
         const auto obj_idx = get_or_add_node(
            config_.symbol_type_id, obj_key, builder, node_indices, node_names
         );
         const std::string pos_str = std::to_string(object_keys.size() + i);
         append_edges(builder, config_.symbol_type_id, pos_str, node_type, obj_idx, relation_idx);
         append_edges(builder, node_type, pos_str, config_.symbol_type_id, relation_idx, obj_idx);
      }

      const std::string rel_key = relation_key(node_type, node_key);
      auto& symbols = relation_to_symbols[rel_key];
      for(const auto& obj_key : object_keys) {
         symbols.insert(obj_key);
         symbol_to_relations[obj_key].insert(rel_key);
      }
      for(const auto& obj_key : extra_objects) {
         symbols.insert(obj_key);
         symbol_to_relations[obj_key].insert(rel_key);
      }
   }
}

void HGraphEncoderEngine::encode_actions(
   std::span< const mimir::formalism::GroundAction > actions,
   BatchBuilder& builder,
   hash_map< std::string, hash_map< std::string, int64_t > >& node_indices,
   hash_map< std::string, std::vector< std::string > >& node_names,
   hash_map< std::string, hash_set< std::string > >& relation_to_symbols,
   hash_map< std::string, hash_set< std::string > >& symbol_to_relations,
   std::span< const std::string > extra_objects
)
{
   for(const auto& action : actions) {
      const std::string node_type = RelationFormatter::format_action_schema(*action->get_action());
      const std::string node_key = RelationFormatter::format_action(action);
      const auto relation_idx = get_or_add_node(
         node_type, node_key, builder, node_indices, node_names
      );

      const std::string action_symbol = fmt::format("target:{}|{}", action->get_index(), node_key);
      get_or_add_node(config_.symbol_type_id, action_symbol, builder, node_indices, node_names);

      std::vector< std::string > object_keys;
      object_keys.emplace_back(action_symbol);
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

      for(size_t i = 0; i < extra_objects.size(); ++i) {
         const auto& obj_key = extra_objects[i];
         const auto obj_idx = get_or_add_node(
            config_.symbol_type_id, obj_key, builder, node_indices, node_names
         );
         const std::string pos_str = std::to_string(object_keys.size() + i);
         append_edges(builder, config_.symbol_type_id, pos_str, node_type, obj_idx, relation_idx);
         append_edges(builder, node_type, pos_str, config_.symbol_type_id, relation_idx, obj_idx);
      }

      const std::string rel_key = relation_key(node_type, node_key);
      auto& symbols = relation_to_symbols[rel_key];
      for(const auto& obj_key : object_keys) {
         symbols.insert(obj_key);
         symbol_to_relations[obj_key].insert(rel_key);
      }
      for(const auto& obj_key : extra_objects) {
         symbols.insert(obj_key);
         symbol_to_relations[obj_key].insert(rel_key);
      }
   }
}

template < typename GoalTag >
void HGraphEncoderEngine::encode_goal_satisfaction(
   std::span< const mimir::formalism::GroundLiteral< GoalTag > > goals,
   const hash_map< mimir::formalism::GroundLiteral< GoalTag >, int >& goal_levels,
   const hash_set< std::string >& fact_keys,
   BatchBuilder& builder,
   hash_map< std::string, hash_map< std::string, int64_t > >& node_indices,
   hash_map< std::string, std::vector< std::string > >& node_names,
   hash_map< std::string, hash_set< std::string > >& relation_to_symbols,
   hash_map< std::string, hash_set< std::string > >& symbol_to_relations,
   std::string_view suffix,
   std::span< const std::string > extra_objects
)
{
   for(const auto& goal : goals) {
      const auto atom = goal->get_atom();
      const auto predicate = atom->get_predicate();
      const auto key = RelationFormatter::format_atom< GoalTag >(atom);
      const bool satisfied = fact_keys.contains(key) == goal->get_polarity();
      const GoalSatisfaction sat = satisfied ? GoalSatisfaction::satisfied
                                             : GoalSatisfaction::unsatisfied;
      if(! relation_dict_.goal_satisfaction_derivations.contains(sat)) {
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
            predicate->get_name(), level, sat, goal->get_polarity(), suffix
         );
         node_key = RelationFormatter::format_literal< GoalTag >(
            goal, level, sat, std::nullopt, suffix
         );
      } else {
         node_type = RelationFormatter::format_predicate(
            predicate->get_name(), std::nullopt, sat, goal->get_polarity(), suffix
         );
         node_key = RelationFormatter::format_literal< GoalTag >(
            goal, std::nullopt, sat, std::nullopt, suffix
         );
      }

      std::vector< std::string > object_keys;
      if(predicate->get_arity() == 0) {
         if(! config_.add_nullary_predicates) {
            continue;
         }
         object_keys.emplace_back(config_.nullary_object_name);
      } else {
         for(const auto& obj : atom->get_objects()) {
            object_keys.emplace_back(RelationFormatter::format_object(*obj));
         }
      }
      const auto relation_idx = get_or_add_node(
         node_type, node_key, builder, node_indices, node_names
      );

      for(size_t pos = 0; pos < object_keys.size(); ++pos) {
         const auto& obj_key = object_keys[pos];
         const auto obj_idx = get_or_add_node(
            config_.symbol_type_id, obj_key, builder, node_indices, node_names
         );
         const std::string pos_str = std::to_string(pos);
         append_edges(builder, config_.symbol_type_id, pos_str, node_type, obj_idx, relation_idx);
         append_edges(builder, node_type, pos_str, config_.symbol_type_id, relation_idx, obj_idx);
      }

      for(size_t i = 0; i < extra_objects.size(); ++i) {
         const auto& obj_key = extra_objects[i];
         const auto obj_idx = get_or_add_node(
            config_.symbol_type_id, obj_key, builder, node_indices, node_names
         );
         const std::string pos_str = std::to_string(object_keys.size() + i);
         append_edges(builder, config_.symbol_type_id, pos_str, node_type, obj_idx, relation_idx);
         append_edges(builder, node_type, pos_str, config_.symbol_type_id, relation_idx, obj_idx);
      }

      const std::string rel_key = relation_key(node_type, node_key);
      auto& symbols = relation_to_symbols[rel_key];
      for(const auto& obj_key : object_keys) {
         symbols.insert(obj_key);
         symbol_to_relations[obj_key].insert(rel_key);
      }
      for(const auto& obj_key : extra_objects) {
         symbols.insert(obj_key);
         symbol_to_relations[obj_key].insert(rel_key);
      }
   }
}

void HGraphEncoderEngine::add_lgan_nn_edges(
   BatchBuilder& builder,
   const hash_map< std::string, hash_map< std::string, int64_t > >& node_indices,
   const hash_map< std::string, hash_set< std::string > >& relation_to_symbols,
   const hash_map< std::string, hash_set< std::string > >& symbol_to_relations
)
{
   auto symbol_it = node_indices.find(config_.symbol_type_id);
   if(symbol_it == node_indices.end()) {
      return;
   }

   hash_map< std::string, hash_set< std::string > > target_to_tn;
   for(const auto& [target_key, _] : symbol_it->second) {
      hash_set< std::string > tn{target_key};
      auto rels_it = symbol_to_relations.find(target_key);
      if(rels_it != symbol_to_relations.end()) {
         for(const auto& rel_key : rels_it->second) {
            auto sym_it = relation_to_symbols.find(rel_key);
            if(sym_it == relation_to_symbols.end()) {
               continue;
            }
            tn.insert(sym_it->second.begin(), sym_it->second.end());
         }
      }
      target_to_tn.emplace(target_key, std::move(tn));
   }

   for(const auto& [rel_key, arg_set] : relation_to_symbols) {
      if(arg_set.empty()) {
         continue;
      }
      const auto pos = rel_key.find('\n');
      if(pos == std::string::npos) {
         continue;
      }
      const std::string rel_type = rel_key.substr(0, pos);
      const std::string rel_node = rel_key.substr(pos + 1);

      auto rel_type_it = node_indices.find(rel_type);
      if(rel_type_it == node_indices.end()) {
         continue;
      }
      auto rel_idx_it = rel_type_it->second.find(rel_node);
      if(rel_idx_it == rel_type_it->second.end()) {
         continue;
      }
      const int64_t rel_idx = rel_idx_it->second;

      for(const auto& [target_key, tn] : target_to_tn) {
         if(arg_set.contains(target_key)) {
            continue;
         }
         bool is_subset = true;
         for(const auto& sym : arg_set) {
            if(! tn.contains(sym)) {
               is_subset = false;
               break;
            }
         }
         if(! is_subset) {
            continue;
         }
         auto sym_idx_it = symbol_it->second.find(target_key);
         if(sym_idx_it == symbol_it->second.end()) {
            continue;
         }
         const int64_t sym_idx = sym_idx_it->second;

         append_edges(
            builder, rel_type, config_.lgan_nn_edge_pos, config_.symbol_type_id, rel_idx, sym_idx
         );
         append_edges(
            builder, config_.symbol_type_id, config_.lgan_nn_edge_pos, rel_type, sym_idx, rel_idx
         );
      }
   }
}

void HGraphEncoderEngine::ensure_empty_edge_types(BatchBuilder& builder) const
{
   if(! config_.include_empty_edge_types) {
      return;
   }
   for(const auto& [src, rel, dst] : all_edge_types_) {
      builder.ensure_edge_type(src, rel, dst);
   }
}

void HGraphEncoderEngine::ensure_node_feature_dims(BatchBuilder& builder) const
{
   builder.set_node_feature_dim(config_.symbol_type_id, 1);
   for(const auto& [node_type, arity] : relation_dict_.arity) {
      builder.set_node_feature_dim(node_type, arity);
   }
}

std::string
HGraphEncoderEngine::relation_key(const std::string& node_type, const std::string& node_key)
{
   return node_type + "\n" + node_key;
}

int64_t HGraphEncoderEngine::get_or_add_node(
   const std::string& node_type,
   const std::string& node_key,
   BatchBuilder& builder,
   hash_map< std::string, hash_map< std::string, int64_t > >& node_indices,
   hash_map< std::string, std::vector< std::string > >& node_names
)
{
   auto& indices = node_indices[node_type];
   auto it = indices.find(node_key);
   if(it != indices.end()) {
      return it->second;
   }
   const auto idx = static_cast< int64_t >(node_names[node_type].size());
   indices[node_key] = idx;
   node_names[node_type].emplace_back(node_key);
   builder.add_nodes(node_type, idx + 1);
   return idx;
}

template void HGraphEncoderEngine::encode_step_impl< mimir::formalism::StaticTag >(
   const mimir::search::State& state,
   std::span< const mimir::formalism::GroundLiteral< mimir::formalism::StaticTag > > goals,
   std::span< const mimir::formalism::GroundAction > actions,
   BatchBuilder& builder
);
template void HGraphEncoderEngine::encode_step_impl< mimir::formalism::FluentTag >(
   const mimir::search::State& state,
   std::span< const mimir::formalism::GroundLiteral< mimir::formalism::FluentTag > > goals,
   std::span< const mimir::formalism::GroundAction > actions,
   BatchBuilder& builder
);
template void HGraphEncoderEngine::encode_step_impl< mimir::formalism::DerivedTag >(
   const mimir::search::State& state,
   std::span< const mimir::formalism::GroundLiteral< mimir::formalism::DerivedTag > > goals,
   std::span< const mimir::formalism::GroundAction > actions,
   BatchBuilder& builder
);

template void HGraphEncoderEngine::encode_literals< mimir::formalism::StaticTag >(
   std::span< const mimir::formalism::GroundLiteral< mimir::formalism::StaticTag > > goals,
   const hash_map< mimir::formalism::GroundLiteral< mimir::formalism::StaticTag >, int >&
      goal_levels,
   BatchBuilder& builder,
   hash_map< std::string, hash_map< std::string, int64_t > >& node_indices,
   hash_map< std::string, std::vector< std::string > >& node_names,
   hash_map< std::string, hash_set< std::string > >& relation_to_symbols,
   hash_map< std::string, hash_set< std::string > >& symbol_to_relations,
   std::span< const std::string > extra_objects
);

template void HGraphEncoderEngine::encode_literals< mimir::formalism::FluentTag >(
   std::span< const mimir::formalism::GroundLiteral< mimir::formalism::FluentTag > > goals,
   const hash_map< mimir::formalism::GroundLiteral< mimir::formalism::FluentTag >, int >&
      goal_levels,
   BatchBuilder& builder,
   hash_map< std::string, hash_map< std::string, int64_t > >& node_indices,
   hash_map< std::string, std::vector< std::string > >& node_names,
   hash_map< std::string, hash_set< std::string > >& relation_to_symbols,
   hash_map< std::string, hash_set< std::string > >& symbol_to_relations,
   std::span< const std::string > extra_objects
);

template void HGraphEncoderEngine::encode_literals< mimir::formalism::DerivedTag >(
   std::span< const mimir::formalism::GroundLiteral< mimir::formalism::DerivedTag > > goals,
   const hash_map< mimir::formalism::GroundLiteral< mimir::formalism::DerivedTag >, int >&
      goal_levels,
   BatchBuilder& builder,
   hash_map< std::string, hash_map< std::string, int64_t > >& node_indices,
   hash_map< std::string, std::vector< std::string > >& node_names,
   hash_map< std::string, hash_set< std::string > >& relation_to_symbols,
   hash_map< std::string, hash_set< std::string > >& symbol_to_relations,
   std::span< const std::string > extra_objects
);

template void HGraphEncoderEngine::encode_goal_satisfaction< mimir::formalism::StaticTag >(
   std::span< const mimir::formalism::GroundLiteral< mimir::formalism::StaticTag > > goals,
   const hash_map< mimir::formalism::GroundLiteral< mimir::formalism::StaticTag >, int >&
      goal_levels,
   const hash_set< std::string >& fact_keys,
   BatchBuilder& builder,
   hash_map< std::string, hash_map< std::string, int64_t > >& node_indices,
   hash_map< std::string, std::vector< std::string > >& node_names,
   hash_map< std::string, hash_set< std::string > >& relation_to_symbols,
   hash_map< std::string, hash_set< std::string > >& symbol_to_relations,
   std::string_view suffix,
   std::span< const std::string > extra_objects
);

template void HGraphEncoderEngine::encode_goal_satisfaction< mimir::formalism::FluentTag >(
   std::span< const mimir::formalism::GroundLiteral< mimir::formalism::FluentTag > > goals,
   const hash_map< mimir::formalism::GroundLiteral< mimir::formalism::FluentTag >, int >&
      goal_levels,
   const hash_set< std::string >& fact_keys,
   BatchBuilder& builder,
   hash_map< std::string, hash_map< std::string, int64_t > >& node_indices,
   hash_map< std::string, std::vector< std::string > >& node_names,
   hash_map< std::string, hash_set< std::string > >& relation_to_symbols,
   hash_map< std::string, hash_set< std::string > >& symbol_to_relations,
   std::string_view suffix,
   std::span< const std::string > extra_objects
);

template void HGraphEncoderEngine::encode_goal_satisfaction< mimir::formalism::DerivedTag >(
   std::span< const mimir::formalism::GroundLiteral< mimir::formalism::DerivedTag > > goals,
   const hash_map< mimir::formalism::GroundLiteral< mimir::formalism::DerivedTag >, int >&
      goal_levels,
   const hash_set< std::string >& fact_keys,
   BatchBuilder& builder,
   hash_map< std::string, hash_map< std::string, int64_t > >& node_indices,
   hash_map< std::string, std::vector< std::string > >& node_names,
   hash_map< std::string, hash_set< std::string > >& relation_to_symbols,
   hash_map< std::string, hash_set< std::string > >& symbol_to_relations,
   std::string_view suffix,
   std::span< const std::string > extra_objects
);
}  // namespace mifrost
