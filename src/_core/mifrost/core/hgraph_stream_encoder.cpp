#include "hgraph_stream_encoder.hpp"

#include <fmt/format.h>

#include <algorithm>
#include <cstdlib>
#include <mimir/formalism/action.hpp>
#include <mimir/formalism/problem.hpp>
#include <numeric>
#include <optional>
#include <span>
#include <string>
#include <type_traits>
#include <variant>

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
   builder.add_edge(src_type, rel_type, dst_type, src, dst);
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
   if(not config_.ignore_actions) {
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

HGraphEncoderEngine::HeteroEncodingWorkspace& HGraphEncoderEngine::init_hetero_workspace(
   BatchBuilder& builder
)
{
   ensure_node_feature_dims(builder);
   workspace_.node_indices.clear();
   workspace_.node_indices_i64.clear();
   workspace_.symbol_indices.clear();
   workspace_.special_symbol_ids.clear();
   workspace_.next_special_symbol_id = -1;
   workspace_.node_names.clear();
   workspace_.relation_to_symbols.clear();
   workspace_.symbol_to_relations.clear();

   const size_t type_hint = relation_dict_.arity.size() + 4;
   workspace_.node_indices.reserve(type_hint);
   workspace_.node_indices_i64.reserve(type_hint);
   if(config_.export_node_names) {
      workspace_.node_names.reserve(type_hint);
   }
   workspace_.symbol_indices.reserve(relation_dict_.arity.size() + 8);

   return workspace_;
}

void HGraphEncoderEngine::track_relation_symbols_if_enabled(
   const std::string& rel_key,
   std::span< const int64_t > object_symbol_ids,
   std::span< const int64_t > extra_symbol_ids,
   hash_map< std::string, hash_set< int64_t > >& relation_to_symbols,
   hash_map< int64_t, hash_set< std::string > >& symbol_to_relations
)
{
   if(not config_.include_lgan_edges) {
      return;
   }
   auto& symbols = relation_to_symbols[rel_key];
   symbols.reserve(symbols.size() + object_symbol_ids.size() + extra_symbol_ids.size());
   for(const auto symbol_id : object_symbol_ids) {
      symbols.insert(symbol_id);
      symbol_to_relations[symbol_id].insert(rel_key);
   }
   for(const auto symbol_id : extra_symbol_ids) {
      symbols.insert(symbol_id);
      symbol_to_relations[symbol_id].insert(rel_key);
   }
}

void HGraphEncoderEngine::track_relation_symbols_if_enabled(
   const std::string& rel_key,
   std::span< const std::string > object_keys,
   std::span< const std::string > extra_objects,
   hash_map< std::string, hash_set< int64_t > >& relation_to_symbols,
   hash_map< int64_t, hash_set< std::string > >& symbol_to_relations
)
{
   if(not config_.include_lgan_edges) {
      return;
   }
   std::vector< int64_t > object_symbol_ids;
   object_symbol_ids.reserve(object_keys.size());
   for(const auto& key : object_keys) {
      object_symbol_ids.emplace_back(get_or_assign_special_symbol_id(key));
   }
   std::vector< int64_t > extra_symbol_ids;
   extra_symbol_ids.reserve(extra_objects.size());
   for(const auto& key : extra_objects) {
      extra_symbol_ids.emplace_back(get_or_assign_special_symbol_id(key));
   }
   track_relation_symbols_if_enabled(
      rel_key,
      std::span{object_symbol_ids},
      std::span{extra_symbol_ids},
      relation_to_symbols,
      symbol_to_relations
   );
}

void HGraphEncoderEngine::track_relation_symbols_if_enabled(
   const std::string& rel_key,
   std::span< const std::string > object_keys,
   std::span< const std::string > extra_objects,
   hash_map< std::string, hash_set< std::string > >& relation_to_symbols,
   hash_map< std::string, hash_set< std::string > >& symbol_to_relations
) const
{
   if(not config_.include_lgan_edges) {
      return;
   }
   auto& symbols = relation_to_symbols[rel_key];
   symbols.reserve(symbols.size() + object_keys.size() + extra_objects.size());
   for(const auto& obj_key : object_keys) {
      symbols.insert(obj_key);
      symbol_to_relations[obj_key].insert(rel_key);
   }
   for(const auto& obj_key : extra_objects) {
      symbols.insert(obj_key);
      symbol_to_relations[obj_key].insert(rel_key);
   }
}

void HGraphEncoderEngine::encode_goal_inputs(
   const GoalInputs& goals,
   BatchBuilder& builder,
   HeteroEncodingWorkspace& workspace,
   std::span< const std::string > extra_objects
)
{
   encode_literals(
      std::span{goals.static_goals},
      goals.static_goal_levels,
      builder,
      workspace.node_indices,
      workspace.node_names,
      workspace.relation_to_symbols,
      workspace.symbol_to_relations,
      extra_objects
   );
   encode_literals(
      std::span{goals.fluent_goals},
      goals.fluent_goal_levels,
      builder,
      workspace.node_indices,
      workspace.node_names,
      workspace.relation_to_symbols,
      workspace.symbol_to_relations,
      extra_objects
   );
   encode_literals(
      std::span{goals.derived_goals},
      goals.derived_goal_levels,
      builder,
      workspace.node_indices,
      workspace.node_names,
      workspace.relation_to_symbols,
      workspace.symbol_to_relations,
      extra_objects
   );
}

void HGraphEncoderEngine::encode_goal_satisfaction_inputs(
   const GoalInputs& goals,
   const hash_set< std::string >& fact_keys,
   BatchBuilder& builder,
   HeteroEncodingWorkspace& workspace,
   std::string_view suffix,
   std::span< const std::string > extra_objects
)
{
   if(not goals.static_goals.empty()) {
      encode_goal_satisfaction(
         std::span{goals.static_goals},
         goals.static_goal_levels,
         fact_keys,
         builder,
         workspace.node_indices,
         workspace.node_names,
         workspace.relation_to_symbols,
         workspace.symbol_to_relations,
         suffix,
         extra_objects
      );
   }
   if(not goals.fluent_goals.empty()) {
      encode_goal_satisfaction(
         std::span{goals.fluent_goals},
         goals.fluent_goal_levels,
         fact_keys,
         builder,
         workspace.node_indices,
         workspace.node_names,
         workspace.relation_to_symbols,
         workspace.symbol_to_relations,
         suffix,
         extra_objects
      );
   }
   if(not goals.derived_goals.empty()) {
      encode_goal_satisfaction(
         std::span{goals.derived_goals},
         goals.derived_goal_levels,
         fact_keys,
         builder,
         workspace.node_indices,
         workspace.node_names,
         workspace.relation_to_symbols,
         workspace.symbol_to_relations,
         suffix,
         extra_objects
      );
   }
}

void HGraphEncoderEngine::maybe_add_lgan_edges(
   BatchBuilder& builder,
   const HeteroEncodingWorkspace& workspace
)
{
   if(config_.include_lgan_edges) {
      add_lgan_nn_edges(
         builder,
         workspace.node_indices,
         workspace.node_indices_i64,
         workspace.symbol_indices,
         workspace.relation_to_symbols,
         workspace.symbol_to_relations
      );
   }
}

void HGraphEncoderEngine::finalize_hetero_encoding(
   BatchBuilder& builder,
   const HeteroEncodingWorkspace& workspace,
   const std::vector< std::string >* object_names_override
) const
{
   if(not config_.export_node_names) {
      ensure_empty_edge_types(builder);
      return;
   }

   for(const auto& [node_type, _] : relation_dict_.arity) {
      if(not workspace.node_names.contains(node_type)) {
         builder.set_node_names(node_type, {});
      }
   }

   if(not workspace.node_names.contains(config_.symbol_type_id)) {
      builder.set_node_names(config_.symbol_type_id, {});
      builder.set_object_names({});
   } else {
      const auto& symbol_names = workspace.node_names.at(config_.symbol_type_id);
      builder.set_node_names(config_.symbol_type_id, symbol_names);
      if(object_names_override != nullptr) {
         builder.set_object_names(*object_names_override);
      } else {
         builder.set_object_names(symbol_names);
      }
   }

   for(const auto& [node_type, names] : workspace.node_names) {
      if(node_type == config_.symbol_type_id) {
         continue;
      }
      builder.set_node_names(node_type, names);
   }

   ensure_empty_edge_types(builder);
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
   encode_impl_core(state, goals, actions, {}, std::nullopt, builder);
}

void HGraphEncoderEngine::encode_impl_core(
   const mimir::search::State& state,
   const GoalInputs& goals,
   std::span< const mimir::formalism::GroundAction > actions,
   std::span< const HistorySubgoal > history_subgoals,
   std::optional< int > history_max_steps,
   BatchBuilder& builder
)
{
   auto& workspace = init_hetero_workspace(builder);

   encode_objects(state, builder, workspace.node_indices, workspace.node_names);
   const auto fact_keys = encode_facts(
      state,
      builder,
      workspace.node_indices,
      workspace.node_names,
      workspace.relation_to_symbols,
      workspace.symbol_to_relations
   );
   encode_goal_inputs(goals, builder, workspace);
   if(! history_subgoals.empty()) {
      encode_history(
         history_subgoals,
         history_max_steps,
         builder,
         workspace.node_indices,
         workspace.node_names,
         workspace.relation_to_symbols,
         workspace.symbol_to_relations
      );
   }
   if(not config_.ignore_actions) {
      encode_actions(
         actions,
         builder,
         workspace.node_indices,
         workspace.node_names,
         workspace.relation_to_symbols,
         workspace.symbol_to_relations
      );
   }
   encode_goal_satisfaction_inputs(goals, fact_keys, builder, workspace);
   maybe_add_lgan_edges(builder, workspace);
   finalize_hetero_encoding(builder, workspace);
}

void HGraphEncoderEngine::encode_objects(
   const mimir::search::State& state,
   BatchBuilder& builder,
   hash_map< std::string, hash_map< std::string, int64_t > >& node_indices,
   hash_map< std::string, std::vector< std::string > >& node_names,
   std::span< const std::string > extra_objects
)
{
   (void) node_indices;
   const auto& problem = state.get_problem();
   const auto& objects = problem.get_problem_and_domain_objects();

   std::vector< mimir::formalism::Object > ordered(objects.begin(), objects.end());
   std::sort(ordered.begin(), ordered.end(), [](auto lhs, auto rhs) {
      return lhs->get_index() < rhs->get_index();
   });

   const size_t symbol_hint = ordered.size() + extra_objects.size()
                              + static_cast< size_t >(config_.add_nullary_predicates ? 1 : 0);
   workspace_.symbol_indices.reserve(workspace_.symbol_indices.size() + symbol_hint);
   if(config_.export_node_names) {
      auto& symbol_names = node_names[config_.symbol_type_id];
      symbol_names.reserve(symbol_names.size() + symbol_hint);
   }
   auto& symbol_indices = node_indices[config_.symbol_type_id];

   for(const auto& obj : ordered) {
      const auto idx = get_or_add_symbol_object_node(obj, builder, node_names);
      symbol_indices.try_emplace(symbol_node_key(obj), idx);
   }
   for(const auto& symbol_name : extra_objects) {
      const auto idx = get_or_add_symbol_special_node(
         symbol_name, symbol_name, builder, node_names
      );
      symbol_indices.try_emplace(symbol_name, idx);
   }
   if(config_.add_nullary_predicates) {
      const auto idx = get_or_add_symbol_special_node(
         config_.nullary_object_name, config_.nullary_object_name, builder, node_names
      );
      symbol_indices.try_emplace(config_.nullary_object_name, idx);
   }
}

hash_set< std::string > HGraphEncoderEngine::encode_facts(
   const mimir::search::State& state,
   BatchBuilder& builder,
   hash_map< std::string, hash_map< std::string, int64_t > >& node_indices,
   hash_map< std::string, std::vector< std::string > >& node_names,
   hash_map< std::string, hash_set< int64_t > >& relation_to_symbols,
   hash_map< int64_t, hash_set< std::string > >& symbol_to_relations,
   std::span< const std::string > extra_objects
)
{
   hash_set< std::string > fact_keys;
   const auto& problem = state.get_problem();
   const auto& repos = problem.get_repositories();

   auto handle_atom = [&]< typename Tag >(mimir::formalism::GroundAtom< Tag > atom) {
      const auto predicate = atom->get_predicate();
      if(predicate->get_arity() == 0 and not config_.add_nullary_predicates) {
         return;
      }
      const std::string node_type = RelationFormatter::format_predicate(predicate);
      const int64_t relation_key = static_cast< int64_t >(atom->get_index());
      const std::string atom_text = RelationFormatter::format_atom< Tag >(atom);
      const std::string node_name = config_.export_node_names ? atom_text : "";
      const auto relation_idx = get_or_add_relation_node_i64(
         node_type, relation_key, builder, node_indices, node_names, node_name
      );

      std::vector< int64_t > object_symbol_ids;
      object_symbol_ids.reserve(static_cast< size_t >(predicate->get_arity()));
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

      for(size_t pos = 0; pos < object_symbol_ids.size(); ++pos) {
         const int64_t symbol_id = object_symbol_ids[pos];
         const auto obj_idx = workspace_.symbol_indices.at(symbol_id);
         const std::string pos_str = std::to_string(pos);
         append_edges(builder, config_.symbol_type_id, pos_str, node_type, obj_idx, relation_idx);
         append_edges(builder, node_type, pos_str, config_.symbol_type_id, relation_idx, obj_idx);
      }

      std::vector< int64_t > extra_symbol_ids;
      extra_symbol_ids.reserve(extra_objects.size());
      for(size_t i = 0; i < extra_objects.size(); ++i) {
         const auto& symbol_name = extra_objects[i];
         const auto obj_idx = get_or_add_symbol_special_node(
            symbol_name, symbol_name, builder, node_names
         );
         const int64_t symbol_id = get_or_assign_special_symbol_id(symbol_name);
         extra_symbol_ids.emplace_back(symbol_id);
         const std::string pos_str = std::to_string(object_symbol_ids.size() + i);
         append_edges(builder, config_.symbol_type_id, pos_str, node_type, obj_idx, relation_idx);
         append_edges(builder, node_type, pos_str, config_.symbol_type_id, relation_idx, obj_idx);
      }

      const std::string rel_key = relation_key_i64(node_type, relation_key);
      track_relation_symbols_if_enabled(
         rel_key,
         std::span{object_symbol_ids},
         std::span{extra_symbol_ids},
         relation_to_symbols,
         symbol_to_relations
      );

      fact_keys.insert(atom_text);
   };

   if(config_.include_static) {
      const auto& literals = problem.get_initial_literals< mimir::formalism::StaticTag >();
      for(const auto& literal : literals) {
         if(not literal->get_polarity()) {
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

void HGraphEncoderEngine::encode_actions(
   std::span< const mimir::formalism::GroundAction > actions,
   BatchBuilder& builder,
   hash_map< std::string, hash_map< std::string, int64_t > >& node_indices,
   hash_map< std::string, std::vector< std::string > >& node_names,
   hash_map< std::string, hash_set< int64_t > >& relation_to_symbols,
   hash_map< int64_t, hash_set< std::string > >& symbol_to_relations,
   std::span< const std::string > extra_objects
)
{
   for(const auto& action : actions) {
      const std::string node_type = RelationFormatter::format_action_schema(*action->get_action());
      const int64_t relation_key = static_cast< int64_t >(action->get_index());
      const std::string node_name = config_.export_node_names
                                       ? RelationFormatter::format_action(action)
                                       : "";
      const auto relation_idx = get_or_add_relation_node_i64(
         node_type, relation_key, builder, node_indices, node_names, node_name
      );

      const std::string action_symbol_key = fmt::format("target:{}", action->get_index());
      const std::string action_symbol_name = config_.export_node_names
                                                ? fmt::format(
                                                     "target:{}|{}", action->get_index(), node_name
                                                  )
                                                : action_symbol_key;
      const auto action_symbol_idx = get_or_add_symbol_special_node(
         action_symbol_key, action_symbol_name, builder, node_names
      );
      const auto action_symbol_id = get_or_assign_special_symbol_id(action_symbol_key);
      (void) action_symbol_idx;

      std::vector< int64_t > object_symbol_ids;
      object_symbol_ids.reserve(action->get_objects().size() + 1);
      object_symbol_ids.emplace_back(action_symbol_id);
      for(const auto& obj : action->get_objects()) {
         const auto obj_idx = get_or_add_symbol_object_node(obj, builder, node_names);
         (void) obj_idx;
         object_symbol_ids.emplace_back(static_cast< int64_t >(obj->get_index()));
      }

      for(size_t pos = 0; pos < object_symbol_ids.size(); ++pos) {
         const int64_t symbol_id = object_symbol_ids[pos];
         const auto obj_idx = workspace_.symbol_indices.at(symbol_id);
         const std::string pos_str = std::to_string(pos);
         append_edges(builder, config_.symbol_type_id, pos_str, node_type, obj_idx, relation_idx);
         append_edges(builder, node_type, pos_str, config_.symbol_type_id, relation_idx, obj_idx);
      }

      std::vector< int64_t > extra_symbol_ids;
      extra_symbol_ids.reserve(extra_objects.size());
      for(size_t i = 0; i < extra_objects.size(); ++i) {
         const auto& symbol_name = extra_objects[i];
         const auto obj_idx = get_or_add_symbol_special_node(
            symbol_name, symbol_name, builder, node_names
         );
         const int64_t symbol_id = get_or_assign_special_symbol_id(symbol_name);
         extra_symbol_ids.emplace_back(symbol_id);
         const std::string pos_str = std::to_string(object_symbol_ids.size() + i);
         append_edges(builder, config_.symbol_type_id, pos_str, node_type, obj_idx, relation_idx);
         append_edges(builder, node_type, pos_str, config_.symbol_type_id, relation_idx, obj_idx);
      }

      const std::string rel_key = relation_key_i64(node_type, relation_key);
      track_relation_symbols_if_enabled(
         rel_key,
         std::span{object_symbol_ids},
         std::span{extra_symbol_ids},
         relation_to_symbols,
         symbol_to_relations
      );
   }
}

void HGraphEncoderEngine::encode_history(
   std::span< const HistorySubgoal > history_subgoals,
   std::optional< int > history_max_steps,
   BatchBuilder& builder,
   hash_map< std::string, hash_map< std::string, int64_t > >& node_indices,
   hash_map< std::string, std::vector< std::string > >& node_names,
   hash_map< std::string, hash_set< int64_t > >& relation_to_symbols,
   hash_map< int64_t, hash_set< std::string > >& symbol_to_relations
)
{
   builder.set_node_feature_dim("history", 1);

   struct HistoryEntry {
      int dt = 0;
      std::vector< LiteralVariant > literals;
   };

   std::vector< HistoryEntry > entries;
   entries.reserve(history_subgoals.size());
   for(const auto& entry : history_subgoals) {
      const int dt = entry.first;
      if(dt >= 0) {
         throw std::invalid_argument("history_subgoals expects negative dt values");
      }
      if(history_max_steps.has_value()) {
         if(std::abs(dt) > *history_max_steps) {
            continue;
         }
      }
      entries.push_back({dt, entry.second});
   }

   std::ranges::stable_sort(entries, [](const auto& lhs, const auto& rhs) {
      return lhs.dt < rhs.dt;
   });

   std::vector< float > history_dt;
   history_dt.reserve(entries.size());
   bool wrote_history = false;

   for(size_t entry_idx = 0; entry_idx < entries.size(); ++entry_idx) {
      const auto& entry = entries[entry_idx];
      const std::string history_key = fmt::format("history:{}#{}", entry.dt, entry_idx);
      const auto history_idx = get_or_add_node(
         "history", history_key, builder, node_indices, node_names, config_.export_node_names
      );
      history_dt.push_back(static_cast< float >(entry.dt));

      for(const auto& literal_variant : entry.literals) {
         std::visit(
            [&]< typename Tag >(const mimir::formalism::GroundLiteral< Tag >& literal) {
               const auto atom = literal->get_atom();
               const auto predicate = atom->get_predicate();
               if(predicate->get_arity() == 0 and not config_.add_nullary_predicates) {
                  return;
               }

               const std::string node_type = RelationFormatter::format_predicate(
                  predicate, std::nullopt, std::nullopt, literal->get_polarity()
               );
               const std::string node_key = RelationFormatter::format_literal< Tag >(
                  literal, std::nullopt
               );

               auto& indices = node_indices[node_type];
               const bool is_new = indices.find(node_key) == indices.end();
               const auto relation_idx = get_or_add_node(
                  node_type, node_key, builder, node_indices, node_names, config_.export_node_names
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

               if(is_new) {
                  for(size_t pos = 0; pos < object_symbol_ids.size(); ++pos) {
                     const auto obj_idx = workspace_.symbol_indices.at(object_symbol_ids[pos]);
                     const std::string pos_str = std::to_string(pos);
                     append_edges(
                        builder, config_.symbol_type_id, pos_str, node_type, obj_idx, relation_idx
                     );
                     append_edges(
                        builder, node_type, pos_str, config_.symbol_type_id, relation_idx, obj_idx
                     );
                  }
               }

               const std::string rel_key = relation_key(node_type, node_key);
               track_relation_symbols_if_enabled(
                  rel_key,
                  std::span{object_symbol_ids},
                  std::span< const int64_t >{},
                  relation_to_symbols,
                  symbol_to_relations
               );

               append_edges(
                  builder,
                  node_type,
                  config_.history_link_relation,
                  "history",
                  relation_idx,
                  history_idx
               );
               append_edges(
                  builder,
                  "history",
                  config_.history_link_relation,
                  node_type,
                  history_idx,
                  relation_idx
               );
               wrote_history = true;
            },
            literal_variant
         );
      }
   }

   if(not history_dt.empty()) {
      builder.add_node_features("history", "history_dt", history_dt, 1);
   }
   if(wrote_history) {
      builder.set_schema_flag("history", true);
   }
}

void HGraphEncoderEngine::add_lgan_nn_edges(
   BatchBuilder& builder,
   const hash_map< std::string, hash_map< std::string, int64_t > >& node_indices,
   const hash_map< std::string, hash_map< int64_t, int64_t > >& node_indices_i64,
   const hash_map< int64_t, int64_t >& symbol_indices,
   const hash_map< std::string, hash_set< int64_t > >& relation_to_symbols,
   const hash_map< int64_t, hash_set< std::string > >& symbol_to_relations
)
{
   if(symbol_indices.empty()) {
      return;
   }

   hash_map< int64_t, hash_set< int64_t > > target_to_tn;
   for(const auto& [target_id, _] : symbol_indices) {
      hash_set< int64_t > tn{target_id};
      auto rels_it = symbol_to_relations.find(target_id);
      if(rels_it != symbol_to_relations.end()) {
         for(const auto& rel_key : rels_it->second) {
            auto sym_it = relation_to_symbols.find(rel_key);
            if(sym_it == relation_to_symbols.end()) {
               continue;
            }
            tn.insert(sym_it->second.begin(), sym_it->second.end());
         }
      }
      target_to_tn.emplace(target_id, std::move(tn));
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
      const std::string rel_node_token = rel_key.substr(pos + 1);

      std::optional< int64_t > rel_idx = std::nullopt;
      if(rel_node_token.rfind("i64:", 0) == 0) {
         const std::string numeric = rel_node_token.substr(4);
         try {
            const int64_t relation_key = std::stoll(numeric);
            auto rel_type_it = node_indices_i64.find(rel_type);
            if(rel_type_it != node_indices_i64.end()) {
               auto rel_idx_it = rel_type_it->second.find(relation_key);
               if(rel_idx_it != rel_type_it->second.end()) {
                  rel_idx = rel_idx_it->second;
               }
            }
         } catch(const std::exception&) {
         }
      } else {
         auto rel_type_it = node_indices.find(rel_type);
         if(rel_type_it != node_indices.end()) {
            auto rel_idx_it = rel_type_it->second.find(rel_node_token);
            if(rel_idx_it != rel_type_it->second.end()) {
               rel_idx = rel_idx_it->second;
            }
         }
      }
      if(not rel_idx.has_value()) {
         continue;
      }

      for(const auto& [target_id, tn] : target_to_tn) {
         if(arg_set.contains(target_id)) {
            continue;
         }
         bool is_subset = true;
         for(const auto& sym : arg_set) {
            if(not tn.contains(sym)) {
               is_subset = false;
               break;
            }
         }
         if(not is_subset) {
            continue;
         }
         auto sym_idx_it = symbol_indices.find(target_id);
         if(sym_idx_it == symbol_indices.end()) {
            continue;
         }
         const int64_t sym_idx = sym_idx_it->second;

         append_edges(
            builder, rel_type, config_.lgan_nn_edge_pos, config_.symbol_type_id, *rel_idx, sym_idx
         );
         append_edges(
            builder, config_.symbol_type_id, config_.lgan_nn_edge_pos, rel_type, sym_idx, *rel_idx
         );
      }
   }
}

void HGraphEncoderEngine::ensure_empty_edge_types(BatchBuilder& builder) const
{
   if(not config_.include_empty_edge_types) {
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

std::string HGraphEncoderEngine::symbol_node_key(const mimir::formalism::Object& obj) const
{
   return RelationFormatter::format_object(*obj);
}

std::string HGraphEncoderEngine::relation_key_i64(const std::string& node_type, int64_t node_key)
{
   return relation_key(node_type, fmt::format("i64:{}", node_key));
}

int64_t HGraphEncoderEngine::get_or_assign_special_symbol_id(std::string_view symbol_name)
{
   auto it = workspace_.special_symbol_ids.find(std::string(symbol_name));
   if(it != workspace_.special_symbol_ids.end()) {
      return it->second;
   }
   const int64_t symbol_id = workspace_.next_special_symbol_id--;
   workspace_.special_symbol_ids.emplace(std::string(symbol_name), symbol_id);
   return symbol_id;
}

int64_t HGraphEncoderEngine::get_or_add_symbol_object_node(
   const mimir::formalism::Object& obj,
   BatchBuilder& builder,
   hash_map< std::string, std::vector< std::string > >& node_names
)
{
   const int64_t symbol_id = static_cast< int64_t >(obj->get_index());
   auto it = workspace_.symbol_indices.find(symbol_id);
   if(it != workspace_.symbol_indices.end()) {
      return it->second;
   }
   const int64_t idx = static_cast< int64_t >(workspace_.symbol_indices.size());
   workspace_.symbol_indices.emplace(symbol_id, idx);
   builder.add_nodes(config_.symbol_type_id, idx + 1);
   if(config_.export_node_names) {
      node_names[config_.symbol_type_id].emplace_back(RelationFormatter::format_object(*obj));
   }
   return idx;
}

int64_t HGraphEncoderEngine::get_or_add_symbol_special_node(
   std::string_view symbol_key,
   std::string_view symbol_name,
   BatchBuilder& builder,
   hash_map< std::string, std::vector< std::string > >& node_names
)
{
   const int64_t symbol_id = get_or_assign_special_symbol_id(symbol_key);
   auto it = workspace_.symbol_indices.find(symbol_id);
   if(it != workspace_.symbol_indices.end()) {
      return it->second;
   }
   const int64_t idx = static_cast< int64_t >(workspace_.symbol_indices.size());
   workspace_.symbol_indices.emplace(symbol_id, idx);
   builder.add_nodes(config_.symbol_type_id, idx + 1);
   if(config_.export_node_names) {
      node_names[config_.symbol_type_id].emplace_back(symbol_name);
   }
   return idx;
}

int64_t HGraphEncoderEngine::get_or_add_relation_node_i64(
   const std::string& node_type,
   int64_t key,
   BatchBuilder& builder,
   hash_map< std::string, hash_map< std::string, int64_t > >& node_indices,
   hash_map< std::string, std::vector< std::string > >& node_names,
   std::string_view node_name
)
{
   auto& indices = workspace_.node_indices_i64[node_type];
   auto it = indices.find(key);
   if(it != indices.end()) {
      return it->second;
   }
   const auto idx = static_cast< int64_t >(indices.size());
   indices[key] = idx;
   if(config_.include_lgan_edges) {
      node_indices[node_type][fmt::format("i64:{}", key)] = idx;
   }
   if(config_.export_node_names) {
      node_names[node_type].emplace_back(node_name);
   }
   builder.add_nodes(node_type, idx + 1);
   return idx;
}

int64_t HGraphEncoderEngine::get_or_add_node(
   const std::string& node_type,
   const std::string& node_key,
   BatchBuilder& builder,
   hash_map< std::string, hash_map< std::string, int64_t > >& node_indices,
   hash_map< std::string, std::vector< std::string > >& node_names,
   bool store_node_name
)
{
   auto& indices = node_indices[node_type];
   auto it = indices.find(node_key);
   if(it != indices.end()) {
      return it->second;
   }
   const auto idx = static_cast< int64_t >(indices.size());
   indices[node_key] = idx;
   if(store_node_name) {
      node_names[node_type].emplace_back(node_key);
   }
   builder.add_nodes(node_type, idx + 1);
   return idx;
}
}  // namespace mifrost
