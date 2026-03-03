#include "hgraph_stream_encoder.hpp"

#include <fmt/format.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <mimir/formalism/action.hpp>
#include <mimir/formalism/problem.hpp>
#include <numeric>
#include <optional>
#include <span>
#include <string>
#include <type_traits>
#include <variant>

#include "mifrost/core/schema_key_separators.hpp"
#include "mifrost/input_handling/batch_input_parser.hpp"

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
   relation_dict_ = RelationDict(domain_, actions, rel_config);
   rebuild_all_edge_types();
}

void HGraphEncoderEngine::rebuild_all_edge_types()
{
   all_edge_types_.clear();
   for(const auto& [node_type, arity] : relation_dict_.arity) {
      const int effective_arity = (config_.add_nullary_predicates and arity == 0) ? 1 : arity;
      for(int pos = 0; pos < effective_arity; ++pos) {
         const std::string pos_str = std::to_string(pos);
         all_edge_types_.emplace_back(config_.symbol_type_id, pos_str, node_type);
         all_edge_types_.emplace_back(node_type, pos_str, config_.symbol_type_id);
      }
      if(config_.include_lgan_edges) {
         all_edge_types_.emplace_back(node_type, config_.lgan_tn_edge_pos, config_.symbol_type_id);
         all_edge_types_.emplace_back(node_type, config_.lgan_nn_edge_pos, config_.symbol_type_id);
      }
   }
   std::ranges::sort(all_edge_types_);
   all_edge_types_.erase(std::ranges::unique(all_edge_types_).begin(), all_edge_types_.end());
}

void HGraphEncoderEngine::update_relations(RelationDict relation_dict)
{
   relation_dict_ = std::move(relation_dict);
   rebuild_all_edge_types();
}

HGraphEncoderEngine::HeteroEncodingWorkspace& HGraphEncoderEngine::init_hetero_workspace(
   BatchBuilder& builder
)
{
   ensure_node_feature_dims(builder);
   workspace_.node_indices.clear();
   workspace_.node_indices_i64.clear();
   workspace_.node_indices_u64.clear();
   workspace_.symbol_indices.clear();
   workspace_.symbol_key_to_id.clear();
   workspace_.special_symbol_ids.clear();
   workspace_.next_special_symbol_id = -1;
   workspace_.node_names.clear();
   workspace_.relation_to_symbols.clear();
   workspace_.symbol_to_relations.clear();
   workspace_.relation_type_ids.clear();
   workspace_.relation_type_names.clear();
   workspace_.lgan_target_symbol_ids.clear();
   workspace_.action_targets.clear();

   const size_t type_hint = relation_dict_.arity.size() + 4;
   workspace_.node_indices.reserve(type_hint);
   workspace_.node_indices_i64.reserve(type_hint);
   workspace_.node_indices_u64.reserve(type_hint);
   if(config_.export_node_names) {
      workspace_.node_names.reserve(type_hint);
   }
   workspace_.relation_type_ids.reserve(type_hint);
   workspace_.relation_type_names.reserve(type_hint);
   workspace_.symbol_indices.reserve(relation_dict_.arity.size() + 8);
   workspace_.symbol_key_to_id.reserve(relation_dict_.arity.size() + 8);
   workspace_.lgan_target_symbol_ids.reserve(8);

   return workspace_;
}

void HGraphEncoderEngine::track_relation_symbols_if_enabled(
   RelationRef rel_ref,
   std::span< const int64_t > object_symbol_ids,
   std::span< const int64_t > extra_symbol_ids,
   hash_map< RelationRef, hash_set< int64_t > >& relation_to_symbols,
   hash_map< int64_t, hash_set< RelationRef > >& symbol_to_relations
)
{
   if(not config_.include_lgan_edges) {
      return;
   }
   auto& symbols = relation_to_symbols[rel_ref];
   symbols.reserve(symbols.size() + object_symbol_ids.size() + extra_symbol_ids.size());
   for(const auto symbol_id : object_symbol_ids) {
      symbols.insert(symbol_id);
      symbol_to_relations[symbol_id].insert(rel_ref);
   }
   for(const auto symbol_id : extra_symbol_ids) {
      symbols.insert(symbol_id);
      symbol_to_relations[symbol_id].insert(rel_ref);
   }
}

void HGraphEncoderEngine::track_relation_symbols_if_enabled(
   RelationRef rel_ref,
   std::span< const std::string > object_keys,
   std::span< const std::string > extra_objects,
   hash_map< RelationRef, hash_set< int64_t > >& relation_to_symbols,
   hash_map< int64_t, hash_set< RelationRef > >& symbol_to_relations
)
{
   if(not config_.include_lgan_edges) {
      return;
   }
   std::vector< int64_t > object_symbol_ids;
   object_symbol_ids.reserve(object_keys.size());
   for(const auto& key : object_keys) {
      auto it = workspace_.symbol_key_to_id.find(key);
      if(it != workspace_.symbol_key_to_id.end()) {
         object_symbol_ids.emplace_back(it->second);
      }
   }
   std::vector< int64_t > extra_symbol_ids;
   extra_symbol_ids.reserve(extra_objects.size());
   for(const auto& key : extra_objects) {
      auto it = workspace_.symbol_key_to_id.find(key);
      if(it != workspace_.symbol_key_to_id.end()) {
         extra_symbol_ids.emplace_back(it->second);
      }
   }
   track_relation_symbols_if_enabled(
      rel_ref,
      std::span{object_symbol_ids},
      std::span{extra_symbol_ids},
      relation_to_symbols,
      symbol_to_relations
   );
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
   const hash_set< uint64_t >& fact_keys,
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
      add_lgan_edges(
         builder,
         workspace.lgan_target_symbol_ids,
         workspace.symbol_indices,
         workspace.relation_to_symbols,
         workspace.symbol_to_relations,
         workspace.relation_type_names
      );
   }
}

void HGraphEncoderEngine::finalize_hetero_encoding(
   BatchBuilder& builder,
   const HeteroEncodingWorkspace& workspace,
   const std::vector< std::string >* object_names_override
) const
{
   if(config_.export_node_names) {
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
   }

   if(config_.export_action_targets) {
      const TargetMetadataEmitConfig emit_config{
         .symbol_type_id = config_.symbol_type_id,
         .symbol_prefix = std::string(kDefaultTargetSymbolPrefix),
         .include_depth = false,
         .parent_relation = std::nullopt,
      };
      emit_target_metadata(builder, workspace.action_targets, emit_config);
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
   if(not history_subgoals.empty()) {
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

hash_set< uint64_t > HGraphEncoderEngine::encode_facts(
   const mimir::search::State& state,
   BatchBuilder& builder,
   hash_map< std::string, hash_map< std::string, int64_t > >& node_indices,
   hash_map< std::string, std::vector< std::string > >& node_names,
   hash_map< RelationRef, hash_set< int64_t > >& relation_to_symbols,
   hash_map< int64_t, hash_set< RelationRef > >& symbol_to_relations,
   std::span< const std::string > extra_objects
)
{
   hash_set< uint64_t > fact_keys;
   const auto& problem = state.get_problem();
   const auto& repos = problem.get_repositories();

   auto handle_atom = [&]< typename Tag >(mimir::formalism::GroundAtom< Tag > atom) {
      const auto predicate = atom->get_predicate();
      if(predicate->get_arity() == 0 and not config_.add_nullary_predicates) {
         return;
      }
      const std::string node_type = RelationFormatter::format_predicate(predicate);
      const int64_t relation_key = static_cast< int64_t >(atom->get_index());
      const std::string node_name = config_.export_node_names
                                       ? RelationFormatter::format_atom< Tag >(atom)
                                       : "";
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

      const auto rel_ref = relation_ref_for(node_type, relation_idx);
      track_relation_symbols_if_enabled(
         rel_ref,
         std::span{object_symbol_ids},
         std::span{extra_symbol_ids},
         relation_to_symbols,
         symbol_to_relations
      );

      uint32_t tag_id = 0;
      if constexpr(std::is_same_v< Tag, mimir::formalism::StaticTag >) {
         tag_id = 1;
      } else if constexpr(std::is_same_v< Tag, mimir::formalism::FluentTag >) {
         tag_id = 2;
      } else {
         tag_id = 3;
      }
      fact_keys.insert(
         pack_u32_u32(static_cast< uint32_t >(atom->get_index()), static_cast< uint32_t >(tag_id))
      );
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
   hash_map< RelationRef, hash_set< int64_t > >& relation_to_symbols,
   hash_map< int64_t, hash_set< RelationRef > >& symbol_to_relations,
   std::span< const std::string > extra_objects
)
{
   if(config_.export_action_targets) {
      workspace_.action_targets.reserve(actions.size(), /*include_depth=*/false);
   }

   for(size_t action_pos = 0; action_pos < actions.size(); ++action_pos) {
      const auto& action = actions[action_pos];
      const std::string node_type = RelationFormatter::format_action_schema(*action->get_action());
      const int64_t relation_key = static_cast< int64_t >(action->get_index());
      const std::string action_name = (config_.export_node_names or config_.export_action_targets)
                                         ? RelationFormatter::format_action(action)
                                         : "";
      const std::string node_name = config_.export_node_names ? action_name : "";
      const auto relation_idx = get_or_add_relation_node_i64(
         node_type, relation_key, builder, node_indices, node_names, node_name
      );

      const std::string action_symbol_key = fmt::format(
         "{}{}", kDefaultTargetSymbolPrefix, action->get_index()
      );
      const std::string action_symbol_name = config_.export_node_names
                                                ? fmt::format(
                                                     "{}{}{}{}",
                                                     kDefaultTargetSymbolPrefix,
                                                     action->get_index(),
                                                     schema_key::kEdgeTypeSeparator,
                                                     action_name
                                                  )
                                                : action_symbol_key;
      const auto action_symbol_idx = get_or_add_symbol_special_node(
         action_symbol_key, action_symbol_name, builder, node_names
      );
      const auto action_symbol_id = get_or_assign_special_symbol_id(action_symbol_key);
      if(config_.include_lgan_edges) {
         workspace_.lgan_target_symbol_ids.insert(action_symbol_id);
      }
      if(config_.export_action_targets) {
         workspace_.action_targets.append(
            TargetRecord{
               .position = action_symbol_idx,
               .index = static_cast< int64_t >(action_pos),
               .candidate_id = static_cast< int64_t >(action_pos),
               .depth = std::nullopt,
               .name = action_name,
            },
            /*include_depth=*/false
         );
      }

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

      const auto rel_ref = relation_ref_for(node_type, relation_idx);
      track_relation_symbols_if_enabled(
         rel_ref,
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
   hash_map< RelationRef, hash_set< int64_t > >& relation_to_symbols,
   hash_map< int64_t, hash_set< RelationRef > >& symbol_to_relations
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
      const uint64_t history_key = pack_i32_u32(entry.dt, static_cast< uint32_t >(entry_idx));
      const std::string history_name = config_.export_node_names
                                          ? fmt::format("history:{}#{}", entry.dt, entry_idx)
                                          : "";
      const auto history_idx = get_or_add_relation_node_u64(
         "history", history_key, builder, node_indices, node_names, history_name
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

               auto& indices = workspace_.node_indices_i64[node_type];
               const int64_t relation_key = static_cast< int64_t >(atom->get_index());
               const bool is_new = indices.find(relation_key) == indices.end();
               const std::string node_name = config_.export_node_names
                                                ? RelationFormatter::format_literal< Tag >(
                                                     literal, std::nullopt
                                                  )
                                                : "";
               const auto relation_idx = get_or_add_relation_node_i64(
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

               const auto rel_ref = relation_ref_for(node_type, relation_idx);
               track_relation_symbols_if_enabled(
                  rel_ref,
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

void HGraphEncoderEngine::add_lgan_edges(
   BatchBuilder& builder,
   const hash_set< int64_t >& lgan_target_symbol_ids,
   const hash_map< int64_t, int64_t >& symbol_indices,
   const hash_map< RelationRef, hash_set< int64_t > >& relation_to_symbols,
   const hash_map< int64_t, hash_set< RelationRef > >& symbol_to_relations,
   const std::vector< std::string >& relation_type_names
)
{
   if(lgan_target_symbol_ids.empty()) {
      throw std::invalid_argument(
         "include_lgan_edges=true requires explicit target symbols, but none were encoded. "
         "For HGraph/Successor, pass actions with ignore_actions=false; for Horizon, ensure "
         "candidate target symbols exist (exclude_root_candidate may remove all)."
      );
   }
   if(symbol_indices.empty() or relation_to_symbols.empty()) {
      return;
   }

   auto decode_relation_ref =
      [&](RelationRef rel_ref) -> std::optional< std::pair< std::string, int64_t > > {
      const auto type_id = static_cast< uint32_t >(rel_ref >> 32);
      const auto relation_idx = static_cast< int64_t >(rel_ref & 0xffffffffULL);
      if(type_id >= relation_type_names.size()) {
         return std::nullopt;
      }
      return std::pair{relation_type_names[type_id], relation_idx};
   };

   auto emit_rel_to_target =
      [&](RelationRef rel_ref, const std::string& edge_label, int64_t target_idx) {
         const auto decoded = decode_relation_ref(rel_ref);
         if(not decoded.has_value()) {
            return;
         }
         const auto& [rel_type, relation_idx] = *decoded;
         append_edges(
            builder, rel_type, edge_label, config_.symbol_type_id, relation_idx, target_idx
         );
      };

   hash_map< RelationRef, hash_set< RelationRef > > rr_edges;
   rr_edges.reserve(relation_to_symbols.size());

   for(const auto target_symbol_id : lgan_target_symbol_ids) {
      auto target_idx_it = symbol_indices.find(target_symbol_id);
      if(target_idx_it == symbol_indices.end()) {
         continue;
      }
      const int64_t target_idx = target_idx_it->second;

      auto tn_it = symbol_to_relations.find(target_symbol_id);
      if(tn_it == symbol_to_relations.end()) {
         continue;
      }
      const auto& tn_relations = tn_it->second;
      if(tn_relations.empty()) {
         continue;
      }

      hash_set< int64_t > local_symbols;
      for(const auto rel_ref : tn_relations) {
         auto rel_symbols_it = relation_to_symbols.find(rel_ref);
         if(rel_symbols_it == relation_to_symbols.end()) {
            continue;
         }
         local_symbols.insert(rel_symbols_it->second.begin(), rel_symbols_it->second.end());
      }
      if(local_symbols.empty()) {
         continue;
      }

      hash_set< RelationRef > local_relations;
      for(const auto symbol_id : local_symbols) {
         auto rels_it = symbol_to_relations.find(symbol_id);
         if(rels_it == symbol_to_relations.end()) {
            continue;
         }
         for(const auto rel_ref : rels_it->second) {
            auto rel_symbols_it = relation_to_symbols.find(rel_ref);
            if(rel_symbols_it == relation_to_symbols.end() or rel_symbols_it->second.empty()) {
               continue;
            }
            bool fully_local = true;
            for(const auto arg_symbol : rel_symbols_it->second) {
               if(not local_symbols.contains(arg_symbol)) {
                  fully_local = false;
                  break;
               }
            }
            if(fully_local) {
               local_relations.insert(rel_ref);
            }
         }
      }

      hash_set< RelationRef > nn_relations;
      nn_relations.reserve(local_relations.size());
      for(const auto rel_ref : local_relations) {
         if(not tn_relations.contains(rel_ref)) {
            nn_relations.insert(rel_ref);
         }
      }

      for(const auto rel_ref : tn_relations) {
         emit_rel_to_target(rel_ref, config_.lgan_tn_edge_pos, target_idx);
      }
      for(const auto rel_ref : nn_relations) {
         emit_rel_to_target(rel_ref, config_.lgan_nn_edge_pos, target_idx);
      }

      hash_map< int64_t, std::vector< RelationRef > > local_symbol_to_relations;
      local_symbol_to_relations.reserve(local_symbols.size());
      for(const auto rel_ref : local_relations) {
         auto rel_symbols_it = relation_to_symbols.find(rel_ref);
         if(rel_symbols_it == relation_to_symbols.end()) {
            continue;
         }
         for(const auto symbol_id : rel_symbols_it->second) {
            if(local_symbols.contains(symbol_id)) {
               local_symbol_to_relations[symbol_id].push_back(rel_ref);
            }
         }
      }

      for(auto& [_, rels] : local_symbol_to_relations) {
         if(rels.size() < 2) {
            continue;
         }
         std::ranges::sort(rels);
         rels.erase(std::ranges::unique(rels).begin(), rels.end());
         if(rels.size() < 2) {
            continue;
         }
         for(size_t i = 0; i < rels.size(); ++i) {
            for(size_t j = i + 1; j < rels.size(); ++j) {
               const auto src_ref = rels[i];
               const auto dst_ref = rels[j];
               rr_edges[src_ref].insert(dst_ref);
               rr_edges[dst_ref].insert(src_ref);
            }
         }
      }
   }

   for(const auto& [src_ref, dst_refs] : rr_edges) {
      const auto src_decoded = decode_relation_ref(src_ref);
      if(not src_decoded.has_value()) {
         continue;
      }
      const auto& [src_type, src_idx] = *src_decoded;
      for(const auto dst_ref : dst_refs) {
         const auto dst_decoded = decode_relation_ref(dst_ref);
         if(not dst_decoded.has_value()) {
            continue;
         }
         const auto& [dst_type, dst_idx] = *dst_decoded;
         append_edges(builder, src_type, config_.lgan_rr_edge_pos, dst_type, src_idx, dst_idx);
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

uint64_t HGraphEncoderEngine::pack_u32_u32(uint32_t a, uint32_t b)
{
   return (static_cast< uint64_t >(a) << 32) | static_cast< uint64_t >(b);
}

uint64_t HGraphEncoderEngine::pack_i32_u32(int32_t a, uint32_t b)
{
   return pack_u32_u32(static_cast< uint32_t >(a), b);
}

HGraphEncoderEngine::RelationRef
HGraphEncoderEngine::relation_ref_from_parts(uint32_t type_id, uint32_t relation_idx)
{
   return pack_u32_u32(type_id, relation_idx);
}

uint32_t HGraphEncoderEngine::get_or_assign_relation_type_id(const std::string& node_type)
{
   auto it = workspace_.relation_type_ids.find(node_type);
   if(it != workspace_.relation_type_ids.end()) {
      return it->second;
   }
   const auto next = workspace_.relation_type_names.size();
   if(next >= std::numeric_limits< uint32_t >::max()) {
      throw std::overflow_error("too many relation node types for RelationRef");
   }
   const auto type_id = static_cast< uint32_t >(next);
   workspace_.relation_type_ids.emplace(node_type, type_id);
   workspace_.relation_type_names.emplace_back(node_type);
   return type_id;
}

std::string HGraphEncoderEngine::symbol_node_key(const mimir::formalism::Object& obj) const
{
   return RelationFormatter::format_object(*obj);
}

HGraphEncoderEngine::RelationRef
HGraphEncoderEngine::relation_ref_for(const std::string& node_type, int64_t relation_idx)
{
   if(relation_idx < 0
      or relation_idx > static_cast< int64_t >(std::numeric_limits< uint32_t >::max())) {
      throw std::overflow_error("relation index out of u32 range for RelationRef");
   }
   const auto type_id = get_or_assign_relation_type_id(node_type);
   return relation_ref_from_parts(type_id, static_cast< uint32_t >(relation_idx));
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
   const std::string key = symbol_node_key(obj);
   return get_or_add_symbol_node(symbol_id, key, key, builder, node_names);
}

int64_t HGraphEncoderEngine::get_or_add_symbol_special_node(
   std::string_view symbol_key,
   std::string_view symbol_name,
   BatchBuilder& builder,
   hash_map< std::string, std::vector< std::string > >& node_names
)
{
   const int64_t symbol_id = get_or_assign_special_symbol_id(symbol_key);
   return get_or_add_symbol_node(symbol_id, symbol_key, symbol_name, builder, node_names);
}

int64_t HGraphEncoderEngine::get_or_add_symbol_node(
   int64_t symbol_id,
   std::string_view symbol_key,
   std::string_view symbol_name,
   BatchBuilder& builder,
   hash_map< std::string, std::vector< std::string > >& node_names
)
{
   auto& symbol_nodes = workspace_.node_indices[config_.symbol_type_id];
   const std::string key(symbol_key);
   auto key_it = symbol_nodes.find(key);
   int64_t idx = -1;
   if(key_it != symbol_nodes.end()) {
      idx = key_it->second;
   } else {
      idx = static_cast< int64_t >(symbol_nodes.size());
      symbol_nodes.emplace(key, idx);
      builder.add_nodes(config_.symbol_type_id, idx + 1);
      if(config_.export_node_names) {
         node_names[config_.symbol_type_id].emplace_back(symbol_name);
      }
   }
   workspace_.symbol_indices[symbol_id] = idx;
   workspace_.symbol_key_to_id[std::string(symbol_key)] = symbol_id;
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
   (void) node_indices;
   auto& indices = workspace_.node_indices_i64[node_type];
   auto it = indices.find(key);
   if(it != indices.end()) {
      return it->second;
   }
   const auto idx = static_cast< int64_t >(indices.size());
   indices[key] = idx;
   if(config_.export_node_names) {
      node_names[node_type].emplace_back(node_name);
   }
   builder.add_nodes(node_type, idx + 1);
   return idx;
}

int64_t HGraphEncoderEngine::get_or_add_relation_node_u64(
   const std::string& node_type,
   uint64_t key,
   BatchBuilder& builder,
   hash_map< std::string, hash_map< std::string, int64_t > >& node_indices,
   hash_map< std::string, std::vector< std::string > >& node_names,
   std::string_view node_name
)
{
   (void) node_indices;
   auto& indices = workspace_.node_indices_u64[node_type];
   auto it = indices.find(key);
   if(it != indices.end()) {
      return it->second;
   }
   const auto idx = static_cast< int64_t >(indices.size());
   indices[key] = idx;
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
   if(node_type == config_.symbol_type_id) {
      const int64_t symbol_id = get_or_assign_special_symbol_id(node_key);
      return get_or_add_symbol_node(symbol_id, node_key, node_key, builder, node_names);
   }

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

BatchBuilder::BatchEncoding HGraphEncoderEngine::encode_batch(
   const batch_input::parsed::HGraphBatchInputs& inputs,
   std::optional< int > history_max_steps
)
{
   const batch_input::parsed::ActionPayload empty_actions{};
   BatchBuilder builder;
   builder.set_graph_kind("hetero");

   const size_t state_count = inputs.states.states.size();
   for(size_t idx = 0; idx < state_count; ++idx) {
      const auto& state_entry = inputs.states.states[idx];
      const auto& goals_entry = inputs.goals.at(idx);
      const auto& actions_entry = inputs.actions.at(idx);
      const auto& subgoal_layers_entry = inputs.subgoal_layers.at(idx);
      const auto& history_entry = inputs.history_subgoals.at(idx);

      const batch_input::parsed::ActionPayload& actions_payload = actions_entry.has_value()
                                                                     ? *actions_entry
                                                                     : empty_actions;
      const bool has_aux_payload = subgoal_layers_entry.has_value() or not actions_payload.empty()
                                   or history_entry.has_value();

      if(not goals_entry.has_value() and not has_aux_payload) {
         encode(state_entry.state, builder);
         builder.next_graph();
         continue;
      }

      GoalInputs goal_inputs;
      if(goals_entry.has_value()) {
         const auto* layers_ptr = subgoal_layers_entry.has_value() ? &(*subgoal_layers_entry)
                                                                   : nullptr;
         goal_inputs = batch_input::compose_goal_inputs(*goals_entry, layers_ptr);
      } else {
         goal_inputs = batch_input::default_goal_inputs_for_batch_state(state_entry);
         if(subgoal_layers_entry.has_value()) {
            size_t level = 1;
            for(const auto& layer : *subgoal_layers_entry) {
               goal_inputs.extend(layer, level);
               ++level;
            }
         }
      }

      if(history_entry.has_value()) {
         encode(
            state_entry.state,
            goal_inputs,
            actions_payload,
            *history_entry,
            history_max_steps,
            builder
         );
      } else {
         encode(state_entry.state, goal_inputs, actions_payload, builder);
      }
      builder.next_graph();
   }

   return builder.build();
}
}  // namespace mifrost
