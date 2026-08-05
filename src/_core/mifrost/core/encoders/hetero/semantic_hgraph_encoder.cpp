#include "semantic_hgraph_encoder.hpp"

#include <fmt/format.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <optional>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string_view>
#include <tuple>
#include <utility>

#include "mifrost/core/common_types.hpp"
#include "mifrost/core/encoders/common/relation_key.hpp"
#include "mifrost/core/encoders/common/target_metadata.hpp"
#include "mifrost/core/encoders/flat/semantic_flat_relation_view_bridge.hpp"
#include "mifrost/core/encoders/hetero/hetero_relation_schema.hpp"
#include "mifrost/core/encoders/hetero/semantic_horizon_hgraph_encoder.hpp"
#include "mifrost/core/schema_key_separators.hpp"

namespace mifrost {
namespace {

const std::shared_ptr< const SemanticTaskContext >& require_task_context(
   const std::shared_ptr< const SemanticTaskContext >& task_context
)
{
   if(not task_context) {
      throw std::invalid_argument("Semantic HGraph task context must not be null");
   }
   return task_context;
}

constexpr std::array< SemanticPredicateCategory, 3 > kCategoryOrder = {
   SemanticPredicateCategory::static_predicate,
   SemanticPredicateCategory::fluent,
   SemanticPredicateCategory::derived,
};
constexpr std::array< std::string_view, 4 > kGoalSuffixes = {
   "[g]",
   "[sg]",
   "[ssg]",
   "[sssg]",
};

std::string_view derivation_suffix(GoalDerivation derivation)
{
   switch(derivation) {
      case GoalDerivation::plain: return "[g]";
      case GoalDerivation::satisfied: return "[sat]";
      case GoalDerivation::unsatisfied: return "[unsat]";
      case GoalDerivation::added_satisfied: return "[sat+]";
      case GoalDerivation::added_unsatisfied: return "[sat-]";
   }
   return "";
}

std::string atom_name(
   const SemanticAtom& atom,
   const std::vector< SemanticPredicateSpec >& predicates,
   const std::vector< std::string >& objects,
   std::string_view predicate_suffix = {}
)
{
   std::string result = "(" + predicates.at(static_cast< size_t >(atom.predicate)).name;
   result += predicate_suffix;
   for(const auto argument : atom.arguments) {
      result += " ";
      result += objects.at(static_cast< size_t >(argument));
   }
   result += ")";
   return result;
}

std::string literal_name(
   const SemanticLiteral& literal,
   const std::vector< SemanticPredicateSpec >& predicates,
   const std::vector< std::string >& objects,
   std::optional< size_t > goal_level,
   std::optional< GoalDerivation > derivation = std::nullopt,
   std::string_view predicate_suffix = {}
)
{
   std::string result = literal.positive ? "[+]" : "[-]";
   result += atom_name(literal.atom, predicates, objects, predicate_suffix);
   if(goal_level) {
      result += kGoalSuffixes.at(*goal_level);
   }
   if(derivation) {
      result += derivation_suffix(*derivation);
   }
   return result;
}

std::string action_name(
   const SemanticGroundAction& action,
   const std::vector< SemanticActionSpec >& actions,
   const std::vector< std::string >& objects
)
{
   std::string result = "(" + actions.at(static_cast< size_t >(action.action)).name + " ";
   for(size_t idx = 0; idx < action.arguments.size(); ++idx) {
      if(idx > 0) {
         result += " ";
      }
      result += objects.at(static_cast< size_t >(action.arguments[idx]));
   }
   result += ")";
   return result;
}

std::string semantic_atom_key(const SemanticAtom& atom)
{
   std::string key = std::to_string(atom.predicate);
   for(const auto argument : atom.arguments) {
      key += ":" + std::to_string(argument);
   }
   return key;
}

void validate_schema(
   const std::vector< SemanticPredicateSpec >& predicates,
   const std::vector< SemanticActionSpec >& actions
)
{
   std::set< std::string > predicate_names;
   for(const auto& predicate : predicates) {
      if(predicate.name.empty()) {
         throw std::invalid_argument("predicate name must not be empty");
      }
      if(predicate.arity < 0) {
         throw std::invalid_argument("predicate arity must be non-negative");
      }
      if(not predicate_names.insert(predicate.name).second) {
         throw std::invalid_argument("Semantic HGraph requires unique predicate names");
      }
   }
   std::set< std::string > action_names;
   for(const auto& action : actions) {
      if(action.name.empty()) {
         throw std::invalid_argument("action name must not be empty");
      }
      if(action.arity < 0) {
         throw std::invalid_argument("action arity must be non-negative");
      }
      if(not action_names.insert(action.name).second) {
         throw std::invalid_argument("Semantic HGraph requires unique action names");
      }
   }
}

void validate_atom(
   const SemanticAtom& atom,
   const std::vector< SemanticPredicateSpec >& predicates,
   size_t object_count
)
{
   if(atom.predicate < 0 or static_cast< size_t >(atom.predicate) >= predicates.size()) {
      throw std::invalid_argument("predicate index out of range");
   }
   const auto arity = predicates[static_cast< size_t >(atom.predicate)].arity;
   if(static_cast< int64_t >(atom.arguments.size()) != arity) {
      throw std::invalid_argument("predicate argument count does not match schema arity");
   }
   for(const auto argument : atom.arguments) {
      if(argument < 0 or static_cast< size_t >(argument) >= object_count) {
         throw std::invalid_argument("object index out of range");
      }
   }
}

void validate_input(
   const SemanticFlatRelationInput& input,
   const std::vector< SemanticPredicateSpec >& predicates,
   const std::vector< SemanticActionSpec >& actions,
   const SemanticHGraphEncoderConfig& config
)
{
   const auto& objects = semantic_objects(input);
   const auto& goals = semantic_goals(input);
   const auto& static_facts = semantic_static_facts(input);
   if(not config.allow_subgoal_layers_beyond_max_goal_level
      and input.subgoal_layers.size() > config.max_goal_level) {
      throw std::invalid_argument("subgoal layer exceeds max_goal_level");
   }
   std::set< std::string > object_names;
   for(const auto& object : objects) {
      if(object.empty()) {
         throw std::invalid_argument("object name must not be empty");
      }
      if(not object_names.insert(object).second) {
         throw std::invalid_argument("Semantic HGraph requires unique object names");
      }
   }
   for(const auto& fact : input.state_facts) {
      validate_atom(fact, predicates, objects.size());
   }
   for(const auto& fact : static_facts) {
      validate_atom(fact, predicates, objects.size());
   }
   const auto validate_literal = [&](const SemanticLiteral& literal) {
      validate_atom(literal.atom, predicates, objects.size());
   };
   for(const auto& goal : goals) {
      validate_literal(goal);
   }
   for(const auto& layer : input.subgoal_layers) {
      for(const auto& goal : layer) {
         validate_literal(goal);
      }
   }
   for(const auto& entry : input.history) {
      if(entry.dt >= 0) {
         throw std::invalid_argument("history_subgoals expects negative dt values");
      }
      for(const auto& literal : entry.literals) {
         validate_literal(literal);
      }
   }
   for(const auto& action : input.actions) {
      if(action.action < 0 or static_cast< size_t >(action.action) >= actions.size()) {
         throw std::invalid_argument("action index out of range");
      }
      if(static_cast< int64_t >(action.arguments.size())
         != actions[static_cast< size_t >(action.action)].arity) {
         throw std::invalid_argument("action argument count does not match schema arity");
      }
      for(const auto argument : action.arguments) {
         if(argument < 0 or static_cast< size_t >(argument) >= objects.size()) {
            throw std::invalid_argument("object index out of range");
         }
      }
   }
}

struct PreparedGoal {
   SemanticLiteral literal;
   size_t level = 0;
};

std::vector< PreparedGoal > prepare_goals(const SemanticFlatRelationInput& input)
{
   const auto& goals = semantic_goals(input);
   const auto levels = semantic_goal_levels(input);
   std::vector< SemanticLiteral > ordered;
   ordered.reserve(goals.size());
   for(const auto& goal : goals) {
      ordered.push_back(goal);
   }
   for(size_t layer = 0; layer < input.subgoal_layers.size(); ++layer) {
      for(const auto& goal : input.subgoal_layers[layer]) {
         ordered.push_back(goal);
      }
   }
   std::vector< PreparedGoal > result;
   result.reserve(ordered.size());
   for(const auto& goal : ordered) {
      result.push_back({goal, semantic_goal_level(levels, goal)});
   }
   return result;
}

}  // namespace

struct SemanticHGraphEncoderEngine::Impl {
   using RelationRef = uint64_t;

   struct Workspace {
      std::map< std::string, std::map< std::string, int64_t > > node_indices;
      std::map< std::string, std::map< SemanticAtom, int64_t > > atom_indices;
      std::map< std::string, std::map< SemanticGroundAction, int64_t > > action_indices;
      std::map< SemanticGroundAction, int64_t > action_symbol_ids;
      std::map< std::string, std::map< uint64_t, int64_t > > u64_indices;
      std::map< std::string, std::vector< std::string > > node_names;
      std::map< int64_t, int64_t > symbol_indices;
      std::map< std::string, int64_t > symbol_key_to_id;
      std::map< std::string, int64_t > special_symbol_ids;
      int64_t next_special_symbol_id = -1;
      std::map< RelationRef, std::set< int64_t > > relation_to_symbols;
      std::map< int64_t, std::set< RelationRef > > symbol_to_relations;
      std::map< std::string, uint32_t > relation_type_ids;
      std::vector< std::string > relation_type_names;
      std::set< int64_t > lgan_target_symbol_ids;
      TargetColumns targets;
      std::vector< std::string > target_groups;
      std::map< TargetSource, int64_t > target_group_ids;
      int64_t next_target_index = 0;
   };

   std::shared_ptr< const SemanticTaskContext > task_context;
   const std::vector< SemanticPredicateSpec >& predicates;
   const std::vector< SemanticActionSpec >& actions;
   Config config;
   // Persistent structured relation schema; resolves node/edge-type strings for both declared
   // and undeclared keys without re-deriving them per atom/literal (see
   // `HeteroRelationSchema::name_for`). Rebuilt whenever the relation universe changes
   // (construction, `configure_horizon()`); `update_relations()` never touches it.
   HeteroRelationSchema schema_;
   // Caller-mutable compat/export view, derived from `schema_` whenever it is rebuilt, and
   // independently replaceable via `update_relations()`.
   std::map< std::string, int > relation_arities;
   std::vector< std::tuple< std::string, std::string, std::string > > all_edge_types;

   [[nodiscard]] std::string_view predicate_name(int64_t predicate_index) const
   {
      return predicates.at(static_cast< size_t >(predicate_index)).name;
   }

   /// A `SemanticHGraphEncoderEngine` may legitimately be given empty predicate/action lists (the
   /// caller relies purely on topology relations, added to `relation_arities` after this call
   /// returns) — unlike `HeteroRelationSchemaBuilder::finalize()`, which throws on an empty
   /// builder, an empty predicate/action-derived schema here is not an error.
   [[nodiscard]] static HeteroRelationSchema finalize_schema(
      HeteroRelationSchemaBuilder builder,
      int max_goal_level,
      bool support_literals,
      const std::set< GoalDerivation >& goal_derivations,
      std::string_view empty_error_message
   )
   {
      if(builder.size() == 0) {
         return HeteroRelationSchema{};
      }
      return std::move(builder).finalize(
         max_goal_level, support_literals, goal_derivations, empty_error_message
      );
   }

   Impl(
      std::vector< SemanticPredicateSpec > predicate_specs,
      std::vector< SemanticActionSpec > action_specs,
      Config encoder_config
   )
       : Impl(
            std::make_shared< SemanticTaskContext >(SemanticTaskContext{
               .predicates = std::move(predicate_specs),
               .actions = std::move(action_specs),
            }),
            std::move(encoder_config)
         )
   {
   }

   Impl(std::shared_ptr< const SemanticTaskContext > context, Config encoder_config)
       : task_context(require_task_context(context)),
         predicates(task_context->predicates),
         actions(task_context->actions),
         config(std::move(encoder_config))
   {
      validate_schema(predicates, actions);
      if(config.max_goal_level >= kGoalSuffixes.size()) {
         throw std::invalid_argument("Semantic HGraph supports at most three subgoal layers");
      }
      build_relation_arities();
   }

   [[nodiscard]] bool has_target_source(TargetSource source) const
   {
      return config.target_sources.contains(source);
   }

   [[nodiscard]] bool has_anchor_source(TargetSource source) const
   {
      return has_target_source(source)
             or (config.include_lgan_edges and config.lgan_anchor_sources.contains(source));
   }

   void rebuild_all_edge_types()
   {
      all_edge_types.clear();
      for(const auto& [type, arity] : relation_arities) {
         const int effective = config.add_nullary_predicates and arity == 0 ? 1 : arity;
         for(int position = 0; position < effective; ++position) {
            const auto pos = std::to_string(position);
            all_edge_types.emplace_back(config.symbol_type_id, pos, type);
            all_edge_types.emplace_back(type, pos, config.symbol_type_id);
         }
         if(config.include_lgan_edges) {
            all_edge_types.emplace_back(type, config.lgan_tn_edge_pos, config.symbol_type_id);
            all_edge_types.emplace_back(type, config.lgan_nn_edge_pos, config.symbol_type_id);
         }
      }
      std::ranges::sort(all_edge_types);
      all_edge_types.erase(std::ranges::unique(all_edge_types).begin(), all_edge_types.end());
   }

   void build_relation_arities()
   {
      const std::set< std::string > top_types = {
         "object", "number", config.symbol_type_id, "_action_"
      };
      std::vector< std::pair< std::string, int64_t > > regular;
      HeteroRelationSchemaBuilder builder;
      for(const auto category : kCategoryOrder) {
         for(const auto& predicate : predicates) {
            if(predicate.category != category) {
               continue;
            }
            const auto arity = predicate.arity;
            builder.register_relation(
               predicate_relation_key(predicate.name),
               HeteroRelationLayout{arity},
               RelationUsage::state
            );
            if(not top_types.contains(predicate.name)) {
               regular.emplace_back(predicate.name, arity);
            }
         }
      }
      if(includes_plain_goal_derivation(config.goal_derivations)) {
         for(const auto& [name, arity] : regular) {
            for(size_t level = 0; level <= config.max_goal_level; ++level) {
               const GoalLevel goal_level(level);
               for(const bool positive : {true, false}) {
                  builder.register_relation(
                     predicate_relation_key(name, positive, goal_level),
                     HeteroRelationLayout{arity},
                     RelationUsage::goal
                  );
               }
            }
            if(config.support_literals) {
               for(const bool positive : {true, false}) {
                  builder.register_relation(
                     predicate_relation_key(name, positive),
                     HeteroRelationLayout{arity},
                     RelationUsage::goal
                  );
               }
            }
         }
      }
      for(const auto derivation : goal_satisfaction_derivations(config.goal_derivations)) {
         for(const auto& [name, arity] : regular) {
            for(size_t level = 0; level <= config.max_goal_level; ++level) {
               const GoalLevel goal_level(level);
               for(const bool positive : {true, false}) {
                  builder.register_relation(
                     predicate_relation_key(name, positive, goal_level, derivation),
                     HeteroRelationLayout{arity},
                     RelationUsage::goal_derivation
                  );
               }
            }
            if(config.support_literals) {
               for(const bool positive : {true, false}) {
                  builder.register_relation(
                     predicate_relation_key(name, positive, std::nullopt, derivation),
                     HeteroRelationLayout{arity},
                     RelationUsage::goal_derivation
                  );
               }
            }
         }
      }
      if(not config.ignore_actions) {
         const int64_t offset = (has_target_source(TargetSource::actions)
                                 or config.include_lgan_edges)
                                   ? 1
                                   : 0;
         for(const auto& action : actions) {
            builder.register_relation(
               action_relation_key(action.name),
               HeteroRelationLayout{action.arity + offset},
               RelationUsage::action
            );
         }
      }
      schema_ = finalize_schema(
         std::move(builder),
         static_cast< int >(config.max_goal_level),
         config.support_literals,
         config.goal_derivations,
         "SemanticHGraphEncoderEngine did not derive any relation types for this predicate/action "
         "schema"
      );
      relation_arities = schema_.relation_dict().arity;
      rebuild_all_edge_types();
   }

   void configure_horizon(const SemanticHorizonHGraphEncoderConfig& horizon)
   {
      const bool root_slot = root_in_state_relations(horizon.root_policy);
      const bool split = horizon.transition_mode == SemanticHorizonMode::full
                         and root_uses_split_state_relations(horizon.root_policy);
      const std::set< std::string > top_types = {
         "object", "number", config.symbol_type_id, "_action_"
      };
      std::vector< std::pair< std::string, int64_t > > regular;
      HeteroRelationSchemaBuilder builder;

      const auto register_relation = [&](RelationKey key, int64_t arity, RelationUsage usage) {
         builder.register_relation(std::move(key), HeteroRelationLayout{arity}, usage);
      };
      const auto root_relation = [&](const RelationKey& key, int64_t arity, RelationUsage usage) {
         register_relation(key, arity + (root_slot ? 1 : 0), usage);
      };
      const auto full_relation = [&](const RelationKey& key, int64_t arity, RelationUsage usage) {
         root_relation(key, arity, usage);
         if(split) {
            RelationKey anchored = key;
            anchored.state_anchored = true;
            register_relation(std::move(anchored), arity + 1, usage);
         }
      };
      for(const auto category : kCategoryOrder) {
         for(const auto& predicate : predicates) {
            if(predicate.category != category) {
               continue;
            }
            const auto arity = predicate.arity;
            full_relation(predicate_relation_key(predicate.name), arity, RelationUsage::state);
            if(not top_types.contains(predicate.name)) {
               regular.emplace_back(predicate.name, arity);
            }
         }
      }

      for(const auto& [name, arity] : regular) {
         if(includes_plain_goal_derivation(config.goal_derivations)) {
            for(size_t level = 0; level <= config.max_goal_level; ++level) {
               const GoalLevel goal_level(level);
               for(const bool positive : {true, false}) {
                  root_relation(
                     predicate_relation_key(name, positive, goal_level), arity, RelationUsage::goal
                  );
               }
            }
         }
         if(config.support_literals) {
            for(const bool positive : {true, false}) {
               const auto key = predicate_relation_key(name, positive);
               if(horizon.transition_mode == SemanticHorizonMode::delta) {
                  register_relation(key, arity + 1, RelationUsage::state);
               } else {
                  root_relation(key, arity, RelationUsage::state);
               }
            }
         }
         for(const auto derivation : goal_satisfaction_derivations(config.goal_derivations)) {
            if(derivation == GoalDerivation::satisfied
               or derivation == GoalDerivation::unsatisfied) {
               for(size_t level = 0; level <= config.max_goal_level; ++level) {
                  const GoalLevel goal_level(level);
                  for(const bool positive : {true, false}) {
                     full_relation(
                        predicate_relation_key(name, positive, goal_level, derivation),
                        arity,
                        RelationUsage::goal_satisfaction
                     );
                  }
               }
            } else if(horizon.transition_mode == SemanticHorizonMode::delta) {
               for(size_t level = 0; level <= config.max_goal_level; ++level) {
                  const GoalLevel goal_level(level);
                  for(const bool positive : {true, false}) {
                     register_relation(
                        predicate_relation_key(name, positive, goal_level, derivation),
                        arity + 1,
                        RelationUsage::goal_satisfaction
                     );
                  }
               }
            }
         }
      }
      if(not config.ignore_actions) {
         for(const auto& action : actions) {
            register_relation(
               action_relation_key(action.name), action.arity + 1, RelationUsage::action
            );
         }
      }

      schema_ = finalize_schema(
         std::move(builder),
         static_cast< int >(config.max_goal_level),
         config.support_literals,
         config.goal_derivations,
         "SemanticHGraphEncoderEngine (horizon) did not derive any relation types for this "
         "predicate/action schema"
      );
      relation_arities = schema_.relation_dict().arity;

      // Topology relations are fully user-configurable names, not schema-derived, so they bypass
      // the immutable `schema_` and go straight into the compat map (mirrors
      // `HorizonHGraphEncoderEngine::register_relation_type`).
      if(horizon.enable_parent_relation) {
         relation_arities[horizon.parent_relation] = 2;
      }
      if(horizon.enable_sibling_relation) {
         relation_arities[horizon.sibling_relation] = 2;
      }
      if(horizon.enable_cousin_relation) {
         relation_arities[horizon.cousin_relation] = 2;
      }

      rebuild_all_edge_types();
   }

   Workspace initialize_workspace(BatchBuilder& builder) const
   {
      builder.set_graph_kind("hetero");
      builder.set_node_feature_dim(config.symbol_type_id, 1);
      for(const auto& [type, arity] : relation_arities) {
         builder.set_node_feature_dim(type, arity);
      }
      Workspace workspace;
      for(const auto source : kCanonicalTargetSourceOrder) {
         if(config.target_sources.contains(source)) {
            workspace.target_group_ids.emplace(
               source, static_cast< int64_t >(workspace.target_groups.size())
            );
            workspace.target_groups.emplace_back(target_source_group_name(source));
         }
      }
      return workspace;
   }

   static void add_edge(
      BatchBuilder& builder,
      const std::string& src,
      const std::string& relation,
      const std::string& dst,
      int64_t src_index,
      int64_t dst_index
   )
   {
      builder.add_edge(src, relation, dst, src_index, dst_index);
   }

   int64_t special_symbol_id(Workspace& workspace, std::string_view key) const
   {
      const std::string owned(key);
      if(const auto it = workspace.special_symbol_ids.find(owned);
         it != workspace.special_symbol_ids.end()) {
         return it->second;
      }
      const auto id = workspace.next_special_symbol_id--;
      workspace.special_symbol_ids.emplace(owned, id);
      return id;
   }

   int64_t symbol_node(
      Workspace& workspace,
      int64_t id,
      std::string_view key,
      std::string_view name,
      BatchBuilder& builder
   ) const
   {
      auto& indices = workspace.node_indices[config.symbol_type_id];
      const std::string owned_key(key);
      auto [it, inserted] = indices.try_emplace(owned_key, static_cast< int64_t >(indices.size()));
      if(inserted) {
         builder.add_nodes(config.symbol_type_id, it->second + 1);
         if(config.export_node_names) {
            workspace.node_names[config.symbol_type_id].emplace_back(name);
         }
      }
      workspace.symbol_indices[id] = it->second;
      workspace.symbol_key_to_id[owned_key] = id;
      return it->second;
   }

   int64_t object_node(
      Workspace& workspace,
      int64_t object,
      const SemanticFlatRelationInput& input,
      BatchBuilder& builder
   ) const
   {
      const auto& name = semantic_objects(input).at(static_cast< size_t >(object));
      return symbol_node(workspace, object, name, name, builder);
   }

   int64_t special_node(
      Workspace& workspace,
      std::string_view key,
      std::string_view name,
      BatchBuilder& builder
   ) const
   {
      return symbol_node(workspace, special_symbol_id(workspace, key), key, name, builder);
   }

   int64_t atom_node(
      Workspace& workspace,
      const std::string& type,
      const SemanticAtom& atom,
      std::string_view name,
      BatchBuilder& builder
   ) const
   {
      auto& indices = workspace.atom_indices[type];
      auto [it, inserted] = indices.try_emplace(atom, static_cast< int64_t >(indices.size()));
      if(inserted) {
         builder.add_nodes(type, it->second + 1);
         if(config.export_node_names) {
            workspace.node_names[type].emplace_back(name);
         }
      }
      return it->second;
   }

   int64_t action_node(
      Workspace& workspace,
      const std::string& type,
      const SemanticGroundAction& action,
      std::string_view name,
      BatchBuilder& builder
   ) const
   {
      auto& indices = workspace.action_indices[type];
      auto [it, inserted] = indices.try_emplace(action, static_cast< int64_t >(indices.size()));
      if(inserted) {
         builder.add_nodes(type, it->second + 1);
         if(config.export_node_names) {
            workspace.node_names[type].emplace_back(name);
         }
      }
      return it->second;
   }

   int64_t action_symbol_id(Workspace& workspace, const SemanticGroundAction& action) const
   {
      auto [it, _] = workspace.action_symbol_ids.try_emplace(
         action, static_cast< int64_t >(workspace.action_symbol_ids.size())
      );
      return it->second;
   }

   int64_t history_node(
      Workspace& workspace,
      uint64_t key,
      std::string_view name,
      BatchBuilder& builder
   ) const
   {
      auto& indices = workspace.u64_indices["history"];
      auto [it, inserted] = indices.try_emplace(key, static_cast< int64_t >(indices.size()));
      if(inserted) {
         builder.add_nodes("history", it->second + 1);
         if(config.export_node_names) {
            workspace.node_names["history"].emplace_back(name);
         }
      }
      return it->second;
   }

   uint32_t relation_type_id(Workspace& workspace, const std::string& type) const
   {
      if(const auto it = workspace.relation_type_ids.find(type);
         it != workspace.relation_type_ids.end()) {
         return it->second;
      }
      if(workspace.relation_type_names.size() >= std::numeric_limits< uint32_t >::max()) {
         throw std::overflow_error("too many relation node types for RelationRef");
      }
      const auto id = static_cast< uint32_t >(workspace.relation_type_names.size());
      workspace.relation_type_ids.emplace(type, id);
      workspace.relation_type_names.push_back(type);
      return id;
   }

   RelationRef relation_ref(Workspace& workspace, const std::string& type, int64_t index) const
   {
      if(index < 0 or index > std::numeric_limits< uint32_t >::max()) {
         throw std::overflow_error("relation index out of u32 range for RelationRef");
      }
      return (static_cast< uint64_t >(relation_type_id(workspace, type)) << 32)
             | static_cast< uint32_t >(index);
   }

   void track(Workspace& workspace, RelationRef relation, std::span< const int64_t > symbols) const
   {
      if(not config.include_lgan_edges) {
         return;
      }
      for(const auto symbol : symbols) {
         workspace.relation_to_symbols[relation].insert(symbol);
         workspace.symbol_to_relations[symbol].insert(relation);
      }
   }

   int64_t append_target(
      Workspace& workspace,
      int64_t position,
      TargetSource source,
      std::string name
   ) const
   {
      auto group_it = workspace.target_group_ids.find(source);
      if(group_it == workspace.target_group_ids.end()) {
         const auto id = static_cast< int64_t >(workspace.target_groups.size());
         group_it = workspace.target_group_ids.emplace(source, id).first;
         workspace.target_groups.emplace_back(target_source_group_name(source));
      }
      const auto index = workspace.next_target_index++;
      append_target_candidate_row(
         workspace.targets,
         TargetCandidateRow{
            .position = position,
            .index = index,
            .candidate_id = index,
            .depth = std::nullopt,
            .group_id = group_it->second,
            .name = std::move(name),
         },
         TargetCandidateAppendConfig{.include_depth = false, .include_group = true}
      );
      return index;
   }

   std::vector< int64_t > atom_symbols(
      Workspace& workspace,
      const SemanticAtom& atom,
      const SemanticFlatRelationInput& input,
      BatchBuilder& builder
   ) const
   {
      std::vector< int64_t > result;
      if(atom.arguments.empty()) {
         if(not config.add_nullary_predicates) {
            return result;
         }
         special_node(workspace, config.nullary_object_name, config.nullary_object_name, builder);
         result.push_back(special_symbol_id(workspace, config.nullary_object_name));
      } else {
         result.reserve(atom.arguments.size());
         for(const auto object : atom.arguments) {
            object_node(workspace, object, input, builder);
            result.push_back(object);
         }
      }
      return result;
   }

   void connect_symbols(
      Workspace& workspace,
      BatchBuilder& builder,
      const std::string& type,
      int64_t relation_index,
      std::span< const int64_t > symbols
   ) const
   {
      for(size_t position = 0; position < symbols.size(); ++position) {
         const auto symbol_index = workspace.symbol_indices.at(symbols[position]);
         const auto label = std::to_string(position);
         add_edge(builder, config.symbol_type_id, label, type, symbol_index, relation_index);
         add_edge(builder, type, label, config.symbol_type_id, relation_index, symbol_index);
      }
   }

   void encode_fact(
      Workspace& workspace,
      const SemanticAtom& atom,
      const SemanticFlatRelationInput& input,
      BatchBuilder& builder
   ) const
   {
      const auto& predicate = predicates.at(static_cast< size_t >(atom.predicate));
      if(atom.arguments.empty() and not config.add_nullary_predicates) {
         return;
      }
      const auto index = atom_node(
         workspace,
         predicate.name,
         atom,
         config.export_node_names ? atom_name(atom, predicates, semantic_objects(input)) : "",
         builder
      );
      const auto symbols = atom_symbols(workspace, atom, input, builder);
      connect_symbols(workspace, builder, predicate.name, index, symbols);
      track(workspace, relation_ref(workspace, predicate.name, index), symbols);
   }

   void encode_successor_fact(
      Workspace& workspace,
      const SemanticAtom& atom,
      const SemanticFlatRelationInput& input,
      BatchBuilder& builder,
      std::string_view predicate_suffix,
      std::optional< bool > polarity = std::nullopt
   ) const
   {
      if(atom.arguments.empty() and not config.add_nullary_predicates) {
         return;
      }
      const std::string type = schema_.name_for(predicate_relation_key(
         predicate_name(atom.predicate), polarity, std::nullopt, std::nullopt, predicate_suffix
      ));
      auto formatted = config.export_node_names
                          ? atom_name(atom, predicates, semantic_objects(input), predicate_suffix)
                          : std::string{};
      if(polarity) {
         formatted = (*polarity ? "[+]" : "[-]") + formatted;
      }
      const auto index = atom_node(
         workspace, type, atom, config.export_node_names ? formatted : "", builder
      );
      const auto symbols = atom_symbols(workspace, atom, input, builder);
      connect_symbols(workspace, builder, type, index, symbols);
      track(workspace, relation_ref(workspace, type, index), symbols);
   }

   void encode_goal(
      Workspace& workspace,
      const PreparedGoal& prepared,
      const SemanticFlatRelationInput& input,
      BatchBuilder& builder,
      std::optional< GoalDerivation > derivation = std::nullopt,
      std::string_view predicate_suffix = {}
   ) const
   {
      const auto& literal = prepared.literal;
      const auto& predicate = predicates.at(static_cast< size_t >(literal.atom.predicate));
      if(literal.atom.arguments.empty() and not config.add_nullary_predicates) {
         return;
      }
      const std::string type = schema_.name_for(predicate_relation_key(
         predicate_name(literal.atom.predicate),
         literal.positive,
         GoalLevel(prepared.level),
         derivation,
         predicate_suffix
      ));
      const auto formatted = config.export_node_names ? literal_name(
                                                           literal,
                                                           predicates,
                                                           semantic_objects(input),
                                                           prepared.level,
                                                           derivation,
                                                           predicate_suffix
                                                        )
                                                      : std::string{};
      if(predicate_suffix.empty()) {
         const auto arity = predicate.arity == 0 and config.add_nullary_predicates
                               ? 1
                               : predicate.arity;
         builder.set_node_feature_dim(type, arity);
      }
      const auto index = atom_node(
         workspace, type, literal.atom, config.export_node_names ? formatted : "", builder
      );
      auto symbols = atom_symbols(workspace, literal.atom, input, builder);
      size_t offset = 0;
      const auto source = prepared.level > 0 ? TargetSource::subgoals : TargetSource::goals;
      std::vector< int64_t > tracked_symbols = symbols;
      if(not derivation and has_anchor_source(source)) {
         const auto key = fmt::format(
            "{}{}{}{}{}{}{}{}{}{}",
            config.target_symbol_prefix,
            target_source_group_name(source),
            schema_key::kEdgeTypeSeparator,
            type,
            schema_key::kEdgeTypeSeparator,
            literal.positive ? 1 : 0,
            schema_key::kEdgeTypeSeparator,
            semantic_atom_key(literal.atom),
            schema_key::kEdgeTypeSeparator,
            prepared.level
         );
         const auto name = config.export_node_names
                              ? fmt::format(
                                   "{}{}{}", key, schema_key::kEdgeTypeSeparator, formatted
                                )
                              : key;
         const auto symbol_index = special_node(workspace, key, name, builder);
         const auto id = special_symbol_id(workspace, key);
         tracked_symbols.push_back(id);
         workspace.lgan_target_symbol_ids.insert(id);
         if(has_target_source(source)) {
            append_target(workspace, symbol_index, source, formatted);
         }
         add_edge(builder, config.symbol_type_id, "0", type, symbol_index, index);
         add_edge(builder, type, "0", config.symbol_type_id, index, symbol_index);
         offset = 1;
      }
      for(size_t position = 0; position < symbols.size(); ++position) {
         const auto symbol_index = workspace.symbol_indices.at(symbols[position]);
         const auto label = std::to_string(position + offset);
         add_edge(builder, config.symbol_type_id, label, type, symbol_index, index);
         add_edge(builder, type, label, config.symbol_type_id, index, symbol_index);
      }
      track(workspace, relation_ref(workspace, type, index), tracked_symbols);
   }

   void encode_actions(
      Workspace& workspace,
      const SemanticFlatRelationInput& input,
      BatchBuilder& builder
   ) const
   {
      if(config.ignore_actions) {
         return;
      }
      const bool target_actions = has_target_source(TargetSource::actions);
      const bool target_symbol = target_actions or config.include_lgan_edges;
      for(const auto& action : input.actions) {
         const auto& spec = actions.at(static_cast< size_t >(action.action));
         const auto formatted = config.export_node_names
                                   ? action_name(action, actions, semantic_objects(input))
                                   : std::string{};
         const auto index = action_node(
            workspace, spec.name, action, config.export_node_names ? formatted : "", builder
         );
         std::vector< int64_t > symbols;
         if(target_symbol) {
            const auto key = fmt::format(
               "{}{}", config.target_symbol_prefix, action_symbol_id(workspace, action)
            );
            const auto name = config.export_node_names
                                 ? fmt::format(
                                      "{}{}{}", key, schema_key::kEdgeTypeSeparator, formatted
                                   )
                                 : key;
            const auto target_index = special_node(workspace, key, name, builder);
            const auto target_id = special_symbol_id(workspace, key);
            symbols.push_back(target_id);
            if(config.include_lgan_edges) {
               workspace.lgan_target_symbol_ids.insert(target_id);
            }
            if(target_actions) {
               append_target(workspace, target_index, TargetSource::actions, formatted);
            }
         }
         for(const auto object : action.arguments) {
            object_node(workspace, object, input, builder);
            symbols.push_back(object);
         }
         connect_symbols(workspace, builder, spec.name, index, symbols);
         track(workspace, relation_ref(workspace, spec.name, index), symbols);
      }
   }

   void encode_history(
      Workspace& workspace,
      const SemanticFlatRelationInput& input,
      BatchBuilder& builder
   ) const
   {
      struct Entry {
         int64_t dt;
         std::vector< SemanticLiteral > literals;
      };
      std::vector< Entry > entries;
      for(const auto& entry : input.history) {
         if(input.history_max_steps and std::abs(entry.dt) > *input.history_max_steps) {
            continue;
         }
         entries.push_back({entry.dt, entry.literals});
      }
      std::ranges::stable_sort(entries, {}, &Entry::dt);
      if(entries.empty()) {
         return;
      }
      builder.set_node_feature_dim("history", 1);
      std::vector< float > history_dt;
      bool wrote_history = false;
      for(size_t entry_index = 0; entry_index < entries.size(); ++entry_index) {
         const auto& entry = entries[entry_index];
         const uint64_t key = (static_cast< uint64_t >(static_cast< uint32_t >(entry.dt)) << 32)
                              | static_cast< uint32_t >(entry_index);
         const auto history_index = history_node(
            workspace,
            key,
            config.export_node_names ? fmt::format("history:{}#{}", entry.dt, entry_index) : "",
            builder
         );
         history_dt.push_back(static_cast< float >(entry.dt));
         for(const auto& literal : entry.literals) {
            if(literal.atom.arguments.empty() and not config.add_nullary_predicates) {
               continue;
            }
            const std::string type = schema_.name_for(
               predicate_relation_key(predicate_name(literal.atom.predicate), literal.positive)
            );
            const auto formatted = config.export_node_names ? literal_name(
                                                                 literal,
                                                                 predicates,
                                                                 semantic_objects(input),
                                                                 std::nullopt
                                                              )
                                                            : std::string{};
            auto& indices = workspace.atom_indices[type];
            const bool is_new = not indices.contains(literal.atom);
            const auto relation_index = atom_node(
               workspace, type, literal.atom, config.export_node_names ? formatted : "", builder
            );
            auto symbols = atom_symbols(workspace, literal.atom, input, builder);
            std::vector< int64_t > tracked = symbols;
            if(has_anchor_source(TargetSource::history)) {
               const auto target_name = fmt::format(
                  "history:{}#{}:{}", entry.dt, entry_index, formatted
               );
               const auto target_key = fmt::format(
                  "{}{}{}{}{}{}{}{}{}{}{}{}",
                  config.target_symbol_prefix,
                  target_source_group_name(TargetSource::history),
                  schema_key::kEdgeTypeSeparator,
                  entry.dt,
                  schema_key::kEdgeTypeSeparator,
                  entry_index,
                  schema_key::kEdgeTypeSeparator,
                  type,
                  schema_key::kEdgeTypeSeparator,
                  literal.positive ? 1 : 0,
                  schema_key::kEdgeTypeSeparator,
                  semantic_atom_key(literal.atom)
               );
               const auto target_symbol_name = config.export_node_names
                                                  ? fmt::format(
                                                       "{}{}{}",
                                                       target_key,
                                                       schema_key::kEdgeTypeSeparator,
                                                       target_name
                                                    )
                                                  : target_key;
               const auto target_index = special_node(
                  workspace, target_key, target_symbol_name, builder
               );
               const auto target_id = special_symbol_id(workspace, target_key);
               tracked.push_back(target_id);
               if(config.include_lgan_edges) {
                  workspace.lgan_target_symbol_ids.insert(target_id);
               }
               if(has_target_source(TargetSource::history)) {
                  append_target(workspace, target_index, TargetSource::history, target_name);
               }
               const auto label = std::to_string(symbols.size());
               add_edge(builder, config.symbol_type_id, label, type, target_index, relation_index);
               add_edge(builder, type, label, config.symbol_type_id, relation_index, target_index);
            }
            if(is_new) {
               connect_symbols(workspace, builder, type, relation_index, symbols);
            }
            track(workspace, relation_ref(workspace, type, relation_index), tracked);
            add_edge(
               builder, type, config.history_link_relation, "history", relation_index, history_index
            );
            add_edge(
               builder, "history", config.history_link_relation, type, history_index, relation_index
            );
            wrote_history = true;
         }
      }
      if(not history_dt.empty()) {
         builder.add_node_features("history", "history_dt", history_dt, 1);
      }
      if(wrote_history) {
         builder.set_schema_flag("history", true);
      }
   }

   void add_lgan(Workspace& workspace, BatchBuilder& builder) const
   {
      if(not config.include_lgan_edges) {
         return;
      }
      if(workspace.lgan_target_symbol_ids.empty()) {
         throw std::invalid_argument(
            "include_lgan_edges=true requires explicit target symbols, but none were encoded. "
            "For HGraph/Successor, pass actions with ignore_actions=false or enable "
            "lgan_anchor_sources/target_sources such as 'goal', 'subgoal', or 'history'; "
            "for Horizon, ensure candidate target symbols exist "
            "(root_policy may remove all)."
         );
      }
      const auto decode =
         [&](RelationRef ref) -> std::optional< std::pair< std::string, int64_t > > {
         const auto type_id = static_cast< uint32_t >(ref >> 32);
         if(type_id >= workspace.relation_type_names.size()) {
            return std::nullopt;
         }
         return std::pair{
            workspace.relation_type_names[type_id],
            static_cast< int64_t >(ref & 0xffffffffULL),
         };
      };
      std::map< RelationRef, std::set< RelationRef > > rr_edges;
      for(const auto target_symbol : workspace.lgan_target_symbol_ids) {
         if(not workspace.symbol_indices.contains(target_symbol)
            or not workspace.symbol_to_relations.contains(target_symbol)) {
            continue;
         }
         const auto target_index = workspace.symbol_indices.at(target_symbol);
         const auto& tn = workspace.symbol_to_relations.at(target_symbol);
         std::set< int64_t > local_symbols;
         for(const auto relation : tn) {
            if(workspace.relation_to_symbols.contains(relation)) {
               local_symbols.insert(
                  workspace.relation_to_symbols.at(relation).begin(),
                  workspace.relation_to_symbols.at(relation).end()
               );
            }
         }
         std::set< RelationRef > local_relations;
         for(const auto symbol : local_symbols) {
            if(not workspace.symbol_to_relations.contains(symbol)) {
               continue;
            }
            for(const auto relation : workspace.symbol_to_relations.at(symbol)) {
               const auto& relation_symbols = workspace.relation_to_symbols.at(relation);
               if(std::ranges::all_of(relation_symbols, [&](int64_t item) {
                     return local_symbols.contains(item);
                  })) {
                  local_relations.insert(relation);
               }
            }
         }
         for(const auto relation : local_relations) {
            const auto decoded = decode(relation);
            if(not decoded) {
               continue;
            }
            const auto& [type, index] = *decoded;
            add_edge(
               builder,
               type,
               tn.contains(relation) ? config.lgan_tn_edge_pos : config.lgan_nn_edge_pos,
               config.symbol_type_id,
               index,
               target_index
            );
         }
         std::map< int64_t, std::vector< RelationRef > > symbol_relations;
         for(const auto relation : local_relations) {
            for(const auto symbol : workspace.relation_to_symbols.at(relation)) {
               if(local_symbols.contains(symbol)) {
                  symbol_relations[symbol].push_back(relation);
               }
            }
         }
         for(auto& [_, relations] : symbol_relations) {
            std::ranges::sort(relations);
            relations.erase(std::ranges::unique(relations).begin(), relations.end());
            for(size_t lhs = 0; lhs < relations.size(); ++lhs) {
               for(size_t rhs = lhs + 1; rhs < relations.size(); ++rhs) {
                  rr_edges[relations[lhs]].insert(relations[rhs]);
                  rr_edges[relations[rhs]].insert(relations[lhs]);
               }
            }
         }
      }
      for(const auto& [src_ref, dst_refs] : rr_edges) {
         const auto src = decode(src_ref);
         if(not src) {
            continue;
         }
         for(const auto dst_ref : dst_refs) {
            const auto dst = decode(dst_ref);
            if(dst) {
               add_edge(
                  builder, src->first, config.lgan_rr_edge_pos, dst->first, src->second, dst->second
               );
            }
         }
      }
   }

   void finalize(
      Workspace& workspace,
      BatchBuilder& builder,
      const std::vector< std::string >* object_names_override = nullptr
   ) const
   {
      if(config.export_node_names) {
         for(const auto& [type, _] : relation_arities) {
            if(not workspace.node_names.contains(type)) {
               builder.set_node_names(type, {});
            }
         }
         for(const auto& [type, names] : workspace.node_names) {
            builder.set_node_names(type, names);
         }
         if(not workspace.node_names.contains(config.symbol_type_id)) {
            builder.set_node_names(config.symbol_type_id, {});
            builder.set_object_names({});
         } else if(object_names_override != nullptr) {
            builder.set_object_names(*object_names_override);
         } else {
            builder.set_object_names(workspace.node_names.at(config.symbol_type_id));
         }
      }
      const bool emit_targets = has_target_source(TargetSource::actions)
                                or has_target_source(TargetSource::goals)
                                or has_target_source(TargetSource::subgoals)
                                or has_target_source(TargetSource::history);
      if(emit_targets) {
         emit_target_metadata(
            builder,
            workspace.targets,
            TargetMetadataEmitConfig{
               .position_node_type_id = config.symbol_type_id,
               .symbol_prefix = config.target_symbol_prefix,
               .include_depth = false,
               .include_group = true,
               .include_names = false,
               .groups = workspace.target_groups,
               .parent_relation = std::nullopt,
            }
         );
         if(config.export_node_names) {
            if(workspace.targets.names.empty()) {
               builder.set_graph_attr(std::string(kTargetNamesAttr), std::vector< std::string >{});
            } else {
               builder.add_lazy_target_names(std::span(workspace.targets.names));
            }
         }
      }
      if(config.include_empty_edge_types) {
         for(const auto& [src, relation, dst] : all_edge_types) {
            builder.ensure_edge_type(src, relation, dst);
         }
      }
   }

   void encode(const SemanticFlatRelationInput& input, BatchBuilder& builder) const
   {
      validate_input(input, predicates, actions, config);
      const auto& objects = semantic_objects(input);
      const auto& static_facts = semantic_static_facts(input);
      auto workspace = initialize_workspace(builder);
      for(size_t object = 0; object < objects.size(); ++object) {
         object_node(workspace, static_cast< int64_t >(object), input, builder);
      }
      if(config.add_nullary_predicates) {
         special_node(workspace, config.nullary_object_name, config.nullary_object_name, builder);
      }
      for(const auto category : kCategoryOrder) {
         if(category == SemanticPredicateCategory::static_predicate and not config.include_static) {
            continue;
         }
         const auto encode_facts = [&](const std::vector< SemanticAtom >& facts) {
            for(const auto& fact : facts) {
               if(predicates.at(static_cast< size_t >(fact.predicate)).category == category) {
                  encode_fact(workspace, fact, input, builder);
               }
            }
         };
         encode_facts(static_facts);
         encode_facts(input.state_facts);
      }
      const auto prepared = prepare_goals(input);
      if(includes_plain_goal_derivation(config.goal_derivations)) {
         for(const auto category : kCategoryOrder) {
            for(const auto& goal : prepared) {
               if(predicates.at(static_cast< size_t >(goal.literal.atom.predicate)).category
                  == category) {
                  encode_goal(workspace, goal, input, builder);
               }
            }
         }
      }
      encode_actions(workspace, input, builder);
      encode_history(workspace, input, builder);
      if(has_non_plain_goal_derivations(config.goal_derivations)) {
         std::set< std::pair< SemanticPredicateCategory, SemanticAtom > > facts;
         for(const auto& fact : static_facts) {
            facts.emplace(SemanticPredicateCategory::static_predicate, fact);
         }
         for(const auto& fact : input.state_facts) {
            const auto category = predicates.at(static_cast< size_t >(fact.predicate)).category;
            facts.emplace(category, fact);
         }
         for(const auto category : kCategoryOrder) {
            for(const auto& goal : prepared) {
               if(predicates.at(static_cast< size_t >(goal.literal.atom.predicate)).category
                  != category) {
                  continue;
               }
               const bool present = facts.contains({category, goal.literal.atom});
               const auto derivation = present == goal.literal.positive
                                          ? GoalDerivation::satisfied
                                          : GoalDerivation::unsatisfied;
               if(config.goal_derivations.contains(derivation)) {
                  encode_goal(workspace, goal, input, builder, derivation);
               }
            }
         }
      }
      add_lgan(workspace, builder);
      finalize(workspace, builder);
   }

   void encode_successor(
      const SemanticFlatRelationInput& current,
      const SemanticFlatRelationInput& successor,
      bool delta_mode,
      std::string_view successor_suffix,
      bool include_successor_goal_satisfaction,
      BatchBuilder& builder
   ) const
   {
      validate_input(current, predicates, actions, config);
      validate_input(successor, predicates, actions, config);
      if(semantic_objects(current) != semantic_objects(successor)) {
         throw std::invalid_argument(
            "current and successor semantic inputs require identical ordered object tables"
         );
      }

      auto workspace = initialize_workspace(builder);
      for(size_t object = 0; object < semantic_objects(current).size(); ++object) {
         object_node(workspace, static_cast< int64_t >(object), current, builder);
      }
      if(config.add_nullary_predicates) {
         special_node(workspace, config.nullary_object_name, config.nullary_object_name, builder);
      }

      for(const auto category : kCategoryOrder) {
         if(category == SemanticPredicateCategory::static_predicate and not config.include_static) {
            continue;
         }
         const auto encode_facts = [&](const std::vector< SemanticAtom >& facts) {
            for(const auto& fact : facts) {
               if(predicates.at(static_cast< size_t >(fact.predicate)).category == category) {
                  encode_fact(workspace, fact, current, builder);
               }
            }
         };
         encode_facts(semantic_static_facts(current));
         encode_facts(current.state_facts);
      }

      std::set< SemanticAtom > current_dynamic_facts;
      std::set< SemanticAtom > successor_dynamic_facts;
      for(const auto& fact : current.state_facts) {
         if(predicates.at(static_cast< size_t >(fact.predicate)).category
            != SemanticPredicateCategory::static_predicate) {
            current_dynamic_facts.insert(fact);
         }
      }
      for(const auto& fact : successor.state_facts) {
         if(predicates.at(static_cast< size_t >(fact.predicate)).category
            != SemanticPredicateCategory::static_predicate) {
            successor_dynamic_facts.insert(fact);
         }
      }

      if(delta_mode) {
         for(const auto& fact : successor_dynamic_facts) {
            if(not current_dynamic_facts.contains(fact)) {
               encode_successor_fact(workspace, fact, successor, builder, successor_suffix, true);
            }
         }
         for(const auto& fact : current_dynamic_facts) {
            if(not successor_dynamic_facts.contains(fact)) {
               encode_successor_fact(workspace, fact, current, builder, successor_suffix, false);
            }
         }
      } else {
         for(const auto category : {
                SemanticPredicateCategory::fluent,
                SemanticPredicateCategory::derived,
             }) {
            for(const auto& fact : successor.state_facts) {
               if(predicates.at(static_cast< size_t >(fact.predicate)).category == category) {
                  encode_successor_fact(workspace, fact, successor, builder, successor_suffix);
               }
            }
         }
      }

      const auto prepared = prepare_goals(current);
      if(includes_plain_goal_derivation(config.goal_derivations)) {
         for(const auto category : kCategoryOrder) {
            for(const auto& goal : prepared) {
               if(predicates.at(static_cast< size_t >(goal.literal.atom.predicate)).category
                  == category) {
                  encode_goal(workspace, goal, current, builder);
               }
            }
         }
      }
      encode_actions(workspace, current, builder);
      encode_history(workspace, current, builder);

      if(not delta_mode and has_non_plain_goal_derivations(config.goal_derivations)) {
         // Accepts either a hash_set (pure membership test, `current_facts`
         // below) or a std::set (`successor_dynamic_facts`, whose sorted
         // iteration elsewhere in encode_successor determines deterministic
         // delta emission order and so must stay a std::set); this closure
         // only ever calls .contains() on it.
         const auto encode_satisfaction = [&](
                                             const auto& facts, std::string_view predicate_suffix
                                          ) {
            for(const auto category : kCategoryOrder) {
               for(const auto& goal : prepared) {
                  if(predicates.at(static_cast< size_t >(goal.literal.atom.predicate)).category
                     != category) {
                     continue;
                  }
                  const bool present = facts.contains(goal.literal.atom);
                  const auto derivation = present == goal.literal.positive
                                             ? GoalDerivation::satisfied
                                             : GoalDerivation::unsatisfied;
                  if(config.goal_derivations.contains(derivation)) {
                     encode_goal(workspace, goal, current, builder, derivation, predicate_suffix);
                  }
               }
            }
         };

         hash_set< SemanticAtom, SemanticAtomHash > current_facts;
         current_facts.insert(
            semantic_static_facts(current).begin(), semantic_static_facts(current).end()
         );
         for(const auto& fact : current.state_facts) {
            const auto category = predicates.at(static_cast< size_t >(fact.predicate)).category;
            current_facts.insert(fact);
         }
         encode_satisfaction(current_facts, {});
         if(include_successor_goal_satisfaction) {
            encode_satisfaction(successor_dynamic_facts, successor_suffix);
         }
      }

      add_lgan(workspace, builder);
      finalize(workspace, builder);
   }

   void encode_horizon(
      const SemanticTransitionDAG& dag,
      const SemanticHorizonHGraphEncoderConfig& horizon,
      BatchBuilder& builder
   ) const
   {
      if(horizon.transition_mode == SemanticHorizonMode::action and config.ignore_actions) {
         throw std::invalid_argument("Action horizon encoding requires ignore_actions=false.");
      }
      const auto& nodes = dag.nodes();
      const auto& root_input = nodes.front().state;
      const auto& root_objects = semantic_objects(root_input);
      const auto& root_static_facts = semantic_static_facts(root_input);
      auto workspace = initialize_workspace(builder);
      const auto target_key = [&](int64_t index) {
         return config.target_symbol_prefix + std::to_string(index);
      };
      for(const auto& node : nodes) {
         const auto key = target_key(node.index);
         special_node(workspace, key, key, builder);
         if(config.include_lgan_edges
            and (node.index != 0 or root_in_target_metadata(horizon.root_policy))) {
            workspace.lgan_target_symbol_ids.insert(special_symbol_id(workspace, key));
         }
      }
      for(size_t object = 0; object < root_objects.size(); ++object) {
         object_node(workspace, static_cast< int64_t >(object), root_input, builder);
      }
      if(config.add_nullary_predicates) {
         special_node(workspace, config.nullary_object_name, config.nullary_object_name, builder);
      }
      std::vector< std::string > physical_object_names;
      if(config.export_node_names) {
         physical_object_names.reserve(root_objects.size());
         for(const auto& object : root_objects) {
            physical_object_names.push_back(object);
         }
         if(config.add_nullary_predicates) {
            physical_object_names.push_back(config.nullary_object_name);
         }
      }
      const auto prefix_for = [&](int64_t index) {
         return target_key(index) + std::string(1, schema_key::kEdgeTypeSeparator);
      };

      const auto relation_node =
         [&](const std::string& type, const std::string& key, const std::string& name) {
            auto& indices = workspace.node_indices[type];
            auto [it, inserted] = indices.try_emplace(key, static_cast< int64_t >(indices.size()));
            if(inserted) {
               builder.add_nodes(type, it->second + 1);
               if(config.export_node_names) {
                  workspace.node_names[type].push_back(name);
               }
            }
            return it->second;
         };

      const auto encode_atom_relation = [&](
                                           const SemanticAtom& atom,
                                           const std::string& type,
                                           const std::string& node_prefix,
                                           const std::string& display,
                                           const std::optional< std::string >& state_key
                                        ) {
         if(atom.arguments.empty() and not config.add_nullary_predicates) {
            return;
         }
         const auto key = node_prefix + type + ":" + semantic_atom_key(atom);
         const auto relation_index = relation_node(
            type, key, config.export_node_names ? node_prefix + display : ""
         );
         std::vector< int64_t > symbols;
         if(state_key.has_value()) {
            special_node(workspace, *state_key, *state_key, builder);
            symbols.push_back(special_symbol_id(workspace, *state_key));
         }
         if(atom.arguments.empty()) {
            symbols.push_back(special_symbol_id(workspace, config.nullary_object_name));
         } else {
            for(const auto object : atom.arguments) {
               object_node(workspace, object, root_input, builder);
               symbols.push_back(object);
            }
         }
         connect_symbols(workspace, builder, type, relation_index, symbols);
         track(workspace, relation_ref(workspace, type, relation_index), symbols);
      };
      const auto encode_action_relation =
         [&](const SemanticGroundAction& action, int64_t node_index) {
            const auto& spec = actions.at(static_cast< size_t >(action.action));
            const auto prefix = prefix_for(node_index);
            const auto display = config.export_node_names
                                    ? action_name(action, actions, root_objects)
                                    : std::string{};
            const auto key = prefix + spec.name + ":" + std::to_string(action.action) + ":"
                             + std::to_string(action_symbol_id(workspace, action));
            const auto relation_index = relation_node(
               spec.name, key, config.export_node_names ? prefix + display : ""
            );
            std::vector< int64_t > symbols;
            const auto state_key = target_key(node_index);
            symbols.push_back(special_symbol_id(workspace, state_key));
            for(const auto object : action.arguments) {
               object_node(workspace, object, root_input, builder);
               symbols.push_back(object);
            }
            connect_symbols(workspace, builder, spec.name, relation_index, symbols);
            track(workspace, relation_ref(workspace, spec.name, relation_index), symbols);
         };

      const bool root_has_state_slot = root_in_state_relations(horizon.root_policy);
      const bool split_state = horizon.transition_mode == SemanticHorizonMode::full
                               and root_uses_split_state_relations(horizon.root_policy);
      const std::string root_prefix = root_has_state_slot ? prefix_for(0) : "_root|";
      const auto root_state_key = root_has_state_slot ? std::optional< std::string >(target_key(0))
                                                      : std::nullopt;
      hash_set< SemanticAtom, SemanticAtomHash > root_encoded_facts;
      for(const auto category : kCategoryOrder) {
         if(category == SemanticPredicateCategory::static_predicate and not config.include_static) {
            continue;
         }
         const auto encode_facts = [&](const std::vector< SemanticAtom >& facts) {
            for(const auto& fact : facts) {
               if(predicates.at(static_cast< size_t >(fact.predicate)).category != category) {
                  continue;
               }
               if(fact.arguments.empty() and not config.add_nullary_predicates) {
                  continue;
               }
               const auto& predicate = predicates.at(static_cast< size_t >(fact.predicate));
               encode_atom_relation(
                  fact,
                  predicate.name,
                  root_prefix,
                  config.export_node_names ? atom_name(fact, predicates, root_objects) : "",
                  root_state_key
               );
               root_encoded_facts.insert(fact);
            }
         };
         encode_facts(root_static_facts);
         encode_facts(root_input.state_facts);
      }

      const auto prepared_goals = prepare_goals(root_input);
      const auto encode_prepared_goal = [&](
                                           const PreparedGoal& goal,
                                           int64_t node_index,
                                           const std::string& node_prefix,
                                           const std::optional< std::string >& state_key,
                                           std::optional< GoalDerivation > derivation,
                                           bool state_suffix
                                        ) {
         const std::string type = schema_.name_for(predicate_relation_key(
            predicate_name(goal.literal.atom.predicate),
            goal.literal.positive,
            GoalLevel(goal.level),
            derivation,
            /*modifier=*/"",
            state_suffix
         ));
         encode_atom_relation(
            goal.literal.atom,
            type,
            node_prefix,
            config.export_node_names
               ? literal_name(goal.literal, predicates, root_objects, goal.level, derivation)
               : "",
            state_key
         );
      };
      if(includes_plain_goal_derivation(config.goal_derivations)) {
         for(const auto category : kCategoryOrder) {
            for(const auto& goal : prepared_goals) {
               if(predicates.at(static_cast< size_t >(goal.literal.atom.predicate)).category
                  == category) {
                  encode_prepared_goal(goal, 0, root_prefix, root_state_key, std::nullopt, false);
               }
            }
         }
      }
      if(has_non_plain_goal_derivations(config.goal_derivations)) {
         for(const auto category : kCategoryOrder) {
            for(const auto& goal : prepared_goals) {
               if(predicates.at(static_cast< size_t >(goal.literal.atom.predicate)).category
                  != category) {
                  continue;
               }
               const bool present = root_encoded_facts.contains(goal.literal.atom);
               const auto derivation = present == goal.literal.positive
                                          ? GoalDerivation::satisfied
                                          : GoalDerivation::unsatisfied;
               if(config.goal_derivations.contains(derivation)) {
                  encode_prepared_goal(goal, 0, root_prefix, root_state_key, derivation, false);
               }
            }
         }
      }

      std::set< SemanticAtom > root_dynamic;
      for(const auto& fact : root_input.state_facts) {
         if(predicates.at(static_cast< size_t >(fact.predicate)).category
               != SemanticPredicateCategory::static_predicate
            and (not fact.arguments.empty() or config.add_nullary_predicates)) {
            root_dynamic.insert(fact);
         }
      }
      for(const auto& node : nodes) {
         if(node.index == 0) {
            continue;
         }
         const auto prefix = prefix_for(node.index);
         const auto state_key = std::optional< std::string >(target_key(node.index));
         if(horizon.transition_mode == SemanticHorizonMode::full) {
            hash_set< SemanticAtom, SemanticAtomHash > successor_facts;
            for(const auto category : {
                   SemanticPredicateCategory::fluent,
                   SemanticPredicateCategory::derived,
                }) {
               for(const auto& fact : node.state.state_facts) {
                  if(predicates.at(static_cast< size_t >(fact.predicate)).category != category
                     or (fact.arguments.empty() and not config.add_nullary_predicates)) {
                     continue;
                  }
                  const auto& predicate = predicates.at(static_cast< size_t >(fact.predicate));
                  const std::string type = schema_.name_for(predicate_relation_key(
                     predicate.name,
                     std::nullopt,
                     std::nullopt,
                     std::nullopt,
                     /*modifier=*/"",
                     split_state
                  ));
                  encode_atom_relation(
                     fact,
                     type,
                     prefix,
                     config.export_node_names ? atom_name(fact, predicates, root_objects) : "",
                     state_key
                  );
                  successor_facts.insert(fact);
               }
            }
            if(not config.ignore_actions and node.incoming_action.has_value()) {
               encode_action_relation(*node.incoming_action, node.index);
            }
            if(has_non_plain_goal_derivations(config.goal_derivations)) {
               for(const auto category : kCategoryOrder) {
                  for(const auto& goal : prepared_goals) {
                     if(predicates.at(static_cast< size_t >(goal.literal.atom.predicate)).category
                        != category) {
                        continue;
                     }
                     const bool present = successor_facts.contains(goal.literal.atom);
                     const auto derivation = present == goal.literal.positive
                                                ? GoalDerivation::satisfied
                                                : GoalDerivation::unsatisfied;
                     if(config.goal_derivations.contains(derivation)) {
                        encode_prepared_goal(
                           goal, node.index, prefix, state_key, derivation, split_state
                        );
                     }
                  }
               }
            }
         } else if(horizon.transition_mode == SemanticHorizonMode::delta) {
            std::vector< SemanticLiteral > deltas;
            if(node.delta_literals.has_value()) {
               deltas = *node.delta_literals;
            } else {
               std::set< SemanticAtom > successor_dynamic;
               for(const auto& fact : node.state.state_facts) {
                  if(predicates.at(static_cast< size_t >(fact.predicate)).category
                        != SemanticPredicateCategory::static_predicate
                     and (not fact.arguments.empty() or config.add_nullary_predicates)) {
                     successor_dynamic.insert(fact);
                  }
               }
               for(const auto& fact : successor_dynamic) {
                  if(not root_dynamic.contains(fact)) {
                     deltas.push_back({fact, true});
                  }
               }
               for(const auto& fact : root_dynamic) {
                  if(not successor_dynamic.contains(fact)) {
                     deltas.push_back({fact, false});
                  }
               }
            }
            std::set< SemanticAtom > added;
            std::set< SemanticAtom > removed;
            for(const auto& literal : deltas) {
               const auto& predicate = predicates.at(static_cast< size_t >(literal.atom.predicate));
               const std::string type = schema_.name_for(
                  predicate_relation_key(predicate.name, literal.positive)
               );
               encode_atom_relation(
                  literal.atom,
                  type,
                  prefix,
                  config.export_node_names
                     ? literal_name(literal, predicates, root_objects, std::nullopt)
                     : "",
                  state_key
               );
               if(predicate.category != SemanticPredicateCategory::static_predicate) {
                  (literal.positive ? added : removed).insert(literal.atom);
               }
            }
            if(not config.ignore_actions and node.incoming_action.has_value()) {
               encode_action_relation(*node.incoming_action, node.index);
            }
            if(has_non_plain_goal_derivations(config.goal_derivations)) {
               for(const auto& goal : prepared_goals) {
                  const auto derivation = delta_goal_satisfaction_derivation(
                     goal.literal.positive,
                     added.contains(goal.literal.atom),
                     removed.contains(goal.literal.atom)
                  );
                  if(derivation.has_value() and config.goal_derivations.contains(*derivation)) {
                     encode_prepared_goal(goal, node.index, prefix, state_key, *derivation, false);
                  }
               }
            }
         } else if(node.incoming_action.has_value()) {
            encode_action_relation(*node.incoming_action, node.index);
         }
      }

      const bool exclude_root_topology = horizon.root_policy == RootPolicy::exclude;
      const auto encode_topology = [&](const std::string& type, int64_t src, int64_t dst) {
         const auto key = type + ":" + std::to_string(src) + ":" + std::to_string(dst);
         const auto name = fmt::format("{}({}->{})", type, src, dst);
         const auto relation_index = relation_node(type, key, config.export_node_names ? name : "");
         const std::array< int64_t, 2 > symbols = {
            special_symbol_id(workspace, target_key(src)),
            special_symbol_id(workspace, target_key(dst)),
         };
         connect_symbols(workspace, builder, type, relation_index, symbols);
         track(workspace, relation_ref(workspace, type, relation_index), symbols);
      };
      if(horizon.enable_parent_relation) {
         for(const auto& [parent, child] : dag.edges()) {
            if(not(exclude_root_topology and parent == 0)) {
               encode_topology(horizon.parent_relation, parent, child);
            }
         }
      }
      std::map< int64_t, std::vector< int64_t > > parent_to_children;
      if(horizon.enable_sibling_relation or horizon.enable_cousin_relation) {
         for(const auto& [parent, child] : dag.edges()) {
            if(not(exclude_root_topology and parent == 0)) {
               parent_to_children[parent].push_back(child);
            }
         }
      }
      std::set< std::pair< int64_t, int64_t > > sibling_pairs;
      if(horizon.enable_sibling_relation) {
         for(auto& [_, children] : parent_to_children) {
            std::ranges::sort(children);
            for(size_t lhs = 0; lhs < children.size(); ++lhs) {
               for(size_t rhs = lhs + 1; rhs < children.size(); ++rhs) {
                  const auto pair = std::minmax(children[lhs], children[rhs]);
                  if(sibling_pairs.emplace(pair.first, pair.second).second) {
                     encode_topology(horizon.sibling_relation, pair.first, pair.second);
                     encode_topology(horizon.sibling_relation, pair.second, pair.first);
                  }
               }
            }
         }
      }
      if(horizon.enable_cousin_relation) {
         std::set< std::pair< int64_t, int64_t > > cousin_pairs;
         for(const auto& [_, parents] : parent_to_children) {
            for(size_t lhs = 0; lhs < parents.size(); ++lhs) {
               for(size_t rhs = lhs + 1; rhs < parents.size(); ++rhs) {
                  const auto left = parent_to_children.find(parents[lhs]);
                  const auto right = parent_to_children.find(parents[rhs]);
                  if(left == parent_to_children.end() or right == parent_to_children.end()) {
                     continue;
                  }
                  for(const auto left_child : left->second) {
                     for(const auto right_child : right->second) {
                        if(left_child == right_child) {
                           continue;
                        }
                        const auto pair = std::minmax(left_child, right_child);
                        if(not sibling_pairs.contains(pair)
                           and cousin_pairs.emplace(pair.first, pair.second).second) {
                           encode_topology(horizon.cousin_relation, pair.first, pair.second);
                           encode_topology(horizon.cousin_relation, pair.second, pair.first);
                        }
                     }
                  }
               }
            }
         }
      }

      add_lgan(workspace, builder);

      if(has_target_source(TargetSource::states)) {
         const bool include_root = root_in_target_metadata(horizon.root_policy);
         dag.validate_candidate_ids(include_root);
         const bool explicit_ids = dag.candidate_id_coverage(include_root)
                                   == SemanticCandidateIdCoverage::complete;
         TargetColumns columns;
         std::vector< std::string > target_names;
         const auto group_id = workspace.target_group_ids.at(TargetSource::states);
         for(const auto& node : nodes) {
            if(node.index == 0 and not include_root) {
               continue;
            }
            if(config.export_node_names and not node.display_name.has_value()) {
               throw std::invalid_argument(
                  "Semantic Horizon requires display_name for every exported target node"
               );
            }
            const auto id = special_symbol_id(workspace, target_key(node.index));
            const auto position = workspace.symbol_indices.at(id);
            append_target_candidate_row(
               columns,
               TargetCandidateRow{
                  .position = position,
                  .index = node.index,
                  .candidate_id = explicit_ids ? node.candidate_id
                                               : std::optional< int64_t >(node.index),
                  .depth = node.depth,
                  .group_id = group_id,
                  .name = node.display_name.value_or(""),
               },
               TargetCandidateAppendConfig{
                  .include_depth = true,
                  .include_group = true,
                  .missing_candidate_id_prefix = "missing candidate_id for target node index ",
                  .duplicate_candidate_id_prefix = "duplicate candidate_id ",
               }
            );
            if(config.export_node_names) {
               target_names.push_back(*node.display_name);
            }
         }
         emit_target_metadata(
            builder,
            columns,
            TargetMetadataEmitConfig{
               .position_node_type_id = config.symbol_type_id,
               .symbol_prefix = config.target_symbol_prefix,
               .include_depth = true,
               .include_group = true,
               .include_names = false,
               .groups = workspace.target_groups,
               .parent_relation = horizon.parent_relation,
            }
         );
         if(config.export_node_names) {
            if(target_names.empty()) {
               builder.set_graph_attr(std::string(kTargetNamesAttr), std::vector< std::string >{});
            } else {
               builder.add_lazy_target_names(std::span(target_names));
            }
         }
      }

      finalize(workspace, builder, config.export_node_names ? &physical_object_names : nullptr);
   }
};

SemanticHGraphEncoderEngine::SemanticHGraphEncoderEngine(
   std::vector< SemanticPredicateSpec > predicates,
   std::vector< SemanticActionSpec > actions,
   Config config
)
    : impl_(std::make_unique< Impl >(std::move(predicates), std::move(actions), std::move(config)))
{
}

SemanticHGraphEncoderEngine::SemanticHGraphEncoderEngine(
   std::shared_ptr< const SemanticTaskContext > task_context,
   Config config
)
    : impl_(std::make_unique< Impl >(std::move(task_context), std::move(config)))
{
}

SemanticHGraphEncoderEngine::SemanticHGraphEncoderEngine(SemanticHGraphEncoderEngine&&) noexcept =
   default;
SemanticHGraphEncoderEngine& SemanticHGraphEncoderEngine::operator=(
   SemanticHGraphEncoderEngine&&
) noexcept = default;
SemanticHGraphEncoderEngine::~SemanticHGraphEncoderEngine() = default;

BatchBuilder::BatchEncoding SemanticHGraphEncoderEngine::encode(
   const SemanticFlatRelationInput& input
) const
{
   BatchBuilder builder;
   impl_->encode(input, builder);
   builder.next_graph();
   return builder.build();
}

BatchBuilder::BatchEncoding SemanticHGraphEncoderEngine::encode_views(
   const canonical::FlatRelationViewInput& input
) const
{
   BatchBuilder builder;
   encode_views(input, builder);
   builder.next_graph();
   return builder.build();
}

void SemanticHGraphEncoderEngine::encode(
   const SemanticFlatRelationInput& input,
   BatchBuilder& builder
) const
{
   impl_->encode(input, builder);
}

void SemanticHGraphEncoderEngine::encode_views(
   const canonical::FlatRelationViewInput& input,
   BatchBuilder& builder
) const
{
   impl_->encode(canonical::materialize_semantic_flat_view_input(input), builder);
}

void SemanticHGraphEncoderEngine::encode_successor(
   const SemanticFlatRelationInput& current,
   const SemanticFlatRelationInput& successor,
   bool delta_mode,
   std::string_view successor_suffix,
   bool include_successor_goal_satisfaction,
   BatchBuilder& builder
) const
{
   impl_->encode_successor(
      current, successor, delta_mode, successor_suffix, include_successor_goal_satisfaction, builder
   );
}

void SemanticHGraphEncoderEngine::configure_horizon(
   const SemanticHorizonHGraphEncoderConfig& config
)
{
   impl_->configure_horizon(config);
}

void SemanticHGraphEncoderEngine::encode_horizon(
   const SemanticTransitionDAG& dag,
   const SemanticHorizonHGraphEncoderConfig& config,
   BatchBuilder& builder
) const
{
   impl_->encode_horizon(dag, config, builder);
}

BatchBuilder::BatchEncoding SemanticHGraphEncoderEngine::encode_batch(
   const std::vector< SemanticFlatRelationInput >& inputs
) const
{
   BatchBuilder builder;
   builder.set_graph_kind("hetero");
   for(const auto& input : inputs) {
      impl_->encode(input, builder);
      builder.next_graph();
   }
   return builder.build();
}

const std::shared_ptr< const SemanticTaskContext >&
SemanticHGraphEncoderEngine::get_task_context() const
{
   return impl_->task_context;
}

const SemanticHGraphEncoderEngine::Config& SemanticHGraphEncoderEngine::get_config() const
{
   return impl_->config;
}

const std::vector< SemanticPredicateSpec >& SemanticHGraphEncoderEngine::get_predicates() const
{
   return impl_->predicates;
}

const std::vector< SemanticActionSpec >& SemanticHGraphEncoderEngine::get_actions() const
{
   return impl_->actions;
}

const std::map< std::string, int >& SemanticHGraphEncoderEngine::get_relation_arities() const
{
   return impl_->relation_arities;
}

void SemanticHGraphEncoderEngine::update_relations(std::map< std::string, int > relation_arities)
{
   for(const auto& [name, arity] : relation_arities) {
      if(name.empty()) {
         throw std::invalid_argument("relation name must not be empty");
      }
      if(arity < 0) {
         throw std::invalid_argument("relation arity must be non-negative");
      }
   }
   impl_->relation_arities = std::move(relation_arities);
   impl_->all_edge_types.clear();
   for(const auto& [type, arity] : impl_->relation_arities) {
      const int effective = impl_->config.add_nullary_predicates and arity == 0 ? 1 : arity;
      for(int position = 0; position < effective; ++position) {
         const auto pos = std::to_string(position);
         impl_->all_edge_types.emplace_back(impl_->config.symbol_type_id, pos, type);
         impl_->all_edge_types.emplace_back(type, pos, impl_->config.symbol_type_id);
      }
      if(impl_->config.include_lgan_edges) {
         impl_->all_edge_types.emplace_back(
            type, impl_->config.lgan_tn_edge_pos, impl_->config.symbol_type_id
         );
         impl_->all_edge_types.emplace_back(
            type, impl_->config.lgan_nn_edge_pos, impl_->config.symbol_type_id
         );
      }
   }
   std::ranges::sort(impl_->all_edge_types);
   impl_->all_edge_types.erase(
      std::ranges::unique(impl_->all_edge_types).begin(), impl_->all_edge_types.end()
   );
}

}  // namespace mifrost
