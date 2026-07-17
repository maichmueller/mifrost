/**
 * @file semantic_flat_relation_encoder.cpp
 * @brief Native encoding of owned, planning-backend-neutral flat graph inputs.
 */
#include "semantic_flat_relation_encoder.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <functional>
#include <map>
#include <numeric>
#include <ranges>
#include <set>
#include <span>
#include <stdexcept>
#include <string_view>
#include <tuple>
#include <utility>

#include "flat_encoder_common.hpp"
#include "flat_lgan.hpp"
#include "flat_relation_schema.hpp"
#include "flat_tuple_layout.hpp"
#include "mifrost/core/encoders/common/target_metadata.hpp"
#include "mifrost/core/encoders/flat/semantic_flat_horizon_encoder.hpp"
#include "mifrost/core/semantic/semantic_transition_dag.hpp"

namespace mifrost {

namespace {

const std::shared_ptr< const SemanticTaskContext >& require_task_context(
   const std::shared_ptr< const SemanticTaskContext >& task_context,
   std::string_view encoder_name
)
{
   if(not task_context) {
      throw std::invalid_argument(std::string(encoder_name) + " task context must not be null");
   }
   return task_context;
}

constexpr std::array< SemanticPredicateCategory, 3 > kCategoryOrder = {
   SemanticPredicateCategory::static_predicate,
   SemanticPredicateCategory::fluent,
   SemanticPredicateCategory::derived,
};

constexpr std::array< std::string_view, 4 > kGoalLevelSuffixes = {
   "[g]",
   "[sg]",
   "[ssg]",
   "[sssg]",
};

const std::set< std::string, std::less<> > kTopTypePredicates = {
   "object",
   "number",
   "symbol",
   "_action_",
};

std::string goal_relation_name(
   std::string_view predicate,
   bool positive,
   std::optional< size_t > level,
   std::optional< GoalDerivation > derivation,
   std::string_view trailing_suffix = ""
)
{
   std::string result = positive ? "[+]" : "[-]";
   result += predicate;
   result += trailing_suffix;
   if(level.has_value()) {
      result += kGoalLevelSuffixes.at(*level);
   }
   if(derivation.has_value()) {
      switch(*derivation) {
         case GoalDerivation::plain: break;
         case GoalDerivation::satisfied: result += "[sat]"; break;
         case GoalDerivation::unsatisfied: result += "[unsat]"; break;
         case GoalDerivation::added_satisfied: result += "[sat+]"; break;
         case GoalDerivation::added_unsatisfied: result += "[sat-]"; break;
      }
   }
   return result;
}

bool supports_semantic_goal_derivation(GoalDerivation derivation)
{
   return derivation == GoalDerivation::plain or derivation == GoalDerivation::satisfied
          or derivation == GoalDerivation::unsatisfied;
}

bool has_target_source(const FlatRelationEncoderConfig& config, TargetSource source)
{
   return config.target_sources.contains(source);
}

bool has_lgan_anchor_source(const FlatRelationEncoderConfig& config, TargetSource source)
{
   return config.include_lgan_edges and config.lgan_anchor_sources.contains(source);
}

bool has_anchor_entity_source(const FlatRelationEncoderConfig& config, TargetSource source)
{
   return has_target_source(config, source) or has_lgan_anchor_source(config, source);
}

std::optional< TargetSource >
source_for_goal_level(const FlatRelationEncoderConfig& config, size_t level)
{
   const auto source = level > 0 ? TargetSource::subgoals : TargetSource::goals;
   return has_anchor_entity_source(config, source) ? std::optional(source) : std::nullopt;
}

FlatTupleLayout
semantic_goal_layout(const FlatRelationEncoderConfig& config, int logical_arity, size_t level)
{
   std::vector< FlatSlotRole > roles;
   if(const auto source = source_for_goal_level(config, level); source.has_value()) {
      roles.push_back(slot_role_for_target_source(*source));
   }
   return make_predicate_tuple_layout(
      logical_arity, std::span{roles}, config.use_predicate_virtual_nodes
   );
}

FlatTupleLayout semantic_history_layout(const FlatRelationEncoderConfig& config, int logical_arity)
{
   std::vector< FlatSlotRole > roles;
   if(has_anchor_entity_source(config, TargetSource::history)) {
      roles.push_back(FlatSlotRole::history_target_slot);
   }
   roles.push_back(FlatSlotRole::history_slot);
   return make_predicate_tuple_layout(
      logical_arity, std::span{roles}, config.use_predicate_virtual_nodes
   );
}

std::string atom_display_name(
   const SemanticAtom& atom,
   const std::vector< SemanticPredicateSpec >& predicates,
   const std::vector< std::string >& objects
)
{
   std::string result = "(" + predicates.at(static_cast< size_t >(atom.predicate)).name;
   for(const int64_t object : atom.arguments) {
      result += " ";
      result += objects.at(static_cast< size_t >(object));
   }
   result += ")";
   return result;
}

std::string goal_display_name(
   const SemanticLiteral& literal,
   size_t level,
   const std::vector< SemanticPredicateSpec >& predicates,
   const std::vector< std::string >& objects
)
{
   std::string result = literal.positive ? "[+]" : "[-]";
   result += atom_display_name(literal.atom, predicates, objects);
   result += kGoalLevelSuffixes.at(level);
   return result;
}

std::string action_display_name(
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

void validate_name(std::string_view name, std::string_view kind)
{
   if(name.empty()) {
      throw std::invalid_argument(std::string(kind) + " name must not be empty");
   }
}

template < typename Spec >
void validate_unique_names(const std::vector< Spec >& specs, std::string_view kind)
{
   std::set< std::string, std::less<> > names;
   for(const auto& spec : specs) {
      validate_name(spec.name, kind);
      if(spec.arity < 0) {
         throw std::invalid_argument(std::string(kind) + " arity must be non-negative");
      }
      if(not names.emplace(spec.name).second) {
         throw std::invalid_argument(
            "Semantic flat encoder requires unique " + std::string(kind) + " names"
         );
      }
   }
}

struct GoalEntityKey {
   TargetSource source = TargetSource::goals;
   SemanticLiteral literal;
   size_t level = 0;

   auto operator<=>(const GoalEntityKey&) const = default;
};

inline void mix_semantic_hash(size_t& value, int64_t part)
{
   value ^= std::hash< int64_t >{}(part) + 0x9e3779b97f4a7c15ULL + (value << 6U) + (value >> 2U);
}

struct SemanticAtomHash {
   size_t operator()(const SemanticAtom& atom) const noexcept
   {
      size_t value = 0;
      mix_semantic_hash(value, atom.predicate);
      for(const auto argument : atom.arguments) {
         mix_semantic_hash(value, argument);
      }
      return value;
   }
};

struct SemanticLiteralHash {
   size_t operator()(const SemanticLiteral& literal) const noexcept
   {
      size_t value = SemanticAtomHash{}(literal.atom);
      mix_semantic_hash(value, literal.positive ? 1 : 0);
      return value;
   }
};

struct SemanticGroundActionHash {
   size_t operator()(const SemanticGroundAction& action) const noexcept
   {
      size_t value = 0;
      mix_semantic_hash(value, action.action);
      for(const auto argument : action.arguments) {
         mix_semantic_hash(value, argument);
      }
      return value;
   }
};

struct GoalEntityKeyHash {
   size_t operator()(const GoalEntityKey& key) const noexcept
   {
      size_t value = SemanticLiteralHash{}(key.literal);
      mix_semantic_hash(value, static_cast< int64_t >(key.source));
      mix_semantic_hash(value, static_cast< int64_t >(key.level));
      return value;
   }
};

struct HistoryEntityKey {
   int64_t dt = 0;
   size_t entry_index = 0;
   SemanticLiteral literal;

   auto operator<=>(const HistoryEntityKey&) const = default;
};

struct HistoryEntityKeyHash {
   size_t operator()(const HistoryEntityKey& key) const noexcept
   {
      size_t value = SemanticLiteralHash{}(key.literal);
      mix_semantic_hash(value, key.dt);
      mix_semantic_hash(value, static_cast< int64_t >(key.entry_index));
      return value;
   }
};

struct PreparedHistoryEntry {
   int64_t dt = 0;
   size_t entry_index = 0;
   std::vector< SemanticLiteral > literals;
   int64_t entity_index = -1;
};

struct SemanticEncodingContext {
   int64_t entity_count = 0;
   std::vector< std::string > entity_names;
   std::vector< int64_t > entity_role_ids;
   std::vector< int64_t > object_indices;
   std::vector< int64_t > predicate_entity_indices;
   std::vector< int64_t > state_entity_indices;
   std::vector< int64_t > history_entity_indices;
   std::vector< int64_t > history_entity_dt;
   std::vector< int64_t > target_entity_indices;
   std::vector< int64_t > target_entity_group_ids;
   hash_map< GoalEntityKey, int64_t, GoalEntityKeyHash > goal_entity_indices;
   hash_map< SemanticGroundAction, int64_t, SemanticGroundActionHash > action_entity_indices;
   hash_map< HistoryEntityKey, int64_t, HistoryEntityKeyHash > history_target_entity_indices;
   std::vector< SemanticGroundAction > unique_actions;
   std::vector< PreparedHistoryEntry > history_entries;
   TargetColumns target_columns;
};

constexpr std::string_view kCandidateRelationSuffix = "[state]";

std::string state_anchored_relation_name(std::string_view name)
{
   return std::string(name) + std::string(kCandidateRelationSuffix);
}

bool split_full_state_relations(const SemanticFlatHorizonEncoderConfig& config)
{
   return config.transition_mode == SemanticHorizonMode::full
          and root_uses_split_state_relations(config.root_policy);
}

}  // namespace

struct SemanticFlatRelationEncoderEngine::Impl {
   struct GoalRelationIds {
      std::array< std::array< int, 3 >, 2 > by_polarity = {
         std::array< int, 3 >{-1, -1, -1},
         std::array< int, 3 >{-1, -1, -1},
      };
   };
   struct HorizonGoalRelationIds {
      std::array< std::array< int, 5 >, 2 > root = {
         std::array< int, 5 >{-1, -1, -1, -1, -1},
         std::array< int, 5 >{-1, -1, -1, -1, -1},
      };
      std::array< std::array< int, 5 >, 2 > candidate = {
         std::array< int, 5 >{-1, -1, -1, -1, -1},
         std::array< int, 5 >{-1, -1, -1, -1, -1},
      };
   };

   Config config;
   std::shared_ptr< const SemanticTaskContext > task_context;
   const std::vector< SemanticPredicateSpec >& predicates;
   const std::vector< SemanticActionSpec >& actions;
   FlatRelationSchemaMetadata metadata;
   std::vector< std::string > target_entity_group_names;
   std::map< TargetSource, int64_t > target_entity_group_ids;
   std::vector< std::string > target_group_names;
   std::map< TargetSource, int64_t > target_group_ids;
   std::vector< int > state_relation_ids;
   std::vector< int > action_relation_ids;
   std::vector< std::array< int, 2 > > history_relation_ids;
   std::vector< std::vector< GoalRelationIds > > goal_relation_ids;
   std::vector< int > horizon_state_relation_ids;
   std::vector< int > horizon_state_anchored_relation_ids;
   std::vector< int > horizon_action_relation_ids;
   std::vector< std::array< int, 2 > > horizon_literal_relation_ids;
   std::vector< std::vector< HorizonGoalRelationIds > > horizon_goal_relation_ids;
   int horizon_parent_relation_id = -1;
   int horizon_sibling_relation_id = -1;
   int horizon_cousin_relation_id = -1;

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
       : config(std::move(encoder_config)),
         task_context(require_task_context(context, "Semantic flat")),
         predicates(task_context->predicates),
         actions(task_context->actions)
   {
      validate_config();
      validate_unique_names(predicates, "predicate");
      validate_unique_names(actions, "action");
      build_groups();
      build_schema();
      build_relation_ids();
   }

   static size_t goal_derivation_index(std::optional< GoalDerivation > derivation)
   {
      if(not derivation.has_value() or *derivation == GoalDerivation::plain) {
         return 0;
      }
      if(*derivation == GoalDerivation::satisfied) {
         return 1;
      }
      if(*derivation == GoalDerivation::unsatisfied) {
         return 2;
      }
      throw std::invalid_argument("unsupported semantic Flat goal derivation");
   }

   static size_t horizon_goal_derivation_index(std::optional< GoalDerivation > derivation)
   {
      if(not derivation.has_value() or *derivation == GoalDerivation::plain) {
         return 0;
      }
      switch(*derivation) {
         case GoalDerivation::satisfied: return 1;
         case GoalDerivation::unsatisfied: return 2;
         case GoalDerivation::added_satisfied: return 3;
         case GoalDerivation::added_unsatisfied: return 4;
         case GoalDerivation::plain: break;
      }
      throw std::invalid_argument("unsupported semantic Flat Horizon goal derivation");
   }

   int relation_id_at_construction(std::string_view name) const
   {
      const auto it = metadata.relation_name_to_id.find(std::string(name));
      if(it == metadata.relation_name_to_id.end()) {
         throw std::invalid_argument("missing precomputed semantic Flat relation");
      }
      return it->second;
   }

   int optional_relation_id_at_construction(std::string_view name) const
   {
      const auto it = metadata.relation_name_to_id.find(std::string(name));
      return it == metadata.relation_name_to_id.end() ? -1 : it->second;
   }

   int required_relation_id(int id) const
   {
      if(id < 0) {
         throw std::invalid_argument("semantic Flat input requires a disabled relation");
      }
      return id;
   }

   void build_relation_ids()
   {
      state_relation_ids.assign(predicates.size(), -1);
      history_relation_ids.assign(predicates.size(), std::array< int, 2 >{-1, -1});
      goal_relation_ids.assign(
         predicates.size(), std::vector< GoalRelationIds >(config.max_goal_level + 1)
      );
      for(size_t predicate_index = 0; predicate_index < predicates.size(); ++predicate_index) {
         const auto& predicate = predicates[predicate_index];
         if(config.ignore_zero_arity_relations and predicate.arity == 0) {
            continue;
         }
         state_relation_ids[predicate_index] = relation_id_at_construction(predicate.name);
         for(const bool positive : {false, true}) {
            const auto polarity_index = positive ? size_t{1} : size_t{0};
            history_relation_ids[predicate_index][polarity_index] = relation_id_at_construction(
               goal_relation_name(predicate.name, positive, std::nullopt, std::nullopt, "[hist]")
            );
            if(kTopTypePredicates.contains(predicate.name)) {
               continue;
            }
            for(size_t level = 0; level <= config.max_goal_level; ++level) {
               for(const auto derivation : {
                      std::optional< GoalDerivation >{},
                      std::optional< GoalDerivation >{GoalDerivation::satisfied},
                      std::optional< GoalDerivation >{GoalDerivation::unsatisfied},
                   }) {
                  const auto slot = goal_derivation_index(derivation);
                  const auto name = goal_relation_name(predicate.name, positive, level, derivation);
                  const auto it = metadata.relation_name_to_id.find(name);
                  if(it != metadata.relation_name_to_id.end()) {
                     goal_relation_ids[predicate_index][level]
                        .by_polarity[polarity_index][slot] = it->second;
                  }
               }
            }
         }
      }
      action_relation_ids.resize(actions.size());
      for(size_t action_index = 0; action_index < actions.size(); ++action_index) {
         action_relation_ids[action_index] = relation_id_at_construction(
            actions[action_index].name
         );
      }
   }

   void validate_config() const
   {
      if(config.max_goal_level >= kGoalLevelSuffixes.size()) {
         throw std::invalid_argument("Semantic flat max_goal_level must be in [0, 3]");
      }
      auto validate_sources = [](const std::set< TargetSource >& sources, std::string_view field) {
         for(const auto source : sources) {
            if(source == TargetSource::goals or source == TargetSource::subgoals
               or source == TargetSource::actions or source == TargetSource::history) {
               continue;
            }
            throw std::invalid_argument(
               "Semantic flat " + std::string(field)
               + " supports action, goal, subgoal, and history only"
            );
         }
      };
      validate_sources(config.target_sources, "target_sources");
      validate_sources(config.lgan_anchor_sources, "lgan_anchor_sources");
      for(const auto derivation : config.goal_derivations) {
         if(not supports_semantic_goal_derivation(derivation)) {
            throw std::invalid_argument(
               "Semantic flat encoder supports plain/satisfied/unsatisfied goal derivations only"
            );
         }
      }
   }

   void build_groups()
   {
      auto append = [](std::vector< std::string >& names,
                       std::map< TargetSource, int64_t >& ids,
                       TargetSource source) {
         ids.emplace(source, static_cast< int64_t >(names.size()));
         names.emplace_back(target_source_group_name(source));
      };
      for(const auto source : kCanonicalTargetSourceOrder) {
         if(source == TargetSource::states) {
            continue;
         }
         if(source == TargetSource::actions or has_anchor_entity_source(config, source)) {
            append(target_entity_group_names, target_entity_group_ids, source);
         }
         if(has_target_source(config, source)) {
            append(target_group_names, target_group_ids, source);
         }
      }
   }

   void build_schema()
   {
      FlatRelationSchemaRegistry registry;
      for(const auto& predicate : predicates) {
         const auto arity = static_cast< int >(predicate.arity);
         if(config.ignore_zero_arity_relations and arity == 0) {
            continue;
         }
         registry.add(
            predicate.name,
            make_predicate_tuple_layout(arity, {}, config.use_predicate_virtual_nodes),
            "state"
         );
         if(not kTopTypePredicates.contains(predicate.name)) {
            if(config.goal_derivations.contains(GoalDerivation::plain)) {
               for(size_t level = 0; level <= config.max_goal_level; ++level) {
                  for(const bool positive : {true, false}) {
                     registry.add(
                        goal_relation_name(predicate.name, positive, level, std::nullopt),
                        semantic_goal_layout(config, arity, level),
                        "goal"
                     );
                  }
               }
               if(config.support_literals) {
                  for(const bool positive : {true, false}) {
                     std::vector< FlatSlotRole > roles;
                     if(has_anchor_entity_source(config, TargetSource::goals)) {
                        roles.push_back(FlatSlotRole::goal_target_slot);
                     }
                     registry.add(
                        goal_relation_name(predicate.name, positive, std::nullopt, std::nullopt),
                        make_predicate_tuple_layout(
                           arity, std::span{roles}, config.use_predicate_virtual_nodes
                        ),
                        "goal"
                     );
                  }
               }
            }
            for(const auto derivation : config.goal_derivations) {
               if(derivation == GoalDerivation::plain) {
                  continue;
               }
               for(size_t level = 0; level <= config.max_goal_level; ++level) {
                  for(const bool positive : {true, false}) {
                     registry.add(
                        goal_relation_name(predicate.name, positive, level, derivation),
                        make_predicate_tuple_layout(arity, {}, config.use_predicate_virtual_nodes),
                        "goal_derivation"
                     );
                  }
               }
               if(config.support_literals) {
                  for(const bool positive : {true, false}) {
                     registry.add(
                        goal_relation_name(predicate.name, positive, std::nullopt, derivation),
                        make_predicate_tuple_layout(arity, {}, config.use_predicate_virtual_nodes),
                        "goal_derivation"
                     );
                  }
               }
            }
         }
         for(const bool positive : {true, false}) {
            registry.add(
               goal_relation_name(predicate.name, positive, std::nullopt, std::nullopt, "[hist]"),
               semantic_history_layout(config, arity),
               "history"
            );
         }
      }
      for(const auto& action : actions) {
         registry.add(
            action.name,
            make_nonpredicate_tuple_layout(
               static_cast< int >(action.arity), {FlatSlotRole::action_slot}
            ),
            "action"
         );
      }
      metadata = build_flat_relation_schema_metadata(
         registry,
         static_cast< int >(config.max_goal_level),
         config.support_literals,
         config.goal_derivations,
         "SemanticFlatRelationEncoderEngine requires at least one relation"
      );
   }

   void configure_horizon(const SemanticFlatHorizonEncoderConfig& horizon)
   {
      if(horizon.max_goal_level >= kGoalLevelSuffixes.size()) {
         throw std::invalid_argument("Semantic flat Horizon max_goal_level must be in [0, 3]");
      }
      if(horizon.transition_mode == SemanticHorizonMode::action and horizon.ignore_actions) {
         throw std::invalid_argument("Action flat horizon encoding requires ignore_actions=false.");
      }

      FlatRelationSchemaRegistry registry;
      const bool root_state_slot = root_in_state_relations(horizon.root_policy);
      const bool split_candidates = split_full_state_relations(horizon);
      auto predicate_layout = [&](int arity, bool state_slot) {
         const std::array roles = {FlatSlotRole::state_slot};
         return make_predicate_tuple_layout(
            arity,
            state_slot ? std::span{roles} : std::span< const FlatSlotRole >{},
            horizon.use_predicate_virtual_nodes
         );
      };
      auto add_root = [&](std::string name, int arity, std::string source) {
         registry.add_or_validate(
            std::move(name), predicate_layout(arity, root_state_slot), std::move(source)
         );
      };
      auto add_full = [&](const std::string& name, int arity, const std::string& source) {
         add_root(name, arity, source);
         if(split_candidates) {
            registry.add_or_validate(
               state_anchored_relation_name(name), predicate_layout(arity, true), source
            );
         }
      };
      auto add_candidate = [&](std::string name, int arity, std::string source) {
         registry.add_or_validate(
            std::move(name), predicate_layout(arity, true), std::move(source)
         );
      };

      for(const auto& predicate : predicates) {
         const int arity = static_cast< int >(predicate.arity);
         add_full(predicate.name, arity, "state");
         if(kTopTypePredicates.contains(predicate.name)) {
            continue;
         }
         if(horizon.goal_derivations.contains(GoalDerivation::plain)) {
            for(size_t level = 0; level <= horizon.max_goal_level; ++level) {
               for(const bool positive : {true, false}) {
                  add_root(
                     goal_relation_name(predicate.name, positive, level, std::nullopt),
                     arity,
                     "goal"
                  );
               }
            }
         }
         if(horizon.support_literals) {
            for(const bool positive : {true, false}) {
               auto name = goal_relation_name(predicate.name, positive, std::nullopt, std::nullopt);
               if(horizon.transition_mode == SemanticHorizonMode::delta) {
                  add_candidate(std::move(name), arity, "state");
               } else {
                  add_root(std::move(name), arity, "state");
               }
            }
         }
         for(const auto derivation : horizon.goal_derivations) {
            if(derivation == GoalDerivation::plain) {
               continue;
            }
            const bool root_derivation = derivation == GoalDerivation::satisfied
                                         or derivation == GoalDerivation::unsatisfied;
            const bool delta_derivation = derivation == GoalDerivation::added_satisfied
                                          or derivation == GoalDerivation::added_unsatisfied;
            if(root_derivation) {
               for(size_t level = 0; level <= horizon.max_goal_level; ++level) {
                  for(const bool positive : {true, false}) {
                     add_root(
                        goal_relation_name(predicate.name, positive, level, derivation),
                        arity,
                        "goal_satisfaction"
                     );
                  }
               }
               if(horizon.support_literals) {
                  for(const bool positive : {true, false}) {
                     add_root(
                        goal_relation_name(predicate.name, positive, std::nullopt, derivation),
                        arity,
                        "goal_satisfaction"
                     );
                  }
               }
               if(horizon.transition_mode == SemanticHorizonMode::full and split_candidates) {
                  for(size_t level = 0; level <= horizon.max_goal_level; ++level) {
                     for(const bool positive : {true, false}) {
                        registry.add_or_validate(
                           state_anchored_relation_name(
                              goal_relation_name(predicate.name, positive, level, derivation)
                           ),
                           predicate_layout(arity, true),
                           "goal_satisfaction"
                        );
                     }
                  }
                  if(horizon.support_literals) {
                     for(const bool positive : {true, false}) {
                        registry.add_or_validate(
                           state_anchored_relation_name(
                              goal_relation_name(predicate.name, positive, std::nullopt, derivation)
                           ),
                           predicate_layout(arity, true),
                           "goal_satisfaction"
                        );
                     }
                  }
               }
            }
            if(horizon.transition_mode == SemanticHorizonMode::delta and delta_derivation) {
               for(size_t level = 0; level <= horizon.max_goal_level; ++level) {
                  for(const bool positive : {true, false}) {
                     add_candidate(
                        goal_relation_name(predicate.name, positive, level, derivation),
                        arity,
                        "goal_satisfaction"
                     );
                  }
               }
               if(horizon.support_literals) {
                  for(const bool positive : {true, false}) {
                     add_candidate(
                        goal_relation_name(predicate.name, positive, std::nullopt, derivation),
                        arity,
                        "goal_satisfaction"
                     );
                  }
               }
            }
         }
      }
      if(not horizon.ignore_actions) {
         for(const auto& action : actions) {
            registry.add_or_validate(
               action.name,
               make_nonpredicate_tuple_layout(
                  static_cast< int >(action.arity), {FlatSlotRole::state_slot}
               ),
               "action"
            );
         }
      }
      const auto topology_layout = make_nonpredicate_tuple_layout(
         0, {FlatSlotRole::state_slot, FlatSlotRole::state_slot}
      );
      if(horizon.enable_parent_relation) {
         registry.add_or_validate(horizon.parent_relation, topology_layout, "parent");
      }
      if(horizon.enable_sibling_relation) {
         registry.add_or_validate(horizon.sibling_relation, topology_layout, "sibling");
      }
      if(horizon.enable_cousin_relation) {
         registry.add_or_validate(horizon.cousin_relation, topology_layout, "cousin");
      }
      metadata = build_flat_relation_schema_metadata(
         registry,
         static_cast< int >(horizon.max_goal_level),
         horizon.support_literals,
         horizon.goal_derivations,
         "SemanticFlatHorizonEncoderEngine requires at least one relation"
      );
      build_horizon_relation_ids(horizon);
   }

   void build_horizon_relation_ids(const SemanticFlatHorizonEncoderConfig& horizon)
   {
      const bool split_candidates = split_full_state_relations(horizon);
      horizon_state_relation_ids.assign(predicates.size(), -1);
      horizon_state_anchored_relation_ids.assign(predicates.size(), -1);
      horizon_literal_relation_ids.assign(predicates.size(), std::array< int, 2 >{-1, -1});
      horizon_goal_relation_ids.assign(
         predicates.size(), std::vector< HorizonGoalRelationIds >(horizon.max_goal_level + 1)
      );

      for(size_t predicate_index = 0; predicate_index < predicates.size(); ++predicate_index) {
         const auto& predicate = predicates[predicate_index];
         const int state_id = relation_id_at_construction(predicate.name);
         horizon_state_relation_ids[predicate_index] = state_id;
         horizon_state_anchored_relation_ids
            [predicate_index] = split_candidates ? relation_id_at_construction(
                                                      state_anchored_relation_name(predicate.name)
                                                   )
                                                 : state_id;
         if(kTopTypePredicates.contains(predicate.name)) {
            continue;
         }
         for(const bool positive : {false, true}) {
            const auto polarity = positive ? size_t{1} : size_t{0};
            if(horizon.support_literals) {
               horizon_literal_relation_ids
                  [predicate_index][polarity] = optional_relation_id_at_construction(
                     goal_relation_name(predicate.name, positive, std::nullopt, std::nullopt)
                  );
            }
            for(size_t level = 0; level <= horizon.max_goal_level; ++level) {
               auto& ids = horizon_goal_relation_ids[predicate_index][level];
               for(const auto derivation : {
                      std::optional< GoalDerivation >{},
                      std::optional< GoalDerivation >{GoalDerivation::satisfied},
                      std::optional< GoalDerivation >{GoalDerivation::unsatisfied},
                      std::optional< GoalDerivation >{GoalDerivation::added_satisfied},
                      std::optional< GoalDerivation >{GoalDerivation::added_unsatisfied},
                   }) {
                  const auto slot = horizon_goal_derivation_index(derivation);
                  const auto relation = goal_relation_name(
                     predicate.name, positive, level, derivation
                  );
                  const int root = optional_relation_id_at_construction(relation);
                  ids.root[polarity][slot] = root;
                  ids.candidate[polarity][slot] = root;
                  if(split_candidates
                     and (derivation == GoalDerivation::satisfied or derivation == GoalDerivation::unsatisfied)) {
                     ids.candidate[polarity][slot] = optional_relation_id_at_construction(
                        state_anchored_relation_name(relation)
                     );
                  }
               }
            }
         }
      }

      horizon_action_relation_ids.assign(actions.size(), -1);
      if(not horizon.ignore_actions) {
         for(size_t action_index = 0; action_index < actions.size(); ++action_index) {
            horizon_action_relation_ids[action_index] = relation_id_at_construction(
               actions[action_index].name
            );
         }
      }
      horizon_parent_relation_id = horizon.enable_parent_relation
                                      ? relation_id_at_construction(horizon.parent_relation)
                                      : -1;
      horizon_sibling_relation_id = horizon.enable_sibling_relation
                                       ? relation_id_at_construction(horizon.sibling_relation)
                                       : -1;
      horizon_cousin_relation_id = horizon.enable_cousin_relation
                                      ? relation_id_at_construction(horizon.cousin_relation)
                                      : -1;
   }

   void prepare_horizon_builder(
      BatchBuilder& builder,
      const SemanticFlatHorizonEncoderConfig& horizon
   ) const
   {
      const std::vector< std::string > groups = {
         std::string(target_source_group_name(TargetSource::states))
      };
      set_flat_graph_attrs(
         builder,
         metadata,
         FlatBuilderGraphConfig{
            .include_lgan_edges = horizon.include_lgan_edges,
            .use_predicate_virtual_nodes = horizon.use_predicate_virtual_nodes,
            .target_symbol_prefix = horizon.target_symbol_prefix,
            .target_entity_group_names = groups,
            .lgan_tn_edge_pos = horizon.lgan_tn_edge_pos,
            .lgan_nn_edge_pos = horizon.lgan_nn_edge_pos,
            .lgan_rr_edge_pos = horizon.lgan_rr_edge_pos,
            .pack_relation_args_relation_major = horizon.pack_relation_args_relation_major,
         }
      );
      register_flat_entity_fields(builder);
      register_flat_target_entity_fields(builder);
      builder.register_field(
         std::string(kTargetSizesField),
         GraphFieldSpec{.dtype = GraphFieldDType::I64, .mode = GraphFieldMode::STACK, .dim = 1}
      );
      const TargetMetadataEmitConfig target_config{
         .position_node_type_id = std::string(kFlatEntityNodeType),
         .symbol_prefix = horizon.target_symbol_prefix,
         .include_depth = true,
         .include_group = true,
         .include_names = false,
         .groups = groups,
         .parent_relation = horizon.parent_relation,
      };
      register_target_fields(builder, target_config);
      builder.set_graph_attr(std::string(kTargetGroupsAttr), groups);
      builder.set_graph_attr(std::string(kTargetSymbolPrefixAttr), horizon.target_symbol_prefix);
      builder.set_graph_attr(std::string(kParentRelationAttr), horizon.parent_relation);
      register_flat_relation_instance_fields(
         builder, static_cast< int >(metadata.relation_names.size())
      );
      if(horizon.include_lgan_edges) {
         register_flat_lgan_fields(builder);
      }
   }

   void prepare_builder(BatchBuilder& builder) const
   {
      set_flat_graph_attrs(
         builder,
         metadata,
         FlatBuilderGraphConfig{
            .include_lgan_edges = config.include_lgan_edges,
            .use_predicate_virtual_nodes = config.use_predicate_virtual_nodes,
            .target_sources = source_names_for(config.target_sources),
            .lgan_anchor_sources = source_names_for(config.lgan_anchor_sources),
            .target_symbol_prefix = config.target_symbol_prefix,
            .target_entity_group_names = target_entity_group_names,
            .lgan_tn_edge_pos = config.lgan_tn_edge_pos,
            .lgan_nn_edge_pos = config.lgan_nn_edge_pos,
            .lgan_rr_edge_pos = config.lgan_rr_edge_pos,
            .pack_relation_args_relation_major = config.pack_relation_args_relation_major,
         }
      );
      register_flat_entity_fields(builder);
      register_flat_history_entity_fields(builder);
      register_flat_target_entity_fields(builder);
      if(not target_group_names.empty()) {
         builder.register_field(
            std::string(kTargetSizesField),
            GraphFieldSpec{
               .dtype = GraphFieldDType::I64,
               .mode = GraphFieldMode::STACK,
               .dim = 1,
            }
         );
         register_target_fields(
            builder,
            TargetMetadataEmitConfig{
               .position_node_type_id = std::string(kFlatEntityNodeType),
               .symbol_prefix = config.target_symbol_prefix,
               .include_depth = false,
               .include_group = true,
               .include_names = false,
               .groups = target_group_names,
               .parent_relation = std::nullopt,
            }
         );
         builder.set_graph_attr(std::string(kTargetGroupsAttr), target_group_names);
         builder.set_graph_attr(std::string(kTargetSymbolPrefixAttr), config.target_symbol_prefix);
      }
      register_flat_relation_instance_fields(
         builder, static_cast< int >(metadata.relation_names.size())
      );
      if(config.include_lgan_edges) {
         register_flat_lgan_fields(builder);
      }
   }

   void validate_atom(const SemanticAtom& atom, size_t object_count, std::string_view lane) const
   {
      if(atom.predicate < 0 or static_cast< size_t >(atom.predicate) >= predicates.size()) {
         throw std::invalid_argument(
            "Semantic flat " + std::string(lane) + " predicate out of range"
         );
      }
      const auto& predicate = predicates[static_cast< size_t >(atom.predicate)];
      if(atom.arguments.size() != static_cast< size_t >(predicate.arity)) {
         throw std::invalid_argument(
            "Semantic flat " + std::string(lane) + " arity does not match predicate schema"
         );
      }
      for(const int64_t object : atom.arguments) {
         if(object < 0 or static_cast< size_t >(object) >= object_count) {
            throw std::invalid_argument(
               "Semantic flat " + std::string(lane) + " object index out of range"
            );
         }
      }
   }

   void validate_input(const SemanticFlatRelationInput& input) const
   {
      const auto& objects = semantic_objects(input);
      const auto& goals = semantic_goals(input);
      const auto& static_facts = semantic_static_facts(input);
      std::set< std::string, std::less<> > object_names;
      for(const auto& object : objects) {
         validate_name(object, "object");
         if(not object_names.emplace(object).second) {
            throw std::invalid_argument("Semantic flat object names must be unique");
         }
      }
      for(const auto& fact : input.state_facts) {
         validate_atom(fact, objects.size(), "state fact");
      }
      for(const auto& fact : static_facts) {
         validate_atom(fact, objects.size(), "static fact");
      }
      for(const auto& goal : goals) {
         validate_atom(goal.atom, objects.size(), "goal");
      }
      if(input.subgoal_layers.size() > config.max_goal_level) {
         throw std::invalid_argument("Semantic flat subgoal layer count exceeds max_goal_level");
      }
      for(const auto& layer : input.subgoal_layers) {
         for(const auto& goal : layer) {
            validate_atom(goal.atom, objects.size(), "subgoal");
         }
      }
      for(const auto& action : input.actions) {
         if(action.action < 0 or static_cast< size_t >(action.action) >= actions.size()) {
            throw std::invalid_argument("Semantic flat action schema index out of range");
         }
         if(action.arguments.size()
            != static_cast< size_t >(actions[static_cast< size_t >(action.action)].arity)) {
            throw std::invalid_argument("Semantic flat ground action arity mismatch");
         }
         for(const int64_t object : action.arguments) {
            if(object < 0 or static_cast< size_t >(object) >= objects.size()) {
               throw std::invalid_argument("Semantic flat action object index out of range");
            }
         }
      }
      for(const auto& entry : input.history) {
         if(entry.dt >= 0) {
            throw std::invalid_argument("Semantic flat history requires negative dt values");
         }
         for(const auto& literal : entry.literals) {
            validate_atom(literal.atom, objects.size(), "history literal");
         }
      }
   }

   int64_t ensure_predicate_entity(SemanticEncodingContext& context, int64_t predicate) const
   {
      if(predicate < 0 or static_cast< size_t >(predicate) >= predicates.size()) {
         throw std::invalid_argument("semantic Flat predicate entity index out of range");
      }
      auto& existing = context.predicate_entity_indices[static_cast< size_t >(predicate)];
      if(existing >= 0) {
         return existing;
      }
      const int64_t index = context.entity_count++;
      existing = index;
      if(config.export_node_names) {
         context.entity_names.emplace_back(
            "predicate:" + predicates.at(static_cast< size_t >(predicate)).name
         );
      }
      context.entity_role_ids.push_back(static_cast< int64_t >(FlatEntityRole::predicate_virtual));
      return index;
   }

   FlatTupleArguments tuple_args(
      SemanticEncodingContext& context,
      const SemanticAtom& atom,
      std::span< const int64_t > auxiliary = {}
   ) const
   {
      const std::optional< int64_t > predicate_entity = config.use_predicate_virtual_nodes
                                                           ? std::optional(ensure_predicate_entity(
                                                                context, atom.predicate
                                                             ))
                                                           : std::nullopt;
      return build_flat_tuple_args(std::span{atom.arguments}, auxiliary, predicate_entity);
   }

   void append_target_row(
      SemanticEncodingContext& context,
      TargetSource source,
      int64_t position,
      std::string display_name
   ) const
   {
      const int64_t index = static_cast< int64_t >(context.target_columns.size());
      context.target_columns.append(
         TargetRecord{
            .position = position,
            .index = index,
            .candidate_id = index,
            .depth = std::nullopt,
            .group_id = target_group_ids.at(source),
            .name = std::move(display_name),
         },
         false,
         true,
         config.export_node_names
      );
   }

   template < typename Key, typename Map >
   int64_t ensure_target_entity(
      SemanticEncodingContext& context,
      Map& indices,
      Key key,
      TargetSource source,
      FlatEntityRole role,
      const std::string& display_name,
      bool& inserted
   ) const
   {
      if(const auto it = indices.find(key); it != indices.end()) {
         inserted = false;
         return it->second;
      }
      inserted = true;
      const int64_t index = context.entity_count++;
      indices.emplace(std::move(key), index);
      if(config.export_node_names) {
         context.entity_names.push_back(display_name);
      }
      context.entity_role_ids.push_back(static_cast< int64_t >(role));
      context.target_entity_indices.push_back(index);
      context.target_entity_group_ids.push_back(target_entity_group_ids.at(source));
      return index;
   }

   SemanticEncodingContext make_context(
      const SemanticFlatRelationInput& input,
      const std::vector< SemanticLiteral >& grouped_goals,
      const std::vector< SemanticGoalLevel >& goal_levels
   ) const
   {
      SemanticEncodingContext context;
      const auto& objects = semantic_objects(input);
      context.entity_count = static_cast< int64_t >(objects.size());
      if(config.export_node_names) {
         context.entity_names = objects;
      }
      context.entity_role_ids.assign(
         objects.size(), static_cast< int64_t >(FlatEntityRole::object)
      );
      context.object_indices.resize(objects.size());
      std::iota(context.object_indices.begin(), context.object_indices.end(), int64_t{0});
      context.predicate_entity_indices.assign(predicates.size(), -1);

      for(const auto source : {TargetSource::goals, TargetSource::subgoals}) {
         if(not has_anchor_entity_source(config, source)) {
            continue;
         }
         for(const auto& literal : grouped_goals) {
            const auto& predicate = predicates.at(static_cast< size_t >(literal.atom.predicate));
            if(config.ignore_zero_arity_relations and predicate.arity == 0) {
               continue;
            }
            const size_t level = semantic_goal_level(goal_levels, literal);
            if((source == TargetSource::goals and level > 0)
               or (source == TargetSource::subgoals and level == 0)) {
               continue;
            }
            const auto display = config.export_node_names
                                    ? goal_display_name(literal, level, predicates, objects)
                                    : std::string{};
            bool inserted = false;
            const auto key = GoalEntityKey{source, literal, level};
            const int64_t position = ensure_target_entity(
               context,
               context.goal_entity_indices,
               key,
               source,
               entity_role_for_target_source(source),
               display,
               inserted
            );
            if(has_target_source(config, source)) {
               append_target_row(context, source, position, display);
            }
         }
      }

      for(const auto& action : input.actions) {
         const auto display = config.export_node_names
                                 ? action_display_name(action, actions, objects)
                                 : std::string{};
         bool inserted = false;
         const int64_t position = ensure_target_entity(
            context,
            context.action_entity_indices,
            action,
            TargetSource::actions,
            FlatEntityRole::action,
            display,
            inserted
         );
         if(inserted) {
            context.unique_actions.push_back(action);
         }
         if(has_target_source(config, TargetSource::actions)) {
            append_target_row(context, TargetSource::actions, position, display);
         }
      }

      context.history_entries.reserve(input.history.size());
      for(const auto& entry : input.history) {
         if(input.history_max_steps.has_value() and std::abs(entry.dt) > *input.history_max_steps) {
            continue;
         }
         context.history_entries.push_back(
            PreparedHistoryEntry{
               .dt = entry.dt,
               .entry_index = context.history_entries.size(),
               .literals = entry.literals,
            }
         );
      }
      std::ranges::stable_sort(context.history_entries, {}, &PreparedHistoryEntry::dt);
      for(size_t idx = 0; idx < context.history_entries.size(); ++idx) {
         auto& entry = context.history_entries[idx];
         entry.entry_index = idx;
         entry.entity_index = context.entity_count++;
         if(config.export_node_names) {
            context.entity_names.push_back(
               "history:" + std::to_string(entry.dt) + "#" + std::to_string(idx)
            );
         }
         context.entity_role_ids.push_back(static_cast< int64_t >(FlatEntityRole::history));
         context.history_entity_indices.push_back(entry.entity_index);
         context.history_entity_dt.push_back(entry.dt);
      }

      if(has_anchor_entity_source(config, TargetSource::history)) {
         for(const auto& entry : context.history_entries) {
            for(const auto& literal : entry.literals) {
               const auto& predicate = predicates.at(static_cast< size_t >(literal.atom.predicate));
               if(config.ignore_zero_arity_relations and predicate.arity == 0) {
                  continue;
               }
               const auto display = config.export_node_names
                                       ? "history:" + std::to_string(entry.dt) + "#"
                                            + std::to_string(entry.entry_index) + ":"
                                            + std::string(literal.positive ? "[+]" : "[-]")
                                            + atom_display_name(literal.atom, predicates, objects)
                                       : std::string{};
               const auto key = HistoryEntityKey{entry.dt, entry.entry_index, literal};
               bool inserted = false;
               const int64_t position = ensure_target_entity(
                  context,
                  context.history_target_entity_indices,
                  key,
                  TargetSource::history,
                  FlatEntityRole::history_target,
                  display,
                  inserted
               );
               if(has_target_source(config, TargetSource::history)) {
                  append_target_row(context, TargetSource::history, position, display);
               }
            }
         }
      }
      return context;
   }

   void encode_into(
      const SemanticFlatRelationInput& input,
      BatchBuilder& builder,
      std::vector< std::string >& batch_target_names
   ) const
   {
      validate_input(input);
      const auto& objects = semantic_objects(input);
      const auto& goals = semantic_goals(input);
      const auto& static_facts = semantic_static_facts(input);

      const auto goal_levels = semantic_goal_levels(input);
      std::vector< SemanticLiteral > grouped_goals;
      for(const auto category : kCategoryOrder) {
         auto append_category = [&](const std::vector< SemanticLiteral >& literals) {
            for(const auto& literal : literals) {
               if(predicates.at(static_cast< size_t >(literal.atom.predicate)).category
                  == category) {
                  grouped_goals.push_back(literal);
               }
            }
         };
         append_category(goals);
         for(const auto& layer : input.subgoal_layers) {
            append_category(layer);
         }
      }

      auto context = make_context(input, grouped_goals, goal_levels);
      FlatRelationSink sink(metadata.relation_names.size(), config.include_lgan_edges);
      auto emit =
         [&](int relation_id, const SemanticAtom& atom, std::span< const int64_t > auxiliary = {}) {
            const auto& predicate = predicates.at(static_cast< size_t >(atom.predicate));
            if(config.ignore_zero_arity_relations and predicate.arity == 0) {
               return;
            }
            const auto args = tuple_args(context, atom, auxiliary);
            sink.emit(required_relation_id(relation_id), args);
         };

      std::set< SemanticAtom > fact_keys;
      const auto append_facts = [&](const std::vector< SemanticAtom >& facts, bool emit_facts) {
         for(const auto& fact : facts) {
            const auto category = predicates.at(static_cast< size_t >(fact.predicate)).category;
            if(emit_facts
               and (category != SemanticPredicateCategory::static_predicate or config.include_static)) {
               emit(state_relation_ids.at(static_cast< size_t >(fact.predicate)), fact);
            }
            fact_keys.emplace(fact);
         }
      };
      append_facts(static_facts, config.include_static);
      append_facts(input.state_facts, true);

      for(const auto& literal : grouped_goals) {
         const auto& predicate = predicates.at(static_cast< size_t >(literal.atom.predicate));
         if(kTopTypePredicates.contains(predicate.name)
            or (config.ignore_zero_arity_relations and predicate.arity == 0)) {
            continue;
         }
         const size_t level = semantic_goal_level(goal_levels, literal);
         if(config.goal_derivations.contains(GoalDerivation::plain)) {
            std::array< int64_t, 1 > auxiliary{};
            std::span< const int64_t > auxiliary_span;
            if(const auto source = source_for_goal_level(config, level); source.has_value()) {
               auxiliary[0] = context.goal_entity_indices.at(
                  GoalEntityKey{*source, literal, level}
               );
               auxiliary_span = std::span{auxiliary};
            }
            emit(
               goal_relation_ids.at(static_cast< size_t >(literal.atom.predicate))
                  .at(level)
                  .by_polarity[literal.positive ? 1 : 0][goal_derivation_index(std::nullopt)],
               literal.atom,
               auxiliary_span
            );
         }
         const bool satisfied = fact_keys.contains(literal.atom) == literal.positive;
         const auto derivation = satisfied ? GoalDerivation::satisfied
                                           : GoalDerivation::unsatisfied;
         if(config.goal_derivations.contains(derivation)) {
            emit(
               goal_relation_ids.at(static_cast< size_t >(literal.atom.predicate))
                  .at(level)
                  .by_polarity[literal.positive ? 1 : 0][goal_derivation_index(derivation)],
               literal.atom
            );
         }
      }

      for(const auto& action : context.unique_actions) {
         FlatTupleArguments args;
         args.reserve(action.arguments.size() + 1);
         args.push_back(context.action_entity_indices.at(action));
         args.insert(args.end(), action.arguments.begin(), action.arguments.end());
         sink.emit(
            required_relation_id(action_relation_ids.at(static_cast< size_t >(action.action))), args
         );
      }

      for(const auto& entry : context.history_entries) {
         for(const auto& literal : entry.literals) {
            const auto& predicate = predicates.at(static_cast< size_t >(literal.atom.predicate));
            if(config.ignore_zero_arity_relations and predicate.arity == 0) {
               continue;
            }
            std::array< int64_t, 2 > auxiliary{};
            std::span< const int64_t > auxiliary_span;
            if(has_anchor_entity_source(config, TargetSource::history)) {
               auxiliary[0] = context.history_target_entity_indices.at(
                  HistoryEntityKey{entry.dt, entry.entry_index, literal}
               );
               auxiliary[1] = entry.entity_index;
               auxiliary_span = std::span{auxiliary};
            } else {
               auxiliary[0] = entry.entity_index;
               auxiliary_span = std::span{auxiliary}.first(1);
            }
            emit(
               history_relation_ids.at(
                  static_cast< size_t >(literal.atom.predicate)
               )[literal.positive ? 1 : 0],
               literal.atom,
               auxiliary_span
            );
         }
      }

      std::vector< float > zeros(static_cast< size_t >(context.entity_count), 0.0F);
      builder.add_node_features(std::string(kFlatEntityNodeType), "x", std::span{zeros}, 1);
      if(config.export_node_names) {
         builder.set_node_names(std::string(kFlatEntityNodeType), context.entity_names);
         builder.set_object_names(objects);
      }

      const int64_t node_size = context.entity_count;
      const int64_t object_size = static_cast< int64_t >(context.object_indices.size());
      const int64_t history_size = static_cast< int64_t >(context.history_entity_indices.size());
      const int64_t target_entity_size = static_cast< int64_t >(
         context.target_entity_indices.size()
      );
      builder.set_field(std::string(kNodeSizesField), std::span{&node_size, size_t{1}});
      builder.set_field(std::string(kObjectSizesField), std::span{&object_size, size_t{1}});
      builder.set_field(std::string(kObjectIndicesField), std::span{context.object_indices});
      builder.set_field(std::string(kEntityRoleIdsField), std::span{context.entity_role_ids});
      builder.set_field(std::string(kHistoryEntitySizesField), std::span{&history_size, size_t{1}});
      builder.set_field(
         std::string(kHistoryEntityIndicesField), std::span{context.history_entity_indices}
      );
      builder.set_field(std::string(kHistoryEntityDtField), std::span{context.history_entity_dt});
      builder.set_field(
         std::string(kTargetEntitySizesField), std::span{&target_entity_size, size_t{1}}
      );
      builder.set_field(
         std::string(kTargetEntityIndicesField), std::span{context.target_entity_indices}
      );
      builder.set_field(
         std::string(kTargetEntityGroupIdsField), std::span{context.target_entity_group_ids}
      );

      if(not target_group_names.empty()) {
         const int64_t target_size = static_cast< int64_t >(context.target_columns.size());
         builder.set_field(std::string(kTargetSizesField), std::span{&target_size, size_t{1}});
         const TargetMetadataEmitConfig target_config{
            .position_node_type_id = std::string(kFlatEntityNodeType),
            .symbol_prefix = config.target_symbol_prefix,
            .include_depth = false,
            .include_group = true,
            .include_names = false,
            .groups = target_group_names,
            .parent_relation = std::nullopt,
         };
         set_target_fields(builder, context.target_columns, target_config);
         set_target_graph_attrs(builder, context.target_columns, target_config);
         if(config.export_node_names) {
            batch_target_names.insert(
               batch_target_names.end(),
               context.target_columns.names.begin(),
               context.target_columns.names.end()
            );
         }
      }

      builder.set_field(std::string(kRelationCountsField), std::span{sink.relation_counts()});
      const int64_t relation_instance_size = sink.relation_instance_count();
      builder.set_field(
         std::string(kRelationInstanceSizesField), std::span{&relation_instance_size, size_t{1}}
      );
      builder.set_field(std::string(kRelationArgsField), std::span{sink.relation_args()});

      if(config.include_lgan_edges) {
         if(context.target_entity_indices.empty()) {
            throw std::invalid_argument(
               "Semantic flat include_lgan_edges=true requires LGAN anchor entity rows"
            );
         }
         const auto lgan = build_flat_lgan(sink, std::span{context.target_entity_indices});
         const int64_t tn_size = static_cast< int64_t >(lgan.tn_relation_indices.size());
         const int64_t nn_size = static_cast< int64_t >(lgan.nn_relation_indices.size());
         const int64_t rr_size = static_cast< int64_t >(lgan.rr_src_relation_indices.size());
         builder.set_field(std::string(kLGANTNSizesField), std::span{&tn_size, size_t{1}});
         builder.set_field(
            std::string(kLGANTNRelationIndicesField), std::span{lgan.tn_relation_indices}
         );
         builder.set_field(
            std::string(kLGANTNEntityIndicesField), std::span{lgan.tn_entity_indices}
         );
         builder.set_field(std::string(kLGANNNSizesField), std::span{&nn_size, size_t{1}});
         builder.set_field(
            std::string(kLGANNNRelationIndicesField), std::span{lgan.nn_relation_indices}
         );
         builder.set_field(
            std::string(kLGANNNEntityIndicesField), std::span{lgan.nn_entity_indices}
         );
         builder.set_field(std::string(kLGANRRSizesField), std::span{&rr_size, size_t{1}});
         builder.set_field(
            std::string(kLGANRRSrcRelationIndicesField), std::span{lgan.rr_src_relation_indices}
         );
         builder.set_field(
            std::string(kLGANRRDstRelationIndicesField), std::span{lgan.rr_dst_relation_indices}
         );
      }
   }

   void encode_horizon(
      const SemanticTransitionDAG& dag,
      const SemanticFlatHorizonEncoderConfig& horizon,
      BatchBuilder& builder
   ) const
   {
      if(dag.predicates() != predicates or dag.actions() != actions) {
         throw std::invalid_argument(
            "Semantic flat Horizon DAG schema must exactly match the encoder schema"
         );
      }
      const auto& root = dag.root().state;
      const auto& root_objects = semantic_objects(root);
      const auto& root_goals = semantic_goals(root);
      if(root.subgoal_layers.size() > horizon.max_goal_level) {
         throw std::invalid_argument(
            "Semantic flat Horizon subgoal layer count exceeds max_goal_level"
         );
      }

      SemanticEncodingContext context;
      context.entity_count = static_cast< int64_t >(root_objects.size());
      if(horizon.export_node_names) {
         context.entity_names = root_objects;
      }
      context.entity_role_ids.assign(
         root_objects.size(), static_cast< int64_t >(FlatEntityRole::object)
      );
      context.object_indices.resize(root_objects.size());
      std::iota(context.object_indices.begin(), context.object_indices.end(), int64_t{0});
      context.predicate_entity_indices.assign(predicates.size(), -1);
      context.state_entity_indices.assign(dag.nodes().size(), -1);

      std::vector< TargetCandidateRow > target_rows;
      for(const auto& node : dag.nodes()) {
         const int64_t position = context.entity_count++;
         context.state_entity_indices.at(static_cast< size_t >(node.index)) = position;
         const bool public_root = root_in_public_carrier(horizon.root_policy) or node.index != 0;
         if(horizon.export_node_names) {
            context.entity_names.push_back(
               public_root ? horizon.target_symbol_prefix + std::to_string(node.index)
                           : "_root_state_"
            );
         }
         context.entity_role_ids.push_back(static_cast< int64_t >(FlatEntityRole::state));
         if(not root_in_target_metadata(horizon.root_policy) and node.index == 0) {
            continue;
         }
         context.target_entity_indices.push_back(position);
         context.target_entity_group_ids.push_back(0);
         target_rows.push_back(
            TargetCandidateRow{
               .position = position,
               .index = node.index,
               .candidate_id = node.candidate_id,
               .depth = node.depth,
               .group_id = int64_t{0},
               .name = horizon.export_node_names
                          ? node.display_name.value_or("state:" + std::to_string(node.index))
                          : std::string{},
            }
         );
      }
      append_target_candidate_rows(
         context.target_columns,
         target_rows,
         TargetCandidateAppendConfig{
            .include_depth = true,
            .include_group = true,
            .include_names = horizon.export_node_names,
            .missing_candidate_id_prefix = "missing candidate_id for target node index ",
            .duplicate_candidate_id_prefix = "duplicate candidate_id ",
         }
      );

      const auto goal_levels = semantic_goal_levels(root);
      std::vector< SemanticLiteral > goals;
      for(const auto category : kCategoryOrder) {
         auto append_category = [&](const std::vector< SemanticLiteral >& literals) {
            for(const auto& literal : literals) {
               if(predicates.at(static_cast< size_t >(literal.atom.predicate)).category
                  == category) {
                  goals.push_back(literal);
               }
            }
         };
         append_category(root_goals);
         for(const auto& layer : root.subgoal_layers) {
            append_category(layer);
         }
      }

      FlatRelationSink sink(metadata.relation_names.size(), horizon.include_lgan_edges);
      auto state_position = [&](int64_t node_index) {
         if(node_index < 0
            or static_cast< size_t >(node_index) >= context.state_entity_indices.size()
            or context.state_entity_indices[static_cast< size_t >(node_index)] < 0) {
            throw std::invalid_argument(
               "Semantic flat Horizon encountered missing state carrier for node index "
               + std::to_string(node_index)
            );
         }
         return context.state_entity_indices[static_cast< size_t >(node_index)];
      };
      auto emit_atom = [&](
                          int relation_id,
                          const SemanticAtom& atom,
                          std::optional< int64_t > state_anchor = std::nullopt
                       ) {
         const auto& predicate = predicates.at(static_cast< size_t >(atom.predicate));
         if(horizon.ignore_zero_arity_relations and predicate.arity == 0) {
            return;
         }
         const std::array< int64_t, 1 > auxiliary = {state_anchor.value_or(0)};
         sink.emit(
            required_relation_id(relation_id),
            tuple_args(
               context,
               atom,
               state_anchor.has_value() ? std::span{auxiliary} : std::span< const int64_t >{}
            )
         );
      };
      auto emit_state = [&](
                           const SemanticFlatRelationInput& input,
                           int64_t node_index,
                           bool include_static,
                           bool include_anchor
                        ) {
         std::set< SemanticAtom > facts;
         const auto anchor = include_anchor ? std::optional(state_position(node_index))
                                            : std::nullopt;
         const auto append_facts = [&](const std::vector< SemanticAtom >& atoms) {
            for(const auto& atom : atoms) {
               const auto& predicate = predicates.at(static_cast< size_t >(atom.predicate));
               if(predicate.category == SemanticPredicateCategory::static_predicate
                  and not include_static) {
                  continue;
               }
               if(horizon.ignore_zero_arity_relations and predicate.arity == 0) {
                  continue;
               }
               const auto predicate_index = static_cast< size_t >(atom.predicate);
               emit_atom(
                  anchor.has_value() ? horizon_state_anchored_relation_ids.at(predicate_index)
                                     : horizon_state_relation_ids.at(predicate_index),
                  atom,
                  anchor
               );
               facts.insert(atom);
            }
         };
         append_facts(semantic_static_facts(input));
         append_facts(input.state_facts);
         return facts;
      };
      auto emit_goal = [&](
                          const SemanticLiteral& literal,
                          size_t level,
                          int64_t node_index,
                          bool include_anchor,
                          std::optional< GoalDerivation > derivation
                       ) {
         const auto& predicate = predicates.at(static_cast< size_t >(literal.atom.predicate));
         if(kTopTypePredicates.contains(predicate.name)
            or (horizon.ignore_zero_arity_relations and predicate.arity == 0)) {
            return;
         }
         const auto anchor = include_anchor ? std::optional(state_position(node_index))
                                            : std::nullopt;
         const auto& ids = horizon_goal_relation_ids
                              .at(static_cast< size_t >(literal.atom.predicate))
                              .at(level);
         const auto relation_id = (anchor.has_value() ? ids.candidate : ids.root)
            [literal.positive ? 1 : 0][horizon_goal_derivation_index(derivation)];
         emit_atom(relation_id, literal.atom, anchor);
      };

      const bool root_anchor = root_in_state_relations(horizon.root_policy);
      const auto root_facts = emit_state(root, 0, horizon.include_static, root_anchor);
      for(const auto& literal : goals) {
         const size_t level = semantic_goal_level(goal_levels, literal);
         if(horizon.goal_derivations.contains(GoalDerivation::plain)) {
            emit_goal(literal, level, 0, root_anchor, std::nullopt);
         }
         const bool satisfied = root_facts.contains(literal.atom) == literal.positive;
         const auto derivation = satisfied ? GoalDerivation::satisfied
                                           : GoalDerivation::unsatisfied;
         if(horizon.goal_derivations.contains(derivation)) {
            emit_goal(literal, level, 0, root_anchor, derivation);
         }
      }

      auto emit_action = [&](const SemanticGroundAction& action, int64_t node_index) {
         FlatTupleArguments args;
         args.reserve(action.arguments.size() + 1);
         args.push_back(state_position(node_index));
         args.insert(args.end(), action.arguments.begin(), action.arguments.end());
         sink.emit(
            required_relation_id(
               horizon_action_relation_ids.at(static_cast< size_t >(action.action))
            ),
            args
         );
      };
      auto emit_delta = [&](const SemanticLiteral& literal, int64_t node_index) {
         const auto& predicate = predicates.at(static_cast< size_t >(literal.atom.predicate));
         if(horizon.ignore_zero_arity_relations and predicate.arity == 0) {
            return;
         }
         emit_atom(
            horizon_literal_relation_ids.at(
               static_cast< size_t >(literal.atom.predicate)
            )[literal.positive ? 1 : 0],
            literal.atom,
            state_position(node_index)
         );
      };
      auto emit_delta_satisfaction = [&](
                                        const SemanticLiteral& goal,
                                        size_t level,
                                        const std::set< SemanticAtom >& added,
                                        const std::set< SemanticAtom >& removed,
                                        int64_t node_index
                                     ) {
         const auto derivation = delta_goal_satisfaction_derivation(
            goal.positive, added.contains(goal.atom), removed.contains(goal.atom)
         );
         if(not derivation.has_value() or not horizon.goal_derivations.contains(*derivation)) {
            return;
         }
         emit_goal(goal, level, node_index, true, derivation);
      };

      std::set< SemanticAtom > root_fluent;
      std::set< SemanticAtom > root_derived;
      if(horizon.transition_mode == SemanticHorizonMode::delta) {
         for(const auto& atom : root.state_facts) {
            const auto category = predicates.at(static_cast< size_t >(atom.predicate)).category;
            if(category == SemanticPredicateCategory::fluent) {
               root_fluent.insert(atom);
            } else if(category == SemanticPredicateCategory::derived) {
               root_derived.insert(atom);
            }
         }
      }

      const bool encode_actions = not horizon.ignore_actions
                                  or horizon.transition_mode == SemanticHorizonMode::action;
      for(size_t index = 1; index < dag.nodes().size(); ++index) {
         const auto& node = dag.nodes()[index];
         if(horizon.transition_mode == SemanticHorizonMode::full) {
            const auto candidate_facts = emit_state(node.state, node.index, false, true);
            if(encode_actions and node.incoming_action.has_value()) {
               emit_action(*node.incoming_action, node.index);
            }
            for(const auto& goal : goals) {
               const bool satisfied = candidate_facts.contains(goal.atom) == goal.positive;
               const auto derivation = satisfied ? GoalDerivation::satisfied
                                                 : GoalDerivation::unsatisfied;
               if(horizon.goal_derivations.contains(derivation)) {
                  emit_goal(
                     goal, semantic_goal_level(goal_levels, goal), node.index, true, derivation
                  );
               }
            }
            continue;
         }
         if(horizon.transition_mode == SemanticHorizonMode::action) {
            if(encode_actions and node.incoming_action.has_value()) {
               emit_action(*node.incoming_action, node.index);
            }
            continue;
         }

         std::set< SemanticAtom > added_fluent;
         std::set< SemanticAtom > removed_fluent;
         std::set< SemanticAtom > added_derived;
         std::set< SemanticAtom > removed_derived;
         if(node.delta_literals.has_value()) {
            for(const auto& literal : *node.delta_literals) {
               emit_delta(literal, node.index);
               const auto category = predicates.at(static_cast< size_t >(literal.atom.predicate))
                                        .category;
               auto* changed = category == SemanticPredicateCategory::fluent
                                  ? (literal.positive ? &added_fluent : &removed_fluent)
                               : category == SemanticPredicateCategory::derived
                                  ? (literal.positive ? &added_derived : &removed_derived)
                                  : nullptr;
               if(changed != nullptr) {
                  changed->insert(literal.atom);
               }
            }
         } else {
            std::set< SemanticAtom > candidate_fluent;
            std::set< SemanticAtom > candidate_derived;
            for(const auto& atom : node.state.state_facts) {
               const auto category = predicates.at(static_cast< size_t >(atom.predicate)).category;
               if(category == SemanticPredicateCategory::fluent) {
                  candidate_fluent.insert(atom);
               } else if(category == SemanticPredicateCategory::derived) {
                  candidate_derived.insert(atom);
               }
            }
            for(const auto& atom : candidate_fluent) {
               if(not root_fluent.contains(atom)) {
                  added_fluent.insert(atom);
                  emit_delta(SemanticLiteral{atom, true}, node.index);
               }
            }
            for(const auto& atom : root_fluent) {
               if(not candidate_fluent.contains(atom)) {
                  removed_fluent.insert(atom);
                  emit_delta(SemanticLiteral{atom, false}, node.index);
               }
            }
            for(const auto& atom : candidate_derived) {
               if(not root_derived.contains(atom)) {
                  added_derived.insert(atom);
                  emit_delta(SemanticLiteral{atom, true}, node.index);
               }
            }
            for(const auto& atom : root_derived) {
               if(not candidate_derived.contains(atom)) {
                  removed_derived.insert(atom);
                  emit_delta(SemanticLiteral{atom, false}, node.index);
               }
            }
         }
         if(encode_actions and node.incoming_action.has_value()) {
            emit_action(*node.incoming_action, node.index);
         }
         for(const auto& goal : goals) {
            const auto category = predicates.at(static_cast< size_t >(goal.atom.predicate))
                                     .category;
            if(category == SemanticPredicateCategory::fluent) {
               emit_delta_satisfaction(
                  goal,
                  semantic_goal_level(goal_levels, goal),
                  added_fluent,
                  removed_fluent,
                  node.index
               );
            } else if(category == SemanticPredicateCategory::derived) {
               emit_delta_satisfaction(
                  goal,
                  semantic_goal_level(goal_levels, goal),
                  added_derived,
                  removed_derived,
                  node.index
               );
            }
         }
      }

      const bool exclude_root_topology = horizon.root_policy == RootPolicy::exclude;
      if(horizon.enable_parent_relation) {
         for(const auto& [parent, child] : dag.edges()) {
            if(exclude_root_topology and parent == 0) {
               continue;
            }
            const std::array< int64_t, 2 > args = {state_position(parent), state_position(child)};
            sink.emit(required_relation_id(horizon_parent_relation_id), args);
         }
      }
      std::vector< std::vector< int64_t > > children(dag.nodes().size());
      for(const auto& [parent, child] : dag.edges()) {
         if(not exclude_root_topology or parent != 0) {
            children[static_cast< size_t >(parent)].push_back(child);
         }
      }
      auto emit_pair = [&](int relation_id, int64_t source, int64_t target) {
         const std::array< int64_t, 2 > args = {state_position(source), state_position(target)};
         sink.emit(required_relation_id(relation_id), args);
      };
      struct PairHash {
         size_t operator()(const std::pair< int64_t, int64_t >& value) const noexcept
         {
            return std::hash< int64_t >{}(value.first)
                   ^ (std::hash< int64_t >{}(value.second) << 1);
         }
      };
      hash_set< std::pair< int64_t, int64_t >, PairHash > siblings;
      if(horizon.enable_sibling_relation) {
         for(auto& values : children) {
            std::ranges::sort(values);
            for(size_t lhs = 0; lhs < values.size(); ++lhs) {
               for(size_t rhs = lhs + 1; rhs < values.size(); ++rhs) {
                  if(siblings.emplace(values[lhs], values[rhs]).second) {
                     emit_pair(horizon_sibling_relation_id, values[lhs], values[rhs]);
                     emit_pair(horizon_sibling_relation_id, values[rhs], values[lhs]);
                  }
               }
            }
         }
      }
      if(horizon.enable_cousin_relation) {
         hash_set< std::pair< int64_t, int64_t >, PairHash > cousins;
         for(const auto& parents : children) {
            for(size_t lhs = 0; lhs < parents.size(); ++lhs) {
               for(size_t rhs = lhs + 1; rhs < parents.size(); ++rhs) {
                  const auto& left = children.at(static_cast< size_t >(parents[lhs]));
                  const auto& right = children.at(static_cast< size_t >(parents[rhs]));
                  for(const auto u : left) {
                     for(const auto v : right) {
                        if(u == v) {
                           continue;
                        }
                        const auto pair = std::minmax(u, v);
                        if(siblings.contains(pair) or not cousins.emplace(pair).second) {
                           continue;
                        }
                        emit_pair(horizon_cousin_relation_id, u, v);
                        emit_pair(horizon_cousin_relation_id, v, u);
                     }
                  }
               }
            }
         }
      }

      std::vector< float > zeros(static_cast< size_t >(context.entity_count), 0.0F);
      builder.add_node_features(std::string(kFlatEntityNodeType), "x", std::span{zeros}, 1);
      if(horizon.export_node_names) {
         builder.set_node_names(std::string(kFlatEntityNodeType), context.entity_names);
         builder.set_object_names(root_objects);
      }
      const int64_t node_size = context.entity_count;
      const int64_t object_size = static_cast< int64_t >(context.object_indices.size());
      const int64_t target_entity_size = static_cast< int64_t >(
         context.target_entity_indices.size()
      );
      const int64_t target_size = static_cast< int64_t >(context.target_columns.size());
      builder.set_field(std::string(kNodeSizesField), std::span{&node_size, size_t{1}});
      builder.set_field(std::string(kObjectSizesField), std::span{&object_size, size_t{1}});
      builder.set_field(std::string(kObjectIndicesField), std::span{context.object_indices});
      builder.set_field(std::string(kEntityRoleIdsField), std::span{context.entity_role_ids});
      builder.set_field(
         std::string(kTargetEntitySizesField), std::span{&target_entity_size, size_t{1}}
      );
      builder.set_field(
         std::string(kTargetEntityIndicesField), std::span{context.target_entity_indices}
      );
      builder.set_field(
         std::string(kTargetEntityGroupIdsField), std::span{context.target_entity_group_ids}
      );
      builder.set_field(std::string(kTargetSizesField), std::span{&target_size, size_t{1}});
      const std::vector< std::string > groups = {
         std::string(target_source_group_name(TargetSource::states))
      };
      const TargetMetadataEmitConfig target_config{
         .position_node_type_id = std::string(kFlatEntityNodeType),
         .symbol_prefix = horizon.target_symbol_prefix,
         .include_depth = true,
         .include_group = true,
         .include_names = false,
         .groups = groups,
         .parent_relation = horizon.parent_relation,
      };
      set_target_fields(builder, context.target_columns, target_config);
      set_target_graph_attrs(builder, context.target_columns, target_config);
      if(horizon.export_node_names) {
         if(context.target_columns.names.empty()) {
            builder.set_graph_attr(std::string(kTargetNamesAttr), std::vector< std::string >{});
         } else {
            builder.add_lazy_target_names(std::span{context.target_columns.names});
         }
      }
      builder.set_field(std::string(kRelationCountsField), std::span{sink.relation_counts()});
      const int64_t relation_size = sink.relation_instance_count();
      builder.set_field(
         std::string(kRelationInstanceSizesField), std::span{&relation_size, size_t{1}}
      );
      builder.set_field(std::string(kRelationArgsField), std::span{sink.relation_args()});

      if(horizon.include_lgan_edges) {
         if(context.target_columns.positions.empty()) {
            throw std::invalid_argument(
               "FlatHorizonEncoder include_lgan_edges=true requires surviving candidate state "
               "rows, but none were encoded. Ensure the horizon DAG exposes at least one "
               "selectable candidate state."
            );
         }
         const auto lgan = build_flat_lgan(sink, std::span{context.target_columns.positions});
         const int64_t tn_size = static_cast< int64_t >(lgan.tn_relation_indices.size());
         const int64_t nn_size = static_cast< int64_t >(lgan.nn_relation_indices.size());
         const int64_t rr_size = static_cast< int64_t >(lgan.rr_src_relation_indices.size());
         builder.set_field(std::string(kLGANTNSizesField), std::span{&tn_size, size_t{1}});
         builder.set_field(
            std::string(kLGANTNRelationIndicesField), std::span{lgan.tn_relation_indices}
         );
         builder.set_field(
            std::string(kLGANTNEntityIndicesField), std::span{lgan.tn_entity_indices}
         );
         builder.set_field(std::string(kLGANNNSizesField), std::span{&nn_size, size_t{1}});
         builder.set_field(
            std::string(kLGANNNRelationIndicesField), std::span{lgan.nn_relation_indices}
         );
         builder.set_field(
            std::string(kLGANNNEntityIndicesField), std::span{lgan.nn_entity_indices}
         );
         builder.set_field(std::string(kLGANRRSizesField), std::span{&rr_size, size_t{1}});
         builder.set_field(
            std::string(kLGANRRSrcRelationIndicesField), std::span{lgan.rr_src_relation_indices}
         );
         builder.set_field(
            std::string(kLGANRRDstRelationIndicesField), std::span{lgan.rr_dst_relation_indices}
         );
      }
   }

   void finalize_horizon_encoding(
      BatchBuilder::BatchEncoding& encoding,
      const SemanticFlatHorizonEncoderConfig& horizon
   ) const
   {
      if(horizon.pack_relation_args_relation_major) {
         pack_flat_relation_args_relation_major(encoding, std::span{metadata.relation_arities});
      }
   }

   BatchBuilder::BatchEncoding encode_many(
      std::span< const SemanticFlatRelationInput > inputs
   ) const
   {
      BatchBuilder builder;
      prepare_builder(builder);
      std::vector< std::string > target_names;
      for(const auto& input : inputs) {
         encode_into(input, builder, target_names);
         builder.next_graph();
      }
      if(not target_group_names.empty() and config.export_node_names) {
         if(target_names.empty()) {
            builder.set_graph_attr(std::string(kTargetNamesAttr), std::vector< std::string >{});
         } else {
            builder.add_lazy_target_names(std::span{target_names});
         }
      }
      auto encoding = builder.build();
      finalize_batch_encoding(encoding);
      return encoding;
   }

   void encode_one_into(const SemanticFlatRelationInput& input, BatchBuilder& builder) const
   {
      prepare_builder(builder);
      std::vector< std::string > target_names;
      encode_into(input, builder, target_names);
      if(not target_group_names.empty() and config.export_node_names) {
         if(target_names.empty()) {
            builder.set_graph_attr(std::string(kTargetNamesAttr), std::vector< std::string >{});
         } else {
            builder.add_lazy_target_names(std::span{target_names});
         }
      }
   }

   void finalize_batch_encoding(BatchBuilder::BatchEncoding& encoding) const
   {
      if(config.pack_relation_args_relation_major) {
         pack_flat_relation_args_relation_major(encoding, std::span{metadata.relation_arities});
      }
   }
};

SemanticFlatRelationEncoderEngine::SemanticFlatRelationEncoderEngine(
   std::vector< SemanticPredicateSpec > predicates,
   std::vector< SemanticActionSpec > actions,
   Config config
)
    : impl_(std::make_unique< Impl >(std::move(predicates), std::move(actions), std::move(config)))
{
}

SemanticFlatRelationEncoderEngine::SemanticFlatRelationEncoderEngine(
   std::shared_ptr< const SemanticTaskContext > task_context,
   Config config
)
    : impl_(std::make_unique< Impl >(std::move(task_context), std::move(config)))
{
}

SemanticFlatRelationEncoderEngine::SemanticFlatRelationEncoderEngine(
   SemanticFlatRelationEncoderEngine&&
) noexcept = default;

SemanticFlatRelationEncoderEngine& SemanticFlatRelationEncoderEngine::operator=(
   SemanticFlatRelationEncoderEngine&&
) noexcept = default;

SemanticFlatRelationEncoderEngine::~SemanticFlatRelationEncoderEngine() = default;

BatchBuilder::BatchEncoding SemanticFlatRelationEncoderEngine::encode(
   const SemanticFlatRelationInput& input
) const
{
   return impl_->encode_many(std::span{&input, size_t{1}});
}

void SemanticFlatRelationEncoderEngine::encode(
   const SemanticFlatRelationInput& input,
   BatchBuilder& builder
) const
{
   impl_->encode_one_into(input, builder);
}

BatchBuilder::BatchEncoding SemanticFlatRelationEncoderEngine::encode_batch(
   const std::vector< SemanticFlatRelationInput >& inputs
) const
{
   return impl_->encode_many(std::span{inputs});
}

void SemanticFlatRelationEncoderEngine::finalize_batch_encoding(
   BatchBuilder::BatchEncoding& encoding
) const
{
   impl_->finalize_batch_encoding(encoding);
}

const SemanticFlatRelationEncoderEngine::Config&
SemanticFlatRelationEncoderEngine::get_config() const
{
   return impl_->config;
}

const std::shared_ptr< const SemanticTaskContext >&
SemanticFlatRelationEncoderEngine::get_task_context() const
{
   return impl_->task_context;
}

const std::vector< SemanticPredicateSpec >&
SemanticFlatRelationEncoderEngine::get_predicates() const
{
   return impl_->predicates;
}

const std::vector< SemanticActionSpec >& SemanticFlatRelationEncoderEngine::get_actions() const
{
   return impl_->actions;
}

const std::vector< std::string >& SemanticFlatRelationEncoderEngine::get_relation_names() const
{
   return impl_->metadata.relation_names;
}

const std::vector< int64_t >& SemanticFlatRelationEncoderEngine::get_relation_arities() const
{
   return impl_->metadata.relation_arities;
}

const std::vector< std::string >& SemanticFlatRelationEncoderEngine::get_relation_sources() const
{
   return impl_->metadata.relation_sources;
}

const std::vector< int64_t >&
SemanticFlatRelationEncoderEngine::get_relation_logical_arities() const
{
   return impl_->metadata.relation_logical_arities;
}

const std::vector< int64_t >&
SemanticFlatRelationEncoderEngine::get_relation_encoded_arities() const
{
   return impl_->metadata.relation_encoded_arities;
}

const std::vector< int64_t >& SemanticFlatRelationEncoderEngine::get_relation_slot_roles() const
{
   return impl_->metadata.relation_slot_roles;
}

const std::vector< int64_t >&
SemanticFlatRelationEncoderEngine::get_relation_slot_role_offsets() const
{
   return impl_->metadata.relation_slot_role_offsets;
}

const std::vector< std::string >& SemanticFlatRelationEncoderEngine::get_slot_role_names() const
{
   return impl_->metadata.slot_role_names;
}

void SemanticFlatRelationEncoderEngine::configure_horizon(
   const SemanticFlatHorizonEncoderConfig& config
)
{
   impl_->configure_horizon(config);
}

void SemanticFlatRelationEncoderEngine::prepare_horizon_builder(
   BatchBuilder& builder,
   const SemanticFlatHorizonEncoderConfig& config
) const
{
   impl_->prepare_horizon_builder(builder, config);
}

void SemanticFlatRelationEncoderEngine::encode_horizon(
   const SemanticTransitionDAG& dag,
   const SemanticFlatHorizonEncoderConfig& config,
   BatchBuilder& builder
) const
{
   impl_->encode_horizon(dag, config, builder);
}

void SemanticFlatRelationEncoderEngine::finalize_horizon_encoding(
   BatchBuilder::BatchEncoding& encoding,
   const SemanticFlatHorizonEncoderConfig& config
) const
{
   impl_->finalize_horizon_encoding(encoding, config);
}

}  // namespace mifrost
