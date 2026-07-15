/**
 * @file semantic_flat_relation_encoder.cpp
 * @brief Native encoding of owned, planning-backend-neutral flat graph inputs.
 */
#include "semantic_flat_relation_encoder.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
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

namespace mifrost {

namespace {

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

struct HistoryEntityKey {
   int64_t dt = 0;
   size_t entry_index = 0;
   SemanticLiteral literal;

   auto operator<=>(const HistoryEntityKey&) const = default;
};

struct PreparedHistoryEntry {
   int64_t dt = 0;
   size_t entry_index = 0;
   std::vector< SemanticLiteral > literals;
   int64_t entity_index = -1;
};

struct SemanticEncodingContext {
   std::vector< std::string > entity_names;
   std::vector< int64_t > entity_role_ids;
   std::vector< int64_t > object_indices;
   std::map< int64_t, int64_t > predicate_entity_indices;
   std::vector< int64_t > history_entity_indices;
   std::vector< int64_t > history_entity_dt;
   std::vector< int64_t > target_entity_indices;
   std::vector< int64_t > target_entity_group_ids;
   std::map< GoalEntityKey, int64_t > goal_entity_indices;
   std::map< SemanticGroundAction, int64_t > action_entity_indices;
   std::map< HistoryEntityKey, int64_t > history_target_entity_indices;
   std::vector< SemanticGroundAction > unique_actions;
   std::vector< PreparedHistoryEntry > history_entries;
   TargetColumns target_columns;
};

}  // namespace

struct SemanticFlatRelationEncoderEngine::Impl {
   Config config;
   std::vector< SemanticPredicateSpec > predicates;
   std::vector< SemanticActionSpec > actions;
   FlatRelationSchemaMetadata metadata;
   std::vector< std::string > target_entity_group_names;
   std::map< TargetSource, int64_t > target_entity_group_ids;
   std::vector< std::string > target_group_names;
   std::map< TargetSource, int64_t > target_group_ids;

   Impl(
      std::vector< SemanticPredicateSpec > predicate_specs,
      std::vector< SemanticActionSpec > action_specs,
      Config encoder_config
   )
       : config(std::move(encoder_config)),
         predicates(std::move(predicate_specs)),
         actions(std::move(action_specs))
   {
      validate_config();
      validate_unique_names(predicates, "predicate");
      validate_unique_names(actions, "action");
      build_groups();
      build_schema();
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
      std::set< std::string, std::less<> > object_names;
      for(const auto& object : input.objects) {
         validate_name(object, "object");
         if(not object_names.emplace(object).second) {
            throw std::invalid_argument("Semantic flat object names must be unique");
         }
      }
      for(const auto& fact : input.state_facts) {
         validate_atom(fact, input.objects.size(), "state fact");
      }
      for(const auto& goal : input.goals) {
         validate_atom(goal.atom, input.objects.size(), "goal");
      }
      if(input.subgoal_layers.size() > config.max_goal_level) {
         throw std::invalid_argument("Semantic flat subgoal layer count exceeds max_goal_level");
      }
      for(const auto& layer : input.subgoal_layers) {
         for(const auto& goal : layer) {
            validate_atom(goal.atom, input.objects.size(), "subgoal");
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
            if(object < 0 or static_cast< size_t >(object) >= input.objects.size()) {
               throw std::invalid_argument("Semantic flat action object index out of range");
            }
         }
      }
      for(const auto& entry : input.history) {
         if(entry.dt >= 0) {
            throw std::invalid_argument("Semantic flat history requires negative dt values");
         }
         for(const auto& literal : entry.literals) {
            validate_atom(literal.atom, input.objects.size(), "history literal");
         }
      }
   }

   int relation_id(std::string_view name) const
   {
      const auto it = metadata.relation_name_to_id.find(std::string(name));
      if(it == metadata.relation_name_to_id.end()) {
         throw std::invalid_argument("Unknown semantic flat relation '" + std::string(name) + "'");
      }
      return it->second;
   }

   int64_t ensure_predicate_entity(SemanticEncodingContext& context, int64_t predicate) const
   {
      if(const auto it = context.predicate_entity_indices.find(predicate);
         it != context.predicate_entity_indices.end()) {
         return it->second;
      }
      const int64_t index = static_cast< int64_t >(context.entity_names.size());
      context.predicate_entity_indices.emplace(predicate, index);
      context.entity_names.emplace_back(
         "predicate:" + predicates.at(static_cast< size_t >(predicate)).name
      );
      context.entity_role_ids.push_back(static_cast< int64_t >(FlatEntityRole::predicate_virtual));
      return index;
   }

   std::vector< int64_t > tuple_args(
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
         true
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
      const int64_t index = static_cast< int64_t >(context.entity_names.size());
      indices.emplace(std::move(key), index);
      context.entity_names.push_back(display_name);
      context.entity_role_ids.push_back(static_cast< int64_t >(role));
      context.target_entity_indices.push_back(index);
      context.target_entity_group_ids.push_back(target_entity_group_ids.at(source));
      return index;
   }

   SemanticEncodingContext make_context(
      const SemanticFlatRelationInput& input,
      const std::vector< SemanticLiteral >& grouped_goals,
      const std::map< SemanticLiteral, size_t >& goal_levels
   ) const
   {
      SemanticEncodingContext context;
      context.entity_names = input.objects;
      context.entity_role_ids.assign(
         input.objects.size(), static_cast< int64_t >(FlatEntityRole::object)
      );
      context.object_indices.resize(input.objects.size());
      std::iota(context.object_indices.begin(), context.object_indices.end(), int64_t{0});

      for(const auto source : {TargetSource::goals, TargetSource::subgoals}) {
         if(not has_anchor_entity_source(config, source)) {
            continue;
         }
         for(const auto& literal : grouped_goals) {
            const auto& predicate = predicates.at(static_cast< size_t >(literal.atom.predicate));
            if(config.ignore_zero_arity_relations and predicate.arity == 0) {
               continue;
            }
            const size_t level = goal_levels.at(literal);
            if((source == TargetSource::goals and level > 0)
               or (source == TargetSource::subgoals and level == 0)) {
               continue;
            }
            const auto display = goal_display_name(literal, level, predicates, input.objects);
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
         const auto display = action_display_name(action, actions, input.objects);
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
         entry.entity_index = static_cast< int64_t >(context.entity_names.size());
         context.entity_names.push_back(
            "history:" + std::to_string(entry.dt) + "#" + std::to_string(idx)
         );
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
               const auto display = "history:" + std::to_string(entry.dt) + "#"
                                    + std::to_string(entry.entry_index) + ":"
                                    + std::string(literal.positive ? "[+]" : "[-]")
                                    + atom_display_name(literal.atom, predicates, input.objects);
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

      std::map< SemanticLiteral, size_t > goal_levels;
      for(const auto& literal : input.goals) {
         goal_levels[literal] = 0;
      }
      for(size_t layer = 0; layer < input.subgoal_layers.size(); ++layer) {
         for(const auto& literal : input.subgoal_layers[layer]) {
            goal_levels[literal] = layer + 1;
         }
      }
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
         append_category(input.goals);
         for(const auto& layer : input.subgoal_layers) {
            append_category(layer);
         }
      }

      auto context = make_context(input, grouped_goals, goal_levels);
      FlatRelationSink sink(metadata.relation_names.size(), config.include_lgan_edges);
      auto emit = [&](
                     std::string_view name,
                     const SemanticAtom& atom,
                     std::span< const int64_t > auxiliary = {}
                  ) {
         const auto& predicate = predicates.at(static_cast< size_t >(atom.predicate));
         if(config.ignore_zero_arity_relations and predicate.arity == 0) {
            return;
         }
         const auto args = tuple_args(context, atom, auxiliary);
         sink.emit(relation_id(name), args);
      };

      std::set< SemanticAtom > fact_keys;
      for(const auto& fact : input.state_facts) {
         const auto category = predicates.at(static_cast< size_t >(fact.predicate)).category;
         if(category == SemanticPredicateCategory::static_predicate and not config.include_static) {
            continue;
         }
         emit(predicates.at(static_cast< size_t >(fact.predicate)).name, fact);
         fact_keys.emplace(fact);
      }

      for(const auto& literal : grouped_goals) {
         const auto& predicate = predicates.at(static_cast< size_t >(literal.atom.predicate));
         if(kTopTypePredicates.contains(predicate.name)
            or (config.ignore_zero_arity_relations and predicate.arity == 0)) {
            continue;
         }
         const size_t level = goal_levels.at(literal);
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
               goal_relation_name(predicate.name, literal.positive, level, std::nullopt),
               literal.atom,
               auxiliary_span
            );
         }
         const bool satisfied = fact_keys.contains(literal.atom) == literal.positive;
         const auto derivation = satisfied ? GoalDerivation::satisfied
                                           : GoalDerivation::unsatisfied;
         if(config.goal_derivations.contains(derivation)) {
            emit(
               goal_relation_name(predicate.name, literal.positive, level, derivation), literal.atom
            );
         }
      }

      for(const auto& action : context.unique_actions) {
         std::vector< int64_t > args;
         args.reserve(action.arguments.size() + 1);
         args.push_back(context.action_entity_indices.at(action));
         args.insert(args.end(), action.arguments.begin(), action.arguments.end());
         sink.emit(relation_id(actions.at(static_cast< size_t >(action.action)).name), args);
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
               goal_relation_name(
                  predicate.name, literal.positive, std::nullopt, std::nullopt, "[hist]"
               ),
               literal.atom,
               auxiliary_span
            );
         }
      }

      std::vector< float > zeros(context.entity_names.size(), 0.0F);
      builder.add_node_features(std::string(kFlatEntityNodeType), "x", std::span{zeros}, 1);
      if(config.export_node_names) {
         builder.set_node_names(std::string(kFlatEntityNodeType), context.entity_names);
         builder.set_object_names(input.objects);
      }

      const int64_t node_size = static_cast< int64_t >(context.entity_names.size());
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
      if(config.pack_relation_args_relation_major) {
         pack_flat_relation_args_relation_major(encoding, std::span{metadata.relation_arities});
      }
      return encoding;
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

BatchBuilder::BatchEncoding SemanticFlatRelationEncoderEngine::encode_batch(
   const std::vector< SemanticFlatRelationInput >& inputs
) const
{
   return impl_->encode_many(std::span{inputs});
}

const SemanticFlatRelationEncoderEngine::Config&
SemanticFlatRelationEncoderEngine::get_config() const
{
   return impl_->config;
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

}  // namespace mifrost
