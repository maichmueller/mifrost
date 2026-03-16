#include "flat_relation_encoder.hpp"

#include <fmt/format.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <mimir/formalism/problem.hpp>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <string_view>
#include <utility>

#include "flat_lgan.hpp"
#include "mifrost/input_handling/batch_input_parser.hpp"
#include "state_fact_iteration.hpp"

namespace mifrost {

namespace {

constexpr std::string_view kEntityNodeType = "entity";
constexpr std::string_view kFlatEntityTypeAttr = "entity_node_type";
constexpr std::string_view kIncludeLGANEdgesAttr = "include_lgan_edges";
constexpr std::string_view kTargetSourcesAttr = "target_sources";
constexpr std::string_view kLGANAnchorSourcesAttr = "lgan_anchor_sources";
constexpr std::string_view kRelationNamesAttr = "relation_names";
constexpr std::string_view kRelationAritiesAttr = "relation_arities";
constexpr std::string_view kRelationSourcesAttr = "relation_sources";
constexpr std::string_view kNodeSizesField = "node_sizes";
constexpr std::string_view kObjectSizesField = "object_sizes";
constexpr std::string_view kObjectIndicesField = "object_indices";
constexpr std::string_view kHistoryEntitySizesField = "history_entity_sizes";
constexpr std::string_view kHistoryEntityIndicesField = "history_entity_indices";
constexpr std::string_view kHistoryEntityDtField = "history_entity_dt";
constexpr std::string_view kTargetEntitySizesField = "target_entity_sizes";
constexpr std::string_view kTargetEntityIndicesField = "target_entity_indices";
constexpr std::string_view kTargetEntityGroupIdsField = "target_entity_group_ids";
constexpr std::string_view kTargetEntityGroupsAttr = "target_entity_groups";
constexpr std::string_view kTargetSizesField = "target_sizes";
constexpr std::string_view kRelationInstanceSizesField = "relation_instance_sizes";
constexpr std::string_view kRelationCountsField = "relation_counts";
constexpr std::string_view kRelationArgsField = "relation_args";
constexpr std::string_view kLGANTNSizesField = "lgan_tn_sizes";
constexpr std::string_view kLGANTNRelationIndicesField = "lgan_tn_relation_indices";
constexpr std::string_view kLGANTNEntityIndicesField = "lgan_tn_entity_indices";
constexpr std::string_view kLGANNNSizesField = "lgan_nn_sizes";
constexpr std::string_view kLGANNNRelationIndicesField = "lgan_nn_relation_indices";
constexpr std::string_view kLGANNNEntityIndicesField = "lgan_nn_entity_indices";
constexpr std::string_view kLGANRRSizesField = "lgan_rr_sizes";
constexpr std::string_view kLGANRRSrcRelationIndicesField = "lgan_rr_src_relation_indices";
constexpr std::string_view kLGANRRDstRelationIndicesField = "lgan_rr_dst_relation_indices";
constexpr std::string_view kLGANTNEdgePosAttr = "lgan_tn_edge_pos";
constexpr std::string_view kLGANNNEdgePosAttr = "lgan_nn_edge_pos";
constexpr std::string_view kLGANRREdgePosAttr = "lgan_rr_edge_pos";
constexpr std::string_view kHistoryRelationSuffix = "[hist]";

GoalInputs default_goal_inputs_for_state(const mimir::search::State& state)
{
   GoalInputs inputs;
   const auto& problem = state.get_problem();
   for(const auto& goal : problem.get_goal_literals< mimir::formalism::StaticTag >()) {
      inputs.append(goal, 0);
   }
   for(const auto& goal : problem.get_goal_literals< mimir::formalism::FluentTag >()) {
      inputs.append(goal, 0);
   }
   for(const auto& goal : problem.get_goal_literals< mimir::formalism::DerivedTag >()) {
      inputs.append(goal, 0);
   }
   return inputs;
}

std::vector< std::string > source_names_for(const std::set< TargetSource >& sources)
{
   std::vector< std::string > out;
   out.reserve(sources.size());
   for(const auto source : kCanonicalTargetSourceOrder) {
      if(sources.contains(source)) {
         out.emplace_back(target_source_group_name(source));
      }
   }
   return out;
}

class RelationSchemaRegistry {
  public:
   struct Entry {
      int arity = 0;
      std::string source;
   };

   void add(std::string name, int arity, std::string source)
   {
      auto [it, inserted] = entries_.try_emplace(std::move(name), Entry{arity, std::move(source)});
      if(not inserted) {
         throw std::invalid_argument(
            "Flat relation schema collision for relation '" + it->first + "'"
         );
      }
   }

   [[nodiscard]] bool contains(const std::string& name) const { return entries_.contains(name); }

   [[nodiscard]] size_t size() const { return entries_.size(); }

   [[nodiscard]] const std::map< std::string, Entry >& entries() const { return entries_; }

   std::map< std::string, Entry > entries_;
};

bool has_lgan_anchor_source(const FlatRelationEncoderEngine::Config& config, TargetSource source)
{
   return config.include_lgan_edges and config.lgan_anchor_sources.contains(source);
}

bool has_anchor_entity_source(const FlatRelationEncoderEngine::Config& config, TargetSource source)
{
   return config.target_sources.contains(source) or has_lgan_anchor_source(config, source);
}

std::optional< TargetSource > anchor_source_for_goal_level(
   const FlatRelationEncoderEngine::Config& config,
   const std::optional< size_t >& goal_level
)
{
   if(goal_level.has_value() and *goal_level > 0) {
      if(has_anchor_entity_source(config, TargetSource::subgoals)) {
         return TargetSource::subgoals;
      }
      return std::nullopt;
   }
   if(has_anchor_entity_source(config, TargetSource::goals)) {
      return TargetSource::goals;
   }
   return std::nullopt;
}

int goal_relation_arity(
   const FlatRelationEncoderEngine& engine,
   int base_arity,
   const std::optional< size_t >& goal_level
)
{
   return base_arity
          + static_cast< int >(
             anchor_source_for_goal_level(engine.get_config(), goal_level).has_value()
          );
}

template < typename GoalTag >
FlatRelationEncoderEngine::TargetEntityKey goal_target_entity_key(
   TargetSource source,
   const mimir::formalism::GroundLiteral< GoalTag >& literal,
   const std::optional< size_t >& goal_level
)
{
   return FlatRelationEncoderEngine::TargetEntityKey{
      .source = source,
      .discriminator = static_cast< int64_t >(state_fact_tag_id< GoalTag >()),
      .primary = static_cast< int64_t >(literal->get_atom()->get_index()),
      .secondary = literal->get_polarity() ? 1 : 0,
      .tertiary = goal_level.has_value() ? static_cast< int64_t >(*goal_level) : -1,
      .quaternary = 0,
   };
}

FlatRelationEncoderEngine::TargetEntityKey action_target_entity_key(
   const mimir::formalism::GroundAction& action
)
{
   return FlatRelationEncoderEngine::TargetEntityKey{
      .source = TargetSource::actions,
      .discriminator = 0,
      .primary = static_cast< int64_t >(action->get_index()),
      .secondary = 0,
      .tertiary = 0,
      .quaternary = 0,
   };
}

template < typename HistoryTag >
FlatRelationEncoderEngine::TargetEntityKey history_target_entity_key(
   int dt,
   size_t entry_idx,
   const mimir::formalism::GroundLiteral< HistoryTag >& literal
)
{
   return FlatRelationEncoderEngine::TargetEntityKey{
      .source = TargetSource::history,
      .discriminator = static_cast< int64_t >(state_fact_tag_id< HistoryTag >()),
      .primary = static_cast< int64_t >(dt),
      .secondary = static_cast< int64_t >(entry_idx),
      .tertiary = static_cast< int64_t >(literal->get_atom()->get_index()),
      .quaternary = literal->get_polarity() ? 1 : 0,
   };
}

template < typename GoalTag >
std::string goal_target_display_name(
   const mimir::formalism::GroundLiteral< GoalTag >& literal,
   const std::optional< size_t >& goal_level
)
{
   if(goal_level.has_value()) {
      return RelationFormatter::format_literal< GoalTag >(literal, GoalLevel(*goal_level));
   }
   return RelationFormatter::format_literal< GoalTag >(literal, std::nullopt);
}

template < typename HistoryTag >
std::string history_target_display_name(
   int dt,
   size_t entry_idx,
   const mimir::formalism::GroundLiteral< HistoryTag >& literal
)
{
   return fmt::format(
      "history:{}#{}:{}",
      dt,
      entry_idx,
      RelationFormatter::format_literal< HistoryTag >(literal, std::nullopt)
   );
}

struct PreparedHistoryEntry {
   int dt = 0;
   size_t entry_idx = 0;
   std::vector< LiteralVariant > literals;
};

std::vector< PreparedHistoryEntry > prepare_history_entries(
   std::span< const FlatRelationEncoderEngine::HistorySubgoal > history_subgoals,
   std::optional< int > history_max_steps
)
{
   std::vector< PreparedHistoryEntry > entries;
   entries.reserve(history_subgoals.size());
   for(const auto& [dt, literals] : history_subgoals) {
      if(dt >= 0) {
         throw std::invalid_argument("history_subgoals expects negative dt values");
      }
      if(history_max_steps.has_value() and std::abs(dt) > *history_max_steps) {
         continue;
      }
      entries.push_back(
         PreparedHistoryEntry{
            .dt = dt,
            .entry_idx = entries.size(),
            .literals = literals,
         }
      );
   }

   std::ranges::stable_sort(entries, [](const auto& lhs, const auto& rhs) {
      return lhs.dt < rhs.dt;
   });
   for(size_t idx = 0; idx < entries.size(); ++idx) {
      entries[idx].entry_idx = idx;
   }

   return entries;
}

template < typename HistoryTag >
std::string
history_relation_name(const mimir::formalism::Predicate< HistoryTag >& predicate, bool polarity)
{
   return RelationFormatter::format_predicate(
      predicate, std::nullopt, std::nullopt, polarity, kHistoryRelationSuffix
   );
}

int history_relation_arity(const FlatRelationEncoderEngine& engine, int base_arity)
{
   return base_arity + 1
          + static_cast< int >(
             has_anchor_entity_source(engine.get_config(), TargetSource::history)
          );
}

template < typename GoalLevelsMap, typename LiteralTag >
std::optional< size_t > goal_level_for(
   const GoalLevelsMap& goal_levels,
   const mimir::formalism::GroundLiteral< LiteralTag >& literal
)
{
   if(const auto it = goal_levels.find(literal); it != goal_levels.end()) {
      return it->second;
   }
   return std::nullopt;
}

template < typename AtomTag >
std::vector< int64_t > local_arg_rows_for_atom(
   const FlatRelationEncoderEngine::EncodingContext& context,
   const mimir::formalism::GroundAtom< AtomTag >& atom
)
{
   std::vector< int64_t > args;
   args.reserve(atom->get_objects().size());
   for(const auto& obj : atom->get_objects()) {
      const auto it = context.entity_index_by_object_id.find(
         static_cast< int64_t >(obj->get_index())
      );
      if(it == context.entity_index_by_object_id.end()) {
         throw std::invalid_argument(
            "Flat relation encoder encountered object not present in entity table: "
            + RelationFormatter::format_object(*obj)
         );
      }
      args.push_back(it->second);
   }
   return args;
}

int64_t lookup_target_entity_index(
   const FlatRelationEncoderEngine::EncodingContext& context,
   const FlatRelationEncoderEngine::TargetEntityKey& key,
   const std::string& missing_message
)
{
   const auto it = context.target_entity_index_by_key.find(key);
   if(it == context.target_entity_index_by_key.end()) {
      throw std::invalid_argument(missing_message);
   }
   return it->second;
}

template < typename GoalTag >
int64_t lookup_goal_target_entity_index(
   const FlatRelationEncoderEngine::EncodingContext& context,
   TargetSource source,
   const mimir::formalism::GroundLiteral< GoalTag >& literal,
   const std::optional< size_t >& goal_level
)
{
   return lookup_target_entity_index(
      context,
      goal_target_entity_key(source, literal, goal_level),
      "Flat relation encoder encountered goal literal without a target entity row: "
         + goal_target_display_name(literal, goal_level)
   );
}

template < typename HistoryTag >
int64_t lookup_history_target_entity_index(
   const FlatRelationEncoderEngine::EncodingContext& context,
   int dt,
   size_t entry_idx,
   const mimir::formalism::GroundLiteral< HistoryTag >& literal
)
{
   return lookup_target_entity_index(
      context,
      history_target_entity_key(dt, entry_idx, literal),
      "Flat relation encoder encountered history literal without a target entity row: "
         + history_target_display_name(dt, entry_idx, literal)
   );
}

std::vector< int64_t > local_arg_rows_for_action(
   const FlatRelationEncoderEngine::EncodingContext& context,
   const mimir::formalism::GroundAction& action
)
{
   std::vector< int64_t > args;
   args.reserve(action->get_objects().size() + 1);

   args.push_back(lookup_target_entity_index(
      context,
      action_target_entity_key(action),
      "Flat relation encoder encountered grounded action without a target entity row: "
         + RelationFormatter::format_action(action)
   ));

   for(const auto& obj : action->get_objects()) {
      const auto object_it = context.entity_index_by_object_id.find(
         static_cast< int64_t >(obj->get_index())
      );
      if(object_it == context.entity_index_by_object_id.end()) {
         throw std::invalid_argument(
            "Flat relation encoder encountered action object not present in entity table: "
            + RelationFormatter::format_object(*obj)
         );
      }
      args.push_back(object_it->second);
   }

   return args;
}

template < typename GoalTag >
std::vector< int64_t > local_arg_rows_for_goal_literal(
   const FlatRelationEncoderEngine& engine,
   const FlatRelationEncoderEngine::EncodingContext& context,
   const mimir::formalism::GroundLiteral< GoalTag >& literal,
   const std::optional< size_t >& goal_level
)
{
   auto args = local_arg_rows_for_atom(context, literal->get_atom());
   if(const auto target_source = anchor_source_for_goal_level(engine.get_config(), goal_level);
      target_source.has_value()) {
      args.insert(
         args.begin(), lookup_goal_target_entity_index(context, *target_source, literal, goal_level)
      );
   }
   return args;
}

template < typename HistoryTag >
std::vector< int64_t > local_arg_rows_for_history_literal(
   const FlatRelationEncoderEngine& engine,
   const FlatRelationEncoderEngine::EncodingContext& context,
   int64_t history_entity_index,
   int dt,
   size_t entry_idx,
   const mimir::formalism::GroundLiteral< HistoryTag >& literal
)
{
   auto args = local_arg_rows_for_atom(context, literal->get_atom());
   args.insert(args.begin(), history_entity_index);
   if(has_anchor_entity_source(engine.get_config(), TargetSource::history)) {
      args.insert(
         args.begin(), lookup_history_target_entity_index(context, dt, entry_idx, literal)
      );
   }
   return args;
}

}  // namespace

uint64_t FlatRelationEncoderEngine::TargetEntityKeyHash::operator()(
   const TargetEntityKey& key
) const noexcept
{
   auto mix = [](uint64_t seed, uint64_t value) {
      seed ^= value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
      return seed;
   };

   uint64_t seed = ankerl::unordered_dense::hash< int64_t >{}(static_cast< int64_t >(key.source));
   seed = mix(seed, ankerl::unordered_dense::hash< int64_t >{}(key.discriminator));
   seed = mix(seed, ankerl::unordered_dense::hash< int64_t >{}(key.primary));
   seed = mix(seed, ankerl::unordered_dense::hash< int64_t >{}(key.secondary));
   seed = mix(seed, ankerl::unordered_dense::hash< int64_t >{}(key.tertiary));
   seed = mix(seed, ankerl::unordered_dense::hash< int64_t >{}(key.quaternary));
   return seed;
}

class FlatRelationEncoderEngine::RelationComponent {
  public:
   virtual ~RelationComponent() = default;
   virtual void declare_schema(
      const FlatRelationEncoderEngine& engine,
      RelationSchemaRegistry& registry
   ) const = 0;
   virtual void emit(
      const FlatRelationEncoderEngine& engine,
      const mimir::search::State& state,
      const GoalInputs& goals,
      const hash_set< uint64_t >& fact_keys,
      const EncodingContext& context,
      FlatRelationSink& sink
   ) const = 0;
};

class FlatRelationEncoderEngine::StateFactsComponent final:
    public FlatRelationEncoderEngine::RelationComponent {
  public:
   void declare_schema(
      const FlatRelationEncoderEngine& engine,
      RelationSchemaRegistry& registry
   ) const override
   {
      for(const auto& spec : engine.predicate_specs_) {
         if(engine.config_.ignore_zero_arity_relations and spec.arity == 0) {
            continue;
         }
         registry.add(spec.name, spec.arity, "state");
      }
   }

   void emit(
      const FlatRelationEncoderEngine& engine,
      const mimir::search::State& state,
      const GoalInputs&,
      const hash_set< uint64_t >&,
      const EncodingContext& context,
      FlatRelationSink& sink
   ) const override
   {
      auto emit_atom = [&](const auto& atom) {
         const int arity = static_cast< int >(atom->get_predicate()->get_arity());
         if(engine.config_.ignore_zero_arity_relations and arity == 0) {
            return;
         }
         const auto relation_id = engine.relation_id_for(
            RelationFormatter::format_predicate(atom->get_predicate())
         );
         const auto args = local_arg_rows_for_atom(context, atom);
         sink.emit(relation_id, args);
      };
      for_each_state_fact_atom(state, engine.config_.include_static, emit_atom);
   }
};

class FlatRelationEncoderEngine::GoalFactsComponent final:
    public FlatRelationEncoderEngine::RelationComponent {
  public:
   void declare_schema(
      const FlatRelationEncoderEngine& engine,
      RelationSchemaRegistry& registry
   ) const override
   {
      if(not engine.config_.goal_derivations.contains(GoalDerivation::plain)) {
         return;
      }
      for(const auto& spec : engine.regular_predicate_specs_) {
         if(engine.config_.ignore_zero_arity_relations and spec.arity == 0) {
            continue;
         }
         for(size_t level = 0; level <= engine.config_.max_goal_level; ++level) {
            const GoalLevel goal_level(level);
            for(bool polarity : {true, false}) {
               registry.add(
                  RelationFormatter::format_predicate(
                     spec.name, goal_level, std::nullopt, polarity
                  ),
                  goal_relation_arity(engine, spec.arity, level),
                  "goal"
               );
            }
         }
         if(engine.config_.support_literals) {
            for(bool polarity : {true, false}) {
               registry.add(
                  RelationFormatter::format_predicate(
                     spec.name, std::nullopt, std::nullopt, polarity
                  ),
                  goal_relation_arity(engine, spec.arity, std::nullopt),
                  "goal"
               );
            }
         }
      }
   }

   void emit(
      const FlatRelationEncoderEngine& engine,
      const mimir::search::State&,
      const GoalInputs& goals,
      const hash_set< uint64_t >&,
      const EncodingContext& context,
      FlatRelationSink& sink
   ) const override
   {
      if(not engine.config_.goal_derivations.contains(GoalDerivation::plain)) {
         return;
      }
      emit_literals(std::span{goals.static_goals}, goals.static_goal_levels, engine, context, sink);
      emit_literals(std::span{goals.fluent_goals}, goals.fluent_goal_levels, engine, context, sink);
      emit_literals(
         std::span{goals.derived_goals}, goals.derived_goal_levels, engine, context, sink
      );
   }

  private:
   template < typename GoalTag >
   static void emit_literals(
      std::span< const mimir::formalism::GroundLiteral< GoalTag > > literals,
      const auto& goal_levels,
      const FlatRelationEncoderEngine& engine,
      const EncodingContext& context,
      FlatRelationSink& sink
   )
   {
      for(const auto& literal : literals) {
         const auto predicate = literal->get_atom()->get_predicate();
         const int arity = static_cast< int >(predicate->get_arity());
         if(engine.config_.ignore_zero_arity_relations and arity == 0) {
            continue;
         }

         const auto level = goal_level_for(goal_levels, literal);
         std::string relation_name;
         if(level.has_value()) {
            relation_name = RelationFormatter::format_predicate(
               predicate, GoalLevel(*level), std::nullopt, literal->get_polarity()
            );
         } else {
            relation_name = RelationFormatter::format_predicate(
               predicate, std::nullopt, std::nullopt, literal->get_polarity()
            );
         }

         const auto relation_id = engine.relation_id_for(relation_name);
         const auto args = local_arg_rows_for_goal_literal(engine, context, literal, level);
         sink.emit(relation_id, args);
      }
   }
};

class FlatRelationEncoderEngine::GoalDerivationComponent final:
    public FlatRelationEncoderEngine::RelationComponent {
  public:
   void declare_schema(
      const FlatRelationEncoderEngine& engine,
      RelationSchemaRegistry& registry
   ) const override
   {
      if(not has_non_plain_goal_derivations(engine.config_.goal_derivations)) {
         return;
      }
      for(const auto& spec : engine.regular_predicate_specs_) {
         if(engine.config_.ignore_zero_arity_relations and spec.arity == 0) {
            continue;
         }
         for(const auto derivation : engine.config_.goal_derivations) {
            if(derivation == GoalDerivation::plain) {
               continue;
            }
            for(size_t level = 0; level <= engine.config_.max_goal_level; ++level) {
               const GoalLevel goal_level(level);
               for(bool polarity : {true, false}) {
                  registry.add(
                     RelationFormatter::format_predicate(
                        spec.name, goal_level, derivation, polarity
                     ),
                     spec.arity,
                     "goal_derivation"
                  );
               }
            }
            if(engine.config_.support_literals) {
               for(bool polarity : {true, false}) {
                  registry.add(
                     RelationFormatter::format_predicate(
                        spec.name, std::nullopt, derivation, polarity
                     ),
                     spec.arity,
                     "goal_derivation"
                  );
               }
            }
         }
      }
   }

   void emit(
      const FlatRelationEncoderEngine& engine,
      const mimir::search::State&,
      const GoalInputs& goals,
      const hash_set< uint64_t >& fact_keys,
      const EncodingContext& context,
      FlatRelationSink& sink
   ) const override
   {
      if(not has_non_plain_goal_derivations(engine.config_.goal_derivations)) {
         return;
      }
      emit_literals(
         std::span{goals.static_goals}, goals.static_goal_levels, fact_keys, engine, context, sink
      );
      emit_literals(
         std::span{goals.fluent_goals}, goals.fluent_goal_levels, fact_keys, engine, context, sink
      );
      emit_literals(
         std::span{goals.derived_goals}, goals.derived_goal_levels, fact_keys, engine, context, sink
      );
   }

  private:
   template < typename GoalTag >
   static void emit_literals(
      std::span< const mimir::formalism::GroundLiteral< GoalTag > > literals,
      const auto& goal_levels,
      const hash_set< uint64_t >& fact_keys,
      const FlatRelationEncoderEngine& engine,
      const EncodingContext& context,
      FlatRelationSink& sink
   )
   {
      for(const auto& literal : literals) {
         const auto predicate = literal->get_atom()->get_predicate();
         const int arity = static_cast< int >(predicate->get_arity());
         if(engine.config_.ignore_zero_arity_relations and arity == 0) {
            continue;
         }

         const uint64_t fact_key = state_fact_key_for_atom(literal->get_atom());
         const bool satisfied = fact_keys.contains(fact_key) == literal->get_polarity();
         const GoalDerivation satisfaction = satisfied ? GoalDerivation::satisfied
                                                       : GoalDerivation::unsatisfied;
         if(not engine.config_.goal_derivations.contains(satisfaction)) {
            continue;
         }

         const auto level = goal_level_for(goal_levels, literal);
         std::string relation_name;
         if(level.has_value()) {
            relation_name = RelationFormatter::format_predicate(
               predicate, GoalLevel(*level), satisfaction, literal->get_polarity()
            );
         } else {
            relation_name = RelationFormatter::format_predicate(
               predicate, std::nullopt, satisfaction, literal->get_polarity()
            );
         }

         const auto relation_id = engine.relation_id_for(relation_name);
         const auto args = local_arg_rows_for_atom(context, literal->get_atom());
         sink.emit(relation_id, args);
      }
   }
};

class FlatRelationEncoderEngine::GroundActionsComponent final:
    public FlatRelationEncoderEngine::RelationComponent {
  public:
   void declare_schema(
      const FlatRelationEncoderEngine& engine,
      RelationSchemaRegistry& registry
   ) const override
   {
      for(const auto& spec : engine.action_specs_) {
         registry.add(spec.name, spec.arity, "action");
      }
   }

   void emit(
      const FlatRelationEncoderEngine& engine,
      const mimir::search::State&,
      const GoalInputs&,
      const hash_set< uint64_t >&,
      const EncodingContext& context,
      FlatRelationSink& sink
   ) const override
   {
      for(const auto& action : context.unique_actions) {
         const auto relation_id = engine.relation_id_for(
            RelationFormatter::format_action_schema(*action->get_action())
         );
         const auto args = local_arg_rows_for_action(context, action);
         sink.emit(relation_id, args);
      }
   }
};

class FlatRelationEncoderEngine::HistoryFactsComponent final:
    public FlatRelationEncoderEngine::RelationComponent {
  public:
   void declare_schema(
      const FlatRelationEncoderEngine& engine,
      RelationSchemaRegistry& registry
   ) const override
   {
      for(const auto& spec : engine.predicate_specs_) {
         if(engine.config_.ignore_zero_arity_relations and spec.arity == 0) {
            continue;
         }
         for(bool polarity : {true, false}) {
            registry.add(
               RelationFormatter::format_predicate(
                  spec.name, std::nullopt, std::nullopt, polarity, kHistoryRelationSuffix
               ),
               history_relation_arity(engine, spec.arity),
               "history"
            );
         }
      }
   }

   void emit(
      const FlatRelationEncoderEngine& engine,
      const mimir::search::State&,
      const GoalInputs&,
      const hash_set< uint64_t >&,
      const EncodingContext& context,
      FlatRelationSink& sink
   ) const override
   {
      for(const auto& entry : context.history_entries) {
         for(const auto& literal_variant : entry.literals) {
            std::visit(
               [&]< typename HistoryTag >(
                  const mimir::formalism::GroundLiteral< HistoryTag >& literal
               ) {
                  const auto predicate = literal->get_atom()->get_predicate();
                  const int arity = static_cast< int >(predicate->get_arity());
                  if(engine.config_.ignore_zero_arity_relations and arity == 0) {
                     return;
                  }
                  const auto relation_id = engine.relation_id_for(
                     history_relation_name(predicate, literal->get_polarity())
                  );
                  const auto args = local_arg_rows_for_history_literal(
                     engine, context, entry.entity_index, entry.dt, entry.entry_idx, literal
                  );
                  sink.emit(relation_id, args);
               },
               literal_variant
            );
         }
      }
   }
};

FlatRelationEncoderEngine::FlatRelationEncoderEngine(const mimir::formalism::DomainImpl& domain)
    : FlatRelationEncoderEngine(domain, Config{})
{
}

FlatRelationEncoderEngine::FlatRelationEncoderEngine(
   const mimir::formalism::DomainImpl& domain,
   Config config
)
    : domain_(domain), config_(std::move(config))
{
   validate_config();
   initialize_from_domain();
}

FlatRelationEncoderEngine::FlatRelationEncoderEngine(mimir::formalism::Domain domain)
    : FlatRelationEncoderEngine(std::move(domain), Config{})
{
}

FlatRelationEncoderEngine::FlatRelationEncoderEngine(mimir::formalism::Domain domain, Config config)
    : domain_holder_(std::move(domain)), domain_(*domain_holder_), config_(std::move(config))
{
   validate_config();
   initialize_from_domain();
}

FlatRelationEncoderEngine::~FlatRelationEncoderEngine() = default;

void FlatRelationEncoderEngine::validate_config() const
{
   auto validate_sources = [](std::span< const TargetSource > sources,
                              std::string_view field_name) {
      for(const auto source : sources) {
         if(source == TargetSource::actions or source == TargetSource::goals
            or source == TargetSource::subgoals or source == TargetSource::history) {
            continue;
         }
         throw std::invalid_argument(
            fmt::format(
               "FlatRelationEncoder currently supports {}={{'action', 'goal', "
               "'subgoal', 'history'}} only; 'state' is reserved for the upcoming flat "
               "successor/horizon encoders",
               field_name
            )
         );
      }
   };
   std::vector< TargetSource > target_sources(
      config_.target_sources.begin(), config_.target_sources.end()
   );
   validate_sources(std::span{target_sources}, "target_sources");
   std::vector< TargetSource > lgan_anchor_sources(
      config_.lgan_anchor_sources.begin(), config_.lgan_anchor_sources.end()
   );
   validate_sources(std::span{lgan_anchor_sources}, "lgan_anchor_sources");
}

void FlatRelationEncoderEngine::initialize_from_domain()
{
   predicate_specs_.clear();
   regular_predicate_specs_.clear();
   action_specs_.clear();

   RelationDictConfig rel_config;
   rel_config.max_goal_level = static_cast< int >(config_.max_goal_level);
   rel_config.support_literals = config_.support_literals;
   rel_config.goal_derivations = config_.goal_derivations;

   std::vector< mimir::formalism::Action > actions;
   actions.assign(domain_.get_actions().begin(), domain_.get_actions().end());

   const auto top_type_predicates = rel_config.top_type_predicates;
   auto collect_predicates = [&]< typename Tag >(Tag) {
      for(const auto predicate : domain_.get_predicates< Tag >()) {
         PredicateSpec spec{
            .name = RelationFormatter::format_predicate(predicate),
            .arity = static_cast< int >(predicate->get_arity()),
         };
         predicate_specs_.push_back(spec);
         if(not top_type_predicates.contains(predicate->get_name())) {
            regular_predicate_specs_.push_back(spec);
         }
      }
   };
   collect_predicates(mimir::formalism::StaticTag{});
   collect_predicates(mimir::formalism::FluentTag{});
   collect_predicates(mimir::formalism::DerivedTag{});

   auto predicate_order = [](const PredicateSpec& lhs, const PredicateSpec& rhs) {
      return lhs.name < rhs.name;
   };
   std::ranges::sort(predicate_specs_, predicate_order);
   std::ranges::sort(regular_predicate_specs_, predicate_order);

   for(const auto& action : actions) {
      action_specs_.push_back(
         PredicateSpec{
            .name = RelationFormatter::format_action_schema(*action),
            .arity = static_cast< int >(action->get_arity()) + 1,
         }
      );
   }
   std::ranges::sort(action_specs_, predicate_order);

   target_entity_group_names_.clear();
   target_entity_group_ids_.clear();
   target_metadata_group_names_.clear();
   target_metadata_group_ids_.clear();

   auto append_group = [](std::vector< std::string >& group_names,
                          std::map< TargetSource, int64_t >& group_ids,
                          TargetSource source) {
      if(group_ids.contains(source)) {
         return;
      }
      const auto next_id = static_cast< int64_t >(group_names.size());
      group_ids.emplace(source, next_id);
      group_names.emplace_back(target_source_group_name(source));
   };

   for(const auto source : kCanonicalTargetSourceOrder) {
      if(source == TargetSource::actions or has_anchor_entity_source(source)) {
         if(source == TargetSource::states) {
            continue;
         }
         append_group(target_entity_group_names_, target_entity_group_ids_, source);
      }
      if(has_target_source(source)) {
         append_group(target_metadata_group_names_, target_metadata_group_ids_, source);
      }
   }

   components_.clear();
   components_.push_back(std::make_unique< StateFactsComponent >());
   components_.push_back(std::make_unique< GoalFactsComponent >());
   components_.push_back(std::make_unique< GoalDerivationComponent >());
   components_.push_back(std::make_unique< GroundActionsComponent >());
   components_.push_back(std::make_unique< HistoryFactsComponent >());

   rebuild_schema();
}

void FlatRelationEncoderEngine::rebuild_schema()
{
   RelationSchemaRegistry registry;
   for(const auto& component : components_) {
      component->declare_schema(*this, registry);
   }

   relation_names_.clear();
   relation_arities_.clear();
   relation_sources_.clear();
   relation_name_to_id_.clear();

   relation_names_.reserve(registry.size());
   relation_arities_.reserve(registry.size());
   relation_sources_.reserve(registry.size());
   relation_name_to_id_.reserve(registry.size());

   std::map< std::string, int > relation_dict_arity;
   for(const auto& [name, entry] : registry.entries()) {
      relation_name_to_id_.emplace(name, static_cast< int >(relation_names_.size()));
      relation_names_.push_back(name);
      relation_arities_.push_back(entry.arity);
      relation_sources_.push_back(entry.source);
      relation_dict_arity.emplace(name, entry.arity);
   }

   if(relation_names_.empty()) {
      throw std::invalid_argument(
         "FlatRelationEncoderEngine did not derive any relation types for this domain/config"
      );
   }

   relation_dict_ = RelationDict(
      std::move(relation_dict_arity),
      static_cast< int >(config_.max_goal_level),
      config_.support_literals,
      config_.goal_derivations
   );
}

void FlatRelationEncoderEngine::prepare_builder(BatchBuilder& builder) const
{
   builder.set_graph_kind("flat");
   builder.set_schema_flag("flat_relations", true);
   builder.set_graph_attr(std::string(kFlatEntityTypeAttr), std::string(kEntityNodeType));
   builder.set_graph_attr(
      std::string(kIncludeLGANEdgesAttr), static_cast< int64_t >(config_.include_lgan_edges)
   );
   builder.set_graph_attr(
      std::string(kTargetSourcesAttr), source_names_for(config_.target_sources)
   );
   builder.set_graph_attr(
      std::string(kLGANAnchorSourcesAttr), source_names_for(config_.lgan_anchor_sources)
   );
   builder.set_graph_attr(std::string(kRelationNamesAttr), relation_names_);
   builder.set_graph_attr(std::string(kRelationAritiesAttr), relation_arities_);
   builder.set_graph_attr(std::string(kRelationSourcesAttr), relation_sources_);
   builder.set_graph_attr(std::string(kTargetEntityGroupsAttr), target_entity_group_names_);
   builder.set_graph_attr(std::string(kTargetSymbolPrefixAttr), config_.target_symbol_prefix);
   builder.set_graph_attr(std::string(kLGANTNEdgePosAttr), config_.lgan_tn_edge_pos);
   builder.set_graph_attr(std::string(kLGANNNEdgePosAttr), config_.lgan_nn_edge_pos);
   builder.set_graph_attr(std::string(kLGANRREdgePosAttr), config_.lgan_rr_edge_pos);

   builder.register_field(
      std::string(kNodeSizesField),
      GraphFieldSpec{
         .dtype = GraphFieldDType::I64,
         .mode = GraphFieldMode::STACK,
         .dim = 1,
      }
   );
   builder.register_field(
      std::string(kObjectSizesField),
      GraphFieldSpec{
         .dtype = GraphFieldDType::I64,
         .mode = GraphFieldMode::STACK,
         .dim = 1,
      }
   );
   builder.register_field(
      std::string(kObjectIndicesField),
      GraphFieldSpec{
         .dtype = GraphFieldDType::I64,
         .mode = GraphFieldMode::CAT,
         .dim = 1,
         .inc = GraphFieldInc{
            .kind = GraphFieldInc::Kind::NODE_OFFSET,
            .node_type = std::string(kEntityNodeType),
         },
      }
   );
   builder.register_field(
      std::string(kHistoryEntitySizesField),
      GraphFieldSpec{
         .dtype = GraphFieldDType::I64,
         .mode = GraphFieldMode::STACK,
         .dim = 1,
      }
   );
   builder.register_field(
      std::string(kHistoryEntityIndicesField),
      GraphFieldSpec{
         .dtype = GraphFieldDType::I64,
         .mode = GraphFieldMode::CAT,
         .dim = 1,
         .inc = GraphFieldInc{
            .kind = GraphFieldInc::Kind::NODE_OFFSET,
            .node_type = std::string(kEntityNodeType),
         },
      }
   );
   builder.register_field(
      std::string(kHistoryEntityDtField),
      GraphFieldSpec{
         .dtype = GraphFieldDType::I64,
         .mode = GraphFieldMode::CAT,
         .dim = 1,
      }
   );
   builder.register_field(
      std::string(kTargetEntitySizesField),
      GraphFieldSpec{
         .dtype = GraphFieldDType::I64,
         .mode = GraphFieldMode::STACK,
         .dim = 1,
      }
   );
   builder.register_field(
      std::string(kTargetEntityIndicesField),
      GraphFieldSpec{
         .dtype = GraphFieldDType::I64,
         .mode = GraphFieldMode::CAT,
         .dim = 1,
         .inc = GraphFieldInc{
            .kind = GraphFieldInc::Kind::NODE_OFFSET,
            .node_type = std::string(kEntityNodeType),
         },
      }
   );
   builder.register_field(
      std::string(kTargetEntityGroupIdsField),
      GraphFieldSpec{
         .dtype = GraphFieldDType::I64,
         .mode = GraphFieldMode::CAT,
         .dim = 1,
      }
   );
   if(supports_target_metadata()) {
      builder.register_field(
         std::string(kTargetSizesField),
         GraphFieldSpec{
            .dtype = GraphFieldDType::I64,
            .mode = GraphFieldMode::STACK,
            .dim = 1,
         }
      );
      const TargetMetadataEmitConfig target_emit_config{
         .position_node_type_id = std::string(kEntityNodeType),
         .symbol_prefix = config_.target_symbol_prefix,
         .include_depth = false,
         .include_group = true,
         .include_names = false,
         .groups = target_metadata_group_names_,
         .parent_relation = std::nullopt,
      };
      register_target_fields(builder, target_emit_config);
      builder.set_graph_attr(std::string(kTargetGroupsAttr), target_metadata_group_names_);
      builder.set_graph_attr(std::string(kTargetSymbolPrefixAttr), config_.target_symbol_prefix);
   }
   builder.register_field(
      std::string(kRelationCountsField),
      GraphFieldSpec{
         .dtype = GraphFieldDType::I64,
         .mode = GraphFieldMode::STACK,
         .dim = static_cast< int >(relation_names_.size()),
      }
   );
   builder.register_field(
      std::string(kRelationInstanceSizesField),
      GraphFieldSpec{
         .dtype = GraphFieldDType::I64,
         .mode = GraphFieldMode::STACK,
         .dim = 1,
      }
   );
   builder.register_field(
      std::string(kRelationArgsField),
      GraphFieldSpec{
         .dtype = GraphFieldDType::I64,
         .mode = GraphFieldMode::CAT,
         .dim = 1,
         .inc = GraphFieldInc{
            .kind = GraphFieldInc::Kind::NODE_OFFSET,
            .node_type = std::string(kEntityNodeType),
         },
      }
   );
   if(config_.include_lgan_edges) {
      builder.register_field(
         std::string(kLGANTNSizesField),
         GraphFieldSpec{
            .dtype = GraphFieldDType::I64,
            .mode = GraphFieldMode::STACK,
            .dim = 1,
         }
      );
      builder.register_field(
         std::string(kLGANTNRelationIndicesField),
         GraphFieldSpec{
            .dtype = GraphFieldDType::I64,
            .mode = GraphFieldMode::CAT,
            .dim = 1,
            .inc = GraphFieldInc{
               .kind = GraphFieldInc::Kind::FIELD_OFFSET,
               .field_key = std::string(kRelationInstanceSizesField),
            },
         }
      );
      builder.register_field(
         std::string(kLGANTNEntityIndicesField),
         GraphFieldSpec{
            .dtype = GraphFieldDType::I64,
            .mode = GraphFieldMode::CAT,
            .dim = 1,
            .inc = GraphFieldInc{
               .kind = GraphFieldInc::Kind::NODE_OFFSET,
               .node_type = std::string(kEntityNodeType),
            },
         }
      );
      builder.register_field(
         std::string(kLGANNNSizesField),
         GraphFieldSpec{
            .dtype = GraphFieldDType::I64,
            .mode = GraphFieldMode::STACK,
            .dim = 1,
         }
      );
      builder.register_field(
         std::string(kLGANNNRelationIndicesField),
         GraphFieldSpec{
            .dtype = GraphFieldDType::I64,
            .mode = GraphFieldMode::CAT,
            .dim = 1,
            .inc = GraphFieldInc{
               .kind = GraphFieldInc::Kind::FIELD_OFFSET,
               .field_key = std::string(kRelationInstanceSizesField),
            },
         }
      );
      builder.register_field(
         std::string(kLGANNNEntityIndicesField),
         GraphFieldSpec{
            .dtype = GraphFieldDType::I64,
            .mode = GraphFieldMode::CAT,
            .dim = 1,
            .inc = GraphFieldInc{
               .kind = GraphFieldInc::Kind::NODE_OFFSET,
               .node_type = std::string(kEntityNodeType),
            },
         }
      );
      builder.register_field(
         std::string(kLGANRRSizesField),
         GraphFieldSpec{
            .dtype = GraphFieldDType::I64,
            .mode = GraphFieldMode::STACK,
            .dim = 1,
         }
      );
      builder.register_field(
         std::string(kLGANRRSrcRelationIndicesField),
         GraphFieldSpec{
            .dtype = GraphFieldDType::I64,
            .mode = GraphFieldMode::CAT,
            .dim = 1,
            .inc = GraphFieldInc{
               .kind = GraphFieldInc::Kind::FIELD_OFFSET,
               .field_key = std::string(kRelationInstanceSizesField),
            },
         }
      );
      builder.register_field(
         std::string(kLGANRRDstRelationIndicesField),
         GraphFieldSpec{
            .dtype = GraphFieldDType::I64,
            .mode = GraphFieldMode::CAT,
            .dim = 1,
            .inc = GraphFieldInc{
               .kind = GraphFieldInc::Kind::FIELD_OFFSET,
               .field_key = std::string(kRelationInstanceSizesField),
            },
         }
      );
   }
}

FlatRelationEncoderEngine::EncodingContext FlatRelationEncoderEngine::make_context(
   const mimir::search::State& state,
   const GoalInputs& goals,
   std::span< const mimir::formalism::GroundAction > actions,
   std::span< const HistorySubgoal > history_subgoals,
   std::optional< int > history_max_steps
) const
{
   EncodingContext context;
   const auto& objects = state.get_problem().get_problem_and_domain_objects();
   std::vector< mimir::formalism::Object > ordered(objects.begin(), objects.end());
   std::ranges::sort(ordered, [](const auto& lhs, const auto& rhs) {
      return lhs->get_index() < rhs->get_index();
   });
   const auto prepared_history = prepare_history_entries(history_subgoals, history_max_steps);

   context.entity_names.reserve(ordered.size() + actions.size() + prepared_history.size());
   context.object_names.reserve(ordered.size());
   context.object_indices.reserve(ordered.size());
   context.entity_index_by_object_id.reserve(ordered.size());
   context.history_entity_indices.reserve(prepared_history.size());
   context.history_entity_dt.reserve(prepared_history.size());
   context.target_entity_indices.reserve(ordered.size() + actions.size() + prepared_history.size());
   context.target_entity_group_ids.reserve(actions.size() + prepared_history.size());
   context.target_entity_index_by_key.reserve(actions.size() + prepared_history.size());
   context.history_entries.reserve(prepared_history.size());
   context.unique_actions.reserve(actions.size());
   if(supports_target_metadata()) {
      const size_t total_goal_literals = goals.static_goals.size() + goals.fluent_goals.size()
                                         + goals.derived_goals.size();
      size_t total_history_literals = 0;
      for(const auto& entry : prepared_history) {
         total_history_literals += entry.literals.size();
      }
      context.target_columns.reserve(
         total_goal_literals + actions.size() + total_history_literals,
         /*include_depth=*/false,
         /*include_group=*/true
      );
   }

   for(size_t i = 0; i < ordered.size(); ++i) {
      const auto& obj = ordered[i];
      const int64_t local_index = static_cast< int64_t >(i);
      context.entity_index_by_object_id.emplace(
         static_cast< int64_t >(obj->get_index()), local_index
      );
      const std::string object_name = RelationFormatter::format_object(*obj);
      context.entity_names.push_back(object_name);
      context.object_names.push_back(object_name);
      context.object_indices.push_back(local_index);
   }

   auto ensure_target_entity =
      [&](
         const TargetEntityKey& key,
         TargetSource source,
         const std::string& name,
         const std::optional< mimir::formalism::GroundAction >& action = std::nullopt
      ) {
         const auto it = context.target_entity_index_by_key.find(key);
         if(it != context.target_entity_index_by_key.end()) {
            return it->second;
         }
         const int64_t local_index = static_cast< int64_t >(context.entity_names.size());
         context.target_entity_index_by_key.emplace(key, local_index);
         context.entity_names.push_back(name);
         context.target_entity_indices.push_back(local_index);
         context.target_entity_group_ids.push_back(target_entity_group_id(source));
         if(action.has_value()) {
            context.unique_actions.push_back(*action);
         }
         return local_index;
      };

   auto append_history_entity = [&](int dt, size_t entry_idx) {
      const int64_t local_index = static_cast< int64_t >(context.entity_names.size());
      context.entity_names.push_back(fmt::format("history:{}#{}", dt, entry_idx));
      context.history_entity_indices.push_back(local_index);
      context.history_entity_dt.push_back(static_cast< int64_t >(dt));
      return local_index;
   };

   auto append_target_row = [&](TargetSource source, int64_t position, const std::string& name) {
      const int64_t target_index = static_cast< int64_t >(context.target_columns.size());
      append_target_candidate_row(
         context.target_columns,
         TargetCandidateRow{
            .position = position,
            .index = target_index,
            .candidate_id = target_index,
            .depth = std::nullopt,
            .group_id = target_metadata_group_id(source),
            .name = name,
         },
         TargetCandidateAppendConfig{
            .include_depth = false,
            .include_group = true,
         }
      );
   };

   auto collect_goal_targets =
      [&]< typename GoalTag >(
         std::span< const mimir::formalism::GroundLiteral< GoalTag > > literals,
         const auto& goal_levels,
         TargetSource source
      ) {
         for(const auto& literal : literals) {
            const auto predicate = literal->get_atom()->get_predicate();
            const int arity = static_cast< int >(predicate->get_arity());
            if(config_.ignore_zero_arity_relations and arity == 0) {
               continue;
            }
            const auto goal_level = goal_level_for(goal_levels, literal);
            const bool is_subgoal = goal_level.has_value() and *goal_level > 0;
            if((source == TargetSource::goals and is_subgoal)
               or (source == TargetSource::subgoals and not is_subgoal)) {
               continue;
            }
            const auto display_name = goal_target_display_name(literal, goal_level);
            const auto local_index = ensure_target_entity(
               goal_target_entity_key(source, literal, goal_level), source, display_name
            );
            if(has_target_source(source)) {
               append_target_row(source, local_index, display_name);
            }
         }
      };

   if(has_anchor_entity_source(TargetSource::goals)) {
      collect_goal_targets(
         std::span{goals.static_goals}, goals.static_goal_levels, TargetSource::goals
      );
      collect_goal_targets(
         std::span{goals.fluent_goals}, goals.fluent_goal_levels, TargetSource::goals
      );
      collect_goal_targets(
         std::span{goals.derived_goals}, goals.derived_goal_levels, TargetSource::goals
      );
   }

   if(has_anchor_entity_source(TargetSource::subgoals)) {
      collect_goal_targets(
         std::span{goals.static_goals}, goals.static_goal_levels, TargetSource::subgoals
      );
      collect_goal_targets(
         std::span{goals.fluent_goals}, goals.fluent_goal_levels, TargetSource::subgoals
      );
      collect_goal_targets(
         std::span{goals.derived_goals}, goals.derived_goal_levels, TargetSource::subgoals
      );
   }

   for(const auto& action : actions) {
      const auto action_name = RelationFormatter::format_action(action);
      const auto local_index = ensure_target_entity(
         action_target_entity_key(action), TargetSource::actions, action_name, action
      );
      if(has_target_source(TargetSource::actions)) {
         append_target_row(TargetSource::actions, local_index, action_name);
      }
   }

   for(const auto& entry : prepared_history) {
      const int64_t history_entity_index = append_history_entity(entry.dt, entry.entry_idx);
      context.history_entries.push_back(
         EncodingContext::HistoryEntry{
            .dt = entry.dt,
            .entry_idx = entry.entry_idx,
            .entity_index = history_entity_index,
            .literals = entry.literals,
         }
      );
   }

   if(has_anchor_entity_source(TargetSource::history)) {
      for(const auto& entry : context.history_entries) {
         for(const auto& literal_variant : entry.literals) {
            std::visit(
               [&]< typename HistoryTag >(
                  const mimir::formalism::GroundLiteral< HistoryTag >& literal
               ) {
                  const auto predicate = literal->get_atom()->get_predicate();
                  const int arity = static_cast< int >(predicate->get_arity());
                  if(config_.ignore_zero_arity_relations and arity == 0) {
                     return;
                  }
                  const auto display_name = history_target_display_name(
                     entry.dt, entry.entry_idx, literal
                  );
                  const auto local_index = ensure_target_entity(
                     history_target_entity_key(entry.dt, entry.entry_idx, literal),
                     TargetSource::history,
                     display_name
                  );
                  if(has_target_source(TargetSource::history)) {
                     append_target_row(TargetSource::history, local_index, display_name);
                  }
               },
               literal_variant
            );
         }
      }
   }

   return context;
}

int FlatRelationEncoderEngine::relation_id_for(const std::string& name) const
{
   const auto it = relation_name_to_id_.find(name);
   if(it == relation_name_to_id_.end()) {
      throw std::invalid_argument("Unknown flat relation name '" + name + "'");
   }
   return it->second;
}

bool FlatRelationEncoderEngine::has_target_source(TargetSource source) const
{
   return config_.target_sources.contains(source);
}

bool FlatRelationEncoderEngine::has_lgan_anchor_source(TargetSource source) const
{
   return config_.include_lgan_edges and config_.lgan_anchor_sources.contains(source);
}

bool FlatRelationEncoderEngine::has_anchor_entity_source(TargetSource source) const
{
   return has_target_source(source) or has_lgan_anchor_source(source);
}

bool FlatRelationEncoderEngine::supports_target_metadata() const
{
   return not target_metadata_group_names_.empty();
}

int64_t FlatRelationEncoderEngine::target_entity_group_id(TargetSource source) const
{
   const auto it = target_entity_group_ids_.find(source);
   if(it == target_entity_group_ids_.end()) {
      throw std::invalid_argument(
         "FlatRelationEncoder does not define a target-entity group for source '"
         + std::string(target_source_group_name(source)) + "'"
      );
   }
   return it->second;
}

int64_t FlatRelationEncoderEngine::target_metadata_group_id(TargetSource source) const
{
   const auto it = target_metadata_group_ids_.find(source);
   if(it == target_metadata_group_ids_.end()) {
      throw std::invalid_argument(
         "FlatRelationEncoder does not define target metadata for source '"
         + std::string(target_source_group_name(source)) + "'"
      );
   }
   return it->second;
}

void FlatRelationEncoderEngine::encode_default_goals(
   const mimir::search::State& state,
   std::span< const mimir::formalism::GroundAction > actions,
   std::span< const HistorySubgoal > history_subgoals,
   std::optional< int > history_max_steps,
   BatchBuilder& builder
)
{
   encode_impl(
      state,
      default_goal_inputs_for_state(state),
      actions,
      history_subgoals,
      history_max_steps,
      builder
   );
}

void FlatRelationEncoderEngine::encode(const mimir::search::State& state, BatchBuilder& builder)
{
   encode_default_goals(state, {}, {}, std::nullopt, builder);
}

void FlatRelationEncoderEngine::encode(
   const mimir::search::State& state,
   std::span< const mimir::formalism::GroundAction > actions,
   BatchBuilder& builder
)
{
   encode_default_goals(state, actions, {}, std::nullopt, builder);
}

void FlatRelationEncoderEngine::encode(
   const mimir::search::State& state,
   const GoalInputs& goals,
   BatchBuilder& builder
)
{
   encode_impl(state, goals, {}, {}, std::nullopt, builder);
}

void FlatRelationEncoderEngine::encode(
   const mimir::search::State& state,
   const GoalInputs& goals,
   std::span< const mimir::formalism::GroundAction > actions,
   BatchBuilder& builder
)
{
   encode_impl(state, goals, actions, {}, std::nullopt, builder);
}

void FlatRelationEncoderEngine::encode(
   const mimir::search::State& state,
   const GoalInputs& goals,
   std::span< const mimir::formalism::GroundAction > actions,
   std::span< const HistorySubgoal > history_subgoals,
   std::optional< int > history_max_steps,
   BatchBuilder& builder
)
{
   encode_impl(state, goals, actions, history_subgoals, history_max_steps, builder);
}

void FlatRelationEncoderEngine::encode_impl(
   const mimir::search::State& state,
   const GoalInputs& goals,
   std::span< const mimir::formalism::GroundAction > actions,
   std::span< const HistorySubgoal > history_subgoals,
   std::optional< int > history_max_steps,
   BatchBuilder& builder,
   std::vector< std::string >* batch_target_names,
   bool prepare_builder_once
)
{
   if(prepare_builder_once) {
      prepare_builder(builder);
   }
   const auto context = make_context(state, goals, actions, history_subgoals, history_max_steps);
   FlatRelationSink sink(relation_names_.size(), config_.include_lgan_edges);

   hash_set< uint64_t > fact_keys;
   collect_state_fact_keys(state, config_.include_static, fact_keys);

   for(const auto& component : components_) {
      component->emit(*this, state, goals, fact_keys, context, sink);
   }

   std::vector< float > zeros(context.entity_names.size(), 0.0f);
   builder.add_node_features(
      std::string(kEntityNodeType), "x", std::span< const float >(zeros.data(), zeros.size()), 1
   );

   if(config_.export_node_names) {
      builder.set_node_names(std::string(kEntityNodeType), context.entity_names);
      builder.set_object_names(context.object_names);
   }

   const int64_t node_size = static_cast< int64_t >(context.entity_names.size());
   const int64_t object_size = static_cast< int64_t >(context.object_indices.size());
   const int64_t history_entity_size = static_cast< int64_t >(
      context.history_entity_indices.size()
   );
   const int64_t target_entity_size = static_cast< int64_t >(context.target_entity_indices.size());
   builder.set_field(std::string(kNodeSizesField), std::span< const int64_t >(&node_size, 1));
   builder.set_field(std::string(kObjectSizesField), std::span< const int64_t >(&object_size, 1));
   builder.set_field(
      std::string(kObjectIndicesField),
      std::span< const int64_t >(context.object_indices.data(), context.object_indices.size())
   );
   builder.set_field(
      std::string(kHistoryEntitySizesField), std::span< const int64_t >(&history_entity_size, 1)
   );
   builder.set_field(
      std::string(kHistoryEntityIndicesField),
      std::span< const int64_t >(
         context.history_entity_indices.data(), context.history_entity_indices.size()
      )
   );
   builder.set_field(
      std::string(kHistoryEntityDtField),
      std::span< const int64_t >(context.history_entity_dt.data(), context.history_entity_dt.size())
   );
   builder.set_field(
      std::string(kTargetEntitySizesField), std::span< const int64_t >(&target_entity_size, 1)
   );
   builder.set_field(
      std::string(kTargetEntityIndicesField),
      std::span< const int64_t >(
         context.target_entity_indices.data(), context.target_entity_indices.size()
      )
   );
   builder.set_field(
      std::string(kTargetEntityGroupIdsField),
      std::span< const int64_t >(
         context.target_entity_group_ids.data(), context.target_entity_group_ids.size()
      )
   );
   if(supports_target_metadata()) {
      const int64_t target_size = static_cast< int64_t >(context.target_columns.size());
      builder.set_field(
         std::string(kTargetSizesField), std::span< const int64_t >(&target_size, 1)
      );
      const TargetMetadataEmitConfig target_emit_config{
         .position_node_type_id = std::string(kEntityNodeType),
         .symbol_prefix = config_.target_symbol_prefix,
         .include_depth = false,
         .include_group = true,
         .include_names = false,
         .groups = target_metadata_group_names_,
         .parent_relation = std::nullopt,
      };
      set_target_fields(builder, context.target_columns, target_emit_config);
      set_target_graph_attrs(builder, context.target_columns, target_emit_config);
      if(config_.export_node_names) {
         if(batch_target_names != nullptr) {
            if(not context.target_columns.names.empty()) {
               batch_target_names->insert(
                  batch_target_names->end(),
                  context.target_columns.names.begin(),
                  context.target_columns.names.end()
               );
            }
         } else if(context.target_columns.names.empty()) {
            builder.set_graph_attr(std::string(kTargetNamesAttr), std::vector< std::string >{});
         } else {
            builder.add_lazy_target_names(std::span(context.target_columns.names));
         }
      }
   }
   builder.set_field(
      std::string(kRelationCountsField),
      std::span< const int64_t >(sink.relation_counts().data(), sink.relation_counts().size())
   );
   const int64_t relation_instance_size = sink.relation_instance_count();
   builder.set_field(
      std::string(kRelationInstanceSizesField),
      std::span< const int64_t >(&relation_instance_size, 1)
   );
   builder.set_field(
      std::string(kRelationArgsField),
      std::span< const int64_t >(sink.relation_args().data(), sink.relation_args().size())
   );
   if(config_.include_lgan_edges) {
      if(context.target_entity_indices.empty()) {
         throw std::invalid_argument(
            "FlatRelationEncoder include_lgan_edges=true requires LGAN anchor entity rows, "
            "but none were encoded. Encode explicit actions or enable anchor-row-emitting "
            "lgan_anchor_sources/target_sources such as 'goal', 'subgoal', or 'history'."
         );
      }
      const auto lgan = build_flat_lgan(sink, std::span{context.target_entity_indices});
      const int64_t tn_size = static_cast< int64_t >(lgan.tn_relation_indices.size());
      const int64_t nn_size = static_cast< int64_t >(lgan.nn_relation_indices.size());
      const int64_t rr_size = static_cast< int64_t >(lgan.rr_src_relation_indices.size());
      builder.set_field(std::string(kLGANTNSizesField), std::span< const int64_t >(&tn_size, 1));
      builder.set_field(
         std::string(kLGANTNRelationIndicesField),
         std::span< const int64_t >(
            lgan.tn_relation_indices.data(), lgan.tn_relation_indices.size()
         )
      );
      builder.set_field(
         std::string(kLGANTNEntityIndicesField),
         std::span< const int64_t >(lgan.tn_entity_indices.data(), lgan.tn_entity_indices.size())
      );
      builder.set_field(std::string(kLGANNNSizesField), std::span< const int64_t >(&nn_size, 1));
      builder.set_field(
         std::string(kLGANNNRelationIndicesField),
         std::span< const int64_t >(
            lgan.nn_relation_indices.data(), lgan.nn_relation_indices.size()
         )
      );
      builder.set_field(
         std::string(kLGANNNEntityIndicesField),
         std::span< const int64_t >(lgan.nn_entity_indices.data(), lgan.nn_entity_indices.size())
      );
      builder.set_field(std::string(kLGANRRSizesField), std::span< const int64_t >(&rr_size, 1));
      builder.set_field(
         std::string(kLGANRRSrcRelationIndicesField),
         std::span< const int64_t >(
            lgan.rr_src_relation_indices.data(), lgan.rr_src_relation_indices.size()
         )
      );
      builder.set_field(
         std::string(kLGANRRDstRelationIndicesField),
         std::span< const int64_t >(
            lgan.rr_dst_relation_indices.data(), lgan.rr_dst_relation_indices.size()
         )
      );
   }
}

BatchBuilder::BatchEncoding FlatRelationEncoderEngine::encode_batch(
   const batch_input::parsed::FlatBatchInputs& inputs,
   std::optional< int > history_max_steps
)
{
   BatchBuilder builder;
   builder.set_graph_kind("flat");
   prepare_builder(builder);

   const size_t state_count = inputs.states.states.size();
   std::vector< std::string > batch_target_names;
   for(size_t idx = 0; idx < state_count; ++idx) {
      const auto& state_entry = inputs.states.states[idx];
      const auto& goals_entry = inputs.goals.at(idx);
      const auto& actions_entry = inputs.actions.at(idx);
      const auto& subgoal_layers_entry = inputs.subgoal_layers.at(idx);
      const auto& history_entry = inputs.history_subgoals.at(idx);

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

      const auto actions_span = actions_entry.has_value()
                                   ? std::span< const mimir::formalism::GroundAction >(
                                        *actions_entry
                                     )
                                   : std::span< const mimir::formalism::GroundAction >{};
      const auto history_span = history_entry.has_value()
                                   ? std::span< const HistorySubgoal >(*history_entry)
                                   : std::span< const HistorySubgoal >{};
      encode_impl(
         state_entry.state,
         goal_inputs,
         actions_span,
         history_span,
         history_max_steps,
         builder,
         &batch_target_names,
         /*prepare_builder_once=*/false
      );
      builder.next_graph();
   }

   if(supports_target_metadata()) {
      if(config_.export_node_names) {
         if(batch_target_names.empty()) {
            builder.set_graph_attr(std::string(kTargetNamesAttr), std::vector< std::string >{});
         } else {
            builder.add_lazy_target_names(std::span(batch_target_names));
         }
      }
      builder.set_graph_attr(std::string(kTargetGroupsAttr), target_metadata_group_names_);
      builder.set_graph_attr(std::string(kTargetSymbolPrefixAttr), config_.target_symbol_prefix);
   }

   return builder.build();
}

}  // namespace mifrost
