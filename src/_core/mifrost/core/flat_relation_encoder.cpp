#include "flat_relation_encoder.hpp"

#include <fmt/format.h>

#include <algorithm>
#include <mimir/formalism/problem.hpp>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <string_view>
#include <utility>

#include "mifrost/input_handling/batch_input_parser.hpp"

namespace mifrost {

namespace {

constexpr std::string_view kEntityNodeType = "entity";
constexpr std::string_view kFlatEntityTypeAttr = "entity_node_type";
constexpr std::string_view kRelationNamesAttr = "relation_names";
constexpr std::string_view kRelationAritiesAttr = "relation_arities";
constexpr std::string_view kRelationSourcesAttr = "relation_sources";
constexpr std::string_view kNodeSizesField = "node_sizes";
constexpr std::string_view kObjectSizesField = "object_sizes";
constexpr std::string_view kObjectIndicesField = "object_indices";
constexpr std::string_view kTargetEntitySizesField = "target_entity_sizes";
constexpr std::string_view kTargetEntityIndicesField = "target_entity_indices";
constexpr std::string_view kTargetSizesField = "target_sizes";
constexpr std::string_view kRelationCountsField = "relation_counts";
constexpr std::string_view kRelationArgsField = "relation_args";

template < typename LiteralTag >
uint32_t fact_tag_id()
{
   if constexpr(std::is_same_v< LiteralTag, mimir::formalism::StaticTag >) {
      return 1U;
   }
   if constexpr(std::is_same_v< LiteralTag, mimir::formalism::FluentTag >) {
      return 2U;
   }
   return 3U;
}

uint64_t pack_u32_u32(uint32_t hi, uint32_t lo)
{
   return (static_cast< uint64_t >(hi) << 32) | static_cast< uint64_t >(lo);
}

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

TargetColumns build_action_target_columns(
   std::span< const mimir::formalism::GroundAction > actions,
   const FlatRelationEncoderEngine::EncodingContext& context
)
{
   TargetColumns columns;
   columns.reserve(actions.size(), /*include_depth=*/false, /*include_group=*/true);
   for(size_t action_pos = 0; action_pos < actions.size(); ++action_pos) {
      const auto& action = actions[action_pos];
      const auto it = context.target_entity_index_by_action_id.find(
         static_cast< int64_t >(action->get_index())
      );
      if(it == context.target_entity_index_by_action_id.end()) {
         throw std::invalid_argument(
            "Flat relation encoder encountered grounded action without a target entity row: "
            + RelationFormatter::format_action(action)
         );
      }
      columns.append(
         TargetRecord{
            .position = it->second,
            .index = static_cast< int64_t >(action_pos),
            .candidate_id = static_cast< int64_t >(action_pos),
            .depth = std::nullopt,
            .group_id = 0,
            .name = RelationFormatter::format_action(action),
         },
         /*include_depth=*/false,
         /*include_group=*/true
      );
   }
   return columns;
}

class RelationSchemaRegistry {
  public:
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

   [[nodiscard]] int arity_for(const std::string& name) const { return entries_.at(name).arity; }

   [[nodiscard]] const std::string& source_for(const std::string& name) const
   {
      return entries_.at(name).source;
   }

   [[nodiscard]] size_t size() const { return entries_.size(); }

  private:
   struct Entry {
      int arity = 0;
      std::string source;
   };

   hash_map< std::string, Entry > entries_;
};

class FlatRelationSink {
  public:
   explicit FlatRelationSink(size_t relation_count)
       : relation_counts_(relation_count, 0), relation_args_by_relation_(relation_count)
   {
   }

   void emit(int relation_id, std::span< const int64_t > args)
   {
      if(relation_id < 0 or static_cast< size_t >(relation_id) >= relation_counts_.size()) {
         throw std::invalid_argument("FlatRelationSink relation id out of range");
      }
      relation_counts_[static_cast< size_t >(relation_id)] += 1;
      auto& bucket = relation_args_by_relation_[static_cast< size_t >(relation_id)];
      bucket.insert(bucket.end(), args.begin(), args.end());
      relation_args_dirty_ = true;
   }

   [[nodiscard]] const std::vector< int64_t >& relation_counts() const { return relation_counts_; }

   [[nodiscard]] const std::vector< int64_t >& relation_args() const
   {
      if(relation_args_dirty_) {
         relation_args_.clear();
         size_t total_slots = 0;
         for(const auto& bucket : relation_args_by_relation_) {
            total_slots += bucket.size();
         }
         relation_args_.reserve(total_slots);
         for(const auto& bucket : relation_args_by_relation_) {
            relation_args_.insert(relation_args_.end(), bucket.begin(), bucket.end());
         }
         relation_args_dirty_ = false;
      }
      return relation_args_;
   }

  private:
   std::vector< int64_t > relation_counts_;
   mutable std::vector< std::vector< int64_t > > relation_args_by_relation_;
   mutable std::vector< int64_t > relation_args_;
   mutable bool relation_args_dirty_ = false;
};

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

std::vector< int64_t > local_arg_rows_for_action(
   const FlatRelationEncoderEngine::EncodingContext& context,
   const mimir::formalism::GroundAction& action
)
{
   std::vector< int64_t > args;
   args.reserve(action->get_objects().size() + 1);

   const auto action_it = context.target_entity_index_by_action_id.find(
      static_cast< int64_t >(action->get_index())
   );
   if(action_it == context.target_entity_index_by_action_id.end()) {
      throw std::invalid_argument(
         "Flat relation encoder encountered grounded action without a target entity row: "
         + RelationFormatter::format_action(action)
      );
   }
   args.push_back(action_it->second);

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

}  // namespace

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
      const auto& problem = state.get_problem();
      const auto& repos = problem.get_repositories();

      auto emit_atom = [&]< typename Tag >(mimir::formalism::GroundAtom< Tag > atom) {
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

      if(engine.config_.include_static) {
         for(const auto& literal : problem.get_initial_literals< mimir::formalism::StaticTag >()) {
            if(not literal->get_polarity()) {
               continue;
            }
            emit_atom(literal->get_atom());
         }
      }

      const auto fluent_atoms = repos.get_ground_atoms_from_indices< mimir::formalism::FluentTag >(
         state.get_atoms< mimir::formalism::FluentTag >()
      );
      for(const auto& atom : fluent_atoms) {
         emit_atom(atom);
      }

      const auto derived_atoms = repos
                                    .get_ground_atoms_from_indices< mimir::formalism::DerivedTag >(
                                       state.get_atoms< mimir::formalism::DerivedTag >()
                                    );
      for(const auto& atom : derived_atoms) {
         emit_atom(atom);
      }
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
                  spec.arity,
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
                  spec.arity,
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
         const auto args = local_arg_rows_for_atom(context, literal->get_atom());
         sink.emit(relation_id, args);
      }
   }
};

class FlatRelationEncoderEngine::GoalSatisfactionComponent final:
    public FlatRelationEncoderEngine::RelationComponent {
  public:
   void declare_schema(
      const FlatRelationEncoderEngine& engine,
      RelationSchemaRegistry& registry
   ) const override
   {
      for(const auto& spec : engine.regular_predicate_specs_) {
         if(engine.config_.ignore_zero_arity_relations and spec.arity == 0) {
            continue;
         }
         for(const auto satisfaction : engine.config_.goal_satisfaction_derivations) {
            for(size_t level = 0; level <= engine.config_.max_goal_level; ++level) {
               const GoalLevel goal_level(level);
               for(bool polarity : {true, false}) {
                  registry.add(
                     RelationFormatter::format_predicate(
                        spec.name, goal_level, satisfaction, polarity
                     ),
                     spec.arity,
                     "goal_satisfaction"
                  );
               }
            }
            if(engine.config_.support_literals) {
               for(bool polarity : {true, false}) {
                  registry.add(
                     RelationFormatter::format_predicate(
                        spec.name, std::nullopt, satisfaction, polarity
                     ),
                     spec.arity,
                     "goal_satisfaction"
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

         const uint64_t fact_key = pack_u32_u32(
            static_cast< uint32_t >(literal->get_atom()->get_index()), fact_tag_id< GoalTag >()
         );
         const bool satisfied = fact_keys.contains(fact_key) == literal->get_polarity();
         const GoalSatisfaction satisfaction = satisfied ? GoalSatisfaction::satisfied
                                                         : GoalSatisfaction::unsatisfied;
         if(not engine.config_.goal_satisfaction_derivations.contains(satisfaction)) {
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
      for(const auto& action : context.target_entity_actions) {
         const auto relation_id = engine.relation_id_for(
            RelationFormatter::format_action_schema(*action->get_action())
         );
         const auto args = local_arg_rows_for_action(context, action);
         sink.emit(relation_id, args);
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
   for(const auto source : config_.target_sources) {
      if(source != TargetSource::Actions) {
         throw std::invalid_argument(
            "FlatRelationEncoder currently supports target_sources={'action'} only"
         );
      }
   }
}

void FlatRelationEncoderEngine::initialize_from_domain()
{
   predicate_specs_.clear();
   regular_predicate_specs_.clear();
   action_specs_.clear();

   RelationDictConfig rel_config;
   rel_config.max_goal_level = static_cast< int >(config_.max_goal_level);
   rel_config.support_literals = config_.support_literals;
   rel_config.goal_satisfaction_derivations = config_.goal_satisfaction_derivations;

   std::vector< mimir::formalism::Action > actions;
   actions.assign(domain_.get_actions().begin(), domain_.get_actions().end());
   relation_dict_ = RelationDict(domain_, actions, rel_config, 0, 1);

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

   components_.clear();
   components_.push_back(std::make_unique< StateFactsComponent >());
   components_.push_back(std::make_unique< GoalFactsComponent >());
   components_.push_back(std::make_unique< GoalSatisfactionComponent >());
   components_.push_back(std::make_unique< GroundActionsComponent >());

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

   for(const auto& [name, arity] : relation_dict_.arity) {
      if(not registry.contains(name)) {
         continue;
      }
      if(registry.arity_for(name) != arity) {
         throw std::invalid_argument(
            "Flat relation schema arity mismatch for relation '" + name + "'"
         );
      }
      relation_name_to_id_.emplace(name, static_cast< int >(relation_names_.size()));
      relation_names_.push_back(name);
      relation_arities_.push_back(arity);
      relation_sources_.push_back(registry.source_for(name));
   }

   if(relation_names_.empty()) {
      throw std::invalid_argument(
         "FlatRelationEncoderEngine did not derive any relation types for this domain/config"
      );
   }
}

void FlatRelationEncoderEngine::prepare_builder(BatchBuilder& builder) const
{
   builder.set_graph_kind("homo");
   builder.set_schema_flag("flat_relations", true);
   builder.set_graph_attr(std::string(kFlatEntityTypeAttr), std::string(kEntityNodeType));
   builder.set_graph_attr(std::string(kRelationNamesAttr), relation_names_);
   builder.set_graph_attr(std::string(kRelationAritiesAttr), relation_arities_);
   builder.set_graph_attr(std::string(kRelationSourcesAttr), relation_sources_);

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
   if(has_target_source(TargetSource::Actions)) {
      builder.register_field(
         std::string(kTargetSizesField),
         GraphFieldSpec{
            .dtype = GraphFieldDType::I64,
            .mode = GraphFieldMode::STACK,
            .dim = 1,
         }
      );
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
}

FlatRelationEncoderEngine::EncodingContext FlatRelationEncoderEngine::make_context(
   const mimir::search::State& state,
   std::span< const mimir::formalism::GroundAction > actions
) const
{
   EncodingContext context;
   const auto& objects = state.get_problem().get_problem_and_domain_objects();
   std::vector< mimir::formalism::Object > ordered(objects.begin(), objects.end());
   std::ranges::sort(ordered, [](const auto& lhs, const auto& rhs) {
      return lhs->get_index() < rhs->get_index();
   });

   context.entity_names.reserve(ordered.size() + actions.size());
   context.object_names.reserve(ordered.size());
   context.object_indices.reserve(ordered.size());
   context.entity_index_by_object_id.reserve(ordered.size());
   context.target_entity_indices.reserve(actions.size());
   context.target_entity_index_by_action_id.reserve(actions.size());
   context.target_entity_actions.reserve(actions.size());

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

   hash_set< int64_t > seen_action_ids;
   seen_action_ids.reserve(actions.size());
   for(const auto& action : actions) {
      const int64_t action_id = static_cast< int64_t >(action->get_index());
      if(not seen_action_ids.emplace(action_id).second) {
         continue;
      }
      const int64_t local_index = static_cast< int64_t >(context.entity_names.size());
      context.target_entity_index_by_action_id.emplace(action_id, local_index);
      context.entity_names.push_back(RelationFormatter::format_action(action));
      context.target_entity_indices.push_back(local_index);
      context.target_entity_actions.push_back(action);
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

void FlatRelationEncoderEngine::encode_default_goals(
   const mimir::search::State& state,
   std::span< const mimir::formalism::GroundAction > actions,
   BatchBuilder& builder
)
{
   encode_impl(state, default_goal_inputs_for_state(state), actions, builder);
}

void FlatRelationEncoderEngine::encode(const mimir::search::State& state, BatchBuilder& builder)
{
   encode_default_goals(state, {}, builder);
}

void FlatRelationEncoderEngine::encode(
   const mimir::search::State& state,
   std::span< const mimir::formalism::GroundAction > actions,
   BatchBuilder& builder
)
{
   encode_default_goals(state, actions, builder);
}

void FlatRelationEncoderEngine::encode(
   const mimir::search::State& state,
   const GoalInputs& goals,
   BatchBuilder& builder
)
{
   encode_impl(state, goals, {}, builder);
}

void FlatRelationEncoderEngine::encode(
   const mimir::search::State& state,
   const GoalInputs& goals,
   std::span< const mimir::formalism::GroundAction > actions,
   BatchBuilder& builder
)
{
   encode_impl(state, goals, actions, builder);
}

void FlatRelationEncoderEngine::encode_impl(
   const mimir::search::State& state,
   const GoalInputs& goals,
   std::span< const mimir::formalism::GroundAction > actions,
   BatchBuilder& builder
)
{
   prepare_builder(builder);
   const auto context = make_context(state, actions);
   FlatRelationSink sink(relation_names_.size());

   hash_set< uint64_t > fact_keys;
   const auto& problem = state.get_problem();
   const auto& repos = problem.get_repositories();

   auto collect_fact_key = [&]< typename Tag >(mimir::formalism::GroundAtom< Tag > atom) {
      fact_keys.insert(
         pack_u32_u32(static_cast< uint32_t >(atom->get_index()), fact_tag_id< Tag >())
      );
   };

   if(config_.include_static) {
      for(const auto& literal : problem.get_initial_literals< mimir::formalism::StaticTag >()) {
         if(not literal->get_polarity()) {
            continue;
         }
         collect_fact_key(literal->get_atom());
      }
   }

   const auto fluent_atoms = repos.get_ground_atoms_from_indices< mimir::formalism::FluentTag >(
      state.get_atoms< mimir::formalism::FluentTag >()
   );
   for(const auto& atom : fluent_atoms) {
      collect_fact_key(atom);
   }

   const auto derived_atoms = repos.get_ground_atoms_from_indices< mimir::formalism::DerivedTag >(
      state.get_atoms< mimir::formalism::DerivedTag >()
   );
   for(const auto& atom : derived_atoms) {
      collect_fact_key(atom);
   }

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
   const int64_t target_entity_size = static_cast< int64_t >(context.target_entity_indices.size());
   builder.set_field(std::string(kNodeSizesField), std::span< const int64_t >(&node_size, 1));
   builder.set_field(std::string(kObjectSizesField), std::span< const int64_t >(&object_size, 1));
   builder.set_field(
      std::string(kObjectIndicesField),
      std::span< const int64_t >(context.object_indices.data(), context.object_indices.size())
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
   if(has_target_source(TargetSource::Actions)) {
      const auto target_columns = build_action_target_columns(actions, context);
      const int64_t target_size = static_cast< int64_t >(target_columns.size());
      builder.set_field(
         std::string(kTargetSizesField), std::span< const int64_t >(&target_size, 1)
      );
      const TargetMetadataEmitConfig target_emit_config{
         .position_node_type_id = std::string(kEntityNodeType),
         .symbol_prefix = config_.target_symbol_prefix,
         .include_depth = false,
         .include_group = true,
         .groups = {std::string(target_source_group_name(TargetSource::Actions))},
         .parent_relation = std::nullopt,
      };
      emit_target_metadata(builder, target_columns, target_emit_config);
   }
   builder.set_field(
      std::string(kRelationCountsField),
      std::span< const int64_t >(sink.relation_counts().data(), sink.relation_counts().size())
   );
   builder.set_field(
      std::string(kRelationArgsField),
      std::span< const int64_t >(sink.relation_args().data(), sink.relation_args().size())
   );
}

BatchBuilder::BatchEncoding FlatRelationEncoderEngine::encode_batch(
   const batch_input::parsed::FlatBatchInputs& inputs
)
{
   BatchBuilder builder;
   builder.set_graph_kind("homo");
   prepare_builder(builder);

   const size_t state_count = inputs.states.states.size();
   std::vector< std::string > batch_target_names;
   for(size_t idx = 0; idx < state_count; ++idx) {
      const auto& state_entry = inputs.states.states[idx];
      const auto& goals_entry = inputs.goals.at(idx);
      const auto& actions_entry = inputs.actions.at(idx);
      const auto& subgoal_layers_entry = inputs.subgoal_layers.at(idx);

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
      if(has_target_source(TargetSource::Actions)) {
         batch_target_names.reserve(batch_target_names.size() + actions_span.size());
         for(const auto& action : actions_span) {
            batch_target_names.push_back(RelationFormatter::format_action(action));
         }
      }
      encode_impl(state_entry.state, goal_inputs, actions_span, builder);
      builder.next_graph();
   }

   if(has_target_source(TargetSource::Actions)) {
      builder.set_graph_attr(std::string(kTargetNamesAttr), std::move(batch_target_names));
      builder.set_graph_attr(
         std::string(kTargetGroupsAttr),
         std::vector< std::string >{std::string(target_source_group_name(TargetSource::Actions))}
      );
      builder.set_graph_attr(std::string(kTargetSymbolPrefixAttr), config_.target_symbol_prefix);
   }

   return builder.build();
}

}  // namespace mifrost
