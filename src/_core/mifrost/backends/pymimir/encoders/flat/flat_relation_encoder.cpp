#include "mifrost/backends/pymimir/encoders/flat/flat_relation_encoder.hpp"

#include <algorithm>
#include <ranges>
#include <type_traits>
#include <utility>

#include "mifrost/backends/pymimir/semantic_views.hpp"
#include "mifrost/core/encoders/flat/semantic_flat_relation_view_bridge.hpp"
#include "mifrost/input_handling/batch_input_parser.hpp"

namespace mifrost {

struct FlatRelationEncoderEngine::SemanticImpl {
   std::unique_ptr< pymimir::SemanticProblemAdapter > problem_adapter;
   std::unique_ptr< SemanticFlatRelationEncoderEngine > encoder;
   const mimir::formalism::ProblemImpl* problem = nullptr;

   SemanticImpl(const mimir::formalism::DomainImpl& domain, const Config& config)
   {
      auto context = std::make_shared< SemanticTaskContext >();
      append_schema(domain, *context);
      encoder = std::make_unique< SemanticFlatRelationEncoderEngine >(std::move(context), config);
   }

   void ensure_problem(const mimir::search::State& state, const Config& config)
   {
      const auto* problem_value = &state.get_problem();
      if(problem != nullptr) {
         if(problem != problem_value) {
            throw std::invalid_argument(
               "FlatRelationEncoder state belongs to a different planning problem"
            );
         }
         return;
      }
      problem = problem_value;
      problem_adapter = std::make_unique< pymimir::SemanticProblemAdapter >(*problem);
      encoder = std::make_unique< SemanticFlatRelationEncoderEngine >(
         problem_adapter->get_task_context(), config
      );
   }

   [[nodiscard]] SemanticFlatRelationInput make_input(
      const mimir::search::State& state,
      const GoalInputs* goals,
      std::span< const mimir::formalism::GroundAction > actions,
      std::span< const FlatRelationEncoderEngine::HistorySubgoal > history,
      std::optional< int > history_max_steps,
      const Config& config
   )
   {
      ensure_problem(state, config);
      auto result = goals == nullptr ? problem_adapter->make_input(state)
                                     : problem_adapter->make_input(state, *goals);
      pymimir::views::Context view_context(state.get_problem());
      for(const auto& action : actions) {
         using NativeAction = std::remove_cvref_t< decltype(action) >;
         result.actions.push_back(
            canonical::materialize_semantic_action(
               pymimir::views::GroundActionView< NativeAction >{action, view_context}
            )
         );
      }
      result.history_max_steps = history_max_steps;
      for(const auto& [dt, literals] : history) {
         SemanticHistoryEntry entry;
         entry.dt = dt;
         for(const auto& literal : literals) {
            std::visit(
               [&](const auto& value) {
                  entry.literals.push_back(materialize_history_literal(value, view_context));
               },
               literal
            );
         }
         result.history.push_back(std::move(entry));
      }
      return result;
   }

   [[nodiscard]] SemanticFlatRelationSink make_sink(
      const mimir::search::State& state,
      const GoalInputs* goals,
      std::span< const mimir::formalism::GroundAction > actions,
      std::span< const FlatRelationEncoderEngine::HistorySubgoal > history,
      std::optional< int > history_max_steps,
      const Config& config
   )
   {
      ensure_problem(state, config);
      auto result = goals == nullptr ? problem_adapter->make_sink(state, actions)
                                     : problem_adapter->make_sink(state, *goals, actions);
      pymimir::views::Context view_context(state.get_problem());
      for(const auto& [dt, literals] : history) {
         SemanticHistoryEntry entry;
         entry.dt = dt;
         for(const auto& literal : literals) {
            std::visit(
               [&](const auto& value) {
                  entry.literals.push_back(materialize_history_literal(value, view_context));
               },
               literal
            );
         }
         result.history.push_back(std::move(entry));
      }
      result.history_max_steps = history_max_steps;
      return result;
   }

  private:
   template < typename Tag >
   static SemanticLiteral materialize_history_literal(
      const mimir::formalism::GroundLiteral< Tag >& literal,
      const pymimir::views::Context& view_context
   )
   {
      constexpr auto category = [] {
         if constexpr(std::is_same_v< Tag, mimir::formalism::StaticTag >) {
            return pymimir::views::Category::static_predicate;
         } else if constexpr(std::is_same_v< Tag, mimir::formalism::FluentTag >) {
            return pymimir::views::Category::fluent;
         } else {
            return pymimir::views::Category::derived;
         }
      }();
      return canonical::materialize_semantic_literal(
         pymimir::views::LiteralView< mimir::formalism::GroundLiteral< Tag >, category >{
            literal, view_context
         }
      );
   }

   static void
   append_schema(const mimir::formalism::DomainImpl& domain, SemanticTaskContext& context)
   {
      const auto append_predicates = [&]< typename Tag >(Tag, SemanticPredicateCategory category) {
         auto predicates = domain.template get_predicates< Tag >();
         std::ranges::sort(predicates, [](const auto lhs, const auto rhs) {
            return std::tuple{lhs->get_name(), lhs->get_arity(), lhs->get_index()}
                   < std::tuple{rhs->get_name(), rhs->get_arity(), rhs->get_index()};
         });
         for(const auto predicate : predicates) {
            context.predicates.push_back(
               SemanticPredicateSpec{
                  category,
                  std::string(predicate->get_name()),
                  static_cast< int64_t >(predicate->get_arity())
               }
            );
         }
      };
      append_predicates(mimir::formalism::StaticTag{}, SemanticPredicateCategory::static_predicate);
      append_predicates(mimir::formalism::FluentTag{}, SemanticPredicateCategory::fluent);
      append_predicates(mimir::formalism::DerivedTag{}, SemanticPredicateCategory::derived);

      auto actions = domain.get_actions();
      std::ranges::sort(actions, [](const auto lhs, const auto rhs) {
         return std::tuple{lhs->get_name(), lhs->get_arity(), lhs->get_index()}
                < std::tuple{rhs->get_name(), rhs->get_arity(), rhs->get_index()};
      });
      for(const auto action : actions) {
         context.actions.push_back(
            SemanticActionSpec{
               std::string(action->get_name()), static_cast< int64_t >(action->get_arity())
            }
         );
      }
   }
};

namespace {
void validate_config(const FlatRelationEncoderConfig& config)
{
   const auto validate_sources = [](const std::ranges::range auto& sources,
                                    std::string_view field_name) {
      for(const TargetSource source : sources) {
         if(source == TargetSource::actions or source == TargetSource::goals
            or source == TargetSource::subgoals or source == TargetSource::history) {
            continue;
         }
         throw std::invalid_argument(
            std::string("FlatRelationEncoder currently supports ") + std::string(field_name)
            + "={'action', 'goal', 'subgoal', 'history'} only; 'state' is reserved for the "
              "upcoming flat successor/horizon encoders"
         );
      }
   };
   validate_sources(config.target_sources, "target_sources");
   validate_sources(config.lgan_anchor_sources, "lgan_anchor_sources");
}
}  // namespace

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
   validate_config(config_);
   std::vector< mimir::formalism::Action > actions(
      domain_.get_actions().begin(), domain_.get_actions().end()
   );
   RelationDictConfig relation_config;
   relation_config.max_goal_level = static_cast< int >(config_.max_goal_level);
   relation_config.support_literals = config_.support_literals;
   relation_config.goal_derivations = config_.goal_derivations;
   relation_dict_ = build_pymimir_relation_dict(domain_, actions, relation_config);
   semantic_ = std::make_unique< SemanticImpl >(domain_, config_);
   const auto& names = semantic_->encoder->get_relation_names();
   const auto& arities = semantic_->encoder->get_relation_arities();
   relation_dict_.arity.clear();
   for(size_t index = 0; index < names.size(); ++index) {
      relation_dict_.arity[names[index]] = static_cast< int >(arities[index]);
   }
}

FlatRelationEncoderEngine::FlatRelationEncoderEngine(mimir::formalism::Domain domain)
    : FlatRelationEncoderEngine(std::move(domain), Config{})
{
}

FlatRelationEncoderEngine::FlatRelationEncoderEngine(mimir::formalism::Domain domain, Config config)
    : domain_holder_(std::move(domain)), domain_(*domain_holder_), config_(std::move(config))
{
   validate_config(config_);
   std::vector< mimir::formalism::Action > actions(
      domain_.get_actions().begin(), domain_.get_actions().end()
   );
   RelationDictConfig relation_config;
   relation_config.max_goal_level = static_cast< int >(config_.max_goal_level);
   relation_config.support_literals = config_.support_literals;
   relation_config.goal_derivations = config_.goal_derivations;
   relation_dict_ = build_pymimir_relation_dict(domain_, actions, relation_config);
   semantic_ = std::make_unique< SemanticImpl >(domain_, config_);
   const auto& names = semantic_->encoder->get_relation_names();
   const auto& arities = semantic_->encoder->get_relation_arities();
   relation_dict_.arity.clear();
   for(size_t index = 0; index < names.size(); ++index) {
      relation_dict_.arity[names[index]] = static_cast< int >(arities[index]);
   }
}

FlatRelationEncoderEngine::~FlatRelationEncoderEngine() = default;

void FlatRelationEncoderEngine::encode_default_goals(
   const mimir::search::State& state,
   std::span< const mimir::formalism::GroundAction > actions,
   std::span< const HistorySubgoal > history_subgoals,
   std::optional< int > history_max_steps,
   BatchBuilder& builder
)
{
   semantic_->ensure_problem(state, config_);
   if(not history_subgoals.empty() or history_max_steps.has_value()) {
      semantic_->encoder->encode(
         semantic_->make_sink(
            state, nullptr, actions, history_subgoals, history_max_steps, config_
         ),
         builder
      );
      return;
   }
   semantic_->encoder->encode(
      semantic_->make_sink(state, nullptr, actions, {}, std::nullopt, config_), builder
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
   std::vector< std::string >*,
   bool
)
{
   semantic_->ensure_problem(state, config_);
   semantic_->encoder->encode(
      semantic_->make_sink(state, &goals, actions, history_subgoals, history_max_steps, config_),
      builder
   );
}

BatchBuilder::BatchEncoding FlatRelationEncoderEngine::encode_batch(
   const batch_input::parsed::FlatBatchInputs& inputs,
   std::optional< int > history_max_steps
)
{
   const size_t state_count = inputs.states.states.size();
   if(state_count == 0) {
      return semantic_->encoder->encode_batch(std::vector< SemanticFlatRelationSink >{});
   }
   semantic_->ensure_problem(inputs.states.states.front().state, config_);
   std::vector< SemanticFlatRelationSink > entries;
   entries.reserve(state_count);
   for(size_t idx = 0; idx < state_count; ++idx) {
      const auto& state_entry = inputs.states.states[idx];
      const auto& goals_entry = inputs.goals.at(idx);
      const auto& actions_entry = inputs.actions.at(idx);
      const auto& subgoal_layers_entry = inputs.subgoal_layers.at(idx);
      const auto& history_entry = inputs.history_subgoals.at(idx);

      GoalInputs goals;
      if(goals_entry.has_value()) {
         const auto* layers = subgoal_layers_entry.has_value() ? &*subgoal_layers_entry : nullptr;
         goals = batch_input::compose_goal_inputs(*goals_entry, layers);
      } else {
         goals = batch_input::default_goal_inputs_for_batch_state(state_entry);
         if(subgoal_layers_entry.has_value()) {
            size_t level = 1;
            for(const auto& layer : *subgoal_layers_entry) {
               goals.extend(layer, level++);
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
      entries.push_back(semantic_->make_sink(
         state_entry.state, &goals, actions_span, history_span, history_max_steps, config_
      ));
   }
   return semantic_->encoder->encode_batch(entries);
}

void FlatRelationEncoderEngine::finalize_batch_encoding(BatchBuilder::BatchEncoding& encoding) const
{
   semantic_->encoder->finalize_batch_encoding(encoding);
}

const RelationDict& FlatRelationEncoderEngine::get_relation_dict() const
{
   return relation_dict_;
}
const std::vector< std::string >& FlatRelationEncoderEngine::get_relation_names() const
{
   return semantic_->encoder->get_relation_names();
}
const std::vector< int64_t >& FlatRelationEncoderEngine::get_relation_arities() const
{
   return semantic_->encoder->get_relation_arities();
}
const std::vector< std::string >& FlatRelationEncoderEngine::get_relation_sources() const
{
   return semantic_->encoder->get_relation_sources();
}
const std::vector< int64_t >& FlatRelationEncoderEngine::get_relation_logical_arities() const
{
   return semantic_->encoder->get_relation_logical_arities();
}
const std::vector< int64_t >& FlatRelationEncoderEngine::get_relation_encoded_arities() const
{
   return semantic_->encoder->get_relation_encoded_arities();
}
const std::vector< int64_t >& FlatRelationEncoderEngine::get_relation_slot_roles() const
{
   return semantic_->encoder->get_relation_slot_roles();
}
const std::vector< int64_t >& FlatRelationEncoderEngine::get_relation_slot_role_offsets() const
{
   return semantic_->encoder->get_relation_slot_role_offsets();
}
const std::vector< std::string >& FlatRelationEncoderEngine::get_slot_role_names() const
{
   return semantic_->encoder->get_slot_role_names();
}

}  // namespace mifrost
