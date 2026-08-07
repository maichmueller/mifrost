#include "mifrost/backends/pymimir/encoders/flat/flat_relation_encoder.hpp"

#include <algorithm>
#include <memory>
#include <mimir/search/state_repository.hpp>
#include <ranges>
#include <utility>

#include "mifrost/backends/pymimir/semantic_views.hpp"
#include "mifrost/core/semantic/views.hpp"
#include "mifrost/input_handling/batch_input_parser.hpp"

namespace mifrost {

struct FlatRelationEncoderEngine::SemanticImpl {
   /// One problem's encoding machinery. The task context an encoder is built
   /// from carries that problem's object table (see `build_task_context`),
   /// which is its only problem-dependent content -- predicates and actions
   /// come from the domain -- so a binding is exactly "the object table to
   /// encode against", and nothing about the relation schema.
   struct ProblemBinding {
      /// Guards the raw map key. A `ProblemImpl*` is unique only among *live*
      /// problems, so a dropped problem's address can be recycled onto a new
      /// one and silently hand back the wrong object table. Rebuilding when
      /// this no longer names the problem the entry was built for is what
      /// makes the cache safe against that.
      std::weak_ptr< mimir::formalism::ProblemImpl > problem;
      std::unique_ptr< pymimir::SemanticProblemAdapter > adapter;
      std::unique_ptr< SemanticFlatRelationEncoderEngine > encoder;
   };

   /// Domain-level encoder. It never encodes a state and so is never bound to
   /// a problem: it answers the relation-schema queries (names, arities,
   /// sources, slot roles) and finalizes batches, all of which are functions
   /// of the domain alone.
   std::unique_ptr< SemanticFlatRelationEncoderEngine > schema_encoder;
   ankerl::unordered_dense::map< const mimir::formalism::ProblemImpl*, ProblemBinding > bindings;

   SemanticImpl(const mimir::formalism::DomainImpl& domain, const Config& config)
   {
      auto context = std::make_shared< SemanticTaskContext >();
      append_schema(domain, *context);
      schema_encoder = std::make_unique< SemanticFlatRelationEncoderEngine >(
         std::move(context), config
      );
   }

   /// The binding for `state`'s problem, built on first use.
   ///
   /// Encoding resolves per problem rather than once per encoder because a
   /// batch is a sequence of independent graphs -- `encode_batch` closes each
   /// item with `next_graph()`, so rows are graph-local -- and one batch may
   /// therefore legitimately span instances. Binding the whole encoder to the
   /// first problem it happened to see made that impossible to express.
   ProblemBinding& binding_for(const mimir::search::State& state, const Config& config)
   {
      // Taken from the state's repository rather than `state.get_problem()` so
      // that the key, the liveness guard and the adapter are the same object by
      // construction. Every mimir state is unpacked against its repository's
      // problem, so the two agree.
      const auto problem = state.get_state_repository()->get_problem();
      const auto* key = problem.get();
      auto& binding = bindings[key];
      if(binding.encoder == nullptr or binding.problem.lock().get() != key) {
         binding.problem = problem;
         binding.adapter = std::make_unique< pymimir::SemanticProblemAdapter >(*problem);
         binding.encoder = std::make_unique< SemanticFlatRelationEncoderEngine >(
            binding.adapter->get_task_context(), config
         );
      }
      return binding;
   }

   void encode_impl(
      const mimir::search::State& state,
      const GoalInputs* goals,
      std::span< const mimir::formalism::GroundAction > actions,
      std::span< const FlatRelationEncoderEngine::HistorySubgoal > history,
      std::optional< int > history_max_steps,
      const Config& config,
      BatchBuilder& builder
   )
   {
      auto& binding = binding_for(state, config);
      auto* const problem_adapter = binding.adapter.get();
      auto* const encoder = binding.encoder.get();
      const auto state_view = problem_adapter->make_state_view(state);
      const auto action_views = problem_adapter->make_action_views(actions);
      if(goals == nullptr and history.empty() and not history_max_steps.has_value()) {
         const auto goal_views = problem_adapter->make_default_goal_views();
         encoder->encode(state_view, goal_views.goals_view(), action_views, builder);
         return;
      }
      const auto history_view = pymimir::make_history_view(
         history, problem_adapter->get_view_context()
      );
      if(goals != nullptr) {
         const auto goal_views = problem_adapter->make_goal_views(*goals);
         encoder->encode(
            state_view,
            goal_views.goals_view(),
            goal_views.subgoal_layers_view(),
            action_views,
            history_view,
            history_max_steps,
            builder
         );
         return;
      }
      const auto goal_views = problem_adapter->make_default_goal_views();
      encoder->encode(
         state_view,
         goal_views.goals_view(),
         goal_views.subgoal_layers_view(),
         action_views,
         history_view,
         history_max_steps,
         builder
      );
   }

  private:
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
   const auto& names = semantic_->schema_encoder->get_relation_names();
   const auto& arities = semantic_->schema_encoder->get_relation_arities();
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
   const auto& names = semantic_->schema_encoder->get_relation_names();
   const auto& arities = semantic_->schema_encoder->get_relation_arities();
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
   semantic_->encode_impl(
      state, nullptr, actions, history_subgoals, history_max_steps, config_, builder
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
   semantic_->encode_impl(
      state, &goals, actions, history_subgoals, history_max_steps, config_, builder
   );
}

BatchBuilder::BatchEncoding FlatRelationEncoderEngine::encode_batch(
   const batch_input::parsed::FlatBatchInputs& inputs,
   std::optional< int > history_max_steps
)
{
   const size_t state_count = inputs.states.states.size();
   if(state_count == 0) {
      return semantic_->schema_encoder->encode_batch(std::vector< SemanticFlatRelationInput >{});
   }
   BatchBuilder builder;
   builder.set_graph_kind("flat");
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
      semantic_->encode_impl(
         state_entry.state, &goals, actions_span, history_span, history_max_steps, config_, builder
      );
      builder.next_graph();
   }
   auto result = builder.build();
   semantic_->schema_encoder->finalize_batch_encoding(result);
   return result;
}

void FlatRelationEncoderEngine::finalize_batch_encoding(BatchBuilder::BatchEncoding& encoding) const
{
   semantic_->schema_encoder->finalize_batch_encoding(encoding);
}

const RelationDict& FlatRelationEncoderEngine::get_relation_dict() const
{
   return relation_dict_;
}
const std::vector< std::string >& FlatRelationEncoderEngine::get_relation_names() const
{
   return semantic_->schema_encoder->get_relation_names();
}
const std::vector< int64_t >& FlatRelationEncoderEngine::get_relation_arities() const
{
   return semantic_->schema_encoder->get_relation_arities();
}
const std::vector< std::string >& FlatRelationEncoderEngine::get_relation_sources() const
{
   return semantic_->schema_encoder->get_relation_sources();
}
const std::vector< int64_t >& FlatRelationEncoderEngine::get_relation_logical_arities() const
{
   return semantic_->schema_encoder->get_relation_logical_arities();
}
const std::vector< int64_t >& FlatRelationEncoderEngine::get_relation_encoded_arities() const
{
   return semantic_->schema_encoder->get_relation_encoded_arities();
}
const std::vector< int64_t >& FlatRelationEncoderEngine::get_relation_slot_roles() const
{
   return semantic_->schema_encoder->get_relation_slot_roles();
}
const std::vector< int64_t >& FlatRelationEncoderEngine::get_relation_slot_role_offsets() const
{
   return semantic_->schema_encoder->get_relation_slot_role_offsets();
}
const std::vector< std::string >& FlatRelationEncoderEngine::get_slot_role_names() const
{
   return semantic_->schema_encoder->get_slot_role_names();
}

}  // namespace mifrost
