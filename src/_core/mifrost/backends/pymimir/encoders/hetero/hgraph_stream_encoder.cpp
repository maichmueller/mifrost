#include "hgraph_stream_encoder.hpp"

#include <stdexcept>
#include <utility>

#include "mifrost/input_handling/batch_input_parser.hpp"

namespace mifrost {
namespace {

bool same_schema(
   const pymimir::hetero_bridge::Schema& lhs,
   const pymimir::hetero_bridge::Schema& rhs
)
{
   return lhs.predicates == rhs.predicates and lhs.actions == rhs.actions;
}

SemanticHGraphEncoderConfig semantic_config(const HGraphEncoderEngine::Config& config)
{
   SemanticHGraphEncoderConfig result;
   result.symbol_type_id = config.symbol_type_id;
   result.target_symbol_prefix = config.target_symbol_prefix;
   result.nullary_object_name = config.nullary_object_name;
   result.lgan_tn_edge_pos = config.lgan_tn_edge_pos;
   result.lgan_nn_edge_pos = config.lgan_nn_edge_pos;
   result.lgan_rr_edge_pos = config.lgan_rr_edge_pos;
   result.history_link_relation = config.history_link_relation;
   result.max_goal_level = config.max_goal_level;
   result.support_literals = config.support_literals;
   result.add_nullary_predicates = config.add_nullary_predicates;
   result.ignore_actions = config.ignore_actions;
   result.include_lgan_edges = config.include_lgan_edges;
   result.include_static = config.include_static;
   result.include_empty_edge_types = config.include_empty_edge_types;
   result.export_node_names = config.export_node_names;
   result.allow_subgoal_layers_beyond_max_goal_level = true;
   result.lgan_anchor_sources = config.lgan_anchor_sources;
   result.target_sources = config.target_sources;
   result.goal_derivations = config.goal_derivations;
   return result;
}

RelationDict
relation_dict(const mimir::formalism::DomainImpl& domain, const HGraphEncoderEngine::Config& config)
{
   RelationDictConfig relation_config;
   relation_config.max_goal_level = static_cast< int >(config.max_goal_level);
   relation_config.support_literals = config.support_literals;
   relation_config.goal_derivations = config.goal_derivations;
   relation_config.top_type_predicates.insert(config.symbol_type_id);
   std::vector< mimir::formalism::Action > actions;
   if(not config.ignore_actions) {
      actions.assign(domain.get_actions().begin(), domain.get_actions().end());
   }
   const int action_offset = (config.target_sources.contains(TargetSource::actions)
                              or config.include_lgan_edges)
                                ? 1
                                : 0;
   return build_pymimir_relation_dict(domain, actions, relation_config, 0, action_offset);
}

}  // namespace

HGraphEncoderEngine::HGraphEncoderEngine(const mimir::formalism::DomainImpl& domain)
    : HGraphEncoderEngine(domain, Config{})
{
}

HGraphEncoderEngine::HGraphEncoderEngine(const mimir::formalism::DomainImpl& domain, Config config)
    : HGraphEncoderEngine(domain, config, semantic_config(config))
{
}

HGraphEncoderEngine::HGraphEncoderEngine(mimir::formalism::Domain domain)
    : HGraphEncoderEngine(std::move(domain), Config{})
{
}

HGraphEncoderEngine::HGraphEncoderEngine(mimir::formalism::Domain domain, Config config)
    : HGraphEncoderEngine(std::move(domain), config, semantic_config(config))
{
}

HGraphEncoderEngine::HGraphEncoderEngine(
   const mimir::formalism::DomainImpl& domain,
   Config config,
   SemanticHGraphEncoderConfig semantic_config_value
)
    : config_(std::move(config)),
      domain_(domain),
      schema_(pymimir::hetero_bridge::schema(domain)),
      semantic_(
         std::make_unique< SemanticHGraphEncoderEngine >(
            schema_.predicates,
            schema_.actions,
            std::move(semantic_config_value)
         )
      ),
      relation_dict_(relation_dict(domain, config_))
{
}

HGraphEncoderEngine::HGraphEncoderEngine(
   mimir::formalism::Domain domain,
   Config config,
   SemanticHGraphEncoderConfig semantic_config_value
)
    : config_(std::move(config)),
      domain_holder_(std::move(domain)),
      domain_(*domain_holder_),
      schema_(pymimir::hetero_bridge::schema(domain_)),
      semantic_(
         std::make_unique< SemanticHGraphEncoderEngine >(
            schema_.predicates,
            schema_.actions,
            std::move(semantic_config_value)
         )
      ),
      relation_dict_(relation_dict(domain_, config_))
{
}

HGraphEncoderEngine::~HGraphEncoderEngine() = default;

std::shared_ptr< const SemanticTaskContext > HGraphEncoderEngine::make_task_context(
   const mimir::search::State& state
) const
{
   ensure_problem(state);
   return problem_adapter_->get_task_context();
}

void HGraphEncoderEngine::ensure_problem(const mimir::search::State& state) const
{
   const auto* problem = &state.get_problem();
   if(problem_ == problem) {
      return;
   }
   problem_ = problem;
   problem_adapter_ = std::make_unique< pymimir::SemanticProblemAdapter >(*problem);
   semantic_ = std::make_unique< SemanticHGraphEncoderEngine >(
      problem_adapter_->get_task_context(), semantic_config(config_)
   );
}

const pymimir::views::Context& HGraphEncoderEngine::view_context(
   const mimir::search::State& state
) const
{
   ensure_problem(state);
   return problem_adapter_->get_view_context();
}

SemanticFlatRelationInput HGraphEncoderEngine::make_input(
   const mimir::search::State& state,
   const GoalInputs& goals,
   std::span< const mimir::formalism::GroundAction > actions,
   const std::vector< HistorySubgoal >& history,
   std::optional< int > history_max_steps
) const
{
   auto input = pymimir::hetero_bridge::input(make_task_context(state), state, goals, actions);
   if(not history.empty()) {
      pymimir::hetero_bridge::add_history(input, history, view_context(state));
   }
   input.history_max_steps = history_max_steps;
   return input;
}

SemanticFlatRelationInput HGraphEncoderEngine::make_default_input(
   const mimir::search::State& state
) const
{
   ensure_problem(state);
   return problem_adapter_->make_input(state);
}

void HGraphEncoderEngine::encode_semantic(
   const mimir::search::State& state,
   SemanticFlatRelationInput input,
   BatchBuilder& builder
) const
{
   const auto state_schema = pymimir::hetero_bridge::schema(*state.get_problem().get_domain());
   if(same_schema(state_schema, schema_)) {
      semantic_->encode(input, builder);
      return;
   }
   SemanticHGraphEncoderEngine compatible_engine(
      state_schema.predicates, state_schema.actions, semantic_config(config_)
   );
   compatible_engine.encode(input, builder);
}

void HGraphEncoderEngine::encode_impl(
   const mimir::search::State& state,
   const GoalInputs& goals,
   std::span< const mimir::formalism::GroundAction > actions,
   const std::vector< HistorySubgoal >& history,
   std::optional< int > history_max_steps,
   BatchBuilder& builder
) const
{
   ensure_problem(state);
   const auto state_view = problem_adapter_->make_state_view(state);
   const auto action_views = problem_adapter_->make_action_views(actions);
   const auto state_schema = pymimir::hetero_bridge::schema(*state.get_problem().get_domain());
   const auto goal_views = problem_adapter_->make_goal_views(goals);
   const auto history_view = pymimir::make_history_view(
      history, problem_adapter_->get_view_context()
   );
   if(same_schema(state_schema, schema_)) {
      semantic_->encode(
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
   SemanticHGraphEncoderEngine compatible_engine(
      problem_adapter_->get_task_context(), semantic_config(config_)
   );
   compatible_engine.encode(
      state_view,
      goal_views.goals_view(),
      goal_views.subgoal_layers_view(),
      action_views,
      history_view,
      history_max_steps,
      builder
   );
}

void HGraphEncoderEngine::encode(const mimir::search::State& state, BatchBuilder& builder)
{
   ensure_problem(state);
   const auto state_view = problem_adapter_->make_state_view(state);
   const auto action_views = problem_adapter_->make_action_views(
      std::span< const mimir::formalism::GroundAction >{}
   );
   const auto goal_views = problem_adapter_->make_default_goal_views();
   semantic_->encode(state_view, goal_views.goals_view(), action_views, builder);
}

void HGraphEncoderEngine::encode(
   const mimir::search::State& state,
   const GoalInputs& goals,
   std::span< const mimir::formalism::GroundAction > actions,
   BatchBuilder& builder
)
{
   encode_impl(state, goals, actions, {}, std::nullopt, builder);
}

void HGraphEncoderEngine::encode(
   const mimir::search::State& state,
   const GoalInputs& goals,
   std::span< const mimir::formalism::GroundAction > actions,
   const std::vector< HistorySubgoal >& history,
   std::optional< int > history_max_steps,
   BatchBuilder& builder
)
{
   encode_impl(state, goals, actions, history, history_max_steps, builder);
}

void HGraphEncoderEngine::update_relations(RelationDict relation_dict_value)
{
   semantic_->update_relations(relation_dict_value.arity);
   relation_dict_ = std::move(relation_dict_value);
}

BatchBuilder::BatchEncoding HGraphEncoderEngine::encode_batch(
   const batch_input::parsed::HGraphBatchInputs& inputs,
   std::optional< int > history_max_steps
)
{
   BatchBuilder builder;
   builder.set_graph_kind("hetero");
   for(size_t index = 0; index < inputs.states.states.size(); ++index) {
      const auto& state = inputs.states.states[index];
      const auto goals_payload = inputs.goals.at(index);
      const auto layers_payload = inputs.subgoal_layers.at(index);
      GoalInputs goals = goals_payload
                            ? batch_input::compose_goal_inputs(
                                 *goals_payload, layers_payload ? &*layers_payload : nullptr
                              )
                            : batch_input::default_goal_inputs_for_batch_state(state);
      const auto actions_payload = inputs.actions.at(index);
      const std::vector< mimir::formalism::GroundAction > empty_actions;
      const auto& actions = actions_payload ? *actions_payload : empty_actions;
      const auto history_payload = inputs.history_subgoals.at(index);
      const std::vector< HistorySubgoal > empty_history;
      const auto& history = history_payload ? *history_payload : empty_history;
      if(goals_payload or actions_payload or not history.empty()) {
         encode(state.state, goals, actions, history, history_max_steps, builder);
      } else {
         encode(state.state, builder);
      }
      builder.next_graph();
   }
   return builder.build();
}

int64_t HGraphMutableStreamEncoder::append(const mimir::search::State& state)
{
   HGraphStepInput step{.state = &state};
   return StreamEncoderBase::append(step);
}

int64_t HGraphMutableStreamEncoder::append(
   const mimir::search::State& state,
   const GoalInputs& goals,
   const std::vector< mimir::formalism::GroundAction >& actions
)
{
   HGraphStepInput step{.state = &state, .goals = &goals, .actions = &actions};
   return StreamEncoderBase::append(step);
}

int64_t HGraphMutableStreamEncoder::append(
   const mimir::search::State& state,
   const GoalInputs& goals,
   const std::vector< mimir::formalism::GroundAction >& actions,
   const std::vector< HGraphEncoderEngine::HistorySubgoal >& history,
   std::optional< int > history_max_steps
)
{
   HGraphStepInput step{
      .state = &state,
      .goals = &goals,
      .actions = &actions,
      .history = &history,
      .history_max_steps = history_max_steps,
   };
   return StreamEncoderBase::append(step);
}

void HGraphMutableStreamEncoder::update(int64_t id, const mimir::search::State& state)
{
   StreamEncoderBase::update(id, HGraphStepInput{.state = &state});
}

void HGraphMutableStreamEncoder::update(
   int64_t id,
   const mimir::search::State& state,
   const GoalInputs& goals,
   const std::vector< mimir::formalism::GroundAction >& actions
)
{
   StreamEncoderBase::update(
      id, HGraphStepInput{.state = &state, .goals = &goals, .actions = &actions}
   );
}

void HGraphMutableStreamEncoder::update(
   int64_t id,
   const mimir::search::State& state,
   const GoalInputs& goals,
   const std::vector< mimir::formalism::GroundAction >& actions,
   const std::vector< HGraphEncoderEngine::HistorySubgoal >& history,
   std::optional< int > history_max_steps
)
{
   StreamEncoderBase::update(
      id,
      HGraphStepInput{
         .state = &state,
         .goals = &goals,
         .actions = &actions,
         .history = &history,
         .history_max_steps = history_max_steps,
      }
   );
}

void HGraphMutableStreamEncoder::encode_step(const HGraphStepInput& step, BatchBuilder& builder)
{
   if(engine_ == nullptr or step.state == nullptr) {
      throw std::invalid_argument("HGraphMutableStreamEncoder requires a valid engine/state");
   }
   if(step.goals == nullptr) {
      if(step.history != nullptr and not step.history->empty()) {
         throw std::invalid_argument("History encoding requires explicit GoalInputs");
      }
      engine_->encode(*step.state, builder);
      return;
   }
   const std::span< const mimir::formalism::GroundAction >
      actions = step.actions ? std::span{*step.actions}
                             : std::span< const mimir::formalism::GroundAction >{};
   if(step.history != nullptr and not step.history->empty()) {
      engine_->encode(
         *step.state, *step.goals, actions, *step.history, step.history_max_steps, builder
      );
   } else {
      engine_->encode(*step.state, *step.goals, actions, builder);
   }
}

int64_t HGraphStreamEncoder::append(const mimir::search::State& state)
{
   ensure_valid();
   engine_->encode(state, builder_);
   builder_.next_graph();
   return next_id_++;
}

int64_t HGraphStreamEncoder::append(
   const mimir::search::State& state,
   const GoalInputs& goals,
   const std::vector< mimir::formalism::GroundAction >& actions
)
{
   ensure_valid();
   engine_->encode(state, goals, actions, builder_);
   builder_.next_graph();
   return next_id_++;
}

int64_t HGraphStreamEncoder::append(
   const mimir::search::State& state,
   const GoalInputs& goals,
   const std::vector< mimir::formalism::GroundAction >& actions,
   const std::vector< HGraphEncoderEngine::HistorySubgoal >& history,
   std::optional< int > history_max_steps
)
{
   ensure_valid();
   engine_->encode(state, goals, actions, history, history_max_steps, builder_);
   builder_.next_graph();
   return next_id_++;
}

BatchEncoding HGraphStreamEncoder::flush()
{
   ensure_valid();
   auto result = builder_.build();
   reset();
   return result;
}

void HGraphStreamEncoder::reset()
{
   builder_.reset();
   builder_.set_graph_kind("hetero");
   next_id_ = 0;
}

void HGraphStreamEncoder::ensure_valid() const
{
   if(engine_ == nullptr) {
      throw std::invalid_argument("HGraphStreamEncoder requires a valid engine");
   }
}

}  // namespace mifrost
