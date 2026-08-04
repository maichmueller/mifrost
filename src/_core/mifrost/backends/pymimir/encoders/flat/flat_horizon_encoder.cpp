/**
 * @file flat_horizon_encoder.cpp
 * @brief Pymimir adapter for the backend-neutral flat Horizon encoder.
 */
#include "flat_horizon_encoder.hpp"

#include <algorithm>
#include <mimir/search/formatter.hpp>
#include <sstream>
#include <tuple>
#include <utility>

#include "mifrost/backends/pymimir/encoders/hetero/semantic_hgraph_view_bridge.hpp"
#include "mifrost/backends/pymimir/semantic_views.hpp"
#include "mifrost/core/encoders/flat/semantic_flat_horizon_encoder.hpp"
#include "mifrost/input_handling/batch_input_parser.hpp"

namespace mifrost {
namespace {

SemanticFlatHorizonEncoderConfig semantic_config(const FlatHorizonEncoderEngine::Config& config)
{
   SemanticFlatHorizonEncoderConfig result;
   result.max_goal_level = config.max_goal_level;
   result.support_literals = config.support_literals;
   result.include_static = config.include_static;
   result.export_node_names = config.export_node_names;
   result.ignore_zero_arity_relations = config.ignore_zero_arity_relations;
   result.use_predicate_virtual_nodes = config.use_predicate_virtual_nodes;
   result.include_lgan_edges = config.include_lgan_edges;
   result.target_symbol_prefix = config.target_symbol_prefix;
   result.lgan_tn_edge_pos = config.lgan_tn_edge_pos;
   result.lgan_nn_edge_pos = config.lgan_nn_edge_pos;
   result.lgan_rr_edge_pos = config.lgan_rr_edge_pos;
   result.goal_derivations = config.goal_derivations;
   result.ignore_actions = config.ignore_actions;
   result.transition_mode = config.transition_mode == FlatHorizonEncoderEngine::Mode::delta
                               ? SemanticHorizonMode::delta
                            : config.transition_mode == FlatHorizonEncoderEngine::Mode::action
                               ? SemanticHorizonMode::action
                               : SemanticHorizonMode::full;
   result.parent_relation = config.parent_relation;
   result.sibling_relation = config.sibling_relation;
   result.cousin_relation = config.cousin_relation;
   result.enable_parent_relation = config.enable_parent_relation;
   result.enable_sibling_relation = config.enable_sibling_relation;
   result.enable_cousin_relation = config.enable_cousin_relation;
   result.root_policy = config.root_policy;
   result.pack_relation_args_relation_major = config.pack_relation_args_relation_major;
   return result;
}

std::vector< SemanticActionSpec > semantic_actions(const mimir::formalism::DomainImpl& domain)
{
   auto actions = domain.get_actions();
   std::ranges::sort(actions, [](const auto lhs, const auto rhs) {
      return std::tuple{lhs->get_name(), lhs->get_arity(), lhs->get_index()}
             < std::tuple{rhs->get_name(), rhs->get_arity(), rhs->get_index()};
   });
   std::vector< SemanticActionSpec > result;
   result.reserve(actions.size());
   for(const auto action : actions) {
      result.push_back(
         SemanticActionSpec{
            std::string(action->get_name()), static_cast< int64_t >(action->get_arity())
         }
      );
   }
   return result;
}

std::shared_ptr< SemanticTaskContext > schema_context(const mimir::formalism::DomainImpl& domain)
{
   auto result = std::make_shared< SemanticTaskContext >();
   result->predicates = pymimir::make_semantic_predicates(domain);
   result->actions = semantic_actions(domain);
   return result;
}

std::string state_display_name(const mimir::search::State& state)
{
   std::ostringstream stream;
   stream << state;
   return stream.str();
}

SemanticTransitionDAG materialize_dag(
   const TransitionDAG& dag,
   const mimir::search::State& root,
   const GoalInputs& goals,
   const pymimir::SemanticProblemAdapter& adapter
)
{
   std::vector< SemanticTransitionDAG::Node > nodes;
   nodes.reserve(dag.nodes().size());
   const auto view_context = pymimir::views::make_context(root.get_problem());
   for(const auto& node : dag.nodes()) {
      SemanticTransitionDAG::Node semantic_node;
      semantic_node.state = node.index == dag.root_index() ? adapter.make_input(root, goals)
                                                           : adapter.make_input(node.state);
      semantic_node.index = node.index;
      semantic_node.depth = node.depth;
      if(adapter.make_input(root).task_context) {
         semantic_node.display_name = state_display_name(node.state);
      }
      if(node.action.has_value()) {
         semantic_node.incoming_action = pymimir::hetero_bridge::materialize_action(
            *node.action, view_context
         );
      }
      semantic_node.candidate_id = node.candidate_id;
      if(node.delta_literals.has_value()) {
         std::vector< SemanticLiteral > literals;
         literals.reserve(node.delta_literals->size());
         for(const auto& literal : *node.delta_literals) {
            std::visit(
               [&](const auto& native) {
                  literals.push_back(
                     pymimir::hetero_bridge::materialize_literal(native, view_context)
                  );
               },
               literal
            );
         }
         semantic_node.delta_literals = std::move(literals);
      }
      nodes.push_back(std::move(semantic_node));
   }

   std::vector< SemanticTransitionDAG::Edge > edges;
   for(const auto [parent, child] : dag.transitions()) {
      edges.emplace_back(parent, child);
   }
   const auto context = adapter.make_input(root).task_context;
   return SemanticTransitionDAG{
      context->predicates,
      context->actions,
      std::move(nodes),
      std::move(edges),
   };
}

void fill_relation_dict(
   RelationDict& result,
   const SemanticFlatHorizonEncoderEngine& encoder,
   const FlatHorizonEncoderEngine::Config& config
)
{
   result.max_goal_level = static_cast< int >(config.max_goal_level);
   result.support_literals = config.support_literals;
   result.goal_derivations = config.goal_derivations;
   const auto& names = encoder.get_relation_names();
   const auto& arities = encoder.get_relation_arities();
   for(size_t index = 0; index < names.size(); ++index) {
      result.arity[names[index]] = static_cast< int >(arities[index]);
   }
}

}  // namespace

struct FlatHorizonEncoderEngine::SemanticImpl {
   Config config;
   SemanticFlatHorizonEncoderConfig semantic_config;
   std::unique_ptr< SemanticFlatHorizonEncoderEngine > encoder;
   std::unique_ptr< pymimir::SemanticProblemAdapter > problem_adapter;
   RelationDict relation_dict;
   const mimir::formalism::ProblemImpl* problem = nullptr;

   SemanticImpl(const mimir::formalism::DomainImpl& domain, Config value)
       : config(std::move(value)),
         semantic_config(::mifrost::semantic_config(config)),
         encoder(
            std::make_unique< SemanticFlatHorizonEncoderEngine >(
               schema_context(domain),
               semantic_config
            )
         )
   {
      fill_relation_dict(relation_dict, *encoder, config);
   }

   void ensure_problem(const mimir::search::State& state)
   {
      const auto* value = &state.get_problem();
      if(problem != nullptr) {
         if(problem != value) {
            throw std::invalid_argument(
               "FlatHorizonEncoder state belongs to a different planning problem"
            );
         }
         return;
      }
      problem = value;
      problem_adapter = std::make_unique< pymimir::SemanticProblemAdapter >(*problem);
      encoder = std::make_unique< SemanticFlatHorizonEncoderEngine >(
         problem_adapter->make_input(state).task_context, semantic_config
      );
      fill_relation_dict(relation_dict, *encoder, config);
   }

   SemanticTransitionDAG
   make_dag(const mimir::search::State& root, const TransitionDAG& dag, const GoalInputs& goals)
   {
      ensure_problem(root);
      return materialize_dag(dag, root, goals, *problem_adapter);
   }
};

FlatHorizonEncoderEngine::FlatHorizonEncoderEngine(const mimir::formalism::DomainImpl& domain)
    : FlatHorizonEncoderEngine(domain, Config{})
{
}

FlatHorizonEncoderEngine::FlatHorizonEncoderEngine(
   const mimir::formalism::DomainImpl& domain,
   Config config
)
    : domain_(domain),
      config_(std::move(config)),
      semantic_(std::make_unique< SemanticImpl >(domain, config_))
{
}

FlatHorizonEncoderEngine::FlatHorizonEncoderEngine(mimir::formalism::Domain domain)
    : FlatHorizonEncoderEngine(std::move(domain), Config{})
{
}

FlatHorizonEncoderEngine::FlatHorizonEncoderEngine(mimir::formalism::Domain domain, Config config)
    : domain_holder_(std::move(domain)),
      domain_(*domain_holder_),
      config_(std::move(config)),
      semantic_(std::make_unique< SemanticImpl >(domain_, config_))
{
}

FlatHorizonEncoderEngine::~FlatHorizonEncoderEngine() = default;

void FlatHorizonEncoderEngine::encode(
   const mimir::search::State& root,
   const TransitionDAG& dag,
   const GoalInputs& goals,
   BatchBuilder& builder
)
{
   auto semantic_dag = semantic_->make_dag(root, dag, goals);
   semantic_->encoder->encode(semantic_dag, builder);
}

BatchBuilder::BatchEncoding FlatHorizonEncoderEngine::encode_batch(
   const batch_input::parsed::HorizonBatchInputs& inputs
)
{
   std::vector< SemanticTransitionDAG > dags;
   dags.reserve(inputs.roots.states.size());
   for(size_t index = 0; index < inputs.roots.states.size(); ++index) {
      const auto& root = inputs.roots.states[index].state;
      const auto& dag_payload = inputs.dags.at(index);
      const auto& goals_payload = inputs.goals.at(index);
      const auto& layers_payload = inputs.subgoal_layers.at(index);
      const GoalInputs goals = goals_payload
                                  ? batch_input::compose_goal_inputs(
                                       *goals_payload, layers_payload ? &*layers_payload : nullptr
                                    )
                                  : batch_input::default_goal_inputs_for_batch_state(
                                       inputs.roots.states[index]
                                    );
      const TransitionDAG fallback(root);
      dags.push_back(semantic_->make_dag(root, dag_payload ? *dag_payload : fallback, goals));
   }
   auto result = semantic_->encoder->encode_batch(dags);
   finalize_batch_encoding(result);
   return result;
}

void FlatHorizonEncoderEngine::finalize_batch_encoding(BatchBuilder::BatchEncoding& encoding) const
{
   semantic_->encoder->finalize_batch_encoding(encoding);
}

const RelationDict& FlatHorizonEncoderEngine::get_relation_dict() const
{
   static const RelationDict empty;
   return semantic_ ? semantic_->relation_dict : empty;
}

const std::vector< std::string >& FlatHorizonEncoderEngine::get_relation_names() const
{
   return semantic_->encoder->get_relation_names();
}

const std::vector< int64_t >& FlatHorizonEncoderEngine::get_relation_arities() const
{
   return semantic_->encoder->get_relation_arities();
}

const std::vector< std::string >& FlatHorizonEncoderEngine::get_relation_sources() const
{
   return semantic_->encoder->get_relation_sources();
}

const std::vector< int64_t >& FlatHorizonEncoderEngine::get_relation_logical_arities() const
{
   return semantic_->encoder->get_relation_logical_arities();
}

const std::vector< int64_t >& FlatHorizonEncoderEngine::get_relation_encoded_arities() const
{
   return semantic_->encoder->get_relation_encoded_arities();
}

const std::vector< int64_t >& FlatHorizonEncoderEngine::get_relation_slot_roles() const
{
   return semantic_->encoder->get_relation_slot_roles();
}

const std::vector< int64_t >& FlatHorizonEncoderEngine::get_relation_slot_role_offsets() const
{
   return semantic_->encoder->get_relation_slot_role_offsets();
}

const std::vector< std::string >& FlatHorizonEncoderEngine::get_slot_role_names() const
{
   return semantic_->encoder->get_slot_role_names();
}

}  // namespace mifrost
