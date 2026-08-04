#include "horizon_hgraph_encoder.hpp"

#include <mimir/search/formatter.hpp>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "mifrost/input_handling/batch_input_parser.hpp"

namespace mifrost {
namespace {

class OwnedTargetNameBatch final: public DeferredStringBatch {
  public:
   explicit OwnedTargetNameBatch(std::vector< std::string > names) : names_(std::move(names)) {}

   std::vector< std::string > materialize() const override { return names_; }

  private:
   std::vector< std::string > names_;
};

HGraphEncoderEngine::Config base_config(const HorizonHGraphEncoderEngine::Config& config)
{
   HGraphEncoderEngine::Config result = config;
   if(config.transition_mode == HorizonHGraphEncoderEngine::Mode::delta) {
      result.support_literals = true;
   }
   return result;
}

}  // namespace

SemanticHorizonHGraphEncoderConfig HorizonHGraphEncoderEngine::semantic_config(const Config& config)
{
   const auto base = base_config(config);
   SemanticHorizonHGraphEncoderConfig result;
   result.symbol_type_id = base.symbol_type_id;
   result.target_symbol_prefix = base.target_symbol_prefix;
   result.nullary_object_name = base.nullary_object_name;
   result.lgan_tn_edge_pos = base.lgan_tn_edge_pos;
   result.lgan_nn_edge_pos = base.lgan_nn_edge_pos;
   result.lgan_rr_edge_pos = base.lgan_rr_edge_pos;
   result.history_link_relation = base.history_link_relation;
   result.max_goal_level = base.max_goal_level;
   result.support_literals = base.support_literals;
   result.add_nullary_predicates = base.add_nullary_predicates;
   result.ignore_actions = base.ignore_actions;
   result.include_lgan_edges = base.include_lgan_edges;
   result.include_static = base.include_static;
   result.include_empty_edge_types = base.include_empty_edge_types;
   result.export_node_names = base.export_node_names;
   result.allow_subgoal_layers_beyond_max_goal_level = true;
   result.lgan_anchor_sources = base.lgan_anchor_sources;
   result.target_sources = base.target_sources;
   result.goal_derivations = base.goal_derivations;
   result.transition_mode = config.transition_mode == Mode::delta    ? SemanticHorizonMode::delta
                            : config.transition_mode == Mode::action ? SemanticHorizonMode::action
                                                                     : SemanticHorizonMode::full;
   result.parent_relation = config.parent_relation;
   result.sibling_relation = config.sibling_relation;
   result.cousin_relation = config.cousin_relation;
   result.enable_parent_relation = config.enable_parent_relation;
   result.enable_sibling_relation = config.enable_sibling_relation;
   result.enable_cousin_relation = config.enable_cousin_relation;
   result.root_policy = config.root_policy;
   return result;
}

HorizonHGraphEncoderEngine::HorizonHGraphEncoderEngine(const mimir::formalism::DomainImpl& domain)
    : HorizonHGraphEncoderEngine(domain, Config{})
{
}

HorizonHGraphEncoderEngine::HorizonHGraphEncoderEngine(
   const mimir::formalism::DomainImpl& domain,
   Config config
)
    : HGraphEncoderEngine(domain, base_config(config)),
      horizon_config_(std::move(config)),
      semantic_horizon_(
         std::make_unique< SemanticHorizonHGraphEncoderEngine >(
            schema_.predicates,
            schema_.actions,
            semantic_config(horizon_config_)
         )
      )
{
   const auto base = base_config(horizon_config_);
   relation_dict_.arity = semantic_horizon_->get_relation_arities();
   relation_dict_.max_goal_level = static_cast< int >(base.max_goal_level);
   relation_dict_.support_literals = base.support_literals;
   relation_dict_.goal_derivations = base.goal_derivations;
}

HorizonHGraphEncoderEngine::HorizonHGraphEncoderEngine(mimir::formalism::Domain domain)
    : HorizonHGraphEncoderEngine(std::move(domain), Config{})
{
}

HorizonHGraphEncoderEngine::HorizonHGraphEncoderEngine(
   mimir::formalism::Domain domain,
   Config config
)
    : HGraphEncoderEngine(std::move(domain), base_config(config)),
      horizon_config_(std::move(config)),
      semantic_horizon_(
         std::make_unique< SemanticHorizonHGraphEncoderEngine >(
            schema_.predicates,
            schema_.actions,
            semantic_config(horizon_config_)
         )
      )
{
   const auto base = base_config(horizon_config_);
   relation_dict_.arity = semantic_horizon_->get_relation_arities();
   relation_dict_.max_goal_level = static_cast< int >(base.max_goal_level);
   relation_dict_.support_literals = base.support_literals;
   relation_dict_.goal_derivations = base.goal_derivations;
}

SemanticTransitionDAG HorizonHGraphEncoderEngine::materialize_dag(
   const TransitionDAG& dag,
   const std::shared_ptr< const SemanticTaskContext >& context,
   const pymimir::hetero_bridge::Schema& schema,
   const GoalInputs& goals
)
{
   std::vector< SemanticTransitionDAG::Node > nodes;
   nodes.reserve(dag.nodes().size());
   const auto root_view_context = pymimir::views::make_context(dag.root().get_problem());
   for(const auto& node : dag.nodes()) {
      auto input = pymimir::hetero_bridge::state_input(context, node.state);
      if(node.index == dag.root_index()) {
         input = pymimir::hetero_bridge::input(context, node.state, goals);
      }
      SemanticTransitionDAG::Node semantic_node;
      semantic_node.state = std::move(input);
      semantic_node.index = node.index;
      semantic_node.depth = node.depth;
      std::ostringstream display_name;
      display_name << node.state;
      semantic_node.display_name = display_name.str();
      if(node.action.has_value() and node.index != dag.root_index()) {
         semantic_node.incoming_action = pymimir::hetero_bridge::materialize_action(
            *node.action, root_view_context
         );
      }
      semantic_node.candidate_id = node.candidate_id;
      if(node.delta_literals.has_value()) {
         std::vector< SemanticLiteral > literals;
         for(const auto& literal : *node.delta_literals) {
            std::visit(
               [&](const auto& native) {
                  literals.push_back(
                     pymimir::hetero_bridge::materialize_literal(native, root_view_context)
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
   return SemanticTransitionDAG{
      schema.predicates,
      schema.actions,
      std::move(nodes),
      std::move(edges),
      true,
   };
}

void HorizonHGraphEncoderEngine::encode(
   const mimir::search::State& root,
   const TransitionDAG& dag,
   const GoalInputs& goals,
   BatchBuilder& builder
)
{
   const bool include_root = root_in_target_metadata(horizon_config_.root_policy);
   const auto& nodes = dag.nodes();
   const size_t first_target = include_root ? 0 : 1;
   bool has_candidate_id = false;
   for(size_t index = first_target; index < nodes.size(); ++index) {
      has_candidate_id = has_candidate_id or nodes[index].candidate_id.has_value();
   }
   if(has_candidate_id) {
      for(size_t index = first_target; index < nodes.size(); ++index) {
         if(not nodes[index].candidate_id.has_value()) {
            throw std::invalid_argument(
               "missing candidate_id for target node index " + std::to_string(nodes[index].index)
            );
         }
      }
   }

   auto context = make_task_context(root);
   auto semantic_dag = materialize_dag(dag, context, schema_, goals);
   semantic_horizon_->encode(semantic_dag, builder);
   if(not builder.lazy_target_name_strings.empty()) {
      auto names = std::move(builder.lazy_target_name_strings);
      builder.lazy_target_name_strings.clear();
      builder.add_lazy_target_name_batch(
         std::make_shared< OwnedTargetNameBatch >(std::move(names))
      );
   }
}

void HorizonHGraphEncoderEngine::update_relations(RelationDict relation_dict_value)
{
   if(horizon_config_.enable_parent_relation) {
      relation_dict_value.arity[horizon_config_.parent_relation] = 2;
   }
   if(horizon_config_.enable_sibling_relation) {
      relation_dict_value.arity[horizon_config_.sibling_relation] = 2;
   }
   if(horizon_config_.enable_cousin_relation) {
      relation_dict_value.arity[horizon_config_.cousin_relation] = 2;
   }
   semantic_horizon_->update_relations(relation_dict_value.arity);
   HGraphEncoderEngine::update_relations(std::move(relation_dict_value));
}

BatchBuilder::BatchEncoding HorizonHGraphEncoderEngine::encode_batch(
   const batch_input::parsed::HorizonBatchInputs& inputs
)
{
   BatchBuilder builder;
   builder.set_graph_kind("hetero");
   for(size_t index = 0; index < inputs.roots.states.size(); ++index) {
      const auto& root = inputs.roots.states[index];
      const auto dag_payload = inputs.dags.at(index);
      TransitionDAG dag = dag_payload ? *dag_payload : TransitionDAG(root.state);
      const auto goals_payload = inputs.goals.at(index);
      const auto layers_payload = inputs.subgoal_layers.at(index);
      const GoalInputs goals = goals_payload
                                  ? batch_input::compose_goal_inputs(
                                       *goals_payload, layers_payload ? &*layers_payload : nullptr
                                    )
                                  : batch_input::default_goal_inputs_for_batch_state(root);
      encode(root.state, dag, goals, builder);
      builder.next_graph();
   }
   return builder.build();
}

int64_t HorizonStreamEncoder::append(
   const mimir::search::State& root,
   const TransitionDAG& dag,
   const GoalInputs& goals
)
{
   return StreamEncoderBase::append(HorizonStepInput{.root = &root, .dag = &dag, .goals = &goals});
}

int64_t HorizonStreamEncoder::append(const mimir::search::State& root, const GoalInputs& goals)
{
   auto dag = std::make_shared< TransitionDAG >(root);
   auto step = HorizonStepInput{
      .root = &root, .dag = dag.get(), .goals = &goals, .owned_dag = std::move(dag)
   };
   return StreamEncoderBase::append(step);
}

void HorizonStreamEncoder::update(
   int64_t id,
   const mimir::search::State& root,
   const TransitionDAG& dag,
   const GoalInputs& goals
)
{
   StreamEncoderBase::update(id, HorizonStepInput{.root = &root, .dag = &dag, .goals = &goals});
}

void HorizonStreamEncoder::update(
   int64_t id,
   const mimir::search::State& root,
   const GoalInputs& goals
)
{
   auto dag = std::make_shared< TransitionDAG >(root);
   StreamEncoderBase::update(
      id,
      HorizonStepInput{
         .root = &root, .dag = dag.get(), .goals = &goals, .owned_dag = std::move(dag)
      }
   );
}

void HorizonStreamEncoder::encode_step(const HorizonStepInput& step, BatchBuilder& builder)
{
   if(engine_ == nullptr or step.root == nullptr or step.dag == nullptr or step.goals == nullptr) {
      throw std::invalid_argument("HorizonStreamEncoder requires root/dag/goals");
   }
   engine_->encode(*step.root, *step.dag, *step.goals, builder);
}

}  // namespace mifrost
