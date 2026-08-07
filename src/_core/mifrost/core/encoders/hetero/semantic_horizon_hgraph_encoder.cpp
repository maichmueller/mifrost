#include "semantic_horizon_hgraph_encoder.hpp"

#include <stdexcept>
#include <utility>

namespace mifrost {
namespace {

SemanticHorizonHGraphEncoderConfig normalize_horizon_config(
   SemanticHorizonHGraphEncoderConfig config
)
{
   if(config.transition_mode == SemanticHorizonMode::delta) {
      config.support_literals = true;
   }
   return config;
}

SemanticHGraphEncoderConfig base_config(const SemanticHorizonHGraphEncoderConfig& config)
{
   return config;
}

}  // namespace

struct SemanticHorizonHGraphEncoderEngine::Impl {
   Config config;
   SemanticHGraphEncoderEngine hgraph;

   Impl(
      std::vector< SemanticPredicateSpec > predicates,
      std::vector< SemanticActionSpec > actions,
      Config encoder_config
   )
       : config(normalize_horizon_config(std::move(encoder_config))),
         hgraph(std::move(predicates), std::move(actions), base_config(config))
   {
      hgraph.configure_horizon(config);
   }

   Impl(std::shared_ptr< const SemanticSchemaContext > schema, Config encoder_config)
       : config(normalize_horizon_config(std::move(encoder_config))),
         hgraph(std::move(schema), base_config(config))
   {
      hgraph.configure_horizon(config);
   }
};

SemanticHorizonHGraphEncoderEngine::SemanticHorizonHGraphEncoderEngine(
   std::vector< SemanticPredicateSpec > predicates,
   std::vector< SemanticActionSpec > actions,
   Config config
)
    : impl_(std::make_unique< Impl >(std::move(predicates), std::move(actions), std::move(config)))
{
}

SemanticHorizonHGraphEncoderEngine::SemanticHorizonHGraphEncoderEngine(
   std::shared_ptr< const SemanticSchemaContext > schema,
   Config config
)
    : impl_(std::make_unique< Impl >(std::move(schema), std::move(config)))
{
}

SemanticHorizonHGraphEncoderEngine::SemanticHorizonHGraphEncoderEngine(
   SemanticHorizonHGraphEncoderEngine&&
) noexcept = default;
SemanticHorizonHGraphEncoderEngine& SemanticHorizonHGraphEncoderEngine::operator=(
   SemanticHorizonHGraphEncoderEngine&&
) noexcept = default;
SemanticHorizonHGraphEncoderEngine::~SemanticHorizonHGraphEncoderEngine() = default;

BatchBuilder::BatchEncoding SemanticHorizonHGraphEncoderEngine::encode(
   const SemanticTransitionDAG& dag
) const
{
   BatchBuilder builder;
   encode(dag, builder);
   builder.next_graph();
   return builder.build();
}

void SemanticHorizonHGraphEncoderEngine::encode(
   const SemanticTransitionDAG& dag,
   BatchBuilder& builder
) const
{
   if(dag.predicates() != impl_->hgraph.get_predicates()
      or dag.actions() != impl_->hgraph.get_actions()) {
      throw std::invalid_argument(
         "Semantic Horizon DAG schema must exactly match the encoder schema"
      );
   }
   impl_->hgraph.encode_horizon(dag, impl_->config, builder);
}

BatchBuilder::BatchEncoding SemanticHorizonHGraphEncoderEngine::encode_batch(
   const std::vector< SemanticTransitionDAG >& dags
) const
{
   BatchBuilder builder;
   builder.set_graph_kind("hetero");
   for(const auto& dag : dags) {
      encode(dag, builder);
      builder.next_graph();
   }
   return builder.build();
}

const SemanticHorizonHGraphEncoderEngine::Config&
SemanticHorizonHGraphEncoderEngine::get_config() const
{
   return impl_->config;
}

const std::vector< SemanticPredicateSpec >&
SemanticHorizonHGraphEncoderEngine::get_predicates() const
{
   return impl_->hgraph.get_predicates();
}

const std::vector< SemanticActionSpec >& SemanticHorizonHGraphEncoderEngine::get_actions() const
{
   return impl_->hgraph.get_actions();
}

const std::map< std::string, int >& SemanticHorizonHGraphEncoderEngine::get_relation_arities() const
{
   return impl_->hgraph.get_relation_arities();
}

void SemanticHorizonHGraphEncoderEngine::update_relations(
   std::map< std::string, int > relation_arities
)
{
   impl_->hgraph.update_relations(std::move(relation_arities));
}

}  // namespace mifrost
