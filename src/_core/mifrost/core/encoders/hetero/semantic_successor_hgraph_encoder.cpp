#include "semantic_successor_hgraph_encoder.hpp"

#include <stdexcept>
#include <utility>

#include "mifrost/core/encoders/flat/semantic_flat_relation_view_bridge.hpp"

namespace mifrost {
namespace {

SemanticSuccessorHGraphEncoderConfig normalize_successor_config(
   SemanticSuccessorHGraphEncoderConfig config
)
{
   if(config.successor_mode == SemanticSuccessorMode::delta and not config.support_literals) {
      config.support_literals = true;
   }
   return config;
}

SemanticHGraphEncoderConfig base_config(const SemanticSuccessorHGraphEncoderConfig& config)
{
   return config;
}

}  // namespace

struct SemanticSuccessorHGraphEncoderEngine::Impl {
   Config config;
   SemanticHGraphEncoderEngine hgraph;

   Impl(
      std::vector< SemanticPredicateSpec > predicates,
      std::vector< SemanticActionSpec > actions,
      Config encoder_config
   )
       : config(normalize_successor_config(std::move(encoder_config))),
         hgraph(std::move(predicates), std::move(actions), base_config(config))
   {
   }

   Impl(std::shared_ptr< const SemanticTaskContext > task_context, Config encoder_config)
       : config(normalize_successor_config(std::move(encoder_config))),
         hgraph(std::move(task_context), base_config(config))
   {
   }
};

SemanticSuccessorHGraphEncoderEngine::SemanticSuccessorHGraphEncoderEngine(
   std::vector< SemanticPredicateSpec > predicates,
   std::vector< SemanticActionSpec > actions,
   Config config
)
    : impl_(std::make_unique< Impl >(std::move(predicates), std::move(actions), std::move(config)))
{
}

SemanticSuccessorHGraphEncoderEngine::SemanticSuccessorHGraphEncoderEngine(
   std::shared_ptr< const SemanticTaskContext > task_context,
   Config config
)
    : impl_(std::make_unique< Impl >(std::move(task_context), std::move(config)))
{
}

SemanticSuccessorHGraphEncoderEngine::SemanticSuccessorHGraphEncoderEngine(
   SemanticSuccessorHGraphEncoderEngine&&
) noexcept = default;
SemanticSuccessorHGraphEncoderEngine& SemanticSuccessorHGraphEncoderEngine::operator=(
   SemanticSuccessorHGraphEncoderEngine&&
) noexcept = default;
SemanticSuccessorHGraphEncoderEngine::~SemanticSuccessorHGraphEncoderEngine() = default;

BatchBuilder::BatchEncoding SemanticSuccessorHGraphEncoderEngine::encode(
   const SemanticFlatRelationInput& current,
   const SemanticFlatRelationInput& successor
) const
{
   BatchBuilder builder;
   encode(current, successor, builder);
   builder.next_graph();
   return builder.build();
}

BatchBuilder::BatchEncoding SemanticSuccessorHGraphEncoderEngine::encode_views(
   const canonical::FlatRelationViewInput& current,
   const canonical::FlatRelationViewInput& successor
) const
{
   BatchBuilder builder;
   encode_views(current, successor, builder);
   builder.next_graph();
   return builder.build();
}

void SemanticSuccessorHGraphEncoderEngine::encode(
   const SemanticFlatRelationInput& current,
   const SemanticFlatRelationInput& successor,
   BatchBuilder& builder
) const
{
   impl_->hgraph.encode_successor(
      current,
      successor,
      impl_->config.successor_mode == SemanticSuccessorMode::delta,
      impl_->config.successor_suffix,
      impl_->config.include_successor_goal_satisfaction,
      builder
   );
}

void SemanticSuccessorHGraphEncoderEngine::encode_views(
   const canonical::FlatRelationViewInput& current,
   const canonical::FlatRelationViewInput& successor,
   BatchBuilder& builder
) const
{
   encode(
      canonical::materialize_semantic_flat_view_input(current),
      canonical::materialize_semantic_flat_view_input(successor),
      builder
   );
}

BatchBuilder::BatchEncoding SemanticSuccessorHGraphEncoderEngine::encode_batch(
   const std::vector< SemanticFlatRelationInput >& currents,
   const std::vector< SemanticFlatRelationInput >& successors
) const
{
   if(currents.size() != successors.size()) {
      throw std::invalid_argument("current and successor batches must have equal length");
   }
   BatchBuilder builder;
   builder.set_graph_kind("hetero");
   for(size_t index = 0; index < currents.size(); ++index) {
      encode(currents[index], successors[index], builder);
      builder.next_graph();
   }
   return builder.build();
}

const SemanticSuccessorHGraphEncoderEngine::Config&
SemanticSuccessorHGraphEncoderEngine::get_config() const
{
   return impl_->config;
}

const std::vector< SemanticPredicateSpec >&
SemanticSuccessorHGraphEncoderEngine::get_predicates() const
{
   return impl_->hgraph.get_predicates();
}

const std::vector< SemanticActionSpec >& SemanticSuccessorHGraphEncoderEngine::get_actions() const
{
   return impl_->hgraph.get_actions();
}

const std::map< std::string, int >&
SemanticSuccessorHGraphEncoderEngine::get_relation_arities() const
{
   return impl_->hgraph.get_relation_arities();
}

void SemanticSuccessorHGraphEncoderEngine::update_relations(
   std::map< std::string, int > relation_arities
)
{
   impl_->hgraph.update_relations(std::move(relation_arities));
}

}  // namespace mifrost
