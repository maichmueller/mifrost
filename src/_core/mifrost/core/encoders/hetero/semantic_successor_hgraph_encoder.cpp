#include "semantic_successor_hgraph_encoder.hpp"

#include <stdexcept>
#include <utility>

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

   Impl(std::shared_ptr< const SemanticSchemaContext > schema, Config encoder_config)
       : config(normalize_successor_config(std::move(encoder_config))),
         hgraph(std::move(schema), base_config(config))
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
   std::shared_ptr< const SemanticSchemaContext > schema,
   Config config
)
    : impl_(std::make_unique< Impl >(std::move(schema), std::move(config)))
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
   const canonical::detail::ViewPreparation& current,
   const canonical::detail::ViewPreparation& successor,
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

BatchBuilder::BatchEncoding SemanticSuccessorHGraphEncoderEngine::encode_batch(
   std::span< const canonical::detail::ViewPreparation* const > currents,
   std::span< const canonical::detail::ViewPreparation* const > successors
) const
{
   if(currents.size() != successors.size()) {
      throw std::invalid_argument("current and successor batches must have equal length");
   }
   const auto& schema = get_schema_context();
   BatchBuilder builder;
   builder.set_graph_kind("hetero");
   for(size_t index = 0; index < currents.size(); ++index) {
      const auto* current = currents[index];
      const auto* successor = successors[index];
      if(current == nullptr or successor == nullptr) {
         throw std::invalid_argument("semantic successor batch preparations must not be null");
      }
      require_semantic_schema_compatible(
         current->problem_context, schema, "semantic successor batch preparation"
      );
      require_semantic_schema_compatible(
         successor->problem_context, schema, "semantic successor batch preparation"
      );
      // A transition is one instance's: the two lanes share an object table, so
      // the pair must agree even though pairs across the batch need not.
      if(current->problem_context != successor->problem_context) {
         throw std::invalid_argument(
            "semantic successor batch pairs must come from one problem per transition"
         );
      }
      encode_views(*current, *successor, builder);
      builder.next_graph();
   }
   return builder.build();
}

const SemanticSuccessorHGraphEncoderEngine::Config&
SemanticSuccessorHGraphEncoderEngine::get_config() const
{
   return impl_->config;
}

const std::shared_ptr< const SemanticSchemaContext >&
SemanticSuccessorHGraphEncoderEngine::get_schema_context() const
{
   return impl_->hgraph.get_schema_context();
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
