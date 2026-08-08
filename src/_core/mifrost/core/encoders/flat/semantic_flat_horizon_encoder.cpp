#include "semantic_flat_horizon_encoder.hpp"

#include <algorithm>
#include <utility>

#include "flat_composition.hpp"

namespace mifrost {
namespace {

FlatRelationEncoderConfig base_config(const SemanticFlatHorizonEncoderConfig& config)
{
   FlatRelationEncoderConfig result = config;
   std::erase_if(result.goal_derivations, [](const GoalDerivation derivation) {
      return derivation == GoalDerivation::added_satisfied
             or derivation == GoalDerivation::added_unsatisfied;
   });
   result.target_sources.clear();
   result.lgan_anchor_sources.clear();
   return result;
}

SemanticFlatHorizonEncoderConfig normalize_config(SemanticFlatHorizonEncoderConfig config)
{
   if(config.transition_mode == SemanticHorizonMode::delta) {
      config.support_literals = true;
   }
   return config;
}

std::shared_ptr< const SemanticSchemaContext > make_schema_context(
   std::vector< SemanticPredicateSpec > predicates,
   std::vector< SemanticActionSpec > actions
)
{
   return std::make_shared< SemanticSchemaContext >(SemanticSchemaContext{
      .predicates = std::move(predicates),
      .actions = std::move(actions),
   });
}

}  // namespace

bool SemanticFlatHorizonAnnotations::contains(std::string_view key) const
{
   return SemanticAnnotations::contains(key);
}

SemanticFlatHorizonInput::SemanticFlatHorizonInput(
   const SemanticTransitionDAG& graph,
   SemanticFlatHorizonAnnotations annotations
)
    : Base(graph, std::move(annotations))
{
}

SemanticFlatHorizonInput::SemanticFlatHorizonInput(
   std::shared_ptr< const SemanticTransitionDAG > graph,
   SemanticFlatHorizonAnnotations annotations
)
    : Base(std::move(graph), std::move(annotations))
{
}

const SemanticTransitionDAG& SemanticFlatHorizonInput::graph() const
{
   return source();
}

struct SemanticFlatHorizonEncoderEngine::Impl {
   Config config;
   SemanticFlatRelationEncoderEngine flat;

   Impl(
      std::shared_ptr< const SemanticSchemaContext > schema,
      Config encoder_config,
      std::vector< std::shared_ptr< FlatEmitterComponent > > components
   )
       : config(normalize_config(std::move(encoder_config))),
         flat(
            detail::SemanticFlatHorizonRelationEngineAccess::make(
               std::move(schema),
               base_config(config)
            )
         )
   {
      flat.configure_horizon(config, std::move(components));
   }
};

struct SemanticFlatHorizonAssemblyBuilder::Impl {
   std::shared_ptr< const SemanticSchemaContext > schema;
   Config config;
   SemanticAssemblyComponents< FlatEmitterComponent > components;
};

SemanticFlatHorizonEncoderEngine::SemanticFlatHorizonEncoderEngine(
   std::vector< SemanticPredicateSpec > predicates,
   std::vector< SemanticActionSpec > actions,
   Config config
)
    : SemanticFlatHorizonEncoderEngine(
         make_schema_context(std::move(predicates), std::move(actions)),
         std::move(config),
         {}
      )
{
}

SemanticFlatHorizonEncoderEngine::SemanticFlatHorizonEncoderEngine(
   std::shared_ptr< const SemanticSchemaContext > schema,
   Config config
)
    : SemanticFlatHorizonEncoderEngine(std::move(schema), std::move(config), {})
{
}

SemanticFlatHorizonEncoderEngine::SemanticFlatHorizonEncoderEngine(
   std::shared_ptr< const SemanticSchemaContext > schema,
   Config config,
   std::vector< std::shared_ptr< FlatEmitterComponent > > components
)
    : impl_(std::make_unique< Impl >(std::move(schema), std::move(config), std::move(components)))
{
}

SemanticFlatHorizonEncoderEngine::SemanticFlatHorizonEncoderEngine(
   SemanticFlatHorizonEncoderEngine&&
) noexcept = default;
SemanticFlatHorizonEncoderEngine& SemanticFlatHorizonEncoderEngine::operator=(
   SemanticFlatHorizonEncoderEngine&&
) noexcept = default;
SemanticFlatHorizonEncoderEngine::~SemanticFlatHorizonEncoderEngine() = default;

BatchBuilder::BatchEncoding SemanticFlatHorizonEncoderEngine::encode(
   const SemanticTransitionDAG& dag
) const
{
   return encode(SemanticFlatHorizonInput(dag));
}

BatchBuilder::BatchEncoding SemanticFlatHorizonEncoderEngine::encode(
   const SemanticFlatHorizonInput& input
) const
{
   return impl_->flat.encode_horizon_composed(input, impl_->config);
}

void SemanticFlatHorizonEncoderEngine::encode(
   const SemanticTransitionDAG& dag,
   BatchBuilder& builder
) const
{
   encode(SemanticFlatHorizonInput(dag), builder);
}

void SemanticFlatHorizonEncoderEngine::encode(
   const SemanticFlatHorizonInput& input,
   BatchBuilder& builder
) const
{
   impl_->flat.encode_horizon(input, impl_->config, builder);
}

BatchBuilder::BatchEncoding SemanticFlatHorizonEncoderEngine::encode_batch(
   const std::vector< SemanticTransitionDAG >& dags
) const
{
   std::vector< SemanticFlatHorizonInput > inputs;
   inputs.reserve(dags.size());
   for(const auto& dag : dags) {
      inputs.emplace_back(dag);
   }
   return encode_batch(inputs);
}

BatchBuilder::BatchEncoding SemanticFlatHorizonEncoderEngine::encode_batch(
   std::initializer_list< SemanticTransitionDAG > dags
) const
{
   return encode_batch(std::vector< SemanticTransitionDAG >(dags));
}

BatchBuilder::BatchEncoding SemanticFlatHorizonEncoderEngine::encode_batch(
   std::span< const SemanticFlatHorizonInput > inputs
) const
{
   return impl_->flat.encode_horizon_composed_batch(inputs, impl_->config);
}

void SemanticFlatHorizonEncoderEngine::finalize_batch_encoding(
   BatchBuilder::BatchEncoding& encoding
) const
{
   impl_->flat.finalize_horizon_encoding(encoding, impl_->config);
}

const SemanticFlatHorizonEncoderEngine::Config& SemanticFlatHorizonEncoderEngine::get_config() const
{
   return impl_->config;
}

const std::vector< SemanticPredicateSpec >& SemanticFlatHorizonEncoderEngine::get_predicates() const
{
   return impl_->flat.get_predicates();
}

const std::vector< SemanticActionSpec >& SemanticFlatHorizonEncoderEngine::get_actions() const
{
   return impl_->flat.get_actions();
}

const std::vector< std::string >& SemanticFlatHorizonEncoderEngine::get_relation_names() const
{
   return impl_->flat.get_relation_names();
}

const std::vector< int64_t >& SemanticFlatHorizonEncoderEngine::get_relation_arities() const
{
   return impl_->flat.get_relation_arities();
}

const std::vector< std::string >& SemanticFlatHorizonEncoderEngine::get_relation_sources() const
{
   return impl_->flat.get_relation_sources();
}

const std::vector< int64_t >& SemanticFlatHorizonEncoderEngine::get_relation_logical_arities() const
{
   return impl_->flat.get_relation_logical_arities();
}

const std::vector< int64_t >& SemanticFlatHorizonEncoderEngine::get_relation_encoded_arities() const
{
   return impl_->flat.get_relation_encoded_arities();
}

const std::vector< int64_t >& SemanticFlatHorizonEncoderEngine::get_relation_slot_roles() const
{
   return impl_->flat.get_relation_slot_roles();
}

const std::vector< int64_t >&
SemanticFlatHorizonEncoderEngine::get_relation_slot_role_offsets() const
{
   return impl_->flat.get_relation_slot_role_offsets();
}

const std::vector< std::string >& SemanticFlatHorizonEncoderEngine::get_slot_role_names() const
{
   return impl_->flat.get_slot_role_names();
}

SemanticFlatHorizonAssemblyBuilder::SemanticFlatHorizonAssemblyBuilder(
   std::vector< SemanticPredicateSpec > predicates,
   std::vector< SemanticActionSpec > actions,
   Config config
)
    : SemanticFlatHorizonAssemblyBuilder(
         make_schema_context(std::move(predicates), std::move(actions)),
         std::move(config)
      )
{
}

SemanticFlatHorizonAssemblyBuilder::SemanticFlatHorizonAssemblyBuilder(
   std::shared_ptr< const SemanticSchemaContext > schema,
   Config config
)
    : impl_(
         std::make_unique< Impl >(
            Impl{.schema = std::move(schema), .config = std::move(config), .components = {}}
         )
      )
{
   if(not impl_->schema) {
      throw std::invalid_argument("Semantic Horizon assembly schema must not be null");
   }
}

SemanticFlatHorizonAssemblyBuilder::SemanticFlatHorizonAssemblyBuilder(
   SemanticFlatHorizonAssemblyBuilder&&
) noexcept = default;

SemanticFlatHorizonAssemblyBuilder& SemanticFlatHorizonAssemblyBuilder::operator=(
   SemanticFlatHorizonAssemblyBuilder&&
) noexcept = default;

SemanticFlatHorizonAssemblyBuilder::~SemanticFlatHorizonAssemblyBuilder() = default;

void SemanticFlatHorizonAssemblyBuilder::add_component(
   std::unique_ptr< FlatEmitterComponent > component
)
{
   if(not impl_) {
      throw std::logic_error("Semantic Horizon assembly builder was already compiled");
   }
   impl_->components.add(std::move(component));
}

SemanticFlatHorizonEncoderEngine SemanticFlatHorizonAssemblyBuilder::compile() &&
{
   if(not impl_) {
      throw std::logic_error("Semantic Horizon assembly builder was already compiled");
   }
   auto state = std::move(impl_);
   auto components = std::move(state->components).freeze();
   return SemanticFlatHorizonEncoderEngine(
      std::move(state->schema), std::move(state->config), std::move(components)
   );
}

}  // namespace mifrost
