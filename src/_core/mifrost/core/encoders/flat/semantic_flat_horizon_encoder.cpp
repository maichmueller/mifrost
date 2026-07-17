#include "semantic_flat_horizon_encoder.hpp"

#include <algorithm>
#include <utility>

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

}  // namespace

struct SemanticFlatHorizonEncoderEngine::Impl {
   Config config;
   SemanticFlatRelationEncoderEngine flat;

   Impl(
      std::vector< SemanticPredicateSpec > predicates,
      std::vector< SemanticActionSpec > actions,
      Config encoder_config
   )
       : config(normalize_config(std::move(encoder_config))),
         flat(std::move(predicates), std::move(actions), base_config(config))
   {
      flat.configure_horizon(config);
   }

   Impl(std::shared_ptr< const SemanticTaskContext > task_context, Config encoder_config)
       : config(normalize_config(std::move(encoder_config))),
         flat(std::move(task_context), base_config(config))
   {
      flat.configure_horizon(config);
   }
};

SemanticFlatHorizonEncoderEngine::SemanticFlatHorizonEncoderEngine(
   std::vector< SemanticPredicateSpec > predicates,
   std::vector< SemanticActionSpec > actions,
   Config config
)
    : impl_(std::make_unique< Impl >(std::move(predicates), std::move(actions), std::move(config)))
{
}

SemanticFlatHorizonEncoderEngine::SemanticFlatHorizonEncoderEngine(
   std::shared_ptr< const SemanticTaskContext > task_context,
   Config config
)
    : impl_(std::make_unique< Impl >(std::move(task_context), std::move(config)))
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
   BatchBuilder builder;
   builder.set_graph_kind("flat");
   impl_->flat.prepare_horizon_builder(builder, impl_->config);
   impl_->flat.encode_horizon(dag, impl_->config, builder);
   builder.next_graph();
   auto encoding = builder.build();
   finalize_batch_encoding(encoding);
   return encoding;
}

void SemanticFlatHorizonEncoderEngine::encode(
   const SemanticTransitionDAG& dag,
   BatchBuilder& builder
) const
{
   impl_->flat.prepare_horizon_builder(builder, impl_->config);
   impl_->flat.encode_horizon(dag, impl_->config, builder);
}

BatchBuilder::BatchEncoding SemanticFlatHorizonEncoderEngine::encode_batch(
   const std::vector< SemanticTransitionDAG >& dags
) const
{
   BatchBuilder builder;
   builder.set_graph_kind("flat");
   impl_->flat.prepare_horizon_builder(builder, impl_->config);
   for(const auto& dag : dags) {
      impl_->flat.encode_horizon(dag, impl_->config, builder);
      builder.next_graph();
   }
   auto encoding = builder.build();
   finalize_batch_encoding(encoding);
   return encoding;
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

}  // namespace mifrost
