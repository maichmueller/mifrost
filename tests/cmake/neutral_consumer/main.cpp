#include <mifrost/core/encoders/flat/semantic_flat_horizon_encoder.hpp>
#include <mifrost/core/encoders/flat/semantic_flat_relation_encoder.hpp>

int main()
{
   const mifrost::SemanticPredicateSpec predicate{
      .category = mifrost::SemanticPredicateCategory::fluent,
      .name = "ready",
      .arity = 0,
   };
   const mifrost::SemanticFlatHorizonEncoderConfig horizon;
   return predicate.name == "ready" && horizon.ignore_actions ? 0 : 1;
}
