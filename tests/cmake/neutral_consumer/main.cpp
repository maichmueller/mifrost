#include <mifrost/core/encoders/flat/semantic_flat_relation_encoder.hpp>

int main()
{
   const mifrost::SemanticPredicateSpec predicate{
      .category = mifrost::SemanticPredicateCategory::fluent,
      .name = "ready",
      .arity = 0,
   };
   return predicate.name == "ready" ? 0 : 1;
}
