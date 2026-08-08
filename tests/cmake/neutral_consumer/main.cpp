#include <mifrost/core/encoders/flat/semantic_flat_horizon_encoder.hpp>
#include <mifrost/core/encoders/flat/semantic_flat_relation_encoder.hpp>
#include <utility>
#include <vector>

int main()
{
   const mifrost::SemanticPredicateSpec predicate{
      .category = mifrost::SemanticPredicateCategory::fluent,
      .name = "ready",
      .arity = 1,
   };
   mifrost::SemanticAnnotations annotations;
   annotations.emplace< int >("sdk_probe", 7);

   mifrost::SemanticFlatRelationAssemblyBuilder relation_builder(
      std::vector{predicate}, std::vector< mifrost::SemanticActionSpec >{}
   );
   auto relation = std::move(relation_builder).compile();

   mifrost::SemanticFlatHorizonAssemblyBuilder horizon_builder(
      std::vector{predicate}, std::vector< mifrost::SemanticActionSpec >{}
   );
   auto horizon = std::move(horizon_builder).compile();

   return predicate.name == "ready" && annotations.contains("sdk_probe")
                && not relation.get_relation_names().empty()
                && not horizon.get_relation_names().empty()
             ? 0
             : 1;
}
