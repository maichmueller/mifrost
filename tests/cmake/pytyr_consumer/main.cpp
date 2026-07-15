#include <mifrost/backends/pytyr/semantic_flat_encoder.hpp>

int main()
{
   const auto method = &mifrost::pytyr::SemanticFlatRelationEncoder::get_engine;
   return method == nullptr ? 1 : 0;
}
