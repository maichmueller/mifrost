#include "semantic_flat_relation_view_bridge.hpp"

namespace mifrost::canonical {

void require_semantic_view_context(const std::shared_ptr< const SemanticTaskContext >& context)
{
   if(not context) {
      throw std::invalid_argument("semantic View task context must not be null");
   }
}

}  // namespace mifrost::canonical
