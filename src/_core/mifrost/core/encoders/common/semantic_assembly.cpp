#include "semantic_assembly.hpp"

namespace mifrost {

bool SemanticAnnotations::contains(std::string_view key) const
{
   return entries_.contains(std::string(key));
}

}  // namespace mifrost
