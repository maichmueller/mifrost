#include "relation_catalog.hpp"

namespace mifrost {

void RelationCatalog::add(RelationSpec spec)
{
   specs_.push_back(std::move(spec));
}

RelationCatalog duplicate_predicate_relations_with_modifier(
   const RelationCatalog& catalog,
   std::string_view modifier
)
{
   RelationCatalog out;
   for(const auto& spec : catalog.specs()) {
      out.add(spec);
      if(spec.key.family != RelationFamily::predicate) {
         continue;
      }
      RelationSpec duplicate = spec;
      duplicate.key.modifiers.emplace_back(modifier);
      out.add(std::move(duplicate));
   }
   return out;
}

}  // namespace mifrost
