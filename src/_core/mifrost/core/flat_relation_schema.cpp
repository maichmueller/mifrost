#include "flat_relation_schema.hpp"

#include <stdexcept>
#include <utility>

namespace mifrost {

namespace {

bool schema_entries_match(const FlatRelationSchemaEntry& lhs, const FlatRelationSchemaEntry& rhs)
{
   return lhs.layout.logical_arity == rhs.layout.logical_arity
          && lhs.layout.include_predicate_virtual_node == rhs.layout.include_predicate_virtual_node
          && lhs.layout.auxiliary_slot_roles == rhs.layout.auxiliary_slot_roles
          && lhs.source == rhs.source;
}

}  // namespace

void FlatRelationSchemaRegistry::add(std::string name, FlatTupleLayout layout, std::string source)
{
   auto [it, inserted] = entries_.try_emplace(
      std::move(name), FlatRelationSchemaEntry{std::move(layout), std::move(source)}
   );
   if(not inserted) {
      throw std::invalid_argument(
         "Flat relation schema collision for relation '" + it->first + "'"
      );
   }
}

void FlatRelationSchemaRegistry::add_or_validate(
   std::string name,
   FlatTupleLayout layout,
   std::string source
)
{
   auto [it, inserted] = entries_.try_emplace(
      std::move(name), FlatRelationSchemaEntry{std::move(layout), std::move(source)}
   );
   if(inserted) {
      return;
   }
   if(not schema_entries_match(
         it->second, FlatRelationSchemaEntry{std::move(layout), std::move(source)}
      )) {
      throw std::invalid_argument(
         "Flat relation schema collision for relation '" + it->first + "'"
      );
   }
}

bool FlatRelationSchemaRegistry::contains(const std::string& name) const
{
   return entries_.contains(name);
}

size_t FlatRelationSchemaRegistry::size() const
{
   return entries_.size();
}

const std::map< std::string, FlatRelationSchemaEntry >& FlatRelationSchemaRegistry::entries() const
{
   return entries_;
}

FlatRelationSchemaMetadata build_flat_relation_schema_metadata(
   const FlatRelationSchemaRegistry& registry,
   int max_goal_level,
   bool support_literals,
   const std::set< GoalDerivation >& goal_derivations,
   std::string_view empty_error_message
)
{
   FlatRelationSchemaMetadata metadata;
   metadata.slot_role_names = flat_slot_role_names();
   metadata.relation_names.reserve(registry.size());
   metadata.relation_arities.reserve(registry.size());
   metadata.relation_sources.reserve(registry.size());
   metadata.relation_logical_arities.reserve(registry.size());
   metadata.relation_encoded_arities.reserve(registry.size());
   metadata.relation_name_to_id.reserve(registry.size());
   metadata.relation_slot_role_offsets.reserve(registry.size() + 1);
   metadata.relation_slot_role_offsets.push_back(0);

   std::map< std::string, int > relation_dict_arity;
   for(const auto& [name, entry] : registry.entries()) {
      metadata.relation_name_to_id.emplace(
         name, static_cast< int >(metadata.relation_names.size())
      );
      metadata.relation_names.push_back(name);
      metadata.relation_arities.push_back(entry.layout.encoded_arity());
      metadata.relation_sources.push_back(entry.source);
      metadata.relation_logical_arities.push_back(entry.layout.logical_arity);
      metadata.relation_encoded_arities.push_back(entry.layout.encoded_arity());
      const auto slot_roles = entry.layout.slot_role_ids();
      metadata.relation_slot_roles.insert(
         metadata.relation_slot_roles.end(), slot_roles.begin(), slot_roles.end()
      );
      metadata.relation_slot_role_offsets.push_back(
         static_cast< int64_t >(metadata.relation_slot_roles.size())
      );
      relation_dict_arity.emplace(name, entry.layout.encoded_arity());
   }

   if(metadata.relation_names.empty()) {
      throw std::invalid_argument(std::string(empty_error_message));
   }

   metadata.relation_dict = RelationDict(
      std::move(relation_dict_arity), max_goal_level, support_literals, goal_derivations
   );
   return metadata;
}

}  // namespace mifrost
