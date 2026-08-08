/**
 * @file flat_relation_schema.cpp
 * @brief Flat relation-schema builder and immutable-schema implementation.
 */
#include "flat_relation_schema.hpp"

#include <algorithm>
#include <map>
#include <stdexcept>
#include <utility>

namespace mifrost {

namespace {

bool layouts_compatible(const FlatTupleLayout& lhs, const FlatTupleLayout& rhs)
{
   return lhs.logical_arity == rhs.logical_arity
          && lhs.include_predicate_virtual_node == rhs.include_predicate_virtual_node
          && lhs.auxiliary_slot_roles == rhs.auxiliary_slot_roles;
}

}  // namespace

void FlatRelationSchemaBuilder::register_relation(
   RelationKey key,
   FlatTupleLayout layout,
   RelationUsage usage
)
{
   const auto it = entries_.find(key);
   if(it == entries_.end()) {
      entries_.emplace(std::move(key), PendingEntry{std::move(layout), usage});
      return;
   }
   if(not layouts_compatible(it->second.layout, layout)) {
      throw std::invalid_argument(
         "Flat relation schema layout mismatch for relation '" + format_relation_name(key) + "'"
      );
   }
   if(it->second.usage != usage) {
      throw std::invalid_argument(
         "Flat relation schema usage mismatch for relation '" + format_relation_name(key) + "'"
      );
   }
}

FlatRelationSchema FlatRelationSchemaBuilder::finalize(
   int max_goal_level,
   bool support_literals,
   const std::set< GoalDerivation >& goal_derivations,
   std::string_view empty_error_message
) &&
{
   if(entries_.empty()) {
      throw std::invalid_argument(std::string(empty_error_message));
   }

   struct Row {
      std::string name;
      const RelationKey* key = nullptr;
      const PendingEntry* pending = nullptr;
   };
   std::vector< Row > rows;
   rows.reserve(entries_.size());
   for(const auto& [key, pending] : entries_) {
      rows.push_back(Row{format_relation_name(key), &key, &pending});
   }
   std::ranges::sort(rows, {}, &Row::name);

   for(size_t idx = 1; idx < rows.size(); ++idx) {
      if(rows[idx].name == rows[idx - 1].name) {
         throw std::invalid_argument(
            "Flat relation schema name collision between distinct relation keys for '"
            + rows[idx].name + "'"
         );
      }
   }

   FlatRelationSchema schema;
   auto& metadata = schema.metadata_;
   metadata.slot_role_names = flat_slot_role_names();
   metadata.relation_names.reserve(rows.size());
   metadata.relation_arities.reserve(rows.size());
   metadata.relation_sources.reserve(rows.size());
   metadata.relation_logical_arities.reserve(rows.size());
   metadata.relation_encoded_arities.reserve(rows.size());
   metadata.relation_name_to_id.reserve(rows.size());
   metadata.relation_slot_role_offsets.reserve(rows.size() + 1);
   metadata.relation_slot_role_offsets.push_back(0);
   schema.key_to_id_.reserve(rows.size());

   std::map< std::string, int > relation_dict_arity;
   for(size_t idx = 0; idx < rows.size(); ++idx) {
      const auto& row = rows[idx];
      const auto relation_id = static_cast< int >(idx);
      metadata.relation_name_to_id.emplace(row.name, relation_id);
      schema.key_to_id_.emplace(*row.key, relation_id);
      metadata.relation_names.push_back(row.name);
      metadata.relation_arities.push_back(row.pending->layout.encoded_arity());
      metadata.relation_sources.emplace_back(relation_usage_source_label(row.pending->usage));
      metadata.relation_logical_arities.push_back(row.pending->layout.logical_arity);
      metadata.relation_encoded_arities.push_back(row.pending->layout.encoded_arity());
      const auto slot_roles = row.pending->layout.slot_role_ids();
      metadata.relation_slot_roles.insert(
         metadata.relation_slot_roles.end(), slot_roles.begin(), slot_roles.end()
      );
      metadata.relation_slot_role_offsets.push_back(
         static_cast< int64_t >(metadata.relation_slot_roles.size())
      );
      relation_dict_arity.emplace(row.name, row.pending->layout.encoded_arity());
   }

   metadata.relation_dict = RelationDict(
      std::move(relation_dict_arity), max_goal_level, support_literals, goal_derivations
   );
   return schema;
}

int FlatRelationSchema::id_for(const RelationKey& key) const
{
   const auto it = key_to_id_.find(key);
   if(it == key_to_id_.end()) {
      throw std::invalid_argument("Unknown flat relation key '" + format_relation_name(key) + "'");
   }
   return it->second;
}

std::optional< int > FlatRelationSchema::try_id_for(const RelationKey& key) const
{
   const auto it = key_to_id_.find(key);
   if(it == key_to_id_.end()) {
      return std::nullopt;
   }
   return it->second;
}

const RelationKey& FlatRelationSchema::key_for_id(int relation_id) const
{
   if(relation_id < 0 or static_cast< size_t >(relation_id) >= size()) {
      throw std::out_of_range("Flat relation id is out of range");
   }
   const auto entry = std::ranges::find(key_to_id_, relation_id, [](const auto& item) {
      return item.second;
   });
   if(entry == key_to_id_.end()) {
      throw std::logic_error("Flat relation schema is missing a structured key");
   }
   return entry->first;
}

int FlatRelationSchema::id_for(const std::string& name) const
{
   const auto it = metadata_.relation_name_to_id.find(name);
   if(it == metadata_.relation_name_to_id.end()) {
      throw std::invalid_argument("Unknown flat relation name '" + name + "'");
   }
   return it->second;
}

std::optional< int > FlatRelationSchema::try_id_for(const std::string& name) const
{
   const auto it = metadata_.relation_name_to_id.find(name);
   if(it == metadata_.relation_name_to_id.end()) {
      return std::nullopt;
   }
   return it->second;
}

}  // namespace mifrost
