#include "hetero_relation_schema.hpp"

#include <algorithm>
#include <map>
#include <stdexcept>
#include <utility>

namespace mifrost {

void HeteroRelationSchemaBuilder::register_relation(
   RelationKey key,
   HeteroRelationLayout layout,
   RelationUsage usage
)
{
   const auto it = entries_.find(key);
   if(it == entries_.end()) {
      entries_.emplace(std::move(key), PendingEntry{layout, usage});
      return;
   }
   if(not(it->second.layout == layout)) {
      throw std::invalid_argument(
         "Hetero relation schema arity mismatch for relation '" + format_relation_name(key) + "'"
      );
   }
   // Same key, same arity: idempotent no-op. The first-registered usage is kept.
}

HeteroRelationSchema HeteroRelationSchemaBuilder::finalize(
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
            "Hetero relation schema name collision between distinct relation keys for '"
            + rows[idx].name + "'"
         );
      }
   }

   HeteroRelationSchema schema;
   schema.names_.reserve(rows.size());
   schema.arities_.reserve(rows.size());
   schema.sources_.reserve(rows.size());
   schema.declared_id_by_key_.reserve(rows.size());

   std::map< std::string, int > relation_dict_arity;
   for(size_t idx = 0; idx < rows.size(); ++idx) {
      const auto& row = rows[idx];
      schema.declared_id_by_key_.emplace(*row.key, static_cast< int64_t >(idx));
      schema.names_.push_back(row.name);
      schema.arities_.push_back(row.pending->layout.arity);
      schema.sources_.emplace_back(relation_usage_source_label(row.pending->usage));
      relation_dict_arity.emplace(row.name, static_cast< int >(row.pending->layout.arity));
   }

   schema.relation_dict_ = RelationDict(
      std::move(relation_dict_arity), max_goal_level, support_literals, goal_derivations
   );
   return schema;
}

const std::string& HeteroRelationSchema::name_for(const RelationKey& key) const
{
   if(const auto it = declared_id_by_key_.find(key); it != declared_id_by_key_.end()) {
      return names_[static_cast< size_t >(it->second)];
   }
   if(const auto it = undeclared_name_cache_.find(key); it != undeclared_name_cache_.end()) {
      return it->second;
   }
   const auto [it, _] = undeclared_name_cache_.emplace(key, format_relation_name(key));
   return it->second;
}

}  // namespace mifrost
