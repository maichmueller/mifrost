#include "flat_lgan.hpp"

#include <algorithm>
#include <numeric>
#include <set>
#include <stdexcept>

namespace mifrost {

FlatRelationSink::FlatRelationSink(size_t relation_count, bool track_relation_instances)
    : relation_counts_(relation_count, 0),
      relation_args_by_relation_(relation_count),
      relation_instances_by_relation_(track_relation_instances ? relation_count : 0),
      track_relation_instances_(track_relation_instances)
{
}

void FlatRelationSink::emit(int relation_id, std::span< const int64_t > args)
{
   if(relation_id < 0 or static_cast< size_t >(relation_id) >= relation_counts_.size()) {
      throw std::invalid_argument("FlatRelationSink relation id out of range");
   }
   relation_counts_[static_cast< size_t >(relation_id)] += 1;
   auto& bucket = relation_args_by_relation_[static_cast< size_t >(relation_id)];
   bucket.insert(bucket.end(), args.begin(), args.end());
   relation_args_dirty_ = true;
   if(track_relation_instances_) {
      relation_instances_by_relation_[static_cast< size_t >(relation_id)].emplace_back(
         args.begin(), args.end()
      );
   }
}

const std::vector< int64_t >& FlatRelationSink::relation_counts() const
{
   return relation_counts_;
}

const std::vector< int64_t >& FlatRelationSink::relation_args() const
{
   if(relation_args_dirty_) {
      relation_args_.clear();
      size_t total_slots = 0;
      for(const auto& bucket : relation_args_by_relation_) {
         total_slots += bucket.size();
      }
      relation_args_.reserve(total_slots);
      for(const auto& bucket : relation_args_by_relation_) {
         relation_args_.insert(relation_args_.end(), bucket.begin(), bucket.end());
      }
      relation_args_dirty_ = false;
   }
   return relation_args_;
}

int64_t FlatRelationSink::relation_instance_count() const
{
   return std::accumulate(relation_counts_.begin(), relation_counts_.end(), int64_t{0});
}

bool FlatRelationSink::tracks_relation_instances() const
{
   return track_relation_instances_;
}

const std::vector< std::vector< std::vector< int64_t > > >&
FlatRelationSink::relation_instances_by_relation() const
{
   if(not track_relation_instances_) {
      throw std::logic_error(
         "FlatRelationSink relation_instances_by_relation requires tracking to be enabled"
      );
   }
   return relation_instances_by_relation_;
}

namespace {

using RelationIndexSet = hash_set< int64_t >;

std::vector< int64_t > sorted_unique(std::span< const int64_t > values)
{
   std::vector< int64_t > out(values.begin(), values.end());
   std::ranges::sort(out);
   out.erase(std::ranges::unique(out).begin(), out.end());
   return out;
}

}  // namespace

FlatLGANFields
build_flat_lgan(const FlatRelationSink& sink, std::span< const int64_t > anchor_entity_indices)
{
   if(anchor_entity_indices.empty()) {
      throw std::invalid_argument("Flat LGAN requires at least one anchor entity row");
   }
   if(not sink.tracks_relation_instances()) {
      throw std::logic_error(
         "Flat LGAN derivation requires FlatRelationSink relation-instance tracking"
      );
   }

   FlatLGANFields out;
   if(sink.relation_instance_count() == 0) {
      return out;
   }

   const auto& instances_by_relation = sink.relation_instances_by_relation();
   std::vector< int64_t > relation_offsets(instances_by_relation.size() + 1, 0);
   for(size_t relation_id = 0; relation_id < instances_by_relation.size(); ++relation_id) {
      relation_offsets[relation_id + 1] = relation_offsets[relation_id]
                                          + static_cast< int64_t >(
                                             instances_by_relation[relation_id].size()
                                          );
   }

   std::vector< std::vector< int64_t > > entities_by_instance(
      static_cast< size_t >(sink.relation_instance_count())
   );
   hash_map< int64_t, RelationIndexSet > relation_indices_by_entity;
   relation_indices_by_entity.reserve(entities_by_instance.size());

   for(size_t relation_id = 0; relation_id < instances_by_relation.size(); ++relation_id) {
      const auto relation_offset = relation_offsets[relation_id];
      const auto& instances = instances_by_relation[relation_id];
      for(size_t instance_idx = 0; instance_idx < instances.size(); ++instance_idx) {
         const int64_t global_relation_index = relation_offset
                                               + static_cast< int64_t >(instance_idx);
         auto unique_entities = sorted_unique(std::span{instances[instance_idx]});
         entities_by_instance[static_cast< size_t >(global_relation_index)] = unique_entities;
         for(const auto entity_index : unique_entities) {
            relation_indices_by_entity[entity_index].insert(global_relation_index);
         }
      }
   }

   auto sorted_anchor_entities = sorted_unique(anchor_entity_indices);
   std::set< std::pair< int64_t, int64_t > > rr_edges;

   for(const auto anchor_entity_index : sorted_anchor_entities) {
      const auto tn_it = relation_indices_by_entity.find(anchor_entity_index);
      if(tn_it == relation_indices_by_entity.end()) {
         continue;
      }

      std::vector< int64_t > tn_relations(tn_it->second.begin(), tn_it->second.end());
      std::ranges::sort(tn_relations);
      if(tn_relations.empty()) {
         continue;
      }

      hash_set< int64_t > local_entity_set;
      for(const auto relation_index : tn_relations) {
         const auto& entities = entities_by_instance[static_cast< size_t >(relation_index)];
         local_entity_set.insert(entities.begin(), entities.end());
      }
      if(local_entity_set.empty()) {
         continue;
      }

      RelationIndexSet local_relation_candidates;
      for(const auto entity_index : local_entity_set) {
         const auto rel_it = relation_indices_by_entity.find(entity_index);
         if(rel_it == relation_indices_by_entity.end()) {
            continue;
         }
         local_relation_candidates.insert(rel_it->second.begin(), rel_it->second.end());
      }

      std::vector< int64_t > local_relations;
      local_relations.reserve(local_relation_candidates.size());
      for(const auto relation_index : local_relation_candidates) {
         const auto& entities = entities_by_instance[static_cast< size_t >(relation_index)];
         const bool fully_local = std::ranges::all_of(entities, [&](const int64_t entity_index) {
            return local_entity_set.contains(entity_index);
         });
         if(fully_local) {
            local_relations.push_back(relation_index);
         }
      }
      std::ranges::sort(local_relations);

      RelationIndexSet tn_relation_set(tn_relations.begin(), tn_relations.end());
      for(const auto relation_index : tn_relations) {
         out.tn_relation_indices.push_back(relation_index);
         out.tn_entity_indices.push_back(anchor_entity_index);
      }
      for(const auto relation_index : local_relations) {
         if(tn_relation_set.contains(relation_index)) {
            continue;
         }
         out.nn_relation_indices.push_back(relation_index);
         out.nn_entity_indices.push_back(anchor_entity_index);
      }

      hash_map< int64_t, std::vector< int64_t > > local_relations_by_entity;
      local_relations_by_entity.reserve(local_entity_set.size());
      for(const auto relation_index : local_relations) {
         const auto& entities = entities_by_instance[static_cast< size_t >(relation_index)];
         for(const auto entity_index : entities) {
            if(local_entity_set.contains(entity_index)) {
               local_relations_by_entity[entity_index].push_back(relation_index);
            }
         }
      }

      for(auto& [_, relations] : local_relations_by_entity) {
         std::ranges::sort(relations);
         relations.erase(std::ranges::unique(relations).begin(), relations.end());
         if(relations.size() < 2) {
            continue;
         }
         for(size_t i = 0; i < relations.size(); ++i) {
            for(size_t j = i + 1; j < relations.size(); ++j) {
               rr_edges.emplace(relations[i], relations[j]);
               rr_edges.emplace(relations[j], relations[i]);
            }
         }
      }
   }

   out.rr_src_relation_indices.reserve(rr_edges.size());
   out.rr_dst_relation_indices.reserve(rr_edges.size());
   for(const auto& [src_relation_index, dst_relation_index] : rr_edges) {
      out.rr_src_relation_indices.push_back(src_relation_index);
      out.rr_dst_relation_indices.push_back(dst_relation_index);
   }

   return out;
}

}  // namespace mifrost
