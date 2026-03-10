#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "common_types.hpp"

namespace mifrost {

class FlatRelationSink {
  public:
   explicit FlatRelationSink(size_t relation_count, bool track_relation_instances = false);

   void emit(int relation_id, std::span< const int64_t > args);

   [[nodiscard]] const std::vector< int64_t >& relation_counts() const;
   [[nodiscard]] const std::vector< int64_t >& relation_args() const;
   [[nodiscard]] int64_t relation_instance_count() const;
   [[nodiscard]] bool tracks_relation_instances() const;
   [[nodiscard]] const std::vector< std::vector< std::vector< int64_t > > >&
   relation_instances_by_relation() const;

  private:
   std::vector< int64_t > relation_counts_;
   mutable std::vector< std::vector< int64_t > > relation_args_by_relation_;
   std::vector< std::vector< std::vector< int64_t > > > relation_instances_by_relation_;
   mutable std::vector< int64_t > relation_args_;
   mutable bool relation_args_dirty_ = false;
   bool track_relation_instances_ = false;
};

struct FlatLGANFields {
   std::vector< int64_t > tn_relation_indices;
   std::vector< int64_t > tn_entity_indices;
   std::vector< int64_t > nn_relation_indices;
   std::vector< int64_t > nn_entity_indices;
   std::vector< int64_t > rr_src_relation_indices;
   std::vector< int64_t > rr_dst_relation_indices;
};

[[nodiscard]] FlatLGANFields
build_flat_lgan(const FlatRelationSink& sink, std::span< const int64_t > anchor_entity_indices);

}  // namespace mifrost
