#pragma once

#include <nanobind/nanobind.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "batch_builder.hpp"

namespace mifrost {

namespace nb = nanobind;
using PartsNative = BatchBuilder::PartsNative;

/**
 * @brief CRTP base for stream encoders with static dispatch.
 *
 * The stream encoder API is defined by a Step payload type owned by the
 * derived encoder, avoiding virtual dispatch and ambiguous state-only calls.
 */
template < typename Derived, typename Step >
class StreamEncoderBase {
  public:
   StreamEncoderBase() = default;

   /// Encode one step and cache the resulting parts.
   int64_t append(const Step& step)
   {
      PartsNative parts = encode_step_to_parts(step);
      return insert_entry(std::move(parts));
   }

   /// Remove a cached graph by id.
   void remove(int64_t id)
   {
      if(id < 0 or static_cast< size_t >(id) >= entries_.size()) {
         throw std::out_of_range("StreamEncoderBase::remove invalid id");
      }
      entries_[static_cast< size_t >(id)].active = false;
      entries_[static_cast< size_t >(id)].parts = PartsNative{};
   }

   /// Re-encode a cached graph by id.
   void update(int64_t id, const Step& step)
   {
      if(id < 0 or static_cast< size_t >(id) >= entries_.size()) {
         throw std::out_of_range("StreamEncoderBase::update invalid id");
      }
      if(not entries_[static_cast< size_t >(id)].active) {
         throw std::invalid_argument("StreamEncoderBase::update on inactive entry");
      }
      entries_[static_cast< size_t >(id)].parts = encode_step_to_parts(step);
   }

   /// Build parts for the accumulated graphs (Python dict form).
   nb::dict flush_parts() { return build_merged_builder().build_parts(); }

   /// Build a PyG object for the accumulated graphs.
   nb::object flush() { return build_merged_builder().build(); }

   /// Build parts for the accumulated graphs (native C++ form).
   PartsNative flush_parts_native() { return build_merged_builder().build_parts_native(); }

   /// Reset the stream to an empty cache.
   void reset() { entries_.clear(); }

   /// Toggle reuse of removed ids (replace-in-order semantics).
   void set_reuse_removed(bool value) { reuse_removed_ = value; }

  protected:
   struct Entry {
      PartsNative parts;
      bool active = true;
   };

   int64_t insert_entry(PartsNative parts)
   {
      if(reuse_removed_) {
         for(size_t idx = 0; idx < entries_.size(); ++idx) {
            if(not entries_[idx].active) {
               entries_[idx].parts = std::move(parts);
               entries_[idx].active = true;
               return static_cast< int64_t >(idx);
            }
         }
      }
      entries_.push_back(Entry{std::move(parts), true});
      return static_cast< int64_t >(entries_.size() - 1);
   }

   PartsNative encode_step_to_parts(const Step& step)
   {
      BatchBuilder builder;
      builder.set_graph_kind(std::string(Derived::graph_kind()));
      static_cast< Derived* >(this)->encode_step(step, builder);
      builder.next_graph();
      return builder.build_parts_native();
   }

   BatchBuilder build_merged_builder()
   {
      BatchBuilder builder;
      builder.set_graph_kind(std::string(Derived::graph_kind()));
      for(const auto& entry : entries_) {
         if(not entry.active) {
            continue;
         }
         builder.append_parts(entry.parts);
      }
      return builder;
   }

   std::vector< Entry > entries_;
   bool reuse_removed_ = false;
};

}  // namespace mifrost
