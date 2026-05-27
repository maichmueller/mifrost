/**
 * @file stream_encoder_base.hpp
 * @brief Shared CRTP base for stream encoders that cache and merge steps.
 *
 * Each stream encoder provides a small Step struct and an `encode_step`
 * function. This base handles lifecycle, id reuse, and
 * batch concatenation without introducing virtual dispatch.
 */
#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "mifrost/core/batch_builder.hpp"

#if defined(MIFROST_ENABLE_PYTHON_API)
   #include <nanobind/nanobind.h>
#endif

namespace mifrost {

using BatchEncoding = BatchBuilder::BatchEncoding;

/**
 * @brief CRTP base for stream encoders with static dispatch.
 *
 * The stream encoder API is defined by a Step type owned by the
 * derived encoder, avoiding virtual dispatch and ambiguous state-only calls.
 *
 * Invariant:
 *  - Each active entry stores a complete single-graph `BatchEncoding`.
 *  - `flush()` concatenates those encodings in insertion order, skipping
 *    inactive entries.
 */
template < typename Derived, typename Step >
class StreamEncoderBase {
  public:
   StreamEncoderBase() = default;

   /// Encode one step and cache the resulting batch encoding.
   int64_t append(const Step& step)
   {
      BatchEncoding batch_encoding = encode_step_to_batch_encoding(step);
      return insert_entry(std::move(batch_encoding));
   }

   /// Remove a cached graph by id.
   void remove(int64_t id)
   {
      if(id < 0 or static_cast< size_t >(id) >= entries_.size()) {
         throw std::out_of_range("StreamEncoderBase::remove invalid id");
      }
      entries_[static_cast< size_t >(id)].active = false;
      entries_[static_cast< size_t >(id)].batch_encoding = BatchEncoding{};
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
      entries_[static_cast< size_t >(id)].batch_encoding = encode_step_to_batch_encoding(step);
   }

   /// Build native batch encoding for the accumulated graphs.
   BatchEncoding flush() { return build_merged_builder().build(); }

#if defined(MIFROST_ENABLE_PYTHON_API)
   /// Build a PyG Batch for the accumulated graphs.
   nanobind::object flush_pyg() { return build_merged_builder().build_pyg(); }
#endif

   /// Reset the stream to an empty cache.
   void reset() { entries_.clear(); }

   /// Toggle reuse of removed ids (replace-in-order semantics).
   void set_reuse_removed(bool value) { reuse_removed_ = value; }

  protected:
   struct Entry {
      BatchEncoding batch_encoding;
      bool active = true;
   };

   int64_t insert_entry(BatchEncoding batch_encoding)
   {
      if(reuse_removed_) {
         for(size_t idx = 0; idx < entries_.size(); ++idx) {
            if(not entries_[idx].active) {
               entries_[idx].batch_encoding = std::move(batch_encoding);
               entries_[idx].active = true;
               return static_cast< int64_t >(idx);
            }
         }
      }
      entries_.push_back(Entry{std::move(batch_encoding), true});
      return static_cast< int64_t >(entries_.size() - 1);
   }

   BatchEncoding encode_step_to_batch_encoding(const Step& step)
   {
      BatchBuilder builder;
      builder.set_graph_kind(std::string(Derived::graph_kind()));
      static_cast< Derived* >(this)->encode_step(step, builder);
      builder.next_graph();
      return builder.build();
   }

   BatchBuilder build_merged_builder()
   {
      BatchBuilder builder;
      builder.set_graph_kind(std::string(Derived::graph_kind()));
      for(const auto& entry : entries_) {
         if(not entry.active) {
            continue;
         }
         builder.append_batch_encoding(entry.batch_encoding);
      }
      return builder;
   }

   std::vector< Entry > entries_;
   bool reuse_removed_ = false;
};

}  // namespace mifrost
