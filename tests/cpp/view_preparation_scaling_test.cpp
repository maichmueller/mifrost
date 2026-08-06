/**
 * @file view_preparation_scaling_test.cpp
 * @brief Growth-rate coverage for the compact graph-identity intern pools.
 *
 * `ViewPreparation` interns each unique atom and ground action once and keeps
 * only compact indices in the input lanes. Interning used to linearly scan the
 * pool, which made preparation quadratic in the number of unique identities.
 * These tests pin the *shape* of the cost curve rather than an absolute time,
 * so they stay meaningful on a loaded or slow machine.
 *
 * This file also guards the header layering: it includes
 * `core/views/semantic_preparation.hpp` and nothing else from the encoder
 * stack, so it stops compiling if the View layer regains a dependency on an
 * encoder header.
 */
#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <vector>

#include "mifrost/core/views/semantic_preparation.hpp"

namespace {

using mifrost::SemanticAtom;
using mifrost::SemanticGroundAction;
using mifrost::canonical::detail::ViewPreparation;

[[nodiscard]] SemanticAtom make_atom(int64_t index)
{
   SemanticAtom atom;
   atom.predicate = index % 7;
   atom.arguments.push_back(index);
   atom.arguments.push_back(index * 2);
   return atom;
}

[[nodiscard]] SemanticGroundAction make_action(int64_t index)
{
   SemanticGroundAction action;
   action.action = index % 5;
   action.arguments.push_back(index);
   return action;
}

/** Wall-clock cost of interning `count` distinct atoms into a fresh pool. */
[[nodiscard]] double intern_atoms_seconds(int64_t count)
{
   ViewPreparation preparation;
   const auto start = std::chrono::steady_clock::now();
   for(int64_t index = 0; index < count; ++index) {
      (void) preparation.intern_atom(make_atom(index));
   }
   const auto elapsed = std::chrono::steady_clock::now() - start;
   EXPECT_EQ(preparation.atom_pool.size(), static_cast< size_t >(count));
   return std::chrono::duration< double >(elapsed).count();
}

TEST(ViewPreparationScaling, AtomInterningGrowsApproximatelyLinearly)
{
   constexpr int64_t small = 4'000;
   constexpr int64_t large = 32'000;
   static_assert(large == small * 8);

   // Warm the allocator and the instruction cache so the first measurement is
   // not systematically penalised.
   (void) intern_atoms_seconds(small);

   const auto small_seconds = intern_atoms_seconds(small);
   const auto large_seconds = intern_atoms_seconds(large);

   // Linear growth predicts an 8x ratio for an 8x input; quadratic predicts
   // 64x. Allow a very wide margin for timer noise and rehashing, while still
   // failing loudly if the linear scan comes back.
   const auto floor_seconds = 1e-6;
   const auto ratio = large_seconds / std::max(small_seconds, floor_seconds);
   EXPECT_LT(ratio, 24.0) << "small=" << small_seconds << "s large=" << large_seconds
                          << "s ratio=" << ratio;
}

TEST(ViewPreparationScaling, RepeatedAtomsReuseOnePoolEntry)
{
   ViewPreparation preparation;
   std::vector< size_t > indices;
   for(int repeat = 0; repeat < 3; ++repeat) {
      for(int64_t index = 0; index < 16; ++index) {
         indices.push_back(preparation.intern_atom(make_atom(index)));
      }
   }

   EXPECT_EQ(preparation.atom_pool.size(), 16U);
   ASSERT_EQ(indices.size(), 48U);
   // Every repeat resolves to the same compact index, and pool order is first
   // use order, so emission stays deterministic.
   for(size_t position = 0; position < indices.size(); ++position) {
      EXPECT_EQ(indices[position], position % 16U);
   }
   for(size_t index = 0; index < preparation.atom_pool.size(); ++index) {
      EXPECT_EQ(preparation.atom_pool[index], make_atom(static_cast< int64_t >(index)));
   }
}

TEST(ViewPreparationScaling, RepeatedActionsShareAPoolEntryButKeepOccurrences)
{
   ViewPreparation preparation;
   const std::vector< int64_t > occurrences{0, 1, 0, 2, 1, 0};
   for(const auto value : occurrences) {
      preparation.action_occurrence_indices.push_back(
         preparation.intern_action(make_action(value))
      );
   }

   EXPECT_EQ(preparation.action_pool.size(), 3U);
   ASSERT_EQ(preparation.action_occurrence_indices.size(), occurrences.size());
   // Multiplicity and lane order both survive deduplication.
   const std::vector< size_t > expected{0, 1, 0, 2, 1, 0};
   EXPECT_EQ(preparation.action_occurrence_indices, expected);
}

}  // namespace
