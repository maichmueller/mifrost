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
#include <memory>
#include <span>
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

/**
 * An atom lane that is only good for one traversal.
 *
 * `StateView` promises no more than an `input_range`, so a backend is entitled
 * to back a lane with a single-pass source. This View makes that explicit: the
 * first request yields the atoms, every later one yields nothing. Preparation
 * sizes and traverses each lane, and if it requests the lane twice it silently
 * encodes an empty state.
 */
struct ProbeAtom {
   const SemanticAtom* atom = nullptr;

   [[nodiscard]] mifrost::views::PredicateId predicate_id() const { return atom->predicate; }
   [[nodiscard]] std::span< const int64_t > arguments() const
   {
      return std::span< const int64_t >{atom->arguments.data(), atom->arguments.size()};
   }
};

class SinglePassAtoms {
  public:
   SinglePassAtoms() = default;
   explicit SinglePassAtoms(std::span< const SemanticAtom > atoms) : atoms_(atoms) {}

   [[nodiscard]] auto begin() const { return Iterator{atoms_.data()}; }
   [[nodiscard]] auto end() const { return Iterator{atoms_.data() + atoms_.size()}; }
   [[nodiscard]] size_t size() const { return atoms_.size(); }

  private:
   struct Iterator {
      using difference_type = std::ptrdiff_t;
      using value_type = ProbeAtom;

      const SemanticAtom* position = nullptr;

      ProbeAtom operator*() const { return ProbeAtom{position}; }
      Iterator& operator++()
      {
         ++position;
         return *this;
      }
      void operator++(int) { ++position; }
      bool operator==(const Iterator& other) const { return position == other.position; }
   };

   std::span< const SemanticAtom > atoms_;
};

class SinglePassState: public mifrost::views::StateViewBase< SinglePassState > {
  public:
   SinglePassState(std::span< const SemanticAtom > fluent, std::span< const SemanticAtom > derived)
       : fluent_(fluent), derived_(derived)
   {
   }

   [[nodiscard]] SinglePassAtoms fluent_atoms() const { return take(fluent_, fluent_taken_); }
   [[nodiscard]] SinglePassAtoms derived_atoms() const { return take(derived_, derived_taken_); }

   [[nodiscard]] int fluent_requests() const { return fluent_requests_; }
   [[nodiscard]] int derived_requests() const { return derived_requests_; }

  private:
   SinglePassAtoms take(std::span< const SemanticAtom >& lane, bool& taken) const
   {
      (&lane == &fluent_ ? fluent_requests_ : derived_requests_) += 1;
      if(taken) {
         return SinglePassAtoms{};
      }
      taken = true;
      return SinglePassAtoms{lane};
   }

   mutable std::span< const SemanticAtom > fluent_;
   mutable std::span< const SemanticAtom > derived_;
   mutable bool fluent_taken_ = false;
   mutable bool derived_taken_ = false;
   mutable int fluent_requests_ = 0;
   mutable int derived_requests_ = 0;
};

static_assert(mifrost::views::StateView< SinglePassState >);

TEST(ViewPreparationContract, StateLanesAreRequestedExactlyOnce)
{
   std::vector< SemanticAtom > fluent{make_atom(1), make_atom(2), make_atom(3)};
   std::vector< SemanticAtom > derived{make_atom(4), make_atom(5)};
   const SinglePassState state{std::span{fluent}, std::span{derived}};
   const auto context = std::make_shared< const mifrost::SemanticTaskContext >();

   const auto prepared = mifrost::canonical::detail::make_state_only_view_preparation(
      context, state
   );

   EXPECT_EQ(state.fluent_requests(), 1);
   EXPECT_EQ(state.derived_requests(), 1);
   EXPECT_EQ(prepared.state_facts.size(), fluent.size() + derived.size());
}

}  // namespace
