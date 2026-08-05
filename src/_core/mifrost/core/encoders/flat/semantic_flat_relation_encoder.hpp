/**
 * @file semantic_flat_relation_encoder.hpp
 * @brief Planning-backend-neutral native flat relation encoder.
 */
#pragma once

#include <algorithm>
#include <boost/container/small_vector.hpp>
#include <compare>
#include <cstdint>
#include <iterator>
#include <memory>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <string>
#include <vector>

#include "flat_relation_config.hpp"
#include "mifrost/core/api.hpp"
#include "mifrost/core/batch_builder.hpp"
#include "mifrost/core/views/concepts.hpp"

namespace mifrost {

class SemanticFlatHorizonEncoderEngine;
class SemanticTransitionDAG;
struct SemanticFlatHorizonEncoderConfig;

/** Category of a predicate in the semantic input schema. */
enum class SemanticPredicateCategory : int64_t {
   static_predicate = 0,
   fluent = 1,
   derived = 2,
};

/** One predicate declaration. Predicate indices refer to this vector. */
struct SemanticPredicateSpec {
   SemanticPredicateCategory category = SemanticPredicateCategory::fluent;
   std::string name;
   int64_t arity = 0;

   auto operator<=>(const SemanticPredicateSpec&) const = default;
};

/** One action declaration. Action indices refer to this vector. */
struct SemanticActionSpec {
   std::string name;
   int64_t arity = 0;

   auto operator<=>(const SemanticActionSpec&) const = default;
};

/**
 * Inline argument storage for common PDDL arities.
 *
 * Four arguments cover the ordinary planning workloads while retaining an
 * owning dynamically sized representation for arbitrary arities. The explicit
 * vector constructor keeps the public compact-input ABI convenient for Python
 * bindings and C++ callers.
 */
class SemanticArguments: public boost::container::small_vector< int64_t, 4 > {
  public:
   using Base = boost::container::small_vector< int64_t, 4 >;
   using Base::Base;

   SemanticArguments() = default;
   SemanticArguments(const std::vector< int64_t >& values) : Base(values.begin(), values.end()) {}
   SemanticArguments(std::vector< int64_t >&& values) : Base(values.begin(), values.end()) {}

   [[nodiscard]] std::strong_ordering operator<=>(const SemanticArguments& other) const
   {
      return std::lexicographical_compare_three_way(
         begin(), end(), other.begin(), other.end(), std::compare_three_way{}
      );
   }
   [[nodiscard]] bool operator==(const SemanticArguments& other) const
   {
      return size() == other.size() and std::equal(begin(), end(), other.begin());
   }
};

/** Ground atom using schema-local predicate and graph-local object indices. */
struct SemanticAtom {
   int64_t predicate = -1;
   SemanticArguments arguments;

   auto operator<=>(const SemanticAtom&) const = default;
};

/** Boost-style hash-combine mixer shared by every semantic-record hash functor. */
inline void mix_semantic_hash(size_t& value, int64_t part)
{
   value ^= std::hash< int64_t >{}(part) + 0x9e3779b97f4a7c15ULL + (value << 6U) + (value >> 2U);
}

/**
 * Hash functor for `SemanticAtom`, for use as `hash_set<SemanticAtom,
 * SemanticAtomHash>`/`hash_map<SemanticAtom, T, SemanticAtomHash>` where a
 * per-state membership set never needs `std::set`'s sorted iteration order
 * (see per-state goal-satisfaction and static/fluent membership tests in
 * semantic_flat_relation_encoder.cpp and semantic_hgraph_encoder.cpp). Do
 * not use this for a set/map whose iteration order is read back to
 * determine emission order (e.g. Horizon/Transition added/removed-fact
 * delta computation): those rely on `SemanticAtom::operator<=>` and
 * `std::set`'s sorted iteration for deterministic output and must stay
 * `std::set`.
 */
struct SemanticAtomHash {
   [[nodiscard]] size_t operator()(const SemanticAtom& atom) const noexcept
   {
      size_t value = 0;
      mix_semantic_hash(value, atom.predicate);
      for(const auto argument : atom.arguments) {
         mix_semantic_hash(value, argument);
      }
      return value;
   }
};

/** Polarized ground literal. Its category is defined by the predicate schema. */
struct SemanticLiteral {
   SemanticAtom atom;
   bool positive = true;

   auto operator<=>(const SemanticLiteral&) const = default;
};

/** Ground action using schema-local action and graph-local object indices. */
struct SemanticGroundAction {
   int64_t action = -1;
   SemanticArguments arguments;

   auto operator<=>(const SemanticGroundAction&) const = default;
};

/** One history row before stable time-delta ordering and distance filtering. */
struct SemanticHistoryEntry {
   int64_t dt = 0;
   std::vector< SemanticLiteral > literals;

   auto operator<=>(const SemanticHistoryEntry&) const = default;
};

/**
 * Immutable schema/problem data shared by all semantic state inputs for one
 * planning task. Backends translate their stable local identities into the
 * compact IDs in this context once during adapter construction.
 */
struct SemanticTaskContext {
   std::vector< SemanticPredicateSpec > predicates;
   std::vector< SemanticActionSpec > actions;
   std::vector< std::string > objects;
   std::vector< SemanticAtom > static_facts;
   std::vector< SemanticLiteral > default_goals;
};

/**
 * @brief Owned semantic input for one flat graph.
 *
 * All numeric indices are deliberately local to this value and its engine
 * schema. They are compact transport into the hot native loop, not persistent
 * or cross-repository identities.
 */
struct SemanticFlatRelationInput {
   /** Shared immutable schema/problem data; absent for legacy standalone inputs. */
   std::shared_ptr< const SemanticTaskContext > task_context;
   /** Legacy object table when no task context is supplied. */
   std::vector< std::string > objects;
   /** Dynamic state facts only when task_context is supplied. */
   std::vector< SemanticAtom > state_facts;
   /** Explicit goal override; defaults come from task_context when requested. */
   std::vector< SemanticLiteral > goals;
   bool use_default_goals = false;
   std::vector< SemanticGroundAction > actions;
   std::vector< std::vector< SemanticLiteral > > subgoal_layers;
   std::vector< SemanticHistoryEntry > history;
   std::optional< int64_t > history_max_steps = std::nullopt;
};

[[nodiscard]] inline const std::vector< std::string >& semantic_objects(
   const SemanticFlatRelationInput& input
)
{
   return input.task_context ? input.task_context->objects : input.objects;
}

[[nodiscard]] inline const std::vector< SemanticLiteral >& semantic_goals(
   const SemanticFlatRelationInput& input
)
{
   return input.task_context and input.use_default_goals ? input.task_context->default_goals
                                                         : input.goals;
}

[[nodiscard]] inline const std::vector< SemanticAtom >& semantic_static_facts(
   const SemanticFlatRelationInput& input
)
{
   static const std::vector< SemanticAtom > empty;
   return input.task_context ? input.task_context->static_facts : empty;
}

/** One sorted literal-to-effective-goal-level entry. */
struct SemanticGoalLevel {
   SemanticLiteral literal;
   size_t level = 0;

   auto operator<=>(const SemanticGoalLevel&) const = default;
};

/**
 * Return a sorted compact lookup for effective goal levels.
 *
 * Later subgoal lanes intentionally override the same literal from an earlier
 * lane, matching the historical map-assignment semantics without one tree node
 * allocation per literal.
 */
[[nodiscard]] inline std::vector< SemanticGoalLevel > semantic_goal_levels(
   const SemanticFlatRelationInput& input
)
{
   const auto& goals = semantic_goals(input);
   size_t count = goals.size();
   for(const auto& layer : input.subgoal_layers) {
      count += layer.size();
   }
   std::vector< SemanticGoalLevel > levels;
   levels.reserve(count);
   for(const auto& literal : goals) {
      levels.push_back({literal, 0});
   }
   for(size_t index = 0; index < input.subgoal_layers.size(); ++index) {
      for(const auto& literal : input.subgoal_layers[index]) {
         levels.push_back({literal, index + 1});
      }
   }
   std::ranges::sort(levels);
   return levels;
}

[[nodiscard]] inline size_t
semantic_goal_level(const std::vector< SemanticGoalLevel >& levels, const SemanticLiteral& literal)
{
   const auto first = std::lower_bound(
      levels.begin(), levels.end(), literal, [](const SemanticGoalLevel& entry, const auto& value) {
         return entry.literal < value;
      }
   );
   if(first == levels.end() or first->literal != literal) {
      throw std::invalid_argument("semantic goal literal is not present in its goal-level lookup");
   }
   const auto past = std::upper_bound(
      first, levels.end(), literal, [](const auto& value, const SemanticGoalLevel& entry) {
         return value < entry.literal;
      }
   );
   return std::prev(past)->level;
}

[[nodiscard]] inline bool semantic_inputs_share_task(
   const SemanticFlatRelationInput& lhs,
   const SemanticFlatRelationInput& rhs
)
{
   return lhs.task_context and lhs.task_context == rhs.task_context;
}

/**
 * @brief Encode owned semantic values directly to native BatchEncoding.
 *
 * The engine owns only predicate/action schema and encoding policy. It never
 * accepts or retains a Mimir, Tyr, or other planning-repository view.
 */
class MIFROST_API SemanticFlatRelationEncoderEngine {
  public:
   using Config = FlatRelationEncoderConfig;

   SemanticFlatRelationEncoderEngine(
      std::vector< SemanticPredicateSpec > predicates,
      std::vector< SemanticActionSpec > actions,
      Config config = {}
   );
   SemanticFlatRelationEncoderEngine(
      std::shared_ptr< const SemanticTaskContext > task_context,
      Config config = {}
   );
   SemanticFlatRelationEncoderEngine(const SemanticFlatRelationEncoderEngine&) = delete;
   SemanticFlatRelationEncoderEngine& operator=(const SemanticFlatRelationEncoderEngine&) = delete;
   SemanticFlatRelationEncoderEngine(SemanticFlatRelationEncoderEngine&&) noexcept;
   SemanticFlatRelationEncoderEngine& operator=(SemanticFlatRelationEncoderEngine&&) noexcept;
   ~SemanticFlatRelationEncoderEngine();

   [[nodiscard]] BatchBuilder::BatchEncoding encode(const SemanticFlatRelationInput& input) const;
   void encode(const SemanticFlatRelationInput& input, BatchBuilder& builder) const;

   template < views::StateView State, views::GroundActionRange Actions >
   [[nodiscard]] BatchBuilder::BatchEncoding encode(const State& state, Actions&& actions) const;

   template < views::StateView State, views::GroundActionRange Actions >
   void encode(const State& state, Actions&& actions, BatchBuilder& builder) const;

   template < views::StateView State, views::LiteralRange Goals, views::GroundActionRange Actions >
   [[nodiscard]] BatchBuilder::BatchEncoding
   encode(const State& state, Goals&& goals, Actions&& actions) const;

   template < views::StateView State, views::LiteralRange Goals, views::GroundActionRange Actions >
   void encode(const State& state, Goals&& goals, Actions&& actions, BatchBuilder& builder) const;

   [[nodiscard]] BatchBuilder::BatchEncoding encode_batch(
      const std::vector< SemanticFlatRelationInput >& inputs
   ) const;
   void finalize_batch_encoding(BatchBuilder::BatchEncoding& encoding) const;

   [[nodiscard]] const Config& get_config() const;
   [[nodiscard]] const std::shared_ptr< const SemanticTaskContext >& get_task_context() const;
   [[nodiscard]] const std::vector< SemanticPredicateSpec >& get_predicates() const;
   [[nodiscard]] const std::vector< SemanticActionSpec >& get_actions() const;
   [[nodiscard]] const std::vector< std::string >& get_relation_names() const;
   [[nodiscard]] const std::vector< int64_t >& get_relation_arities() const;
   [[nodiscard]] const std::vector< std::string >& get_relation_sources() const;
   [[nodiscard]] const std::vector< int64_t >& get_relation_logical_arities() const;
   [[nodiscard]] const std::vector< int64_t >& get_relation_encoded_arities() const;
   [[nodiscard]] const std::vector< int64_t >& get_relation_slot_roles() const;
   [[nodiscard]] const std::vector< int64_t >& get_relation_slot_role_offsets() const;
   [[nodiscard]] const std::vector< std::string >& get_slot_role_names() const;

  private:
   friend class SemanticFlatHorizonEncoderEngine;

   void configure_horizon(const SemanticFlatHorizonEncoderConfig& config);
   void encode_horizon(
      const SemanticTransitionDAG& dag,
      const SemanticFlatHorizonEncoderConfig& config,
      BatchBuilder& builder
   ) const;
   [[nodiscard]] BatchBuilder::BatchEncoding encode_horizon_composed(
      const SemanticTransitionDAG& dag,
      const SemanticFlatHorizonEncoderConfig& config
   ) const;
   [[nodiscard]] BatchBuilder::BatchEncoding encode_horizon_composed_batch(
      const std::vector< SemanticTransitionDAG >& dags,
      const SemanticFlatHorizonEncoderConfig& config
   ) const;
   void finalize_horizon_encoding(
      BatchBuilder::BatchEncoding& encoding,
      const SemanticFlatHorizonEncoderConfig& config
   ) const;

   struct Impl;
   std::unique_ptr< Impl > impl_;
};

}  // namespace mifrost

#include "mifrost/core/encoders/flat/semantic_flat_relation_view_bridge.hpp"
