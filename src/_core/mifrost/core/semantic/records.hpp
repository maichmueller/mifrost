/**
 * @file records.hpp
 * @brief Backend-neutral semantic record and key definitions.
 *
 * This is the bottom layer of the encoder stack. It declares the owned
 * semantic value types (atoms, literals, ground actions, history entries, the
 * shared schema and problem contexts and the owning
 * `SemanticFlatRelationInput` compatibility DTO) plus their hash/ordering
 * helpers, and nothing else: no encoder engine, no View adapter, no batch
 * machinery.
 *
 * Keeping these definitions here is what lets the View layer
 * (`core/views/semantic_preparation.hpp`) describe borrowed inputs in terms of
 * semantic records without including a *encoder* header. The dependency
 * direction is:
 *
 *     core/semantic/records.hpp
 *         -> core/views/{concepts,semantic_preparation}.hpp
 *             -> core/encoders/<family>/...
 *                 -> backends/<backend>/...
 *
 * `core/encoders/flat/semantic_flat_relation_encoder.hpp` includes this header
 * and re-exports every name from namespace `mifrost`, so existing includes and
 * the exported ABI are unchanged.
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
#include <string_view>
#include <utility>
#include <vector>

namespace mifrost {

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

/**
 * Hash functor for `SemanticGroundAction`. Same caveat as `SemanticAtomHash`:
 * only for membership/identity lookup, never for a container whose iteration
 * order determines emission order.
 */
struct SemanticGroundActionHash {
   [[nodiscard]] size_t operator()(const SemanticGroundAction& action) const noexcept
   {
      size_t value = 0;
      mix_semantic_hash(value, action.action);
      for(const auto argument : action.arguments) {
         mix_semantic_hash(value, argument);
      }
      return value;
   }
};

/** One history row before stable time-delta ordering and distance filtering. */
struct SemanticHistoryEntry {
   int64_t dt = 0;
   std::vector< SemanticLiteral > literals;

   auto operator<=>(const SemanticHistoryEntry&) const = default;
};

/**
 * Immutable *domain*-level schema: the predicate and action declarations every
 * problem of one domain shares.
 *
 * This is the only context an encoder engine needs. Its relation universe is a
 * function of the predicate/action schemas and the encoder configuration, not
 * of how many objects a particular instance has, what its static facts are, or
 * what it is asked to achieve. Keeping the schema separate is what lets one
 * engine encode graphs from many problems into a single batch.
 */
struct SemanticSchemaContext {
   std::vector< SemanticPredicateSpec > predicates;
   std::vector< SemanticActionSpec > actions;

   auto operator<=>(const SemanticSchemaContext&) const = default;
};

/**
 * Immutable *problem*-level data for one planning instance.
 *
 * Backends translate their stable local identities into the compact IDs in
 * this context once during adapter construction. Every graph carries its own
 * problem context (through its input or its `ViewPreparation`); the engine
 * never holds one, so graphs from different instances batch together as long
 * as their schemas agree.
 *
 * Object indices are deliberately *graph*-local: object 0 of one problem and
 * object 0 of another are unrelated nodes, and `BatchBuilder` offsets them
 * independently. Predicate and action indices, by contrast, are schema-local
 * and therefore identical across every problem of the same domain.
 */
struct SemanticProblemContext {
   std::shared_ptr< const SemanticSchemaContext > schema;
   std::vector< std::string > objects;
   std::vector< SemanticAtom > static_facts;
   std::vector< SemanticLiteral > default_goals;

   [[nodiscard]] const std::vector< SemanticPredicateSpec >& predicates() const
   {
      return require_schema().predicates;
   }
   [[nodiscard]] const std::vector< SemanticActionSpec >& actions() const
   {
      return require_schema().actions;
   }

  private:
   [[nodiscard]] const SemanticSchemaContext& require_schema() const
   {
      if(not schema) {
         throw std::invalid_argument("semantic problem context carries no schema context");
      }
      return *schema;
   }
};

/**
 * Whether a graph built against `lhs` may be encoded by an engine holding `rhs`.
 *
 * Pointer equality is the fast path -- backends share one schema instance per
 * domain -- but two structurally identical schemas from independently parsed
 * copies of the same domain are equally encodable, so fall back to comparing
 * the declarations themselves. What must *not* be required is that both graphs
 * came from the same problem.
 */
[[nodiscard]] inline bool semantic_schema_compatible(
   const std::shared_ptr< const SemanticSchemaContext >& lhs,
   const std::shared_ptr< const SemanticSchemaContext >& rhs
)
{
   if(lhs == rhs) {
      return true;
   }
   if(not lhs or not rhs) {
      return false;
   }
   return *lhs == *rhs;
}

/**
 * Reject a graph whose schema the engine cannot encode.
 *
 * This is the *only* cross-graph precondition a batch has: what a batch may
 * not mix is incompatible domains, not distinct problems.
 */
inline void require_semantic_schema_compatible(
   const std::shared_ptr< const SemanticProblemContext >& graph_context,
   const std::shared_ptr< const SemanticSchemaContext >& engine_schema,
   std::string_view what
)
{
   if(not graph_context) {
      throw std::invalid_argument(std::string(what) + " carries no problem context");
   }
   if(not semantic_schema_compatible(graph_context->schema, engine_schema)) {
      throw std::invalid_argument(
         std::string(what) + " was built against a schema this encoder cannot encode"
      );
   }
}

/**
 * @brief Owned semantic input for one flat graph.
 *
 * All numeric indices are deliberately local to this value and its engine
 * schema. They are compact transport into the hot native loop, not persistent
 * or cross-repository identities.
 */
struct SemanticFlatRelationInput {
   /** Shared immutable problem data; absent for legacy standalone inputs. */
   std::shared_ptr< const SemanticProblemContext > problem_context;
   /** Legacy object table when no problem context is supplied. */
   std::vector< std::string > objects;
   /** Dynamic state facts only when problem_context is supplied. */
   std::vector< SemanticAtom > state_facts;
   /** Explicit goal override; defaults come from problem_context when requested. */
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
   return input.problem_context ? input.problem_context->objects : input.objects;
}

[[nodiscard]] inline const std::vector< SemanticLiteral >& semantic_goals(
   const SemanticFlatRelationInput& input
)
{
   return input.problem_context and input.use_default_goals ? input.problem_context->default_goals
                                                            : input.goals;
}

[[nodiscard]] inline const std::vector< SemanticAtom >& semantic_static_facts(
   const SemanticFlatRelationInput& input
)
{
   static const std::vector< SemanticAtom > empty;
   return input.problem_context ? input.problem_context->static_facts : empty;
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
template < typename Input >
[[nodiscard]] inline std::vector< SemanticGoalLevel > semantic_goal_levels(const Input& input)
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
   size_t index = 0;
   for(const auto& layer : input.subgoal_layers) {
      for(const auto& literal : layer) {
         levels.push_back({literal, index + 1});
      }
      ++index;
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
   return lhs.problem_context and lhs.problem_context == rhs.problem_context;
}

}  // namespace mifrost
