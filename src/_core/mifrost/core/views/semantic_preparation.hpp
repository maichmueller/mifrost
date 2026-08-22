/**
 * @file semantic_preparation.hpp
 * @brief Compile-time semantic View adapters and compact encoder preparation.
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <iterator>
#include <memory>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

// This layer sits *below* the encoder families: it may depend on the semantic
// record definitions and the View concepts, never on an encoder header.
#include "mifrost/core/common_types.hpp"
#include "mifrost/core/semantic/records.hpp"
#include "mifrost/core/views/concepts.hpp"

namespace mifrost::canonical::detail {

/**
 * References into the compact graph-working pools built from borrowed Views.
 *
 * The pools contain unique graph identities; the dynamic input lanes retain
 * only references into those pools. This preserves repeated action/fact
 * occurrences without constructing a second semantic input object.
 */
struct ViewLiteralRef {
   size_t atom_index = 0;
   bool positive = true;
};

struct ViewGoalLevelRef {
   ViewLiteralRef literal;
   size_t level = 0;
};

struct ViewHistoryEntryData {
   int64_t dt = 0;
   std::vector< ViewLiteralRef > literals;
};

class SemanticActionRange;
class SemanticLiteralRange;
class SemanticHistoryRange;
class SemanticGoalRange;

struct ViewPreparation {
   /**
    * The problem this graph came from. Held per preparation, never per engine:
    * a batch may mix preparations from different problems of one domain.
    */
   std::shared_ptr< const SemanticProblemContext > problem_context;
   /**
    * Compact graph-derived identity pools.
    *
    * `atom_pool` and `action_pool` hold each unique identity once, in first-use
    * order; every lane keeps only indices into them, which preserves both lane
    * order and multiplicity. `atom_indices` / `action_indices` exist purely for
    * O(1) lookup during interning -- iteration order is always taken from the
    * pools, never from the maps, so emission stays deterministic.
    */
   std::vector< SemanticAtom > atom_pool;
   hash_map< SemanticAtom, size_t, SemanticAtomHash > atom_indices;
   /**
    * State facts in native emission order, stored contiguously.
    *
    * These are not pool references. A backend state View yields lazy atom
    * proxies, so the state lane has to become owned `SemanticAtom` values
    * somewhere regardless; keeping them contiguous lets every consumer take a
    * plain `std::span` -- the same shape the owning compatibility DTO already
    * has -- instead of a two-mode carrier that branches on storage per element.
    * It also costs one vector rather than a pool entry plus an index.
    *
    * The pool remains for the lanes that genuinely need stable multi-pass
    * identity across repeated occurrences: goals and history.
    */
   std::vector< SemanticAtom > state_facts;
   std::vector< ViewGoalLevelRef > goal_level_refs;
   size_t goal_layer_count = 0;
   std::vector< SemanticGroundAction > action_pool;
   hash_map< SemanticGroundAction, size_t, SemanticGroundActionHash > action_indices;
   std::vector< size_t > action_occurrence_indices;
   std::vector< ViewHistoryEntryData > history_data;
   hash_set< SemanticAtom, SemanticAtomHash > fact_lookup;

   /** Reserve pool storage when an input lane can cheaply report its size. */
   void reserve_atoms(size_t count)
   {
      atom_pool.reserve(atom_pool.size() + count);
      atom_indices.reserve(atom_indices.size() + count);
      fact_lookup.reserve(fact_lookup.size() + count);
   }

   void reserve_state_facts(size_t count)
   {
      state_facts.reserve(state_facts.size() + count);
      fact_lookup.reserve(fact_lookup.size() + count);
   }

   void reserve_actions(size_t count)
   {
      action_pool.reserve(action_pool.size() + count);
      action_indices.reserve(action_indices.size() + count);
      action_occurrence_indices.reserve(action_occurrence_indices.size() + count);
   }

   [[nodiscard]] size_t intern_atom(SemanticAtom atom)
   {
      const auto [it, inserted] = atom_indices.try_emplace(atom, atom_pool.size());
      if(inserted) {
         atom_pool.push_back(std::move(atom));
      }
      return it->second;
   }

   [[nodiscard]] size_t intern_action(SemanticGroundAction action)
   {
      const auto [it, inserted] = action_indices.try_emplace(action, action_pool.size());
      if(inserted) {
         action_pool.push_back(std::move(action));
      }
      return it->second;
   }

   [[nodiscard]] SemanticLiteral materialize_literal(const ViewLiteralRef& literal) const
   {
      return SemanticLiteral{atom_pool.at(literal.atom_index), literal.positive};
   }

   [[nodiscard]] SemanticActionRange actions() const;
   [[nodiscard]] const SemanticGroundAction& action_at(size_t index) const
   {
      return action_pool.at(index);
   }
   [[nodiscard]] SemanticHistoryRange history() const;
   [[nodiscard]] SemanticGoalRange goal_levels() const;
};

/**
 * The unique ground actions interned for one graph.
 *
 * Single-mode by construction: it always reads the preparation's action pool.
 * The owning compatibility DTO already stores its actions contiguously and is
 * consumed as a plain span, so no storage-mode check reaches per-element code.
 */
class SemanticActionRange {
  public:
   explicit SemanticActionRange(const ViewPreparation* preparation) : preparation_(preparation) {}

   class iterator {
     public:
      using iterator_category = std::forward_iterator_tag;
      using iterator_concept = std::forward_iterator_tag;
      using value_type = SemanticGroundAction;
      using difference_type = std::ptrdiff_t;
      using reference = const SemanticGroundAction&;

      iterator() = default;
      iterator(const SemanticActionRange* owner, size_t index) : owner_(owner), index_(index) {}

      [[nodiscard]] const SemanticGroundAction& operator*() const { return owner_->at(index_); }
      iterator& operator++()
      {
         ++index_;
         return *this;
      }
      [[nodiscard]] bool operator==(const iterator& other) const
      {
         return owner_ == other.owner_ and index_ == other.index_;
      }

     private:
      const SemanticActionRange* owner_ = nullptr;
      size_t index_ = 0;
   };

   [[nodiscard]] iterator begin() const { return iterator{this, 0}; }
   [[nodiscard]] iterator end() const { return iterator{this, size()}; }
   [[nodiscard]] size_t size() const { return preparation_->action_pool.size(); }
   [[nodiscard]] const SemanticGroundAction& at(size_t index) const
   {
      return preparation_->action_pool.at(index);
   }

  private:
   const ViewPreparation* preparation_ = nullptr;
};

class SemanticLiteralRange {
  public:
   SemanticLiteralRange(
      const ViewPreparation* preparation,
      const std::vector< ViewLiteralRef >* references
   )
       : preparation_(preparation), references_(references)
   {
   }

   class iterator {
     public:
      using iterator_category = std::forward_iterator_tag;
      using iterator_concept = std::forward_iterator_tag;
      using value_type = SemanticLiteral;
      using difference_type = std::ptrdiff_t;
      using reference = SemanticLiteral;

      iterator() = default;
      iterator(const SemanticLiteralRange* owner, size_t index) : owner_(owner), index_(index) {}

      [[nodiscard]] SemanticLiteral operator*() const { return owner_->at(index_); }
      iterator& operator++()
      {
         ++index_;
         return *this;
      }
      [[nodiscard]] bool operator==(const iterator& other) const
      {
         return owner_ == other.owner_ and index_ == other.index_;
      }

     private:
      const SemanticLiteralRange* owner_ = nullptr;
      size_t index_ = 0;
   };

   [[nodiscard]] iterator begin() const { return iterator{this, 0}; }
   [[nodiscard]] iterator end() const { return iterator{this, size()}; }
   [[nodiscard]] size_t size() const { return references_->size(); }
   [[nodiscard]] SemanticLiteral at(size_t index) const
   {
      return preparation_->materialize_literal(references_->at(index));
   }

  private:
   const ViewPreparation* preparation_ = nullptr;
   const std::vector< ViewLiteralRef >* references_ = nullptr;
};

class SemanticHistoryRange {
  public:
   struct Entry {
      int64_t dt = 0;
      SemanticLiteralRange literals{nullptr, nullptr};
   };

   explicit SemanticHistoryRange(const ViewPreparation* preparation) : preparation_(preparation) {}

   class iterator {
     public:
      using iterator_category = std::forward_iterator_tag;
      using iterator_concept = std::forward_iterator_tag;
      using value_type = Entry;
      using difference_type = std::ptrdiff_t;
      using reference = Entry;

      iterator() = default;
      iterator(const SemanticHistoryRange* owner, size_t index) : owner_(owner), index_(index) {}

      [[nodiscard]] Entry operator*() const { return owner_->at(index_); }
      iterator& operator++()
      {
         ++index_;
         return *this;
      }
      [[nodiscard]] bool operator==(const iterator& other) const
      {
         return owner_ == other.owner_ and index_ == other.index_;
      }

     private:
      const SemanticHistoryRange* owner_ = nullptr;
      size_t index_ = 0;
   };

   [[nodiscard]] iterator begin() const { return iterator{this, 0}; }
   [[nodiscard]] iterator end() const { return iterator{this, size()}; }
   [[nodiscard]] size_t size() const { return preparation_->history_data.size(); }
   [[nodiscard]] Entry at(size_t index) const
   {
      const auto& entry = preparation_->history_data.at(index);
      return Entry{
         .dt = entry.dt,
         .literals = SemanticLiteralRange{preparation_, &entry.literals},
      };
   }

  private:
   const ViewPreparation* preparation_ = nullptr;
};

class SemanticGoalRange {
  public:
   SemanticGoalRange(const ViewPreparation* preparation) : preparation_(preparation) {}

   class iterator {
     public:
      using iterator_category = std::forward_iterator_tag;
      using iterator_concept = std::forward_iterator_tag;
      using value_type = SemanticGoalLevel;
      using difference_type = std::ptrdiff_t;
      using reference = SemanticGoalLevel;

      iterator() = default;
      iterator(const SemanticGoalRange* owner, size_t index) : owner_(owner), index_(index) {}

      [[nodiscard]] SemanticGoalLevel operator*() const { return owner_->at(index_); }
      iterator& operator++()
      {
         ++index_;
         return *this;
      }
      [[nodiscard]] bool operator==(const iterator& other) const
      {
         return owner_ == other.owner_ and index_ == other.index_;
      }

     private:
      const SemanticGoalRange* owner_ = nullptr;
      size_t index_ = 0;
   };

   [[nodiscard]] iterator begin() const { return iterator{this, 0}; }
   [[nodiscard]] iterator end() const { return iterator{this, size()}; }
   [[nodiscard]] size_t size() const { return preparation_->goal_level_refs.size(); }
   [[nodiscard]] SemanticGoalLevel at(size_t index) const
   {
      const auto& goal = preparation_->goal_level_refs.at(index);
      return SemanticGoalLevel{
         .literal = preparation_->materialize_literal(goal.literal),
         .level = goal.level,
      };
   }

  private:
   const ViewPreparation* preparation_ = nullptr;
};

inline SemanticActionRange ViewPreparation::actions() const
{
   return SemanticActionRange{this};
}

inline SemanticHistoryRange ViewPreparation::history() const
{
   return SemanticHistoryRange{this};
}

inline SemanticGoalRange ViewPreparation::goal_levels() const
{
   return SemanticGoalRange{this};
}

inline void append_view_default_goals(ViewPreparation& preparation);
inline void append_view_static_facts(ViewPreparation& preparation);
template < typename Layer >
[[nodiscard]] size_t view_layer_level(const Layer& layer, size_t fallback);

/**
 * Compile-time borrowed input aggregate used at the direct encoder boundary.
 *
 * Range members are `views::all` wrappers, so lvalue backend ranges remain
 * references and rvalue ranges remain owned by the short encode call. The
 * aggregate itself contains no semantic-record lane.
 */
template < typename State, typename Goals, typename Layers, typename Actions, typename History >
struct BorrowedViewInput {
   const State* state = nullptr;
   Goals goals;
   Layers layers;
   Actions actions;
   History history;
   std::optional< int64_t > history_max_steps = std::nullopt;
};

template < typename Range >
using borrowed_range_t = decltype(std::views::all(std::declval< Range >()));

template < typename State, typename Goals, typename Layers, typename Actions, typename History >
[[nodiscard]] ViewPreparation prepare_borrowed_view_input(
   const std::shared_ptr< const SemanticProblemContext >& context,
   const BorrowedViewInput< State, Goals, Layers, Actions, History >& input
)
{
   if(not context) {
      throw std::invalid_argument("semantic View input requires a problem context");
   }
   ViewPreparation result{.problem_context = context};
   append_view_static_facts(result);
   append_view_state(*input.state, result);
   if constexpr(not std::same_as< Goals, std::monostate >) {
      append_view_goals(input.goals, 0, result);
   } else {
      append_view_default_goals(result);
   }
   if constexpr(not std::same_as< Layers, std::monostate >) {
      for(const auto& layer : input.layers) {
         const auto level = view_layer_level(layer, result.goal_layer_count + 1);
         append_view_goals(layer, level, result);
         result.goal_layer_count = std::max(result.goal_layer_count, level);
      }
   }
   append_view_actions(input.actions, result);
   if constexpr(not std::same_as< History, std::monostate >) {
      append_view_history(input.history, input.history_max_steps, result);
   }
   return result;
}

template < views::AtomView Atom >
[[nodiscard]] SemanticAtom view_materialize_atom(const Atom& atom)
{
   SemanticAtom result;
   result.predicate = static_cast< int64_t >(atom.predicate_id());
   if(result.predicate < 0) {
      throw std::invalid_argument("semantic View contains an invalid predicate ID");
   }
   for(const auto object : atom.arguments()) {
      const auto id = static_cast< int64_t >(object);
      if(id < 0) {
         throw std::invalid_argument("semantic View contains an invalid object ID");
      }
      result.arguments.push_back(id);
   }
   return result;
}

template < views::LiteralView Literal >
[[nodiscard]] SemanticLiteral view_materialize_literal(const Literal& literal)
{
   return {view_materialize_atom(literal.atom()), not static_cast< bool >(literal.is_negated())};
}

template < views::GroundActionView Action >
[[nodiscard]] SemanticGroundAction view_materialize_action(const Action& action)
{
   SemanticGroundAction result;
   result.action = static_cast< int64_t >(action.schema_id());
   if(result.action < 0) {
      throw std::invalid_argument("semantic View contains an invalid action schema ID");
   }
   for(const auto object : action.arguments()) {
      const auto id = static_cast< int64_t >(object);
      if(id < 0) {
         throw std::invalid_argument("semantic View contains an invalid action object ID");
      }
      result.arguments.push_back(id);
   }
   return result;
}

inline SemanticAtom view_materialize_atom(const SemanticAtom& atom)
{
   return atom;
}
inline SemanticLiteral view_materialize_literal(const SemanticLiteral& literal)
{
   return literal;
}
inline SemanticGroundAction view_materialize_action(const SemanticGroundAction& action)
{
   return action;
}

inline void add_fact(ViewPreparation& preparation, SemanticAtom atom)
{
   preparation.fact_lookup.emplace(atom);
   preparation.state_facts.push_back(std::move(atom));
}

template < views::StateView State >
void append_view_state(const State& state, ViewPreparation& preparation)
{
   // Request each lane exactly once. `StateView` only promises an input_range,
   // so a second call is free to return a different range -- or, for a
   // single-pass source, an already-consumed one. Sizing and traversal must
   // therefore share the range, which also means it cannot be held by const
   // reference: an input_range is iterated by mutating it.
   auto fluent_atoms = state.fluent_atoms();
   auto derived_atoms = state.derived_atoms();
   if constexpr(std::ranges::sized_range< decltype(fluent_atoms) >
                and std::ranges::sized_range< decltype(derived_atoms) >) {
      preparation.reserve_state_facts(
         std::ranges::size(fluent_atoms) + std::ranges::size(derived_atoms)
      );
   }
   for(const auto atom : fluent_atoms) {
      add_fact(preparation, view_materialize_atom(atom));
   }
   for(const auto atom : derived_atoms) {
      add_fact(preparation, view_materialize_atom(atom));
   }
}

template < views::LiteralRange Goals >
void append_view_goals(Goals&& goals, size_t level, ViewPreparation& preparation)
{
   if constexpr(std::ranges::sized_range< Goals >) {
      preparation.goal_level_refs.reserve(
         preparation.goal_level_refs.size() + std::ranges::size(goals)
      );
   }
   for(const auto& goal : goals) {
      const auto literal = view_materialize_literal(goal);
      preparation.goal_level_refs.push_back(
         ViewGoalLevelRef{
            .literal =
               ViewLiteralRef{
                  .atom_index = preparation.intern_atom(literal.atom),
                  .positive = literal.positive,
               },
            .level = level,
         }
      );
   }
}

template < typename Layer >
[[nodiscard]] size_t view_layer_level(const Layer& layer, size_t fallback)
{
   if constexpr(requires { layer.level(); }) {
      return static_cast< size_t >(layer.level());
   } else {
      return fallback;
   }
}

template < views::GroundActionRange Actions >
void append_view_actions(Actions&& actions, ViewPreparation& preparation)
{
   if constexpr(std::ranges::sized_range< Actions >) {
      preparation.reserve_actions(std::ranges::size(actions));
   }
   // Deduplicate into the action pool while recording one occurrence index per
   // input action, so repeated occurrences survive even though graph nodes are
   // shared.
   for(const auto& action : actions) {
      preparation.action_occurrence_indices.push_back(
         preparation.intern_action(view_materialize_action(action))
      );
   }
}

template < views::HistoryRange History >
void append_view_history(
   History&& history,
   std::optional< int64_t > history_max_steps,
   ViewPreparation& preparation
)
{
   for(const auto& entry : history) {
      const auto dt = static_cast< int64_t >(entry.dt());
      if(history_max_steps and std::abs(dt) > *history_max_steps) {
         continue;
      }
      auto& target = preparation.history_data.emplace_back();
      target.dt = dt;
      for(const auto& literal : entry.literals()) {
         const auto value = view_materialize_literal(literal);
         target.literals.push_back(
            ViewLiteralRef{
               .atom_index = preparation.intern_atom(value.atom),
               .positive = value.positive,
            }
         );
      }
   }
   std::ranges::stable_sort(preparation.history_data, {}, &ViewHistoryEntryData::dt);
}

inline void append_view_default_goals(ViewPreparation& preparation)
{
   for(const auto& goal : preparation.problem_context->default_goals) {
      preparation.goal_level_refs.push_back(
         ViewGoalLevelRef{
            .literal =
               ViewLiteralRef{
                  .atom_index = preparation.intern_atom(goal.atom),
                  .positive = goal.positive,
               },
            .level = 0,
         }
      );
   }
}

inline void append_view_static_facts(ViewPreparation& preparation)
{
   const auto& static_facts = preparation.problem_context->static_facts;
   preparation.fact_lookup.reserve(preparation.fact_lookup.size() + static_facts.size());
   for(const auto& fact : static_facts) {
      preparation.fact_lookup.emplace(fact);
   }
}

/**
 * Lane-aware preparation for paths that read only objects and state facts.
 *
 * The successor side of the successor-HGraph algorithm reads exactly two things
 * from its input: the object table (for name formatting and the shared-table
 * check) and the successor state facts. It never inspects successor goals,
 * subgoal layers, actions or history. Running the full preparation there
 * materialized the problem context's default goals into the atom pool and the
 * goal-level lane on every encode, purely to be discarded.
 *
 * This builds only the static-fact membership set and the state-fact lane; the
 * other lanes stay empty. Output is unchanged, because no consumer of a
 * state-only preparation reads them.
 */
template < views::StateView State >
[[nodiscard]] ViewPreparation make_state_only_view_preparation(
   const std::shared_ptr< const SemanticProblemContext >& context,
   const State& state
)
{
   if(not context) {
      throw std::invalid_argument("semantic View input requires a problem context");
   }
   ViewPreparation result{.problem_context = context};
   append_view_static_facts(result);
   append_view_state(state, result);
   return result;
}

template < views::StateView State, views::GroundActionRange Actions >
[[nodiscard]] ViewPreparation make_view_preparation(
   const std::shared_ptr< const SemanticProblemContext >& context,
   const State& state,
   Actions&& actions
)
{
   using ActionRange = borrowed_range_t< Actions&& >;
   return prepare_borrowed_view_input(
      context,
      BorrowedViewInput< State, std::monostate, std::monostate, ActionRange, std::monostate >{
         .state = &state,
         .actions = std::views::all(std::forward< Actions >(actions)),
      }
   );
}

template < views::StateView State, views::LiteralRange Goals, views::GroundActionRange Actions >
[[nodiscard]] ViewPreparation make_view_preparation(
   const std::shared_ptr< const SemanticProblemContext >& context,
   const State& state,
   Goals&& goals,
   Actions&& actions
)
{
   using GoalRange = borrowed_range_t< Goals&& >;
   using ActionRange = borrowed_range_t< Actions&& >;
   return prepare_borrowed_view_input(
      context,
      BorrowedViewInput< State, GoalRange, std::monostate, ActionRange, std::monostate >{
         .state = &state,
         .goals = std::views::all(std::forward< Goals >(goals)),
         .actions = std::views::all(std::forward< Actions >(actions)),
      }
   );
}

template <
   views::StateView State,
   views::LiteralRange Goals,
   views::LiteralLayerRange Layers,
   views::GroundActionRange Actions >
[[nodiscard]] ViewPreparation make_view_preparation(
   const std::shared_ptr< const SemanticProblemContext >& context,
   const State& state,
   Goals&& goals,
   Layers&& layers,
   Actions&& actions
)
{
   using GoalRange = borrowed_range_t< Goals&& >;
   using LayerRange = borrowed_range_t< Layers&& >;
   using ActionRange = borrowed_range_t< Actions&& >;
   return prepare_borrowed_view_input(
      context,
      BorrowedViewInput< State, GoalRange, LayerRange, ActionRange, std::monostate >{
         .state = &state,
         .goals = std::views::all(std::forward< Goals >(goals)),
         .layers = std::views::all(std::forward< Layers >(layers)),
         .actions = std::views::all(std::forward< Actions >(actions)),
      }
   );
}

template <
   views::StateView State,
   views::LiteralRange Goals,
   views::LiteralLayerRange Layers,
   views::GroundActionRange Actions >
[[nodiscard]] ViewPreparation make_hgraph_view_preparation(
   const std::shared_ptr< const SemanticProblemContext >& context,
   const State& state,
   Goals&& goals,
   Layers&& layers,
   Actions&& actions
)
{
   return make_view_preparation(
      context,
      state,
      std::forward< Goals >(goals),
      std::forward< Layers >(layers),
      std::forward< Actions >(actions)
   );
}

template <
   views::StateView State,
   views::LiteralRange Goals,
   views::LiteralLayerRange Layers,
   views::GroundActionRange Actions,
   views::HistoryRange History >
[[nodiscard]] ViewPreparation make_view_preparation(
   const std::shared_ptr< const SemanticProblemContext >& context,
   const State& state,
   Goals&& goals,
   Layers&& layers,
   Actions&& actions,
   History&& history,
   std::optional< int64_t > history_max_steps = std::nullopt
)
{
   using GoalRange = borrowed_range_t< Goals&& >;
   using LayerRange = borrowed_range_t< Layers&& >;
   using ActionRange = borrowed_range_t< Actions&& >;
   using HistoryRange = borrowed_range_t< History&& >;
   return prepare_borrowed_view_input(
      context,
      BorrowedViewInput< State, GoalRange, LayerRange, ActionRange, HistoryRange >{
         .state = &state,
         .goals = std::views::all(std::forward< Goals >(goals)),
         .layers = std::views::all(std::forward< Layers >(layers)),
         .actions = std::views::all(std::forward< Actions >(actions)),
         .history = std::views::all(std::forward< History >(history)),
         .history_max_steps = history_max_steps,
      }
   );
}

template < views::StateView State, views::GroundActionRange Actions >
[[nodiscard]] ViewPreparation make_flat_view_preparation(
   const std::shared_ptr< const SemanticProblemContext >& context,
   const State& state,
   Actions&& actions
)
{
   return make_view_preparation(context, state, std::forward< Actions >(actions));
}

template < views::StateView State, views::LiteralRange Goals, views::GroundActionRange Actions >
[[nodiscard]] ViewPreparation make_flat_view_preparation(
   const std::shared_ptr< const SemanticProblemContext >& context,
   const State& state,
   Goals&& goals,
   Actions&& actions
)
{
   return make_view_preparation(
      context, state, std::forward< Goals >(goals), std::forward< Actions >(actions)
   );
}

template <
   views::StateView State,
   views::LiteralRange Goals,
   views::LiteralLayerRange Layers,
   views::GroundActionRange Actions,
   views::HistoryRange History >
[[nodiscard]] ViewPreparation make_flat_view_preparation(
   const std::shared_ptr< const SemanticProblemContext >& context,
   const State& state,
   Goals&& goals,
   Layers&& layers,
   Actions&& actions,
   History&& history,
   std::optional< int64_t > history_max_steps = std::nullopt
)
{
   return make_view_preparation(
      context,
      state,
      std::forward< Goals >(goals),
      std::forward< Layers >(layers),
      std::forward< Actions >(actions),
      std::forward< History >(history),
      history_max_steps
   );
}

template < views::StateView State, views::GroundActionRange Actions >
[[nodiscard]] ViewPreparation make_hgraph_view_preparation(
   const std::shared_ptr< const SemanticProblemContext >& context,
   const State& state,
   Actions&& actions
)
{
   return make_view_preparation(context, state, std::forward< Actions >(actions));
}

template < views::StateView State, views::LiteralRange Goals, views::GroundActionRange Actions >
[[nodiscard]] ViewPreparation make_hgraph_view_preparation(
   const std::shared_ptr< const SemanticProblemContext >& context,
   const State& state,
   Goals&& goals,
   Actions&& actions
)
{
   return make_view_preparation(
      context, state, std::forward< Goals >(goals), std::forward< Actions >(actions)
   );
}

template <
   views::StateView State,
   views::LiteralRange Goals,
   views::LiteralLayerRange Layers,
   views::GroundActionRange Actions,
   views::HistoryRange History >
[[nodiscard]] ViewPreparation make_hgraph_view_preparation(
   const std::shared_ptr< const SemanticProblemContext >& context,
   const State& state,
   Goals&& goals,
   Layers&& layers,
   Actions&& actions,
   History&& history,
   std::optional< int64_t > history_max_steps = std::nullopt
)
{
   return make_view_preparation(
      context,
      state,
      std::forward< Goals >(goals),
      std::forward< Layers >(layers),
      std::forward< Actions >(actions),
      std::forward< History >(history),
      history_max_steps
   );
}

template < views::StateView State, views::GroundActionRange Actions >
[[nodiscard]] ViewPreparation make_color_view_preparation(
   const std::shared_ptr< const SemanticProblemContext >& context,
   const State& state,
   Actions&& actions
)
{
   return make_view_preparation(context, state, std::forward< Actions >(actions));
}

template < views::StateView State, views::LiteralRange Goals, views::GroundActionRange Actions >
[[nodiscard]] ViewPreparation make_color_view_preparation(
   const std::shared_ptr< const SemanticProblemContext >& context,
   const State& state,
   Goals&& goals,
   Actions&& actions
)
{
   return make_view_preparation(
      context, state, std::forward< Goals >(goals), std::forward< Actions >(actions)
   );
}

template <
   views::StateView State,
   views::LiteralRange Goals,
   views::LiteralLayerRange Layers,
   views::GroundActionRange Actions >
[[nodiscard]] ViewPreparation make_color_view_preparation(
   const std::shared_ptr< const SemanticProblemContext >& context,
   const State& state,
   Goals&& goals,
   Layers&& layers,
   Actions&& actions
)
{
   return make_view_preparation(
      context,
      state,
      std::forward< Goals >(goals),
      std::forward< Layers >(layers),
      std::forward< Actions >(actions)
   );
}

template < views::StateView State, views::GroundActionRange Actions >
[[nodiscard]] ViewPreparation make_derived_view_preparation(
   const std::shared_ptr< const SemanticProblemContext >& context,
   const State& state,
   Actions&& actions
)
{
   return make_view_preparation(context, state, std::forward< Actions >(actions));
}

template < views::StateView State, views::LiteralRange Goals, views::GroundActionRange Actions >
[[nodiscard]] ViewPreparation make_derived_view_preparation(
   const std::shared_ptr< const SemanticProblemContext >& context,
   const State& state,
   Goals&& goals,
   Actions&& actions
)
{
   return make_view_preparation(
      context, state, std::forward< Goals >(goals), std::forward< Actions >(actions)
   );
}

template <
   views::StateView State,
   views::LiteralRange Goals,
   views::LiteralLayerRange Layers,
   views::GroundActionRange Actions >
[[nodiscard]] ViewPreparation make_derived_view_preparation(
   const std::shared_ptr< const SemanticProblemContext >& context,
   const State& state,
   Goals&& goals,
   Layers&& layers,
   Actions&& actions
)
{
   return make_view_preparation(
      context,
      state,
      std::forward< Goals >(goals),
      std::forward< Layers >(layers),
      std::forward< Actions >(actions)
   );
}

template <
   views::StateView State,
   views::LiteralRange Goals,
   views::LiteralLayerRange Layers,
   views::GroundActionRange Actions,
   views::HistoryRange History >
[[nodiscard]] ViewPreparation make_derived_view_preparation(
   const std::shared_ptr< const SemanticProblemContext >& context,
   const State& state,
   Goals&& goals,
   Layers&& layers,
   Actions&& actions,
   History&& history,
   std::optional< int64_t > history_max_steps = std::nullopt
)
{
   return make_view_preparation(
      context,
      state,
      std::forward< Goals >(goals),
      std::forward< Layers >(layers),
      std::forward< Actions >(actions),
      std::forward< History >(history),
      history_max_steps
   );
}

inline const std::vector< std::string >& semantic_objects(const ViewPreparation& input)
{
   if(not input.problem_context) {
      throw std::invalid_argument("semantic View preparation requires an object table");
   }
   return input.problem_context->objects;
}

inline const std::vector< SemanticAtom >& semantic_static_facts(const ViewPreparation& input)
{
   return input.problem_context->static_facts;
}

}  // namespace mifrost::canonical::detail

namespace mifrost::detail {

inline const std::vector< std::string >& semantic_objects(
   const canonical::detail::ViewPreparation& input
)
{
   return canonical::detail::semantic_objects(input);
}

inline const std::vector< SemanticAtom >& semantic_static_facts(
   const canonical::detail::ViewPreparation& input
)
{
   return canonical::detail::semantic_static_facts(input);
}

}  // namespace mifrost::detail

namespace mifrost {

template < typename Input >
[[nodiscard]] decltype(auto) semantic_state_facts(const Input& input)
{
   if constexpr(requires { input.state_facts(); }) {
      return input.state_facts();
   } else {
      return (input.state_facts);
   }
}

template < typename Input >
[[nodiscard]] decltype(auto) semantic_actions(const Input& input)
{
   if constexpr(requires { input.actions(); }) {
      return input.actions();
   } else {
      return (input.actions);
   }
}

template < typename Input >
[[nodiscard]] decltype(auto) semantic_history(const Input& input)
{
   if constexpr(requires { input.history(); }) {
      return input.history();
   } else {
      return (input.history);
   }
}

template < typename Input >
[[nodiscard]] decltype(auto) semantic_action_at(const Input& input, size_t index)
{
   if constexpr(requires { input.action_at(index); }) {
      return input.action_at(index);
   } else {
      return (input.actions.at(index));
   }
}

template < typename Input >
[[nodiscard]] decltype(auto) semantic_history_at(const Input& input, size_t index)
{
   if constexpr(requires { input.history().at(index); }) {
      return input.history().at(index);
   } else {
      return (input.history.at(index));
   }
}

[[nodiscard]] inline std::vector< SemanticGoalLevel > semantic_goal_levels(
   const canonical::detail::ViewPreparation& input
)
{
   std::vector< SemanticGoalLevel > result;
   result.reserve(input.goal_level_refs.size());
   for(const auto& goal : input.goal_levels()) {
      result.push_back(goal);
   }
   return result;
}

}  // namespace mifrost
