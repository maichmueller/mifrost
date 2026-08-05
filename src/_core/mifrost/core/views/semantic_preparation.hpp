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
#include <map>
#include <memory>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "mifrost/core/common_types.hpp"
#include "mifrost/core/encoders/flat/semantic_flat_relation_encoder.hpp"

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

class SemanticAtomRange;
class SemanticActionRange;
class SemanticLiteralRange;
class SemanticHistoryRange;
class SemanticGoalRange;

struct ViewPreparation {
   std::shared_ptr< const SemanticTaskContext > task_context;
   std::vector< SemanticAtom > atom_pool;
   std::vector< size_t > state_fact_indices;
   std::vector< ViewGoalLevelRef > goal_level_refs;
   size_t goal_layer_count = 0;
   std::vector< SemanticGroundAction > action_pool;
   std::vector< size_t > action_occurrence_indices;
   std::vector< ViewHistoryEntryData > history_data;
   hash_set< SemanticAtom, SemanticAtomHash > fact_lookup;

   [[nodiscard]] size_t intern_atom(SemanticAtom atom)
   {
      for(size_t index = 0; index < atom_pool.size(); ++index) {
         if(atom_pool[index] == atom) {
            return index;
         }
      }
      atom_pool.push_back(std::move(atom));
      return atom_pool.size() - 1;
   }

   [[nodiscard]] SemanticLiteral materialize_literal(const ViewLiteralRef& literal) const
   {
      return SemanticLiteral{atom_pool.at(literal.atom_index), literal.positive};
   }

   [[nodiscard]] SemanticAtomRange state_facts() const;
   [[nodiscard]] SemanticActionRange actions() const;
   [[nodiscard]] const SemanticGroundAction& action_at(size_t index) const
   {
      return action_pool.at(index);
   }
   [[nodiscard]] SemanticHistoryRange history() const;
   [[nodiscard]] SemanticGoalRange goal_levels() const;
};

class SemanticAtomRange {
  public:
   SemanticAtomRange(
      const std::vector< SemanticAtom >* owned,
      const ViewPreparation* preparation,
      const std::vector< size_t >* references
   )
       : owned_(owned), preparation_(preparation), references_(references)
   {
   }

   class iterator {
     public:
      using iterator_category = std::forward_iterator_tag;
      using iterator_concept = std::forward_iterator_tag;
      using value_type = SemanticAtom;
      using difference_type = std::ptrdiff_t;
      using reference = const SemanticAtom&;

      iterator() = default;
      iterator(const SemanticAtomRange* owner, size_t index) : owner_(owner), index_(index) {}

      [[nodiscard]] const SemanticAtom& operator*() const { return owner_->at(index_); }
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
      const SemanticAtomRange* owner_ = nullptr;
      size_t index_ = 0;
   };

   [[nodiscard]] iterator begin() const { return iterator{this, 0}; }
   [[nodiscard]] iterator end() const { return iterator{this, size()}; }
   [[nodiscard]] size_t size() const
   {
      return owned_ != nullptr ? owned_->size() : references_->size();
   }

  private:
   [[nodiscard]] const SemanticAtom& at(size_t index) const
   {
      return owned_ != nullptr ? owned_->at(index)
                               : preparation_->atom_pool.at(references_->at(index));
   }

   const std::vector< SemanticAtom >* owned_ = nullptr;
   const ViewPreparation* preparation_ = nullptr;
   const std::vector< size_t >* references_ = nullptr;
};

class SemanticActionRange {
  public:
   SemanticActionRange(
      const std::vector< SemanticGroundAction >* owned,
      const ViewPreparation* preparation
   )
       : owned_(owned), preparation_(preparation)
   {
   }

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
   [[nodiscard]] size_t size() const
   {
      return owned_ != nullptr ? owned_->size() : preparation_->action_pool.size();
   }
   [[nodiscard]] const SemanticGroundAction& at(size_t index) const
   {
      return owned_ != nullptr ? owned_->at(index) : preparation_->action_pool.at(index);
   }

  private:
   const std::vector< SemanticGroundAction >* owned_ = nullptr;
   const ViewPreparation* preparation_ = nullptr;
};

class SemanticLiteralRange {
  public:
   SemanticLiteralRange(
      const std::vector< SemanticLiteral >* owned,
      const ViewPreparation* preparation,
      const std::vector< ViewLiteralRef >* references
   )
       : owned_(owned), preparation_(preparation), references_(references)
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
   [[nodiscard]] size_t size() const
   {
      return owned_ != nullptr ? owned_->size() : references_->size();
   }
   [[nodiscard]] SemanticLiteral at(size_t index) const
   {
      return owned_ != nullptr ? owned_->at(index)
                               : preparation_->materialize_literal(references_->at(index));
   }

  private:
   const std::vector< SemanticLiteral >* owned_ = nullptr;
   const ViewPreparation* preparation_ = nullptr;
   const std::vector< ViewLiteralRef >* references_ = nullptr;
};

class SemanticHistoryRange {
  public:
   struct Entry {
      int64_t dt = 0;
      SemanticLiteralRange literals{nullptr, nullptr, nullptr};
   };

   SemanticHistoryRange(
      const std::vector< SemanticHistoryEntry >* owned,
      const ViewPreparation* preparation
   )
       : owned_(owned), preparation_(preparation)
   {
   }

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
   [[nodiscard]] size_t size() const
   {
      return owned_ != nullptr ? owned_->size() : preparation_->history_data.size();
   }
   [[nodiscard]] Entry at(size_t index) const
   {
      if(owned_ != nullptr) {
         const auto& entry = owned_->at(index);
         return Entry{
            .dt = entry.dt,
            .literals = SemanticLiteralRange{&entry.literals, nullptr, nullptr},
         };
      }
      const auto& entry = preparation_->history_data.at(index);
      return Entry{
         .dt = entry.dt,
         .literals = SemanticLiteralRange{nullptr, preparation_, &entry.literals},
      };
   }

  private:
   const std::vector< SemanticHistoryEntry >* owned_ = nullptr;
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

inline SemanticAtomRange ViewPreparation::state_facts() const
{
   return SemanticAtomRange{nullptr, this, &state_fact_indices};
}

inline SemanticActionRange ViewPreparation::actions() const
{
   return SemanticActionRange{nullptr, this};
}

inline SemanticHistoryRange ViewPreparation::history() const
{
   return SemanticHistoryRange{nullptr, this};
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
   const std::shared_ptr< const SemanticTaskContext >& context,
   const BorrowedViewInput< State, Goals, Layers, Actions, History >& input
)
{
   if(not context) {
      throw std::invalid_argument("semantic View input requires a task context");
   }
   ViewPreparation result{.task_context = context};
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
   preparation.state_fact_indices.push_back(preparation.intern_atom(std::move(atom)));
}

template < views::StateView State >
void append_view_state(const State& state, ViewPreparation& preparation)
{
   for(const auto atom : state.fluent_atoms()) {
      add_fact(preparation, view_materialize_atom(atom));
   }
   for(const auto atom : state.derived_atoms()) {
      add_fact(preparation, view_materialize_atom(atom));
   }
}

template < views::LiteralRange Goals >
void append_view_goals(Goals&& goals, size_t level, ViewPreparation& preparation)
{
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
   std::map< SemanticGroundAction, size_t > indices;
   for(const auto& action : actions) {
      auto value = view_materialize_action(action);
      const auto [it, inserted] = indices.emplace(value, preparation.action_pool.size());
      if(inserted) {
         preparation.action_pool.push_back(std::move(value));
      }
      preparation.action_occurrence_indices.push_back(it->second);
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
   for(const auto& goal : preparation.task_context->default_goals) {
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
   for(const auto& fact : preparation.task_context->static_facts) {
      preparation.fact_lookup.emplace(fact);
   }
}

template < views::StateView State, views::GroundActionRange Actions >
[[nodiscard]] ViewPreparation make_view_preparation(
   const std::shared_ptr< const SemanticTaskContext >& context,
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
   const std::shared_ptr< const SemanticTaskContext >& context,
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
   const std::shared_ptr< const SemanticTaskContext >& context,
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
   const std::shared_ptr< const SemanticTaskContext >& context,
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
   const std::shared_ptr< const SemanticTaskContext >& context,
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
   const std::shared_ptr< const SemanticTaskContext >& context,
   const State& state,
   Actions&& actions
)
{
   return make_view_preparation(context, state, std::forward< Actions >(actions));
}

template < views::StateView State, views::LiteralRange Goals, views::GroundActionRange Actions >
[[nodiscard]] ViewPreparation make_flat_view_preparation(
   const std::shared_ptr< const SemanticTaskContext >& context,
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
   const std::shared_ptr< const SemanticTaskContext >& context,
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
   const std::shared_ptr< const SemanticTaskContext >& context,
   const State& state,
   Actions&& actions
)
{
   return make_view_preparation(context, state, std::forward< Actions >(actions));
}

template < views::StateView State, views::LiteralRange Goals, views::GroundActionRange Actions >
[[nodiscard]] ViewPreparation make_hgraph_view_preparation(
   const std::shared_ptr< const SemanticTaskContext >& context,
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
   const std::shared_ptr< const SemanticTaskContext >& context,
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
   const std::shared_ptr< const SemanticTaskContext >& context,
   const State& state,
   Actions&& actions
)
{
   return make_view_preparation(context, state, std::forward< Actions >(actions));
}

template < views::StateView State, views::LiteralRange Goals, views::GroundActionRange Actions >
[[nodiscard]] ViewPreparation make_color_view_preparation(
   const std::shared_ptr< const SemanticTaskContext >& context,
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
   const std::shared_ptr< const SemanticTaskContext >& context,
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

inline const std::vector< std::string >& semantic_objects(const ViewPreparation& input)
{
   if(not input.task_context) {
      throw std::invalid_argument("semantic View preparation requires an object table");
   }
   return input.task_context->objects;
}

inline const std::vector< SemanticAtom >& semantic_static_facts(const ViewPreparation& input)
{
   return input.task_context->static_facts;
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
