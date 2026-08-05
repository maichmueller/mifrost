/**
 * @file views.hpp
 * @brief Granular non-owning Views over retained semantic records.
 */
#pragma once

#include <cassert>
#include <cstddef>
#include <iterator>
#include <span>
#include <string_view>
#include <type_traits>
#include <vector>

#include "mifrost/core/encoders/flat/semantic_flat_relation_encoder.hpp"
#include "mifrost/core/views/concepts.hpp"
#include "mifrost/core/views/ids.hpp"
#include "mifrost/core/views/ranges.hpp"

namespace mifrost::semantic {

using views::ActionSchemaId;
using views::ObjectId;
using views::PredicateCategory;
using views::PredicateId;

struct PredicateView: views::PredicateViewBase< PredicateView > {
   const SemanticPredicateSpec* value = nullptr;
   PredicateId value_id = -1;

   constexpr PredicateView() = default;
   constexpr PredicateView(const SemanticPredicateSpec* predicate, PredicateId id)
       : value(predicate), value_id(id)
   {
   }

   [[nodiscard]] PredicateId id_impl() const noexcept { return value_id; }
   [[nodiscard]] std::string_view name_impl() const noexcept
   {
      return value == nullptr ? std::string_view{} : value->name;
   }
   [[nodiscard]] std::size_t arity_impl() const noexcept
   {
      return value == nullptr ? 0U : static_cast< std::size_t >(value->arity);
   }
   [[nodiscard]] PredicateCategory category_impl() const noexcept
   {
      if(value == nullptr) {
         return PredicateCategory::fluent;
      }
      return static_cast< PredicateCategory >(static_cast< int8_t >(value->category));
   }
};

struct ObjectView: views::ObjectViewBase< ObjectView > {
   ObjectId value = -1;
   std::string_view object_name{};

   constexpr ObjectView() = default;
   constexpr ObjectView(ObjectId id, std::string_view name) : value(id), object_name(name) {}

   [[nodiscard]] ObjectId id_impl() const noexcept { return value; }
   [[nodiscard]] std::string_view name_impl() const noexcept { return object_name; }
};

struct AtomView: views::AtomViewBase< AtomView > {
   const SemanticAtom* value = nullptr;

   constexpr AtomView() = default;
   constexpr explicit AtomView(const SemanticAtom* atom) : value(atom) {}

   [[nodiscard]] PredicateId predicate_id_impl() const noexcept
   {
      return value == nullptr ? -1 : value->predicate;
   }
   [[nodiscard]] std::span< const ObjectId > arguments_impl() const noexcept
   {
      if(value == nullptr) {
         return {};
      }
      return {value->arguments.data(), value->arguments.size()};
   }
};

struct LiteralView: views::LiteralViewBase< LiteralView > {
   const SemanticLiteral* value = nullptr;

   constexpr LiteralView() = default;
   constexpr explicit LiteralView(const SemanticLiteral* literal) : value(literal) {}

   [[nodiscard]] bool is_negated_impl() const noexcept
   {
      return value != nullptr and not value->positive;
   }
   [[nodiscard]] AtomView atom_impl() const noexcept
   {
      return value == nullptr ? AtomView{} : AtomView{&value->atom};
   }
};

struct ActionSchemaView: views::ActionSchemaViewBase< ActionSchemaView > {
   const SemanticActionSpec* value = nullptr;
   ActionSchemaId value_id = -1;

   constexpr ActionSchemaView() = default;
   constexpr ActionSchemaView(const SemanticActionSpec* action, ActionSchemaId id)
       : value(action), value_id(id)
   {
   }

   [[nodiscard]] ActionSchemaId id_impl() const noexcept { return value_id; }
   [[nodiscard]] std::string_view name_impl() const noexcept
   {
      return value == nullptr ? std::string_view{} : value->name;
   }
   [[nodiscard]] std::size_t arity_impl() const noexcept
   {
      return value == nullptr ? 0U : static_cast< std::size_t >(value->arity);
   }
};

struct GroundActionView: views::GroundActionViewBase< GroundActionView > {
   const SemanticGroundAction* value = nullptr;

   constexpr GroundActionView() = default;
   constexpr explicit GroundActionView(const SemanticGroundAction* action) : value(action) {}

   [[nodiscard]] ActionSchemaId schema_id_impl() const noexcept
   {
      return value == nullptr ? -1 : value->action;
   }
   [[nodiscard]] std::span< const ObjectId > arguments_impl() const noexcept
   {
      if(value == nullptr) {
         return {};
      }
      return {value->arguments.data(), value->arguments.size()};
   }
};

class AtomsView {
  public:
   explicit AtomsView(std::span< const SemanticAtom > values) : values_(values) {}

   class iterator {
      using Base = std::span< const SemanticAtom >::iterator;

     public:
      using iterator_category = std::forward_iterator_tag;
      using value_type = AtomView;
      using difference_type = std::ptrdiff_t;

      explicit iterator(Base value = {}) : value_(value) {}
      [[nodiscard]] AtomView operator*() const noexcept { return AtomView{&*value_}; }
      iterator& operator++() noexcept
      {
         ++value_;
         return *this;
      }
      iterator operator++(int) noexcept
      {
         auto copy = *this;
         ++value_;
         return copy;
      }
      friend bool operator==(const iterator&, const iterator&) = default;

     private:
      Base value_;
   };

   [[nodiscard]] iterator begin() const noexcept { return iterator(values_.begin()); }
   [[nodiscard]] iterator end() const noexcept { return iterator(values_.end()); }
   [[nodiscard]] std::size_t size() const noexcept { return values_.size(); }

  private:
   std::span< const SemanticAtom > values_;
};

class LiteralsView {
  public:
   explicit LiteralsView(std::span< const SemanticLiteral > values) : values_(values) {}

   class iterator {
      using Base = std::span< const SemanticLiteral >::iterator;

     public:
      using iterator_category = std::forward_iterator_tag;
      using value_type = LiteralView;
      using difference_type = std::ptrdiff_t;

      explicit iterator(Base value = {}) : value_(value) {}
      [[nodiscard]] LiteralView operator*() const noexcept { return LiteralView{&*value_}; }
      iterator& operator++() noexcept
      {
         ++value_;
         return *this;
      }
      iterator operator++(int) noexcept
      {
         auto copy = *this;
         ++value_;
         return copy;
      }
      friend bool operator==(const iterator&, const iterator&) = default;

     private:
      Base value_;
   };

   [[nodiscard]] iterator begin() const noexcept { return iterator(values_.begin()); }
   [[nodiscard]] iterator end() const noexcept { return iterator(values_.end()); }
   [[nodiscard]] std::size_t size() const noexcept { return values_.size(); }

  private:
   std::span< const SemanticLiteral > values_;
};

class GroundActionsView {
  public:
   explicit GroundActionsView(std::span< const SemanticGroundAction > values) : values_(values) {}

   class iterator {
      using Base = std::span< const SemanticGroundAction >::iterator;

     public:
      using iterator_category = std::forward_iterator_tag;
      using value_type = GroundActionView;
      using difference_type = std::ptrdiff_t;

      explicit iterator(Base value = {}) : value_(value) {}
      [[nodiscard]] GroundActionView operator*() const noexcept
      {
         return GroundActionView{&*value_};
      }
      iterator& operator++() noexcept
      {
         ++value_;
         return *this;
      }
      iterator operator++(int) noexcept
      {
         auto copy = *this;
         ++value_;
         return copy;
      }
      friend bool operator==(const iterator&, const iterator&) = default;

     private:
      Base value_;
   };

   [[nodiscard]] iterator begin() const noexcept { return iterator(values_.begin()); }
   [[nodiscard]] iterator end() const noexcept { return iterator(values_.end()); }
   [[nodiscard]] std::size_t size() const noexcept { return values_.size(); }

  private:
   std::span< const SemanticGroundAction > values_;
};

class SubgoalLayersView {
   using Base = std::span< const std::vector< SemanticLiteral > >::iterator;

  public:
   explicit SubgoalLayersView(std::span< const std::vector< SemanticLiteral > > values)
       : values_(values)
   {
   }

   class iterator {
     public:
      using iterator_category = std::forward_iterator_tag;
      using value_type = LiteralsView;
      using difference_type = std::ptrdiff_t;

      explicit iterator(Base value = {}) : value_(value) {}
      [[nodiscard]] LiteralsView operator*() const noexcept
      {
         return LiteralsView(std::span{value_->data(), value_->size()});
      }
      iterator& operator++() noexcept
      {
         ++value_;
         return *this;
      }
      iterator operator++(int) noexcept
      {
         auto copy = *this;
         ++value_;
         return copy;
      }
      friend bool operator==(const iterator&, const iterator&) = default;

     private:
      Base value_;
   };

   [[nodiscard]] iterator begin() const noexcept { return iterator(values_.begin()); }
   [[nodiscard]] iterator end() const noexcept { return iterator(values_.end()); }
   [[nodiscard]] std::size_t size() const noexcept { return values_.size(); }

  private:
   std::span< const std::vector< SemanticLiteral > > values_;
};

class HistoryEntryView: public views::HistoryEntryViewBase< HistoryEntryView > {
  public:
   explicit HistoryEntryView(const SemanticHistoryEntry* value) : value_(value) {}

   [[nodiscard]] std::int64_t dt_impl() const noexcept
   {
      return value_ == nullptr ? 0 : value_->dt;
   }
   [[nodiscard]] LiteralsView literals_impl() const noexcept
   {
      if(value_ == nullptr) {
         return LiteralsView(std::span< const SemanticLiteral >{});
      }
      return LiteralsView(std::span{value_->literals.data(), value_->literals.size()});
   }

  private:
   const SemanticHistoryEntry* value_;
};

class HistoryView {
   using Base = std::span< const SemanticHistoryEntry >::iterator;

  public:
   explicit HistoryView(std::span< const SemanticHistoryEntry > values) : values_(values) {}

   class iterator {
     public:
      using iterator_category = std::forward_iterator_tag;
      using value_type = HistoryEntryView;
      using difference_type = std::ptrdiff_t;

      explicit iterator(Base value = {}) : value_(value) {}
      [[nodiscard]] HistoryEntryView operator*() const noexcept
      {
         return HistoryEntryView{&*value_};
      }
      iterator& operator++() noexcept
      {
         ++value_;
         return *this;
      }
      iterator operator++(int) noexcept
      {
         auto copy = *this;
         ++value_;
         return copy;
      }
      friend bool operator==(const iterator&, const iterator&) = default;

     private:
      Base value_;
   };

   [[nodiscard]] iterator begin() const noexcept { return iterator(values_.begin()); }
   [[nodiscard]] iterator end() const noexcept { return iterator(values_.end()); }
   [[nodiscard]] std::size_t size() const noexcept { return values_.size(); }

  private:
   std::span< const SemanticHistoryEntry > values_;
};

struct StateView: views::StateViewBase< StateView > {
   AtomsView fluent;
   AtomsView derived;

   StateView(AtomsView fluent_atoms, AtomsView derived_atoms)
       : fluent(fluent_atoms), derived(derived_atoms)
   {
   }

   [[nodiscard]] AtomsView fluent_atoms_impl() const noexcept { return fluent; }
   [[nodiscard]] AtomsView derived_atoms_impl() const noexcept { return derived; }
};

static_assert(views::AtomView< AtomView >);
static_assert(views::LiteralView< LiteralView >);
static_assert(views::GroundActionView< GroundActionView >);
static_assert(views::StateView< StateView >);
static_assert(views::AtomRange< AtomsView >);
static_assert(views::LiteralRange< LiteralsView >);
static_assert(views::GroundActionRange< GroundActionsView >);
static_assert(views::LiteralLayerRange< SubgoalLayersView >);
static_assert(views::HistoryRange< HistoryView >);
static_assert(std::is_trivially_copyable_v< AtomView >);
static_assert(sizeof(AtomView) <= 2 * sizeof(void*));

}  // namespace mifrost::semantic
