/**
 * @file views.hpp
 * @brief Granular non-owning Views over retained semantic records.
 */
#pragma once

#include <cassert>
#include <cstddef>
#include <span>
#include <string_view>
#include <type_traits>

#include "mifrost/core/encoders/flat/semantic_flat_relation_encoder.hpp"
#include "mifrost/core/views/concepts.hpp"
#include "mifrost/core/views/ids.hpp"
#include "mifrost/core/views/ranges.hpp"

namespace mifrost::semantic {

using views::ActionSchemaId;
using views::ObjectId;
using views::PredicateCategory;
using views::PredicateId;

struct PredicateView {
   const SemanticPredicateSpec* value = nullptr;
   PredicateId value_id = -1;

   [[nodiscard]] PredicateId id() const noexcept { return value_id; }
   [[nodiscard]] std::string_view name() const noexcept
   {
      return value == nullptr ? std::string_view{} : value->name;
   }
   [[nodiscard]] std::size_t arity() const noexcept
   {
      return value == nullptr ? 0U : static_cast< std::size_t >(value->arity);
   }
   [[nodiscard]] PredicateCategory category() const noexcept
   {
      if(value == nullptr) {
         return PredicateCategory::fluent;
      }
      return static_cast< PredicateCategory >(static_cast< int8_t >(value->category));
   }
};

struct ObjectView {
   ObjectId value = -1;
   std::string_view object_name{};

   [[nodiscard]] ObjectId id() const noexcept { return value; }
   [[nodiscard]] std::string_view name() const noexcept { return object_name; }
};

struct AtomView {
   const SemanticAtom* value = nullptr;

   [[nodiscard]] PredicateId predicate_id() const noexcept
   {
      return value == nullptr ? -1 : value->predicate;
   }
   [[nodiscard]] std::span< const ObjectId > arguments() const noexcept
   {
      if(value == nullptr) {
         return {};
      }
      return {value->arguments.data(), value->arguments.size()};
   }
};

struct LiteralView {
   const SemanticLiteral* value = nullptr;

   [[nodiscard]] bool is_negated() const noexcept
   {
      return value != nullptr and not value->positive;
   }
   [[nodiscard]] AtomView atom() const noexcept
   {
      return value == nullptr ? AtomView{} : AtomView{&value->atom};
   }
};

struct ActionSchemaView {
   const SemanticActionSpec* value = nullptr;
   ActionSchemaId value_id = -1;

   [[nodiscard]] ActionSchemaId id() const noexcept { return value_id; }
   [[nodiscard]] std::string_view name() const noexcept
   {
      return value == nullptr ? std::string_view{} : value->name;
   }
   [[nodiscard]] std::size_t arity() const noexcept
   {
      return value == nullptr ? 0U : static_cast< std::size_t >(value->arity);
   }
};

struct GroundActionView {
   const SemanticGroundAction* value = nullptr;

   [[nodiscard]] ActionSchemaId schema_id() const noexcept
   {
      return value == nullptr ? -1 : value->action;
   }
   [[nodiscard]] std::span< const ObjectId > arguments() const noexcept
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

struct StateView {
   AtomsView fluent;
   AtomsView derived;

   [[nodiscard]] AtomsView fluent_atoms() const noexcept { return fluent; }
   [[nodiscard]] AtomsView derived_atoms() const noexcept { return derived; }
};

static_assert(views::AtomView< AtomView >);
static_assert(views::LiteralView< LiteralView >);
static_assert(views::GroundActionView< GroundActionView >);
static_assert(views::StateView< StateView >);
static_assert(std::is_trivially_copyable_v< AtomView >);
static_assert(sizeof(AtomView) <= 2 * sizeof(void*));

}  // namespace mifrost::semantic
