/**
 * @file semantic_assembly.hpp
 * @brief Backend-neutral ownership primitives for extensible semantic encoders.
 */
#pragma once

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>
#include <utility>
#include <vector>

#include "mifrost/core/api.hpp"

namespace mifrost {

/**
 * Immutable-by-convention graph-local data carried beside a semantic input.
 *
 * Values are owned through `shared_ptr<const T>`. Copying the map is cheap and
 * keeps every value alive for synchronous single-graph and batch encoding.
 * Engines retain no annotations between calls, so immutable compiled engines
 * can be reused concurrently without global or engine-local mutable state.
 */
class MIFROST_API SemanticAnnotations {
  public:
   SemanticAnnotations() = default;

   template < typename T >
   void set(std::string key, std::shared_ptr< const T > value)
   {
      if(key.empty()) {
         throw std::invalid_argument("Semantic annotation key must not be empty");
      }
      if(not value) {
         throw std::invalid_argument("Semantic annotation value must not be null");
      }
      entries_.insert_or_assign(
         std::move(key), Entry{std::move(value), std::type_index(typeid(T))}
      );
   }

   template < typename T, typename... Args >
   std::shared_ptr< const T > emplace(std::string key, Args&&... args)
   {
      auto value = std::make_shared< const T >(std::forward< Args >(args)...);
      set< T >(std::move(key), value);
      return value;
   }

   template < typename T >
   [[nodiscard]] const T* find(std::string_view key) const
   {
      const auto it = entries_.find(std::string(key));
      if(it == entries_.end()) {
         return nullptr;
      }
      if(it->second.type != std::type_index(typeid(T))) {
         throw std::invalid_argument(
            "Semantic annotation '" + std::string(key) + "' has the wrong type"
         );
      }
      return static_cast< const T* >(it->second.value.get());
   }

   template < typename T >
   [[nodiscard]] const T& get(std::string_view key) const
   {
      const auto* value = find< T >(key);
      if(value == nullptr) {
         throw std::invalid_argument(
            "Semantic annotation '" + std::string(key) + "' is not present"
         );
      }
      return *value;
   }

   [[nodiscard]] bool contains(std::string_view key) const;
   [[nodiscard]] bool empty() const noexcept { return entries_.empty(); }
   [[nodiscard]] size_t size() const noexcept { return entries_.size(); }

  private:
   struct Entry {
      std::shared_ptr< const void > value;
      std::type_index type;
   };
   std::unordered_map< std::string, Entry > entries_;
};

/**
 * Borrowed-or-owned semantic source plus graph-local annotations.
 *
 * A reference-backed input is valid for the duration of synchronous `encode`
 * or `encode_batch`. The shared-pointer form owns the source for as long as the
 * carrier is retained. Rvalues are rejected to prevent accidental dangling
 * references.
 */
template < typename Source >
class SemanticEncoderInput {
  public:
   explicit SemanticEncoderInput(const Source& source, SemanticAnnotations annotations = {})
       : source_(&source), annotations_(std::move(annotations))
   {
   }

   SemanticEncoderInput(Source&&, SemanticAnnotations = {}) = delete;
   SemanticEncoderInput(const Source&&, SemanticAnnotations = {}) = delete;

   explicit SemanticEncoderInput(
      std::shared_ptr< const Source > source,
      SemanticAnnotations annotations = {}
   )
       : owned_source_(std::move(source)),
         source_(owned_source_.get()),
         annotations_(std::move(annotations))
   {
      if(source_ == nullptr) {
         throw std::invalid_argument("Semantic encoder input source must not be null");
      }
   }

   [[nodiscard]] const Source& source() const
   {
      if(source_ == nullptr) {
         throw std::logic_error("Semantic encoder input source is not initialized");
      }
      return *source_;
   }

   [[nodiscard]] const SemanticAnnotations& annotations() const noexcept { return annotations_; }

  private:
   std::shared_ptr< const Source > owned_source_;
   const Source* source_ = nullptr;
   SemanticAnnotations annotations_;
};

/**
 * Exclusive pre-compilation ownership for one encoder family's components.
 *
 * Concrete family builders add schema/configuration and choose their compiled
 * runtime. This common container deliberately does not impose a universal
 * emitter interface on flat, heterogeneous, and homogeneous encoders.
 */
template < typename Component >
class SemanticAssemblyComponents {
  public:
   SemanticAssemblyComponents() = default;
   SemanticAssemblyComponents(const SemanticAssemblyComponents&) = delete;
   SemanticAssemblyComponents& operator=(const SemanticAssemblyComponents&) = delete;
   SemanticAssemblyComponents(SemanticAssemblyComponents&&) noexcept = default;
   SemanticAssemblyComponents& operator=(SemanticAssemblyComponents&&) noexcept = default;

   void add(std::unique_ptr< Component > component)
   {
      if(frozen_) {
         throw std::logic_error("Semantic assembly components were already frozen");
      }
      if(not component) {
         throw std::invalid_argument("Semantic assembly component must not be null");
      }
      components_.push_back(std::move(component));
   }

   template < typename Concrete, typename... Args >
   void emplace(Args&&... args)
   {
      add(std::make_unique< Concrete >(std::forward< Args >(args)...));
   }

   [[nodiscard]] std::vector< std::shared_ptr< Component > > freeze() &&
   {
      if(frozen_) {
         throw std::logic_error("Semantic assembly components were already frozen");
      }
      frozen_ = true;
      std::vector< std::shared_ptr< Component > > result;
      result.reserve(components_.size());
      for(auto& component : components_) {
         result.emplace_back(std::move(component));
      }
      components_.clear();
      return result;
   }

   [[nodiscard]] size_t size() const noexcept { return components_.size(); }

  private:
   std::vector< std::unique_ptr< Component > > components_;
   bool frozen_ = false;
};

}  // namespace mifrost
