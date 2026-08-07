/**
 * @file problem_adapter_cache.hpp
 * @brief Per-problem `SemanticProblemAdapter` cache shared by the Pymimir facades.
 */
#pragma once

#include <ankerl/unordered_dense.h>

#include <memory>
#include <mimir/formalism/domain.hpp>
#include <mimir/formalism/problem.hpp>
#include <mimir/search/state.hpp>
#include <mimir/search/state_repository.hpp>
#include <utility>

#include "mifrost/backends/pymimir/semantic_views.hpp"
#include "mifrost/core/semantic/records.hpp"

namespace mifrost::pymimir {

/**
 * The per-problem half of an encoder, kept out of the encoder itself.
 *
 * An encoder is built from a domain and holds only a `SemanticSchemaContext`.
 * What is genuinely per instance -- the object table, the canonicalized static
 * facts, the default goals and the backend view context -- lives in a
 * `SemanticProblemAdapter`, and this caches one per problem so a stream of
 * states from several instances neither rebuilds them nor forces the encoder
 * to bind to whichever problem it happened to see first.
 *
 * Adapters for problems of this cache's own domain share its schema instance,
 * so the problem contexts handed out compare equal to the encoder's schema by
 * pointer and batches may mix instances freely. A state from a *different*
 * domain still gets an adapter, with a schema derived from its own domain; it
 * is up to the facade to route it to an engine that can encode it.
 */
class ProblemAdapterCache {
  public:
   ProblemAdapterCache(
      const mimir::formalism::DomainImpl& domain,
      std::shared_ptr< const SemanticSchemaContext > schema
   )
       : domain_(&domain), schema_(std::move(schema))
   {
   }

   [[nodiscard]] const std::shared_ptr< const SemanticSchemaContext >& schema() const noexcept
   {
      return schema_;
   }

   /** The adapter for `state`'s problem, built on first use. */
   [[nodiscard]] SemanticProblemAdapter& adapter_for(const mimir::search::State& state) const
   {
      // Taken from the state's repository rather than `state.get_problem()` so
      // that the key, the liveness guard and the adapter are the same object by
      // construction. Every mimir state is unpacked against its repository's
      // problem, so the two agree.
      const auto& problem = state.get_state_repository()->get_problem();
      const auto* key = problem.get();
      auto& binding = bindings_[key];
      if(binding.adapter == nullptr or binding.problem.lock().get() != key) {
         binding.problem = problem;
         // The shared schema is this *domain's*. A caller may hand an encoder a
         // state from another domain entirely -- the facades handle that by
         // falling back to a compatible engine -- and such an adapter must
         // derive its own schema rather than be labelled with this one.
         const bool same_domain = problem->get_domain().get() == domain_;
         binding.adapter = std::make_unique< SemanticProblemAdapter >(
            *problem, same_domain ? schema_ : nullptr
         );
      }
      return *binding.adapter;
   }

   [[nodiscard]] std::shared_ptr< const SemanticProblemContext > problem_context(
      const mimir::search::State& state
   ) const
   {
      return adapter_for(state).get_problem_context();
   }

  private:
   struct Binding {
      /// Guards the raw map key. A `ProblemImpl*` is unique only among *live*
      /// problems, so a dropped problem's address can be recycled onto a new
      /// one and silently hand back the wrong object table. Rebuilding when
      /// this no longer names the problem the entry was built for is what
      /// makes the cache safe against that.
      std::weak_ptr< mimir::formalism::ProblemImpl > problem;
      std::unique_ptr< SemanticProblemAdapter > adapter;
   };

   const mimir::formalism::DomainImpl* domain_;
   std::shared_ptr< const SemanticSchemaContext > schema_;
   mutable ankerl::unordered_dense::map< const mimir::formalism::ProblemImpl*, Binding > bindings_;
};

}  // namespace mifrost::pymimir
