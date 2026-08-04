#include "semantic_flat_encoder.hpp"

#include <algorithm>
#include <cstdint>
#include <ranges>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>

namespace mifrost::pytyr {
namespace {

using Category = SemanticPredicateCategory;

int canonical_category_rank(Category category)
{
   // Match PredicateCategory.value lexical ordering in the Python semantic
   // contract: derived, fluent, static.
   switch(category) {
      case Category::derived: return 0;
      case Category::fluent: return 1;
      case Category::static_predicate: return 2;
   }
   throw std::invalid_argument("unknown semantic predicate category");
}

template < typename Index >
size_t raw_index(const Index& index)
{
   if(index.is_max()) {
      throw std::invalid_argument("PyTyr adapter received an invalid repository index");
   }
   return static_cast< size_t >(index.get_value());
}

void set_dense_id(std::vector< int64_t >& ids, size_t raw, int64_t compact)
{
   if(raw >= ids.size()) {
      ids.resize(raw + 1, -1);
   }
   if(ids[raw] != -1) {
      throw std::invalid_argument("PyTyr adapter encountered a duplicate repository index");
   }
   ids[raw] = compact;
}

void set_dense_atom(std::vector< SemanticAtom >& atoms, size_t raw, SemanticAtom compact)
{
   if(raw >= atoms.size()) {
      atoms.resize(raw + 1);
   }
   if(atoms[raw].predicate != -1) {
      throw std::invalid_argument("PyTyr adapter encountered a duplicate ground atom index");
   }
   atoms[raw] = std::move(compact);
}

void cache_dense_atom(std::vector< SemanticAtom >& atoms, size_t raw, SemanticAtom compact)
{
   if(raw >= atoms.size()) {
      atoms.resize(raw + 1);
   }
   if(atoms[raw].predicate == -1) {
      atoms[raw] = std::move(compact);
   }
}

int64_t dense_id(const std::vector< int64_t >& ids, size_t raw, std::string_view kind)
{
   if(raw >= ids.size() or ids[raw] < 0) {
      throw std::invalid_argument(
         "PyTyr " + std::string(kind) + " is outside the adapter task context"
      );
   }
   return ids[raw];
}

Category category_from_raw(int64_t value)
{
   switch(value) {
      case static_cast< int64_t >(Category::static_predicate): return Category::static_predicate;
      case static_cast< int64_t >(Category::fluent): return Category::fluent;
      case static_cast< int64_t >(Category::derived): return Category::derived;
      default: throw std::invalid_argument("PyTyr literal has an invalid predicate category");
   }
}

}  // namespace

struct SemanticPlanningTaskAdapter::Impl {
   tyr::formalism::planning::PlanningTask task;
   std::shared_ptr< SemanticTaskContext > task_context = std::make_shared< SemanticTaskContext >();
   std::vector< int64_t > static_predicate_ids;
   std::vector< int64_t > fluent_predicate_ids;
   std::vector< int64_t > derived_predicate_ids;
   std::vector< int64_t > action_ids;
   std::vector< int64_t > object_ids;
   views::Context view_context;
   std::vector< SemanticAtom > static_atoms;
   std::vector< SemanticAtom > fluent_atoms;
   std::vector< SemanticAtom > derived_atoms;

   struct PredicateSeed {
      Category category;
      std::string name;
      int64_t arity;
      size_t raw;
   };
   struct ActionSeed {
      std::string name;
      int64_t arity;
      size_t raw;
   };
   struct ObjectSeed {
      std::string name;
      size_t raw;
   };

   explicit Impl(const tyr::formalism::planning::PlanningTask& task_value)
       : task(task_value),
         view_context(
            task,
            static_predicate_ids,
            fluent_predicate_ids,
            derived_predicate_ids,
            action_ids,
            object_ids
         )
   {
      build_schema();
      build_atom_caches();
      build_static_facts();
      build_goals();
   }

   std::vector< int64_t >& predicate_ids(Category category)
   {
      switch(category) {
         case Category::static_predicate: return static_predicate_ids;
         case Category::fluent: return fluent_predicate_ids;
         case Category::derived: return derived_predicate_ids;
      }
      throw std::invalid_argument("unknown PyTyr predicate category");
   }

   const std::vector< int64_t >& predicate_ids(Category category) const
   {
      return const_cast< Impl* >(this)->predicate_ids(category);
   }

   void build_schema()
   {
      auto& context = *task_context;
      std::vector< PredicateSeed > predicates;
      const auto domain = task.get_task().get_domain();
      const auto append_predicates = [&predicates](const auto& views, Category category) {
         for(const auto view : views) {
            predicates.push_back(
               PredicateSeed{
                  .category = category,
                  .name = std::string(view.get_name()),
                  .arity = static_cast< int64_t >(view.get_arity()),
                  .raw = raw_index(view.get_index()),
               }
            );
         }
      };
      append_predicates(
         domain.template get_predicates< tyr::formalism::StaticTag >(), Category::static_predicate
      );
      append_predicates(
         domain.template get_predicates< tyr::formalism::FluentTag >(), Category::fluent
      );
      append_predicates(
         domain.template get_predicates< tyr::formalism::DerivedTag >(), Category::derived
      );
      std::ranges::sort(predicates, [](const PredicateSeed& lhs, const PredicateSeed& rhs) {
         return std::tuple(canonical_category_rank(lhs.category), lhs.name, lhs.arity)
                < std::tuple(canonical_category_rank(rhs.category), rhs.name, rhs.arity);
      });
      context.predicates.reserve(predicates.size());
      for(const auto& predicate : predicates) {
         auto& ids = predicate_ids(predicate.category);
         set_dense_id(ids, predicate.raw, static_cast< int64_t >(context.predicates.size()));
         context.predicates.push_back(
            SemanticPredicateSpec{predicate.category, predicate.name, predicate.arity}
         );
      }

      std::vector< ActionSeed > actions;
      for(const auto action : domain.get_actions()) {
         actions.push_back(
            ActionSeed{
               .name = std::string(action.get_name()),
               .arity = static_cast< int64_t >(action.get_original_arity()),
               .raw = raw_index(action.get_index()),
            }
         );
      }
      std::ranges::sort(actions, [](const ActionSeed& lhs, const ActionSeed& rhs) {
         return std::tie(lhs.name, lhs.arity) < std::tie(rhs.name, rhs.arity);
      });
      context.actions.reserve(actions.size());
      for(const auto& action : actions) {
         set_dense_id(action_ids, action.raw, static_cast< int64_t >(context.actions.size()));
         context.actions.push_back(SemanticActionSpec{action.name, action.arity});
      }

      std::vector< ObjectSeed > objects;
      for(const auto object : task.get_task().get_objects()) {
         objects.push_back(
            ObjectSeed{std::string(object.get_name()), raw_index(object.get_index())}
         );
      }
      std::ranges::sort(objects, {}, &ObjectSeed::name);
      context.objects.reserve(objects.size());
      for(const auto& object : objects) {
         if(not context.objects.empty() and context.objects.back() == object.name) {
            throw std::invalid_argument(
               "PyTyr task contains duplicate object name: " + object.name
            );
         }
         set_dense_id(object_ids, object.raw, static_cast< int64_t >(context.objects.size()));
         context.objects.push_back(object.name);
      }
   }

   template < typename AtomView >
   SemanticAtom atom(const AtomView& value, Category category) const
   {
      const auto predicate = value.get_predicate();
      SemanticAtom result;
      result.predicate = dense_id(
         predicate_ids(category), raw_index(predicate.get_index()), "atom predicate"
      );
      const auto values = value.get_objects();
      result.arguments.reserve(values.size());
      for(const auto object : values) {
         result.arguments.push_back(
            dense_id(object_ids, raw_index(object.get_index()), "atom object")
         );
      }
      return result;
   }

   std::vector< SemanticAtom >& atoms(Category category)
   {
      switch(category) {
         case Category::static_predicate: return static_atoms;
         case Category::fluent: return fluent_atoms;
         case Category::derived: return derived_atoms;
      }
      throw std::invalid_argument("unknown PyTyr predicate category");
   }

   const std::vector< SemanticAtom >& atoms(Category category) const
   {
      return const_cast< Impl* >(this)->atoms(category);
   }

   template < typename AtomRange >
   void build_atom_cache(const AtomRange& values, Category category)
   {
      auto& cache = atoms(category);
      for(const auto value : values) {
         set_dense_atom(cache, raw_index(value.get_index()), atom(value, category));
      }
   }

   void build_atom_caches()
   {
      const auto task_view = task.get_task();
      build_atom_cache(
         task_view.template get_atoms< tyr::formalism::StaticTag >(), Category::static_predicate
      );
      build_atom_cache(
         task_view.template get_atoms< tyr::formalism::FluentTag >(), Category::fluent
      );
      // Task::fluent_atoms contains initial-state facts only. The goal may
      // reference additional FDR-domain atoms, which are still immutable for
      // this task and can be added once without normal-path name extraction.
      const auto goal = task_view.get_goal();
      for(const auto fact : goal.template get_facts< tyr::formalism::PositiveTag >()) {
         if(const auto value = fact.get_atom()) {
            cache_dense_atom(
               fluent_atoms, raw_index(value->get_index()), atom(*value, Category::fluent)
            );
         }
      }
      for(const auto fact : goal.template get_facts< tyr::formalism::NegativeTag >()) {
         if(const auto value = fact.get_atom()) {
            cache_dense_atom(
               fluent_atoms, raw_index(value->get_index()), atom(*value, Category::fluent)
            );
         }
      }
   }

   template < typename AtomView >
   SemanticAtom cached_atom(const AtomView& value, Category category) const
   {
      // Tyr's task inventory exposes static and fluent ground-atom ranges but
      // deliberately has no DerivedTag case. Derived atom views still provide
      // stable indices, but their repository cannot be enumerated through the
      // public PlanningTask API, so normalize their structural fields without
      // names when they occur in a state or literal lane.
      if(category == Category::derived) {
         return atom(value, category);
      }
      // The task-owned static/fluent ground-atom repository enumerated at
      // adapter construction only covers atoms reachable from the initial
      // state and goal. Lifted successor generation grounds additional atoms
      // lazily as new states are discovered, so a cache miss here is a first
      // sighting of a stable index, not an out-of-context error: compute the
      // (name-free, index-based) atom once and memoize it for later states
      // that share it. This mutation is safe only while the adapter is not
      // called concurrently without the GIL held (see pytyr_module.cpp).
      auto& cache = const_cast< Impl* >(this)->atoms(category);
      const auto raw = raw_index(value.get_index());
      if(raw < cache.size() and cache[raw].predicate >= 0) {
         return cache[raw];
      }
      auto result = atom(value, category);
      if(raw >= cache.size()) {
         cache.resize(raw + 1);
      }
      cache[raw] = result;
      return result;
   }

   template < typename LiteralView >
   SemanticLiteral literal(const LiteralView& value, Category category) const
   {
      return {cached_atom(value.get_atom(), category), static_cast< bool >(value.get_polarity())};
   }

   SemanticLiteral raw_literal(
      int64_t raw_category,
      int64_t raw_predicate,
      const std::vector< int64_t >& raw_objects,
      bool positive
   ) const
   {
      if(raw_predicate < 0) {
         throw std::invalid_argument("PyTyr literal has a negative predicate index");
      }
      SemanticAtom result;
      const auto category = category_from_raw(raw_category);
      result.predicate = dense_id(
         predicate_ids(category), static_cast< size_t >(raw_predicate), "literal predicate"
      );
      result.arguments.reserve(raw_objects.size());
      for(const auto raw_object : raw_objects) {
         if(raw_object < 0) {
            throw std::invalid_argument("PyTyr literal has a negative object index");
         }
         result.arguments.push_back(
            dense_id(object_ids, static_cast< size_t >(raw_object), "literal object")
         );
      }
      return SemanticLiteral{std::move(result), positive};
   }

   SemanticGroundAction action(const tyr::formalism::planning::GroundActionView& value) const
   {
      const auto schema = value.get_action();
      SemanticGroundAction result;
      result.action = dense_id(action_ids, raw_index(schema.get_index()), "ground action schema");
      const auto values = value.get_objects();
      result.arguments.reserve(values.size());
      for(const auto object : values) {
         result.arguments.push_back(
            dense_id(object_ids, raw_index(object.get_index()), "ground action object")
         );
      }
      return result;
   }

   void build_static_facts()
   {
      auto& facts = task_context->static_facts;
      facts.reserve(static_atoms.size());
      for(const auto& atom_value : static_atoms) {
         if(atom_value.predicate >= 0) {
            facts.push_back(atom_value);
         }
      }
      std::ranges::sort(facts);
   }

   void build_goals()
   {
      auto& goals = task_context->default_goals;
      const auto goal = task.get_task().get_goal();
      for(const auto value : goal.template get_literals< tyr::formalism::StaticTag >()) {
         goals.push_back(literal(value, Category::static_predicate));
      }
      for(const auto fact : goal.template get_facts< tyr::formalism::PositiveTag >()) {
         if(const auto value = fact.get_atom()) {
            goals.push_back({cached_atom(*value, Category::fluent), true});
         }
      }
      for(const auto fact : goal.template get_facts< tyr::formalism::NegativeTag >()) {
         if(const auto value = fact.get_atom()) {
            goals.push_back({cached_atom(*value, Category::fluent), false});
         }
      }
      for(const auto value : goal.template get_literals< tyr::formalism::DerivedTag >()) {
         goals.push_back(literal(value, Category::derived));
      }
      std::ranges::sort(goals);
   }

   template < typename State >
   SemanticFlatRelationInput make_input(
      const State& state,
      const std::vector< tyr::formalism::planning::GroundActionView >& action_values
   ) const
   {
      SemanticFlatRelationInput result;
      result.task_context = task_context;
      result.use_default_goals = true;

      std::vector< SemanticAtom > fluent_atoms;
      for(const auto fact : state.get_fluent_facts_view()) {
         if(const auto value = fact.get_atom()) {
            fluent_atoms.push_back(cached_atom(*value, Category::fluent));
         }
      }
      std::ranges::sort(fluent_atoms);

      std::vector< SemanticAtom > derived_atoms;
      for(const auto value : state.get_derived_atoms_view()) {
         derived_atoms.push_back(cached_atom(value, Category::derived));
      }
      std::ranges::sort(derived_atoms);

      result.state_facts.reserve(fluent_atoms.size() + derived_atoms.size());
      result.state_facts.insert(result.state_facts.end(), fluent_atoms.begin(), fluent_atoms.end());
      result.state_facts.insert(
         result.state_facts.end(), derived_atoms.begin(), derived_atoms.end()
      );

      result.actions.reserve(action_values.size());
      for(const auto& value : action_values) {
         result.actions.push_back(action(value));
      }
      return result;
   }
};

SemanticPlanningTaskAdapter::SemanticPlanningTaskAdapter(
   const tyr::formalism::planning::PlanningTask& task
)
    : impl_(std::make_unique< Impl >(task))
{
}

SemanticPlanningTaskAdapter::SemanticPlanningTaskAdapter(SemanticPlanningTaskAdapter&&) noexcept =
   default;

SemanticPlanningTaskAdapter& SemanticPlanningTaskAdapter::operator=(
   SemanticPlanningTaskAdapter&&
) noexcept = default;

SemanticPlanningTaskAdapter::~SemanticPlanningTaskAdapter() = default;

SemanticFlatRelationInput SemanticPlanningTaskAdapter::make_input(
   const tyr::planning::StateView< tyr::planning::LiftedTag >& state,
   const std::vector< tyr::formalism::planning::GroundActionView >& actions
) const
{
   return impl_->make_input(state, actions);
}

SemanticLiteral SemanticPlanningTaskAdapter::make_raw_literal(
   int64_t category,
   int64_t predicate_index,
   const std::vector< int64_t >& object_indices,
   bool positive
) const
{
   return impl_->raw_literal(category, predicate_index, object_indices, positive);
}

SemanticFlatRelationInput SemanticPlanningTaskAdapter::make_input(
   const tyr::planning::StateView< tyr::planning::GroundTag >& state,
   const std::vector< tyr::formalism::planning::GroundActionView >& actions
) const
{
   return impl_->make_input(state, actions);
}

std::shared_ptr< const SemanticTaskContext > SemanticPlanningTaskAdapter::get_task_context() const
{
   return impl_->task_context;
}

const views::Context& SemanticPlanningTaskAdapter::get_view_context() const noexcept
{
   return impl_->view_context;
}

}  // namespace mifrost::pytyr
