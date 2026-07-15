#include "semantic_flat_encoder.hpp"

#include <algorithm>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

namespace mifrost::pytyr {
namespace {

using Category = SemanticPredicateCategory;
using PredicateKey = std::tuple< Category, std::string, int64_t >;
using ActionKey = std::pair< std::string, int64_t >;

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

struct PredicateKeyLess {
   bool operator()(const PredicateKey& lhs, const PredicateKey& rhs) const
   {
      const auto& [lhs_category, lhs_name, lhs_arity] = lhs;
      const auto& [rhs_category, rhs_name, rhs_arity] = rhs;
      return std::tuple(canonical_category_rank(lhs_category), lhs_name, lhs_arity)
             < std::tuple(canonical_category_rank(rhs_category), rhs_name, rhs_arity);
   }
};

template < typename View >
std::string name_of(const View& view)
{
   return std::string(view.get_name());
}

template < typename Range >
std::vector< std::string > object_names(const Range& objects)
{
   std::vector< std::string > result;
   result.reserve(objects.size());
   for(const auto object : objects) {
      result.push_back(name_of(object));
   }
   return result;
}

}  // namespace

struct SemanticFlatRelationEncoder::Impl {
   tyr::formalism::planning::PlanningTask task;
   std::map< PredicateKey, int64_t, PredicateKeyLess > predicate_indices;
   std::map< ActionKey, int64_t > action_indices;
   std::map< std::string, int64_t > object_indices;
   std::vector< std::string > objects;
   std::vector< SemanticLiteral > goals;
   SemanticFlatRelationEncoderEngine engine;

   static std::vector< SemanticPredicateSpec > build_predicates(
      const tyr::formalism::planning::PlanningTask& task
   )
   {
      std::vector< SemanticPredicateSpec > result;
      const auto domain = task.get_task().get_domain();

      const auto append = [&result](const auto& predicates, Category category) {
         for(const auto predicate : predicates) {
            result.push_back(
               {category, name_of(predicate), static_cast< int64_t >(predicate.get_arity())}
            );
         }
      };
      append(
         domain.template get_predicates< tyr::formalism::StaticTag >(), Category::static_predicate
      );
      append(domain.template get_predicates< tyr::formalism::FluentTag >(), Category::fluent);
      append(domain.template get_predicates< tyr::formalism::DerivedTag >(), Category::derived);

      std::ranges::sort(result, [](const auto& lhs, const auto& rhs) {
         return std::tuple(canonical_category_rank(lhs.category), lhs.name, lhs.arity)
                < std::tuple(canonical_category_rank(rhs.category), rhs.name, rhs.arity);
      });
      return result;
   }

   static std::vector< SemanticActionSpec > build_actions(
      const tyr::formalism::planning::PlanningTask& task
   )
   {
      std::vector< SemanticActionSpec > result;
      for(const auto action : task.get_task().get_domain().get_actions()) {
         result.push_back({name_of(action), static_cast< int64_t >(action.get_original_arity())});
      }
      std::ranges::sort(result, [](const auto& lhs, const auto& rhs) {
         return std::tie(lhs.name, lhs.arity) < std::tie(rhs.name, rhs.arity);
      });
      return result;
   }

   Impl(const tyr::formalism::planning::PlanningTask& task_value, Config config)
       : task(task_value), engine(build_predicates(task), build_actions(task), std::move(config))
   {
      const auto& predicates = engine.get_predicates();
      for(size_t index = 0; index < predicates.size(); ++index) {
         const auto& predicate = predicates[index];
         predicate_indices.emplace(
            PredicateKey{predicate.category, predicate.name, predicate.arity},
            static_cast< int64_t >(index)
         );
      }

      const auto& actions = engine.get_actions();
      for(size_t index = 0; index < actions.size(); ++index) {
         action_indices.emplace(
            ActionKey{actions[index].name, actions[index].arity}, static_cast< int64_t >(index)
         );
      }

      objects = object_names(task.get_task().get_objects());
      std::ranges::sort(objects);
      for(size_t index = 0; index < objects.size(); ++index) {
         const auto [_, inserted] = object_indices.emplace(
            objects[index], static_cast< int64_t >(index)
         );
         if(! inserted) {
            throw std::invalid_argument(
               "PyTyr task contains duplicate object name: " + objects[index]
            );
         }
      }

      build_goals();
   }

   template < typename AtomView >
   SemanticAtom atom(const AtomView& value, Category category) const
   {
      const auto predicate = value.get_predicate();
      const PredicateKey key{
         category,
         name_of(predicate),
         static_cast< int64_t >(predicate.get_arity()),
      };
      const auto predicate_it = predicate_indices.find(key);
      if(predicate_it == predicate_indices.end()) {
         throw std::invalid_argument("PyTyr atom predicate is outside the adapter domain");
      }

      SemanticAtom result;
      result.predicate = predicate_it->second;
      const auto values = value.get_objects();
      result.arguments.reserve(values.size());
      for(const auto object : values) {
         const auto object_name = name_of(object);
         const auto object_it = object_indices.find(object_name);
         if(object_it == object_indices.end()) {
            throw std::invalid_argument(
               "PyTyr atom object is outside the adapter task: " + object_name
            );
         }
         result.arguments.push_back(object_it->second);
      }
      return result;
   }

   template < typename LiteralView >
   SemanticLiteral literal(const LiteralView& value, Category category) const
   {
      return {atom(value.get_atom(), category), static_cast< bool >(value.get_polarity())};
   }

   SemanticGroundAction action(const tyr::formalism::planning::GroundActionView& value) const
   {
      const auto schema = value.get_action();
      const ActionKey key{name_of(schema), static_cast< int64_t >(schema.get_original_arity())};
      const auto action_it = action_indices.find(key);
      if(action_it == action_indices.end()) {
         throw std::invalid_argument("PyTyr ground action is outside the adapter domain");
      }

      SemanticGroundAction result;
      result.action = action_it->second;
      const auto values = value.get_objects();
      result.arguments.reserve(values.size());
      for(const auto object : values) {
         const auto object_name = name_of(object);
         const auto object_it = object_indices.find(object_name);
         if(object_it == object_indices.end()) {
            throw std::invalid_argument(
               "PyTyr action object is outside the adapter task: " + object_name
            );
         }
         result.arguments.push_back(object_it->second);
      }
      return result;
   }

   void build_goals()
   {
      const auto goal = task.get_task().get_goal();
      for(const auto value : goal.template get_literals< tyr::formalism::StaticTag >()) {
         goals.push_back(literal(value, Category::static_predicate));
      }
      for(const auto fact : goal.template get_facts< tyr::formalism::PositiveTag >()) {
         if(const auto value = fact.get_atom()) {
            goals.push_back({atom(*value, Category::fluent), true});
         }
      }
      for(const auto fact : goal.template get_facts< tyr::formalism::NegativeTag >()) {
         if(const auto value = fact.get_atom()) {
            goals.push_back({atom(*value, Category::fluent), false});
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
      result.objects = objects;
      result.goals = goals;

      std::vector< SemanticAtom > static_atoms;
      for(const auto value : state.get_static_atoms_view()) {
         static_atoms.push_back(atom(value, Category::static_predicate));
      }
      std::ranges::sort(static_atoms);

      std::vector< SemanticAtom > fluent_atoms;
      for(const auto fact : state.get_fluent_facts_view()) {
         if(const auto value = fact.get_atom()) {
            fluent_atoms.push_back(atom(*value, Category::fluent));
         }
      }
      std::ranges::sort(fluent_atoms);

      std::vector< SemanticAtom > derived_atoms;
      for(const auto value : state.get_derived_atoms_view()) {
         derived_atoms.push_back(atom(value, Category::derived));
      }
      std::ranges::sort(derived_atoms);

      result.state_facts.reserve(static_atoms.size() + fluent_atoms.size() + derived_atoms.size());
      result.state_facts.insert(result.state_facts.end(), static_atoms.begin(), static_atoms.end());
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

SemanticFlatRelationEncoder::SemanticFlatRelationEncoder(
   const tyr::formalism::planning::PlanningTask& task,
   Config config
)
    : impl_(std::make_unique< Impl >(task, std::move(config)))
{
}

SemanticFlatRelationEncoder::SemanticFlatRelationEncoder(SemanticFlatRelationEncoder&&) noexcept =
   default;

SemanticFlatRelationEncoder& SemanticFlatRelationEncoder::operator=(
   SemanticFlatRelationEncoder&&
) noexcept = default;

SemanticFlatRelationEncoder::~SemanticFlatRelationEncoder() = default;

SemanticFlatRelationInput SemanticFlatRelationEncoder::make_input(
   const tyr::planning::StateView< tyr::planning::LiftedTag >& state,
   const std::vector< tyr::formalism::planning::GroundActionView >& actions
) const
{
   return impl_->make_input(state, actions);
}

SemanticFlatRelationInput SemanticFlatRelationEncoder::make_input(
   const tyr::planning::StateView< tyr::planning::GroundTag >& state,
   const std::vector< tyr::formalism::planning::GroundActionView >& actions
) const
{
   return impl_->make_input(state, actions);
}

BatchBuilder::BatchEncoding SemanticFlatRelationEncoder::encode(
   const tyr::planning::StateView< tyr::planning::LiftedTag >& state,
   const std::vector< tyr::formalism::planning::GroundActionView >& actions
) const
{
   return impl_->engine.encode(impl_->make_input(state, actions));
}

BatchBuilder::BatchEncoding SemanticFlatRelationEncoder::encode(
   const tyr::planning::StateView< tyr::planning::GroundTag >& state,
   const std::vector< tyr::formalism::planning::GroundActionView >& actions
) const
{
   return impl_->engine.encode(impl_->make_input(state, actions));
}

const SemanticFlatRelationEncoderEngine& SemanticFlatRelationEncoder::get_engine() const
{
   return impl_->engine;
}

}  // namespace mifrost::pytyr
