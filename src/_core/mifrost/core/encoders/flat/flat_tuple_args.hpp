#pragma once

#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "flat_entity_context.hpp"
#include "flat_tuple_layout.hpp"

namespace mifrost {

/// Template implementations

template < typename Context, typename AtomTag >
std::vector< int64_t > flat_logical_arg_rows_for_atom(
   const Context& context,
   const mimir::formalism::GroundAtom< AtomTag >& atom,
   std::string_view missing_object_prefix
)
{
   std::vector< int64_t > args;
   args.reserve(atom->get_objects().size());
   for(const auto& obj : atom->get_objects()) {
      const auto it = context.entity_index_by_object_id.find(
         static_cast< int64_t >(obj->get_index())
      );
      if(it == context.entity_index_by_object_id.end()) {
         throw std::invalid_argument(
            std::string(missing_object_prefix) + RelationFormatter::format_object(*obj)
         );
      }
      args.push_back(it->second);
   }
   return args;
}

template < typename Context, typename AtomTag >
std::vector< int64_t > build_flat_atom_tuple_args(
   Context& context,
   const mimir::formalism::GroundAtom< AtomTag >& atom,
   std::span< const int64_t > auxiliary_args,
   bool use_predicate_virtual_nodes,
   std::string_view missing_object_prefix
)
{
   const auto logical_args = flat_logical_arg_rows_for_atom(context, atom, missing_object_prefix);
   std::optional< int64_t > predicate_virtual_index = std::nullopt;
   if(use_predicate_virtual_nodes) {
      predicate_virtual_index = ensure_predicate_virtual_entity_for_atom(context, atom);
   }
   return build_flat_tuple_args(std::span{logical_args}, auxiliary_args, predicate_virtual_index);
}

template < typename Context >
std::vector< int64_t > build_flat_action_tuple_args(
   const Context& context,
   const mimir::formalism::GroundAction& action,
   std::span< const int64_t > auxiliary_args,
   std::string_view missing_object_prefix
)
{
   std::vector< int64_t > logical_args;
   logical_args.reserve(action->get_objects().size());
   for(const auto& obj : action->get_objects()) {
      const auto it = context.entity_index_by_object_id.find(
         static_cast< int64_t >(obj->get_index())
      );
      if(it == context.entity_index_by_object_id.end()) {
         throw std::invalid_argument(
            std::string(missing_object_prefix) + RelationFormatter::format_object(*obj)
         );
      }
      logical_args.push_back(it->second);
   }
   return build_flat_tuple_args(std::span{logical_args}, auxiliary_args, std::nullopt);
}

}  // namespace mifrost
