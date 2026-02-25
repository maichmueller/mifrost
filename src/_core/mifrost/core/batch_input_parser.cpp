#include "mifrost/core/batch_input_parser.hpp"

#include <fmt/format.h>
#include <nanobind/stl/variant.h>

#include <algorithm>
#include <mimir/formalism/ground_literal.hpp>
#include <optional>
#include <string>
#include <utility>

#include "mifrost/common.hpp"

namespace mifrost::batch_input {

namespace nb = nanobind;

namespace {

constexpr std::string_view kNestedActionsError =
   "Nested/tuple action payloads are not supported by HGraphEncoder; "
   "use HorizonEncoder for IW lookahead.";

nb::handle encoder_types_module()
{
   static nb::object* module = [] {
      return new nb::object(nb::module_::import_("mifrost.encoders.types"));
   }();
   return *module;
}

nb::handle resolve_state_adapter_fn()
{
   static nb::object* fn = [] {
      return new nb::object(encoder_types_module().attr("_resolve_state_adapter"));
   }();
   return *fn;
}

nb::handle resolve_literal_adapter_fn()
{
   static nb::object* fn = [] {
      return new nb::object(encoder_types_module().attr("_resolve_literal_adapter"));
   }();
   return *fn;
}

nb::handle resolve_action_adapter_fn()
{
   static nb::object* fn = [] {
      return new nb::object(encoder_types_module().attr("_resolve_action_adapter"));
   }();
   return *fn;
}

bool is_str_bytes_like(nb::handle value)
{
   return nb::isinstance< nb::str >(value) or nb::isinstance< nb::bytes >(value)
          or PyByteArray_Check(value.ptr()) != 0;
}

bool is_sequence_like_but_not_str_bytes(nb::handle value)
{
   return not is_str_bytes_like(value) and PySequence_Check(value.ptr()) != 0;
}

bool is_iterable_but_not_str_bytes(nb::handle value)
{
   if(is_str_bytes_like(value)) {
      return false;
   }
   PyObject* iter = PyObject_GetIter(value.ptr());
   if(iter == nullptr) {
      PyErr_Clear();
      return false;
   }
   Py_DECREF(iter);
   return true;
}

std::string_view py_type_repr(nb::handle value)
{
   nb::object typ = py::builtins_type_type()(value);
   return nb::cast< std::string_view >(nb::repr(typ));
}

template < typename T >
bool can_cast_noerror(nb::handle value)
{
   try {
      (void) nb::cast< T >(value);
      return true;
   } catch(...) {
      PyErr_Clear();
      return false;
   }
}

nb::object state_object_for_cast(nb::handle value)
{
   if(nb::hasattr(value, "_advanced_state")) {
      return value.attr("_advanced_state");
   }
   return nb::borrow< nb::object >(value);
}

nb::object literal_object_for_cast(nb::handle value)
{
   if(nb::hasattr(value, "_advanced_ground_literal")) {
      return value.attr("_advanced_ground_literal");
   }
   return nb::borrow< nb::object >(value);
}

nb::object action_object_for_cast(nb::handle value)
{
   if(nb::hasattr(value, "_advanced_ground_action")) {
      return value.attr("_advanced_ground_action");
   }
   return nb::borrow< nb::object >(value);
}

bool has_adapter(nb::handle resolver, nb::handle value)
{
   return not resolver(value).is_none();
}

bool has_state_adapter(nb::handle value)
{
   return has_adapter(resolve_state_adapter_fn(), value);
}

bool has_literal_adapter(nb::handle value)
{
   return has_adapter(resolve_literal_adapter_fn(), value);
}

bool has_action_adapter(nb::handle value)
{
   return has_adapter(resolve_action_adapter_fn(), value);
}

bool is_native_state_object(nb::handle value)
{
   return can_cast_noerror< mimir::search::State >(state_object_for_cast(value));
}

bool is_wrapper_state_object(nb::handle value)
{
   return nb::hasattr(value, "_advanced_state");
}

bool is_native_action_object(nb::handle value)
{
   return can_cast_noerror< mimir::formalism::GroundAction >(action_object_for_cast(value));
}

std::string state_adapter_error()
{
   return "Batch parsing does not support state adapters; pass wf.State or ase.State";
}

std::string literal_adapter_error()
{
   return "Batch parsing does not support literal adapters; pass native goal literals";
}

std::string_view action_adapter_error()
{
   return "Batch parsing does not support action adapters; pass native ground actions";
}

std::string states_iterable_error(std::string_view field_name)
{
   if(field_name == "states") {
      return "encode_batch expects a state or an iterable of states";
   }
   if(field_name == "successors") {
      return "successors must be a state or an iterable of states";
   }
   return fmt::format("{} must be a state or an iterable of states", field_name);
}

ParsedStateEntry parse_state_entry(nb::handle value, std::string_view field_name, size_t index)
{
   try {
      return ParsedStateEntry{
         .source = nb::borrow< nb::object >(value),
         .state = nb::cast< mimir::search::State >(state_object_for_cast(value)),
         .is_wrapper_state = is_wrapper_state_object(value),
      };
   } catch(...) {
      PyErr_Clear();
      if(has_state_adapter(value)) {
         throw nb::type_error(state_adapter_error().c_str());
      }
      throw nb::type_error(
         fmt::format(
            "{} entry at index {} has invalid type: {}", field_name, index, py_type_repr(value)
         )
            .c_str()
      );
   }
}

LiteralVariant cast_goal_literal_or_raise(
   nb::handle literal,
   std::string_view field,
   std::optional< size_t > entry_idx,
   std::optional< size_t > literal_idx
)
{
   try {
      return nb::cast< LiteralVariant >(literal_object_for_cast(literal));
   } catch(...) {
      PyErr_Clear();
      std::string location;
      if(entry_idx.has_value()) {
         location = fmt::format(" entry at index {}", *entry_idx);
      }
      if(literal_idx.has_value()) {
         location += fmt::format(" literal at position {}", *literal_idx);
      }
      if(has_literal_adapter(literal)) {
         throw nb::type_error(literal_adapter_error().c_str());
      }
      throw nb::type_error(
         fmt::format(
            "{}{} has invalid goal literal type: {}", field, location, py_type_repr(literal)
         )
            .c_str()
      );
   }
}

mimir::formalism::GroundAction cast_action_or_raise(
   nb::handle action,
   std::string_view field,
   std::optional< size_t > entry_idx,
   std::optional< size_t > action_idx
)
{
   try {
      return nb::cast< mimir::formalism::GroundAction >(action_object_for_cast(action));
   } catch(...) {
      PyErr_Clear();
      std::string location;
      if(entry_idx.has_value()) {
         location = fmt::format(" entry at index {}", *entry_idx);
      }
      if(action_idx.has_value()) {
         location += fmt::format(" action at position {}", *action_idx);
      }
      if(has_action_adapter(action)) {
         throw nb::type_error(action_adapter_error().c_str());
      }
      throw nb::type_error(
         fmt::format("{}{} has invalid action type: {}", field, location, py_type_repr(action))
            .c_str()
      );
   }
}

ParsedGoalPayload parse_goal_literals_iterable(
   nb::handle value,
   std::string_view field,
   std::optional< size_t > entry_idx = std::nullopt
)
{
   if(not is_iterable_but_not_str_bytes(value)) {
      std::string location = entry_idx.has_value() ? fmt::format(" entry at index {}", *entry_idx)
                                                   : "";
      throw nb::type_error(
         fmt::format("{}{} must be an iterable of goal literals or None", field, location).c_str()
      );
   }

   ParsedGoalPayload out;
   const nb::list literals = nb::list(value);
   size_t literal_idx = 0;
   for(nb::handle literal : literals) {
      out.push_back(cast_goal_literal_or_raise(literal, field, entry_idx, literal_idx));
      ++literal_idx;
   }
   return out;
}

ParsedSubgoalLayersPayload parse_subgoal_layers_payload(
   nb::handle value,
   std::string_view field,
   std::optional< size_t > entry_idx = std::nullopt
)
{
   if(not is_iterable_but_not_str_bytes(value)) {
      std::string location = entry_idx.has_value() ? fmt::format(" entry at index {}", *entry_idx)
                                                   : "";
      throw nb::type_error(
         fmt::format("{}{} must be an iterable of goal-literal layers or None", field, location)
            .c_str()
      );
   }

   ParsedSubgoalLayersPayload out;
   const nb::list layers = nb::list(value);
   size_t layer_idx = 0;
   for(nb::handle layer : layers) {
      if(not is_iterable_but_not_str_bytes(layer)) {
         std::string location = entry_idx.has_value()
                                   ? fmt::format(" entry at index {}", *entry_idx)
                                   : "";
         throw nb::type_error(
            fmt::format(
               "{}{} layer at position {} must be an iterable of goal literals",
               field,
               location,
               layer_idx
            )
               .c_str()
         );
      }
      out.push_back(parse_goal_literals_iterable(layer, field, entry_idx));
      ++layer_idx;
   }

   return out;
}

bool is_integral_py(nb::handle value)
{
   return PyLong_Check(value.ptr()) != 0;
}

bool is_history_item(nb::handle value)
{
   if(not nb::isinstance< nb::tuple >(value)) {
      return false;
   }
   const nb::tuple tup = nb::borrow< nb::tuple >(value);
   return nb::len(tup) == 2 and is_integral_py(tup[0]);
}

ParsedHistoryPayload parse_history_payload(
   nb::handle value,
   std::string_view field,
   std::optional< size_t > entry_idx = std::nullopt
)
{
   if(not is_iterable_but_not_str_bytes(value)) {
      std::string location = entry_idx.has_value() ? fmt::format(" entry at index {}", *entry_idx)
                                                   : "";
      throw nb::type_error(
         fmt::format(
            "{}{} must be an iterable of (dt, goal-literals) tuples or None", field, location
         )
            .c_str()
      );
   }

   ParsedHistoryPayload out;
   const nb::list items = nb::list(value);
   size_t history_idx = 0;
   for(nb::handle item : items) {
      if(not is_history_item(item)) {
         std::string location = entry_idx.has_value()
                                   ? fmt::format(" entry at index {}", *entry_idx)
                                   : "";
         throw nb::type_error(
            fmt::format(
               "{}{} item at position {} must be a (dt, literals) tuple",
               field,
               location,
               history_idx
            )
               .c_str()
         );
      }
      const nb::tuple tup = nb::borrow< nb::tuple >(item);
      const int dt = nb::cast< int >(tup[0]);
      const ParsedGoalPayload literals = parse_goal_literals_iterable(tup[1], field, entry_idx);
      out.emplace_back(dt, literals);
      ++history_idx;
   }

   return out;
}

ParsedActionPayload parse_flat_actions_iterable(
   nb::handle value,
   std::string_view field,
   std::optional< size_t > entry_idx = std::nullopt
)
{
   if(not is_iterable_but_not_str_bytes(value)) {
      std::string location = entry_idx.has_value() ? fmt::format(" entry at index {}", *entry_idx)
                                                   : "";
      throw nb::type_error(
         fmt::format("{}{} must be an iterable of actions or None", field, location).c_str()
      );
   }

   ParsedActionPayload out;
   const nb::list actions = nb::list(value);
   size_t action_idx = 0;
   for(nb::handle action : actions) {
      if(is_sequence_like_but_not_str_bytes(action) and not is_native_action_object(action)) {
         throw nb::value_error(std::string(kNestedActionsError).c_str());
      }
      out.push_back(cast_action_or_raise(action, field, entry_idx, action_idx));
      ++action_idx;
   }
   return out;
}

bool goal_entry_indicates_per_state(nb::handle entry)
{
   return entry.is_none() or is_sequence_like_but_not_str_bytes(entry);
}

bool subgoal_entry_indicates_per_state(nb::handle entry)
{
   if(entry.is_none()) {
      return true;
   }
   if(not is_sequence_like_but_not_str_bytes(entry)) {
      return false;
   }
   if(nb::len(entry) == 0) {
      return false;
   }
   return is_sequence_like_but_not_str_bytes(entry[0]);
}

bool history_entry_indicates_per_state(nb::handle entry)
{
   if(entry.is_none()) {
      return true;
   }
   if(is_history_item(entry)) {
      return false;
   }
   if(not is_sequence_like_but_not_str_bytes(entry)) {
      return false;
   }
   if(nb::len(entry) == 0) {
      return false;
   }
   return is_history_item(entry[0]);
}

nb::list materialize_list(nb::handle value, std::string_view error_message)
{
   if(not is_iterable_but_not_str_bytes(value)) {
      throw nb::type_error(std::string(error_message).c_str());
   }
   return nb::list(value);
}

template < typename Payload, typename ConvertFn >
nb::list expand_plan_to_python_list(
   const SharedOrPerState< Payload >& plan,
   size_t state_count,
   ConvertFn&& convert
)
{
   nb::list out;
   if(plan.per_state.has_value()) {
      for(const auto& entry : *plan.per_state) {
         if(entry.has_value()) {
            out.append(convert(*entry));
         } else {
            out.append(nb::none());
         }
      }
      return out;
   }
   for(size_t idx = 0; idx < state_count; ++idx) {
      (void) idx;
      if(plan.shared.has_value()) {
         out.append(convert(*plan.shared));
      } else {
         out.append(nb::none());
      }
   }
   return out;
}

template < typename Payload, typename ConvertFn >
nb::tuple plan_to_python_tuple(const SharedOrPerState< Payload >& plan, ConvertFn&& convert)
{
   if(plan.per_state.has_value()) {
      nb::list out;
      for(const auto& entry : *plan.per_state) {
         if(entry.has_value()) {
            out.append(convert(*entry));
         } else {
            out.append(nb::none());
         }
      }
      return nb::make_tuple(true, out);
   }

   if(plan.shared.has_value()) {
      return nb::make_tuple(false, convert(*plan.shared));
   }
   return nb::make_tuple(false, nb::none());
}

nb::object goal_payload_to_python(const ParsedGoalPayload& payload)
{
   nb::list out;
   for(const auto& literal : payload) {
      out.append(nb::cast(literal));
   }
   return out;
}

nb::object action_payload_to_python(const ParsedActionPayload& payload)
{
   nb::list out;
   for(const auto& action : payload) {
      out.append(nb::cast(action));
   }
   return out;
}

nb::object subgoal_layers_payload_to_python(const ParsedSubgoalLayersPayload& payload)
{
   nb::list layers;
   for(const auto& layer : payload) {
      nb::list literals;
      for(const auto& literal : layer) {
         literals.append(nb::cast(literal));
      }
      layers.append(literals);
   }
   return layers;
}

nb::object history_payload_to_python(const ParsedHistoryPayload& payload)
{
   nb::list out;
   for(const auto& [dt, literals] : payload) {
      nb::list literals_py;
      for(const auto& literal : literals) {
         literals_py.append(nb::cast(literal));
      }
      out.append(nb::make_tuple(dt, literals_py));
   }
   return out;
}

}  // namespace

ParsedStateBatch parse_states_batch_param(nb::handle states, std::string_view field_name)
{
   ParsedStateBatch out;

   if(is_native_state_object(states)) {
      out.states.push_back(parse_state_entry(states, field_name, 0));
      return out;
   }

   if(is_str_bytes_like(states)) {
      throw nb::type_error(states_iterable_error(field_name).c_str());
   }

   if(not is_iterable_but_not_str_bytes(states) and has_state_adapter(states)) {
      throw nb::type_error(state_adapter_error().c_str());
   }

   const nb::list state_list = materialize_list(states, states_iterable_error(field_name));
   out.states.reserve(nb::len(state_list));
   for(size_t idx = 0; idx < static_cast< size_t >(nb::len(state_list)); ++idx) {
      out.states.push_back(parse_state_entry(state_list[idx], field_name, idx));
   }
   return out;
}

ParsedGoalBatch parse_goals_batch_param(nb::handle goals, size_t state_count)
{
   ParsedGoalBatch out;
   if(goals.is_none()) {
      return out;
   }

   if(not is_sequence_like_but_not_str_bytes(goals)) {
      out.shared = parse_goal_literals_iterable(goals, "goals");
      return out;
   }

   const nb::list outer = nb::list(goals);
   bool per_state_like = false;
   for(nb::handle entry : outer) {
      if(goal_entry_indicates_per_state(entry)) {
         per_state_like = true;
         break;
      }
   }

   if(per_state_like) {
      if(static_cast< size_t >(nb::len(outer)) != state_count) {
         throw nb::value_error("goals length must match states length");
      }
      std::vector< std::optional< ParsedGoalPayload > > values;
      values.reserve(state_count);
      for(size_t idx = 0; idx < state_count; ++idx) {
         nb::handle entry = outer[idx];
         if(entry.is_none()) {
            values.emplace_back(std::nullopt);
         } else {
            values.emplace_back(parse_goal_literals_iterable(entry, "goals", idx));
         }
      }
      out.per_state = std::move(values);
      return out;
   }

   try {
      out.shared = parse_goal_literals_iterable(outer, "goals");
      return out;
   } catch(const nb::builtin_exception& ex) {
      if(ex.type() != nb::exception_type::type_error) {
         throw;
      }
      bool candidate_per_state = static_cast< size_t >(nb::len(outer)) == state_count;
      if(candidate_per_state) {
         for(nb::handle entry : outer) {
            if(not entry.is_none() and not is_iterable_but_not_str_bytes(entry)) {
               candidate_per_state = false;
               break;
            }
         }
      }
      if(not candidate_per_state) {
         throw;
      }
      std::vector< std::optional< ParsedGoalPayload > > values;
      values.reserve(state_count);
      for(size_t idx = 0; idx < state_count; ++idx) {
         nb::handle entry = outer[idx];
         if(entry.is_none()) {
            values.emplace_back(std::nullopt);
         } else {
            values.emplace_back(parse_goal_literals_iterable(entry, "goals", idx));
         }
      }
      out.per_state = std::move(values);
      return out;
   }
}

ParsedActionBatch parse_actions_batch_param(nb::handle actions, size_t state_count)
{
   ParsedActionBatch out;
   if(actions.is_none()) {
      out.shared = ParsedActionPayload{};
      return out;
   }

   if(not is_sequence_like_but_not_str_bytes(actions)) {
      out.shared = parse_flat_actions_iterable(actions, "actions");
      return out;
   }

   const nb::list outer = nb::list(actions);
   bool per_state_like = false;
   for(nb::handle item : outer) {
      if(item.is_none() or is_iterable_but_not_str_bytes(item)) {
         per_state_like = true;
         break;
      }
   }

   if(per_state_like) {
      if(static_cast< size_t >(nb::len(outer)) != state_count) {
         throw nb::value_error("actions length must match states length");
      }
      std::vector< std::optional< ParsedActionPayload > > values;
      values.reserve(state_count);
      for(size_t idx = 0; idx < state_count; ++idx) {
         nb::handle entry = outer[idx];
         if(entry.is_none()) {
            values.emplace_back(std::nullopt);
            continue;
         }
         if(not is_iterable_but_not_str_bytes(entry)) {
            throw nb::type_error(
               "per-state action entries must be iterable action collections or None"
            );
         }
         values.emplace_back(parse_flat_actions_iterable(entry, "actions", idx));
      }
      out.per_state = std::move(values);
      return out;
   }

   out.shared = parse_flat_actions_iterable(outer, "actions");
   return out;
}

ParsedSubgoalLayersBatch
parse_subgoal_layers_batch_param(nb::handle subgoal_layers, size_t state_count)
{
   ParsedSubgoalLayersBatch out;
   if(subgoal_layers.is_none()) {
      return out;
   }

   if(not is_sequence_like_but_not_str_bytes(subgoal_layers)) {
      out.shared = parse_subgoal_layers_payload(subgoal_layers, "subgoal_layers");
      return out;
   }

   const nb::list outer = nb::list(subgoal_layers);
   bool per_state_like = false;
   for(nb::handle entry : outer) {
      if(subgoal_entry_indicates_per_state(entry)) {
         per_state_like = true;
         break;
      }
   }

   if(per_state_like) {
      if(static_cast< size_t >(nb::len(outer)) != state_count) {
         throw nb::value_error("subgoal_layers length must match states length");
      }
      std::vector< std::optional< ParsedSubgoalLayersPayload > > values;
      values.reserve(state_count);
      for(size_t idx = 0; idx < state_count; ++idx) {
         nb::handle entry = outer[idx];
         if(entry.is_none()) {
            values.emplace_back(std::nullopt);
         } else {
            values.emplace_back(parse_subgoal_layers_payload(entry, "subgoal_layers", idx));
         }
      }
      out.per_state = std::move(values);
      return out;
   }

   try {
      out.shared = parse_subgoal_layers_payload(outer, "subgoal_layers");
      return out;
   } catch(const nb::builtin_exception& ex) {
      if(ex.type() != nb::exception_type::type_error) {
         throw;
      }
      bool candidate_per_state = static_cast< size_t >(nb::len(outer)) == state_count;
      if(candidate_per_state) {
         for(nb::handle entry : outer) {
            if(not entry.is_none() and not is_iterable_but_not_str_bytes(entry)) {
               candidate_per_state = false;
               break;
            }
         }
      }
      if(not candidate_per_state) {
         throw;
      }
      std::vector< std::optional< ParsedSubgoalLayersPayload > > values;
      values.reserve(state_count);
      for(size_t idx = 0; idx < state_count; ++idx) {
         nb::handle entry = outer[idx];
         if(entry.is_none()) {
            values.emplace_back(std::nullopt);
         } else {
            values.emplace_back(parse_subgoal_layers_payload(entry, "subgoal_layers", idx));
         }
      }
      out.per_state = std::move(values);
      return out;
   }
}

ParsedHistoryBatch
parse_history_subgoals_batch_param(nb::handle history_subgoals, size_t state_count)
{
   ParsedHistoryBatch out;
   if(history_subgoals.is_none()) {
      return out;
   }

   if(not is_sequence_like_but_not_str_bytes(history_subgoals)) {
      out.shared = parse_history_payload(history_subgoals, "history_subgoals");
      return out;
   }

   const nb::list outer = nb::list(history_subgoals);
   bool per_state_like = false;
   for(nb::handle entry : outer) {
      if(history_entry_indicates_per_state(entry)) {
         per_state_like = true;
         break;
      }
   }

   if(per_state_like) {
      if(static_cast< size_t >(nb::len(outer)) != state_count) {
         throw nb::value_error("history_subgoals length must match states length");
      }
      std::vector< std::optional< ParsedHistoryPayload > > values;
      values.reserve(state_count);
      for(size_t idx = 0; idx < state_count; ++idx) {
         nb::handle entry = outer[idx];
         if(entry.is_none()) {
            values.emplace_back(std::nullopt);
         } else {
            values.emplace_back(parse_history_payload(entry, "history_subgoals", idx));
         }
      }
      out.per_state = std::move(values);
      return out;
   }

   try {
      out.shared = parse_history_payload(outer, "history_subgoals");
      return out;
   } catch(const nb::builtin_exception& ex) {
      if(ex.type() != nb::exception_type::type_error) {
         throw;
      }
      bool candidate_per_state = static_cast< size_t >(nb::len(outer)) == state_count;
      if(candidate_per_state) {
         for(nb::handle entry : outer) {
            if(not entry.is_none() and not is_iterable_but_not_str_bytes(entry)) {
               candidate_per_state = false;
               break;
            }
         }
      }
      if(not candidate_per_state) {
         throw;
      }
      std::vector< std::optional< ParsedHistoryPayload > > values;
      values.reserve(state_count);
      for(size_t idx = 0; idx < state_count; ++idx) {
         nb::handle entry = outer[idx];
         if(entry.is_none()) {
            values.emplace_back(std::nullopt);
         } else {
            values.emplace_back(parse_history_payload(entry, "history_subgoals", idx));
         }
      }
      out.per_state = std::move(values);
      return out;
   }
}

std::vector< ParsedStateEntry >
parse_successors_batch_param(nb::handle successors, size_t state_count)
{
   if(successors.is_none()) {
      throw nb::value_error("successors must be provided for transition batch encoding");
   }
   auto parsed = parse_states_batch_param(successors, "successors");
   if(parsed.states.size() != state_count) {
      throw nb::value_error("successors length must match states length");
   }
   return std::move(parsed.states);
}

ParsedDagBatch parse_dags_batch_param(nb::handle dags, size_t state_count)
{
   ParsedDagBatch out;
   if(dags.is_none()) {
      return out;
   }

   if(nb::isinstance< TransitionDAG >(dags)) {
      out.shared = nb::cast< TransitionDAG >(dags);
      return out;
   }

   if(not is_iterable_but_not_str_bytes(dags)) {
      throw nb::type_error("dags must be a TransitionDAG, iterable of dags, or None");
   }

   const nb::list outer = nb::list(dags);
   if(static_cast< size_t >(nb::len(outer)) != state_count) {
      throw nb::value_error("dags length must match states length");
   }

   std::vector< std::optional< TransitionDAG > > values;
   values.reserve(state_count);
   for(size_t idx = 0; idx < state_count; ++idx) {
      nb::handle entry = outer[idx];
      if(entry.is_none()) {
         values.emplace_back(std::nullopt);
         continue;
      }
      if(not nb::isinstance< TransitionDAG >(entry)) {
         throw nb::type_error(
            fmt::format("dags entry at index {} has invalid type: {}", idx, py_type_repr(entry))
               .c_str()
         );
      }
      values.emplace_back(nb::cast< TransitionDAG >(entry));
   }
   out.per_state = std::move(values);
   return out;
}

GoalInputs compose_goal_inputs(
   const ParsedGoalPayload& goals,
   const ParsedSubgoalLayersPayload* subgoal_layers
)
{
   GoalInputs inputs;
   inputs.extend(goals, 0);
   if(subgoal_layers != nullptr) {
      size_t depth = 1;
      for(const auto& layer : *subgoal_layers) {
         inputs.extend(layer, depth);
         ++depth;
      }
   }
   return inputs;
}

GoalInputs default_goal_inputs_for_batch_state(const ParsedStateEntry& state_entry)
{
   if(not state_entry.is_wrapper_state) {
      throw nb::value_error("goals must be provided when passing an advanced state");
   }

   GoalInputs inputs;
   const auto& problem = state_entry.state.get_problem();
   for(const auto& goal : problem.get_goal_literals< mimir::formalism::StaticTag >()) {
      inputs.append(goal, 0);
   }
   for(const auto& goal : problem.get_goal_literals< mimir::formalism::FluentTag >()) {
      inputs.append(goal, 0);
   }
   for(const auto& goal : problem.get_goal_literals< mimir::formalism::DerivedTag >()) {
      inputs.append(goal, 0);
   }
   return inputs;
}

void reject_unsupported_batch_field(
   std::string_view encoder_name,
   std::string_view field_name,
   nb::handle value
)
{
   if(not value.is_none()) {
      throw nb::type_error(
         fmt::format("{} does not accept '{}' in encode_batch", encoder_name, field_name).c_str()
      );
   }
}

BatchBuilder::BatchEncoding hgraph_encode_batch(
   HGraphEncoderEngine& encoder,
   nb::handle states,
   nb::handle goals,
   nb::handle actions,
   nb::handle subgoal_layers,
   nb::handle history_subgoals,
   std::optional< int > history_max_steps
)
{
   auto parsed_states = parse_states_batch_param(states, "states");
   const size_t state_count = parsed_states.states.size();
   const auto parsed_goals = parse_goals_batch_param(goals, state_count);
   const auto parsed_actions = parse_actions_batch_param(actions, state_count);
   const auto parsed_subgoal_layers = parse_subgoal_layers_batch_param(subgoal_layers, state_count);
   const auto parsed_history = parse_history_subgoals_batch_param(history_subgoals, state_count);

   const ParsedActionPayload empty_actions{};
   BatchBuilder builder;
   builder.set_graph_kind("hetero");

   for(size_t idx = 0; idx < state_count; ++idx) {
      const auto& state_entry = parsed_states.states[idx];
      const auto& goals_entry = parsed_goals.at(idx);
      const auto& actions_entry = parsed_actions.at(idx);
      const auto& subgoal_layers_entry = parsed_subgoal_layers.at(idx);
      const auto& history_entry = parsed_history.at(idx);

      const ParsedActionPayload& actions_payload = actions_entry.has_value() ? *actions_entry
                                                                             : empty_actions;
      const bool has_aux_payload = subgoal_layers_entry.has_value() or not actions_payload.empty()
                                   or history_entry.has_value();

      if(not goals_entry.has_value() and not has_aux_payload) {
         encoder.encode(state_entry.state, builder);
         builder.next_graph();
         continue;
      }

      GoalInputs inputs;
      if(goals_entry.has_value()) {
         const auto* layers_ptr = subgoal_layers_entry.has_value() ? &(*subgoal_layers_entry)
                                                                   : nullptr;
         inputs = compose_goal_inputs(*goals_entry, layers_ptr);
      } else {
         inputs = default_goal_inputs_for_batch_state(state_entry);
         if(subgoal_layers_entry.has_value()) {
            size_t level = 1;
            for(const auto& layer : *subgoal_layers_entry) {
               inputs.extend(layer, level);
               ++level;
            }
         }
      }

      if(history_entry.has_value()) {
         encoder.encode(
            state_entry.state, inputs, actions_payload, *history_entry, history_max_steps, builder
         );
      } else {
         encoder.encode(state_entry.state, inputs, actions_payload, builder);
      }
      builder.next_graph();
   }

   return builder.build();
}

BatchBuilder::BatchEncoding color_encode_batch(
   ColorEncoderEngine& encoder,
   std::string_view encoder_name,
   nb::handle states,
   nb::handle goals,
   nb::handle actions,
   nb::handle subgoal_layers
)
{
   reject_unsupported_batch_field(encoder_name, "actions", actions);

   auto parsed_states = parse_states_batch_param(states, "states");
   const size_t state_count = parsed_states.states.size();
   const auto parsed_goals = parse_goals_batch_param(goals, state_count);
   const auto parsed_subgoal_layers = parse_subgoal_layers_batch_param(subgoal_layers, state_count);

   BatchBuilder builder;
   builder.set_graph_kind("homo");

   for(size_t idx = 0; idx < state_count; ++idx) {
      const auto& state_entry = parsed_states.states[idx];
      const auto& goals_entry = parsed_goals.at(idx);
      const auto& subgoal_layers_entry = parsed_subgoal_layers.at(idx);

      if(not goals_entry.has_value() and not subgoal_layers_entry.has_value()) {
         encoder.encode(state_entry.state, builder);
         builder.next_graph();
         continue;
      }

      GoalInputs inputs;
      if(goals_entry.has_value()) {
         const auto* layers_ptr = subgoal_layers_entry.has_value() ? &(*subgoal_layers_entry)
                                                                   : nullptr;
         inputs = compose_goal_inputs(*goals_entry, layers_ptr);
      } else {
         inputs = default_goal_inputs_for_batch_state(state_entry);
         if(subgoal_layers_entry.has_value()) {
            size_t level = 1;
            for(const auto& layer : *subgoal_layers_entry) {
               inputs.extend(layer, level);
               ++level;
            }
         }
      }

      encoder.encode(state_entry.state, inputs, builder);
      builder.next_graph();
   }

   return builder.build();
}

BatchBuilder::BatchEncoding successor_encode_batch(
   SuccessorHGraphEncoderEngine& encoder,
   std::string_view encoder_name,
   nb::handle states,
   nb::handle successors,
   nb::handle goals,
   nb::handle actions,
   nb::handle subgoal_layers,
   nb::handle history_subgoals,
   std::optional< int > history_max_steps
)
{
   reject_unsupported_batch_field(encoder_name, "actions", actions);
   reject_unsupported_batch_field(encoder_name, "history_subgoals", history_subgoals);
   if(history_max_steps.has_value()) {
      reject_unsupported_batch_field(
         encoder_name, "history_max_steps", nb::cast(*history_max_steps)
      );
   }

   auto parsed_states = parse_states_batch_param(states, "states");
   const size_t state_count = parsed_states.states.size();
   const auto parsed_successors = parse_successors_batch_param(successors, state_count);
   const auto parsed_goals = parse_goals_batch_param(goals, state_count);
   const auto parsed_subgoal_layers = parse_subgoal_layers_batch_param(subgoal_layers, state_count);

   BatchBuilder builder;
   builder.set_graph_kind("hetero");

   for(size_t idx = 0; idx < state_count; ++idx) {
      const auto& state_entry = parsed_states.states[idx];
      const auto& successor_entry = parsed_successors[idx];
      const auto& goals_entry = parsed_goals.at(idx);
      const auto& subgoal_layers_entry = parsed_subgoal_layers.at(idx);

      GoalInputs inputs;
      if(goals_entry.has_value()) {
         const auto* layers_ptr = subgoal_layers_entry.has_value() ? &(*subgoal_layers_entry)
                                                                   : nullptr;
         inputs = compose_goal_inputs(*goals_entry, layers_ptr);
      } else {
         inputs = default_goal_inputs_for_batch_state(state_entry);
         if(subgoal_layers_entry.has_value()) {
            size_t level = 1;
            for(const auto& layer : *subgoal_layers_entry) {
               inputs.extend(layer, level);
               ++level;
            }
         }
      }

      encoder.encode(state_entry.state, successor_entry.state, inputs, builder);
      builder.next_graph();
   }

   return builder.build();
}

BatchBuilder::BatchEncoding horizon_encode_batch(
   HorizonHGraphEncoderEngine& encoder,
   std::string_view encoder_name,
   nb::handle roots,
   nb::handle dags,
   nb::handle goals,
   nb::handle actions,
   nb::handle subgoal_layers,
   nb::handle history_subgoals,
   std::optional< int > history_max_steps
)
{
   reject_unsupported_batch_field(encoder_name, "actions", actions);
   reject_unsupported_batch_field(encoder_name, "history_subgoals", history_subgoals);
   if(history_max_steps.has_value()) {
      reject_unsupported_batch_field(
         encoder_name, "history_max_steps", nb::cast(*history_max_steps)
      );
   }

   auto parsed_roots = parse_states_batch_param(roots, "states");
   const size_t state_count = parsed_roots.states.size();
   const auto parsed_dags = parse_dags_batch_param(dags, state_count);
   const auto parsed_goals = parse_goals_batch_param(goals, state_count);
   const auto parsed_subgoal_layers = parse_subgoal_layers_batch_param(subgoal_layers, state_count);

   BatchBuilder builder;
   builder.set_graph_kind("hetero");

   for(size_t idx = 0; idx < state_count; ++idx) {
      const auto& root_entry = parsed_roots.states[idx];
      const auto& dag_entry = parsed_dags.at(idx);
      const auto& goals_entry = parsed_goals.at(idx);
      const auto& subgoal_layers_entry = parsed_subgoal_layers.at(idx);

      const TransitionDAG default_dag(root_entry.state);
      const TransitionDAG& dag_ref = dag_entry.has_value() ? *dag_entry : default_dag;

      GoalInputs inputs;
      if(goals_entry.has_value()) {
         const auto* layers_ptr = subgoal_layers_entry.has_value() ? &(*subgoal_layers_entry)
                                                                   : nullptr;
         inputs = compose_goal_inputs(*goals_entry, layers_ptr);
      } else {
         inputs = default_goal_inputs_for_batch_state(root_entry);
         if(subgoal_layers_entry.has_value()) {
            size_t level = 1;
            for(const auto& layer : *subgoal_layers_entry) {
               inputs.extend(layer, level);
               ++level;
            }
         }
      }

      encoder.encode(root_entry.state, dag_ref, inputs, builder);
      builder.next_graph();
   }

   return builder.build();
}

nb::list parse_states_batch_python(nb::handle states)
{
   const auto parsed = parse_states_batch_param(states, "states");
   nb::list out;
   for(const auto& state : parsed.states) {
      out.append(state.source);
   }
   return out;
}

nb::tuple parse_goals_batch_param_python(nb::handle goals, size_t state_count)
{
   const auto parsed = parse_goals_batch_param(goals, state_count);
   return plan_to_python_tuple(parsed, goal_payload_to_python);
}

nb::tuple parse_actions_batch_param_python(nb::handle actions, size_t state_count)
{
   const auto parsed = parse_actions_batch_param(actions, state_count);
   return plan_to_python_tuple(parsed, action_payload_to_python);
}

nb::tuple parse_subgoal_layers_batch_param_python(nb::handle subgoal_layers, size_t state_count)
{
   const auto parsed = parse_subgoal_layers_batch_param(subgoal_layers, state_count);
   return plan_to_python_tuple(parsed, subgoal_layers_payload_to_python);
}

nb::tuple parse_history_subgoals_batch_param_python(nb::handle history_subgoals, size_t state_count)
{
   const auto parsed = parse_history_subgoals_batch_param(history_subgoals, state_count);
   return plan_to_python_tuple(parsed, history_payload_to_python);
}

nb::list parse_successors_batch_param_python(nb::handle successors, size_t state_count)
{
   const auto parsed = parse_successors_batch_param(successors, state_count);
   nb::list out;
   for(const auto& successor : parsed) {
      out.append(successor.source);
   }
   return out;
}

nb::list parse_dags_batch_param_python(nb::handle dags, size_t state_count)
{
   const auto parsed = parse_dags_batch_param(dags, state_count);
   nb::list out;
   if(parsed.per_state.has_value()) {
      for(const auto& entry : *parsed.per_state) {
         if(entry.has_value()) {
            out.append(nb::cast(*entry));
         } else {
            out.append(nb::none());
         }
      }
      return out;
   }

   for(size_t idx = 0; idx < state_count; ++idx) {
      (void) idx;
      if(parsed.shared.has_value()) {
         out.append(nb::cast(*parsed.shared));
      } else {
         out.append(nb::none());
      }
   }
   return out;
}

nb::tuple parse_ilg_batch_inputs_python(
   nb::handle states,
   nb::handle goals,
   nb::handle actions,
   nb::handle subgoal_layers
)
{
   const auto parsed_states = parse_states_batch_param(states, "states");
   const size_t state_count = parsed_states.states.size();
   const auto parsed_goals = parse_goals_batch_param(goals, state_count);
   const auto parsed_actions = parse_actions_batch_param(actions, state_count);
   const auto parsed_subgoal_layers = parse_subgoal_layers_batch_param(subgoal_layers, state_count);

   nb::list states_out;
   for(const auto& state : parsed_states.states) {
      states_out.append(state.source);
   }

   nb::list goals_out = expand_plan_to_python_list(
      parsed_goals, state_count, goal_payload_to_python
   );
   nb::list actions_out = expand_plan_to_python_list(
      parsed_actions, state_count, action_payload_to_python
   );
   nb::list subgoal_layers_out = expand_plan_to_python_list(
      parsed_subgoal_layers, state_count, subgoal_layers_payload_to_python
   );

   return nb::make_tuple(states_out, goals_out, actions_out, subgoal_layers_out);
}

}  // namespace mifrost::batch_input
