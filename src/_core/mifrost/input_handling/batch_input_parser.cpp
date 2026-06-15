#include "mifrost/input_handling/batch_input_parser.hpp"

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

nb::handle batch_param_type()
{
   static nb::object* typ = [] {
      return new nb::object(encoder_types_module().attr("BatchParam"));
   }();
   return *typ;
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

inline void clear_python_error_if_possible()
{
   if(_PyThreadState_UncheckedGet() != nullptr) {
      PyErr_Clear();
   }
}

bool is_iterable_but_not_str_bytes(nb::handle value)
{
   if(is_str_bytes_like(value)) {
      return false;
   }
   PyObject* iter = PyObject_GetIter(value.ptr());
   if(iter == nullptr) {
      clear_python_error_if_possible();
      return false;
   }
   Py_DECREF(iter);
   return true;
}

std::string py_type_repr(nb::handle value)
{
   nb::object typ = py::builtins_type_type()(value);
   return nb::cast< std::string >(nb::repr(typ));
}

template < typename T >
bool can_cast_noerror(nb::handle value)
{
   try {
      (void) nb::cast< T >(value);
      return true;
   } catch(...) {
      clear_python_error_if_possible();
      return false;
   }
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

bool is_batch_param(nb::handle value)
{
   return value.ptr() != nullptr
          and Py_TYPE(value.ptr()) == reinterpret_cast< PyTypeObject* >(batch_param_type().ptr());
}

std::string batch_param_kind(nb::handle value)
{
   return nb::cast< std::string >(value.attr("kind"));
}

nb::handle batch_param_value(nb::handle value)
{
   return value.attr("value");
}

bool is_native_state_object(nb::handle value)
{
   // Sequences (list, tuple, …) are never native State objects.  Probing them
   // via the mimir caster can SIGSEGV rather than raising a Python exception,
   // so skip the cast entirely for any sequence type.
   if(is_sequence_like_but_not_str_bytes(value)) {
      return false;
   }
   return can_cast_noerror< mimir::search::State >(value);
}

bool is_native_action_object(nb::handle value)
{
   if(is_sequence_like_but_not_str_bytes(value)) {
      return false;
   }
   return can_cast_noerror< mimir::formalism::GroundAction >(value);
}

std::string state_adapter_error()
{
   return "Batch parsing does not support state adapters; pass native advanced states";
}

std::string literal_adapter_error()
{
   return "Batch parsing does not support literal adapters; pass native goal literals";
}

std::string action_adapter_error()
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

parsed::StateEntry parse_state_entry(nb::handle value, std::string_view field_name, size_t index)
{
   try {
      return parsed::StateEntry{
         .source = value.ptr(),
         .state = nb::cast< mimir::search::State >(value),
      };
   } catch(...) {
      clear_python_error_if_possible();
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
      return nb::cast< LiteralVariant >(literal);
   } catch(...) {
      clear_python_error_if_possible();
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
      return nb::cast< mimir::formalism::GroundAction >(action);
   } catch(...) {
      clear_python_error_if_possible();
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

parsed::GoalPayload parse_goal_literals_iterable(
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

   parsed::GoalPayload out;
   const nb::list literals = nb::list(value);
   size_t literal_idx = 0;
   for(nb::handle literal : literals) {
      out.push_back(cast_goal_literal_or_raise(literal, field, entry_idx, literal_idx));
      ++literal_idx;
   }
   return out;
}

parsed::SubgoalLayersPayload parse_subgoal_layers_payload(
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

   parsed::SubgoalLayersPayload out;
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
   const auto tup = nb::borrow< nb::tuple >(value);
   return nb::len(tup) == 2 and is_integral_py(tup[0]);
}

parsed::HistoryPayload parse_history_payload(
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

   parsed::HistoryPayload out;
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
      const auto tup = nb::borrow< nb::tuple >(item);
      const int dt = nb::cast< int >(tup[0]);
      const parsed::GoalPayload literals = parse_goal_literals_iterable(tup[1], field, entry_idx);
      out.emplace_back(dt, literals);
      ++history_idx;
   }

   return out;
}

parsed::ActionPayload parse_flat_actions_iterable(
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

   parsed::ActionPayload out;
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

bool goal_entry_is_per_state_candidate(nb::handle entry)
{
   return entry.is_none() or is_sequence_like_but_not_str_bytes(entry);
}

bool action_entry_indicates_per_state(nb::handle entry)
{
   if(entry.is_none()) {
      return true;
   }
   if(not is_iterable_but_not_str_bytes(entry)) {
      return false;
   }
   return not is_native_action_object(entry);
}

bool action_entry_is_per_state_candidate(nb::handle entry)
{
   return entry.is_none()
          or (is_iterable_but_not_str_bytes(entry) and not is_native_action_object(entry));
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

bool subgoal_entry_is_per_state_candidate(nb::handle entry)
{
   return entry.is_none() or is_sequence_like_but_not_str_bytes(entry);
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

bool history_entry_is_per_state_candidate(nb::handle entry)
{
   return entry.is_none()
          or (is_sequence_like_but_not_str_bytes(entry) and not is_history_item(entry));
}

template <
   typename Payload,
   typename ParseEntryFn,
   typename PerStateIndicatorFn,
   typename PerStateCandidateFn >
BatchParam< Payload > parse_shared_or_per_state(
   nb::handle value,
   size_t state_count,
   std::string_view length_mismatch_message,
   ParseEntryFn&& parse_entry,
   PerStateIndicatorFn&& entry_indicates_per_state,
   PerStateCandidateFn&& entry_is_per_state_candidate
)
{
   if(value.ptr() == nullptr) {
      return BatchParam< Payload >::none();
   }

   if(is_batch_param(value)) {
      const std::string kind = batch_param_kind(value);
      nb::handle payload = batch_param_value(value);

      if(kind == "none") {
         return BatchParam< Payload >::none();
      }
      if(kind == "shared") {
         return BatchParam< Payload >::shared(parse_entry(payload, std::nullopt));
      }
      if(kind == "separate") {
         if(not is_sequence_like_but_not_str_bytes(payload)) {
            throw nb::type_error("BatchParam(separate) value must be a sequence");
         }
         const nb::list outer = nb::list(payload);
         if(nb::len(outer) != state_count) {
            throw nb::value_error(std::string(length_mismatch_message).c_str());
         }
         std::vector< std::optional< Payload > > values;
         values.reserve(state_count);
         for(size_t idx = 0; idx < state_count; ++idx) {
            nb::handle entry = outer[idx];
            if(entry.is_none()) {
               values.emplace_back(std::nullopt);
            } else {
               values.emplace_back(parse_entry(entry, idx));
            }
         }
         return BatchParam< Payload >::per_state(std::move(values));
      }
      throw nb::value_error("BatchParam.kind must be 'shared', 'separate', or 'none'");
   }

   if(value.is_none()) {
      return BatchParam< Payload >::none();
   }

   if(not is_sequence_like_but_not_str_bytes(value)) {
      return BatchParam< Payload >::shared(parse_entry(value, std::nullopt));
   }

   const nb::list outer = nb::list(value);
   bool per_state_like = false;
   for(nb::handle entry : outer) {
      if(entry_indicates_per_state(entry)) {
         per_state_like = true;
         break;
      }
   }

   auto parse_per_state = [&]() {
      if(nb::len(outer) != state_count) {
         throw nb::value_error(std::string(length_mismatch_message).c_str());
      }
      std::vector< std::optional< Payload > > values;
      values.reserve(state_count);
      for(size_t idx = 0; idx < state_count; ++idx) {
         nb::handle entry = outer[idx];
         if(entry.is_none()) {
            values.emplace_back(std::nullopt);
         } else {
            values.emplace_back(parse_entry(entry, idx));
         }
      }
      return BatchParam< Payload >::per_state(std::move(values));
   };

   if(per_state_like) {
      return parse_per_state();
   }

   try {
      return BatchParam< Payload >::shared(parse_entry(outer, std::nullopt));
   } catch(const nb::builtin_exception& ex) {
      if(ex.type() != nb::exception_type::type_error) {
         throw;
      }
      bool candidate_per_state = nb::len(outer) == state_count;
      if(candidate_per_state) {
         for(nb::handle entry : outer) {
            if(not entry_is_per_state_candidate(entry)) {
               candidate_per_state = false;
               break;
            }
         }
      }
      if(not candidate_per_state) {
         throw;
      }
      return parse_per_state();
   }
}

template < typename Payload, typename ConvertFn >
nb::list expand_plan_to_python_list(
   const BatchParam< Payload >& plan,
   size_t state_count,
   ConvertFn&& convert
)
{
   nb::list out;
   if(plan.is_per_state()) {
      for(const auto& entry : plan.per_state()) {
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
      if(plan.is_shared()) {
         out.append(convert(*plan.shared()));
      } else {
         out.append(nb::none());
      }
   }
   return out;
}

template < typename Payload, typename ConvertFn >
nb::tuple plan_to_python_tuple(const BatchParam< Payload >& plan, ConvertFn&& convert)
{
   if(plan.is_per_state()) {
      nb::list out;
      for(const auto& entry : plan.per_state()) {
         if(entry.has_value()) {
            out.append(convert(*entry));
         } else {
            out.append(nb::none());
         }
      }
      return nb::make_tuple(true, out);
   }

   if(plan.is_shared()) {
      return nb::make_tuple(false, convert(*plan.shared()));
   }
   return nb::make_tuple(false, nb::none());
}

nb::object goal_payload_to_python(const parsed::GoalPayload& payload)
{
   nb::list out;
   for(const auto& literal : payload) {
      out.append(nb::cast(literal));
   }
   return out;
}

nb::object action_payload_to_python(const parsed::ActionPayload& payload)
{
   nb::list out;
   for(const auto& action : payload) {
      out.append(nb::cast(action));
   }
   return out;
}

nb::object subgoal_layers_payload_to_python(const parsed::SubgoalLayersPayload& payload)
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

nb::object history_payload_to_python(const parsed::HistoryPayload& payload)
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

parsed::StateBatch parse_states_batch_param(nb::handle states, std::string_view field_name)
{
   parsed::StateBatch out;

   // Fast path: empty sequence → empty batch, no type-probe needed.
   if(is_sequence_like_but_not_str_bytes(states) and nb::len(states) == 0) {
      return out;
   }

   if(is_native_state_object(states)) {
      out.states.push_back(parse_state_entry(states, field_name, 0));
      return out;
   }

   if(is_str_bytes_like(states)) {
      throw nb::type_error(states_iterable_error(field_name).c_str());
   }

   if(not is_iterable_but_not_str_bytes(states)) {
      if(has_state_adapter(states)) {
         throw nb::type_error(state_adapter_error().c_str());
      }
      throw nb::type_error(states_iterable_error(field_name).c_str());
   }

   try {
      if(nb::isinstance< nb::sequence >(states)) {
         const auto seq = nb::borrow< nb::sequence >(states);
         out.states.reserve(static_cast< size_t >(nb::len(seq)));
      }
   } catch(...) {
      clear_python_error_if_possible();
   }

   size_t idx = 0;
   try {
      for(nb::handle entry : nb::borrow< nb::iterable >(states)) {
         out.states.push_back(parse_state_entry(entry, field_name, idx));
         ++idx;
      }
   } catch(const nb::builtin_exception& ex) {
      if(ex.type() != nb::exception_type::type_error) {
         throw;
      }
      throw;
   } catch(...) {
      clear_python_error_if_possible();
      throw nb::type_error(states_iterable_error(field_name).c_str());
   }
   return out;
}

parsed::GoalBatch parse_goals_batch_param(nb::handle goals, size_t state_count)
{
   auto parse_entry = [](nb::handle entry, std::optional< size_t > idx) {
      return parse_goal_literals_iterable(entry, "goals", idx);
   };
   return parse_shared_or_per_state< parsed::GoalPayload >(
      goals,
      state_count,
      "goals length must match states length",
      parse_entry,
      goal_entry_indicates_per_state,
      goal_entry_is_per_state_candidate
   );
}

parsed::ActionBatch parse_actions_batch_param(nb::handle actions, size_t state_count)
{
   auto parse_entry = [](nb::handle entry, std::optional< size_t > idx) {
      return parse_flat_actions_iterable(entry, "actions", idx);
   };
   return parse_shared_or_per_state< parsed::ActionPayload >(
      actions,
      state_count,
      "actions length must match states length",
      parse_entry,
      action_entry_indicates_per_state,
      action_entry_is_per_state_candidate
   );
}

parsed::SubgoalLayersBatch
parse_subgoal_layers_batch_param(nb::handle subgoal_layers, size_t state_count)
{
   auto parse_entry = [](nb::handle entry, std::optional< size_t > idx) {
      return parse_subgoal_layers_payload(entry, "subgoal_layers", idx);
   };
   return parse_shared_or_per_state< parsed::SubgoalLayersPayload >(
      subgoal_layers,
      state_count,
      "subgoal_layers length must match states length",
      parse_entry,
      subgoal_entry_indicates_per_state,
      subgoal_entry_is_per_state_candidate
   );
}

parsed::HistoryBatch
parse_history_subgoals_batch_param(nb::handle history_subgoals, size_t state_count)
{
   auto parse_entry = [](nb::handle entry, std::optional< size_t > idx) {
      return parse_history_payload(entry, "history_subgoals", idx);
   };
   return parse_shared_or_per_state< parsed::HistoryPayload >(
      history_subgoals,
      state_count,
      "history_subgoals length must match states length",
      parse_entry,
      history_entry_indicates_per_state,
      history_entry_is_per_state_candidate
   );
}

parsed::SuccessorBatch parse_successors_batch_param(nb::handle successors, size_t state_count)
{
   if(successors.is_none()) {
      throw nb::value_error("successors must be provided for transition batch encoding");
   }

   auto parse_shared = [](nb::handle value) {
      if(value.is_none()) {
         throw nb::value_error("successors must be provided for transition batch encoding");
      }
      return parse_state_entry(value, "successors", 0);
   };

   auto parse_separate = [&](nb::handle value) {
      if(not is_sequence_like_but_not_str_bytes(value)) {
         throw nb::type_error("BatchParam(separate) value must be a sequence");
      }
      auto parsed = parse_states_batch_param(value, "successors");
      if(parsed.states.size() != state_count) {
         throw nb::value_error("successors length must match states length");
      }

      std::vector< std::optional< parsed::StateEntry > > values;
      values.reserve(state_count);
      for(auto& entry : parsed.states) {
         values.emplace_back(std::move(entry));
      }
      return parsed::SuccessorBatch::per_state(std::move(values));
   };

   if(is_batch_param(successors)) {
      const std::string kind = batch_param_kind(successors);
      nb::handle payload = batch_param_value(successors);

      if(kind == "none") {
         throw nb::value_error("successors must be provided for transition batch encoding");
      }
      if(kind == "shared") {
         return parsed::SuccessorBatch::shared(parse_shared(payload));
      }
      if(kind == "separate") {
         return parse_separate(payload);
      }
      throw nb::value_error("BatchParam.kind must be 'shared', 'separate', or 'none'");
   }

   if(is_native_state_object(successors)) {
      return parsed::SuccessorBatch::shared(parse_shared(successors));
   }

   auto parsed = parse_states_batch_param(successors, "successors");
   if(parsed.states.size() != state_count) {
      throw nb::value_error("successors length must match states length");
   }

   std::vector< std::optional< parsed::StateEntry > > values;
   values.reserve(state_count);
   for(auto& entry : parsed.states) {
      values.emplace_back(std::move(entry));
   }
   return parsed::SuccessorBatch::per_state(std::move(values));
}

parsed::DagBatch parse_dags_batch_param(nb::handle dags, size_t state_count)
{
   if(is_batch_param(dags)) {
      const std::string kind = batch_param_kind(dags);
      nb::handle payload = batch_param_value(dags);

      if(kind == "none") {
         return parsed::DagBatch::none();
      }
      if(kind == "shared") {
         if(not nb::isinstance< TransitionDAG >(payload)) {
            throw nb::type_error("BatchParam(shared) value for dags must be a TransitionDAG");
         }
         return parsed::DagBatch::shared(nb::cast< TransitionDAG >(payload));
      }
      if(kind == "separate") {
         if(not is_sequence_like_but_not_str_bytes(payload)) {
            throw nb::type_error("BatchParam(separate) value must be a sequence");
         }

         const nb::list outer = nb::list(payload);
         if(nb::len(outer) != state_count) {
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
                  fmt::format(
                     "dags entry at index {} has invalid type: {}", idx, py_type_repr(entry)
                  )
                     .c_str()
               );
            }
            values.emplace_back(nb::cast< TransitionDAG >(entry));
         }
         return parsed::DagBatch::per_state(std::move(values));
      }
      throw nb::value_error("BatchParam.kind must be 'shared', 'separate', or 'none'");
   }

   if(dags.is_none()) {
      return parsed::DagBatch::none();
   }

   if(nb::isinstance< TransitionDAG >(dags)) {
      return parsed::DagBatch::shared(nb::cast< TransitionDAG >(dags));
   }

   if(not is_iterable_but_not_str_bytes(dags)) {
      throw nb::type_error("dags must be a TransitionDAG, iterable of dags, or None");
   }

   const nb::list outer = nb::list(dags);
   if(nb::len(outer) != state_count) {
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
   return parsed::DagBatch::per_state(std::move(values));
}

nb::list parse_states_batch_python(nb::handle states)
{
   // Hard fast-path for the common empty-list case. This avoids any adapter or
   // caster probing and guarantees `_parse_states_batch([])` returns `[]`.
   if(PyList_Check(states.ptr()) != 0 && PyList_GET_SIZE(states.ptr()) == 0) {
      return nb::list();
   }

   const auto parsed = parse_states_batch_param(states, "states");
   nb::list out;
   for(const auto& state : parsed.states) {
      out.append(nb::borrow< nb::object >(state.source));
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
   if(parsed.is_per_state()) {
      for(const auto& entry : parsed.per_state()) {
         if(entry.has_value()) {
            out.append(nb::borrow< nb::object >(entry->source));
         } else {
            out.append(nb::none());
         }
      }
      return out;
   }

   for(size_t idx = 0; idx < state_count; ++idx) {
      (void) idx;
      if(parsed.is_shared()) {
         out.append(nb::borrow< nb::object >(parsed.shared()->source));
      } else {
         out.append(nb::none());
      }
   }
   return out;
}

nb::list parse_dags_batch_param_python(nb::handle dags, size_t state_count)
{
   const auto parsed = parse_dags_batch_param(dags, state_count);
   nb::list out;
   if(parsed.is_per_state()) {
      for(const auto& entry : parsed.per_state()) {
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
      if(parsed.is_shared()) {
         out.append(nb::cast(*parsed.shared()));
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
      states_out.append(nb::borrow< nb::object >(state.source));
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
