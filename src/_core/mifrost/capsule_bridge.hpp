#pragma once

#include <Python.h>

#include <memory>
#include <stdexcept>
#include <utility>

namespace mifrost::capsule_bridge {

inline constexpr char config_name[] = "mifrost.FlatRelationEncoderConfig.v1";
inline constexpr char input_name[] = "mifrost.SemanticFlatRelationInput.v1";
inline constexpr char inputs_name[] = "mifrost.SemanticFlatRelationInputs.v1";
inline constexpr char engine_name[] = "mifrost.SemanticFlatRelationEncoderEngine.v1";
inline constexpr char color_config_name[] = "mifrost.SemanticColorEncoderConfig.v1";
inline constexpr char color_engine_name[] = "mifrost.SemanticColorEncoderEngine.v1";
inline constexpr char derived_config_name[] = "mifrost.SemanticDerivedGraphEncoderConfig.v1";
inline constexpr char derived_engine_name[] = "mifrost.SemanticDerivedGraphEncoderEngine.v1";
inline constexpr char hgraph_config_name[] = "mifrost.SemanticHGraphEncoderConfig.v1";
inline constexpr char hgraph_engine_name[] = "mifrost.SemanticHGraphEncoderEngine.v1";
inline constexpr char
   successor_hgraph_config_name[] = "mifrost.SemanticSuccessorHGraphEncoderConfig.v1";
inline constexpr char
   successor_hgraph_engine_name[] = "mifrost.SemanticSuccessorHGraphEncoderEngine.v1";
inline constexpr char
   horizon_hgraph_config_name[] = "mifrost.SemanticHorizonHGraphEncoderConfig.v1";
inline constexpr char
   horizon_hgraph_engine_name[] = "mifrost.SemanticHorizonHGraphEncoderEngine.v1";
inline constexpr char flat_horizon_config_name[] = "mifrost.SemanticFlatHorizonEncoderConfig.v1";
inline constexpr char flat_horizon_engine_name[] = "mifrost.SemanticFlatHorizonEncoderEngine.v1";
inline constexpr char batch_encoding_name[] = "mifrost.BatchEncoding.v1";
/**
 * A direct-View preparation. Unlike the other capsules this one is never
 * consumed by the core module: it stays owned by the backend module that
 * produced it and is borrowed again on every batch flush.
 */
inline constexpr char view_preparation_name[] = "mifrost.ViewPreparation.v1";
inline constexpr char consumed_name[] = "mifrost.consumed.v1";

template < typename T >
void delete_owned(PyObject* capsule) noexcept
{
   const auto* name = PyCapsule_GetName(capsule);
   auto* value = static_cast< T* >(PyCapsule_GetPointer(capsule, name));
   if(value == nullptr) {
      PyErr_Clear();
      return;
   }
   delete value;
}

template < typename T >
PyObject* make_owned(T value, const char* name)
{
   auto owned = std::make_unique< T >(std::move(value));
   auto* capsule = PyCapsule_New(owned.get(), name, &delete_owned< T >);
   if(capsule != nullptr) {
      owned.release();
   }
   return capsule;
}

template < typename T >
T* get(PyObject* capsule, const char* name)
{
   return static_cast< T* >(PyCapsule_GetPointer(capsule, name));
}

template < typename T >
T take(PyObject* capsule, const char* name)
{
   auto* value = get< T >(capsule, name);
   if(value == nullptr) {
      PyErr_Clear();
      throw std::invalid_argument("invalid or already-consumed Mifrost capsule");
   }
   T result = std::move(*value);
   delete value;
   PyCapsule_SetDestructor(capsule, nullptr);
   PyCapsule_SetName(capsule, consumed_name);
   return result;
}

}  // namespace mifrost::capsule_bridge
