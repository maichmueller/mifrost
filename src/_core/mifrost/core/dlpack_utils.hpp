#pragma once

#include <Python.h>
#include <dlpack/dlpack.h>
#include <nanobind/nanobind.h>

#include <cstdint>
#include <type_traits>
#include <vector>

#include "mifrost/core/api.hpp"

namespace mifrost {

namespace nb = nanobind;

namespace dlpack_utils {

inline void dlpack_capsule_destructor(PyObject* capsule)
{
   if(not PyCapsule_IsValid(capsule, "dltensor")) {
      return;
   }
   auto* managed = static_cast< DLManagedTensor* >(PyCapsule_GetPointer(capsule, "dltensor"));
   if(managed != nullptr and managed->deleter != nullptr) {
      managed->deleter(managed);
   }
}

inline bool is_dlpack_capsule(nb::handle value)
{
   return PyCapsule_IsValid(value.ptr(), "dltensor");
}

template < typename T >
constexpr DLDataType dlpack_dtype()
{
   if constexpr(std::is_same_v< T, float >) {
      return DLDataType{.code = static_cast< uint8_t >(kDLFloat), .bits = 32, .lanes = 1};
   } else if constexpr(std::is_same_v< T, int64_t >) {
      return DLDataType{.code = static_cast< uint8_t >(kDLInt), .bits = 64, .lanes = 1};
   } else {
      static_assert(std::is_same_v< T, float > or std::is_same_v< T, int64_t >);
   }
}

template < typename T >
struct MIFROST_LOCAL VectorDlpackCtx {
   std::vector< T >* vec = nullptr;
   nb::object owner;
   bool owns_vec = false;
   DLManagedTensor managed{};
   int64_t shape[2]{0, 0};
};

template < typename T >
void vector_dlpack_managed_deleter(DLManagedTensor* managed) noexcept
{
   if(managed == nullptr) {
      return;
   }
   auto* ctx = static_cast< VectorDlpackCtx< T >* >(managed->manager_ctx);
   if(ctx == nullptr) {
      return;
   }
   if(ctx->owns_vec and ctx->vec != nullptr) {
      delete ctx->vec;
   }
   if(ctx->owner.is_valid()) {
      nb::gil_scoped_acquire guard;
      delete ctx;
      return;
   }
   delete ctx;
}

template < typename T >
nb::object make_dlpack_capsule(VectorDlpackCtx< T >* ctx, int ndim, int64_t rows, int64_t cols)
{
   ctx->managed.dl_tensor.data = ctx->vec->data();
   ctx->managed.dl_tensor.device = DLDevice{.device_type = kDLCPU, .device_id = 0};
   ctx->managed.dl_tensor.ndim = ndim;
   ctx->managed.dl_tensor.dtype = dlpack_dtype< T >();
   ctx->managed.dl_tensor.shape = ctx->shape;
   ctx->managed.dl_tensor.strides = nullptr;
   ctx->managed.dl_tensor.byte_offset = 0;
   ctx->shape[0] = rows;
   ctx->shape[1] = cols;
   ctx->managed.manager_ctx = ctx;
   ctx->managed.deleter = &vector_dlpack_managed_deleter< T >;

   PyObject* capsule = PyCapsule_New(&ctx->managed, "dltensor", &dlpack_capsule_destructor);
   if(capsule == nullptr) {
      ctx->managed.deleter(&ctx->managed);
      throw nb::python_error();
   }
   return nb::steal< nb::object >(capsule);
}

template < typename T >
nb::object vector_to_dlpack_view_1d(std::vector< T >& vec, nb::handle owner)
{
   auto* ctx = new VectorDlpackCtx< T >;
   ctx->vec = &vec;
   ctx->owner = nb::borrow< nb::object >(owner);
   return make_dlpack_capsule(ctx, 1, static_cast< int64_t >(vec.size()), 0);
}

template < typename T >
nb::object
vector_to_dlpack_view_2d(std::vector< T >& vec, size_t rows, size_t cols, nb::handle owner)
{
   auto* ctx = new VectorDlpackCtx< T >;
   ctx->vec = &vec;
   ctx->owner = nb::borrow< nb::object >(owner);
   return make_dlpack_capsule(ctx, 2, static_cast< int64_t >(rows), static_cast< int64_t >(cols));
}

template < typename T >
nb::object vector_to_dlpack_owned_1d(std::vector< T >&& vec)
{
   auto* ctx = new VectorDlpackCtx< T >;
   ctx->vec = new std::vector< T >(std::move(vec));
   ctx->vec->shrink_to_fit();
   ctx->owns_vec = true;
   return make_dlpack_capsule(ctx, 1, static_cast< int64_t >(ctx->vec->size()), 0);
}

template < typename T >
nb::object vector_to_dlpack_owned_2d(std::vector< T >&& vec, size_t rows, size_t cols)
{
   auto* ctx = new VectorDlpackCtx< T >;
   ctx->vec = new std::vector< T >(std::move(vec));
   ctx->vec->shrink_to_fit();
   ctx->owns_vec = true;
   return make_dlpack_capsule(ctx, 2, static_cast< int64_t >(rows), static_cast< int64_t >(cols));
}

template < typename T >
nb::object vector_to_dlpack_owned_copy_1d(const std::vector< T >& vec)
{
   std::vector< T > copy(vec);
   return vector_to_dlpack_owned_1d(std::move(copy));
}

template < typename T >
nb::object vector_to_dlpack_owned_copy_2d(const std::vector< T >& vec, size_t rows, size_t cols)
{
   std::vector< T > copy(vec);
   return vector_to_dlpack_owned_2d(std::move(copy), rows, cols);
}

}  // namespace dlpack_utils

}  // namespace mifrost
