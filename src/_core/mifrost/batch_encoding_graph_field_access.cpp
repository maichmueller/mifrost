#include "mifrost/batch_encoding_graph_field_access.hpp"

#include <nanobind/ndarray.h>

#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace nb = nanobind;

namespace mifrost {

namespace {

enum class GraphFieldLookupKind { VALUE, PTR, MISSING };

struct GraphFieldLookup {
   GraphFieldLookupKind kind = GraphFieldLookupKind::MISSING;
   GraphField* field = nullptr;
};

template < typename T >
void vector_owner_deleter(void* p) noexcept
{
   delete static_cast< std::vector< T >* >(p);
}

template < typename T >
auto vector_to_1d_ndarray_view(std::vector< T >& vec, nb::handle owner)
{
   size_t shape[1] = {vec.size()};
   return nb::ndarray< nb::numpy, T, nb::shape< -1 > >(vec.data(), 1, shape, owner);
}

template < typename T >
auto vector_to_2d_ndarray_view(std::vector< T >& vec, size_t rows, size_t cols, nb::handle owner)
{
   size_t shape[2] = {rows, cols};
   return nb::ndarray< nb::numpy, T, nb::shape< -1, -1 > >(vec.data(), 2, shape, owner);
}

template < typename T >
auto vector_to_1d_ndarray_owned_copy(const std::vector< T >& vec)
{
   auto* heap_vec = new std::vector< T >(vec);
   heap_vec->shrink_to_fit();
   size_t shape[1] = {heap_vec->size()};
   nb::capsule owner(heap_vec, vector_owner_deleter< T >);
   return nb::ndarray< nb::numpy, T, nb::shape< -1 > >(heap_vec->data(), 1, shape, owner);
}

template < typename T >
auto vector_to_2d_ndarray_owned_copy(const std::vector< T >& vec, size_t rows, size_t cols)
{
   auto* heap_vec = new std::vector< T >(vec);
   heap_vec->shrink_to_fit();
   size_t shape[2] = {rows, cols};
   nb::capsule owner(heap_vec, vector_owner_deleter< T >);
   return nb::ndarray< nb::numpy, T, nb::shape< -1, -1 > >(heap_vec->data(), 2, shape, owner);
}

GraphFieldLookup
batch_encoding_lookup_graph_field(BatchBuilder::BatchEncoding& encoding, std::string_view key)
{
   if(const auto it = encoding.graph_fields.find(std::string(key));
      it != encoding.graph_fields.end()) {
      return {.kind = GraphFieldLookupKind::VALUE, .field = &it->second};
   }
   constexpr std::string_view kPtrSuffix = "_ptr";
   if(key.size() <= kPtrSuffix.size()
      || key.compare(key.size() - kPtrSuffix.size(), kPtrSuffix.size(), kPtrSuffix) != 0) {
      return {};
   }
   std::string base(key.substr(0, key.size() - kPtrSuffix.size()));
   if(const auto it = encoding.graph_fields.find(base); it != encoding.graph_fields.end()) {
      if(it->second.spec.mode == GraphFieldMode::RAGGED_CAT) {
         return {.kind = GraphFieldLookupKind::PTR, .field = &it->second};
      }
   }
   return {};
}

nb::object to_torch_tensor(nb::handle array_like)
{
   static const nb::object numpy = nb::module_::import_("numpy");
   static const nb::object torch = nb::module_::import_("torch");
   nb::object array = numpy.attr("asarray")(array_like);
   return torch.attr("from_numpy")(array);
}

nb::object graph_field_values_to_tensor_view(GraphField& field, nb::handle owner)
{
   return std::visit(
      [&](auto& values) -> nb::object {
         using T = std::decay_t< decltype(values) >::value_type;
         if(field.spec.dim == 1) {
            return to_torch_tensor(vector_to_1d_ndarray_view(values, owner).cast());
         }
         const bool cat_dim_one = (field.spec.mode == GraphFieldMode::CAT
                                   || field.spec.mode == GraphFieldMode::RAGGED_CAT)
                                  && graph_field_cat_dim_is_one(field.spec.cat_dim);
         const size_t rows = cat_dim_one ? static_cast< size_t >(field.spec.dim)
                                         : values.size() / static_cast< size_t >(field.spec.dim);
         const size_t cols = cat_dim_one ? values.size() / static_cast< size_t >(field.spec.dim)
                                         : static_cast< size_t >(field.spec.dim);
         return to_torch_tensor(vector_to_2d_ndarray_view(values, rows, cols, owner).cast());
      },
      field.values
   );
}

nb::object graph_field_ptr_to_tensor(const GraphField& field)
{
   return to_torch_tensor(vector_to_1d_ndarray_owned_copy(field.ptr).cast());
}

}  // namespace

std::set< std::string > batch_encoding_native_graph_field_keys(
   const BatchBuilder::BatchEncoding& encoding
)
{
   std::set< std::string > out;
   for(const auto& [key, field] : encoding.graph_fields) {
      out.insert(key);
      if(field.spec.mode == GraphFieldMode::RAGGED_CAT) {
         out.insert(key + "_ptr");
      }
   }
   return out;
}

bool batch_encoding_has_graph_field(
   const BatchBuilder::BatchEncoding& encoding,
   std::string_view key
)
{
   if(encoding.graph_fields.contains(std::string(key))) {
      return true;
   }
   constexpr std::string_view kPtrSuffix = "_ptr";
   if(key.size() <= kPtrSuffix.size()
      || key.compare(key.size() - kPtrSuffix.size(), kPtrSuffix.size(), kPtrSuffix) != 0) {
      return false;
   }
   std::string base(key.substr(0, key.size() - kPtrSuffix.size()));
   if(const auto it = encoding.graph_fields.find(base); it != encoding.graph_fields.end()) {
      return it->second.spec.mode == GraphFieldMode::RAGGED_CAT;
   }
   return false;
}

nb::object batch_encoding_get_graph_field(
   BatchBuilder::BatchEncoding& encoding,
   std::string_view key,
   nb::handle owner
)
{
   const auto lookup = batch_encoding_lookup_graph_field(encoding, key);
   if(lookup.kind == GraphFieldLookupKind::VALUE) {
      return graph_field_values_to_tensor_view(*lookup.field, owner);
   }
   if(lookup.kind == GraphFieldLookupKind::PTR) {
      return graph_field_ptr_to_tensor(*lookup.field);
   }
   throw std::invalid_argument(
      "BatchEncoding.get_graph_field unknown key '" + std::string(key) + "'"
   );
}

void validate_batch_encoding_graph_fields(
   const BatchBuilder::BatchEncoding& encoding,
   std::string_view context
)
{
   for(const auto& [key, field] : encoding.graph_fields) {
      try {
         validate_graph_field_storage(key, field, encoding.num_graphs);
      } catch(const std::exception& ex) {
         throw std::invalid_argument(
            "Invalid graph field '" + key + "' in " + std::string(context) + ": " + ex.what()
         );
      }
   }
}

}  // namespace mifrost
