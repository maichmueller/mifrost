#include "mifrost/init_batch_encoding.hpp"

#include <fmt/format.h>
#include <fmt/ranges.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <ranges>
#include <set>
#include <string>
#include <string_view>

#include "mifrost/batch_encoding_graph_field_access.hpp"
#include "mifrost/batch_encoding_python_collation.hpp"
#include "mifrost/common.hpp"

namespace nb = nanobind;

namespace mifrost {

struct ReprQuoted {
   std::string_view value;
};

struct ReprEdgeType {
   const EdgeType* value = nullptr;
};

struct DisplayEdgeType {
   const EdgeType* value = nullptr;
};

}  // namespace mifrost

template <>
struct fmt::formatter< mifrost::ReprQuoted > {
   constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }

   template < typename FormatContext >
   auto format(const mifrost::ReprQuoted& quoted, FormatContext& ctx) const
   {
      auto out = ctx.out();
      *out++ = '\'';
      for(char ch : quoted.value) {
         if(ch == '\'' or ch == '\\') {
            *out++ = '\\';
         }
         *out++ = ch;
      }
      *out++ = '\'';
      return out;
   }
};

template <>
struct fmt::formatter< mifrost::ReprEdgeType > {
   constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }

   template < typename FormatContext >
   auto format(const mifrost::ReprEdgeType& edge_type, FormatContext& ctx) const
   {
      if(edge_type.value == nullptr) {
         return fmt::format_to(ctx.out(), "(None)");
      }
      return fmt::format_to(
         ctx.out(),
         "({}, {}, {})",
         mifrost::ReprQuoted{edge_type.value->src},
         mifrost::ReprQuoted{edge_type.value->rel},
         mifrost::ReprQuoted{edge_type.value->dst}
      );
   }
};

template <>
struct fmt::formatter< mifrost::DisplayEdgeType > {
   constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }

   template < typename FormatContext >
   auto format(const mifrost::DisplayEdgeType& edge_type, FormatContext& ctx) const
   {
      if(edge_type.value == nullptr) {
         return fmt::format_to(ctx.out(), "(None)");
      }
      return fmt::format_to(
         ctx.out(), "({}, {}, {})", edge_type.value->src, edge_type.value->rel, edge_type.value->dst
      );
   }
};

namespace mifrost {

std::string batch_encoding_repr(nb::handle self, const BatchBuilder::BatchEncoding& encoding)
{
   const auto native_field_keys = batch_encoding_native_graph_field_keys(encoding);
   nb::dict attrs = batch_encoding_python_attrs(self);
   std::set< std::string > python_attr_keys;
   for(auto [key_obj, value_obj] : attrs) {
      (void) value_obj;
      const std::string key = py::to_std_string(key_obj);
      if(is_reserved_python_attr_key(key) or native_field_keys.contains(key)) {
         continue;
      }
      python_attr_keys.insert(key);
   }

   const auto node_type_reprs = encoding.schema.node_types
                                | std::views::transform([](const std::string& value) {
                                     return ReprQuoted{value};
                                  });
   const auto edge_type_reprs = encoding.schema.edge_types
                                | std::views::transform([](const EdgeType& value) {
                                     return ReprEdgeType{&value};
                                  });
   const auto field_key_reprs = native_field_keys | std::views::transform([](const auto& value) {
                                   return ReprQuoted{value};
                                });
   const auto python_attr_reprs = python_attr_keys | std::views::transform([](const auto& value) {
                                     return ReprQuoted{value};
                                  });

   nb::object device = owner_target_device(self);
   if(device.is_none()) {
      return fmt::format(
         "BatchEncoding(graph_kind={}, num_graphs={}, num_nodes={}, num_edges={}, "
         "node_types=[{}], edge_types=[{}], fields=[{}], python_attrs=[{}], device=None)",
         ReprQuoted{encoding.graph_kind},
         encoding.num_graphs,
         batch_encoding_num_nodes(encoding),
         batch_encoding_num_edges(encoding),
         fmt::join(node_type_reprs, ", "),
         fmt::join(edge_type_reprs, ", "),
         fmt::join(field_key_reprs, ", "),
         fmt::join(python_attr_reprs, ", ")
      );
   }
   const std::string device_repr = py::to_std_string(nb::str(device));
   return fmt::format(
      "BatchEncoding(graph_kind={}, num_graphs={}, num_nodes={}, num_edges={}, "
      "node_types=[{}], edge_types=[{}], fields=[{}], python_attrs=[{}], device={})",
      ReprQuoted{encoding.graph_kind},
      encoding.num_graphs,
      batch_encoding_num_nodes(encoding),
      batch_encoding_num_edges(encoding),
      fmt::join(node_type_reprs, ", "),
      fmt::join(edge_type_reprs, ", "),
      fmt::join(field_key_reprs, ", "),
      fmt::join(python_attr_reprs, ", "),
      ReprQuoted{device_repr}
   );
}

std::string batch_encoding_str(nb::handle self, const BatchBuilder::BatchEncoding& encoding)
{
   const auto native_field_keys = batch_encoding_native_graph_field_keys(encoding);
   nb::dict attrs = batch_encoding_python_attrs(self);
   std::set< std::string > python_attr_keys;
   for(auto [key_obj, value_obj] : attrs) {
      (void) value_obj;
      const std::string key = py::to_std_string(key_obj);
      if(is_reserved_python_attr_key(key) or native_field_keys.contains(key)) {
         continue;
      }
      python_attr_keys.insert(key);
   }

   const auto edge_type_views = encoding.schema.edge_types
                                | std::views::transform([](const EdgeType& value) {
                                     return DisplayEdgeType{&value};
                                  });
   nb::object device = owner_target_device(self);
   const std::string device_str = device.is_none() ? "None" : py::to_std_string(nb::str(device));

   return fmt::format(
      "BatchEncoding(graph_kind={}, num_graphs={}, num_nodes={}, num_edges={}, "
      "node_types=[{}], edge_types=[{}], fields=[{}], python_attrs=[{}], device={})",
      encoding.graph_kind,
      encoding.num_graphs,
      batch_encoding_num_nodes(encoding),
      batch_encoding_num_edges(encoding),
      fmt::join(encoding.schema.node_types, ", "),
      fmt::join(edge_type_views, ", "),
      fmt::join(native_field_keys, ", "),
      fmt::join(python_attr_keys, ", "),
      device_str
   );
}

}  // namespace mifrost
