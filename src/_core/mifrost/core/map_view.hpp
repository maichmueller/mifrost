#pragma once

#include <nanobind/make_iterator.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/pair.h>

#include <concepts>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace mifrost {

namespace nb = nanobind;

template < typename T >
inline constexpr bool always_false_v = false;

#define MIFROST_REGISTER_MAPVIEW_TYPE_TOKEN(type_expr, token_literal) \
   template <>                                                        \
   struct map_view_python_type_token_traits< type_expr > {            \
      static constexpr std::string_view value = token_literal;        \
   };

template < typename T >
struct map_view_python_type_token_traits {
   static_assert(always_false_v< T >, "Unsupported map view key/value type token");
};

template < typename T >
struct map_view_python_type_token_traits< T& >: map_view_python_type_token_traits< T > {};

template < typename T >
struct map_view_python_type_token_traits< const T >: map_view_python_type_token_traits< T > {};

template < typename T >
struct map_view_python_type_token_traits< volatile T >: map_view_python_type_token_traits< T > {};

template < typename T >
struct map_view_python_type_token_traits< const volatile T >:
    map_view_python_type_token_traits< T > {};

template < typename T >
   requires(
      std::is_integral_v< std::remove_cvref_t< T > >
      and not std::is_same_v< std::remove_cvref_t< T >, bool >
   )
struct map_view_python_type_token_traits< T > {
   static constexpr std::string_view value = "int";
};

template < typename T >
   requires(std::is_floating_point_v< std::remove_cvref_t< T > >)
struct map_view_python_type_token_traits< T > {
   static constexpr std::string_view value = "float";
};

MIFROST_REGISTER_MAPVIEW_TYPE_TOKEN(std::string, "str")
MIFROST_REGISTER_MAPVIEW_TYPE_TOKEN(bool, "bool")

template < typename T >
std::string map_view_python_type_token()
{
   using U = std::remove_cvref_t< T >;
   constexpr std::string_view override = map_view_python_type_token_traits< U >::value;
   return std::string{override};
}

template < typename T >
nb::object map_view_python_type_object()
{
   using U = std::remove_cvref_t< T >;
   nb::object builtins = nb::module_::import_("builtins");
   if constexpr(std::is_same_v< U, bool >) {
      return builtins.attr("bool");
   } else if constexpr(std::is_same_v< U, std::string >) {
      return builtins.attr("str");
   } else if constexpr(std::is_integral_v< U >) {
      return builtins.attr("int");
   } else if constexpr(std::is_floating_point_v< U >) {
      return builtins.attr("float");
   } else {
      static_assert(always_false_v< U >, "Unsupported map view key/value Python type");
   }
}

#undef MIFROST_REGISTER_MAPVIEW_TYPE_TOKEN

template < typename MapT >
concept MapLike = requires(const MapT& map, const typename MapT::key_type& key) {
   typename MapT::key_type;
   typename MapT::mapped_type;
   { map.size() } -> std::convertible_to< std::size_t >;
   { map.empty() } -> std::convertible_to< bool >;
   { map.begin() };
   { map.end() };
   { map.find(key) };
};

class MapViewBase {
  public:
   explicit MapViewBase(nb::object owner) : owner_(std::move(owner)) {}
   virtual ~MapViewBase() = default;

   [[nodiscard]] virtual std::size_t size() const = 0;
   [[nodiscard]] virtual bool empty() const = 0;
   [[nodiscard]] virtual bool contains_object(nb::handle key) const = 0;
   [[nodiscard]] virtual nb::object at_object(nb::handle key) const = 0;
   [[nodiscard]] virtual nb::object get_object(nb::handle key, nb::handle default_value) const = 0;
   [[nodiscard]] virtual nb::list keys_list() const = 0;
   [[nodiscard]] virtual nb::list values_list() const = 0;
   [[nodiscard]] virtual nb::list items_list() const = 0;
   [[nodiscard]] virtual nb::dict as_dict() const = 0;

  protected:
   nb::object owner_;
};

template < MapLike MapT >
class MapView: public MapViewBase {
  public:
   using map_type = MapT;
   using key_type = typename map_type::key_type;
   using mapped_type = typename map_type::mapped_type;
   using const_iterator = typename map_type::const_iterator;

   MapView(const map_type* map, nb::object owner) : MapViewBase(std::move(owner)), map_(map)
   {
      if(map_ == nullptr) {
         throw std::invalid_argument("MapView requires non-null map pointer");
      }
   }

   [[nodiscard]] const map_type& map() const { return *map_; }

   [[nodiscard]] std::size_t size() const override { return map().size(); }

   [[nodiscard]] bool empty() const override { return map().empty(); }

   [[nodiscard]] bool contains(const key_type& key) const { return map().find(key) != map().end(); }

   [[nodiscard]] const mapped_type& at(const key_type& key) const
   {
      auto it = map().find(key);
      if(it == map().end()) {
         throw nb::key_error();
      }
      return it->second;
   }

   [[nodiscard]] const_iterator begin() const { return map().begin(); }

   [[nodiscard]] const_iterator end() const { return map().end(); }

   [[nodiscard]] bool contains_object(nb::handle key) const override
   {
      try {
         return contains(nb::cast< key_type >(key));
      } catch(const nb::cast_error&) {
         return false;
      }
   }

   [[nodiscard]] nb::object at_object(nb::handle key) const override
   {
      try {
         const auto& value = at(nb::cast< key_type >(key));
         return nb::cast(value);
      } catch(const nb::cast_error&) {
         throw nb::key_error();
      }
   }

   [[nodiscard]] nb::object get_object(nb::handle key, nb::handle default_value) const override
   {
      try {
         const auto typed_key = nb::cast< key_type >(key);
         auto it = map().find(typed_key);
         if(it == map().end()) {
            return nb::borrow< nb::object >(default_value);
         }
         return nb::cast(it->second);
      } catch(const nb::cast_error&) {
         return nb::borrow< nb::object >(default_value);
      }
   }

   [[nodiscard]] nb::list keys_list() const override
   {
      nb::list out;
      for(const auto& [key, _] : map()) {
         out.append(nb::cast(key));
      }
      return out;
   }

   [[nodiscard]] nb::list values_list() const override
   {
      nb::list out;
      for(const auto& [_, value] : map()) {
         out.append(nb::cast(value));
      }
      return out;
   }

   [[nodiscard]] nb::list items_list() const override
   {
      nb::list out;
      for(const auto& [key, value] : map()) {
         out.append(nb::make_tuple(nb::cast(key), nb::cast(value)));
      }
      return out;
   }

   [[nodiscard]] nb::dict as_dict() const override
   {
      nb::dict out;
      for(const auto& [key, value] : map()) {
         out[nb::cast(key)] = nb::cast(value);
      }
      return out;
   }

  private:
   const map_type* map_;
};

template < MapLike MapT >
MapView< MapT > make_map_view(const MapT& map, nb::handle owner)
{
   return MapView< MapT >(&map, nb::borrow< nb::object >(owner));
}

template < MapLike MapT >
void bind_map_view(nb::module_& m, const char* python_name)
{
   using ViewT = MapView< MapT >;
   using Key = typename ViewT::key_type;

   struct KeyView {
      const ViewT* view;
   };
   struct ValueView {
      const ViewT* view;
   };
   struct ItemView {
      const ViewT* view;
   };

   auto view_cls = nb::class_< ViewT, MapViewBase >(m, python_name)
                      .def(
                         "__contains__",
                         [](const ViewT& view, const Key& key) { return view.contains(key); }
                      )
                      .def("__contains__", [](const ViewT&, nb::handle) { return false; })
                      .def(
                         "__getitem__",
                         [](
                            const ViewT& view, const Key& key
                         ) -> const typename ViewT::mapped_type& { return view.at(key); },
                         nb::rv_policy::reference_internal
                      )
                      .def(
                         "__iter__",
                         [](const ViewT& view) {
                            return nb::make_key_iterator(
                               nb::type< ViewT >(), "KeyIterator", view.begin(), view.end()
                            );
                         },
                         nb::keep_alive< 0, 1 >()
                      );
   view_cls.attr("key_type") = map_view_python_type_object< Key >();
   view_cls.attr("value_type") = map_view_python_type_object< typename ViewT::mapped_type >();
   nb::class_< KeyView >(view_cls, "KeyView")
      .def("__len__", [](const KeyView& v) { return v.view->size(); })
      .def("__contains__", [](const KeyView& v, const Key& key) { return v.view->contains(key); })
      .def("__contains__", [](const KeyView&, nb::handle) { return false; })
      .def(
         "__iter__",
         [](const KeyView& v) {
            return nb::make_key_iterator(
               nb::type< ViewT >(), "KeyIterator", v.view->begin(), v.view->end()
            );
         },
         nb::keep_alive< 0, 1 >()
      );

   nb::class_< ValueView >(view_cls, "ValueView")
      .def("__len__", [](const ValueView& v) { return v.view->size(); })
      .def(
         "__iter__",
         [](const ValueView& v) {
            return nb::make_value_iterator(
               nb::type< ViewT >(), "ValueIterator", v.view->begin(), v.view->end()
            );
         },
         nb::keep_alive< 0, 1 >()
      );

   nb::class_< ItemView >(view_cls, "ItemView")
      .def("__len__", [](const ItemView& v) { return v.view->size(); })
      .def(
         "__iter__",
         [](const ItemView& v) {
            return nb::make_iterator(
               nb::type< ViewT >(), "ItemIterator", v.view->begin(), v.view->end()
            );
         },
         nb::keep_alive< 0, 1 >()
      );

   view_cls
      .def(
         "keys",
         [](const ViewT& view) { return new KeyView{&view}; },
         nb::rv_policy::take_ownership,
         nb::keep_alive< 0, 1 >()
      )
      .def(
         "values",
         [](const ViewT& view) { return new ValueView{&view}; },
         nb::rv_policy::take_ownership,
         nb::keep_alive< 0, 1 >()
      )
      .def(
         "items",
         [](const ViewT& view) { return new ItemView{&view}; },
         nb::rv_policy::take_ownership,
         nb::keep_alive< 0, 1 >()
      );
}

inline void register_mapview_base_maybe(nb::module_& m)
{
   if(nb::type< MapViewBase >().is_valid()) {
      return;
   }
   nb::class_< MapViewBase >(m, "MapViewBase")
      .def("__len__", &MapViewBase::size)
      .def("__bool__", [](const MapViewBase& view) { return not view.empty(); })
      .def("__contains__", &MapViewBase::contains_object)
      .def("__getitem__", &MapViewBase::at_object)
      .def("contains", &MapViewBase::contains_object)
      .def("at", &MapViewBase::at_object)
      .def("get", &MapViewBase::get_object, nb::arg("key"), nb::arg("default_value") = nb::none())
      .def("keys_list", &MapViewBase::keys_list)
      .def("values_list", &MapViewBase::values_list)
      .def("items_list", &MapViewBase::items_list)
      .def("as_dict", &MapViewBase::as_dict);
}

template < MapLike MapT >
const std::string& map_view_python_identifier()
{
   using Key = typename MapT::key_type;
   using Value = typename MapT::mapped_type;
   static const std::string name = [] {
      std::string out = "_MapView_";
      out.append(map_view_python_type_token< Key >());
      out.push_back('_');
      out.append(map_view_python_type_token< Value >());
      return out;
   }();
   return name;
}

template < MapLike MapT >
void register_mapview_maybe(nb::module_& m)
{
   register_mapview_base_maybe(m);
   using ViewT = MapView< MapT >;
   if(nb::type< ViewT >().is_valid()) {
      return;
   }
   const auto& identifier = map_view_python_identifier< MapT >();
   bind_map_view< MapT >(m, identifier.c_str());
}

}  // namespace mifrost
