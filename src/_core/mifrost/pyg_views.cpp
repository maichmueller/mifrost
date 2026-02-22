
#include "mifrost/pyg_views.hpp"

#include "core/nb_instance.hpp"

namespace mifrost {

HeteroBatchEncodingView::HeteroBatchEncodingView(nb::object owner) : owner_(std::move(owner))
{
   encoding_ = require_instance_ptr< BatchBuilder::BatchEncoding >(
      owner_, "HeteroBatchEncodingView created with invalid BatchEncoding instance"
   );
}
nb::object HeteroBatchEncodingView::x_dict()
{
   if(x_dict_cache_.is_valid()) {
      return x_dict_cache_;
   }

   nb::dict out;
   for(const auto& node_type : encoding_->schema.node_types) {
      if(const auto key = find_node_attr_key(encoding_->schema, node_type, "x");
         key.has_value() and has_tensor(*key)) {
         out[node_type.c_str()] = tensor(*key);
         continue;
      }
      if(const auto it = encoding_->node_feature_dims.find(node_type);
         it != encoding_->node_feature_dims.end()) {
         int64_t count = 0;
         if(const auto count_it = encoding_->node_counts.find(node_type);
            count_it != encoding_->node_counts.end()) {
            count = std::max< int64_t >(0, count_it->second);
         }
         out[node_type.c_str()] = zeros_f32_on_owner_device(owner_, count, it->second);
      }
   }
   x_dict_cache_ = py::mapping_proxy(out);
   return x_dict_cache_;
}

nb::object HeteroBatchEncodingView::edge_index_dict()
{
   if(edge_index_dict_cache_.is_valid()) {
      return edge_index_dict_cache_;
   }

   nb::dict out;
   for(size_t idx = 0; idx < encoding_->schema.edge_types.size(); ++idx) {
      const auto [key0, key1] = find_edge_index_keys(encoding_->schema, static_cast< int >(idx));
      if(not key0.has_value() or not key1.has_value()) {
         continue;
      }
      if(not has_tensor(*key0) or not has_tensor(*key1)) {
         continue;
      }
      nb::list pair;
      pair.append(tensor(*key0));
      pair.append(tensor(*key1));
      nb::handle torch = py::torch_module();
      out[edge_type_to_tuple(encoding_->schema.edge_types[idx])] = torch.attr("stack")(
         pair, "dim"_a = 0
      );
   }
   edge_index_dict_cache_ = py::mapping_proxy(out);
   return edge_index_dict_cache_;
}

nb::object HeteroBatchEncodingView::batch_dict()
{
   if(batch_dict_cache_.is_valid()) {
      return batch_dict_cache_;
   }

   nb::dict out;
   for(const auto& node_type : encoding_->schema.node_types) {
      const std::string key = node_type + "/batch";
      if(has_tensor(key)) {
         out[node_type.c_str()] = tensor(key);
      }
   }
   batch_dict_cache_ = py::mapping_proxy(out);
   return batch_dict_cache_;
}

nb::object HeteroBatchEncodingView::ptr_dict()
{
   if(ptr_dict_cache_.is_valid()) {
      return ptr_dict_cache_;
   }

   nb::dict out;
   for(const auto& node_type : encoding_->schema.node_types) {
      const std::string key = node_type + "/ptr";
      if(has_tensor(key)) {
         out[node_type.c_str()] = tensor(key);
      }
   }
   ptr_dict_cache_ = py::mapping_proxy(out);
   return ptr_dict_cache_;
}

nb::object HeteroBatchEncodingView::edge_attr_dict()
{
   if(edge_attr_dict_cache_.is_valid()) {
      return edge_attr_dict_cache_;
   }

   nb::dict out;
   for(size_t idx = 0; idx < encoding_->schema.edge_types.size(); ++idx) {
      const auto key = find_edge_attr_key(encoding_->schema, static_cast< int >(idx));
      if(not key.has_value() or not has_tensor(*key)) {
         continue;
      }
      out[edge_type_to_tuple(encoding_->schema.edge_types[idx])] = tensor(*key);
   }
   edge_attr_dict_cache_ = py::mapping_proxy(out);
   return edge_attr_dict_cache_;
}

void HeteroBatchEncodingView::set_device(nb::handle device)
{
   if(device.is_none()) {
      return;
   }
   set_owner_target_device(owner_, device);
   materialize_owner_tensor_cache(owner_, *encoding_);
   clear_caches();
   prewarm_caches();
}

void HeteroBatchEncodingView::clear_caches()
{
   tensor_cache_ = nb::dict();
   x_dict_cache_ = nb::object();
   edge_index_dict_cache_ = nb::object();
   batch_dict_cache_ = nb::object();
   ptr_dict_cache_ = nb::object();
   edge_attr_dict_cache_ = nb::object();
}

void HeteroBatchEncodingView::prewarm_caches()
{
   (void) x_dict();
   (void) edge_index_dict();
   (void) batch_dict();
   (void) ptr_dict();
   (void) edge_attr_dict();
}

nb::object HeteroBatchEncodingView::tensor(const std::string& key)
{
   if(tensor_cache_.contains(key.c_str())) {
      return nb::borrow< nb::object >(tensor_cache_[key.c_str()]);
   }
   if(auto owner_cache = owner_tensor_cache_if_present(owner_);
      owner_cache.has_value() and owner_cache->contains(key.c_str())) {
      auto value = nb::borrow< nb::object >((*owner_cache)[key.c_str()]);
      tensor_cache_[key.c_str()] = value;
      return value;
   }
   nb::object value = batch_encoding_get_native_tensor(*encoding_, key, owner_);
   tensor_cache_[key.c_str()] = value;
   if(auto owner_cache = owner_tensor_cache_if_present(owner_); owner_cache.has_value()) {
      (*owner_cache)[key.c_str()] = value;
   }
   return value;
}

HomoBatchEncodingView::HomoBatchEncodingView(nb::object owner) : owner_(std::move(owner))
{
   encoding_ = require_instance_ptr< BatchBuilder::BatchEncoding >(
      owner_, "HomoBatchEncodingView created with invalid BatchEncoding instance"
   );
}

nb::object HomoBatchEncodingView::x()
{
   if(x_ready_) {
      return x_cache_;
   }
   x_ready_ = true;
   x_cache_ = nb::none();
   if(encoding_->schema.node_types.empty()) {
      return x_cache_;
   }
   const std::string& node_type = encoding_->schema.node_types.front();
   if(const auto key = find_node_attr_key(encoding_->schema, node_type, "x");
      key.has_value() and has_tensor(*key)) {
      x_cache_ = tensor(*key);
      return x_cache_;
   }
   if(const auto it = encoding_->node_feature_dims.find(node_type);
      it != encoding_->node_feature_dims.end()) {
      int64_t count = 0;
      if(const auto count_it = encoding_->node_counts.find(node_type);
         count_it != encoding_->node_counts.end()) {
         count = std::max< int64_t >(0, count_it->second);
      }
      x_cache_ = zeros_f32_on_owner_device(owner_, count, it->second);
   }
   return x_cache_;
}

nb::object HomoBatchEncodingView::edge_index()
{
   if(edge_index_ready_) {
      return edge_index_cache_;
   }
   edge_index_ready_ = true;
   if(encoding_->schema.edge_types.empty()) {
      return edge_index_cache_ = nb::none();
   }
   const auto [key0, key1] = find_edge_index_keys(encoding_->schema, 0);
   if(not key0.has_value() or not key1.has_value() or not has_tensor(*key0)
      or not has_tensor(*key1)) {
      return edge_index_cache_ = nb::none();
   }
   nb::list pair;
   pair.append(tensor(*key0));
   pair.append(tensor(*key1));
   nb::handle torch = py::torch_module();
   edge_index_cache_ = torch.attr("stack")(pair, "dim"_a = 0);
   return edge_index_cache_;
}

nb::object HomoBatchEncodingView::batch()
{
   if(batch_ready_) {
      return batch_cache_;
   }
   batch_ready_ = true;
   std::string key;
   if(encoding_->schema.node_types.empty() or std::invoke([&] {
         key = encoding_->schema.node_types.front() + "/batch";
         return not has_tensor(key);
      })) {
      return batch_cache_ = nb::none();
   }
   return batch_cache_ = tensor(key);
}

nb::object HomoBatchEncodingView::ptr()
{
   if(ptr_ready_) {
      return ptr_cache_;
   }
   ptr_ready_ = true;
   std::string key;
   if(encoding_->schema.node_types.empty() or std::invoke([&] {
         key = encoding_->schema.node_types.front() + "/ptr";
         return not has_tensor(key);
      })) {
      return ptr_cache_ = nb::none();
   }
   return ptr_cache_ = tensor(key);
}

nb::object HomoBatchEncodingView::edge_attr()
{
   if(edge_attr_ready_) {
      return edge_attr_cache_;
   }
   edge_attr_ready_ = true;
   edge_attr_cache_ = nb::none();
   if(encoding_->schema.edge_types.empty()) {
      return edge_attr_cache_;
   }
   const auto key = find_edge_attr_key(encoding_->schema, 0);
   if(key.has_value() and has_tensor(*key)) {
      edge_attr_cache_ = tensor(*key);
   }
   return edge_attr_cache_;
}

void HomoBatchEncodingView::set_device(nb::handle device)
{
   if(device.is_none()) {
      return;
   }
   set_owner_target_device(owner_, device);
   materialize_owner_tensor_cache(owner_, *encoding_);
   clear_caches();
   prewarm_caches();
}

void HomoBatchEncodingView::clear_caches()
{
   tensor_cache_ = nb::dict();
   x_ready_ = false;
   edge_index_ready_ = false;
   batch_ready_ = false;
   ptr_ready_ = false;
   edge_attr_ready_ = false;
   x_cache_ = nb::object();
   edge_index_cache_ = nb::object();
   batch_cache_ = nb::object();
   ptr_cache_ = nb::object();
   edge_attr_cache_ = nb::object();
}

void HomoBatchEncodingView::prewarm_caches()
{
   (void) x();
   (void) edge_index();
   (void) batch();
   (void) ptr();
   (void) edge_attr();
}

nb::object HomoBatchEncodingView::tensor(const std::string& key)
{
   if(tensor_cache_.contains(key.c_str())) {
      return nb::borrow< nb::object >(tensor_cache_[key.c_str()]);
   }
   if(auto owner_cache = owner_tensor_cache_if_present(owner_);
      owner_cache.has_value() and owner_cache->contains(key.c_str())) {
      auto value = nb::borrow< nb::object >((*owner_cache)[key.c_str()]);
      tensor_cache_[key.c_str()] = value;
      return value;
   }
   nb::object value = batch_encoding_get_native_tensor(*encoding_, key, owner_);
   tensor_cache_[key.c_str()] = value;
   if(auto owner_cache = owner_tensor_cache_if_present(owner_); owner_cache.has_value()) {
      (*owner_cache)[key.c_str()] = value;
   }
   return value;
}

}  // namespace mifrost