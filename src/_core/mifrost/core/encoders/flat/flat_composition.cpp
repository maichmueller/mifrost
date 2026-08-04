#include "flat_composition.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <map>
#include <stdexcept>
#include <utility>

namespace mifrost {

namespace {

constexpr std::array< FlatExternalModeContract, 6 > kExternalModeContracts = {{
   {
      FlatExternalMode::concurrent_internal,
      "concurrent_internal",
      FlatExternalComponent::state_facts | FlatExternalComponent::goal_facts
         | FlatExternalComponent::transition_effects,
   },
   {
      FlatExternalMode::concurrent_internal_tree,
      "concurrent_internal_tree",
      FlatExternalComponent::state_facts | FlatExternalComponent::goal_facts
         | FlatExternalComponent::transition_effects | FlatExternalComponent::parent_relations,
   },
   {
      FlatExternalMode::concurrent_internal_tree_rooted,
      "concurrent_internal_tree_rooted",
      FlatExternalComponent::state_facts | FlatExternalComponent::goal_facts
         | FlatExternalComponent::ground_actions | FlatExternalComponent::transition_effects
         | FlatExternalComponent::parent_relations | FlatExternalComponent::root_action_nodes,
   },
   {
      FlatExternalMode::concurrent_internal_comparison_tree,
      "concurrent_internal_comparison_tree",
      FlatExternalComponent::state_facts | FlatExternalComponent::goal_facts
         | FlatExternalComponent::transition_effects | FlatExternalComponent::parent_relations
         | FlatExternalComponent::shared_state,
   },
   {
      FlatExternalMode::concurrent_internal_action_tree,
      "concurrent_internal_action_tree",
      FlatExternalComponent::state_facts | FlatExternalComponent::goal_facts
         | FlatExternalComponent::ground_actions | FlatExternalComponent::parent_relations,
   },
   {
      FlatExternalMode::concurrent_internal_action_hybrid_tree,
      "concurrent_internal_action_hybrid_tree",
      FlatExternalComponent::state_facts | FlatExternalComponent::goal_facts
         | FlatExternalComponent::ground_actions | FlatExternalComponent::transition_effects
         | FlatExternalComponent::parent_relations,
   },
}};

void validate_node_type_id(const FlatNodeSchema& schema, FlatNodeTypeId id)
{
   if(id < 0 or static_cast< size_t >(id) >= schema.size()) {
      throw std::invalid_argument("Flat node type id is out of range");
   }
}

void register_relation_runtime_fields(
   BatchBuilder& builder,
   size_t relation_count,
   std::string_view relation_args_node_type
)
{
   builder.register_field(
      std::string(kRelationInstanceSizesField),
      GraphFieldSpec{.dtype = GraphFieldDType::I64, .mode = GraphFieldMode::STACK, .dim = 1}
   );
   builder.register_field(
      std::string(kRelationCountsField),
      GraphFieldSpec{
         .dtype = GraphFieldDType::I64,
         .mode = GraphFieldMode::STACK,
         .dim = static_cast< int >(relation_count),
      }
   );
   builder.register_field(
      std::string(kRelationArgsField),
      GraphFieldSpec{
         .dtype = GraphFieldDType::I64,
         .mode = GraphFieldMode::CAT,
         .dim = 1,
         .inc = GraphFieldInc{
            .kind = GraphFieldInc::Kind::NODE_OFFSET,
            .node_type = std::string(relation_args_node_type),
         },
      }
   );
}

const FlatFieldPlanEntry&
find_field(const std::vector< FlatFieldPlanEntry >& fields, std::string_view key)
{
   const auto it = std::ranges::find(fields, key, &FlatFieldPlanEntry::key);
   if(it == fields.end()) {
      throw std::invalid_argument("Flat composition field is not declared: " + std::string(key));
   }
   return *it;
}

}  // namespace

const FlatExternalModeContract& flat_external_mode_contract(FlatExternalMode mode)
{
   const auto it = std::ranges::find(kExternalModeContracts, mode, &FlatExternalModeContract::mode);
   if(it == kExternalModeContracts.end()) {
      throw std::invalid_argument("Unknown flat external mode");
   }
   return *it;
}

std::span< const FlatExternalModeContract > flat_external_mode_contracts()
{
   return kExternalModeContracts;
}

const FlatNodeTypeSpec& FlatNodeSchema::spec(FlatNodeTypeId id) const
{
   validate_node_type_id(*this, id);
   return specs_[static_cast< size_t >(id)];
}

FlatNodeTypeId FlatNodeSchema::id_for(std::string_view name) const
{
   const auto it = ids_.find(std::string(name));
   if(it == ids_.end()) {
      throw std::invalid_argument("Unknown flat node type '" + std::string(name) + "'");
   }
   return it->second;
}

std::optional< FlatNodeTypeId > FlatNodeSchema::try_id_for(std::string_view name) const
{
   const auto it = ids_.find(std::string(name));
   if(it == ids_.end()) {
      return std::nullopt;
   }
   return it->second;
}

FlatNodeTypeId FlatNodeSchemaBuilder::declare_node_type(
   std::string name,
   FlatNodeKind kind,
   int feature_dim,
   bool export_names
)
{
   if(name.empty()) {
      throw std::invalid_argument("Flat node type name must not be empty");
   }
   if(feature_dim <= 0) {
      throw std::invalid_argument("Flat node feature dimension must be positive");
   }
   if(const auto it = schema_.ids_.find(name); it != schema_.ids_.end()) {
      const FlatNodeTypeSpec expected{std::move(name), kind, feature_dim, export_names};
      if(schema_.specs_[static_cast< size_t >(it->second)] != expected) {
         throw std::invalid_argument("Flat node type redeclared with incompatible specification");
      }
      return it->second;
   }

   const auto id = static_cast< FlatNodeTypeId >(schema_.specs_.size());
   schema_.ids_.emplace(name, id);
   schema_.specs_.push_back(FlatNodeTypeSpec{std::move(name), kind, feature_dim, export_names});
   return id;
}

FlatNodeSchema FlatNodeSchemaBuilder::finalize() &&
{
   if(schema_.specs_.empty()) {
      throw std::invalid_argument("Flat composition plan declares no node types");
   }
   return std::move(schema_);
}

FlatNodeRef FlatNodePlanBuilder::add_node(FlatNodeTypeId type, std::string key)
{
   validate_node_type_id(schema_, type);
   if(key.empty()) {
      throw std::invalid_argument("Flat graph node key must not be empty");
   }
   const auto type_index = static_cast< size_t >(type);
   if(names_.empty()) {
      names_.resize(schema_.size());
      indices_.resize(schema_.size());
   }
   auto& index_by_key = indices_[type_index];
   if(const auto it = index_by_key.find(key); it != index_by_key.end()) {
      return FlatNodeRef{type, std::move(key)};
   }
   const auto index = static_cast< int64_t >(names_[type_index].size());
   index_by_key.emplace(key, index);
   names_[type_index].push_back(key);
   return FlatNodeRef{type, std::move(key)};
}

FlatNodeRef FlatNodePlanBuilder::add_node(std::string_view type_name, std::string key)
{
   return add_node(schema_.id_for(type_name), std::move(key));
}

FlatNodePlan FlatNodePlanBuilder::finish() &&
{
   if(names_.empty()) {
      names_.resize(schema_.size());
      indices_.resize(schema_.size());
   }
   FlatNodePlan plan;
   plan.schema_ = &schema_;
   plan.names_ = std::move(names_);
   plan.indices_ = std::move(indices_);
   return plan;
}

int64_t FlatNodePlan::index(FlatNodeRef ref) const
{
   return index(ref.type, ref.key);
}

int64_t FlatNodePlan::index(FlatNodeTypeId type, std::string_view key) const
{
   if(schema_ == nullptr) {
      throw std::logic_error("Flat node plan is not initialized");
   }
   validate_node_type_id(*schema_, type);
   const auto& index_by_key = indices_[static_cast< size_t >(type)];
   const auto it = index_by_key.find(std::string(key));
   if(it == index_by_key.end()) {
      throw std::invalid_argument("Unknown graph-local flat node key '" + std::string(key) + "'");
   }
   return it->second;
}

int64_t FlatNodePlan::count(FlatNodeTypeId type) const
{
   validate_node_type_id(*schema_, type);
   return static_cast< int64_t >(names_[static_cast< size_t >(type)].size());
}

const std::vector< std::string >& FlatNodePlan::names(FlatNodeTypeId type) const
{
   validate_node_type_id(*schema_, type);
   return names_[static_cast< size_t >(type)];
}

FlatSlotResolver FlatSlotResolver::source(int slot)
{
   if(slot < 0) {
      throw std::invalid_argument("Flat source slot must be non-negative");
   }
   return FlatSlotResolver{.kind = FlatSlotResolverKind::source_slot, .source_slot = slot};
}

FlatSlotResolver FlatSlotResolver::constant_value(int64_t value)
{
   return FlatSlotResolver{.kind = FlatSlotResolverKind::constant, .constant = value};
}

FlatSlotResolver FlatSlotResolver::node_ref(FlatNodeRef ref)
{
   if(ref.type < 0 or ref.key.empty()) {
      throw std::invalid_argument("Flat node slot resolver requires a valid node reference");
   }
   return FlatSlotResolver{.kind = FlatSlotResolverKind::node, .node = std::move(ref)};
}

std::vector< int64_t > CompiledFlatRelationProjection::project(
   std::span< const int64_t > source_args,
   const FlatNodePlan& nodes
) const
{
   std::vector< int64_t > output;
   output.reserve(slots.size());
   for(const auto& slot : slots) {
      switch(slot.kind) {
         case FlatSlotResolverKind::source_slot:
            if(slot.source_slot < 0
               or static_cast< size_t >(slot.source_slot) >= source_args.size()) {
               throw std::invalid_argument("Flat relation projection source slot is out of range");
            }
            output.push_back(source_args[static_cast< size_t >(slot.source_slot)]);
            break;
         case FlatSlotResolverKind::constant: output.push_back(slot.constant); break;
         case FlatSlotResolverKind::node: output.push_back(nodes.index(slot.node)); break;
      }
   }
   return output;
}

void FlatFieldPlanBuilder::register_field(std::string key, GraphFieldSpec spec)
{
   if(key.empty()) {
      throw std::invalid_argument("Flat graph field key must not be empty");
   }
   validate_graph_field_spec(key, spec);
   if(const auto it = index_by_key_.find(key); it != index_by_key_.end()) {
      auto& existing = entries_[it->second];
      if(existing.spec != spec or existing.owner != owner_) {
         throw std::invalid_argument(
            "Flat graph field '" + key + "' has conflicting declaration or owner"
         );
      }
      return;
   }
   index_by_key_.emplace(key, entries_.size());
   entries_.push_back(FlatFieldPlanEntry{std::move(key), spec, owner_});
}

void FlatSchemaPlanBuilder::register_relation(
   RelationKey key,
   FlatTupleLayout layout,
   RelationUsage usage
)
{
   relation_schema_.register_relation(std::move(key), std::move(layout), usage);
}

void FlatSchemaPlanBuilder::add_projection(FlatRelationProjection projection)
{
   if(projection.slots.empty()) {
      throw std::invalid_argument("Flat relation projection must declare at least one slot");
   }
   projections_.push_back(std::move(projection));
}

FlatNodeTypeId FlatSchemaPlanBuilder::declare_node_type(
   std::string name,
   FlatNodeKind kind,
   int feature_dim,
   bool export_names
)
{
   return node_schema_.declare_node_type(std::move(name), kind, feature_dim, export_names);
}

FlatRelationSchema FlatSchemaPlanBuilder::finalize_schema(const FlatCompositionConfig& config) &&
{
   return std::move(relation_schema_)
      .finalize(
         config.max_goal_level,
         config.support_literals,
         config.goal_derivations,
         config.empty_schema_error
      );
}

void FlatGraphContext::emit(RelationKey key, std::span< const int64_t > args) const
{
   const auto relation_id = schema.id_for(key);
   if(static_cast< size_t >(relation_id) >= schema.arities().size()) {
      throw std::logic_error("Flat relation schema has no arity for relation id");
   }
   if(args.size() != static_cast< size_t >(schema.arities()[static_cast< size_t >(relation_id)])) {
      throw std::invalid_argument("Flat relation emission arity does not match compiled schema");
   }
   relations.emit(relation_id, args);
}

void FlatGraphContext::emit_projection(
   size_t projection_id,
   std::span< const int64_t > source_args
) const
{
   if(projection_id >= projections.size()) {
      throw std::invalid_argument("Flat relation projection id is out of range");
   }
   const auto& projection = projections[projection_id];
   if(source_args.size()
      != static_cast< size_t >(
         schema.arities()[static_cast< size_t >(projection.source_relation_id)]
      )) {
      throw std::invalid_argument("Flat relation projection source arity does not match schema");
   }
   auto output = projection.project(source_args, nodes);
   const auto output_arity = schema.arities()[static_cast< size_t >(projection.output_relation_id)];
   if(output.size() != static_cast< size_t >(output_arity)) {
      throw std::logic_error("Compiled flat relation projection output arity is inconsistent");
   }
   relations.emit(projection.output_relation_id, output);
}

FlatEmitterComponent::~FlatEmitterComponent() = default;

void FlatFieldWriter::validate(std::string_view key, GraphFieldDType dtype) const
{
   const auto& entry = find_field(fields_, key);
   if(entry.owner != owner_) {
      throw std::invalid_argument(
         "Flat graph field '" + std::string(key) + "' is owned by '" + entry.owner + "'"
      );
   }
   if(entry.spec.dtype != dtype) {
      throw std::invalid_argument("Flat graph field '" + std::string(key) + "' has wrong dtype");
   }
}

void FlatFieldWriter::set(std::string_view key, std::span< const int64_t > values) const
{
   validate(key, GraphFieldDType::I64);
   builder_.set_field(std::string(key), values);
}

void FlatFieldWriter::set(std::string_view key, std::span< const float > values) const
{
   validate(key, GraphFieldDType::F32);
   builder_.set_field(std::string(key), values);
}

void CompiledFlatPlan::configure_builder(BatchBuilder& builder) const
{
   set_flat_graph_attrs(builder, schema_.as_metadata(), config_.graph_config);
   for(const auto& spec : node_schema_.specs()) {
      builder.set_node_feature_dim(spec.name, spec.feature_dim);
   }
   for(const auto& field : fields_) {
      builder.register_field(field.key, field.spec);
   }
   if(config_.track_relation_instances or not schema_.names().empty()) {
      register_relation_runtime_fields(builder, schema_.size(), config_.relation_args_node_type);
   }
}

void CompiledFlatPlan::encode(const FlatInputView& input, BatchBuilder& builder) const
{
   configure_builder(builder);

   FlatNodePlanBuilder node_builder(node_schema_);
   for(const auto& component : components_) {
      component->plan_graph(input, node_builder);
   }
   auto node_plan = std::move(node_builder).finish();
   for(size_t node_type = 0; node_type < node_schema_.size(); ++node_type) {
      const auto& spec = node_schema_.spec(static_cast< FlatNodeTypeId >(node_type));
      builder.add_nodes(spec.name, node_plan.count(static_cast< FlatNodeTypeId >(node_type)));
      if(spec.export_names
         and not node_plan.names(static_cast< FlatNodeTypeId >(node_type)).empty()) {
         builder.set_node_names(
            spec.name, node_plan.names(static_cast< FlatNodeTypeId >(node_type))
         );
      }
   }

   FlatRelationSink sink(schema_.size(), config_.track_relation_instances);
   FlatGraphContext context{input, schema_, node_plan, sink, projections_};
   for(const auto& component : components_) {
      component->prepare_graph(input, context);
   }
   for(const auto& component : components_) {
      component->emit(input, context);
   }
   for(const auto& component : components_) {
      FlatFieldWriter writer(builder, fields_, component->name());
      component->write_fields(context, writer);
   }

   const auto& relation_counts = sink.relation_counts();
   const auto& relation_args = sink.relation_args();
   const auto relation_instance_count = sink.relation_instance_count();
   builder.set_field(
      std::string(kRelationInstanceSizesField), std::span{&relation_instance_count, 1}
   );
   builder.set_field(std::string(kRelationCountsField), relation_counts);
   builder.set_field(std::string(kRelationArgsField), relation_args);
   builder.next_graph();
}

BatchBuilder::BatchEncoding CompiledFlatPlan::encode(const FlatInputView& input) const
{
   BatchBuilder builder;
   encode(input, builder);
   auto encoding = builder.build();
   finalize_batch_encoding(encoding);
   return encoding;
}

BatchBuilder::BatchEncoding CompiledFlatPlan::encode_batch(
   std::span< const FlatInputView > inputs
) const
{
   BatchBuilder builder;
   for(const auto& input : inputs) {
      encode(input, builder);
   }
   auto encoding = builder.build();
   finalize_batch_encoding(encoding);
   return encoding;
}

void CompiledFlatPlan::finalize_batch_encoding(BatchBuilder::BatchEncoding& encoding) const
{
   if(config_.pack_relation_args_relation_major) {
      pack_flat_relation_args_relation_major(encoding, schema_.arities());
   }
}

BatchBuilder::BatchEncoding FlatBatchRuntime::encode(const FlatInputView& input) const
{
   return plan_.encode(input);
}

BatchBuilder::BatchEncoding FlatBatchRuntime::encode_batch(
   std::span< const FlatInputView > inputs
) const
{
   return plan_.encode_batch(inputs);
}

void FlatBatchRuntime::encode(const FlatInputView& input, BatchBuilder& builder) const
{
   plan_.encode(input, builder);
}

void FlatBatchRuntime::finalize_batch_encoding(BatchBuilder::BatchEncoding& encoding) const
{
   plan_.finalize_batch_encoding(encoding);
}

void FlatEncoderPlan::add_component(std::shared_ptr< FlatEmitterComponent > component)
{
   if(not component) {
      throw std::invalid_argument("Flat composition component must not be null");
   }
   const auto name = component->name();
   if(name.empty()) {
      throw std::invalid_argument("Flat composition component name must not be empty");
   }
   if(std::ranges::any_of(components_, [&](const auto& existing) {
         return existing->name() == name;
      })) {
      throw std::invalid_argument("Flat composition component names must be unique");
   }
   components_.push_back(std::move(component));
}

CompiledFlatPlan FlatEncoderPlan::compile(FlatCompositionConfig config) const
{
   if(components_.empty()) {
      throw std::invalid_argument("Flat composition plan declares no components");
   }

   FlatSchemaPlanBuilder schema_builder;
   std::vector< FlatFieldPlanEntry > fields;
   std::unordered_map< std::string, std::string > field_owners;
   for(const auto& component : components_) {
      component->declare_schema(schema_builder);
      FlatFieldPlanBuilder field_builder(std::string(component->name()));
      component->declare_fields(field_builder);
      for(const auto& entry : field_builder.entries()) {
         if(const auto owner_it = field_owners.find(entry.key); owner_it != field_owners.end()) {
            throw std::invalid_argument(
               "Flat graph field '" + entry.key + "' declared by both '" + owner_it->second
               + "' and '" + entry.owner + "'"
            );
         }
         field_owners.emplace(entry.key, entry.owner);
         fields.push_back(entry);
      }
   }

   auto schema = std::move(schema_builder).finalize_schema(config);
   auto node_schema = std::move(schema_builder).finalize_nodes();
   if(config.relation_args_node_type.empty()
      or not node_schema.try_id_for(config.relation_args_node_type).has_value()) {
      throw std::invalid_argument(
         "Flat composition relation_args_node_type is not declared: "
         + config.relation_args_node_type
      );
   }

   std::vector< CompiledFlatRelationProjection > projections;
   projections.reserve(schema_builder.projections().size());
   for(const auto& declaration : schema_builder.projections()) {
      const auto source_id = schema.id_for(declaration.source_relation);
      const auto output_id = schema.id_for(declaration.output_relation);
      const auto source_arity = schema.arities()[static_cast< size_t >(source_id)];
      const auto output_arity = schema.arities()[static_cast< size_t >(output_id)];
      if(declaration.slots.size() != static_cast< size_t >(output_arity)) {
         throw std::invalid_argument(
            "Flat relation projection slot count does not match output arity"
         );
      }
      for(const auto& slot : declaration.slots) {
         if(slot.kind == FlatSlotResolverKind::source_slot
            and (slot.source_slot < 0 or slot.source_slot >= source_arity)) {
            throw std::invalid_argument("Flat relation projection source slot is out of range");
         }
         if(slot.kind == FlatSlotResolverKind::node) {
            validate_node_type_id(node_schema, slot.node.type);
         }
      }
      projections.push_back(
         CompiledFlatRelationProjection{source_id, output_id, declaration.slots}
      );
   }

   CompiledFlatPlan compiled;
   compiled.schema_ = std::move(schema);
   compiled.node_schema_ = std::move(node_schema);
   compiled.fields_ = std::move(fields);
   compiled.components_ = components_;
   compiled.projections_ = std::move(projections);
   compiled.config_ = std::move(config);
   return compiled;
}

}  // namespace mifrost
