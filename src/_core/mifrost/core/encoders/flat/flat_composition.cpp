#include "flat_composition.hpp"

#include <algorithm>
#include <array>
#include <iterator>
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
   if(relation_count > static_cast< size_t >(std::numeric_limits< int >::max())) {
      throw std::invalid_argument("Flat composition relation count exceeds graph-field dimension");
   }
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

bool is_reserved_flat_graph_attr(std::string_view key)
{
   constexpr std::array< std::string_view, 20 > reserved = {
      kFlatEntityTypeAttr,         kIncludeLGANEdgesAttr,
      kTargetSourcesAttr,          kLGANAnchorSourcesAttr,
      kEntityRoleNamesAttr,        kRelationNamesAttr,
      kRelationAritiesAttr,        kRelationSourcesAttr,
      kRelationLogicalAritiesAttr, kRelationEncodedAritiesAttr,
      kRelationSlotRolesAttr,      kRelationSlotRoleOffsetsAttr,
      kSlotRoleNamesAttr,          kTargetEntityGroupsAttr,
      kTargetSymbolPrefixAttr,     kLGANTNEdgePosAttr,
      kLGANNNEdgePosAttr,          kLGANRREdgePosAttr,
      kRelationArgsLayoutAttr,     kUsePredicateVirtualNodesAttr,
   };
   return std::ranges::find(reserved, key) != reserved.end();
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

uint32_t flat_external_mode_missing_components(FlatExternalMode mode, uint32_t available_components)
{
   return flat_external_mode_contract(mode).required_components & ~available_components;
}

bool flat_external_mode_satisfied(FlatExternalMode mode, uint32_t available_components)
{
   return flat_external_mode_missing_components(mode, available_components) == 0U;
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
   if(schema_ == nullptr) {
      throw std::logic_error("Flat node plan is not initialized");
   }
   validate_node_type_id(*schema_, type);
   return static_cast< int64_t >(names_[static_cast< size_t >(type)].size());
}

const std::vector< std::string >& FlatNodePlan::names(FlatNodeTypeId type) const
{
   if(schema_ == nullptr) {
      throw std::logic_error("Flat node plan is not initialized");
   }
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

void FlatNodeFeaturePlanBuilder::register_feature(
   std::string node_type,
   std::string attr,
   int feature_dim
)
{
   if(node_type.empty() or attr.empty()) {
      throw std::invalid_argument("Flat node feature node type and attr must not be empty");
   }
   if(feature_dim <= 0) {
      throw std::invalid_argument("Flat node feature dimension must be positive");
   }
   std::string key = node_type + "/" + attr;
   if(const auto it = index_by_key_.find(key); it != index_by_key_.end()) {
      const auto& existing = entries_[it->second];
      if(existing.feature_dim != feature_dim) {
         throw std::invalid_argument("Flat node feature was redeclared with a different dimension");
      }
      return;
   }
   index_by_key_.emplace(key, entries_.size());
   entries_.push_back(
      FlatNodeFeaturePlanEntry{
         .node_type = std::move(node_type),
         .attr = std::move(attr),
         .feature_dim = feature_dim,
         .owner = owner_,
      }
   );
}

void FlatMetadataPlanBuilder::claim_object_names()
{
   if(object_names_claimed_) {
      throw std::invalid_argument(
         "Flat object-name metadata was claimed twice by component '" + owner_ + "'"
      );
   }
   object_names_claimed_ = true;
}

void FlatMetadataPlanBuilder::claim_graph_attr(std::string key)
{
   if(key.empty()) {
      throw std::invalid_argument("Flat graph metadata key must not be empty");
   }
   if(not graph_attrs_.emplace(key, owner_).second) {
      throw std::invalid_argument("Flat graph metadata key was claimed more than once");
   }
}

void FlatSchemaPlanBuilder::register_relation(
   RelationKey key,
   FlatTupleLayout layout,
   RelationUsage usage
)
{
   relation_schema_.register_relation(std::move(key), std::move(layout), usage);
}

void FlatSchemaPlanBuilder::register_relation_alias(RelationKey alias, RelationKey target)
{
   if(alias == target) {
      throw std::invalid_argument("Flat relation alias must differ from its target");
   }
   if(std::ranges::any_of(relation_aliases_, [&](const auto& existing) {
         return existing.alias == alias;
      })) {
      throw std::invalid_argument("Flat relation alias was declared more than once");
   }
   relation_aliases_.push_back(
      FlatRelationAlias{.alias = std::move(alias), .target = std::move(target)}
   );
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

int FlatGraphContext::relation_id(const RelationKey& key) const
{
   if(const auto id = schema.try_id_for(key); id.has_value()) {
      return *id;
   }
   const auto it = std::ranges::find(relation_aliases, key, &FlatRelationAlias::alias);
   if(it != relation_aliases.end()) {
      return it->target_relation_id;
   }
   throw std::invalid_argument("Unknown flat relation key");
}

void FlatGraphContext::emit(int relation_id, std::span< const int64_t > args) const
{
   if(relation_id < 0 or static_cast< size_t >(relation_id) >= schema.arities().size()) {
      throw std::logic_error("Flat relation schema has no arity for relation id");
   }
   if(args.size() != static_cast< size_t >(schema.arities()[static_cast< size_t >(relation_id)])) {
      throw std::invalid_argument("Flat relation emission arity does not match compiled schema");
   }
   relations.emit(relation_id, args);
}

void FlatGraphContext::emit(const RelationKey& key, std::span< const int64_t > args) const
{
   emit(relation_id(key), args);
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

void FlatNodeFeatureWriter::set(
   std::string_view node_type,
   std::string_view attr,
   std::span< const float > values
) const
{
   const auto plan_it = std::ranges::find(
      plan_, std::pair{std::string_view(node_type), std::string_view(attr)}, [](const auto& entry) {
         return std::pair{std::string_view(entry.node_type), std::string_view(entry.attr)};
      }
   );
   if(plan_it == plan_.end() or plan_it->owner != owner_) {
      throw std::invalid_argument(
         "Flat node feature '" + std::string(node_type) + "/" + std::string(attr)
         + "' is not owned by component '" + owner_ + "'"
      );
   }
   const std::string write_key = std::string(node_type) + "/" + std::string(attr);
   if(not written_.insert(write_key).second) {
      throw std::invalid_argument("Flat node feature was written more than once per graph");
   }
   const auto type_id = schema_.id_for(node_type);
   const auto& spec = schema_.spec(type_id);
   if(spec.feature_dim != plan_it->feature_dim) {
      throw std::logic_error("Flat node feature plan dimension does not match node schema");
   }
   const auto count = nodes_.count(type_id);
   const auto expected = static_cast< size_t >(count) * static_cast< size_t >(spec.feature_dim);
   if(values.size() != expected) {
      throw std::invalid_argument(
         "Flat node feature writer received the wrong value count for node type '"
         + std::string(node_type) + "'"
      );
   }
   builder_.add_node_features(std::string(node_type), std::string(attr), values, spec.feature_dim);
}

FlatEmitterComponent::~FlatEmitterComponent() = default;

int FlatCompositionInputBuilder::relation_id(const RelationKey& key) const
{
   if(const auto id = schema_.try_id_for(key); id.has_value()) {
      return *id;
   }
   const auto it = std::ranges::find(aliases_, key, &FlatRelationAlias::alias);
   if(it != aliases_.end()) {
      return it->target_relation_id;
   }
   throw std::invalid_argument("Flat composition relation key is undeclared");
}

void FlatCompositionInputBuilder::add_object(std::string name)
{
   if(name.empty()) {
      throw std::invalid_argument("Flat composition object name must not be empty");
   }
   if(std::ranges::find(input_.objects, name) != input_.objects.end()) {
      throw std::invalid_argument("Flat composition object names must be unique");
   }
   input_.objects.push_back(std::move(name));
}

void FlatCompositionInputBuilder::add_node(std::string node_type, std::string key)
{
   if(node_type.empty() or key.empty()) {
      throw std::invalid_argument("Flat composition node type and key must not be empty");
   }
   if(std::ranges::any_of(input_.nodes, [&](const auto& record) {
         return record.node_type == node_type and record.key == key;
      })) {
      throw std::invalid_argument("Flat composition node keys must be unique per node type");
   }
   input_.nodes.push_back(
      FlatCompositionNodeRecord{.node_type = std::move(node_type), .key = std::move(key)}
   );
}

void FlatCompositionInputBuilder::add_relation(
   int relation_id,
   std::span< const int64_t > args,
   std::string component
)
{
   if(relation_id < 0 or static_cast< size_t >(relation_id) >= schema_.arities().size()) {
      throw std::invalid_argument("Flat composition relation id is out of range");
   }
   if(args.size() != static_cast< size_t >(schema_.arities()[static_cast< size_t >(relation_id)])) {
      throw std::invalid_argument("Flat composition relation arguments have wrong arity");
   }
   input_.relations.push_back(
      FlatCompositionRelationRecord{
         .relation_id = relation_id,
         .args = std::vector< int64_t >(args.begin(), args.end()),
         .component = std::move(component),
      }
   );
}

void FlatCompositionInputBuilder::add_relation(
   const RelationKey& key,
   std::span< const int64_t > args,
   std::string component
)
{
   add_relation(relation_id(key), args, std::move(component));
}

void FlatCompositionInputBuilder::set_field(std::string key, NumericColumnData values)
{
   if(key.empty()) {
      throw std::invalid_argument("Flat composition field key must not be empty");
   }
   if(std::ranges::any_of(input_.fields, [&](const auto& record) { return record.key == key; })) {
      throw std::invalid_argument("Flat composition field keys must be unique");
   }
   input_.fields.push_back(
      FlatCompositionFieldRecord{.key = std::move(key), .values = std::move(values)}
   );
}

void FlatMetadataWriter::set_object_names(std::vector< std::string > names) const
{
   if(not plan_.object_names_owner.has_value() or *plan_.object_names_owner != owner_) {
      throw std::invalid_argument(
         "Flat object-name metadata is not owned by component '" + owner_ + "'"
      );
   }
   builder_.set_object_names(std::move(names));
}

void FlatMetadataWriter::set_graph_attr(
   std::string_view key,
   BatchBuilder::GraphAttrValue value
) const
{
   const auto it = plan_.graph_attr_owners.find(std::string(key));
   if(it == plan_.graph_attr_owners.end() or it->second != owner_) {
      throw std::invalid_argument(
         "Flat graph metadata key '" + std::string(key) + "' is not owned by component '" + owner_
         + "'"
      );
   }
   std::visit(
      [&](auto&& typed_value) {
         builder_.set_graph_attr(
            std::string(key), std::forward< decltype(typed_value) >(typed_value)
         );
      },
      std::move(value)
   );
}

void FlatMetadataWriter::set_graph_attr(std::string_view key, int64_t value) const
{
   set_graph_attr(key, BatchBuilder::GraphAttrValue{value});
}

void FlatMetadataWriter::set_graph_attr(std::string_view key, std::string value) const
{
   set_graph_attr(key, BatchBuilder::GraphAttrValue{std::move(value)});
}

void FlatMetadataWriter::set_graph_attr(std::string_view key, std::vector< int64_t > value) const
{
   set_graph_attr(key, BatchBuilder::GraphAttrValue{std::move(value)});
}

void FlatMetadataWriter::set_graph_attr(
   std::string_view key,
   std::vector< std::string > value
) const
{
   set_graph_attr(key, BatchBuilder::GraphAttrValue{std::move(value)});
}

FlatObjectNodeComponent::FlatObjectNodeComponent(
   std::string component_name,
   std::string node_type,
   FlatNodeKind kind,
   int feature_dim,
   bool export_names
)
    : component_name_(std::move(component_name)),
      node_type_(std::move(node_type)),
      kind_(kind),
      feature_dim_(feature_dim),
      export_names_(export_names)
{
   if(component_name_.empty()) {
      throw std::invalid_argument("Flat object component name must not be empty");
   }
   if(node_type_.empty()) {
      throw std::invalid_argument("Flat object component node type must not be empty");
   }
   if(feature_dim_ <= 0) {
      throw std::invalid_argument("Flat object component feature dimension must be positive");
   }
}

void FlatObjectNodeComponent::declare_schema(FlatSchemaPlanBuilder& builder) const
{
   (void) builder.declare_node_type(node_type_, kind_, feature_dim_, export_names_);
}

void FlatObjectNodeComponent::plan_graph(
   const FlatInputView& input,
   FlatNodePlanBuilder& builder
) const
{
   const auto& composition = input.get< FlatCompositionInput >();
   for(const auto& object : composition.objects) {
      (void) builder.add_node(node_type_, object);
   }
}

void FlatObjectNodeComponent::declare_node_features(FlatNodeFeaturePlanBuilder& builder) const
{
   builder.register_feature(node_type_, "x", feature_dim_);
}

void FlatObjectNodeComponent::declare_metadata(FlatMetadataPlanBuilder& builder) const
{
   if(export_names_) {
      builder.claim_object_names();
   }
}

void FlatObjectNodeComponent::write_node_features(
   const FlatGraphContext& context,
   FlatNodeFeatureWriter& writer
) const
{
   const auto type_id = context.nodes.schema().id_for(node_type_);
   const auto& spec = context.nodes.schema().spec(type_id);
   std::vector< float > zeros(
      static_cast< size_t >(context.nodes.count(type_id)) * static_cast< size_t >(spec.feature_dim),
      0.0F
   );
   writer.set(node_type_, "x", zeros);
}

void FlatObjectNodeComponent::write_metadata(
   const FlatGraphContext& context,
   FlatMetadataWriter& writer
) const
{
   if(export_names_) {
      writer.set_object_names(context.input.get< FlatCompositionInput >().objects);
   }
}

FlatNodeRecordComponent::FlatNodeRecordComponent(
   std::string component_name,
   std::string node_type,
   FlatNodeKind kind,
   int feature_dim,
   bool export_names
)
    : component_name_(std::move(component_name)),
      node_type_(std::move(node_type)),
      kind_(kind),
      feature_dim_(feature_dim),
      export_names_(export_names)
{
   if(component_name_.empty()) {
      throw std::invalid_argument("Flat node-record component name must not be empty");
   }
   if(node_type_.empty()) {
      throw std::invalid_argument("Flat node-record component node type must not be empty");
   }
   if(feature_dim_ <= 0) {
      throw std::invalid_argument("Flat node-record component feature dimension must be positive");
   }
}

void FlatNodeRecordComponent::declare_schema(FlatSchemaPlanBuilder& builder) const
{
   (void) builder.declare_node_type(node_type_, kind_, feature_dim_, export_names_);
}

void FlatNodeRecordComponent::plan_graph(
   const FlatInputView& input,
   FlatNodePlanBuilder& builder
) const
{
   const auto& composition = input.get< FlatCompositionInput >();
   for(const auto& record : composition.nodes) {
      if(record.node_type == node_type_) {
         (void) builder.add_node(node_type_, record.key);
      }
   }
}

void FlatNodeRecordComponent::declare_node_features(FlatNodeFeaturePlanBuilder& builder) const
{
   builder.register_feature(node_type_, "x", feature_dim_);
}

void FlatNodeRecordComponent::write_node_features(
   const FlatGraphContext& context,
   FlatNodeFeatureWriter& writer
) const
{
   const auto type_id = context.nodes.schema().id_for(node_type_);
   const auto& spec = context.nodes.schema().spec(type_id);
   std::vector< float > zeros(
      static_cast< size_t >(context.nodes.count(type_id)) * static_cast< size_t >(spec.feature_dim),
      0.0F
   );
   writer.set(node_type_, "x", zeros);
}

FlatRelationEmitterComponent::FlatRelationEmitterComponent(
   std::string component_name,
   std::vector< FlatCompositionRelationSpec > relations
)
    : component_name_(std::move(component_name)), relations_(std::move(relations))
{
   if(component_name_.empty()) {
      throw std::invalid_argument("Flat relation component name must not be empty");
   }
   if(relations_.empty()) {
      throw std::invalid_argument("Flat relation component must declare at least one relation");
   }
}

void FlatRelationEmitterComponent::declare_schema(FlatSchemaPlanBuilder& builder) const
{
   for(const auto& relation : relations_) {
      builder.register_relation(relation.key, relation.layout, relation.usage);
   }
}

void FlatRelationEmitterComponent::emit(const FlatInputView& input, FlatGraphContext& context) const
{
   const auto& composition = input.get< FlatCompositionInput >();
   for(const auto& relation : composition.relations) {
      if(not relation.component.empty() and relation.component != component_name_) {
         continue;
      }
      context.emit(relation.relation_id, relation.args);
   }
}

FlatFieldEmitterComponent::FlatFieldEmitterComponent(
   std::string component_name,
   std::vector< FieldDeclaration > fields
)
    : component_name_(std::move(component_name)), fields_(std::move(fields))
{
   if(component_name_.empty()) {
      throw std::invalid_argument("Flat field component name must not be empty");
   }
   if(fields_.empty()) {
      throw std::invalid_argument("Flat field component must declare at least one field");
   }
   for(auto it = fields_.begin(); it != fields_.end(); ++it) {
      if(std::ranges::find(
            std::next(it), fields_.end(), it->first, [](const auto& field) { return field.first; }
         )
         != fields_.end()) {
         throw std::invalid_argument("Flat field component field keys must be unique");
      }
   }
}

void FlatFieldEmitterComponent::declare_fields(FlatFieldPlanBuilder& builder) const
{
   for(const auto& [key, spec] : fields_) {
      builder.register_field(key, spec);
   }
}

void FlatFieldEmitterComponent::write_fields(
   const FlatGraphContext& context,
   FlatFieldWriter& writer
) const
{
   const auto& composition = context.input.get< FlatCompositionInput >();
   for(const auto& [key, spec] : fields_) {
      const auto it = std::ranges::find(composition.fields, key, &FlatCompositionFieldRecord::key);
      if(it == composition.fields.end()) {
         throw std::invalid_argument(
            "Flat composition input is missing declared field '" + key + "'"
         );
      }
      if(std::ranges::find(
            std::next(it), composition.fields.end(), key, &FlatCompositionFieldRecord::key
         )
         != composition.fields.end()) {
         throw std::invalid_argument(
            "Flat composition input declares field '" + key + "' more than once"
         );
      }
      if(spec.dtype == GraphFieldDType::I64) {
         const auto* values = std::get_if< std::vector< int64_t > >(&it->values);
         if(values == nullptr) {
            throw std::invalid_argument("Flat composition field '" + key + "' has wrong dtype");
         }
         writer.set(key, *values);
      } else {
         const auto* values = std::get_if< std::vector< float > >(&it->values);
         if(values == nullptr) {
            throw std::invalid_argument("Flat composition field '" + key + "' has wrong dtype");
         }
         writer.set(key, *values);
      }
   }
}

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
   set_flat_graph_attrs(builder, schema_plan_.relation_schema.as_metadata(), config_.graph_config);
   for(const auto& field : schema_plan_.fields) {
      builder.register_field(field.key, field.spec);
   }
   if(config_.track_relation_instances or not schema_plan_.relation_schema.names().empty()) {
      register_relation_runtime_fields(
         builder, schema_plan_.relation_schema.size(), config_.relation_args_node_type
      );
   }
}

void CompiledFlatPlan::encode_graph(const FlatInputView& input, BatchBuilder& builder) const
{
   FlatNodePlanBuilder node_builder(schema_plan_.node_schema);
   for(const auto& component : components_) {
      component->plan_graph(input, node_builder);
   }
   auto node_plan = std::move(node_builder).finish();
   for(size_t node_type = 0; node_type < schema_plan_.node_schema.size(); ++node_type) {
      const auto& spec = schema_plan_.node_schema.spec(static_cast< FlatNodeTypeId >(node_type));
      builder.set_node_feature_dim(spec.name, spec.feature_dim);
      builder.add_nodes(spec.name, node_plan.count(static_cast< FlatNodeTypeId >(node_type)));
      if(spec.export_names
         and not node_plan.names(static_cast< FlatNodeTypeId >(node_type)).empty()) {
         builder.set_node_names(
            spec.name, node_plan.names(static_cast< FlatNodeTypeId >(node_type))
         );
      }
   }

   FlatRelationSink sink(schema_plan_.relation_schema.size(), config_.track_relation_instances);
   FlatGraphContext context{
      input,
      schema_plan_.relation_schema,
      node_plan,
      sink,
      schema_plan_.projections,
      schema_plan_.relation_aliases,
   };
   for(const auto& component : components_) {
      FlatNodeFeatureWriter writer(
         builder, schema_plan_.node_schema, node_plan, schema_plan_.node_features, component->name()
      );
      component->write_node_features(context, writer);
   }
   for(const auto& component : components_) {
      component->prepare_graph(input, context);
   }
   for(const auto& component : components_) {
      component->emit(input, context);
   }
   for(const auto& component : components_) {
      FlatFieldWriter writer(builder, schema_plan_.fields, component->name());
      component->write_fields(context, writer);
   }
   for(const auto& component : components_) {
      FlatMetadataWriter writer(builder, schema_plan_.metadata, component->name());
      component->write_metadata(context, writer);
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

void CompiledFlatPlan::encode(const FlatInputView& input, BatchBuilder& builder) const
{
   configure_builder(builder);
   encode_graph(input, builder);
}

BatchBuilder::BatchEncoding CompiledFlatPlan::encode(const FlatInputView& input) const
{
   BatchBuilder builder;
   configure_builder(builder);
   encode_graph(input, builder);
   auto encoding = builder.build();
   finalize_batch_encoding(encoding);
   return encoding;
}

BatchBuilder::BatchEncoding CompiledFlatPlan::encode_batch(
   std::span< const FlatInputView > inputs
) const
{
   BatchBuilder builder;
   configure_builder(builder);
   for(const auto& input : inputs) {
      encode_graph(input, builder);
   }
   auto encoding = builder.build();
   finalize_batch_encoding(encoding);
   return encoding;
}

void CompiledFlatPlan::finalize_batch_encoding(BatchBuilder::BatchEncoding& encoding) const
{
   FlatRelationMajorWriter writer(
      schema_plan_.relation_schema.arities(), config_.pack_relation_args_relation_major
   );
   writer.finalize(encoding);
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

namespace {

FlatBatchParityResult parity_mismatch(std::string path)
{
   return FlatBatchParityResult{.equal = false, .mismatch = std::move(path)};
}

template < typename Map >
std::set< std::string > map_keys(const Map& values)
{
   std::set< std::string > keys;
   for(const auto& [key, _] : values) {
      keys.insert(key);
   }
   return keys;
}

template < typename Map, typename ValueEqual >
std::optional< std::string > compare_string_maps(
   const Map& expected,
   const Map& actual,
   std::string_view prefix,
   ValueEqual&& value_equal
)
{
   const auto keys = [&] {
      auto result = map_keys(expected);
      const auto actual_keys = map_keys(actual);
      result.insert(actual_keys.begin(), actual_keys.end());
      return result;
   }();
   for(const auto& key : keys) {
      const auto expected_it = expected.find(key);
      const auto actual_it = actual.find(key);
      if(expected_it == expected.end() or actual_it == actual.end()) {
         return std::string(prefix) + "[" + key + "]";
      }
      if(not value_equal(expected_it->second, actual_it->second)) {
         return std::string(prefix) + "[" + key + "]";
      }
   }
   return std::nullopt;
}

bool equal_graph_field(const GraphField& expected, const GraphField& actual)
{
   return expected.spec == actual.spec and expected.values == actual.values
          and expected.ptr == actual.ptr and expected.pending == actual.pending;
}

bool equal_column(const BatchBuilder::Column& expected, const BatchBuilder::Column& actual)
{
   return expected.dim == actual.dim and expected.data == actual.data;
}

bool equal_schema(const Schema& expected, const Schema& actual)
{
   const auto equal_node_tensors = [&] {
      return expected.node_tensors.size() == actual.node_tensors.size()
             and std::ranges::equal(
                expected.node_tensors, actual.node_tensors, [](const auto& lhs, const auto& rhs) {
                   return as_tuple(lhs) == as_tuple(rhs);
                }
             );
   };
   return expected.version == actual.version and expected.graph_kind == actual.graph_kind
          and expected.node_types == actual.node_types and expected.edge_types == actual.edge_types
          and equal_node_tensors() and expected.edge_tensors == actual.edge_tensors
          and expected.graph_tensors == actual.graph_tensors and expected.flags == actual.flags;
}

bool equal_lazy_target_names(
   const BatchBuilder::BatchEncoding& expected,
   const BatchBuilder::BatchEncoding& actual
)
{
   if(expected.lazy_target_name_strings != actual.lazy_target_name_strings
      or expected.lazy_target_name_batches.size() != actual.lazy_target_name_batches.size()) {
      return false;
   }
   for(size_t i = 0; i < expected.lazy_target_name_batches.size(); ++i) {
      const auto& expected_batch = expected.lazy_target_name_batches[i];
      const auto& actual_batch = actual.lazy_target_name_batches[i];
      if(expected_batch == nullptr or actual_batch == nullptr) {
         if(expected_batch != actual_batch) {
            return false;
         }
         continue;
      }
      if(expected_batch->materialize() != actual_batch->materialize()) {
         return false;
      }
   }
   return true;
}

}  // namespace

FlatBatchParityResult compare_flat_batch_encodings(
   const BatchBuilder::BatchEncoding& expected,
   const BatchBuilder::BatchEncoding& actual
)
{
   if(expected.num_graphs != actual.num_graphs) {
      return parity_mismatch("num_graphs");
   }
   if(expected.graph_kind != actual.graph_kind) {
      return parity_mismatch("graph_kind");
   }
   if(expected.object_names != actual.object_names) {
      return parity_mismatch("object_names");
   }
   if(expected.node_names != actual.node_names) {
      return parity_mismatch("node_names");
   }
   if(expected.node_feature_dims != actual.node_feature_dims) {
      return parity_mismatch("node_feature_dims");
   }
   if(expected.node_counts != actual.node_counts) {
      return parity_mismatch("node_counts");
   }
   if(expected.ptrs != actual.ptrs) {
      return parity_mismatch("ptrs");
   }
   if(expected.schema_flags != actual.schema_flags) {
      return parity_mismatch("schema_flags");
   }
   if(not equal_schema(expected.schema, actual.schema)) {
      return parity_mismatch("schema");
   }
   if(not equal_lazy_target_names(expected, actual)) {
      return parity_mismatch("lazy_target_names");
   }

   if(const auto mismatch = compare_string_maps(
         expected.columns, actual.columns, "columns", equal_column
      );
      mismatch.has_value()) {
      return parity_mismatch(*mismatch);
   }
   if(const auto mismatch = compare_string_maps(
         expected.graph_attrs,
         actual.graph_attrs,
         "graph_attrs",
         [](const auto& lhs, const auto& rhs) { return lhs == rhs; }
      );
      mismatch.has_value()) {
      return parity_mismatch(*mismatch);
   }
   if(const auto mismatch = compare_string_maps(
         expected.graph_fields, actual.graph_fields, "graph_fields", equal_graph_field
      );
      mismatch.has_value()) {
      return parity_mismatch(*mismatch);
   }
   return FlatBatchParityResult{};
}

void FlatRelationMajorWriter::finalize(BatchBuilder::BatchEncoding& encoding) const
{
   if(enabled_) {
      pack_flat_relation_args_relation_major(encoding, relation_arities_);
   }
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
   std::vector< FlatNodeFeaturePlanEntry > node_features;
   std::unordered_map< std::string, FlatNodeFeaturePlanEntry > node_feature_declarations;
   FlatMetadataPlan metadata_plan;
   for(const auto& component : components_) {
      component->declare_schema(schema_builder);
      FlatNodeFeaturePlanBuilder node_feature_builder(std::string(component->name()));
      component->declare_node_features(node_feature_builder);
      for(const auto& entry : node_feature_builder.entries()) {
         const auto key = entry.node_type + "/" + entry.attr;
         if(const auto existing = node_feature_declarations.find(key);
            existing != node_feature_declarations.end()) {
            if(existing->second.feature_dim != entry.feature_dim) {
               throw std::invalid_argument(
                  "Flat node feature '" + key + "' has conflicting dimensions"
               );
            }
            throw std::invalid_argument(
               "Flat node feature '" + key + "' declared by both '" + existing->second.owner
               + "' and '" + entry.owner + "'"
            );
         }
         node_feature_declarations.emplace(key, entry);
         node_features.push_back(entry);
      }
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
      FlatMetadataPlanBuilder metadata_builder(std::string(component->name()));
      component->declare_metadata(metadata_builder);
      if(metadata_builder.claims_object_names()) {
         if(metadata_plan.object_names_owner.has_value()) {
            throw std::invalid_argument(
               "Flat object-name metadata declared by both '" + *metadata_plan.object_names_owner
               + "' and '" + std::string(component->name()) + "'"
            );
         }
         metadata_plan.object_names_owner = std::string(component->name());
      }
      for(const auto& [key, owner] : metadata_builder.graph_attrs()) {
         if(is_reserved_flat_graph_attr(key)) {
            throw std::invalid_argument(
               "Flat graph metadata key '" + key + "' is reserved by the graph schema"
            );
         }
         if(const auto owner_it = metadata_plan.graph_attr_owners.find(key);
            owner_it != metadata_plan.graph_attr_owners.end()) {
            throw std::invalid_argument(
               "Flat graph metadata key '" + key + "' declared by both '" + owner_it->second
               + "' and '" + owner + "'"
            );
         }
         metadata_plan.graph_attr_owners.emplace(key, owner);
      }
   }

   const auto projection_declarations = schema_builder.projections();
   const auto alias_declarations = schema_builder.relation_aliases();
   auto schema = std::move(schema_builder).finalize_schema(config);
   auto node_schema = std::move(schema_builder).finalize_nodes();
   for(const auto& feature : node_features) {
      const auto type_id = node_schema.try_id_for(feature.node_type);
      if(not type_id.has_value()) {
         throw std::invalid_argument(
            "Flat node feature references undeclared node type '" + feature.node_type + "'"
         );
      }
      if(node_schema.spec(*type_id).feature_dim != feature.feature_dim) {
         throw std::invalid_argument(
            "Flat node feature dimension does not match node type '" + feature.node_type + "'"
         );
      }
   }
   if(config.relation_args_node_type.empty()
      or not node_schema.try_id_for(config.relation_args_node_type).has_value()) {
      throw std::invalid_argument(
         "Flat composition relation_args_node_type is not declared: "
         + config.relation_args_node_type
      );
   }
   const auto is_runtime_field = [](std::string_view key) {
      return key == kRelationInstanceSizesField or key == kRelationCountsField
             or key == kRelationArgsField;
   };
   const auto has_declared_field = [&](std::string_view key) {
      return std::ranges::any_of(fields, [&](const auto& field) { return field.key == key; });
   };
   for(const auto& field : fields) {
      if(is_runtime_field(field.key)) {
         throw std::invalid_argument(
            "Flat composition field '" + field.key + "' is reserved for the relation writer"
         );
      }
      if(field.spec.inc.kind == GraphFieldInc::Kind::NODE_OFFSET
         and (field.spec.inc.node_type.empty()
             or not node_schema.try_id_for(field.spec.inc.node_type).has_value())) {
         throw std::invalid_argument(
            "Flat composition field '" + field.key + "' references an undeclared node-offset type '"
            + field.spec.inc.node_type + "'"
         );
      }
      if(field.spec.inc.kind == GraphFieldInc::Kind::FIELD_OFFSET
         and (field.spec.inc.field_key.empty()
             or (not is_runtime_field(field.spec.inc.field_key)
                 and not has_declared_field(field.spec.inc.field_key)))) {
         throw std::invalid_argument(
            "Flat composition field '" + field.key + "' references an undeclared field-offset key '"
            + field.spec.inc.field_key + "'"
         );
      }
   }

   std::vector< FlatRelationAlias > relation_aliases;
   relation_aliases.reserve(alias_declarations.size());
   for(const auto& declaration : alias_declarations) {
      const auto target_id = schema.try_id_for(declaration.target);
      if(not target_id.has_value()) {
         throw std::invalid_argument("Flat relation alias target is undeclared");
      }
      if(const auto canonical_id = schema.try_id_for(declaration.alias);
         canonical_id.has_value() and *canonical_id != *target_id) {
         throw std::invalid_argument(
            "Flat relation alias collides with a different canonical relation"
         );
      }
      relation_aliases.push_back(
         FlatRelationAlias{
            .alias = declaration.alias,
            .target = declaration.target,
            .target_relation_id = *target_id,
         }
      );
   }

   const auto resolve_relation_id = [&](const RelationKey& key) {
      if(const auto id = schema.try_id_for(key); id.has_value()) {
         return *id;
      }
      const auto it = std::ranges::find(relation_aliases, key, &FlatRelationAlias::alias);
      if(it != relation_aliases.end()) {
         return it->target_relation_id;
      }
      throw std::invalid_argument("Flat relation projection references an undeclared relation");
   };

   std::vector< CompiledFlatRelationProjection > projections;
   projections.reserve(projection_declarations.size());
   for(const auto& declaration : projection_declarations) {
      const auto source_id = resolve_relation_id(declaration.source_relation);
      const auto output_id = resolve_relation_id(declaration.output_relation);
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
   compiled.schema_plan_ = FlatSchemaPlan{
      .relation_schema = std::move(schema),
      .node_schema = std::move(node_schema),
      .fields = std::move(fields),
      .node_features = std::move(node_features),
      .projections = std::move(projections),
      .metadata = std::move(metadata_plan),
      .relation_aliases = std::move(relation_aliases),
   };
   compiled.components_ = components_;
   compiled.config_ = std::move(config);
   return compiled;
}

}  // namespace mifrost
