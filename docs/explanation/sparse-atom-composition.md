# Sparse Atom-Composition Encoding

`mifrost.encoders.sparse_atom` is a pure-Python (Stage 1: no native engine
involved) encoder for the sparse atom-composition architecture described in
"Sparse Atom Compositions for Relational STRIPS Message Passing". It stores
one persistent vector per *argument occurrence* of a represented atom, rather
than one vector per object, and exposes sparse pair/witness indices so a
consumer model can compose atoms that share an object pair without
materializing the dense object-pair universe.

The consumer is `relmo.models.SparseAtomCompositionGNN` (in the sibling
`relm` repository); `SparseAtomCompositionEncoder` produces its exact input
contract, `SparseAtomCompositionBatch`. `mifrost.encoders.sparse_atom.validate_sparse_atom_composition`
is a standalone reimplementation of every invariant that consumer's
`prepare()` checks, usable without constructing a model.

## The contract, at a glance

| Field | Shape | Meaning |
| --- | --- | --- |
| `atom_args`, `atom_offsets` | `[I]`, `[Q+1]` | Packed ordered atom argument tuples (CSR) |
| `atom_predicate_ids` | `[Q]` | Emitted predicate id -- meaning depends on `status_encoding` (see below) |
| `atom_channel_ids` | `[Q]` | One of the four frozen channels in `"channel"` status encoding; always `0` in `"vocabulary"` |
| `atom_batch`, `object_batch` | `[Q]`, `[O]` | Graph id per atom / per object |
| `pair_objects` | `[P, 2]` | Ordered, non-diagonal, deduplicated object pairs |
| `pair_support_offsets/atom_ids/i/j` | CSR over `[S]` | Every `(atom, i, j)` contributing to a pair |
| `composition_triplets` | `[K, 3]` | `(target_pair, left_pair, right_pair)` witness rows |
| `atom_pair_ids`, `atom_pair_occurrence_i/j` | `[M]` | Atom-to-pair maps, in global occurrence indices |
| `goal_available` | `[B]` | Per-graph zeta flag |
| `object_carrier_occurrence_ids` | `[O]` | One `object(o)` carrier occurrence per object, ordered by object id |
| `counterpart_occurrence_ids` | `[I]`, optional | R11 exact-tuple exchange, `-1` where none |
| `object_type_ids` | `[O]`, optional | R13 per-object declared PDDL type id; `None` when the backend cannot resolve types |

`Q` = atoms, `I` = occurrences (`sum(arity)`), `O` = objects, `P` = pairs,
`S` = pair-support entries, `K` = witness triplets, `M` = atom-pair mappings,
`B` = graphs in the batch. Everything is a plain `torch.long`/`torch.bool`
tensor; `SparseAtomCompositionEncoding` (the encoder's carrier dataclass) has
exactly these attribute names, so it can be passed directly to
`SparseAtomCompositionGNN(...)`.

Predicate arities aren't carried on the tensor contract at all: they are a
*model constructor* argument (`predicate_arities=...`), fixed once from
`SparseAtomCompositionEncoder.schema` (a `SparseAtomPredicateSchema`) at
construction time, shared by every graph the encoder ever produces. The same
goes for the reported channel cardinality (`num_channels`) -- see
[Status encoding](#status-encoding-channel-vs-vocabulary) below.

## Channel convention

Frozen, do not renumber:

| id | name | meaning |
| -- | ---- | ------- |
| 0 | `state` | current true atom (includes static facts, and every `object(o)` carrier) |
| 1 | `satisfied` | supplied goal atom that is currently true |
| 2 | `unsatisfied` | supplied goal atom that is currently false |
| 3 | `auxiliary` | encoding artefacts that are not themselves facts (only the nullary star's carrier) |

In `"channel"` status encoding, the consumer allocates one wide atom MLP per
`(channel, base_predicate)` pair, so `atom_predicate_ids` **must** be the
base predicate id, not a channel-qualified relation id: `on[state]`,
`on[sat]` and `on[unsat]` all share predicate id `on`, differing only in
`atom_channel_ids`. This is what lets the architecture's `M_state,on`,
`M_sat,on`, `M_unsat,on` split exist without tripling the predicate
vocabulary. In `"vocabulary"` status encoding these four rows fold into the
predicate vocabulary instead -- see below.

## Why `plain` is off

The native flat-encoder family's `GoalDerivation` enum has five values
(`plain`, `satisfied`, `unsatisfied`, `added_satisfied`,
`added_unsatisfied`), and its *library default* is `{plain, satisfied}` --
every goal gets one plain atom, and satisfied goals additionally get a
`satisfied` atom, but there is **no `unsatisfied` atom at all**. That
default is a different encoding, tuned for a different architecture. This
encoder never emits `plain`: every supplied goal atom `p(o)` in the goal
conjunction `G` becomes exactly one status atom, `satisfied` **xor**
`unsatisfied`, depending on whether `p(o)` is already a member of the
current true-atom set `S`:

```text
Atoms_goal(S, G) =
    { p_satisfied(o)   : p(o) in G and p(o) in S }
  U { p_unsatisfied(o) : p(o) in G and p(o) not in S }
```

Using the library default here would silently produce a *plain* goal atom
for every goal and *no* `unsatisfied` atoms at all -- voiding the two-tower
Blocksworld separation the architecture relies on (`on_satisfied`/
`on_unsatisfied` needs both a satisfied *and* an unsatisfied label to exist
for the separation argument to apply). `tests/encoding/test_sparse_atom_composition.py::test_goal_status_split_exactly_one_atom_per_goal`
asserts this directly: goal-derived channels are always a subset of
`{satisfied, unsatisfied}` and their count always equals the number of
supplied goal atoms -- never zero, never doubled.

A satisfied goal keeps its current-fact representation too: `q(o)` being
both true and a goal produces *two* atoms (one `state`, one `satisfied`),
not one atom carrying both roles.

## Per-object carriers (R4)

Every object needs a persistent unary "carrier" atom so `object_readout`
and the exact-tuple exchange have somewhere to attach an object-level
representation even when an object appears in no other atom. That carrier
relation is the backend's own `object` predicate -- not an invented one.
Real PDDL backends already expose it: on `blocks:small`, `StateView(problem).predicates`
includes `PredicateInfo(name='object', arity=1, category='static')`, and
`static_facts` is exactly `[Atom('object', ('a',)), Atom('object', ('b',))]`
-- one `object(o)` static fact per domain object, already part of the
current-atom set `S`. There is no name collision to design around: the
`object` relation the backend exposes *is* the carrier relation the
architecture note writes as `object^aux(o)`, so `build_predicate_schema`
reuses it (appending it, with encoded arity 1, only if a hand-built schema
omits it) instead of minting a second, reserved-looking predicate.

Because `object(o)` is a genuine static fact, its carrier occurrence lives in
the **state** channel, as part of the current atoms `S` -- not a separate
auxiliary channel. `object_carrier_occurrence_ids[o]` points at that
occurrence, one per object, ordered by object id (the consumer asserts this
ordering).

**Fallback.** If a domain or backend does not supply `object(o)` for every
object -- a hand-built test schema, or a future backend with gaps -- the
missing carriers are synthesized, still in the state channel, decided
*per object* rather than per domain: whatever is missing gets synthesized,
so an isolated object with no other atom mentioning it still gets a carrier.
On the two real Blocksworld fixtures exercised by
`tests/encoding/test_sparse_atom_composition_relmo_integration.py`, this
fallback path is never taken -- both backends already cover every object --
so it is verified there via hand-built `Atom` lists instead
(`tests/encoding/test_sparse_atom_composition.py::test_missing_object_carrier_is_synthesized_per_object`).

The one object that *never* gets a real `object(o)` fact is the distinguished
nullary object star (see [Nullary normalization](#nullary-normalization)
below): it is not a real PDDL object, so no backend ever emits
`object(star)`. Its carrier is therefore always synthesized, and it stays in
the **auxiliary** channel, since it is an encoding artefact rather than a
true fact -- the only auxiliary-channel atom this encoder ever emits.

## Nullary normalization

A nullary atom such as `handempty()` has no argument to attach a persistent
occurrence vector to. This encoder ports the native hetero family's
`nullary_object_name`/`add_nullary_predicates` approach rather than the flat
family's `ignore_zero_arity_relations=True` default, which would silently
*drop* `handempty` instead of representing it: every nullary atom becomes a
unary atom on a distinguished auxiliary object,
`mifrost.encoders.sparse_atom.NULLARY_OBJECT_NAME`
(`"![nullary_symbol]!"`, the same literal the hetero family already uses).
The star object:

- is added to a graph's object universe only when that graph actually
  contains a nullary atom (a state with no nullary predicates never carries
  a dangling unused star);
- always gets a synthesized `object(star)` carrier in the auxiliary channel
  (see [Per-object carriers](#per-object-carriers-r4) above);
- participates in pairs and witnesses like any other object once it appears
  in a *non*-nullary atom's arguments too (it never does on its own, since
  every atom mentioning it is now unary).

No arity-0 atom is ever emitted by this encoder. The consumer's own
`prepare()` rejects arity 0 outright, naming this normalization in its error
message, and `validate_sparse_atom_composition` checks the same invariant
standalone.

## Status encoding: channel vs. vocabulary

`build_predicate_schema` (and `SparseAtomCompositionEncoder`) take a
`status_encoding` argument with two values, matching the two ways the
architecture document's baseline comparison represents goal status:

- **`"channel"`** (default): as described above -- `atom_predicate_ids` is
  the base predicate, `atom_channel_ids` distinguishes
  state/satisfied/unsatisfied/auxiliary, and the schema reports
  `num_channels == 4`.
- **`"vocabulary"`**: there is no status *channel* at all. Every
  `(base predicate, channel)` pair gets its own vocabulary entry instead --
  the schema is four times as large -- `atom_channel_ids` is always `0`, and
  `num_channels == 1`. This is how `relmo.models.FlatRelationalGNN`
  represents goal status: it extends the *relation vocabulary* with separate
  relations (`P`, or `P_sat`/`P_unsat`) rather than carrying a status
  channel. Vocabulary-mode names reuse the flat native encoder family's own
  `RelationKey` bracket convention (see
  `src/_core/mifrost/core/encoders/common/relation_key.cpp`): `on[state]`,
  `on[sat]`, `on[unsat]`, `on[g]` (the last for the auxiliary channel, which
  has no flat-family analogue of its own).

Both modes encode *exactly* the same atoms and sparse topology --
`pair_objects`, `composition_triplets`, `atom_pair_ids`,
`object_carrier_occurrence_ids`, everything -- only the predicate/channel
labelling differs; `atom_predicate_ids` is precisely
`schema.relation_id(base_predicate_id, channel)` applied pointwise to the
channel-mode ids. This is what makes vocabulary mode useful: it is what the
architecture document's "identical input encodings across architectures
within each comparison" requirement needs for a like-for-like comparison
against `FlatRelationalGNN`, since both models can then be sized from the
same predicate vocabulary (`predicate_arities`, `num_channels`).

That equivalence has one asymmetry worth knowing before benchmarking with
it: `SparseAtomCompositionGNN`'s `join_sharing` ablation derives a per-pair
join-template key from `atom_channel_ids` in `prepare()` (a pair keys on its
support channel when that support is channel-homogeneous, and on a
distinguished "mixed" key otherwise). In `"vocabulary"` status encoding every
atom's channel is `0`, so every pair collapses onto the same template --
correct, not a bug, since the status distinction already lives in the
predicate id there, but it means the join-sharing ablation is only
*meaningful* in `"channel"` status encoding. Use `"vocabulary"` for a
like-for-like vocabulary comparison against `FlatRelationalGNN`; use
`"channel"` when the join-sharing ablation itself is what is being studied.

## Pairs and witnesses (R7/R8/R9)

For distinct objects `u`, `v`, `pair_objects` holds every ordered pair with
at least one supporting occurrence -- `(u, v)` and `(v, u)` are independent
records, and *both* arise automatically from any single atom that mentions
`u` and `v` at two different argument positions (the position loop is over
all ordered `(i, j)`, not just `i < j`). `composition_triplets` is built by
listing every triangle of the resulting undirected support graph exactly
once via degeneracy ordering (`O(alpha * M)`, the same construction
`relmo`'s own topology-stats triangle counter uses) and expanding each into
its 6 ordered `(target, left, right)` rows. `atom_pair_ids` /
`atom_pair_occurrence_i` / `atom_pair_occurrence_j` are the atom-to-pair maps
`D`+`AGGR^args` needs to distribute pair/composition information back to
occurrence stores; occurrence indices there are always *global*
(`atom_offsets[q] + position]`), never atom-local.

## Batching (R12)

`batch_sparse_atom_encodings` concatenates a list of single-graph (or
already-batched) encodings into one, explicitly rebasing every index space
by that graph's own running totals before concatenating -- objects, atoms,
occurrences, pairs, and pair-support entries each get their own running
offset. No pair or witness triplet can span two input graphs: `pair_objects`
only ever references two objects with equal `object_batch` entries, by
construction (each graph's own `pair_objects` already only references its
own objects, and rebasing preserves that).

## Object types (R13)

The architecture's per-occurrence side information `s_{q,j}` includes
argument object types, and the consumer carrier (`object_type_ids`) has a
slot for them. Whether a backend can supply them turned out to be
asymmetric, not a uniformly unfinished feature:

- **pymimir**: yes. The installed pymimir wrapper's `Object` (and `Type`)
  expose `get_bases()` in addition to `get_index`/`get_name` --
  contradicting an earlier assumption recorded in this codebase that only
  the latter two existed. `Object.get_bases()` returns the object's own
  declared type(s), and `Domain.get_types()` always includes the implicit
  PDDL root type `"object"`, even for domains with no `:types` section at
  all (verified against every fixture under `data/pddl/`).
- **pytyr**: no, genuinely. The `pytyr.formalism.planning` task `StateView`
  wraps is a *translated* representation, and PDDL types are compiled away
  before it exists: its `Domain` has no `get_types()`, and its
  `Object`/`Type` really do expose only `get_index`/`get_name` -- confirmed
  at the C++ layer, not just the binding surface. `tyr::formalism::planning
  ::Object`'s `Data` struct (pytyr's own
  `native/include/tyr/formalism/object_data.hpp`) stores only `index` and
  `name`, nothing else. The type hierarchy does exist earlier, on the raw
  parsed AST (`pypddl.formalism.Task.get_objects()[i].get_types()`, with
  `Type.get_bases()` walking the full ancestor chain there), but the
  `PlanningTask` this reader is built from keeps no reference back to that
  AST or to the original PDDL file paths needed to re-parse it. This is a
  capability gap in what the translated task carries, not a missing
  binding to add.

`SparseAtomCompositionEncoder`/`encode_sparse_atom_facts` populate
`object_type_ids` whenever `StateView.object_types` is available and leave
it `None` otherwise -- never a fabricated constant that would silently
present as "every object has the same type." Ids come from a fixed,
*domain-scoped* vocabulary (`SparseAtomTypeSchema`, built by
`build_type_schema` from `StateView.type_names`), the same stability
guarantee `build_predicate_schema` already gives predicate ids: two
problems of one domain that happen to instantiate different subsets of its
declared types still agree on which type gets which id.

**Only the most specific declared type**, never the ancestor chain, is
carried. The one real consumer, `SparseAtomCompositionGNN
.object_type_embedding`, is a single `nn.Embedding` lookup -- a categorical
id, not a set -- and the most specific type is the most informative single
label a PDDL declaration gives (a `truck` implies `locatable` implies
`object`, never the reverse). The ancestor chain remains separately
derivable from the domain's type declarations (walk `pymimir.Type
.get_bases()` from a name in `StateView.type_names`) if some future
consumer needs it; it is not threaded through today because nothing reads
it, and an unread field would be exactly the write-only plumbing this
design avoids. An object declared with a PDDL `either` type (more than one
base) has no single "the" type, so pymimir's snapshot layer raises
`ValueError` for it rather than guessing -- no domain under `data/pddl/`
exercises this today.

**Untyped domains degrade to a single real type id, not an absent field.**
Because pymimir always resolves at least the implicit root type `"object"`
for every object -- typed domain or not -- `object_type_ids` being present
reflects "this backend can classify objects", not "this domain declares
more than one type". A Blocksworld problem therefore gets a real (constant,
all-zero) `object_type_ids` with `num_object_types == 1`, computationally
identical to leaving the field `None` (the consumer's own fallback is also
"treat every object as type 0"), but explicit rather than silent about why.
The synthetic nullary "star" object (see
[Nullary normalization](#nullary-normalization) above) follows the same
logic from the other direction: it is not a real PDDL object, so rather
than leave it without a type (every id in the schema must be a genuine,
resolvable row) it is typed generically as the schema's root entry,
`mifrost.encoders.sparse_atom.ROOT_TYPE_NAME` (`"object"` -- the same
literal as `OBJECT_PREDICATE`, not by coincidence: pymimir reports it as
the real root of every domain's type hierarchy, the same way it reports
`object` as a real static predicate for the R4 carrier).
`build_type_schema` always guarantees this entry, synthesizing it if the
supplied type names omit it -- mirroring how `build_predicate_schema`
always guarantees `OBJECT_PREDICATE`.

Batching (`batch_sparse_atom_encodings`) concatenates `object_type_ids`
**without index offsetting**: it is a domain-scoped categorical *value*,
like `atom_predicate_ids`, not a per-graph local index like `atom_args` or
`pair_objects` (compare `kEntityRoleIdsField` -- `GraphFieldMode::CAT` with
no `inc` -- against `kObjectIndicesField`'s `GraphFieldMode::CAT` plus
`GraphFieldInc::Kind::NODE_OFFSET`, in
`src/_core/mifrost/core/encoders/flat/flat_encoder_common.cpp`, which draws
exactly this distinction for the native flat-encoder family). It must be
present on every encoding in a batch or none: unlike
`counterpart_occurrence_ids` (whose `-1` filler the consumer explicitly
treats as "no counterpart"), every id in `[0, num_object_types)` is a
genuine declared type, so there is no spare sentinel value for a graph with
no types once others in the batch declare real ones -- mixing them raises.

`occurrence_type_ids` -- a different, still-unaddressed R13 component (a
per-argument-*position* type drawn from the predicate/action signature,
rather than a per-*object* type) -- remains out of scope and stays `None`.

## Backend-free core vs. the StateView facade

Two layers are exposed:

- `encode_sparse_atom_facts` / `batch_sparse_atom_encodings` /
  `build_predicate_schema` operate on plain `(predicate, args)` tuples
  (`mifrost.encoders.custom.state_view.Atom`) and never touch pymimir or
  pytyr. This is what most of `tests/encoding/test_sparse_atom_composition.py`
  exercises directly -- hand-built Blocksworld-style and synthetic-graph
  scenarios (six-cycle vs. two triangles, the two-tower goal-witness
  separation, and so on) without needing a PDDL problem at all.
- `SparseAtomCompositionEncoder` is a thin `StateView` facade over a real
  pymimir/pytyr problem. Its `goals=` parameter deliberately differs from
  `CustomGraphEncoder`'s lane convention: `goals=None` means *goal-free*
  (zeta=0) here, not "use the problem's own goal" -- this encoder has a
  genuine goal-free mode, and reusing the other convention would make it
  unreachable. Pass `goals=encoder.view.goal_literals(state)` explicitly for
  the problem's own goal, or `goals=()` for a supplied-empty goal (zeta=1,
  `G=∅`; the same atom set as goal-free, differing only in
  `goal_available`).
