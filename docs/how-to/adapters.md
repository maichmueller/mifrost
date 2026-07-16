# Adapters for Custom Types

Public encoders accept native Pymimir wrapper/advanced values and native PyTyr
planning values. Pass a Pymimir domain or PyTyr `PlanningTask` to the
constructor; the backend is inferred per encoder instance, or can be selected
explicitly with `backend="pymimir"` / `backend="pytyr"`.

The `register_*_adapter` functions are the legacy Pymimir compatibility surface
for custom wrappers that can losslessly expose `pymimir.advanced` values:

```python
import mifrost

mifrost.register_state_adapter(MyStateType, lambda s: s.to_advanced_state())
mifrost.register_domain_adapter(MyDomainType, lambda d: d.to_advanced_domain())
mifrost.register_literal_adapter(MyLiteralType, lambda l: l.to_advanced_literal())
mifrost.register_action_adapter(MyActionType, lambda a: a.to_advanced_action())
```

Adapters are matched by exact concrete type and should return the corresponding `pymimir.advanced` object.

Remove adapters with the matching `unregister_*_adapter` function.

These registrations do not establish a process-global active backend and are
not used by PyTyr runtimes. PyTyr accepts its native task/state/action/literal
objects directly. For planner-independent data producers, use the semantic
records and engines from `mifrost._neutral_core` instead of converting through
Pymimir.
