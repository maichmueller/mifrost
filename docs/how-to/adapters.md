# Adapters for Custom Types

Encoders accept native `pymimir` wrapper/advanced types by default. For custom wrappers, register explicit adapters.

```python
import mifrost

mifrost.register_state_adapter(MyStateType, lambda s: s.to_advanced_state())
mifrost.register_domain_adapter(MyDomainType, lambda d: d.to_advanced_domain())
mifrost.register_literal_adapter(MyLiteralType, lambda l: l.to_advanced_literal())
mifrost.register_action_adapter(MyActionType, lambda a: a.to_advanced_action())
```

Adapters are matched by exact concrete type and should return the corresponding `pymimir.advanced` object.

Remove adapters with the matching `unregister_*_adapter` function.
