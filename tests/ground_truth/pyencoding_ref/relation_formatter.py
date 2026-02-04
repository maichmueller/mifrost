from __future__ import annotations

from functools import cache, singledispatchmethod
from typing import Protocol, runtime_checkable

import pymimir

from mifrost.utils.singleton import PickleSafeSingleton

from .accessors import (
    action_name,
    action_objects,
    literal_atom,
    literal_polarity,
    object_name,
    predicate_name,
)

Node = str


@runtime_checkable
class RelationProto(Protocol):
    def get_name(self) -> str: ...

    def get_arity(self) -> int: ...


class RelationFormatter(metaclass=PickleSafeSingleton):
    positive_prefix = "[+]"
    negative_prefix = "[-]"
    goal_suffix = "[g]"
    subgoal_suffix = "[sg]"
    subsubgoal_suffix = "[ssg]"
    subsubsubgoal_suffix = "[sssg]"
    goal_satisfied_suffix = "[sat]"
    goal_unsatisfied_suffix = "[unsat]"
    goal_satisfied_added_suffix = "[sat+]"
    goal_satisfied_removed_suffix = "[sat-]"

    goal_level_suffixes = {
        0: goal_suffix,
        1: subgoal_suffix,
        2: subsubgoal_suffix,
        3: subsubsubgoal_suffix,
        None: "",
    }
    polarity_prefixes = {
        True: positive_prefix,
        False: negative_prefix,
        None: "",
    }
    goal_satisfaction_suffixes = {
        True: goal_satisfied_suffix,
        False: goal_unsatisfied_suffix,
        "+": goal_satisfied_added_suffix,
        "-": goal_satisfied_removed_suffix,
        None: "",
    }

    default_nullary_symbol_name = "![nullary_symbol]!"

    @singledispatchmethod
    def __call__(self, item, *args, **kwargs) -> Node | None:
        raise NotImplementedError(
            "__call__ is not implemented for type {}".format(type(item))
        )

    @__call__.register
    def atom(
        self,
        atom: pymimir.GroundAtom,
        pos: int | None = None,
        *args,
        **kwargs,
    ) -> Node | None:
        if pos is None:
            return str(atom)
        return f"{atom}:{pos}"

    @__call__.register(object)
    def predicate(
        self,
        predicate: object,
        *,
        goal_level: int | None = None,
        goal_satisfaction: bool | str | None = None,
        polarity: bool = None,
        **kwargs,
    ) -> Node | None:
        if hasattr(predicate, "get_atom") and hasattr(predicate, "get_polarity"):
            return self._format_literal_obj(
                predicate,
                goal_level=goal_level,
                goal_satisfaction=goal_satisfaction,
                polarity=polarity,
            )
        if hasattr(predicate, "get_predicate") and hasattr(predicate, "get_terms"):
            return str(predicate)
        if not hasattr(predicate, "get_name") and not hasattr(predicate, "name"):
            raise NotImplementedError(
                "__call__ is not implemented for type {}".format(type(predicate))
            )
        return (
            f"{self.polarity_prefixes[polarity]}"
            f"{predicate_name(predicate)}"
            f"{self.goal_level_suffixes[goal_level]}"
            f"{self.goal_satisfaction_suffixes[goal_satisfaction]}"
        )

    @cache
    @__call__.register
    def literal(
        self,
        literal: pymimir.GroundLiteral,
        pos: int | None = None,
        *args,
        goal_level: int | None = None,
        goal_satisfaction: bool = None,
        polarity: bool | None = None,
        **kwargs,
    ) -> Node | None:
        return self._format_literal_obj(
            literal,
            pos=pos,
            goal_level=goal_level,
            goal_satisfaction=goal_satisfaction,
            polarity=polarity,
        )

    def _format_literal_obj(
        self,
        literal: object,
        *,
        pos: int | None = None,
        goal_level: int | None = None,
        goal_satisfaction: bool | str | None = None,
        polarity: bool | None = None,
    ) -> Node | None:
        atom = literal_atom(literal)
        lit_polarity = literal_polarity(literal) if polarity is None else polarity
        atom_str = self.atom(atom)
        literal_str = f"{self.polarity_prefixes[lit_polarity]}{atom_str}"
        return (
            f"{literal_str}"
            f"{self.goal_level_suffixes[goal_level]}"
            f"{self.goal_satisfaction_suffixes[goal_satisfaction]}"
            f"{f':{pos}' if pos is not None else ''}"
        )

    @__call__.register
    def action_schema(
        self,
        action: pymimir.Action,
        **kwargs,
    ) -> Node | None:
        return action_name(action)

    @cache
    @__call__.register
    def action(
        self,
        action: pymimir.GroundAction,
        **kwargs,
    ) -> Node | None:
        parts = [action_name(action)]
        parts.extend(object_name(obj) for obj in action_objects(action))
        return "(" + " ".join(parts) + ")"

    @__call__.register
    def object(self, obj: pymimir.Object, *args, **kwargs) -> Node | None:
        return object_name(obj)

    @__call__.register
    def str_(self, s: str, *args, **kwargs) -> Node | None:
        return s

    @__call__.register
    def none(self, none: None, *args, **kwargs) -> Node | None:
        return str(None)


relation_formatter = RelationFormatter()
