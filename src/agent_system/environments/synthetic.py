"""Abstract-component oracle: the semantics of the frozen synthetic model."""

from __future__ import annotations

from .base import Environment, TaskInstance
from ..schemas import ComponentSpec


class SyntheticEnvironment(Environment):
    """Components carry no semantics; an experiment returns only its label."""

    kind = "synthetic"
    notes = (
        "Components are abstract identifiers with no physical meaning. "
        "An experiment reveals the exact label of one component and nothing else."
    )

    def __init__(self, task: TaskInstance) -> None:
        super().__init__(task)
        encoding = task.encoding
        self._library = tuple(
            ComponentSpec(
                family_id=family,
                variant_id=variant,
                name=f"component_{family}_{variant}",
                expression=f"abstract[{family}][{variant}]",
            )
            for family in range(encoding.families)
            for variant in range(encoding.variants)
        )

    def component_library(self) -> tuple[ComponentSpec, ...]:
        return self._library
