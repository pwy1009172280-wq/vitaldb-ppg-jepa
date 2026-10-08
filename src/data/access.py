"""Split-aware data access views."""

from dataclasses import dataclass
from typing import Literal

from .base import BaseDataset, SubjectIdentityError, SubjectLeakageError
from .samples import UnifiedSample
from .splits import SubjectSplit

SplitRole = Literal["train", "validation", "test"]


@dataclass(frozen=True)
class SplitContext:
    """The only context in which samples may be accessed by an experiment."""

    role: SplitRole
    subject_ids: frozenset[str]

    @classmethod
    def from_split(cls, split: SubjectSplit, role: SplitRole) -> "SplitContext":
        return cls(role=role, subject_ids=frozenset(getattr(split, role)))


class SplitAwareDataset:
    """Read-only view that requires and enforces a declared split context."""

    def __init__(self, dataset: BaseDataset, context: SplitContext) -> None:
        if context is None:
            raise TypeError("SplitAwareDataset requires a SplitContext")
        self.dataset = dataset
        self.context = context

    @property
    def name(self) -> str:
        return self.dataset.name

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> UnifiedSample:
        sample = self.dataset[index]
        if sample.subject_id is None:
            raise SubjectIdentityError(
                "SplitAwareDataset requires RESOLVED subjects; "
                f"sample {index} of dataset {self.dataset.name!r} is unresolved"
            )
        if sample.subject_id not in self.context.subject_ids:
            raise SubjectLeakageError(
                f"subject {sample.subject_id!r} is not declared in "
                f"{self.context.role} split"
            )
        return sample

    def subject_ids(self) -> frozenset[str]:
        return frozenset(self[index].subject_id for index in range(len(self)))
