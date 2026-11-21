"""A small fixed-shape answer for the engine tests (the same fields as AutoDocs' write-up)."""

from pydantic import BaseModel


class SampleAnswer(BaseModel):
    """The shape the tests ask the fake model to answer in."""

    title: str
    summary: str
    changes: list[str]
    functions: list[dict]
    release_note: str
    could_not_determine: list[str]
