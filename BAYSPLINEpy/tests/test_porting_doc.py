"""PORTING.md and the tests cannot drift apart.

Every entry ID in the translation record must have a ``test_<ID>`` in
test_port.py, and every DEVIATION or DEFECT must appear in the register.
"""
from __future__ import annotations

import re

import pytest

import test_port

ENTRY = re.compile(r"^### ([A-Z]+-\d+)\s*\n`?([A-Z]+)`?", re.M)


@pytest.fixture(scope="module")
def doc(root):
    path = root / "docs" / "BAYSPLINEpy" / "PORTING.md"
    if not path.is_file():
        pytest.skip("docs/BAYSPLINEpy/PORTING.md not found")
    return path.read_text(encoding="utf-8")


def _entries(doc):
    return {m.group(1): m.group(2) for m in ENTRY.finditer(doc)}


def test_every_entry_has_a_test(doc):
    entries = _entries(doc)
    assert len(entries) >= 25
    missing = [i for i in entries if not hasattr(test_port, "test_" + i.replace("-", "_"))]
    assert not missing, f"PORTING.md entries without a test: {missing}"


def test_every_test_has_an_entry(doc):
    entries = _entries(doc)
    ids = {n[5:].replace("_", "-") for n in dir(test_port)
           if re.fullmatch(r"test_[A-Z]+_\d+", n)}
    orphans = sorted(ids - set(entries))
    assert not orphans, f"tests with no PORTING.md entry: {orphans}"


def test_register_lists_every_behaviour_change(doc):
    register = doc.split("## Register", 1)[1].split("\n---", 1)[0]
    listed = set(re.findall(r"\[`([A-Z]+-\d+)`\]", register))
    must = {i for i, s in _entries(doc).items() if s in ("DEVIATION", "DEFECT", "EXTENSION")}
    assert must <= listed, f"missing from the register: {sorted(must - listed)}"
