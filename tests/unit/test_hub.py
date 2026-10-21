"""The dataset adapter receives bytes only and suppresses credential-bearing errors."""

from types import SimpleNamespace

import pytest

from servegauge.adapters.hub import upload_dataset


def test_hub_adapter_pins_destination_and_commits_exact_frozen_bytes(monkeypatch):
    import huggingface_hub

    seen = []

    class FakeApi:
        def __init__(self, *, endpoint, token):
            assert endpoint == "https://huggingface.co" and token == "fixture-credential"

        def create_commit(self, **kwargs):
            assert kwargs["repo_type"] == "dataset" and kwargs["num_threads"] == 1
            operations = kwargs["operations"]
            assert len(operations) == 1 and operations[0].path_in_repo == "README.md"
            assert operations[0].path_or_fileobj == b"reviewed"
            seen.append(kwargs)
            return SimpleNamespace(oid="a" * 40)

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    assert (
        upload_dataset("fixture-owner/data", "fixture-credential", {"README.md": b"reviewed"})
        == "a" * 40
    )
    assert len(seen) == 1


def test_hub_errors_never_expose_credentials_or_request_details(monkeypatch):
    import huggingface_hub

    class FakeApi:
        def __init__(self, **kwargs):
            raise RuntimeError("fixture-credential private upstream detail")

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    with pytest.raises(ValueError, match="outcome_unverified") as error:
        upload_dataset("fixture-owner/data", "fixture-credential", {})
    assert "fixture-credential" not in str(error.value)
