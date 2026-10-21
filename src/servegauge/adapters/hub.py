"""Owner-invoked dataset commit from an already frozen explicit byte inventory."""

from __future__ import annotations

import re


def upload_dataset(
    repo_id: str,
    token: str,
    payloads: dict[str, bytes],
    *,
    commit_message: str = "Add reviewed ServeGauge numeric observations",
) -> str:
    from huggingface_hub import CommitOperationAdd, HfApi

    try:
        result = HfApi(endpoint="https://huggingface.co", token=token).create_commit(
            repo_id=repo_id,
            repo_type="dataset",
            operations=[
                CommitOperationAdd(path_in_repo=name, path_or_fileobj=blob)
                for name, blob in sorted(payloads.items())
            ],
            commit_message=commit_message,
            num_threads=1,
        )
        identifier = result.oid
        if not re.fullmatch(r"[a-fA-F0-9]{40,64}", identifier):
            raise ValueError("hub_commit_identifier_unavailable")
        return identifier
    except Exception:
        raise ValueError("hub_upload_failed_outcome_unverified") from None
