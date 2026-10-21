"""Explicit sourced declarations are distinct from a requested API parameter."""

from servegauge.core.models import EndpointSpec


def declared_structured(endpoint: EndpointSpec) -> bool | None:
    declaration = endpoint.metadata.get("declared_json_schema")
    if (
        not isinstance(declaration, dict)
        or declaration.get("verified") is not True
        or not isinstance(declaration.get("source"), str)
        or not declaration["source"]
    ):
        return None
    value = declaration.get("value")
    return value if isinstance(value, bool) else None


def strict_schema_contract(endpoint: EndpointSpec, category: str) -> bool:
    declaration = endpoint.metadata.get("strict_schema_guarantee")
    return bool(
        isinstance(declaration, dict)
        and declaration.get("verified") is True
        and isinstance(declaration.get("source"), str)
        and declaration["source"]
        and isinstance(declaration.get("features"), list)
        and all(isinstance(feature, str) for feature in declaration["features"])
        and category in declaration.get("features", [])
    )
