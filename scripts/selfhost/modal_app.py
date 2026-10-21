"""Python Modal app launcher; the owner supplies an existing dedicated app in configuration."""

from servegauge.selfhost.entrypoints import launcher

if __name__ == "__main__":
    launcher("modal")
