"""Python SSH launcher; all actions use the reviewed CLI, shared ledger and remote shutdown."""

from servegauge.selfhost.entrypoints import launcher

if __name__ == "__main__":
    launcher("ssh")
