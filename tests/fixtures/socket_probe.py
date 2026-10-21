"""Intentional negative acceptance probe; invoke this file explicitly, expecting exit 1."""

import socket


def test_socket_creation_must_fail() -> None:
    socket.socket()
