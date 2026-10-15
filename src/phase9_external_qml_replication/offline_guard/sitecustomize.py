"""Hard-disable Python socket networking for Phase 9 offline reproduction."""

import os
import socket

os.environ["PHASE9_NETWORK_DISABLED"] = "1"


def _deny_network(*args, **kwargs):
    raise RuntimeError("Network access is disabled during Phase 9 offline reproduction")


socket.create_connection = _deny_network
socket.socket.connect = _deny_network
socket.socket.connect_ex = _deny_network
