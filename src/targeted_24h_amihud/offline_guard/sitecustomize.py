"""Deny external Python socket traffic during offline reproduction.

AF_UNIX sockets remain available for local process coordination; all IPv4/IPv6
connections, connection attempts, and datagrams are rejected.
"""
import socket

NETWORK_GUARD_ACTIVE = True
_connect = socket.socket.connect
_connect_ex = socket.socket.connect_ex
_sendto = socket.socket.sendto
_getaddrinfo = socket.getaddrinfo

def _deny_inet(self, address):
    if self.family in (socket.AF_INET, socket.AF_INET6):
        raise OSError("offline reproduction: network sockets are disabled")
    return _connect(self, address)

def _deny_inet_ex(self, address):
    if self.family in (socket.AF_INET, socket.AF_INET6):
        raise OSError("offline reproduction: network sockets are disabled")
    return _connect_ex(self, address)

def _deny_inet_sendto(self, data, *args):
    if self.family in (socket.AF_INET, socket.AF_INET6):
        raise OSError("offline reproduction: network sockets are disabled")
    return _sendto(self, data, *args)

def _local_only_getaddrinfo(host, *args, **kwargs):
    if host not in (None, "localhost", "127.0.0.1", "::1"):
        raise OSError("offline reproduction: external name resolution is disabled")
    return _getaddrinfo(host, *args, **kwargs)

socket.socket.connect = _deny_inet
socket.socket.connect_ex = _deny_inet_ex
socket.socket.sendto = _deny_inet_sendto
socket.getaddrinfo = _local_only_getaddrinfo
