"""HTTP CONNECT tunnel through a local forward proxy (stdlib only).

The Neo4j Python driver has no native proxy support. Cloud Bolt endpoints
that must route through a local proxy (e.g. Clash on ``127.0.0.1:7890``)
are reached via loopback listeners: the driver keeps the real hostname
(for SNI and certificate verification) while ``socket.getaddrinfo`` maps
those hosts to ``127.0.0.1:<port>``.  Each hostname gets its own listener
so the CONNECT target matches the host the driver asked for — Aura's
routing table hands out member names such as ``p-mt-….neo4j.io``, and a
single pinned entry-host tunnel would send those connections to the wrong
backend.

TLS still terminates at the real server.  CONNECT uses the hostname (not a
pre-resolved IP) so domain-based proxy rules keep working.
"""

from __future__ import annotations

import socket
import threading
from typing import Self
from urllib.parse import urlsplit

#: CONNECT replies are short; anything larger is a bad sign, but stay generous.
_READ_CHUNK = 4096


def _should_proxy_host(host: str | None) -> bool:
    if not host:
        return False
    name = host.casefold()
    if name in {"127.0.0.1", "localhost", "::1"}:
        return False
    return name.endswith(".neo4j.io") or name.endswith(".databases.neo4j.io")


class ConnectTunnel:
    """Loopback listener that relays each accepted connection via an HTTP proxy.

    One thread per relayed connection carries both directions; everything is a
    daemon so a forgotten ``close()`` never blocks interpreter shutdown.
    """

    def __init__(
        self,
        proxy_url: str,
        target_host: str,
        target_port: int,
        *,
        connect_timeout: float = 10.0,
    ) -> None:
        parsed = urlsplit(proxy_url if "//" in proxy_url else f"//{proxy_url}")
        if parsed.scheme not in ("", "http", "https") or not parsed.hostname:
            raise ValueError(f"proxy url must be http(s)://host:port: {proxy_url!r}")
        self.proxy_host = parsed.hostname
        self.proxy_port = parsed.port or 7890
        self.target_host = target_host
        self.target_port = target_port
        self.connect_timeout = connect_timeout
        self._server: socket.socket | None = None
        self._threads: set[threading.Thread] = set()
        self._lock = threading.Lock()
        self.local_port: int | None = None

    def start(self) -> ConnectTunnel:
        if self._server is not None:
            return self
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", 0))
        server.listen(16)
        server.settimeout(0.5)
        self._server = server
        self.local_port = server.getsockname()[1]
        worker = threading.Thread(target=self._accept_loop, name="connect-tunnel-accept", daemon=True)
        worker.start()
        self._threads.add(worker)
        return self

    def close(self) -> None:
        if self._server is None:
            return
        server, self._server = self._server, None
        try:
            server.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            server.close()
        except OSError:
            pass
        with self._lock:
            threads = list(self._threads)
        for thread in threads:
            thread.join(timeout=1.0)

    def __enter__(self) -> Self:
        return self.start()

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _accept_loop(self) -> None:
        assert self._server is not None
        while self._server is not None:
            try:
                client, _ = self._server.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            relay = threading.Thread(
                target=self._relay, args=(client,), name="connect-tunnel-relay", daemon=True
            )
            relay.start()
            with self._lock:
                self._threads.add(relay)

    def _relay(self, client: socket.socket) -> None:
        upstream: socket.socket | None = None
        try:
            upstream = socket.create_connection(
                (self.proxy_host, self.proxy_port), timeout=self.connect_timeout
            )
            request = (
                f"CONNECT {self.target_host}:{self.target_port} HTTP/1.1\r\n"
                f"Host: {self.target_host}:{self.target_port}\r\n"
                "Proxy-Connection: keep-alive\r\n\r\n"
            )
            upstream.sendall(request.encode("ascii"))
            response = self._read_connect_response(upstream)
            if not response.startswith(b"HTTP/1.1 200") and not response.startswith(b"HTTP/1.0 200"):
                raise ConnectionError(
                    f"proxy CONNECT rejected: {response.split(chr(13).encode('ascii'))[0][:200]!r}"
                )
            left = threading.Thread(
                target=_pipe, args=(client, upstream, client), name="tunnel-a", daemon=True
            )
            right = threading.Thread(
                target=_pipe, args=(upstream, client, upstream), name="tunnel-b", daemon=True
            )
            left.start()
            right.start()
            left.join()
            right.join()
        except OSError:
            pass
        finally:
            for sock in (upstream, client):
                if sock is not None:
                    try:
                        sock.close()
                    except OSError:
                        pass

    @staticmethod
    def _read_connect_response(sock: socket.socket) -> bytes:
        # 只给 CONNECT 响应设置读取超时；隧道建立后必须恢复阻塞模式。
        # 否则空闲超过 10 秒时，_pipe 会把 socket.timeout 当成断链并关闭
        # Neo4j 的长期连接，驱动随后报 ``defunct connection / No data``。
        sock.settimeout(10.0)
        try:
            data = b""
            while b"\r\n\r\n" not in data and len(data) < 16384:
                chunk = sock.recv(_READ_CHUNK)
                if not chunk:
                    break
                data += chunk
            return data
        finally:
            sock.settimeout(None)


class ProxyBroker:
    """One CONNECT tunnel per hostname so Aura routing members stay on-proxy.

    Installing ``remap_getaddrinfo`` on ``socket.getaddrinfo`` keeps the real
    hostname for SNI while the TCP hop lands on the matching loopback tunnel.
    """

    def __init__(self, proxy_url: str, *, connect_timeout: float = 10.0) -> None:
        self.proxy_url = proxy_url
        self.connect_timeout = connect_timeout
        self._tunnels: dict[tuple[str, int], ConnectTunnel] = {}
        self._lock = threading.Lock()

    def ensure(self, host: str, port: int) -> ConnectTunnel:
        key = (host, port)
        with self._lock:
            tunnel = self._tunnels.get(key)
            if tunnel is not None and getattr(tunnel, "_server", None) is None:
                try:
                    tunnel.close()
                except OSError:
                    pass
                tunnel = None
            if tunnel is None:
                tunnel = ConnectTunnel(
                    self.proxy_url, host, port, connect_timeout=self.connect_timeout
                ).start()
                self._tunnels[key] = tunnel
            return tunnel

    def resolve(self, address):
        """Map Aura hosts onto CONNECT tunnels without patching DNS.

        The Neo4j driver keeps the original hostname for routing/SNI.  Patching
        ``socket.getaddrinfo`` made the driver believe it connected to
        127.0.0.1, after which cluster routing failed with
        ``Unable to retrieve routing information``.
        """

        host = getattr(address, "host", None)
        port = getattr(address, "port", None)
        if host is None:
            host = address[0] if address else ""
        if port is None and address is not None and len(address) > 1:
            port = address[1]
        if _should_proxy_host(str(host) if host else None):
            tunnel = self.ensure(str(host), _numeric_port(port))
            assert tunnel.local_port is not None
            return [("127.0.0.1", tunnel.local_port)]
        return [address]

    def remap_getaddrinfo(self, original):
        def remapped(host, port, family: int = 0, type: int = 0, proto: int = 0, flags: int = 0):
            if _should_proxy_host(host):
                target_port = _numeric_port(port)
                tunnel = self.ensure(str(host), target_port)
                assert tunnel.local_port is not None
                host, port = "127.0.0.1", tunnel.local_port
            return original(host, port, family, type, proto, flags)

        return remapped

    def close(self) -> None:
        with self._lock:
            tunnels = list(self._tunnels.values())
            self._tunnels.clear()
        for tunnel in tunnels:
            tunnel.close()


def _numeric_port(port) -> int:
    if isinstance(port, int) and not isinstance(port, bool) and port > 0:
        return port
    if isinstance(port, str) and port.isdigit():
        return int(port)
    return 7687


def _pipe(source: socket.socket, sink: socket.socket, half: socket.socket) -> None:
    """Copy bytes one way; when the source closes, half-close the sink side."""
    try:
        while True:
            chunk = source.recv(_READ_CHUNK)
            if not chunk:
                break
            sink.sendall(chunk)
    except OSError:
        pass
    finally:
        try:
            half.shutdown(socket.SHUT_WR)
        except OSError:
            pass
