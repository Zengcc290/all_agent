"""CONNECT 隧道按主机分流：Aura 成员主机各自落到不同的本地端口。"""

from __future__ import annotations

from core.proxy_tunnel import ConnectTunnel, ProxyBroker, _should_proxy_host


class _ResponseSocket:
    def __init__(self) -> None:
        self.timeouts: list[float | None] = []
        self.chunks = [b"HTTP/1.1 200 Connection Established\r\n\r\n"]

    def settimeout(self, value: float | None) -> None:
        self.timeouts.append(value)

    def recv(self, size: int) -> bytes:
        return self.chunks.pop(0)


def test_connect_response_timeout_is_removed_after_handshake() -> None:
    sock = _ResponseSocket()

    response = ConnectTunnel._read_connect_response(sock)  # type: ignore[arg-type]

    assert response.startswith(b"HTTP/1.1 200")
    assert sock.timeouts == [10.0, None]


def test_should_proxy_only_aura_hosts() -> None:
    assert _should_proxy_host("53ac91cf.databases.neo4j.io")
    assert _should_proxy_host("p-mt-abc.production-orch-0068.neo4j.io")
    assert not _should_proxy_host("127.0.0.1")
    assert not _should_proxy_host("localhost")
    assert not _should_proxy_host("example.com")


def test_broker_opens_a_distinct_tunnel_per_hostname() -> None:
    broker = ProxyBroker("http://127.0.0.1:7890")
    original = lambda host, port, family=0, type=0, proto=0, flags=0: [  # noqa: E731
        (2, 1, 6, "", (host, int(port or 0)))
    ]
    remapped = broker.remap_getaddrinfo(original)
    try:
        first = remapped("a.databases.neo4j.io", 7687)
        second = remapped("b.neo4j.io", 7687)
        skipped = remapped("example.com", 80)

        assert first[0][4][0] == "127.0.0.1"
        assert second[0][4][0] == "127.0.0.1"
        assert first[0][4][1] != second[0][4][1]
        assert skipped[0][4] == ("example.com", 80)
        assert len(broker._tunnels) == 2
        again = remapped("a.databases.neo4j.io", 7687)
        assert again[0][4] == first[0][4]
        assert len(broker._tunnels) == 2
    finally:
        broker.close()

def test_broker_resolver_keeps_non_aura_address() -> None:
    broker = ProxyBroker("http://127.0.0.1:7890")
    try:
        skipped = broker.resolve(("example.com", 80))
        assert skipped == [("example.com", 80)]
        mapped = broker.resolve(("a.databases.neo4j.io", 7687))
        assert mapped[0][0] == "127.0.0.1"
        assert mapped[0][1] != 7687
        again = broker.resolve(("a.databases.neo4j.io", 7687))
        assert again == mapped
        other = broker.resolve(("b.neo4j.io", 7687))
        assert other[0][1] != mapped[0][1]
    finally:
        broker.close()


def test_refcounted_getaddrinfo_patch_installs_and_uninstalls() -> None:
    """补丁引用计数：多实例共存时链式生效，逐个关闭后原子函数还原。"""

    import socket as socket_module

    import memory.storage.graph as graph_module

    original = socket_module.getaddrinfo
    try:
        first = _install_broker()
        second = _install_broker()
        patched = socket_module.getaddrinfo
        assert patched is graph_module._chained_getaddrinfo

        def resolve(host: str) -> str:
            return patched(host, 7687)[0][4][0]

        assert resolve("example.com") == original("example.com", 80)[0][4][0]
        first_port = resolve("a.databases.neo4j.io")
        second_port = resolve("b.neo4j.io")
        assert first_port == "127.0.0.1" and second_port == "127.0.0.1"

        graph_module._uninstall_socket_patch(first)
        assert socket_module.getaddrinfo is graph_module._chained_getaddrinfo
        assert resolve("a.databases.neo4j.io") == "127.0.0.1"

        graph_module._uninstall_socket_patch(second)
        assert socket_module.getaddrinfo is original
    finally:
        if socket_module.getaddrinfo is not original:
            socket_module.getaddrinfo = original
        graph_module._socket_patch_frames.clear()
        graph_module._socket_patch_original = None


def _install_broker():
    import memory.storage.graph as graph_module
    from core.proxy_tunnel import ProxyBroker

    broker = ProxyBroker("http://127.0.0.1:7890")
    return graph_module._install_socket_patch(broker)
