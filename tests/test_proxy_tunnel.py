"""CONNECT 隧道按主机分流：Aura 成员主机各自落到不同的本地端口。"""

from __future__ import annotations

from core.proxy_tunnel import ProxyBroker, _should_proxy_host


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
