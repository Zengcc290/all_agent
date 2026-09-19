"""Neo4j graph relation store with an in-memory fallback."""

from __future__ import annotations

import socket
import threading
from collections.abc import Callable, Mapping
from operator import itemgetter
from typing import Any

from constants import MEMORY_EDGE_WEIGHT_GROWTH, MEMORY_EDGE_WEIGHT_MAX

# ---------------------------------------------------------------------------
# 进程级 socket.getaddrinfo 补丁（Aura 经本地代理访问）。
#
# Neo4j 驱动在连接/路由阶段才按成员主机名解析地址，因此补丁必须陪伴驱动
# 整个生命周期。多实例（同一进程里的多个 Neo4jGraphStore）同时存在时，
# 用引用计数 + 链式帧保证：谁安装谁卸载，互不覆盖，非 Aura 主机一律走
# 最初捕获的原函数；同一时间不会有两份互相打架的全局补丁。
# ---------------------------------------------------------------------------
_socket_patch_lock = threading.Lock()
_socket_patch_original: Callable[..., Any] | None = None
_socket_patch_frames: list[Callable[..., Any]] = []


def _chained_getaddrinfo(
    host, port, family: int = 0, type: int = 0, proto: int = 0, flags: int = 0
):
    original = _socket_patch_original
    if original is None:
        raise RuntimeError("socket.getaddrinfo patch installed without a captured original")
    for frame in tuple(reversed(_socket_patch_frames)):
        mapped = frame(host, port, family, type, proto, flags)
        if mapped is not None:
            return mapped
    return original(host, port, family, type, proto, flags)


def _make_proxy_frame(broker: Any) -> Callable[..., Any]:
    """Build one chained frame: Aura hosts -> loopback tunnel, other hosts -> None."""

    from core.proxy_tunnel import _numeric_port, _should_proxy_host

    original = _socket_patch_original
    assert original is not None

    def frame(
        host, port, family: int = 0, type: int = 0, proto: int = 0, flags: int = 0
    ):
        if not _should_proxy_host(host):
            return None
        tunnel = broker.ensure(str(host), _numeric_port(port))
        assert tunnel.local_port is not None
        return original("127.0.0.1", tunnel.local_port, family, type, proto, flags)

    return frame


def _install_socket_patch(broker: Any) -> Callable[..., Any]:
    global _socket_patch_original
    with _socket_patch_lock:
        if not _socket_patch_frames:
            _socket_patch_original = socket.getaddrinfo
            socket.getaddrinfo = _chained_getaddrinfo
        frame = _make_proxy_frame(broker)
        _socket_patch_frames.append(frame)
        return frame


def _uninstall_socket_patch(frame: Callable[..., Any]) -> None:
    global _socket_patch_original
    with _socket_patch_lock:
        try:
            _socket_patch_frames.remove(frame)
        except ValueError:
            return
        if not _socket_patch_frames:
            socket.getaddrinfo = _socket_patch_original
            _socket_patch_original = None


class Neo4jGraphStore:
    """Neo4j relation store with an in-memory fallback for local development."""

    def __init__(
        self,
        uri: str | None = None,
        username: str | None = None,
        password: str | None = None,
        *,
        driver: Any = None,
        database: str | None = None,
        proxy_url: str | None = None,
    ) -> None:
        self.database = database
        self.driver = driver
        self.proxy_url = proxy_url
        self._uri = uri
        self._username = username
        self._password = password
        self._broker: Any = None
        self._socket_patch_frame: Callable[..., Any] | None = None
        self._local: dict[str, list[dict[str, Any]]] = {}
        self._reverse: dict[str, list[tuple[str, dict[str, Any]]]] = {}
        #: 内存回退下的实体属性（Neo4j 侧由 ON CREATE/ON MATCH 维护同样的三项）。
        self._entities: dict[str, dict[str, Any]] = {}
        #: Reified n-ary facts. Each observation owns any number of participants
        #: and therefore does not collapse same-triple events at different times.
        self._observations: dict[str, dict[str, Any]] = {}
        if self.driver is None and uri:
            self._open_driver()

    def _open_driver(self) -> None:
        """Create the Bolt driver; Aura via proxy uses the driver resolver."""

        try:
            from neo4j import GraphDatabase
        except ImportError as exc:
            raise RuntimeError("Neo4jGraphStore requires neo4j") from exc
        if not self._username or self._password is None:
            raise ValueError("username and password are required for Neo4j")
        kwargs: dict[str, Any] = {
            "auth": (self._username, self._password),
            "connection_timeout": 15.0,
            "max_connection_lifetime": 60.0,
            "liveness_check_timeout": 10.0,
        }
        if self.proxy_url:
            # Aura needs neo4j+s routing.  The driver has no native proxy, so
            # *.neo4j.io is remapped onto CONNECT tunnels *for the driver's whole
            # lifetime*（连接与路由阶段都会解析成员主机，不能只在构造时换装）。
            # 补丁引用计数安装：close() 时由本实例自己卸载，不影响同进程其他实例。
            from core.proxy_tunnel import ProxyBroker

            self._broker = ProxyBroker(self.proxy_url)
            self._socket_patch_frame = _install_socket_patch(self._broker)
        try:
            self.driver = GraphDatabase.driver(self._uri, **kwargs)
        except Exception:
            # 驱动构造失败时立即卸载刚安装的补丁，避免泄漏进程级全局补丁。
            self._discard_socket_patch()
            raise

    def _discard_socket_patch(self) -> None:
        if self._socket_patch_frame is not None:
            _uninstall_socket_patch(self._socket_patch_frame)
            self._socket_patch_frame = None
        if self._broker is not None:
            try:
                self._broker.close()
            except Exception:  # noqa: BLE001 - best effort during error path
                pass
            self._broker = None

    def _reopen_driver(self) -> None:
        if self.driver is not None and callable(getattr(self.driver, "close", None)):
            try:
                self.driver.close()
            except Exception:  # noqa: BLE001 - stale driver must not block reconnect
                pass
            self.driver = None
        if self._socket_patch_frame is not None:
            _uninstall_socket_patch(self._socket_patch_frame)
            self._socket_patch_frame = None
        if self._broker is not None:
            try:
                self._broker.close()
            except Exception:  # noqa: BLE001
                pass
            self._broker = None
        self._open_driver()

    def _with_session(self, runner):
        """Run ``runner(session)`` and reconnect once on Aura routing flaps."""

        if self.driver is None:
            return None
        try:
            with self.driver.session(database=self.database) as session:
                return runner(session)
        except Exception as exc:
            name = type(exc).__name__
            message = str(exc)
            if name not in {"ServiceUnavailable", "SessionExpired"} and "routing information" not in message:
                raise
            self._reopen_driver()
            with self.driver.session(database=self.database) as session:
                return runner(session)

    def add_relation(
        self,
        source: str,
        relation: str,
        target: str,
        *,
        properties: Mapping[str, Any] | None = None,
        source_domain: str = "",
        target_domain: str = "",
        source_aliases: list[str] | None = None,
        target_aliases: list[str] | None = None,
        source_importance: float = 0.5,
        target_importance: float = 0.5,
        bump: bool = False,
    ) -> None:
        if not all(
            isinstance(value, str) and value.strip()
            for value in (source, relation, target)
        ):
            raise ValueError("source, relation and target must be non-empty strings")
        if bump:
            # F1 回忆强化：只动已存在边的计数/权重，不写实体、不造边。
            self._bump_relation(source, relation, target)
            return
        props = dict(properties or {})
        if self.driver is None:
            edges = self._local.setdefault(source, [])
            existing = next(
                (
                    edge
                    for edge in edges
                    if edge["relation"] == relation and edge["target"] == target
                ),
                None,
            )
            if existing is None:
                edge = {
                    "source": source,
                    "relation": relation,
                    "target": target,
                    "properties": {
                        **props,
                        # F1：计数/权重由存储层在创建时给默认值，写入方（抽取
                        # 管道）不传它们——否则幂等重写会把计数清零。
                        "weight": 1.0,
                        "recall_count": 0,
                        "last_accessed_at": "",
                    },
                }
                edges.append(edge)
                self._reverse.setdefault(target, []).append((source, edge))
            else:
                existing["properties"].update(props)
            self._merge_entity(source, source_domain, source_aliases, source_importance)
            self._merge_entity(target, target_domain, target_aliases, target_importance)
            return
        # ON MATCH 只更新别名：实体的 domain/importance 由首次创建它的那次抽取决定，
        # 后续边写入不该把它们覆盖成空值。weight/recall_count 只在 ON CREATE 给
        # 默认值（F1）：SET r += $properties 是覆盖语义，常规写入碰它们会把
        # 已累计的回忆计数清零。
        query = (
            "MERGE (a:MemoryEntity {name: $source}) "
            "ON CREATE SET a.domain = $source_domain, a.aliases = $source_aliases, a.importance = $source_importance "
            "ON MATCH SET a.aliases = CASE WHEN size($source_aliases) = 0 THEN a.aliases ELSE $source_aliases END "
            "MERGE (b:MemoryEntity {name: $target}) "
            "ON CREATE SET b.domain = $target_domain, b.aliases = $target_aliases, b.importance = $target_importance "
            "ON MATCH SET b.aliases = CASE WHEN size($target_aliases) = 0 THEN b.aliases ELSE $target_aliases END "
            "MERGE (a)-[r:RELATED {kind: $relation}]->(b) "
            "ON CREATE SET r += $properties, r.weight = 1.0, r.recall_count = 0, r.last_accessed_at = '' "
            "ON MATCH SET r += $properties"
        )
        def _run(session):
            session.run(
                query,
                source=source,
                target=target,
                relation=relation,
                properties=props,
                source_domain=source_domain,
                target_domain=target_domain,
                source_aliases=list(source_aliases or []),
                target_aliases=list(target_aliases or []),
                source_importance=source_importance,
                target_importance=target_importance,
            ).consume()

        self._with_session(_run)

    def add_observation(
        self,
        observation_id: str,
        predicate: str,
        participants: list[Mapping[str, Any]],
        *,
        properties: Mapping[str, Any] | None = None,
    ) -> None:
        """Persist one idempotent n-ary fact as an observation node.

        Participants use ``{name, role, ordinal, domain, aliases, importance,
        entity_type}``.  A separate observation identity preserves repeated
        triples at different times and lets one fact carry arbitrary roles.
        """

        if not isinstance(observation_id, str) or not observation_id.strip():
            raise ValueError("observation_id must be a non-empty string")
        if not isinstance(predicate, str) or not predicate.strip():
            raise ValueError("predicate must be a non-empty string")
        normalized: list[dict[str, Any]] = []
        for ordinal, participant in enumerate(participants):
            name = str(participant.get("name") or "").strip()
            role = str(participant.get("role") or "").strip()
            if not name or not role:
                raise ValueError("observation participants require name and role")
            normalized.append(
                {
                    "name": name,
                    "role": role,
                    "ordinal": int(participant.get("ordinal", ordinal)),
                    "domain": str(participant.get("domain") or ""),
                    "aliases": [str(value) for value in participant.get("aliases") or []],
                    "importance": float(participant.get("importance", 0.5)),
                    "entity_type": str(participant.get("entity_type") or "概念"),
                }
            )
        if len(normalized) < 2 or not any(p["role"] == "subject" for p in normalized):
            raise ValueError("an observation requires a subject and at least one other participant")
        props = dict(properties or {})
        props.update({"predicate": predicate, "observation_id": observation_id})
        if self.driver is None:
            self._observations[observation_id] = {
                "id": observation_id,
                "predicate": predicate,
                "properties": props,
                "participants": normalized,
            }
            for participant in normalized:
                self._merge_entity(
                    participant["name"],
                    participant["domain"],
                    participant["aliases"],
                    participant["importance"],
                )
                self._entities[participant["name"]]["entity_type"] = participant[
                    "entity_type"
                ]
            return
        query = (
            "MERGE (o:MemoryObservation {id: $observation_id}) "
            "SET o += $properties, o.predicate = $predicate, o.name = $observation_id "
            "WITH o OPTIONAL MATCH (o)-[old:HAS_PARTICIPANT]->() DELETE old "
            "WITH DISTINCT o UNWIND $participants AS participant "
            "MERGE (e:MemoryEntity {name: participant.name}) "
            "ON CREATE SET e.domain = participant.domain, e.aliases = participant.aliases, "
            "e.importance = participant.importance, e.entity_type = participant.entity_type "
            "ON MATCH SET e.aliases = CASE WHEN size(participant.aliases) = 0 "
            "THEN e.aliases ELSE participant.aliases END, "
            "e.entity_type = coalesce(e.entity_type, participant.entity_type) "
            "MERGE (o)-[r:HAS_PARTICIPANT {role: participant.role, ordinal: participant.ordinal}]->(e) "
            "SET r.entity_type = participant.entity_type, r.kind = participant.role, "
            "r.weight = coalesce(r.weight, 1.0)"
        )
        def _run(session):
            session.run(
                query,
                observation_id=observation_id,
                predicate=predicate,
                properties=props,
                participants=normalized,
            ).consume()

        self._with_session(_run)

    def _merge_entity(
        self, name: str, domain: str, aliases: list[str] | None, importance: float
    ) -> None:
        """In-memory twin of the Cypher ON CREATE/ON MATCH entity clauses."""

        known = list(aliases or [])
        entity = self._entities.get(name)
        if entity is None:
            self._entities[name] = {
                "domain": domain,
                "aliases": known,
                "importance": importance,
            }
        elif known:
            entity["aliases"] = known

    def _bump_relation(self, source: str, relation: str, target: str) -> bool:
        """F1 回忆强化：给已存在的边 +1 次回忆、权重×增长倍数（有上限）。

        边不存在时什么都不做——强化不能凭空造边。返回是否真的强化了一条边。
        """

        from ..base import utc_now

        accessed_at = utc_now().isoformat()
        if self.driver is None:
            for edge in self._local.get(source, []):
                if edge["relation"] == relation and edge["target"] == target:
                    self._apply_bump(edge["properties"], accessed_at)
                    return True
            return False
        query = (
            "MATCH (a:MemoryEntity {name: $source})-[r:RELATED {kind: $relation}]->(b:MemoryEntity {name: $target}) "
            "SET r.recall_count = coalesce(r.recall_count, 0) + 1, "
            "r.last_accessed_at = $accessed_at, "
            "r.weight = CASE "
            "WHEN coalesce(r.weight, 1.0) * $grow_factor > $weight_max THEN $weight_max "
            "ELSE coalesce(r.weight, 1.0) * $grow_factor END "
            "RETURN 1 AS bumped"
        )
        with self.driver.session(database=self.database) as session:
            record = session.run(
                query,
                source=source,
                relation=relation,
                target=target,
                accessed_at=accessed_at,
                grow_factor=MEMORY_EDGE_WEIGHT_GROWTH,
                weight_max=MEMORY_EDGE_WEIGHT_MAX,
            ).single()
        return record is not None

    @staticmethod
    def _apply_bump(properties: dict[str, Any], accessed_at: str) -> None:
        """In-memory twin of the Cypher bump clauses."""

        properties["recall_count"] = int(properties.get("recall_count", 0) or 0) + 1
        properties["last_accessed_at"] = accessed_at
        grown = float(properties.get("weight", 1.0) or 1.0) * MEMORY_EDGE_WEIGHT_GROWTH
        properties["weight"] = min(grown, MEMORY_EDGE_WEIGHT_MAX)

    def entity(self, name: str) -> dict[str, Any]:
        """Entity attributes: 别名 / 领域 / 重要度。

        启用 Neo4j 时以它为准——``_entities`` 只是本进程写入过的内存镜像，
        新进程里它是空的，直接读它会静默返回「没有别名」。
        """

        if self.driver is None:
            return dict(self._entities.get(name, {}))
        query = (
            "MATCH (e:MemoryEntity {name: $name}) "
            "RETURN e.domain AS domain, e.aliases AS aliases, e.importance AS importance"
        )
        with self.driver.session(database=self.database) as session:
            record = session.run(query, name=name).single()
        if record is None:
            return {}
        importance = record["importance"]
        return {
            "domain": record["domain"] or "",
            "aliases": list(record["aliases"] or []),
            "importance": float(importance) if importance is not None else 0.5,
        }

    def entity_aliases(self) -> dict[str, list[str]]:
        """``{实体名: 别名}`` 一次取回；星云图要给每个实体节点带别名，不能逐个查。

        无 driver 时读内存镜像；有 driver 时一条 Cypher 查完全部实体（避免 N+1）。
        """

        if self.driver is None:
            return {
                name: list(attributes.get("aliases") or [])
                for name, attributes in self._entities.items()
            }
        query = "MATCH (e:MemoryEntity) RETURN e.name AS name, e.aliases AS aliases"
        with self.driver.session(database=self.database) as session:
            return {
                str(record["name"]): [str(alias) for alias in (record["aliases"] or [])]
                for record in session.run(query)
            }

    def graph_snapshot(self, *, at: str | None = None) -> dict[str, Any]:
        """Return the actual graph projection used by the Web visualization.

        ``at`` selects the latest temporal observation per subject/predicate at
        or before that instant.  Without it, all observations are returned so
        the UI can render the complete timeline.
        """

        if at:
            from ..base import ensure_datetime

            if ensure_datetime(at) is None:
                raise ValueError("at must be an ISO-8601 datetime")
        if self.driver is None:
            entities = [
                {"name": name, "properties": dict(properties)}
                for name, properties in self._entities.items()
            ]
            observations = [
                {
                    "id": value["id"],
                    "predicate": value["predicate"],
                    "properties": dict(value["properties"]),
                    "participants": [dict(item) for item in value["participants"]],
                }
                for value in self._observations.values()
            ]
            relations = [
                {
                    "source": edge["source"],
                    "relation": edge["relation"],
                    "target": edge["target"],
                    "properties": dict(edge["properties"]),
                }
                for edges in self._local.values()
                for edge in edges
            ]
        else:
            entity_query = (
                "MATCH (e:MemoryEntity) "
                "RETURN e.name AS name, properties(e) AS properties"
            )
            observation_query = (
                "MATCH (o:MemoryObservation) "
                "OPTIONAL MATCH (o)-[r:HAS_PARTICIPANT]->(e:MemoryEntity) "
                "RETURN o.id AS id, o.predicate AS predicate, properties(o) AS properties, "
                "collect({name: e.name, role: r.role, ordinal: r.ordinal, "
                "entity_type: r.entity_type, domain: e.domain, aliases: e.aliases, "
                "importance: e.importance}) AS participants"
            )
            relation_query = (
                "MATCH (a:MemoryEntity)-[r:RELATED]->(b:MemoryEntity) "
                "RETURN a.name AS source, r.kind AS relation, b.name AS target, "
                "properties(r) AS properties"
            )
            def _run(session):
                fetched_entities = [
                    {
                        "name": str(record["name"]),
                        "properties": dict(record["properties"] or {}),
                    }
                    for record in session.run(entity_query)
                ]
                fetched_observations = [
                    {
                        "id": str(record["id"]),
                        "predicate": str(record["predicate"] or "关联"),
                        "properties": dict(record["properties"] or {}),
                        "participants": [
                            dict(item)
                            for item in (record["participants"] or [])
                            if item and item.get("name")
                        ],
                    }
                    for record in session.run(observation_query)
                ]
                fetched_relations = [dict(record) for record in session.run(relation_query)]
                return fetched_entities, fetched_observations, fetched_relations

            entities, observations, relations = self._with_session(_run)
        return {
            "mode": "neo4j" if self.driver is not None else "inmemory",
            "entities": entities,
            "observations": self._observations_at(observations, at),
            "relations": self._relations_at(relations, at),
        }

    @staticmethod
    def _temporal_bounds(
        properties: Mapping[str, Any], at: str
    ) -> tuple[bool, Any, Any]:
        from ..base import ensure_datetime

        moment = ensure_datetime(at)
        if moment is None:
            raise ValueError("at must be an ISO-8601 datetime")
        try:
            valid_from = ensure_datetime(str(properties.get("valid_from") or ""))
        except ValueError:
            valid_from = None
        try:
            valid_to = ensure_datetime(str(properties.get("valid_to") or ""))
        except ValueError:
            valid_to = None
        in_window = (valid_from is None or valid_from <= moment) and (
            valid_to is None or moment <= valid_to
        )
        try:
            event_at = ensure_datetime(str(properties.get("event_at") or ""))
        except ValueError:
            event_at = None
        return in_window and (event_at is None or event_at <= moment), event_at, moment

    @classmethod
    def _observations_at(
        cls, observations: list[dict[str, Any]], at: str | None
    ) -> list[dict[str, Any]]:
        if not at:
            return observations
        timeless: list[dict[str, Any]] = []
        latest: dict[tuple[str, str], tuple[Any, dict[str, Any]]] = {}
        for observation in observations:
            properties = observation.get("properties") or {}
            visible, event_at, _ = cls._temporal_bounds(properties, at)
            if not visible or str(properties.get("status") or "fact") == "expired":
                continue
            if properties.get("cardinality") != "temporal" or event_at is None:
                timeless.append(observation)
                continue
            subject = next(
                (
                    str(item.get("name") or "")
                    for item in observation.get("participants") or []
                    if item.get("role") == "subject"
                ),
                "",
            )
            key = (subject, str(observation.get("predicate") or ""))
            current = latest.get(key)
            if current is None or event_at > current[0]:
                latest[key] = (event_at, observation)
        return [*timeless, *(value[1] for value in latest.values())]

    @classmethod
    def _relations_at(
        cls, relations: list[dict[str, Any]], at: str | None
    ) -> list[dict[str, Any]]:
        if not at:
            return relations
        visible: list[dict[str, Any]] = []
        latest: dict[tuple[str, str], tuple[Any, dict[str, Any]]] = {}
        for relation in relations:
            properties = relation.get("properties") or {}
            matches, event_at, _ = cls._temporal_bounds(properties, at)
            if not matches or str(properties.get("status") or "fact") == "expired":
                continue
            if properties.get("cardinality") == "temporal" and event_at is not None:
                key = (str(relation.get("source") or ""), str(relation.get("relation") or ""))
                current = latest.get(key)
                if current is None or event_at > current[0]:
                    latest[key] = (event_at, relation)
            else:
                visible.append(relation)
        return [*visible, *(value[1] for value in latest.values())]

    def relation_memory_ids(self) -> list[str]:
        """``memory_id`` of every edge (reconcile compares this against facts)."""

        if self.driver is None:
            return [
                str(edge["properties"].get("memory_id") or "")
                for edges in self._local.values()
                for edge in edges
            ]
        query = "MATCH ()-[r:RELATED]->() RETURN r.memory_id AS memory_id"
        with self.driver.session(database=self.database) as session:
            return [str(record["memory_id"]) for record in session.run(query)]

    def get_relations(
        self,
        entity: str,
        *,
        relation: str | None = None,
        direction: str = "both",
        at: str | None = None,
    ) -> list[dict[str, Any]]:
        if direction not in {"in", "out", "both"}:
            raise ValueError("direction must be in, out, or both")
        if self.driver is None:
            values = (
                list(self._local.get(entity, []))
                if direction in ("out", "both")
                else []
            )
            if direction in ("in", "both"):
                values += [
                    {
                        "source": source,
                        "relation": edge["relation"],
                        "target": entity,
                        "properties": edge["properties"],
                    }
                    for source, edge in self._reverse.get(entity, [])
                ]
            if relation is not None:
                values = [edge for edge in values if edge["relation"] == relation]
            return self._relations_at(values, at)
        if direction == "out":
            match, condition = "(a)-[r:RELATED]->(b)", "a.name = $entity"
        elif direction == "in":
            match, condition = "(a)-[r:RELATED]->(b)", "b.name = $entity"
        else:
            match, condition = (
                "(a)-[r:RELATED]->(b)",
                "a.name = $entity OR b.name = $entity",
            )
        clauses = [condition]
        if relation is not None:
            clauses.append("r.kind = $relation")
        query = f"MATCH {match} WHERE {' AND '.join(clauses)} RETURN a.name AS source, r.kind AS relation, b.name AS target, properties(r) AS properties"
        with self.driver.session(database=self.database) as session:
            values = [
                dict(record)
                for record in session.run(query, entity=entity, relation=relation)
            ]
        return self._relations_at(values, at)

    related = get_relations

    def path_query(
        self, start: str, target: str, *, max_depth: int = 3, limit: int = 50
    ) -> list[dict[str, Any]]:
        """Simple paths between two entities, up to ``max_depth`` hops.

        Each result is ``{"entities": [...], "relations": [{source, relation, target, weight}, ...]}``;
        the in-memory fallback mirrors what the Cypher returns. Results are
        ordered by total path weight (F1) — reinforced paths come first.
        """

        if not all(
            isinstance(value, str) and value.strip() for value in (start, target)
        ):
            raise ValueError("start and target must be non-empty strings")
        if isinstance(max_depth, bool) or not isinstance(max_depth, int) or max_depth < 1:
            raise ValueError("max_depth must be a positive integer")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be a positive integer")
        if self.driver is None:
            return self._local_paths(start, target, max_depth=max_depth, limit=limit)
        # 变长区间的上界不能参数化，只能拼进语句；已用 int() 收敛为整数。
        # path_weight 用 reduce 累加边权重（F1），让 LIMIT 截断发生在权重排序之后。
        query = (
            f"MATCH p=(a:MemoryEntity {{name: $start}})-[*1..{int(max_depth)}]-(b:MemoryEntity {{name: $target}}) "
            "RETURN [n IN nodes(p) | n.name] AS entities, "
            "[r IN relationships(p) | {source: startNode(r).name, relation: r.kind, target: endNode(r).name, weight: coalesce(r.weight, 1.0)}] AS relations, "
            "reduce(w = 0.0, r IN relationships(p) | w + coalesce(r.weight, 1.0)) AS path_weight "
            "ORDER BY path_weight DESC, length(p) ASC "
            "LIMIT $limit"
        )
        with self.driver.session(database=self.database) as session:
            return [
                {
                    "entities": list(record["entities"]),
                    "relations": [dict(relation) for relation in record["relations"]],
                }
                for record in session.run(query, start=start, target=target, limit=limit)
            ]

    def _neighbours(self, entity: str) -> list[tuple[str, str, bool, float]]:
        """``(other_end, relation, is_outgoing, weight)`` for every edge at ``entity``."""

        values = [
            (edge["target"], edge["relation"], True, float(edge["properties"].get("weight", 1.0) or 1.0))
            for edge in self._local.get(entity, [])
        ]
        values += [
            (source, edge["relation"], False, float(edge["properties"].get("weight", 1.0) or 1.0))
            for source, edge in self._reverse.get(entity, [])
        ]
        # 权重优先：强边先入队，limit 截断时留下的是更强的路径（F1）。
        values.sort(key=itemgetter(3), reverse=True)
        return values

    def _local_paths(
        self, start: str, target: str, *, max_depth: int, limit: int
    ) -> list[dict[str, Any]]:
        """BFS over simple paths; shortest paths come first because it is a queue."""

        paths: list[dict[str, Any]] = []
        queue: list[tuple[str, list[str], list[dict[str, Any]]]] = [(start, [start], [])]
        while queue and len(paths) < limit:
            node, entities, relations = queue.pop(0)
            if len(relations) >= max_depth:
                continue
            for other, relation, outgoing, weight in self._neighbours(node):
                if other in entities:  # 简单路径：不重复经过同一实体，天然无环
                    continue
                step = {
                    "source": node if outgoing else other,
                    "relation": relation,
                    "target": other if outgoing else node,
                    "weight": weight,
                }
                if other == target:
                    paths.append(
                        {"entities": [*entities, other], "relations": [*relations, step]}
                    )
                    if len(paths) >= limit:
                        break
                    continue
                queue.append((other, [*entities, other], [*relations, step]))
        return paths

    def delete_memory_relation(self, memory_id: str) -> bool:
        """Remove a relation created for a specific semantic memory item."""
        if not isinstance(memory_id, str) or not memory_id:
            raise ValueError("memory_id must be a non-empty string")
        if self.driver is None:
            removed = self._observations.pop(memory_id, None) is not None
            for source, edges in list(self._local.items()):
                kept = [
                    edge
                    for edge in edges
                    if edge.get("properties", {}).get("memory_id") != memory_id
                ]
                for edge in edges:
                    if edge in kept:
                        continue
                    self._reverse[edge["target"]] = [
                        (edge_source, indexed_edge)
                        for edge_source, indexed_edge in self._reverse.get(
                            edge["target"], []
                        )
                        if indexed_edge is not edge
                    ]
                removed = removed or len(kept) != len(edges)
                if kept:
                    self._local[source] = kept
                else:
                    self._local.pop(source, None)
            return removed
        query = (
            "MATCH ()-[r:RELATED {memory_id: $memory_id}]->() DELETE r "
            "WITH count(r) AS removed "
            "OPTIONAL MATCH (o:MemoryObservation {id: $memory_id}) DETACH DELETE o "
            "RETURN removed"
        )
        with self.driver.session(database=self.database) as session:
            result = session.run(query, memory_id=memory_id)
            record = result.single()
            summary = result.consume()
            return bool(record and record["removed"]) or bool(
                getattr(summary.counters, "nodes_deleted", 0)
            )

    def close(self) -> None:
        if self.driver is not None and callable(getattr(self.driver, "close", None)):
            self.driver.close()
        self._discard_socket_patch()


__all__ = ["Neo4jGraphStore"]
