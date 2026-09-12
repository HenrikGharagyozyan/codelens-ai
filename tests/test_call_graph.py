"""Tests for CallGraph: resolved neighbours, multi-hop chains, depth limits and budgets."""

import pytest

from codelens.graph.call_graph import CallGraph, chain_nodes, is_test_path


@pytest.fixture
def graph(db):
    """A small resolved graph.

    main -> Service.run -> connect
            Service.run -> helper
    test_run (tests/) -> Service.run
    ping <-> pong  (mutual recursion)
    Service.run also calls print (builtin) and query (unresolved).
    """
    symbols = [
        ("app.py::main", "main", "main", "function", "app.py", 20),
        ("app.py::Service", "Service", "Service", "class", "app.py", 1),
        ("app.py::Service.run", "run", "Service.run", "method", "app.py", 5),
        ("app.py::helper", "helper", "helper", "function", "app.py", 30),
        ("db.py::connect", "connect", "connect", "function", "db.py", 1),
        ("tests/test_app.py::test_run", "test_run", "test_run", "function", "tests/test_app.py", 3),
        ("loop.py::ping", "ping", "ping", "function", "loop.py", 1),
        ("loop.py::pong", "pong", "pong", "function", "loop.py", 5),
    ]
    for sid, name, qualname, kind, path, line in symbols:
        db.insert_symbol(sid, name, kind, path, line, qualname=qualname)

    edges = [
        ("app.py::main", "run", 21, "app.py::Service.run", "typed"),
        ("app.py::Service.run", "connect", 6, "db.py::connect", "import"),
        ("app.py::Service.run", "helper", 7, "app.py::helper", "direct"),
        ("app.py::Service.run", "print", 8, None, "builtin"),
        ("app.py::Service.run", "query", 9, None, "unresolved"),
        ("tests/test_app.py::test_run", "run", 4, "app.py::Service.run", "typed"),
        ("loop.py::ping", "pong", 2, "loop.py::pong", "direct"),
        ("loop.py::pong", "ping", 6, "loop.py::ping", "direct"),
    ]
    for caller, name, line, callee_id, how in edges:
        db.insert_call(caller, name, line, callee_id=callee_id, resolution=how)

    return CallGraph(db)


def names(nodes):
    return [node.qualname for node in nodes]


class TestNeighbours:
    def test_callers_put_real_code_before_tests(self, graph):
        edges = graph.callers("app.py::Service.run")

        assert [e.caller.qualname for e in edges] == ["main", "test_run"]
        assert edges[0].line == 21

    def test_callees_follow_only_resolved_edges(self, graph):
        edges = graph.callees("app.py::Service.run")

        assert [e.callee.qualname for e in edges] == ["connect", "helper"]

    def test_an_unknown_symbol_has_no_neighbours(self, graph):
        assert graph.callers("nope") == []
        assert graph.callees("nope") == []


class TestChains:
    def test_caller_chains_run_from_the_outermost_caller_inwards(self, graph):
        chains = graph.caller_chains("db.py::connect", depth=3)

        assert [names(chain_nodes(c)) for c in chains] == [
            ["main", "Service.run", "connect"],
            ["test_run", "Service.run", "connect"],
        ]

    def test_callee_chains_follow_calls_downwards(self, graph):
        chains = graph.callee_chains("app.py::main", depth=3)

        assert [names(chain_nodes(c)) for c in chains] == [
            ["main", "Service.run", "connect"],
            ["main", "Service.run", "helper"],
        ]

    def test_depth_limits_how_far_a_chain_goes(self, graph):
        chains = graph.caller_chains("db.py::connect", depth=1)

        assert [names(chain_nodes(c)) for c in chains] == [["Service.run", "connect"]]

    def test_the_budget_caps_the_number_of_paths(self, graph):
        assert len(graph.caller_chains("db.py::connect", depth=3, max_paths=1)) == 1

    def test_recursion_does_not_loop(self, graph):
        chains = graph.callee_chains("loop.py::ping", depth=5)

        assert [names(chain_nodes(c)) for c in chains] == [["ping", "pong"]]

    def test_a_leaf_has_no_chains(self, graph):
        assert graph.callee_chains("db.py::connect") == []


class TestNeighbourhood:
    def test_reports_hop_distance_in_both_directions(self, graph):
        found = {node.qualname: hops for node, hops in graph.neighbourhood("app.py::Service.run", depth=2)}

        assert found == {"main": 1, "test_run": 1, "connect": 1, "helper": 1}

    def test_two_hops_reach_further(self, graph):
        found = {node.qualname: hops for node, hops in graph.neighbourhood("db.py::connect", depth=2)}

        assert found == {"Service.run": 1, "main": 2, "test_run": 2, "helper": 2}

    def test_the_node_budget_is_respected(self, graph):
        assert len(graph.neighbourhood("db.py::connect", depth=2, max_nodes=2)) == 2


def test_is_test_path_matches_the_hybrid_search_heuristic():
    assert is_test_path("tests/test_app.py")
    assert is_test_path("src/test_utils.py")
    assert not is_test_path("src/app.py")
    assert not is_test_path(None)
