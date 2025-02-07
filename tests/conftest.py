import networkx as nx
import pytest


@pytest.fixture
def tiny_test_graph():
    G = nx.DiGraph()
    nA = (0, 0, 1)
    nB = (0, 1, 3)

    G.add_node(nA, type="rail")
    G.add_node(nB, type="rail")
    G.add_edge(nA, nB, type="dir", l=1)
    G.add_edge(nB, nA, type="dir", l=1)
    return G


@pytest.fixture
def two_by_two_cross_graph():
    G = nx.DiGraph()

    A = (0, 0, 1)
    B1 = (0, 1, 3)
    B = (1, 0, 1)
    A1 = (1, 1, 3)

    for node in [A, B1, B, A1]:
        G.add_node(node, type="rail")

    edges = [(A, B1), (B1, A), (B, A1), (A1, B), (A, B), (B, A), (B1, A1), (A1, B1)]
    for u, v in edges:
        G.add_edge(u, v, type="dir", l=1)

    return G


@pytest.fixture
def passing_node_graph():
    """
    Graph layout:
        (2,1,0) = s1 --- (2,2,0)=t2 --- (2,3,0)=mid --- (2,4,0)=t1 --- (2,5,0)=s2
                                           |
                                        (1,3,0)=top

    If agents don't vanish at goal, PP  can't solve it.
    CBS can solve by routing one agent via (1,3,0).
    """
    G = nx.DiGraph()

    s1 = (2, 1, 0)
    t2 = (2, 2, 0)
    mid = (2, 3, 0)
    top = (1, 3, 0)
    t1 = (2, 4, 0)
    s2 = (2, 5, 0)

    for node in [s1, t2, mid, top, t1, s2]:
        G.add_node(node, type="rail")

    edges = [
        (s1, t2),
        (t2, s1),
        (t2, mid),
        (mid, t2),
        (mid, top),
        (top, mid),
        (mid, t1),
        (t1, mid),
        (t1, s2),
        (s2, t1),
    ]
    for u, v in edges:
        G.add_edge(u, v, type="dir", l=1)

    return G


@pytest.fixture
def two_trains_suboptimal_graph():
    G = nx.DiGraph()
    nodes = [
        (0, 1, 0),
        (1, 0, 0),
        (1, 1, 0),
        (1, 2, 0),
        (1, 3, 0),
        (2, 1, 0),
        (2, 3, 0),
        (3, 1, 0),
        (3, 2, 0),
        (3, 3, 0),
        (4, 1, 0),
        (4, 2, 0),
    ]

    for n in nodes:
        G.add_node(n, type="rail")

    edges = [
        ((0, 1, 0), (1, 1, 0)),
        ((1, 0, 0), (1, 1, 0)),
        ((1, 1, 0), (1, 2, 0)),
        ((1, 2, 0), (1, 3, 0)),
        ((1, 3, 0), (2, 3, 0)),
        ((2, 3, 0), (3, 3, 0)),
        ((1, 1, 0), (2, 1, 0)),
        ((2, 1, 0), (3, 1, 0)),
        ((3, 1, 0), (4, 1, 0)),
        ((3, 1, 0), (3, 2, 0)),
        ((3, 2, 0), (4, 2, 0)),
        ((3, 2, 0), (3, 3, 0)),
    ]

    for u, v in edges:
        G.add_edge(u, v, type="dir", l=1)
        G.add_edge(v, u, type="dir", l=1)
    return G
