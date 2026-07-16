import numpy as np
import pytest

from image_processing_tools.scene_graph_network.cell_merge_inference import (
    merge_fragments, build_prediction_graph, summarize_cells,
    TOPOLOGY_SINGLETON, TOPOLOGY_PATH, TOPOLOGY_BRANCHED, TOPOLOGY_CYCLIC,
)


def _strip(n, gap=4, width=6):
    """n fragments in a row, separated by `gap` background pixels so that no two
    are ever physically touching."""
    H = 10
    W = n * (width + gap) + gap
    ais = np.zeros((H, W), dtype=np.int32)
    for i in range(n):
        x0 = gap + i * (width + gap)
        ais[3:7, x0:x0 + width] = i + 1
    return ais, np.arange(1, n + 1, dtype=np.int32)


def _directed(pairs):
    """Undirected pairs -> a symmetric (2, E) edge_index, as the model sees it."""
    src, dst = [], []
    for u, v in pairs:
        src += [u, v]
        dst += [v, u]
    return np.array([src, dst], dtype=np.int64)


def test_disjoint_fragments_joined_by_an_edge_get_one_label():
    """The case that rules out pixel connected-components: the fragments never
    touch, so only the predicted edges can merge them."""
    from skimage.measure import label as pixel_label
    ais, frag = _strip(3)
    # Pixel connectivity sees three separate blobs and would never merge them.
    assert pixel_label(ais > 0).max() == 3

    ei = _directed([(0, 1), (1, 2)])
    merged, cells, _ = merge_fragments(ais, frag, ei, np.ones(ei.shape[1]))

    assert len(cells) == 1
    fg = merged[ais > 0]
    assert len(np.unique(fg)) == 1 and np.unique(fg)[0] == 1
    assert np.all(merged[ais == 0] == 0)             # background preserved


def test_two_chains_become_two_cells():
    ais, frag = _strip(4)
    ei = _directed([(0, 1), (2, 3)])
    merged, cells, _ = merge_fragments(ais, frag, ei, np.ones(ei.shape[1]))
    assert len(cells) == 2
    assert sorted(np.unique(merged[ais > 0])) == [1, 2]
    assert {c["topology"] for c in cells} == {TOPOLOGY_PATH}


def test_no_positive_edges_leaves_every_fragment_its_own_cell():
    ais, frag = _strip(3)
    ei = _directed([(0, 1), (1, 2)])
    merged, cells, _ = merge_fragments(ais, frag, ei, np.zeros(ei.shape[1]))
    assert len(cells) == 3
    assert all(c["topology"] == TOPOLOGY_SINGLETON for c in cells)
    # Identity relabel: each fragment keeps a distinct id.
    assert len(np.unique(merged[ais > 0])) == 3


def test_partial_predictions_split_a_chain():
    """Fragments 0-1 predicted joined, 2 left out -> two cells."""
    ais, frag = _strip(3)
    ei = _directed([(0, 1), (1, 2)])
    pred = np.array([1, 1, 0, 0], dtype=float)      # only the (0,1) pair fires
    merged, cells, _ = merge_fragments(ais, frag, ei, pred)
    assert len(cells) == 2
    assert sorted(len(c["nodes"]) for c in cells) == [1, 2]


def test_chain_order_runs_endpoint_to_endpoint():
    ais, frag = _strip(4)
    # Deliberately out of order, to prove the order comes from the graph.
    ei = _directed([(2, 3), (0, 1), (1, 2)])
    _, cells, _ = merge_fragments(ais, frag, ei, np.ones(ei.shape[1]))
    assert len(cells) == 1
    cell = cells[0]
    assert cell["topology"] == TOPOLOGY_PATH
    assert cell["nodes"] in ([0, 1, 2, 3], [3, 2, 1, 0])
    assert cell["fragments"] in ([1, 2, 3, 4], [4, 3, 2, 1])


def test_branched_prediction_is_flagged():
    """Node 1 with degree 3 violates the unbranched constraint."""
    ais, frag = _strip(4)
    ei = _directed([(0, 1), (1, 2), (1, 3)])
    _, cells, _ = merge_fragments(ais, frag, ei, np.ones(ei.shape[1]))
    assert len(cells) == 1
    assert cells[0]["topology"] == TOPOLOGY_BRANCHED
    assert len(cells[0]["nodes"]) == 3          # diameter path, not all 4


def test_cyclic_prediction_is_flagged_and_still_merges():
    ais, frag = _strip(3)
    ei = _directed([(0, 1), (1, 2), (2, 0)])
    merged, cells, _ = merge_fragments(ais, frag, ei, np.ones(ei.shape[1]))
    assert len(cells) == 1
    assert cells[0]["topology"] == TOPOLOGY_CYCLIC
    assert len(np.unique(merged[ais > 0])) == 1   # still one cell despite no order


def test_non_contiguous_ais_labels_are_mapped_correctly():
    """AIS label ids need not be 1..N; the node->label map drives the relabel."""
    ais = np.zeros((10, 30), dtype=np.int32)
    ais[3:7, 2:8] = 5
    ais[3:7, 12:18] = 40
    ais[3:7, 22:28] = 9
    frag = np.array([5, 9, 40], dtype=np.int32)   # np.unique / regionprops order
    ei = _directed([(0, 1)])                      # labels 5 and 9
    merged, cells, _ = merge_fragments(ais, frag, ei, np.ones(ei.shape[1]))
    assert len(cells) == 2
    assert merged[ais == 5][0] == merged[ais == 9][0]
    assert merged[ais == 40][0] != merged[ais == 5][0]


def test_graph_carries_attributes_for_downstream_analysis():
    ais, frag = _strip(2)
    ei = _directed([(0, 1)])
    g = build_prediction_graph(
        frag, ei, np.ones(2),
        probs=np.array([0.9, 0.9]), true_labels=np.array([1.0, 1.0]),
        edge_classes=np.array(['TP', 'TP']), attentions=(np.array([0.3, 0.3]), np.array([0.4, 0.4])),
        centroids=np.array([[5.0, 5.0], [5.0, 15.0]]),
    )
    assert g.number_of_nodes() == 2
    assert g.number_of_edges() == 1               # both directions collapse to one
    assert g.nodes[0]["ais_label"] == 1 and g.nodes[0]["y"] == 5.0
    e = g.edges[0, 1]
    assert e["edge_class"] == 'TP' and e["prob"] == pytest.approx(0.9)
    assert e["a1"] == pytest.approx(0.3) and e["a2"] == pytest.approx(0.4)


def test_isolated_nodes_are_present_in_the_graph():
    ais, frag = _strip(3)
    ei = _directed([(0, 1)])
    g = build_prediction_graph(frag, ei, np.ones(2))
    assert g.number_of_nodes() == 3               # node 2 has no edges but exists


def test_summarize_cells_tallies_topologies():
    cells = [{"topology": TOPOLOGY_PATH}, {"topology": TOPOLOGY_PATH},
             {"topology": TOPOLOGY_SINGLETON}, {"topology": TOPOLOGY_CYCLIC}]
    s = summarize_cells(cells)
    assert "4 cells" in s and "2 path" in s and "1 singleton" in s and "1 cyclic" in s
