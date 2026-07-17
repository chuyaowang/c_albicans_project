"""An edge's RoIAlign box is the union of its endpoints' bboxes, so A->B and B->A crop the
identical patch. Half of every forward pass's edge RoIAlign work is therefore duplicated.

These tests pin the two things the deduplication rests on: the boxes really are identical
both ways, and deduplicating reproduces the per-edge answer. They pass before and after the
optimisation on purpose -- their job is to fail if it is ever wrong.

NOT bit-identical, and it cannot be. Identical patches at different offsets in a batched
convolution take different reduction orders, so float32 results drift by ~3e-8. Cropping 899
patches instead of 1798 changes the batch composition and therefore moves the last bits.
ATOL below is set well above that noise and far below anything that could matter: the
features feed a LayerNorm'd fusion MLP.
"""
import torch
from torch_geometric.data import Data
from torchvision.ops import roi_align

from image_processing_tools.scene_graph_network.simple_gnn import Model, _edge_boxes

# Measured float32 conv drift between identical patches at different batch offsets is ~3e-8.
# 1e-6 is two orders above that and still ~7 orders below the feature scale (~0.1-1.0).
ATOL = 1e-6


def _visual_graph(n=8, seed=0):
    """A graph with paired edges and a SAM-like feature map attached."""
    g = torch.Generator().manual_seed(seed)
    pairs = sorted({(min(u, v), max(u, v))
                    for u, v in torch.randint(0, n, (40, 2), generator=g).tolist()
                    if u != v})[:10]
    src = [u for u, v in pairs] + [v for u, v in pairs]
    dst = [v for u, v in pairs] + [u for u, v in pairs]
    ei = torch.tensor([src, dst], dtype=torch.long)

    centroids = torch.rand(n, 2, generator=g) * 100
    x1y1 = torch.rand(n, 2, generator=g) * 80
    boxes = torch.cat([x1y1, x1y1 + 10 + torch.rand(n, 2, generator=g) * 10], dim=1)

    d = Data(x=torch.rand(n, 8, generator=g), edge_index=ei,
             edge_attr=torch.rand(ei.shape[1], 10, generator=g))
    d.centroids = centroids
    d.node_bboxes = boxes
    d.microsam_embedding = torch.rand(256, 16, 16, generator=g)
    d.pixels_per_feature = torch.tensor([8.0])
    return d


def test_the_edge_box_is_identical_in_both_directions():
    """The property the deduplication rests on. It holds because _edge_boxes takes a
    min/max union of the endpoints' bboxes, which cannot depend on their order."""
    d = _visual_graph()
    boxes = _edge_boxes(d.centroids, d.edge_index, 0.15, 20, node_bboxes=d.node_bboxes)
    lookup = {(int(u), int(v)): k for k, (u, v) in enumerate(d.edge_index.t().tolist())}

    checked = 0
    for (u, v), k in lookup.items():
        if u >= v:
            continue
        assert torch.equal(boxes[k], boxes[lookup[(v, u)]]), f"box differs for {u}<->{v}"
        checked += 1
    assert checked > 0, "fixture produced no paired edges, so this asserts nothing"


def test_edge_visual_matches_cropping_every_edge():
    """Deduplicating must not change the answer beyond float32 conv noise. Compares against
    the naive reference: crop all 2E boxes independently and run the CNN over them."""
    model = Model(hidden_channels=16, dropout_p=0.0, use_visual_features=True, d_visual=8)
    d = _visual_graph()
    model.eval()

    with torch.no_grad():
        _, edge_visual = model._extract_visual(d)

        boxes = _edge_boxes(d.centroids, d.edge_index, model.edge_box_margin_frac,
                            model.edge_box_margin_floor, node_bboxes=d.node_bboxes)
        rois = torch.cat([torch.zeros(boxes.size(0), 1), boxes], dim=1)
        patches = roi_align(d.microsam_embedding.unsqueeze(0), rois,
                            output_size=model.roi_output_size,
                            spatial_scale=1.0 / float(d.pixels_per_feature[0]),
                            aligned=True)
        reference = model.edge_visual_cnn(patches)

    assert edge_visual.shape == reference.shape
    assert torch.allclose(edge_visual, reference, atol=ATOL, rtol=0), (
        f"deduplicated edge_visual differs from the per-edge reference beyond float32 conv "
        f"noise; max|diff| = {(edge_visual - reference).abs().max().item():.3e} > {ATOL}"
    )


def test_both_directions_get_the_same_edge_visual():
    """The redundancy the optimisation removes, stated as a property. Not exact even today:
    the two directions crop the identical box, but the CNN's batched reduction order differs
    by offset, so they drift ~3e-8 apart."""
    model = Model(hidden_channels=16, dropout_p=0.0, use_visual_features=True, d_visual=8)
    d = _visual_graph()
    model.eval()
    with torch.no_grad():
        _, edge_visual = model._extract_visual(d)

    lookup = {(int(u), int(v)): k for k, (u, v) in enumerate(d.edge_index.t().tolist())}
    for (u, v), k in lookup.items():
        if u >= v:
            continue
        assert torch.allclose(edge_visual[k], edge_visual[lookup[(v, u)]],
                              atol=ATOL, rtol=0), f"{u}->{v} and {v}->{u} differ"
