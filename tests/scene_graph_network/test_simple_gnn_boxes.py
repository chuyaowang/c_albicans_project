import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")
import torch
from image_processing_tools.scene_graph_network.simple_gnn import _node_boxes, _edge_boxes


def test_node_boxes_use_mask_bboxes_when_given():
    centroids = torch.tensor([[10.0, 10.0], [20.0, 20.0]])
    node_bboxes = torch.tensor([[0.0, 0.0, 5.0, 5.0], [6.0, 6.0, 12.0, 12.0]])
    out = _node_boxes(centroids, box_size=150, node_bboxes=node_bboxes, pad_frac=0.0)
    assert torch.allclose(out, node_bboxes)


def test_node_boxes_fallback_to_centroid_square():
    centroids = torch.tensor([[10.0, 10.0]])
    out = _node_boxes(centroids, box_size=4)     # x1,y1,x2,y2 around (x=10,y=10)
    assert out.tolist() == [[8.0, 8.0, 12.0, 12.0]]


def test_edge_boxes_union_of_mask_bboxes():
    centroids = torch.tensor([[10.0, 10.0], [20.0, 20.0]])
    node_bboxes = torch.tensor([[0.0, 0.0, 5.0, 5.0], [6.0, 6.0, 12.0, 12.0]])
    edge_index = torch.tensor([[0], [1]])
    out = _edge_boxes(centroids, edge_index, 0.0, 0, node_bboxes=node_bboxes)
    # union of the two boxes = [0,0,12,12]
    assert out.tolist() == [[0.0, 0.0, 12.0, 12.0]]