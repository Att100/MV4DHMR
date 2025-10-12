from datasets.closeint.chi3d import CHI3DSequenceMetaLoader
from datasets.pose.panoptic import PanopticSequenceMetaLoader
from datasets.pose.shelf import ShelfSequenceMetaLoader

sequence_loaders = {
    'chi3d': CHI3DSequenceMetaLoader,
    'panoptic': PanopticSequenceMetaLoader,
    'shelf': ShelfSequenceMetaLoader
}