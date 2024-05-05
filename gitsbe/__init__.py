from gitsbe.model.GeneralModel import *
# from gitsbe.model.MultipleInteraction import MultipleInteraction
from gitsbe.model.SingleInteraction import SingleInteraction
# from gitsbe.utils.util import Util

# Press the green button in the gutter to run the script.

if __name__ == '__main__':
    gm = GeneralModel()
    gm.load_sif_file('../example_model_args/toy_ags_network.sif')
    gm.remove_interactions(True, True)
    gm.remove_self_regulated_interactions()
    gm.build_multiple_interactions()
