from gitsbe.model.GeneralModel import GeneralModel

if __name__ == '__main__':
    # GeneralModel
    gm = GeneralModel()
    gm.load_sif_file('../example_model_args/toy_ags_network.sif')
    gm.remove_interactions(True, True)
    gm.remove_self_regulated_interactions()
    gm.build_multiple_interactions()
    print('Interactions')
    print(gm)
