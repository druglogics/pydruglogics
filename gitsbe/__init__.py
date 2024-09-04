from gitsbe.input.Perturbations import Perturbation
from gitsbe.input.TrainingData import TrainingData
from gitsbe.model.BooleanModel import BooleanModel
from gitsbe.model.Evolution import Evolution
from gitsbe.model.InteractionModel import InteractionModel
from gitsbe.input.ModelOutputs import ModelOutputs
from gitsbe.model.ModelPredictions import ModelPredictions


if __name__ == '__main__':
    # Interaction
    interaction = InteractionModel()
    interaction.load_sif_file('../ags_cascade_1.0/network.sif')
    interaction.build_multiple_interactions()
    print('Interactions')
    print(interaction)

    # ModelOutputs
    model_outputs = ModelOutputs('../ags_cascade_1.0/modeloutputs')
    # model_outputs_dict = {
    #     "RPS6KA1": 1.0,
    #     "MYC": 1.0,
    #     "TCF7": 1.0,
    #     "CASP8": -1.0,
    #     "CASP9": -1.0,
    #     "FOXO3": 1.0
    # }
    # model_outputs = ModelOutputs(model_outputs_dict=model_outputs_dict)
    model_outputs.print()

    # TrainingData
    # training_data = TrainingData('../ags_cascade_1.0/training.tab')
    # observations = [(["-"], ["Antisurvival:0", "CASP3:0", "Prosurvival:1", "CCND1:1", "MYC:1",
    #                          "RAC_f:1", "JNK_f:0", "MAPK14:0", "AKT1:1", "MP_f:1",
    #                          "PTEN:0", "ERK_f:1", "KRAS:1", "PIK3CA:1", "S6K_f:1",
    #                          "GSK3_f:0", "TP53:0", "BAX:0", "BCL2:1", "CASP8:0",
    #                          "CTNNB1:1", "TCF7_f:1", "NFKB_f:1"], 1.0)]
    # training_data = TrainingData(observations=observations)
    # training_data.print()

    # DrugPanel
    drug_panel_data = [
        ['PI', 'PIK3CA', 'inhibits'],
        ['PD', 'MAP2K1,MAP2K2'],
        ['CT','GSK3A,GSK3B'],
        ['BI', 'MAPK14'],
        ['PK', 'CTNNB1'],
        ['AK', 'AKT,AKT1,AKT2,AKT3'],
        ['5Z', 'MAP3K7']
    ]

    # Perturbations
    perturbation_data = [
        ['PI'],
        ['PD'],
        ['CT'],
        ['BI'],
        ['PK'],
        ['AK'],
        ['5Z'],
        ['PI', 'PD'],
        ['PI', 'CT'],
        ['PI', 'BI'],
        ['PI', 'PK'],
        ['PI', 'AK'],
        ['PI', '5Z'],
        ['PD', 'CT'],
        ['PD', 'BI'],
        ['PD', 'PK'],
        ['PD', 'AK'],
        ['PD', '5Z'],
        ['CT', 'BI'],
        ['CT', 'PK'],
        ['CT', 'AK'],
        ['CT', '5Z'],
        ['BI', 'PK'],
        ['BI', 'AK'],
        ['BI', '5Z'],
        ['PK', 'AK'],
        ['PK', '5Z'],
        ['AK', '5Z']]
    perturbations = Perturbation(drug_data=drug_panel_data, perturbation_data=perturbation_data)
    perturbations.print()

    # BooleanModel init from .bnet file
    boolean_model_bnet = BooleanModel(file='../ags_cascade_1.0/network.bnet', model_name='test',
                                      mutation_type='mixed', attractor_tool='pyboolnet')
    boolean_model_bnet.print()

    # BooleanModel init from .sif file
    boolean_model_sif = BooleanModel(model=interaction, model_name='test2')
    boolean_model_sif.print()

    # init pygad.GA
    ga_args = {
        'num_generations': 20,
        'num_parents_mating': 2,
        'sol_per_pop': 20,
        'num_genes': 20,
        'parent_selection_type': "sss",
        'crossover_type': "single_point",
        'mutation_type': "random",
        'population_size': 20,
        'mutation_num_genes': 3,
        # 'mutation_percent_genes': 40,
        # 'parallel_processing': 3
    }
    evolution = Evolution(boolean_model=boolean_model_bnet,
                          model_outputs=model_outputs, num_best_solutions=3, num_mutations=5,
                          num_mutations_per_eq=3, num_runs=5, num_cores=4, ga_args=ga_args)
    best_models = evolution.run()
    evolution.save_to_file_models('../solutions')

    # run Drabme
    model_predictions = ModelPredictions(best_models, perturbations, model_outputs, 'hsa')
    model_predictions.run_simulations()
