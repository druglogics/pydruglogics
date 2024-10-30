import logging
from pydruglogics.input.Perturbations import Perturbation
from pydruglogics.input.TrainingData import TrainingData
from pydruglogics.model.BooleanModel import BooleanModel
from pydruglogics.model.Evolution import Evolution
from pydruglogics.model.InteractionModel import InteractionModel
from pydruglogics.input.ModelOutputs import ModelOutputs
from pydruglogics.execution.Executor import Executor
from pydruglogics.model.Statistics import Statistics
from pydruglogics.utils.Logger import Logger

if __name__ == '__main__':
    # Interaction
    interaction = InteractionModel(interactions_file='../ags_cascade_1.0/network.sif', self_regulated_interactions=True,
                                   remove_inputs=True, remove_outputs=False)
    interaction.print()

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
    # training_data = TrainingData(input_file='../ags_cascade_1.0/training')
    observations = [(["CASP3:0", "CASP8:0","CASP9:0","FOXO_f:0","RSK_f:1","CCND1:1",
                      "MYC:1","RAC_f:1","JNK_f:0","MAPK14:0","AKT_f:1","MMP_f:1",
                      "PTEN:0","ERK_f:1","KRAS:1","PIK3CA:1","S6K_f:1","GSK3_f:0",
                      "TP53:0","BAX:0","BCL2:1","CTNNB1:1","TCF7_f:1","NFKB_f:1"], 1.0)]
    training_data = TrainingData(observations=observations)
    training_data.print()

    # DrugPanel
    drug_panel_data = [
        ['PI', 'PIK3CA', 'inhibits'],
        ['PD', 'MEK_f'],
        ['CT','GSK3_f'],
        ['BI', 'MAPK14'],
        ['PK', 'CTNNB1'],
        ['AK', 'AKT_f'],
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
                                      mutation_type='topology', attractor_tool='mpbn', attractor_type='stable_states')


    # BooleanModel init from .sif file
    boolean_model_sif = BooleanModel(model=interaction, model_name='test2',
                                     mutation_type='balanced', attractor_tool='mpbn', attractor_type='trapspaces')

    # init pygad.GA
    ga_args = {
        'num_generations': 20,
        'num_parents_mating': 3,
        'mutation_num_genes': 2,
        'fitness_batch_size': 20,
        'gene_type': int,
        'keep_elitism': 3
    }

    # init evolution params
    ev_args = {
        'num_best_solutions': 3,
        'num_of_runs': 30,
        'num_of_cores': 4,
        'num_of_init_mutation': 6
    }

    # init observed synergy scores
    observed_synergy_scores = ["PI-PD", "PI-5Z", "PD-AK", "AK-5Z"]

    # init executor
    executor = Executor()

    # Running evolution
    # executor.train(
    #     boolean_model=boolean_model_bnet,
    #     model_outputs=model_outputs,
    #     training_data=training_data,
    #     ga_args=ga_args,
    #     ev_args=ev_args,
    #     save_best_models=True,
    #     save_path='./models'
    # )
    #
    # # Running predictions
    # executor.predict(
    #     perturbations=perturbations,
    #     model_outputs=model_outputs,
    #     observed_synergy_scores=observed_synergy_scores,
    #     synergy_method='hsa',
    #     best_boolean_models=None,
    #     model_directory = './models/models_java',
    #     # 'attractor_tool':'mpbn'
    #     # 'attractor_type':'trapspaces'
    # )

    # Running evolution and predictions
    executor.execute(
        train=True,
        predict=True,
        train_params={
            'boolean_model': boolean_model_bnet,
            'model_outputs': model_outputs,
            'training_data': training_data,
            'ga_args': ga_args,
            'ev_args': ev_args,
            'save_best_models': False,
            # 'save_path': './models'
        },
        predict_params={
            'best_boolean_models': None,
            'perturbations': perturbations,
            'model_outputs':model_outputs,
            'observed_synergy_scores': observed_synergy_scores,
            'synergy_method': 'bliss',
            # 'attractor_tool': 'mpbn'
            # 'attractor_type':  'trapspaces'
        }
    )

    # Statistics
    # evolution_calibrated = Evolution(
    #     boolean_model=boolean_model_bnet,
    #     model_outputs=model_outputs,
    #     training_data=training_data,
    #     ga_args=ga_args,
    #     ev_args=ev_args
    # )
    # best_boolean_models_calibrated = evolution_calibrated.run()
    #
    # evolution_random = Evolution(
    #         boolean_model=boolean_model_bnet,
    #         model_outputs=model_outputs,
    #         ga_args=ga_args,
    #         ev_args=ev_args
    #     )
    # best_boolean_models_random = evolution_random.run()
    #
    # statistics = Statistics(best_boolean_models_calibrated, observed_synergy_scores,
    #                         model_outputs, perturbations, 'bliss')
    # statistics.compare_two_simulations(best_boolean_models_calibrated, best_boolean_models_random,
    #                                    'Calibrated (Non-Normalized)', 'Random')
    # statistics.sampling_with_ci(repeat_time=15, plot_discrete=False)
