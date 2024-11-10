import logging
from pydruglogics.input.Perturbations import Perturbation
from pydruglogics.input.TrainingData import TrainingData
from pydruglogics.model.BooleanModel import BooleanModel
from pydruglogics.model.Evolution import Evolution
from pydruglogics.model.InteractionModel import InteractionModel
from pydruglogics.input.ModelOutputs import ModelOutputs
from pydruglogics.execution.Executor import execute, train
from pydruglogics.model.ModelPredictions import ModelPredictions
from pydruglogics.model.Statistics import compare_two_simulations, sampling_with_ci
from pydruglogics.utils.Logger import Logger
from pydruglogics.utils.PlotUtil import PlotUtil

if __name__ == '__main__':
    # Interaction
    interaction = InteractionModel(interactions_file='../ags_cascade_1.0/network.sif',
                                   remove_self_regulated_interactions=False, remove_inputs=False, remove_outputs=False)
    interaction.print()

    # ModelOutputs
    model_outputs = ModelOutputs(input_file='../ags_cascade_1.0/modeloutputs')
    # model_outputs_dict = {
    #     "RPS6KA1": 1.0,
    #     "MYC": 1.0,
    #     "TCF7": 1.0,
    #     "CASP8": -1.0,
    #     "CASP9": -1.0,
    #     "FOXO3": 1.0
    # }
    # model_outputs = ModelOutputs(input_file=model_outputs_dict)
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
                                      mutation_type='balanced', attractor_tool='pyboolnet', attractor_type='stable_states')


    # BooleanModel init from .sif file
    boolean_model_sif = BooleanModel(model=interaction, model_name='test2',
                                     mutation_type='balanced', attractor_tool='mpbn', attractor_type='trapspaces')

    # init pygad.GA
    ga_args = {
        'num_generations': 20,
        'num_parents_mating': 3,
        'mutation_num_genes': 10,
        'fitness_batch_size': 20,
        'gene_type': int,
        'crossover_type': 'single_point',
        'mutation_type': 'random',
        'keep_elitism':6
    }

    # init evolution params
    ev_args = {
        'num_best_solutions': 3,
        'num_of_runs': 50,
        'num_of_cores': 4,
        'num_of_init_mutation': 20
    }

    # init observed synergy scores
    observed_synergy_scores = ["PI-PD", "PI-5Z", "PD-AK", "AK-5Z"]

    train_params = {
        'boolean_model': boolean_model_bnet,
        'model_outputs': model_outputs,
        'training_data': training_data,
        'ga_args': ga_args,
        'ev_args': ev_args,
        'save_best_models': True,
        # 'save_path': './models'
    }
    predict_params = {
        'perturbations': perturbations,
        'model_outputs': model_outputs,
        'observed_synergy_scores': observed_synergy_scores,
        'synergy_method': 'bliss',
        'save_predictions': True,
        'save_path': './predictions',
        # 'model_directory': './models/example_models',
        # 'attractor_tool': 'mpbn',
        # 'attractor_type':  'stable_states'
    }
    execute(train_params=train_params, predict_params=predict_params)

    # Statistics
    # evolution_calibrated = Evolution(
    #     boolean_model=boolean_model_bnet,
    #     model_outputs=model_outputs,
    #     training_data=training_data,
    #     ga_args=ga_args,
    #     ev_args=ev_args
    # )
    # best_boolean_models_calibrated = train(boolean_model=boolean_model_bnet,model_outputs=model_outputs,
    #                                        training_data=training_data, ga_args=ga_args, ev_args=ev_args)
    #
    # best_boolean_models_random = train(boolean_model=boolean_model_bnet, model_outputs=model_outputs,
    #                                    ga_args=ga_args, ev_args=ev_args)
    #
    # compare_two_simulations(best_boolean_models_calibrated, best_boolean_models_random, observed_synergy_scores,
    #                         model_outputs,perturbations, 'bliss','Calibrated (Non-Normalized)',
    #                         'Random', False)
    # sampling_with_ci(best_boolean_models_calibrated, observed_synergy_scores, model_outputs, perturbations,
    #                  repeat_time=15, sub_ratio=0.8, boot_n=1000, plot_discrete=False, with_seeds=True)
