import logging
import os
import timeit
import cProfile
import pstats
import io
from datetime import datetime
import psutil
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from pydruglogics.input.Perturbations import Perturbation
from pydruglogics.input.TrainingData import TrainingData
from pydruglogics.model.BooleanModel import BooleanModel
from pydruglogics.model.InteractionModel import InteractionModel
from pydruglogics.input.ModelOutputs import ModelOutputs
from pydruglogics.execution.Executor import execute

def run_project():
    interaction = InteractionModel(
        interactions_file='../ags_cascade_1.0/network.sif',
        remove_self_regulated_interactions=False,
        remove_inputs=False,
        remove_outputs=False
    )

    model_outputs = ModelOutputs(input_file='../ags_cascade_1.0/modeloutputs')

    observations = [
        (["CASP3:0", "CASP8:0", "CASP9:0", "FOXO_f:0", "RSK_f:1", "CCND1:1",
          "MYC:1", "RAC_f:1", "JNK_f:0", "MAPK14:0", "AKT_f:1", "MMP_f:1",
          "PTEN:0", "ERK_f:1", "KRAS:1", "PIK3CA:1", "S6K_f:1", "GSK3_f:0",
          "TP53:0", "BAX:0", "BCL2:1", "CTNNB1:1", "TCF7_f:1", "NFKB_f:1"], 1.0)
    ]
    training_data = TrainingData(observations=observations)

    drug_panel_data = [
        ['PI', 'PIK3CA', 'inhibits'],
        ['PD', 'MEK_f'],
        ['CT', 'GSK3_f'],
        ['BI', 'MAPK14'],
        ['PK', 'CTNNB1'],
        ['AK', 'AKT_f'],
        ['5Z', 'MAP3K7']
    ]
    perturbation_data = [
        ['PI'], ['PD'], ['CT'], ['BI'], ['PK'], ['AK'], ['5Z'], ['PI', 'PD'], ['PI', 'CT'], ['PI', 'BI'],
        ['PI', 'PK'], ['PI', 'AK'], ['PI', '5Z'], ['PD', 'CT'], ['PD', 'BI'], ['PD', 'PK'], ['PD', 'AK'],
        ['PD', '5Z'], ['CT', 'BI'], ['CT', 'PK'], ['CT', 'AK'], ['CT', '5Z'], ['BI', 'PK'], ['BI', 'AK'],
        ['BI', '5Z'], ['PK', 'AK'], ['PK', '5Z'], ['AK', '5Z']
    ]
    perturbations = Perturbation(drug_data=drug_panel_data, perturbation_data=perturbation_data)

    boolean_model_bnet = BooleanModel(
        file='../ags_cascade_1.0/network.bnet',
        model_name='test',
        mutation_type='balanced',
        attractor_tool='mpbn',
        attractor_type='trapspaces'
    )

    ga_args = {
        'num_generations': 20,
        'num_parents_mating': 3,
        'mutation_num_genes': 10,
        'fitness_batch_size': 20,
        'gene_type': int,
        'crossover_type': 'single_point',
        'mutation_type': 'random',
        'keep_elitism': 6
    }
    ev_args = {
        'num_best_solutions': 3,
        'num_of_runs': 50,
        'num_of_cores': 4,
        'num_of_init_mutation': 20
    }
    observed_synergy_scores = ["PI-PD", "PI-5Z", "PD-AK", "AK-5Z"]

    train_params = {
        'boolean_model': boolean_model_bnet,
        'model_outputs': model_outputs,
        'training_data': training_data,
        'ga_args': ga_args,
        'ev_args': ev_args,
        'save_best_models': True
    }
    predict_params = {
        'perturbations': perturbations,
        'model_outputs': model_outputs,
        'observed_synergy_scores': observed_synergy_scores,
        'synergy_method': 'bliss',
        'save_predictions': True,
        'save_path': './predictions'
    }

    execute(train_params=train_params, predict_params=predict_params)

def monitor_cpu_usage(duration, interval=0.1):
    cpu_usage = []
    for _ in range(int(duration / interval)):
        cpu_usage.append(psutil.cpu_percent(interval=interval))
    return cpu_usage

def create_results_folder(base_folder="benchmark_results"):
    benchmark_dir = os.path.dirname(os.path.abspath(__file__))
    results_folder = os.path.join(benchmark_dir, base_folder)

    os.makedirs(results_folder, exist_ok=True)

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_folder = os.path.join(results_folder, current_time)
    os.makedirs(run_folder, exist_ok=True)

    return run_folder

def benchmark(runs=5, results_folder=None):
    if results_folder is None:
        results_folder = create_results_folder()

    results = []
    cpu_results = []

    for run in range(runs):
        start_time = timeit.default_timer()

        with ThreadPoolExecutor() as executor:
            future_cpu = executor.submit(monitor_cpu_usage, 30)

            profiler = cProfile.Profile()
            profiler.enable()

            run_project()

            profiler.disable()
            elapsed_time = timeit.default_timer() - start_time
            cpu_usage = future_cpu.result()

        iostream = io.StringIO()
        pst_cumulative = pstats.Stats(profiler, stream=iostream).sort_stats('cumulative')
        pst_cumulative.print_stats()
        profile_path = os.path.join(results_folder, f'profile_run_{run + 1}.txt')
        with open(profile_path, 'w') as f:
            f.write(iostream.getvalue())

        results.append(elapsed_time)
        cpu_results.append(cpu_usage)

        logging.info(f"Run {run + 1} completed in {elapsed_time:.2f} seconds.")

    plot_results(results, cpu_results, results_folder)

    return results, cpu_results

def plot_results(runtime_results, cpu_results, results_folder):
    plt.figure(figsize=(10, 5))
    plt.title('Runtime per Simulation')
    plt.plot(range(1, len(runtime_results) + 1), runtime_results, marker='o', linestyle='-', label='Runtime (sec)')
    plt.xlabel('Run')
    plt.ylabel('Time (sec)')
    plt.grid(True, which='both', linestyle='--', color='lightgrey')
    plt.legend()
    runtime_plot_path = os.path.join(results_folder, "runtime_per_simulation.png")
    plt.savefig(runtime_plot_path)
    plt.show()
    plot_cpu_usage(cpu_results, 'All Runs Combined', results_folder)

def plot_single_run(runtime_results, setup_label, results_folder):
    plt.figure(figsize=(10, 5))
    x_labels = [f'Run {i}' for i in range(1, len(runtime_results) + 1)]
    bars = plt.bar(x_labels, runtime_results, color='skyblue')

    for bar, runtime in zip(bars, runtime_results):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{runtime:.2f} s',
                 ha='center', va='bottom', fontsize=10)

    plt.title(f'Runtime per Simulation for {setup_label}')
    plt.xlabel('Run')
    plt.ylabel('Time (seconds)')
    plt.grid(True, axis='y', linestyle='--', color='lightgrey')
    plt.yticks(fontsize=10)

    runtime_bar_chart_path = os.path.join(results_folder, f"runtime_bar_chart_{setup_label.replace(' ', '_').lower()}.png")
    plt.savefig(runtime_bar_chart_path)
    plt.show()

def plot_cpu_usage(cpu_results, setup_label, results_folder):
    plt.figure(figsize=(12, 6))
    colors = plt.colormaps['tab10'].colors

    for idx, cpu_run in enumerate(cpu_results):
        color = colors[idx % len(colors)]
        plt.plot(cpu_run, label=f'Run {idx + 1}', color=color)

    plt.title(f'CPU Usage During All Runs for {setup_label}')
    plt.xlabel('Time (sec)')
    plt.ylabel('CPU Usage (%)')
    plt.grid(True)
    plt.legend()
    cpu_usage_path = os.path.join(results_folder, f"cpu_usage_combined_{setup_label.replace(' ', '_').lower()}.png")
    plt.savefig(cpu_usage_path)
    plt.show()

def benchmark_multiple_setups(run_param_configs, setup_labels=None, runs=5):
    all_results = {}
    results_folder = create_results_folder()

    if setup_labels is None:
        setup_labels = [f'Setup {i + 1}' for i in range(len(run_param_configs))]

    for idx, (setup, label) in enumerate(zip(run_param_configs, setup_labels)):
        logging.info(f"\nBenchmarking {label}...")

        runtime_results, cpu_results = benchmark(runs=runs, results_folder=results_folder)
        all_results[label] = runtime_results

        plot_single_run(runtime_results, label, results_folder)
        plot_cpu_usage(cpu_results, label, results_folder)

    plt.figure(figsize=(12, 6))
    plt.title('Runtime Comparison for Different Setups')
    plt.boxplot(list(all_results.values()), tick_labels=list(all_results.keys()))
    plt.xlabel('Setup')
    plt.ylabel('Time (seconds)')
    plt.grid(True, which='both', linestyle='--', color='lightgrey')
    boxplot_path = os.path.join(results_folder, "runtime_comparison_boxplot_across_setups.png")
    plt.savefig(boxplot_path)
    plt.show()

    return all_results

# Add different configs for the benchmark
setup_configs = [
    {
        'ga_args': {
            'num_generations': 20,
            'num_parents_mating': 3,
            'mutation_num_genes': 10,
            'fitness_batch_size': 20,
            'gene_type': int,
            'crossover_type': 'single_point',
            'mutation_type': 'random',
            'keep_elitism': 6
        },
        'ev_args': {
            'num_best_solutions': 3,
            'num_of_runs': 50,
            'num_of_cores': 4,
            'num_of_init_mutation': 20
        }
    },
    {
        'ga_args': {
            'num_generations': 25,
            'num_parents_mating': 5,
            'mutation_num_genes': 15,
            'fitness_batch_size': 15,
            'gene_type': int,
            'crossover_type': 'two_point',
            'mutation_type': 'adaptive',
            'keep_elitism': 8
        },
        'ev_args': {
            'num_best_solutions': 5,
            'num_of_runs': 60,
            'num_of_cores': 2,
            'num_of_init_mutation': 25
        }
    }
]

# Run the benchmark
runtime_results, cpu_results = benchmark(runs=3)
# all_runtime_results = benchmark_multiple_setups(setup_configs, runs=2)
