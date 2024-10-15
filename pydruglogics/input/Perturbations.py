import itertools
from typing import List, Dict, Union
import logging


class Perturbation:
    def __init__(self, drug_data: List[List[Union[str, None]]] = None, perturbation_data: List[List[str]] = None):
        self._drug_panel = []
        self._perturbations = []
        self._drug_perturbations = []

        if drug_data is not None:
            self._load_drug_panel_from_data(drug_data)
        else:
            raise ValueError('Please provide drug data.')

        if perturbation_data is not None:
            self._load_perturbations_from_data(perturbation_data)
        else:
            self._init_perturbations_from_drugpanel()

    def _load_drug_panel_from_data(self, drug_data: List[List[Union[str, None]]]) -> None:
        try:
            for drug in drug_data:
                if len(drug) < 2:
                    raise ValueError("Each drug entry must contain at least 'name' and 'targets'.")

                name, targets, effect = drug[0], drug[1], drug[2] if len(drug) > 2 else 'inhibits'

                if not name or not targets:
                    raise ValueError("Each drug entry must contain 'name' and 'targets'.")

                self._drug_panel.append({
                    'name': name,
                    'targets': targets.split(',') if isinstance(targets, str) else targets,
                    'effect': effect
                })

            logging.info(f"Drug panel data initialized from list.")

        except Exception as e:
            logging.error(f"Error while loading drug panel data: {str(e)}")
            raise

    def _load_perturbations_from_data(self, perturbation_data: List[List[str]]) -> None:
        try:
            if not all(perturbation_data):
                raise ValueError('Each perturbation entry must contain at least one perturbation.')

            self._drug_perturbations = perturbation_data
            self._init_drug_perturbations()
            logging.info(f"Perturbations initialized from list.")

        except ValueError as ve:
            logging.error(f"Error loading perturbation data: {str(ve)}")
            raise
        except Exception as e:
            logging.error(f"Error while loading perturbation data: {str(e)}")
            raise

    def _init_drug_perturbations(self) -> None:
        try:
            name_to_drug = {drug['name']: drug for drug in self._drug_panel}
            perturbed_drugs = []

            for combination in self._drug_perturbations:
                combo = [name_to_drug.get(name) for name in combination if name in name_to_drug]
                if combo:
                    perturbed_drugs.append(combo)
                else:
                    logging.warning('Some drugs in the perturbation were not found in the drug panel.')

            self._perturbations = perturbed_drugs
            logging.info(f"Drug perturbations loaded.")

        except Exception as e:
            logging.error(f"Error during drug perturbation initialization: {str(e)}")
            raise

    def _init_perturbations_from_drugpanel(self):
        try:
            self._perturbations = [
                list(combination)
                for number_of_combination in range(1, 3)
                for combination in itertools.combinations(self._drug_panel, number_of_combination)
            ]
            logging.info(f"Perturbations generated from drug panel.")

        except Exception as e:
            logging.error(f"Error initializing perturbations from drug panel: {str(e)}")
            raise

    @property
    def drugs(self) -> List[Dict[str, str]]:
        return self._drug_panel

    @property
    def perturbations(self) -> List[List[Dict[str, str]]]:
        return self._perturbations

    @property
    def drug_names(self) -> List[str]:
        return [drug['name'] for drug in self._drug_panel]

    @property
    def drug_effects(self) -> List[str]:
        return [drug['effect'] for drug in self._drug_panel]

    @property
    def drug_targets(self) -> List[List[str]]:
        return [drug['targets'] for drug in self._drug_panel]

    def print(self) -> None:
        try:
            print(str(self))
        except Exception as e:
            print(f"An error occurred while printing Perturbation: {e}")

    def __str__(self) -> str:
        if not self._perturbations:
            return 'No perturbations available.'
        perturbations_str = []
        for perturbation in self._perturbations:
            combo_str = ', '.join([f"{drug['name']} (targets: {', '.join(drug['targets'])})" for drug in perturbation])
            perturbations_str.append(f"[{combo_str}]")
        return '\n'.join(perturbations_str)
