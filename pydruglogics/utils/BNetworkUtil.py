import os
import re
from typing import List
import logging


class BNetworkUtil:

    @staticmethod
    def get_file_extension(file_name: str) -> str:
        if not file_name or '.' not in file_name:
            logging.warning("No extension found in the provided file.")
            return ''

        return file_name.rsplit('.', 1)[-1]

    @staticmethod
    def remove_extension(file_ext: str) -> str:
        file_name = os.path.basename(file_ext)
        name_without_extension = file_name.rsplit('.', 1)[0]
        if name_without_extension:
            return name_without_extension
        return file_name

    @staticmethod
    def read_lines_from_file(file_name: str, skip_empty_lines_and_comments: bool = True) -> List[str]:
        lines = []
        with open(file_name, 'r') as file:
            for line in file:
                line = line.strip()
                if skip_empty_lines_and_comments:
                    if not line.startswith('#') and len(line) > 0:
                        lines.append(line)
                else:
                    lines.append(line)
        return lines

    @staticmethod
    def is_numeric_string(value: str) -> bool:
        if isinstance(value, (int, float)):
            return True
        if isinstance(value, str):
            try:
                float(value)
                return True
            except ValueError:
                return False
        return False

    @staticmethod
    def parse_interaction(interaction: str) -> dict:
        components = interaction.split()

        if len(components) != 3:
            logging.error(f"Invalid interaction format: '{interaction}'. Expected format: "
                          f"'source interaction_type target'")
            raise ValueError(f"ERROR: Wrongly formatted interaction: {interaction}")


        source, interaction_type, target = components
        interaction_map = {
            'activate': 1, 'activates': 1, '->': 1,
            'inhibit': -1, 'inhibits': -1, '-|': -1,
            '<-': 1, '|-': -1
        }

        arc = interaction_map.get(interaction_type)
        if arc is None:
            logging.error(f"Invalid interaction type '{interaction_type}' in interaction '{interaction}'")
            raise ValueError(f"ERROR: Unrecognized interaction type: {interaction_type}")

        return {
            'source': source,
            'target': target,
            'arc': arc,
            'activating_regulators': [],
            'inhibitory_regulators': []
        }

    @staticmethod
    def create_interaction(target: str) -> dict:
        return {
            'target': target,
            'activating_regulators': [],
            'inhibitory_regulators': []
        }

    @staticmethod
    def bnet_string_to_dict(bnet_string: str):
        lines = [line.strip() for line in bnet_string.split('\n') if line.strip()]
        result = {}
        for line in lines:
            node, definition = line.split(',', 1)
            node = node.strip()
            definition = definition.strip()
            result[node] = definition

        return result

    @staticmethod
    def to_bnet_format(boolean_equations):
        """
        Converts Boolean equations to the '.bnet' format.
        :param boolean_equations: Boolean equations to be converted.
        :return: Boolean Equations in BNet format.
        """
        equation_list = []

        for eq in boolean_equations:
            target, activating_regulators, inhibitory_regulators, link = eq

            target_value = f"{target}, "

            activation_terms = [regulator for regulator, value in activating_regulators.items() if value == 1]
            if activation_terms:
                activation_expression = f"({activation_terms[0]})"
                for reg in activation_terms[1:]:
                    activation_expression = f"({activation_expression} | {reg})"
            else:
                activation_expression = ''

            inhibition_terms = [regulator for regulator, value in inhibitory_regulators.items() if value == 1]
            if inhibition_terms:
                inhibition_expression = f"({inhibition_terms[0]})"
                for reg in inhibition_terms[1:]:
                    inhibition_expression = f"({inhibition_expression} | {reg})"
            else:
                inhibition_expression = ''

            if activation_expression and inhibition_expression:
                combined_expression = f"{activation_expression} {link} !{inhibition_expression}"
            elif activation_expression or inhibition_expression:
                combined_expression = activation_expression if activation_expression else f"!{inhibition_expression}"
            else:
                combined_expression = '0'

            equation_line = f"{target_value}{combined_expression}".strip()
            equation_list.append(equation_line)

        final_equation_list = '\n'.join(equation_list)
        return final_equation_list

    @staticmethod
    def create_equation_from_bnet(equation_str):
        equation = equation_str.strip()
        target, regulators = equation.split(',', 1)
        target = target.strip()
        activating_regulators = {}
        inhibitory_regulators = {}
        link = ''

        regulators = regulators.replace('(', '').replace(')', '')
        if '!' in regulators:
            parts = regulators.split('!')
            inhibitory_part = parts[1].strip()
            before_inhibitory = parts[0].strip()
            if '&' in before_inhibitory:
                link = '&'
            elif '|' in before_inhibitory:
                link = '|'

            inhibitory_nodes = re.split(r'[|&]', inhibitory_part)
            for node in inhibitory_nodes:
                node = node.strip()
                if node:
                    inhibitory_regulators[node] = 1

            activating_nodes = re.split(r'[|&]', before_inhibitory)
            for node in activating_nodes:
                node = node.strip()
                if node:
                    activating_regulators[node] = 1
        else:
            activating_nodes = re.split(r'[|&]', regulators)
            for node in activating_nodes:
                node = node.strip()
                if node:
                    activating_regulators[node] = 1

        return target, activating_regulators, inhibitory_regulators, link,
