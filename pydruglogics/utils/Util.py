import os
from typing import List


class Util:
    @staticmethod
    def get_file_extension(file_name: str) -> str:
        try:
            if file_name:
                extension = file_name[file_name.rfind('.') + 1:]
            else:
                extension = ''
        except Exception as ex:
            raise ex
        return extension

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
    def is_numeric_string(value):
        try:
            float(value)
            return True
        except ValueError:
            return False

    @staticmethod
    def parse_interaction(interaction: str) -> dict:
        tmp = interaction.split()
        if len(tmp) != 3:
            raise ValueError(f"ERROR: Wrongly formatted interaction: {interaction}")
        source = tmp[0]
        interaction_type = tmp[1]
        target = tmp[2]

        if interaction_type in ['activate', 'activates', '->']:
            arc = 1
        elif interaction_type in ['inhibit', 'inhibits', '-|']:
            arc = -1
        elif interaction_type in ['<-', '|-']:
            arc = 1 if interaction_type == '<-' else -1
        else:
            print('ERROR: Wrongly formatted interaction type:')
            print(f"Source: {source} Interaction type: {interaction_type} Target: {target}")
            raise SystemExit(1)

        return {
            'source': source,
            'target': target,
            'arc': arc,
            'activating_regulators': [],
            'inhibitory_regulators': [],
            'activating_regulator_complexes': [],
            'inhibitory_regulator_complexes': []
        }

    @staticmethod
    def create_interaction(target: str) -> dict:
        return {
            'target': target,
            'activating_regulators': [],
            'inhibitory_regulators': [],
            'activating_regulator_complexes': [],
            'inhibitory_regulator_complexes': []
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
