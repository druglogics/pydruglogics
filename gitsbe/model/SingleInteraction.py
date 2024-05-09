class SingleInteraction:
    def __init__(self, interaction, first_node=None, second_node=None):
        self.set_interaction(first_node, interaction, second_node)

    def get_interaction(self) -> str:
        return f"{self.source} {'->' if self.arc == 1 else '-|'} {self.target}"

    def get_source(self) -> str:
        return self.source

    def get_target(self) -> str:
        return self.target

    def get_arc(self) -> int:
        return self.arc

    def set_interaction(self, first_node, interaction, second_node) -> None:
        if first_node is None and second_node is None:
            tmp = interaction.split('\t')
            if len(tmp) != 3:
                raise ValueError(f"ERROR: Wrongly formatted interaction: {interaction}")
            first_node = tmp[0]
            interaction = tmp[1]
            second_node = tmp[2]

        match interaction:
            case 'activate' | 'activates' | '->':
                self.arc = 1
                self.source = first_node
                self.target = second_node
            case 'inhibit' | 'inhibits' | '-|':
                self.arc = -1
                self.source = first_node
                self.target = second_node
            case '<-':
                self.arc = 1
                self.source = first_node
                self.target = second_node
            case '|-':
                self.arc = -1
                self.source = first_node
                self.target = second_node
            case '|->' | '<->' | '<-|' | '|-|':
                print('ERROR: Wrongly formatted interaction type:')
                print(f"Source: {first_node} Interaction type: {interaction} Target: {second_node}")
                raise SystemExit(1)

    def __str__(self):
        return f"{self.source} {self.arc} {self.target} "
