from src.problems.base.components import BaseSolution, BaseOperator

class Solution(BaseSolution):
    """The solution of online BPP.
    A list of bins where each bin is represented by a list of items.
    Each sublist contains integers representing the items packed in the bin in the order of insertion.
    """
    def __init__(self, sequences: list[list[int]], current_item: int):
        self.sequences = sequences
        self.current_item = current_item

    def __str__(self) -> str:
        sequences_string = ""
        for index, items in enumerate(self.sequences):
            if items == []:
                continue
            sequences_string += f"bin_{index}: " + ",".join(map(str, items)) + "\n"
        return sequences_string


class AssignBinOperator(BaseOperator):
    """AssignOperator is responsible for assigning an item to a specific bin in the packing sequence.
    This operator appends the given item to the list of items in the specified bin within the solution. 
    The item number is provided externally and is not determined within the operator itself.
    """
    def __init__(self, bin: int):
        self.bin = bin

    def run(self, solution: Solution) -> Solution:
        assert self.bin < len(solution.sequences)
        new_sequences = solution.sequences[:]
        new_sequences[self.bin].append(solution.current_item)
        return Solution(new_sequences, solution.current_item + 1)

class NewBinOperator(BaseOperator):
    """NewBinOperator is responsible for assigning an item to a new bin without any items before.
    """
    def __init__(self):
        pass

    def run(self, solution: Solution) -> Solution:
        new_sequences = solution.sequences[:]
        for sequence in new_sequences:
            if sequence == []:
                sequence.append(solution.current_item)
                break
        return Solution(new_sequences, solution.current_item + 1)