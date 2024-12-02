from src.problems.base.components import BaseSolution, BaseOperator

class Solution(BaseSolution):
    """The solution of DPOSP.
        A list of lists, where each sublist corresponds to a production line's planned sequence of orders.
        Each integer within the sublist represents an order to be fulfilled by the machine, in the sequence they are to be produced.
    """
    def __init__(self, production_schedule: list[list[int]]):
        self.production_schedule = production_schedule

    def __str__(self) -> str:
        production_schedules_str = ""
        for index, production_schedule in enumerate(self.production_schedule):
            production_schedules_str += f"production_line_{index}: " + "->".join(map(str, production_schedule)) + "\n"
        return production_schedules_str


class AppendOperator(BaseOperator):
    """Appends an order to the end of a specified production line's schedule."""
    def __init__(self, production_line_id: int, order_id: int):
        self.production_line_id = production_line_id
        self.order_id = order_id

    def run(self, solution: Solution) -> Solution:
        new_schedule = [order[:] for order in solution.production_schedule]
        new_schedule[self.production_line_id].append(self.order_id)
        return Solution(new_schedule)


class InsertOperator(BaseOperator):
    """Inserts an order at a specified position within a production line's schedule."""
    def __init__(self, production_line_id: int, order_id: int, position: int):
        self.production_line_id = production_line_id
        self.order_id = order_id
        self.position = position

    def run(self, solution: Solution) -> Solution:
        new_schedule = [order[:] for order in solution.production_schedule]
        new_schedule[self.production_line_id].insert(self.position, self.order_id)
        return Solution(new_schedule)


class SwapOperator(BaseOperator):
    """Swaps two orders between or within production lines."""
    def __init__(self, production_line_id1: int, position1: int, production_line_id2: int, position2: int):
        if production_line_id1 <= production_line_id2:
            self.production_line_id1 = production_line_id1
            self.position1 = position1
            self.production_line_id2 = production_line_id2
            self.position2 = position2
        else:
            self.production_line_id1 = production_line_id2
            self.position1 = position2
            self.production_line_id2 = production_line_id1
            self.position2 = position1

    def run(self, solution: Solution) -> Solution:
        new_schedule = [order[:] for order in solution.production_schedule]
        new_schedule[self.production_line_id1][self.position1], new_schedule[self.production_line_id2][self.position2] = new_schedule[self.production_line_id2][self.position2], new_schedule[self.production_line_id1][self.position1]
        return Solution(new_schedule)


class ReverseSegmentOperator(BaseOperator):
    """Reverses a segment of orders within a production line's schedule."""
    def __init__(self, production_line_id: int, segments: list[tuple[int, int]]):
        self.production_line_id = production_line_id
        self.segments = segments

    def run(self, solution: Solution) -> Solution:
        new_schedule = [order[:] for order in solution.production_schedule]

        for segment in self.segments:
            start_index, end_index = segment

            # Ensure the indices are within the bounds of the tour list
            assert 0 <= start_index < len(new_schedule[self.production_line_id])
            assert 0 <= end_index < len(new_schedule[self.production_line_id])

            # Ensure the start index comes before the end index
            if start_index > end_index:
                start_index, end_index = end_index, start_index

            # Reverse the segment between start_index and end_index (inclusive)
            new_schedule[self.production_line_id][start_index:end_index + 1] = reversed(new_schedule[self.production_line_id][start_index:end_index + 1])

        return Solution(new_schedule)


class RelocateOperator(BaseOperator):
    """Moves an order from one position to another within the same or to a different production line."""
    def __init__(self, source_production_line_id: int, source_position: int, target_production_line_id: int, target_position: int):
        self.source_production_line_id = source_production_line_id
        self.source_position = source_position
        self.target_production_line_id = target_production_line_id
        self.target_position = target_position

    def run(self, solution: Solution) -> Solution:
        new_schedule = [order[:] for order in solution.production_schedule]
        order = new_schedule[self.source_production_line_id].pop(self.source_position)
        # If target schedule is empty, append the order directly
        if not new_schedule[self.target_production_line_id]:
            new_schedule[self.target_production_line_id].append(order)
        else:
            # Insert order into target schedule at the specified position
            new_schedule[self.target_production_line_id].insert(self.target_position, order)
        return Solution(new_schedule)


class MergeOperator(BaseOperator):
    """Merges the schedule of a source production line into the beginning of a target production line's schedule. 
    The merged schedule is assigned to the target production line, and the source production line's schedule is cleared."""
    def __init__(self, source_production_line_id: int, target_production_line_id: int):
        self.source_production_line_id = source_production_line_id
        self.target_production_line_id = target_production_line_id

    def run(self, solution: Solution) -> Solution:
        new_schedule = [order[:] for order in solution.production_schedule]
        # Append source schedule to target schedule, then clear the source schedule
        new_schedule[self.target_production_line_id] = new_schedule[self.source_production_line_id] + new_schedule[self.target_production_line_id]
        new_schedule[self.source_production_line_id] = []
        return Solution(new_schedule)
