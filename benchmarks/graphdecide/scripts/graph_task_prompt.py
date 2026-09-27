"""Oracle-free, wire-neutral graph questions and explicit action meanings."""

from dataclasses import replace

from src.clients.base import DecisionOption


def state_explicit_tsp_request(request):
    question = request.question + (
        f" The already-visited prefix is {request.state['partial_tour']}. "
        f"You are NOW at vertex {request.state['current_vertex']}, not restarting "
        "at vertex 0. Select the NEXT vertex reached from this current endpoint. "
        "Account for every remaining unvisited vertex before returning to 0; "
        "the selected vertex is not necessarily the final vertex before 0. "
        "Do not choose a vertex merely because it is closest to vertex 0.")
    return replace(request, question=question)


def canonical_graph_request(task, request):
    if task == "degree_exact":
        suffix = " Return only its decimal integer option ID."
    elif task in ("maxcut_construct", "community_bipartition_construct"):
        suffix = " Return only A or B."
    elif task == "tsp_construct":
        suffix = " Return only the chosen vertex's decimal option ID."
    elif task == "lt_influence_construct":
        suffix = " and return only one remaining vertex's decimal option ID."
    elif task in ("adjacency", "cycle_detection", "pair_connectivity",
                  "distance_threshold", "articulation_point"):
        suffix = " Return only yes or no."
    else:
        raise ValueError("Unknown graph task for canonical rendering")
    if not request.question.endswith(suffix):
        raise ValueError("Task wording changed; review the response-format boundary")
    question = request.question[:-len(suffix)]
    if task == "lt_influence_construct":
        question += ". Select exactly one remaining vertex as the next seed."
    options = []
    for option in request.options:
        if task == "degree_exact":
            description = f"Degree {option.id}: exactly {option.id} distinct neighbors."
        elif task in ("maxcut_construct", "community_bipartition_construct"):
            description = f"Assign vertex {request.state['current_vertex']} to side {option.id}."
        elif task == "tsp_construct":
            description = f"Visit vertex {option.id} next."
        elif task == "lt_influence_construct":
            description = f"Select vertex {option.id} as a new seed."
        else:
            description = ("Yes: the stated graph property holds." if option.id == "yes"
                           else "No: the stated graph property does not hold.")
        options.append(DecisionOption(option.id, description))
    result = replace(request, question=question, options=tuple(options))
    return state_explicit_tsp_request(result) if task == "tsp_construct" else result
