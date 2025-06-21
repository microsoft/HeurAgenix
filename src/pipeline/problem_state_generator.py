import os
import yaml
import importlib
import traceback
from src.util.util import extract, load_function, parse_text_to_dict, search_file
from src.util.llm_client.base_llm_client import BaseLLMClient


class ProblemStateGenerator:
    def __init__(
        self,
        llm_client: BaseLLMClient,
        problem: str
    ) -> None:
        self.llm_client = llm_client
        self.problem = problem
        self.output_dir = self.llm_client.output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_problem_state(self, smoke_test: bool=False, max_try_times: int=5) -> str:
        prompt_dict = self.llm_client.load_background(self.problem, "background_without_code")
        problem_state_description_file = search_file("problem_state_description.txt", problem=self.problem)
        assert problem_state_description_file is not None, f"Problem state description file {problem_state_description_file} does not exist"
        problem_state_description = parse_text_to_dict(open(problem_state_description_file).read())

        prompt_dict["instance_data_introduction"] = problem_state_description["instance_data"]
        prompt_dict["key_item"] = problem_state_description["key_item"].split(":")[0].split("(")[0].replace("-", "").replace(" ", "").replace("\"", "").replace("\"", "")
        prompt_dict["key_item_description"] = problem_state_description["key_item"].split(":")[-1]

        # Get instance problem state
        self.llm_client.load("generate_instance_problem_state", prompt_dict)
        response = self.llm_client.chat()
        instance_problem_states = extract(response, "instance_problem_state", "\n")
        instance_problem_states = ",".join([instance_problem_state.split(";")[0] for instance_problem_state in instance_problem_states])
        prompt_dict["instance_problem_states"] = instance_problem_states

        # Generate instance problem state code
        self.llm_client.load("implement_instance_problem_state_code", prompt_dict)
        response = self.llm_client.chat()
        instance_problem_state_code = extract(response, "python_code")
        self.llm_client.dump(f"instance_problem_state")

        # Get solution problem state
        self.llm_client.load("generate_solution_problem_state", prompt_dict)
        response = self.llm_client.chat()
        solution_problem_states = extract(response, "solution_problem_state", "\n")
        solution_problem_states = ",".join([solution_problem_state.split(";")[0] for solution_problem_state in solution_problem_states])
        prompt_dict["solution_problem_states"] = solution_problem_states

        # Generate solution problem state code
        self.llm_client.load("implement_solution_problem_state_code", prompt_dict)
        response = self.llm_client.chat()
        solution_problem_state_code = extract(response, "python_code")
        solution_problem_state_code = f"from src.problems.{self.problem}.components import Solution\n" + solution_problem_state_code
        self.llm_client.dump(f"solution_problem_state")

        # Get observation problem state
        self.llm_client.load("generate_observation_problem_state", prompt_dict)
        response = self.llm_client.chat()
        observation_problem_states = extract(response, "observation_problem_state", "\n")
        observation_problem_states = ",".join([observation_problem_state.split(";")[0] for observation_problem_state in observation_problem_states])
        prompt_dict["observation_problem_states"] = observation_problem_states

        # Generate observation problem state code
        self.llm_client.load("implement_observation_problem_state_code", prompt_dict)
        response = self.llm_client.chat()
        observation_problem_state_code = extract(response, "python_code")
        self.llm_client.dump(f"observation_problem_state")

        # Verify and revision code
        if smoke_test:
            for _ in range(max_try_times):
                instance_error_message, solution_error_message, observation_error_message = self.smoke_test(instance_problem_state_code, solution_problem_state_code, observation_problem_state_code)
                while instance_error_message or solution_error_message or observation_error_message:
                    self.llm_client.dump(f"problem_state_revision")
                    print(f"Revision problem state code: {instance_error_message}, {solution_error_message}, {observation_error_message}")
                    if instance_error_message:
                        self.llm_client.load(instance_error_message)
                        response = self.llm_client.chat()
                        instance_problem_state_code = extract(response, "python_code")
                    if solution_error_message:
                        self.llm_client.load(solution_error_message)
                        response = self.llm_client.chat()
                        solution_problem_state_code = extract(response, "python_code")
                    if observation_error_message:
                        self.llm_client.load(observation_error_message)
                        response = self.llm_client.chat()
                        observation_problem_state_code = extract(response, "python_code")
                    instance_error_message, solution_error_message, observation_error_message = self.smoke_test(instance_problem_state_code, solution_problem_state_code, observation_problem_state_code)
                if instance_error_message or solution_error_message or observation_error_message:
                    self.llm_client.dump(f"problem_state_abandoned")
                    return None
        # Save problem state code
        problem_state_code_file = os.path.join(self.output_dir, "problem_state.py")
        node = "# This file is generated generate_evaluation_function.py and to renew the function, run \"python generate_evaluation_function.py\""
        problem_state_code = "\n\n".join([node, instance_problem_state_code, solution_problem_state_code, observation_problem_state_code])
        with open(problem_state_code_file, "w") as fp:
            fp.write(problem_state_code)
        print(f"Save problem state in {problem_state_code_file}")

        # Save problem state description
        instance_problem_state_description = self.get_problem_state_description(instance_problem_state_code)
        solution_problem_state_description = self.get_problem_state_description(solution_problem_state_code)
        problem_state_description["instance_problem_state"] = instance_problem_state_description
        problem_state_description["solution_problem_state"] = solution_problem_state_description
        problem_state_descriptions = [line.lstrip() for line in "\n".join(problem_state_description.values()).split("\n")]
        node = "problem_state (dict): The dictionary contains the problem state with:\n    "
        problem_state_description_str = node + "\n    ".join(problem_state_descriptions)# .replace("- \"", "- ").replace("\" (", " (")
        problem_state_description_file = os.path.join(self.output_dir, "problem_state.txt")
        with open(problem_state_description_file, "w") as fp:
            fp.write(problem_state_description_str)
        return problem_state_code_file

    def get_problem_state_description(self, problem_state_code: str) -> None:
        description = problem_state_code.split("\"\"\"")[1].split("problem state with:\n")[-1].strip()
        return description


    def smoke_test(self, instance_problem_state_code: str, solution_problem_state_code: str, observation_problem_state_code: str) -> str:
        # Load smoke data
        smoke_data_dir = search_file("smoke_data", problem=self.problem)
        previous_operations = []
        if os.path.exists(os.path.join(smoke_data_dir, "previous_operations.txt")):
            previous_operations = open(os.path.join(smoke_data_dir, "previous_operations.txt")).readlines()
        smoke_data = [file for file in os.listdir(smoke_data_dir) if file != "previous_operations.txt"][0]
        smoke_data = os.path.join(smoke_data_dir, smoke_data)

        # Prepare env
        module = importlib.import_module(f"src.problems.{self.problem}.env")
        globals()["Env"] = getattr(module, "Env")
        if os.path.exists(os.path.join("src", "problems", self.problem, "components.py")):
            module = importlib.import_module(f"src.problems.{self.problem}.components")
        else:
            module = importlib.import_module(f"src.problems.base.mdp_components")
        names_to_import = (name for name in dir(module) if not name.startswith('_'))
        for name in names_to_import:
            globals()[name] = getattr(module, name)
        env = Env(data_name=smoke_data)
        env.reset()
        for previous_operation in previous_operations:
            env.run_operator(eval(previous_operation.strip()))
        try:
            # Load instance problem state and run
            get_instance_problem_state = load_function(instance_problem_state_code, function_name="get_instance_problem_state")
            instance_problem_state = get_instance_problem_state(env.instance_data)
            assert instance_problem_state is not None
        except Exception as e:
            error_message = traceback.format_exc()
            return f"We got error when run get_instance_problem_state:\n{error_message}. Please fix up the get_instance_problem_state function in same format.", None, None
        try:
            # Load solution problem state and run
            get_solution_problem_state = load_function(solution_problem_state_code, function_name="get_solution_problem_state")
            solution_problem_state = get_solution_problem_state(env.instance_data, env.current_solution, env.get_key_value)
            assert solution_problem_state is not None
        except Exception as e:
            error_message = traceback.format_exc()
            return None, f"We got error when run get_solution_problem_state:\n{error_message}. Please fix up the get_solution_problem_state function in same format.", None
        try:
            # Load observation problem state and run
            get_observation_problem_state = load_function(observation_problem_state_code, function_name="get_observation_problem_state")
            observation_problem_state = get_observation_problem_state(solution_problem_state)
            assert observation_problem_state is not None
        except Exception as e:
            error_message = traceback.format_exc()
            return None, None, f"We got error when run get_observation_problem_state:\n{error_message}. Please fix up the get_observation_problem_state function in same format."
        return None, None, None