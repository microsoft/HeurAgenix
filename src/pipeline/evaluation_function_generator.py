import os
import importlib
import traceback
from src.util.util import extract, load_heuristic, search_file
from src.util.gpt_helper import GPTHelper


class EvaluationFunctionGenerator:
    def __init__(
        self,
        gpt_helper: GPTHelper,
        problem: str
    ) -> None:
        self.gpt_helper = gpt_helper
        self.problem = problem
        self.output_dir = self.gpt_helper.output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_evaluation_function(self, smoke_test: bool=False) -> str:
        prompt_dict = self.gpt_helper.load_background(self.problem)

        # Get global data feature
        self.gpt_helper.load("global_data_feature", prompt_dict)
        response = self.gpt_helper.chat()
        global_data_features = extract(response, "global_data_feature", "\n")
        global_data_features = ",".join([global_data_feature.split(";")[0] for global_data_feature in global_data_features])
        prompt_dict["global_data_features"] = global_data_features

        # Generate global data feature code
        self.gpt_helper.load("implement_global_data_feature_code", prompt_dict)
        response = self.gpt_helper.chat()
        global_data_feature_code = extract(response, "python_code")
        self.gpt_helper.dump(f"global_data_feature")

        # Get state data feature
        self.gpt_helper.load("state_data_feature", prompt_dict)
        response = self.gpt_helper.chat()
        state_data_features = extract(response, "state_data_feature", "\n")
        state_data_features = ",".join([state_data_feature.split(";")[0] for state_data_feature in state_data_features])
        prompt_dict["state_data_features"] = state_data_features

        # Generate state data feature code
        self.gpt_helper.load("implement_state_data_feature_code", prompt_dict)
        response = self.gpt_helper.chat()
        state_data_feature_code = extract(response, "python_code")
        self.gpt_helper.dump(f"state_data_feature")

        # Verify and revision code
        if smoke_test:
            global_error_message, state_error_message = self.smoke_test(global_data_feature_code, state_data_feature_code)
            while global_error_message or state_error_message:
                if global_error_message:
                    self.gpt_helper.load(global_error_message)
                    response = self.gpt_helper.chat()
                    global_data_feature_code = extract(response, "python_code")
                if state_error_message:
                    self.gpt_helper.load(state_error_message)
                    response = self.gpt_helper.chat()
                    state_data_feature_code = extract(response, "python_code")
                global_error_message, state_error_message = self.smoke_test(global_data_feature_code, state_data_feature_code)

        # Save the code
        evaluation_function_file = os.path.join(self.output_dir, "evaluation_function.py")
        node = "# This file is generated generate_evaluation_function.py and to renew the function, run \"python generate_evaluation_function.py\""
        with open(evaluation_function_file, "w") as fp:
            fp.write("\n\n".join([node, global_data_feature_code, state_data_feature_code]))
        print(f"Save get_state_data_feature and get_state_data_feature code to {evaluation_function_file}")
        return evaluation_function_file

    def smoke_test(self, global_data_feature_code: str, state_data_feature_code: str) -> str:
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
            # Load global data feature extractor and run
            global_data_feature_extractor = load_heuristic(global_data_feature_code, function_name="get_global_data_feature")
            global_data_feature = global_data_feature_extractor(env.global_data)
            assert global_data_feature is not None
        except Exception as e:
            error_message = traceback.format_exc()
            return f"We got error when run get_global_data_feature:\n{error_message}. Please fix up the get_global_data_feature function in same format.", None
        try:
            # Load state data feature extractor and run
            state_data_feature_extractor = load_heuristic(state_data_feature_code, function_name="get_state_data_feature")
            state_data_feature = state_data_feature_extractor(env.global_data, env.state_data)
            assert state_data_feature is not None
        except Exception as e:
            error_message = traceback.format_exc()
            return None, f"We got error when run get_state_data_feature:\n{error_message}. Please fix up the get_state_data_feature function in same format."
        return None, None