import os
import json
import importlib
import traceback
from copy import deepcopy
from src.util.util import extract, extract_function_with_short_docstring, filter_dict_to_str, find_key_value, load_heuristic, parse_paper_to_dict, replace_strings_in_dict, sanitize_function_name, load_framework_description
from src.util.gpt_helper import GPTHelper


class HeuristicGenerator:
    def __init__(
        self,
        gpt_helper: GPTHelper,
        problem: str
    ) -> None:
        self.gpt_helper = gpt_helper
        self.problem = problem
        self.output_dir = self.gpt_helper.output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_from_gpt(self, smoke_test: bool=False) -> list[str]:
        heuristic_files = []

        # Load background
        prompt_dict = self.gpt_helper.load_background(self.problem)

        # Generate available heuristic description
        self.gpt_helper.load("generate_from_gpt", prompt_dict)
        response = self.gpt_helper.chat()
        self.gpt_helper.dump("heuristic_from_gpt")
        heuristics = extract(response, "heuristic", sep="\n")

        for heuristic in heuristics:
            # Generate description for single heuristic
            self.gpt_helper.load_chat(os.path.join(self.gpt_helper.output_dir, "heuristic_from_gpt"))
            heuristic_name, description = heuristic.split(":")
            heuristic_files.append(self.generate(heuristic_name, description, smoke_test))

        return heuristic_files

    def generate_from_paper(self, paper_path: str, smoke_test: bool=False) -> str:
        heuristic_file = None
        # Load background
        prompt_dict = self.gpt_helper.load_background(self.problem)

        # Load whole paper
        if os.path.isdir(paper_path):
            paper_content = ""
            for file in os.listdir(paper_path):
                if file.split(".")[-1] == "tex":
                    paper_content += open(os.path.join(paper_path, file)).read()
        elif os.path.isfile(paper_path):
            paper_content = open(paper_path).read()

        # Decompose the paper into dict by sections
        section_dict = parse_paper_to_dict(paper_content)
        if "Title" in section_dict:
            title = section_dict["Title"]
        else:
            title = os.path.splitext(os.path.basename(paper_path))[0]
        if "Abstract" in section_dict:
            abstract = section_dict["Abstract"]
        elif "Introduction" in section_dict:
            abstract = section_dict["Introduction"]
        else:
            abstract = "No abstract found."

        # Read abstract to check whether we can generate heuristic from paper
        prompt_dict["title"] = title
        prompt_dict["abstract"] = abstract
        self.gpt_helper.load("reading_paper_abstract", prompt_dict)
        response = self.gpt_helper.chat()
        related_to_problem = extract(response, "related_to_problem")
        self.gpt_helper.dump("read_paper")

        if "yes" in related_to_problem:
            last_interested_section = "None"
            last_interested_content = "None"
            remaining_section_dict = deepcopy(section_dict)
            remaining_section_dict = replace_strings_in_dict(remaining_section_dict)
            dict_str = json.dumps(remaining_section_dict, indent=4)
            # Read section until implement heuristic or give up.
            while True:
                prompt_dict["last_interested_section"] = last_interested_section
                prompt_dict["last_interested_content"] = last_interested_content
                prompt_dict["remaining_section_dict"] = dict_str
                self.gpt_helper.load("reading_paper_section", prompt_dict)
                response = self.gpt_helper.chat()
                interested_section = extract(response, "interested_section")
                if interested_section is None:
                    self.gpt_helper.dump("abandoned")
                    return None
                interested_content = find_key_value(section_dict, interested_section)
                if interested_content is None:
                    self.gpt_helper.dump(f"generate_from_paper")
                    heuristic_name = interested_section
                    heuristic_file = self.generate(heuristic_name, f"Generate from paper {title}. Please add the notes in code to show the source paper.", smoke_test)
                    return heuristic_file
                last_interested_section = interested_section
                last_interested_content = interested_content
        else:
            self.gpt_helper.dump(f"abandoned")
            return None


    def generate_from_reference(self, related_problems: list[str], smoke_test: bool=False) -> list[str]:
        heuristic_files = []

        # Load background
        prompt_dict = self.gpt_helper.load_background(self.problem)

        # Find similar problem
        description_dict = {
            problem: open(os.path.join("src", "problems", problem, "prompt", "problem_description.txt")).read()
            for problem in related_problems
        }
        studied_problems = "\n\n".join([
            f"problem name: {problem}\ndescription: {description_dict[problem]}"
            for problem in related_problems
        ])
        prompt_dict["studied_problems"] = studied_problems
        self.gpt_helper.load("reference_problem", prompt_dict)
        response = self.gpt_helper.chat()
        related_problems = extract(response, "referenced_problem", ";")

        for referenced_problem in related_problems:
            if referenced_problem not in description_dict:
                continue

            # Find the similarities between referenced problem and new problem
            component_code = open(os.path.join("src", "problems", referenced_problem, "components.py")).read()
            reference_solution_class, reference_operation_class = load_framework_description(component_code)
            description = description_dict[referenced_problem]
            prompt_dict["referenced_problem"] = referenced_problem
            prompt_dict["referenced_problem_description"] = description
            prompt_dict["referenced_problem_solution_class"] = reference_solution_class
            prompt_dict["referenced_problem_operation_class"] = reference_operation_class
            self.gpt_helper.load("mapping_component_in_problem", prompt_dict)
            response = self.gpt_helper.chat()
            similarities = extract(response, "similarities", "\n")
            prompt_dict["similarities_in_problem"] = "\n".join(similarities)

            # Check the referenced heuristic
            referenced_heuristic_dir = os.path.join("src", "problems", referenced_problem, "heuristics", "basic_heuristics")
            referenced_heuristic_names = [heuristic_file.split(".")[0] for heuristic_file in os.listdir(referenced_heuristic_dir)]
            referenced_heuristic_docs = []
            for heuristic_name in referenced_heuristic_names:
                referenced_heuristic_doc = extract_function_with_short_docstring(open(os.path.join(referenced_heuristic_dir, heuristic_name + ".py")).read(), heuristic_name).split(":")[-1]
                referenced_heuristic_docs.append(f"{heuristic_name}:{referenced_heuristic_doc}")
            referenced_heuristic_docs = "\n".join(referenced_heuristic_docs)
            prompt_dict["candidate_heuristic_pool"] = referenced_heuristic_docs
            self.gpt_helper.load("reference_heuristic", prompt_dict)
            response = self.gpt_helper.chat()
            self.gpt_helper.dump(f"reference_heuristics_in_{referenced_problem}")
            reference_heuristics = extract(response, "referenced_heuristics", "\n")

            # Find the similarities between referenced heuristic and new problem
            for reference_heuristic_item in reference_heuristics:
                reference_heuristic = reference_heuristic_item.split(";")[0]
                reference_heuristic_file = os.path.join("src", "problems", referenced_problem, "heuristics", "basic_heuristics", reference_heuristic + ".py")
                reference_heuristic_code = open(reference_heuristic_file).read()
                prompt_dict["referenced_heuristic"] = reference_heuristic
                prompt_dict["referenced_heuristic_code"] = reference_heuristic_code
                prompt_dict["referenced_global_data_introduction"] = open(os.path.join("src", "problems", referenced_problem, "prompt", "global_data.txt")).read()
                prompt_dict["referenced_state_data_introduction"] = open(os.path.join("src", "problems", referenced_problem, "prompt", "state_data.txt")).read()
                self.gpt_helper.load("mapping_component_in_heuristic", prompt_dict)
                response = self.gpt_helper.chat()
                similarities_in_heuristics = extract(response, "similarities", "\n")
                if similarities_in_heuristics:
                    similarities += similarities_in_heuristics

                # Update the description
                self.gpt_helper.load("transfer_heuristic", prompt_dict)
                response = self.gpt_helper.chat()
                heuristic_name, description = extract(response, "heuristic", ";")
                description += f"We hope to transfer {reference_heuristic} in {referenced_problem} into new {heuristic_name} in {self.problem}.\n" \
                    + "Following are some similarities(source_component;target_component;introduction):\n" \
                    + "\n".join(similarities)

                # Generate description for single heuristic
                heuristic_files.append(self.generate(heuristic_name, description, smoke_test))
        return heuristic_files

    def generate(self, heuristic_name: str, description: str, smoke_test: bool=False) -> str:
        # Special remind
        special_remind_file = os.path.join("src", "problems", self.problem, "prompt", "special_remind.txt")
        special_remind = "None"
        if os.path.exists(special_remind_file):
            special_remind = open(special_remind_file).read()

        # Review heuristic and describe the intermediate mathematical processes
        self.gpt_helper.load("detailed_heuristic_design", {"heuristic_name": heuristic_name, "description": description})
        response = self.gpt_helper.chat()
        review_result = extract(response, "mathematical_analysis", "\n")
        assert review_result[0].split(":")[0] == "mathematical_description"

        # Check feasible and add mathematical description
        mathematical_description = review_result[0].split(":")[1].strip()
        if mathematical_description == "None":
            self.gpt_helper.dump(f"{function_name}_abandoned")
            return
        description += f"\nmathematical_description: {mathematical_description}"

        # Generate function name
        function_name = sanitize_function_name(heuristic_name, description)
        prompt_dict = {"problem": self.problem, "heuristic_name": heuristic_name, "description": description, "function_name": function_name, "special_remind": special_remind}

        # Implement code
        if os.path.exists(os.path.join("src", "problems", self.problem, "components")):
            prompt_dict["components_file"] = f"src.problems.{self.problem}.components"
        else:
            prompt_dict["components_file"] = f"src.problems.base.mdp_components"
        self.gpt_helper.load("implement_code", prompt_dict)
        response = self.gpt_helper.chat()
        code = extract(response, "python_code")

        # Verify and revision code
        if smoke_test and not self.smoke_test(code, function_name):
            self.gpt_helper.dump(f"{function_name}_abandoned")
            return None

        self.gpt_helper.dump(f"{function_name}")

        # Save code
        output_heuristic_file = os.path.join(self.output_dir, function_name + ".py")
        print(f"Save {function_name} code to {output_heuristic_file}")
        with open(output_heuristic_file, "w") as fp:
            fp.write(code)
        return output_heuristic_file

    def smoke_test(self, heuristic_code: str, function_name: str, max_try_times: int=5) -> str:
        prompt_dict = {}
        # Load smoke data
        smoke_data_dir = os.path.join("src", "problems", self.problem, "data", "smoke_data")
        previous_operations = open(os.path.join(smoke_data_dir, "previous_operations.txt")).readlines()
        smoke_data = [file for file in os.listdir(smoke_data_dir) if file[:10] == "smoke_data"][0]
        prompt_dict["function_name"] = function_name
        prompt_dict["previous_operations"] = "".join(previous_operations)

        # Prepare env
        module = importlib.import_module(f"src.problems.{self.problem}.env")
        globals()["Env"] = getattr(module, "Env")
        if os.path.exists(os.path.join("src", "problems", self.problem, "components")):
            module = importlib.import_module(f"src.problems.{self.problem}.components")
        else:
            module = importlib.import_module(f"src.problems.base.mdp_components")
        names_to_import = (name for name in dir(module) if not name.startswith('_'))
        for name in names_to_import:
            globals()[name] = getattr(module, name)
        env = Env(data_name=smoke_data)
        for _ in range(max_try_times):
            env.reset()
            prompt_dict["smoke_global_data"] = filter_dict_to_str(env.global_data)
            for previous_operation in previous_operations:
                env.run_operator(eval(previous_operation.strip()))
            prompt_dict["smoke_solution"] = env.current_solution
            prompt_dict["smoke_state_data"] = filter_dict_to_str(env.state_data)
            try:
                # Load heuristic and run once
                heuristic = load_heuristic(heuristic_code, function_name)
                operator = env.run_heuristic(heuristic)
            except Exception as e:
                operator = traceback.format_exc()
            if operator is not False and not isinstance(operator, str):
                # Expected result
                self.gpt_helper.load("smoke_test_expected_result.txt", prompt_dict)
                response = self.gpt_helper.chat()
                expected_result = extract(response, "expected_result")

                # Actual result
                prompt_dict["output_result"] = str(operator)
                prompt_dict["updated_smoke_solution"] = env.current_solution
                prompt_dict["updated_smoke_state_data"] = filter_dict_to_str(env.state_data)

                # Compare
                prompt_dict["expected_result"] = expected_result
                self.gpt_helper.load("smoke_test_compare.txt", prompt_dict)
                response = self.gpt_helper.chat()
                heuristic_code = extract(response, "python_code")
                # Actual result
                if heuristic_code is None:
                    # Give up
                    return None
                elif "correct" in heuristic_code:
                    # Correct
                    return heuristic_code
                else:
                    # Update code
                    continue
            else:
                # Crashed during running the heuristic
                prompt_dict["error_message"] = operator
                self.gpt_helper.load("smoke_test_crashed.txt", prompt_dict)
                response = self.gpt_helper.chat()
                heuristic_code = extract(response, "python_code")
                if heuristic_code is None:
                    # Give up
                    return None
        # Give up due to the try limitation
        return None