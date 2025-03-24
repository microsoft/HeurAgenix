import os
import threading
import queue
import time
import base64
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_theme import st_theme
from streamlit import session_state as state

generated_heuristics = queue.Queue()

def load_text(file_path: str) -> str:
    return open(file_path).read().replace("\n", "<br>")

def sidebar():
    if "scenario" not in state:
        state.scenario = None
    theme = st_theme()
    if theme:
        theme = theme.get("base", "light")
    css = f"""
<style>
    a[href="#_heuristic_generation_agent"], a[href="#_heuristic_evolution_agent"], a[href="#_heuristic_generation_agent"], a[href="#_heuristic_evolution_agent"],{{
        color: {"black" if theme == "light" else "white"};
    }}
</style>
"""
    st.markdown(css, unsafe_allow_html=True)
    toc = """
## Framework
### Heuristic Generation Phase
- [**Heuristic Generation Agent**](#_heuristic_generation_agent)
- [**Heuristic Evolution Agent**](#_heuristic_evolution_agent)
### Problem Solve Phase
- [**Benchmark Evaluation Agent**](#_benchmark_evaluation_agent)
- [**Heuristic Selection Agent**](#_heuristic_selection_agent)
"""
    with st.sidebar:
        st.subheader(":blue[HeurAgenix]", divider="blue")
        st.markdown(toc)
        st.subheader(":orange[Try HeurAgenix]", divider="orange")

        if st.button('Select / Create a Problem', use_container_width=True):
            st.session_state.page = "Select Problem"
            st.rerun()

        if st.button('Generate Heuristics', use_container_width=True, disabled=('selected_problem' not in st.session_state)):
            st.session_state.page = "Generate Heuristics"
            st.rerun()

        if st.button('Evolution Heuristics', use_container_width=True, disabled=('selected_problem' not in st.session_state)):
            st.session_state.page = "Evolve Heuristics"
            st.rerun()

        if st.button('Benchmark Heuristics', use_container_width=True, disabled=('selected_problem' not in st.session_state)):
            st.session_state.page = "Benchmark Heuristics"
            st.rerun()
        
        st.subheader(":green[Navigation]", divider="green")
        if st.button("Back to Introduce", use_container_width=True):
            st.session_state.page = "Introduction"
            st.rerun()

def introduction():
    st.title("HeurAgenix")
    st.write("""
Welcome to the HeurAgenix project demo page!
        
HeurAgenix is a multi-agent framework that utilizes large language models (LLMs) to generate, evolve, evaluate, and select heuristic strategies for solving combinatorial optimization problems.

This innovative framework is designed to effectively generate diverse heuristics for both classic and novel optimization challenges, showcasing remarkable adaptability and flexibility.""")
    st.markdown("---")
    
    st.subheader("Framework")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(r"doc/framework.png", caption="HeurAgenix Framework Overview", use_column_width=True)
    with col2:
        st.write("""
The HeurAgenix framework operates in two main phases: the heuristic generation phase and the problem-solving phase. It consists of four key agents that work collaboratively to address combinatorial optimization problems:

1. **Heuristic Generation Agent**: Responsible for creating initial heuristics using diverse knowledge sources.
2. **Heuristic Evolution Agent**: Focuses on refining and enhancing the generated heuristics through a data-driven process.
3. **Benchmark Evaluation Agent**: Generates detailed feature extractors to provide insights for heuristic selection.
4. **Heuristic Selection Agent**: Dynamically selects the most appropriate heuristic based on current problem characteristics and solution states.
        """)

    with st.expander("Heuristic Generation Agent"):
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(r"doc/heuristic_generation.png", caption="Heuristic Generation Figure", use_column_width=True)
        with col2:
            st.markdown("### <a name='_heuristic_generation_agent'></a>Role", unsafe_allow_html=True)
            st.write("""
The heuristic generation agent is responsible for creating initial heuristic strategies by leveraging a variety of knowledge sources. To mitigate hallucinations in LLMs, it employs smoke tests to ensure the correctness of the generated heuristics. By utilizing these diverse sources, the agent ensures that the heuristics are both novel and applicable to a wide range of optimization problems.

### Input
- Internal knowledge from LLMs.
- Information from reference papers.
- Heuristics from related problems.

### Output
- A set of initial heuristic strategies tailored to the optimization problem at hand.
            """)

    with st.expander("Heuristic Evolution Agent"):
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(r"doc/heuristic_evolution_multiple.png", caption="Heuristic Generation Figure", use_column_width=True)
        with col2:
            st.markdown("### <a name='_heuristic_evolution_agent'></a>Role", unsafe_allow_html=True)
            st.write("""
The heuristic evolution agent refines and enhances the heuristics generated in the initial phase, ensuring they are robust and effective. It identifies bottlenecks in the current solutions and iteratively proposes and verifies improvements. This process uses a data-driven approach, allowing for optimization without the need for prior domain knowledge.

### Input
- Initial heuristics generated by the heuristic generation agent.
- Performance data from test runs.

### Output
- Improved heuristics with enhanced performance metrics.
            """)

    with st.expander("Benchmark Evaluation Agent"):
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(r"doc/problem_solving.png", caption="Heuristic Generation Figure", use_column_width=True)
        with col2:
            st.markdown("### <a name='_benchmark_evaluation_agent'></a>Role", unsafe_allow_html=True)
            st.write("""
This agent creates feature extractors that describe both the problem instances and the current solutions, offering crucial insights for the heuristic selection process. By focusing on capturing distinct characteristics and effective representations, it provides comprehensive evaluations that aid in selecting the most appropriate heuristic strategies.

### Input
- Data from current problem instances.
- Current solution states.

### Output
- Detailed feature sets that characterize the problem and solution space.
            """)

    with st.expander("Heuristic Selection Agent"):
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(r"doc/problem_solving.png", caption="Heuristic Generation Figure", use_column_width=True)
        with col2:
            st.markdown("### <a name='_heuristic_selection_agent'></a>Role", unsafe_allow_html=True)
            st.write("""
The heuristic selection agent dynamically identifies the most suitable heuristic strategy based on the insights provided by the benchmark evaluation agent. It analyzes problem characteristics and evaluates the current solution state, adapting dynamically to specific problem instances and states to ensure robust performance.

### Input
- Features from the benchmark evaluation agent.
- Descriptions of available heuristics.

### Output
- Selected heuristic strategy optimized for the current problem state.
            """)
    st.markdown("---")
    st.subheader("""Try HeurAgenix""")
    st.write("""
Follow these simple steps to get started:

- **Select/Create the Problem**: Begin by selecting an existing combinatorial optimization problem or create a new one with problem description and format of input/output.

- **Generate Heuristic(Optional)**: If heuristics have not been generated previously, use heuristic generation agent to create initial heuristic from LLM, related paper or related problems.

- **Evolve Heuristic(Optional)**: Enhance the generated heuristics by heuristic evolution agent wih training data.

- **Run Heuristic or Heuristics Selectors**: Apply the selected heuristic, or heuristic selection agent to find better solution for given data.

Try to use HeurAgenix and explore its capabilities from selection/creating a problem.""")

    if st.button("Select / Create a Problem"):
        st.session_state.page = "Select Problem"
        st.rerun()

def select_problem():
    existing_problems = [problem for problem in os.listdir(os.path.join("src", "problems")) if problem != "base"]
    options = existing_problems + ["create new problem"]

    st.write(":red[This is a demo to introduce how heuristics works in heuristics generation, evolution and selection.]")
    st.write(":red[We will not running the actual system and the all data are cached before.]")
    col1, col2 = st.columns([1, 3])
    with col1:
        st.header("Select a Problem")
        if 'selected_problem' not in st.session_state:
            st.session_state.selected_problem = options[0]
        problem_choice = st.radio("Choose a problem to work with:", options, index=None)

    if problem_choice == "create new problem":
        with col2:
            st.header(f"Components to build a problem")
            st.image(r"doc/component.png", use_column_width=True)
        st.markdown("---")
        st.write("""
To apply HeurAgenix to a new problem, several essential files are required:
        """)
        problem_choice = st.text_input("Problem name (necessary)")
        problem_description = st.text_area("Problem description.txt (necessary): This text facilitates communication with the LLM and is essential throughout all phases—heuristic generation, evolution, benchmark evaluation, and heuristic selection. It can include natural language explanations, optimization models, Markov decision processes, pseudocode, or specific problem instances.", "Example: \nTraveling Salesman Problem (TSP) is the challenge of finding the shortest possible route that visits a given list of cities exactly once and returns to the origin city, based on the distances between each pair of cities.", height=200)

        global_data = st.text_area("Global data (necessary): This text specifies the global instance data format that provided to heuristics.", "Example:\nglobal_data (dict): The global data dict containing the global instance data with:\n    - \"node_num\" (int): The total number of nodes in the problem.\n    - \"distance_matrix\" (numpy.ndarray): A 2D array representing the distances between nodes.", height=200)

        state_data = st.text_area("State data (necessary): This text specifies the data format for solution state data that provided to heuristics.", "Example:\nstate_data (dict): The state data dict containing the solution state data with:\n    - \"current_solution\" (Solution): An instance of the Solution class representing the current solution.\n    - \"visited_nodes\" (list[int]): A list of integers representing the IDs of nodes that have been visited.\n    - \"current_cost\" (int): The total cost of current solution. The cost to return to the starting point is not included until the path is fully constructed.\n    - \"last_visited\" (int): The last visited node.\n    - \"validation_solution\" (callable): def validation_solution(solution: Solution) -> bool: function to check whether new solution is valid.", height=200)

        st.markdown("---")
        st.write("components.py (necessary): This python file defines the solution class, which records the solution, and operator classes, which are used to modify the solution and apply heuristic algorithms. These operators are crucial during heuristic generation, evolution, and heuristic execution.")
        st.write("To ensure the correct format, please download the template, modify it and then upload it.")
        components_template_file_path = os.path.join("doc", "components.template.py")
        with open(components_template_file_path, 'rb') as file:  
            components_btn = st.download_button(  
                label="Down the components.py template",
                key="components_btn",
                data=file,
                file_name="components.py",
                mime="text/plain"  
            )  
        components_uploaded_file = st.file_uploader("Upload the components.py", type=["py"])  
        if components_uploaded_file is not None:  
            components_content = components_uploaded_file.getvalue()  
            st.text_area("components.py", components_content.decode("utf-8"), height=300)


        st.markdown("---")
        st.write("env.py (optional, only necessary when run the heuristics): This python file defines the environment class to support heuristics as the backend during execution. It includes functions like `load_data`, `init_solution`, `validate_solution`, and data wrapping functions such as `get_global_data` and `get_state_data`, primarily used during heuristic execution, whether for a single heuristic or during heuristic selection.")
        st.write("To ensure the correct format, please download the template, modify it and then upload it.")
        env_template_file_path = os.path.join("doc", "env.template.py")
        with open(env_template_file_path, 'rb') as file:  
            env_btn = st.download_button(  
                label="Down the env.py template",
                key="env_btn",
                data=file,  
                file_name="env.py",
                mime="text/plain"  
            )  
        env_uploaded_file = st.file_uploader("Upload the env.py", type=["py"])  
        if env_uploaded_file is not None:  
            env_content = env_uploaded_file.getvalue()  
            st.text_area("env.py", env_content.decode("utf-8"), height=300)

        st.markdown("---")
        if st.button("Save problem"):
            st.session_state.page = "Select Problem"
            st.rerun()

    elif problem_choice is not None:
        with col2:
            st.header(f"Problem description for {problem_choice}")
            problem_description = load_text(os.path.join("src", "problems", f"{problem_choice}", "prompt", "problem_description.txt"))
            global_data = load_text(os.path.join("src", "problems", f"{problem_choice}", "prompt", "global_data.txt"))
            state_data = load_text(os.path.join("src", "problems", f"{problem_choice}", "prompt", "state_data.txt"))
            st.write(f"""
{problem_description}
#### Global data
{global_data}
#### State data
{state_data}
""", unsafe_allow_html=True)

        st.markdown("---")
        st.header(f"Existing Heuristics for {problem_choice}")
        heuristic_list = os.listdir(os.path.join("src", "problems", problem_choice, "heuristics", "basic_heuristics"))[:4]
        st.session_state.selected_heuristic = heuristic_list[0]

        col1, col2 = st.columns([1, 4])
        with col1:
            for heuristic in heuristic_list:
                if st.button(heuristic[:-8], key=heuristic, use_container_width=True):
                    st.session_state.selected_heuristic = heuristic

        with col2:
            selected_heuristic = st.session_state.selected_heuristic
            heuristic_code = open(os.path.join("src", "problems", problem_choice, "heuristics", "basic_heuristics", selected_heuristic)).read()
            st.code(heuristic_code, language="python")

        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            if st.button(f"Generate More Heuristics for {problem_choice}", use_container_width=True, disabled=('selected_problem' not in st.session_state)):
                st.session_state.page = "Generate Heuristics"
                st.rerun()
        with col2:
            if st.button(f"Evolve Current Heuristics for {problem_choice}", use_container_width=True, disabled=('selected_problem' not in st.session_state)):
                st.session_state.page = "Evolve Heuristics"
                st.rerun()
        with col3:
            if st.button("Benchmark Heuristics", disabled=('selected_problem' not in st.session_state)):
                st.session_state.page = "Benchmark Heuristics"
                st.rerun()

    st.session_state.problem_choice = problem_choice

def generate_heuristic():
    st.title(f"Generate Heuristics for {st.session_state.problem_choice}")

    # Generate from internal knowledge
    st.subheader("Generate from Internal Knowledge")
    st.session_state.generate_from_llm = st.checkbox("Use LLM to generate heuristics")

    st.markdown("---")

    # Learn from paper
    st.subheader("Learn from Paper")
    st.session_state.paper_list = []

    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx"])
    if uploaded_file is not None:
        st.session_state.paper_list.append(uploaded_file.name)

    st.write("Uploaded Papers:")
    for paper in st.session_state.paper_list:
        st.write(f"- {paper}")

    st.markdown("---")

    # Transfer from related problems
    st.subheader("Transfer from Related Problems")
    existing_problems = [problem for problem in os.listdir(os.path.join("src", "problems")) if problem != "base"]
    st.session_state.reference_problems = st.multiselect("Select related problems to transfer heuristics", existing_problems)


    # More optionals
    st.markdown("---")
    st.subheader("More optionals")

    st.session_state.checkbox_smoke_test = st.checkbox("Smoke test")
    if st.button("Upload smoke test data", disabled=not st.session_state.checkbox_smoke_test):  
        st.session_state.smoke_test_data = st.file_uploader("Update the smoke data")
    st.session_state.checkbox_deduplication = st.checkbox("Deduplication")

    st.markdown("---")
    if st.button("Start Generate"):
        st.session_state.generated_heuristics = True

    if "generated_heuristics" in st.session_state and st.session_state.generated_heuristics:
        st.write("Generated heuristics")
        heuristic_list = os.listdir(os.path.join("src", "problems", st.session_state.problem_choice, "heuristics", "basic_heuristics"))[4:8]
        st.session_state.selected_heuristic = heuristic_list[0]
        if "selected_checkboxes" not in st.session_state:
            st.session_state.selected_checkboxes = {}
        col1, col2 = st.columns([1, 3])
        with col1:
            for heuristic in heuristic_list:
                col3, col4 = st.columns([1, 3])
                with col3:
                    st.session_state.selected_checkboxes[heuristic] = st.checkbox("", key=f"checkbox_{heuristic}")
                with col4:
                    if st.button(heuristic[:-8], key=heuristic, use_container_width=True):
                        st.session_state.selected_heuristic = heuristic
        with col2:
            selected_heuristic = st.session_state.selected_heuristic
            heuristic_code = open(os.path.join("src", "problems", st.session_state.problem_choice, "heuristics", "basic_heuristics", selected_heuristic)).read()
            st.code(heuristic_code, language="python")
        col5, col6 = st.columns([1, 1])
        with col5:
            if st.button("Save", use_container_width=True):
                selected_heuristics = ",".join([heuristic for heuristic, checked in st.session_state.selected_checkboxes.items() if checked])
                st.write("Saved heuristics: " + selected_heuristics)
        with col6:
            if st.button("Back", use_container_width=True):
                st.session_state.page = "Select Problem"
                st.rerun()

def evolve_heuristic():
    st.title(f"Evolve Heuristics for {st.session_state.problem_choice}")
    st.write("Select the heuristics to evolve")
    heuristic_list = os.listdir(os.path.join("src", "problems", st.session_state.problem_choice, "heuristics", "basic_heuristics"))[4:8]
    st.session_state.selected_heuristic = heuristic_list[0]
    if "selected_checkboxes" not in st.session_state:
        st.session_state.selected_checkboxes = {}
    col1, col2 = st.columns([1, 3])
    with col1:
        for heuristic in heuristic_list:
            col3, col4 = st.columns([1, 3])
            with col3:
                st.session_state.selected_checkboxes[heuristic] = st.checkbox("", key=f"checkbox_{heuristic}")
            with col4:
                if st.button(heuristic[:-8], key=heuristic, use_container_width=True):
                    st.session_state.selected_heuristic = heuristic
    with col2:
        selected_heuristic = st.session_state.selected_heuristic
        heuristic_code = open(os.path.join("src", "problems", st.session_state.problem_choice, "heuristics", "basic_heuristics", selected_heuristic)).read()
        st.code(heuristic_code, language="python")

    st.markdown("---")
    st.write("Set datasets")
    train_data = []
    uploaded_file = st.file_uploader("Choose evolution data set")
    if uploaded_file is not None:
        train_data.append(uploaded_file)

    validation_data = []
    uploaded_file = st.file_uploader("Choose validation data set")
    if uploaded_file is not None:
        validation_data.append(uploaded_file)
    
    st.markdown("---")
    st.write("Set parameters for evolution")
    perturb_heuristic = st.radio("Choose a problem to work with:", heuristic_list, index=0)
    perturb_ratio = st.selectbox("Perturb ratio", [i / 10 for i in range(1, 11)])
    evolution_rounds = st.selectbox("Evolution rounds", range(1, 8))
    filter_num = st.selectbox("Filter num", range(1, 8))
    st.session_state.checkbox_smoke_test = st.checkbox("Smoke test")
    if st.button("Upload smoke test data", disabled=not st.session_state.checkbox_smoke_test):  
        st.session_state.smoke_test_data = st.file_uploader("Update the smoke data")
    st.session_state.checkbox_deduplication = st.checkbox("Deduplication")

    st.markdown("---")
    if st.button("Start Evolution"):
        st.session_state.evolved_heuristics = True

    if "evolved_heuristics" in st.session_state and st.session_state.evolved_heuristics:
        st.write("Evolved heuristics")
        heuristic_list = os.listdir(os.path.join("src", "problems", st.session_state.problem_choice, "heuristics", "basic_heuristics"))[8:]
        st.session_state.selected_heuristic = heuristic_list[0]
        if "selected_checkboxes" not in st.session_state:
            st.session_state.selected_checkboxes = {}
        col1, col2 = st.columns([1, 3])
        with col1:
            for heuristic in heuristic_list:
                col3, col4 = st.columns([1, 3])
                with col3:
                    st.session_state.selected_checkboxes[heuristic] = st.checkbox("", key=f"checkbox_{heuristic}")
                with col4:
                    if st.button(heuristic[:-8], key=heuristic, use_container_width=True):
                        st.session_state.selected_heuristic = heuristic
        with col2:
            selected_heuristic = st.session_state.selected_heuristic
            heuristic_code = open(os.path.join("src", "problems", st.session_state.problem_choice, "heuristics", "basic_heuristics", selected_heuristic)).read()
            st.code(heuristic_code, language="python")
        col5, col6 = st.columns([1, 1])
        with col5:
            if st.button("Save", use_container_width=True):
                selected_heuristics = ",".join([heuristic for heuristic, checked in st.session_state.selected_checkboxes.items() if checked])
                st.write("Saved heuristics: " + selected_heuristics)
        with col6:
            if st.button("Back", use_container_width=True):
                st.session_state.page = "Select Problem"
                st.rerun()

def run_heuristic():
    st.title(f"Benchmark {st.session_state.problem_choice}")
    heuristic_list = os.listdir(os.path.join("src", "problems", st.session_state.problem_choice, "heuristics", "basic_heuristics"))[:4]
    st.session_state.selected_heuristic = heuristic_list[0]
    if "selected_checkboxes" not in st.session_state:
        st.session_state.selected_checkboxes = {}
    col1, col2 = st.columns([1, 3])
    with col1:
        for heuristic in heuristic_list + ["LLM Rapid Selection", "LLM Comprehensive Selection", "Random Selection"]:
            col3, col4 = st.columns([1, 3])
            with col3:
                st.session_state.selected_checkboxes[heuristic] = st.checkbox("", key=f"checkbox_{heuristic}")
            with col4:
                name = heuristic[:-8] if heuristic in heuristic_list else heuristic
                if st.button(name, key=heuristic, use_container_width=True):
                    st.session_state.selected_heuristic = heuristic
    with col2:
        selected_heuristic = st.session_state.selected_heuristic
        if selected_heuristic in heuristic_list:
            heuristic_code = open(os.path.join("src", "problems", st.session_state.problem_choice, "heuristics", "basic_heuristics", selected_heuristic)).read()
        elif selected_heuristic == "LLM Rapid Selection":
            heuristic_code = "Dynamic select heuristics directly from llm"
        elif selected_heuristic == "LLM Comprehensive Selection":
            heuristic_code = "Dynamic select heuristics with mcts search from llm"
        elif selected_heuristic == "Random Selection":
            heuristic_code = "Dynamic random select heuristics"
        st.code(heuristic_code, language="python")

    st.markdown("---")
    st.write("Set datasets")
    test_data = []
    uploaded_file = st.file_uploader("Choose test data set")
    if uploaded_file is not None:
        test_data.append(uploaded_file)

    st.markdown("---")
    if st.button("Start Run"):
        st.session_state.run_heuristics = True

    if "run_heuristics" in st.session_state and st.session_state.run_heuristics:
        data = [[1,2], [4,5], [7,8]]

        group_data = list(zip(*data))
        n_groups = len(data)
        n_items = len(data[0])

        fig, ax = plt.subplots()
        index = np.arange(n_groups)
        bar_width = 0.35

        for i in range(n_items):
            ax.bar(index + i * bar_width, group_data[i], bar_width, label=f'Item {i+1}')
        
        ax.set_xlabel('Groups')  
        ax.set_ylabel('Values')  
        ax.set_title('Grouped Bar Chart')  
        ax.set_xticks(index + bar_width / 2)  
        ax.set_xticklabels([f'Group {i+1}' for i in range(n_groups)])  
        ax.legend()  
        
        st.pyplot(fig)

    col5, col6 = st.columns([1, 1])
    with col5:
        if st.button("Clean", use_container_width=True):
            st.session_state.run_heuristics = False
            st.rerun()
    with col6:
        if st.button("Back", use_container_width=True):
            st.session_state.page = "Select Problem"
            st.rerun()

# Main function
def main():

    st.set_page_config(layout="wide", page_title="HeurAgenix", page_icon="🔍")

    if "page" not in st.session_state:
        st.session_state.page = "Introduction"

    sidebar()
    # Page navigation based on session state
    if st.session_state.page == "Introduction":
        introduction()
    elif st.session_state.page == "Select Problem":
        select_problem()
    elif st.session_state.page == "Generate Heuristics":
        generate_heuristic()
    elif st.session_state.page == "Evolve Heuristics":
        evolve_heuristic()
    elif st.session_state.page == "Benchmark Heuristics":
        run_heuristic()

if __name__ == "__main__":
    main()