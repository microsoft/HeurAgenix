import os
import threading
import queue
import time
import base64
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

generated_heuristics = queue.Queue()

def button_style():
    st.markdown("""
        <style>
        div.stButton > button {
            background-color: #f0f0f0;
            border: none;
            color: black;
            padding: 0px;
            text-align: center;
            text-decoration: none;
            display: block;
            font-size: 16px;
            margin: -8px 0;
            cursor: pointer;
            width: 100%;
            box-sizing: border-box;
            border-radius: 0px;
        }
        div.stButton > button:focus {
            background-color: white;
        }
        </style>
    """, unsafe_allow_html=True)

def get_binary_file_downloader_html(bin_file, file_label='File'):  
    with open(bin_file, 'rb') as f:
        data = f.read()  
        bin_str = base64.b64encode(data).decode()  
        href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">下载 {file_label}</a>'  
    return href

def heuristic_generator(generate_from_llm, paper_list, reference_problems, checkbox_smoke_test, checkbox_deduplication):
    # Mock generations
    for i in range(10):
        time.sleep(0)
        index = int(i / 2)
        if i % 2 == 0:
            generated_heuristics.put([None, None, f"Generating heu {index} ..."])
        else:
            generated_heuristics.put([f"Heu {index}", f"Heu {index} des", f"Heu {index} generated."])
    generated_heuristics.put([None, None, None])

def introduction():
    st.title("HeurAgenix Experience")
    st.write("""
        Welcome to the HeurAgenix project experience page!
        
        HeurAgenix is a multi-agent framework that utilizes large language models (LLMs) to generate, evolve, evaluate, and select heuristic strategies for solving combinatorial optimization problems. 
        This innovative framework is designed to effectively generate diverse heuristics for both classic and novel optimization challenges, showcasing remarkable adaptability and flexibility.
        
        ### Workflow of HeurAgenix:
        - **Heuristic Generation**: Utilizes internal knowledge, reference papers, or related problem heuristics.
        - **Heuristic Evolution**: Employs a data-driven approach to refine and improve heuristics.
        - **Benchmark Evaluation**: Provides feature extractors for problem instances and solutions.
        - **Heuristic Selection**: Dynamically selects the most appropriate heuristic based on real-time evaluation.

        Explore the capabilities of HeurAgenix by following the steps below.
    """)
    st.image(r"doc/framework.png", caption="HeurAgenix Framework Overview", use_column_width=True)

    if st.button("Start Experience"):
        st.session_state.page = "Select Problem"
        st.rerun()

def select_problem():
    st.header("Select a Problem")

    existing_problems = [problem for problem in os.listdir(os.path.join("src", "problems")) if problem != "base"]
    options = existing_problems + ["create new problem"]

    if 'selected_problem' not in st.session_state:
        st.session_state.selected_problem = options[0]

    problem_choice = st.radio("Choose a problem to work with:", options, index=options.index(st.session_state.selected_problem))

    if problem_choice == "create new problem":

        st.image(r"doc/component.png", caption="Component", use_column_width=True)
        st.write("For new problem, we need to provide problem description text to introduce the problem, global data text to introduce the static instance data, state data text to introduce the dynamic state data")
        problem_description = st.text_area("Problem description", "Input the problem description here. Example: \nTraveling Salesman Problem (TSP) is the challenge of finding the shortest possible route that visits a given list of cities exactly once and returns to the origin city, based on the distances between each pair of cities.", height=200)

        global_data = st.text_area("Global data", "global_data (dict): The global data dict containing the global instance data. Example:\n    - \"node_num\" (int): The total number of nodes in the problem.\n    - \"distance_matrix\" (numpy.ndarray): A 2D array representing the distances between nodes.", height=200)

        state_data = st.text_area("State data", "state_data (dict): The state data dict containing the solution state data. Example:\n    - \"current_solution\" (Solution): An instance of the Solution class representing the current solution.\n    - \"visited_nodes\" (list[int]): A list of integers representing the IDs of nodes that have been visited.\n    - \"current_cost\" (int): The total cost of current solution. The cost to return to the starting point is not included until the path is fully constructed.\n    - \"last_visited\" (int): The last visited node.\n    - \"validation_solution\" (callable): def validation_solution(solution: Solution) -> bool: function to check whether new solution is valid.", height=200)

        # st.markdown(get_binary_file_downloader_html('template.py', 'template'), unsafe_allow_html=True)
  
        st.markdown("---")
        template_file_path = os.path.join("src", "problems", "base", "env.template.py")
        with open(template_file_path, 'rb') as file:  
            env_btn = st.download_button(
                label="Down the env.py template",
                data=file,  
                file_name="env.py",
                mime="text/plain"  
            )
        env_uploaded_file = st.file_uploader("Upload the env.py", type=["py"])  
        if env_uploaded_file is not None:  
            env_content = env_uploaded_file.getvalue()  
            st.text_area("env.py", env_content.decode("utf-8"), height=300)

        st.markdown("---")
        template_file_path = os.path.join("src", "problems", "base", "components.template.py")
        with open(template_file_path, 'rb') as file:  
            components_btn = st.download_button(  
                label="Down the components.py template",  
                data=file,  
                file_name="components.py",
                mime="text/plain"  
            )  
        components_uploaded_file = st.file_uploader("Upload the components.py", type=["py"])  
        if components_uploaded_file is not None:  
            components_content = components_uploaded_file.getvalue()  
            st.text_area("env.py", components_content.decode("utf-8"), height=300)

        st.markdown("---")
        st.session_state.checkbox_smoke_test = st.checkbox("Smoke test")
        if st.session_state.checkbox_smoke_test:
            st.session_state.smoke_test_data = st.file_uploader("Upload the smoke data")
        if st.button("Create the problem"):
            if env_uploaded_file is not None and components_uploaded_file is not None:
                st.write("Problem is saved")
            else:
                st.write("Missing file")
    else:
        problem_description = "This is a description of the selected problem."
        heuristic_list = ["Heu1", "Heu2", "Heu3", "Heu4", "Heu5", "Heu6", "Heu7", "Heu8"]
        heuristic_descriptions = {
            "Heu1": "Description for Heu1",
            "Heu2": "Description for Heu2",
            "Heu3": "Description for Heu3",
            "Heu4": "Description for Heu4",
            "Heu5": "Description for Heu5",
            "Heu6": "Description for Heu6",
            "Heu7": "Description for Heu7",
            "Heu8": "Description for Heu8",
        }
        st.session_state.selected_heuristic = heuristic_list[0]

        col1, col2 = st.columns([1, 3])

        with col1:
            st.subheader("Heuristics")
            button_style()
            for heuristic in heuristic_list:
                if st.button(heuristic, key=heuristic):
                    st.session_state.selected_heuristic = heuristic

        with col2:
            st.subheader("Heuristic Description")
            selected_heuristic = st.session_state.selected_heuristic
            st.write(heuristic_descriptions[selected_heuristic])
            st.write("..............")
            st.write("..............")
            st.write("..............")
            st.write("..............")

        st.markdown("<br><br><br>", unsafe_allow_html=True)
        if st.button("Generate Heuristics"):
            st.session_state.page = "Generate Heuristics Optionals"
            st.rerun()
        if st.button("Evolve Heuristics"):
            st.session_state.page = "Evolve Heuristics"
            st.rerun()
        if st.button("Run Heuristics"):
            st.session_state.page = "Run Heuristics"
            st.rerun()

def generate_heuristics_optionals():
    st.title("Generate Heuristics")

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
    st.session_state.checkbox_deduplication = st.checkbox("Deduplication")
    
    if st.button("Upload smoke test data", disabled=not st.session_state.checkbox_smoke_test):  
        st.session_state.smoke_test_data = st.file_uploader("Update the smoke data")

    st.markdown("---")

    # Start generate button
    if st.button("Start Generate"):
        st.session_state.page = "Generate Heuristics"
        st.rerun()

def generate_heuristic():
    st.title("Generate Heuristics")
    generate_from_llm = st.session_state.generate_from_llm
    paper_list = st.session_state.paper_list
    reference_problems = st.session_state.reference_problems
    checkbox_smoke_test = st.session_state.checkbox_smoke_test
    checkbox_deduplication = st.session_state.checkbox_deduplication
    generated_heuristics.queue.clear()

    
    thread = threading.Thread(target=heuristic_generator, args=(generate_from_llm, paper_list, reference_problems, checkbox_smoke_test, checkbox_deduplication))
    thread.start()
    st.session_state.stop = False
    generated_heuristic_name = []
    generated_heuristic_description = {}

    if st.button("Stop", disabled=True):
        st.session_state.generated_heuristic_name = generated_heuristic_name
        st.session_state.generated_heuristic_description = generated_heuristic_description

    while True:
        heuristic_name, heuristic_description, message = generated_heuristics.get()
        if message == None or st.session_state.stop:        
            st.session_state.generated_heuristic_name = generated_heuristic_name
            st.session_state.generated_heuristic_description = generated_heuristic_description
            break
        if heuristic_name is not None:
            generated_heuristic_name.append(heuristic_name)
            generated_heuristic_description[heuristic_name] = heuristic_description
        st.write(message)

    st.session_state.page = "Generate Heuristics Result"
    st.rerun()

def generate_heuristic_result():
    st.title("Generate Heuristics")
    heuristic_list = st.session_state.generated_heuristic_name
    heuristic_descriptions = st.session_state.generated_heuristic_description

    if len(heuristic_list) > 0:
        col1, col2, col3 = st.columns([0.8, 1, 3])
        st.session_state.selected_heuristic = heuristic_list[0]
        if "selected_checkboxes" not in st.session_state:
            st.session_state.selected_checkboxes = {heuristic: False for heuristic in heuristic_list}


        button_style()
        with col1:
            st.subheader("Select")
            for heuristic in heuristic_list:
                st.session_state.selected_checkboxes[heuristic] = st.checkbox("", key=f"checkbox_{heuristic}")

        with col2:
            st.subheader("Heuristic")
            for heuristic in heuristic_list:
                if st.button(heuristic, key=heuristic):
                    st.session_state.selected_heuristic = heuristic
        with col3:
            st.subheader("Heuristic Description")
            selected_heuristic = st.session_state.selected_heuristic
            st.write(heuristic_descriptions[selected_heuristic])
            st.write("..............")
            st.write("..............")
            st.write("..............")
            st.write("..............")

    st.markdown("<br><br><br>", unsafe_allow_html=True)
    
    if st.button("Save"):
        selected_heuristics = ",".join([heuristic for heuristic, checked in st.session_state.selected_checkboxes.items() if checked])
        st.write("Saved heuristics: " + selected_heuristics)

    if st.button("Back"):
        st.session_state.page = "Select Problem"
        st.rerun()

def evolve_heuristic():
    problem_description = "This is a description of the selected problem."
    heuristic_list = ["Heu1", "Heu2", "Heu3", "Heu4", "Heu5", "Heu6", "Heu7", "Heu8"]
    heuristic_descriptions = {
        "Heu1": "Description for Heu1",
        "Heu2": "Description for Heu2",
        "Heu3": "Description for Heu3",
        "Heu4": "Description for Heu4",
        "Heu5": "Description for Heu5",
        "Heu6": "Description for Heu6",
        "Heu7": "Description for Heu7",
        "Heu8": "Description for Heu8",
    }
    st.session_state.selected_heuristic = heuristic_list[0]

    col1, col2 = st.columns([1, 3])

    with col1:
        st.subheader("Heuristics")
        button_style()
        for heuristic in heuristic_list:
            if st.button(heuristic, key=heuristic):
                st.session_state.selected_heuristic = heuristic

    with col2:
        st.subheader("Heuristic Description")
        selected_heuristic = st.session_state.selected_heuristic
        st.write(heuristic_descriptions[selected_heuristic])
        st.write("..............")
        st.write("..............")
        st.write("..............")
        st.write("..............")

    train_data = []
    uploaded_file = st.file_uploader("Choose evolution data set")
    if uploaded_file is not None:
        train_data.append(uploaded_file)

    validation_data = []
    uploaded_file = st.file_uploader("Choose validation data set")
    if uploaded_file is not None:
        validation_data.append(uploaded_file)
    
    evolution_rounds = st.selectbox("Evolution rounds", range(1, 8))

    if st.button("Evolve"):
        st.write("Evolved heuristics")
        st.write("..............")
        st.write("..............")
        st.write("..............")
        st.write("..............")

    if st.button("Save"):
        selected_heuristics = ",".join([heuristic for heuristic, checked in st.session_state.selected_checkboxes.items() if checked])
        st.write("Saved heuristics: " + selected_heuristics)

    if st.button("Back"):
        st.session_state.page = "Select Problem"
        st.rerun()

def run_heuristic():
    problem_description = "This is a description of the selected problem."
    heuristic_list = ["Heu1", "Heu2", "Heu3", "Heu4", "Heu5", "Heu6", "LLM selection", "Random selection"]
    heuristic_descriptions = {
        "Heu1": "Description for Heu1",
        "Heu2": "Description for Heu2",
        "Heu3": "Description for Heu3",
        "Heu4": "Description for Heu4",
        "Heu5": "Description for Heu5",
        "Heu6": "Description for Heu6",
        "LLM selection": "Description for LLM selection",
        "Random selection": "Random selection",
    }
    if len(heuristic_list) > 0:
        col1, col2, col3 = st.columns([0.8, 1, 3])
        st.session_state.selected_heuristic = heuristic_list[0]
        if "selected_checkboxes" not in st.session_state:
            st.session_state.selected_checkboxes = {heuristic: False for heuristic in heuristic_list}


        button_style()
        with col1:
            st.subheader("Select")
            for heuristic in heuristic_list:
                st.session_state.selected_checkboxes[heuristic] = st.checkbox("", key=f"checkbox_{heuristic}")

        with col2:
            st.subheader("Heuristic")
            for heuristic in heuristic_list:
                if st.button(heuristic, key=heuristic):
                    st.session_state.selected_heuristic = heuristic
        with col3:
            st.subheader("Heuristic Description")
            selected_heuristic = st.session_state.selected_heuristic
            st.write(heuristic_descriptions[selected_heuristic])
            st.write("..............")
            st.write("..............")
            st.write("..............")
            st.write("..............")

    test_data = []
    uploaded_file = st.file_uploader("Choose test data set")
    if uploaded_file is not None:
        test_data.append(uploaded_file)
    
    if st.button("Run"):
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

        if st.button("Clean"):
            st.rerun()

    if st.button("Back"):
        st.session_state.page = "Select Problem"
        st.rerun()


# Main function
def main():
    if "page" not in st.session_state:
        st.session_state.page = "Introduction"

    # Page navigation based on session state
    if st.session_state.page == "Introduction":
        introduction()
    elif st.session_state.page == "Select Problem":
        select_problem()
    elif st.session_state.page == "Generate Heuristics Optionals":
        generate_heuristics_optionals()
    elif st.session_state.page == "Generate Heuristics":
        generate_heuristic()
    elif st.session_state.page == "Generate Heuristics Result":
        generate_heuristic_result()
    elif st.session_state.page == "Evolve Heuristics":
        evolve_heuristic()
    elif st.session_state.page == "Run Heuristics":
        run_heuristic()

if __name__ == "__main__":
    main()