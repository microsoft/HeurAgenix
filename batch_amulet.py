import os

string = """description: search best

environment:
  image: amlt-sing/2.5.0-cuda12.4-cudnn9-devel
  setup:
    - pip install azure-identity==1.16.0
    - pip install matplotlib
    - pip install networkx==2.8.8
    - pip install numpy==1.26.4
    - pip install openai==1.34.0
    - pip install ortools==9.9.3963
    - pip install pandas==2.2.1
    - pip install tsplib95==0.7.1
    - pip install bottleneck==1.3.6
    - pip install dill==0.3.9
    - pip install pandas==2.2.1
    - pip install transformers==4.51.3
    - pip install accelerate
    - pip install scikit-learn
    - pip install gym

code:
  local_dir: F:\\ORLLM\\repo\\max_cut

data:
  local_dir: $CONFIG_DIR
  remote_dir: data

target:
  service: sing
  name: msrresrchvc
  workspace_name: workspace-msra-ml-miic-sg-oc
 
jobs:
"""
for i in range(1):
  for data in os.listdir(os.path.join("data", "max_cut", "test_data")):
      string += f"""
  - name: Search best on {data} try {i}
    sku: 8C30
    identity: managed
    submit_args:
      env:
        _AZUREML_SINGULARITY_JOB_UAI: "/subscriptions/e033d461-1923-44a7-872b-78f1d35a86dd/resourcegroups/rg-msra-oc-ml-miic-sg/providers/Microsoft.ManagedIdentity/userAssignedIdentities/msra-oc-ml-miic-sg-mi"
    command:
      - python search_best.py {data} 1000
    tags:
      ["OR_Reasoner_Model"]
"""
print(string)