from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="VictorYXL/HeurAgenixDataset",
    repo_type="dataset",
    local_dir="data",
    local_dir_use_symlinks=False,
    revision="main"
)