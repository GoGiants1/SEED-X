from huggingface_hub import hf_hub_download


def download_file_from_hf(pretrained_model_path, repo_id):
    if repo_id is None:
        raise ValueError("repo_id is required to download from huggingface")

    # Download from huggingface
    path = hf_hub_download(
        repo_id=repo_id,
        repo_type="model",
        filename=pretrained_model_path,
    )

    return path
