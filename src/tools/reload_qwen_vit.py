import torch
from huggingface_hub import HfApi
from torch import nn
from transformers import AutoModelForCausalLM

torch.manual_seed(1234)

qwen_model_path = "Qwen/Qwen-VL-Chat"
save_path = "qwen_vit_G.pt"

model = AutoModelForCausalLM.from_pretrained(
    qwen_model_path, device_map="cpu", trust_remote_code=True
).eval()

visual_encoder = model.transformer.visual
print(visual_encoder)
torch.save(visual_encoder.state_dict(), save_path)

api = HfApi()
api.upload_file(
    path_or_fileobj=save_path,
    path_in_repo="qwen_vit_G.pt",
    repo_id="GoGiants1/QwenViT",
    repo_type="model",
)

# Save the visual encoder in local disk

# Save the visual encoder in huggingface hub
