import torch

path = "./mars_base_sar_encoder_only.pth"

print("Torch:", torch.__version__)

state = torch.load(
    path,
    map_location="cpu",
    weights_only=True
)

print(type(state))

if isinstance(state, dict):
    print(state.keys())