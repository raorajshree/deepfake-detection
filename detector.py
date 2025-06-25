from model.resnext_lstm_model import DeepFakeModel
import torch

def predict_video(video_path):
    model = DeepFakeModel()
    model.load_state_dict(torch.load("model_weights.pth"))
    model.eval()
    # Dummy prediction
    return "Real"  # or "Fake"
