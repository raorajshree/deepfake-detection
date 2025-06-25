from resnext_lstm_model import DeepFakeModel
from utils import load_data

if __name__ == "__main__":
    train_loader, val_loader = load_data()
    model = DeepFakeModel()
    model.train(train_loader, val_loader)
