
import torch
import torchxrayvision as xrv


def load_densenet(device):
    model = xrv.models.DenseNet(
        weights="densenet121-res224-all"
    )

    model = model.to(device)
    model.eval()

    return model


def predict_densenet(model, image, device):
    tensor = torch.from_numpy(image).unsqueeze(0)
    tensor = tensor.to(device=device, dtype=torch.float32)

    with torch.inference_mode():
        output = model(tensor)[0]

    scores = output.detach().cpu().numpy().tolist()

    return {
        name: float(score)
        for name, score in zip(model.pathologies, scores)
    }
