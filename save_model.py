import os
import torch

def save_checkpoint(model, path):
    """
    Сохраняет веса модели в указанный файл.
    """
    torch.save(model.state_dict(), path)
    print(f"Checkpoint сохранён: {path}")

def save_traced_model(model, path):
    """
    Пытается сохранить через scripting.
    """
    try:
        scripted_model = torch.jit.script(model)
        torch.jit.save(scripted_model, path)
        print(f"Scripted модель сохранена: {path}")
    except Exception as e2:
        print("Ошибка при сохранении через scripting:", "e2")
