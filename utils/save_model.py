import torch
import logging

logger = logging.getLogger(__name__)

def save_checkpoint(model: torch.nn.Module, path: str) -> None:
    """
    Сохраняет веса модели в указанный файл.
    """
    try:
        torch.save(model.state_dict(), path)
        logger.info("Checkpoint сохранён: %s", path)
    except Exception as e:
        logger.error("Ошибка при сохранении чекпоинта: %s", e)

def save_traced_model(model: torch.nn.Module, path: str) -> None:
    """
    Пытается сохранить модель через scripting.
    """
    try:
        scripted_model = torch.jit.script(model)
        torch.jit.save(scripted_model, path)
        logger.info("Scripted модель сохранена: %s", path)
    except Exception as e:
        logger.error("Ошибка при сохранении через scripting: %s", e)
