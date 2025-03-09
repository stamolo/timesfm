import numpy as np

def print_label_statistics(labels, num_classes):
    """
    Выводит статистику по меткам:
    - Общее количество примеров
    - Количество примеров для каждого класса
    """
    flat_labels = labels.flatten()
    counts = np.bincount(flat_labels, minlength=num_classes).astype(np.float32)
    total = flat_labels.shape[0]
    print("Общее количество примеров:", total)
    for i, count in enumerate(counts):
        print(f"Класс {i}: {int(count)} примеров")
