import pandas as pd
import numpy as np
import random

# Пытаемся импортировать Faker, но если его нет, программа все равно будет работать
try:
    from faker import Faker
    fake = Faker()
except ImportError:
    fake = None
    print("Библиотека Faker не найдена. Будет использоваться базовая генерация данных.")

# Инициализация генератора случайных данных
np.random.seed(42)
random.seed(42)

def generate_synthetic_data(n_samples=1000):
    """
    Генерирует синтетические данные для обучения модели выявления битых DOM-элементов

    Args:
        n_samples: количество примеров для генерации

    Returns:
        DataFrame с сгенерированными данными
    """
    data = []

    tags = ['div', 'a', 'button', 'img', 'input', 'span', 'p', 'h1', 'h2', 'h3', 'ul', 'li', 'form']

    for _ in range(n_samples):
        tag = random.choice(tags)
        attr_count = np.random.randint(0, 10)
        has_text = np.random.choice([0, 1])
        depth = np.random.randint(1, 10)
        has_empty_attr = np.random.choice([0, 1], p=[0.7, 0.3])
        visibility = np.random.choice([0, 1], p=[0.1, 0.9])
        is_interactive = 1 if tag in ['a', 'button', 'input', 'form'] else np.random.choice([0, 1], p=[0.8, 0.2])
        has_display_none = np.random.choice([0, 1], p=[0.9, 0.1])

        # Дополнительные признаки
        missing_required_attr = np.random.choice([0, 1], p=[0.7, 0.3])
        css_conflict = np.random.choice([0, 1], p=[0.85, 0.15])
        nested_structure_issue = np.random.choice([0, 1], p=[0.85, 0.15])

        # Определяем, является ли элемент битым на основе заданных критериев
        is_broken = 0

        # Пустые или отсутствующие ключевые атрибуты
        if missing_required_attr == 1 and tag in ['img', 'a', 'input']:
            is_broken = 1

        # Скрытые элементы, участвующие в логике интерфейса
        if has_display_none == 1 and is_interactive == 1 and visibility == 0:
            is_broken = 1

        # Конфликт CSS-стилей
        if css_conflict == 1 and visibility == 0:
            is_broken = 1

        # Повреждённые вложенные структуры
        if nested_structure_issue == 1 and depth > 5:
            is_broken = 1

        data.append({
            'tag': tag,
            'attr_count': attr_count,
            'has_text': has_text,
            'depth': depth,
            'has_empty_attr': has_empty_attr,
            'visibility': visibility,
            'is_interactive': is_interactive,
            'has_display_none': has_display_none,
            'missing_required_attr': missing_required_attr,
            'css_conflict': css_conflict,
            'nested_structure_issue': nested_structure_issue,
            'is_broken': is_broken
        })

    return pd.DataFrame(data)

def save_data(df, filename='dom_elements.csv'):
    """Сохраняет данные в CSV файл"""
    import os

    # Определяем абсолютный путь к директории в зависимости от места запуска скрипта
    # Если запуск из ноутбука, путь будет '../data', если из src - 'data'
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.basename(current_dir) == 'src':
        data_dir = os.path.join(os.path.dirname(current_dir), 'data')
    else:
        # Если запуск из другого места, предполагаем, что мы в корне проекта
        data_dir = os.path.join(current_dir, 'data')

    # Создаем директорию, если она не существует
    os.makedirs(data_dir, exist_ok=True)

    # Сохраняем файл
    file_path = os.path.join(data_dir, filename)
    df.to_csv(file_path, index=False)
    print(f"Данные сохранены в файл {file_path}")

if __name__ == "__main__":
    # Генерируем данные
    df = generate_synthetic_data(n_samples=5000)

    # Выводим статистику
    print(f"Сгенерировано {len(df)} DOM-элементов")
    print(f"Из них битых: {df['is_broken'].sum()} ({df['is_broken'].mean()*100:.2f}%)")

    # Сохраняем данные
    save_data(df)
