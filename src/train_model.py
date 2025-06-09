import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_curve, roc_curve, auc
from sklearn.preprocessing import StandardScaler
import joblib
import os
import time

# Устанавливаем стиль для графиков
sns.set(style='whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

def load_data(file_path='data/dom_elements.csv'):
    """
    Загружает данные из CSV-файла

    Args:
        file_path: путь к файлу с данными

    Returns:
        DataFrame с данными
    """
    df = pd.read_csv(file_path)
    print(f"Загружено {len(df)} записей")
    return df

def explore_data(df):
    """
    Проводит предварительный анализ данных

    Args:
        df: DataFrame с данными
    """
    print("Информация о датасете:")
    print(df.info())

    print("\nСтатистика по числовым признакам:")
    print(df.describe())

    print("\nРаспределение целевой переменной:")
    print(df['is_broken'].value_counts())
    print(f"Доля битых элементов: {df['is_broken'].mean()*100:.2f}%")

    # Визуализация распределения целевой переменной
    plt.figure(figsize=(8, 6))
    sns.countplot(x='is_broken', data=df)
    plt.title('Распределение целевой переменной')
    plt.xlabel('Битый элемент (1 - да, 0 - нет)')
    plt.ylabel('Количество')
    plt.savefig('data/target_distribution.png')

    # Корреляционная матрица
    plt.figure(figsize=(12, 10))
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Корреляционная матрица признаков')
    plt.tight_layout()
    plt.savefig('data/correlation_matrix.png')

    # Распределение по типам тегов
    plt.figure(figsize=(12, 6))
    tag_broken = df.groupby('tag')['is_broken'].mean().sort_values(ascending=False)
    sns.barplot(x=tag_broken.index, y=tag_broken.values)
    plt.title('Доля битых элементов по типам тегов')
    plt.xlabel('Тег')
    plt.ylabel('Доля битых элементов')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('data/tag_broken_ratio.png')

    print("\nАнализ данных завершен. Графики сохранены в директории data/")

def prepare_data(df):
    """
    Подготавливает данные для обучения модели

    Args:
        df: DataFrame с данными

    Returns:
        X_train, X_test, y_train, y_test: разделенные данные для обучения и тестирования
    """
    # Преобразуем категориальные признаки
    df_encoded = pd.get_dummies(df, columns=['tag'], drop_first=True)

    # Разделяем признаки и целевую переменную
    X = df_encoded.drop('is_broken', axis=1)
    y = df_encoded['is_broken']

    # Разделяем данные на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    print(f"Размер обучающей выборки: {X_train.shape}")
    print(f"Размер тестовой выборки: {X_test.shape}")

    return X_train, X_test, y_train, y_test

def train_random_forest(X_train, y_train):
    """
    Обучает модель случайного леса с поиском оптимальных гиперпараметров

    Args:
        X_train: признаки для обучения
        y_train: целевая переменная для обучения

    Returns:
        Обученная модель
    """
    print("Обучение модели Random Forest...")

    # Определяем параметры для поиска
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Создаем базовую модель
    rf = RandomForestClassifier(random_state=42)

    # Поиск оптимальных параметров с помощью перекрестной проверки
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        verbose=1,
        scoring='f1'
    )

    start_time = time.time()
    grid_search.fit(X_train, y_train)
    training_time = time.time() - start_time

    # Выводим лучшие параметры
    print(f"Лучшие параметры: {grid_search.best_params_}")
    print(f"Лучший F1-score: {grid_search.best_score_:.4f}")
    print(f"Время обучения: {training_time:.2f} секунд")

    # Получаем лучшую модель
    best_rf = grid_search.best_estimator_

    # Сохраняем модель
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_rf, 'models/random_forest_model.pkl')
    print("Модель сохранена в models/random_forest_model.pkl")

    return best_rf

def evaluate_model(model, X_test, y_test, X_train, y_train):
    """
    Оценивает качество модели на тестовой выборке

    Args:
        model: обученная модель
        X_test: признаки для тестирования
        y_test: целевая переменная для тестирования
        X_train: признаки для обучения
        y_train: целевая переменная для обучения
    """
    print("Оценка качества модели...")

    # Прогнозы на обучающей выборке
    y_train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)

    # Прогнозы на тестовой выборке
    y_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy на обучающей выборке: {train_accuracy:.4f}")
    print(f"Accuracy на тестовой выборке: {test_accuracy:.4f}")

    print("\nОтчет о классификации:")
    print(classification_report(y_test, y_pred))

    # Матрица ошибок
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Матрица ошибок')
    plt.xlabel('Предсказанный класс')
    plt.ylabel('Истинный класс')
    plt.savefig('data/confusion_matrix.png')

    # ROC-кривая
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig('data/roc_curve.png')

    # Precision-Recall кривая
    precision, recall, _ = precision_recall_curve(y_test, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.savefig('data/precision_recall_curve.png')

    # Важность признаков
    plt.figure(figsize=(12, 10))
    feature_importance = pd.DataFrame(
        {'feature': X_train.columns, 'importance': model.feature_importances_}
    ).sort_values('importance', ascending=False)

    sns.barplot(x='importance', y='feature', data=feature_importance.head(20))
    plt.title('Важность признаков')
    plt.tight_layout()
    plt.savefig('data/feature_importance.png')

    print("\nОценка модели завершена. Графики сохранены в директории data/")

def compare_models():
    """
    Сравнивает различные алгоритмы классификации
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import GradientBoostingClassifier

    # Загружаем данные
    df = load_data()

    # Подготавливаем данные
    df_encoded = pd.get_dummies(df, columns=['tag'], drop_first=True)
    X = df_encoded.drop('is_broken', axis=1)
    y = df_encoded['is_broken']

    # Определяем модели для сравнения
    models = {
        'Логистическая регрессия': LogisticRegression(max_iter=1000, random_state=42),
        'K-ближайших соседей': KNeighborsClassifier(),
        'Дерево решений': DecisionTreeClassifier(random_state=42),
        'Случайный лес': RandomForestClassifier(n_estimators=100, random_state=42),
        'Градиентный бустинг': GradientBoostingClassifier(random_state=42)
    }

    # Сравниваем модели по метрикам
    results = {}
    for name, model in models.items():
        print(f"Оценка модели: {name}...")
        # Используем кросс-валидацию для более надежной оценки
        cv_accuracy = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        cv_f1 = cross_val_score(model, X, y, cv=5, scoring='f1')
        cv_precision = cross_val_score(model, X, y, cv=5, scoring='precision')
        cv_recall = cross_val_score(model, X, y, cv=5, scoring='recall')

        results[name] = {
            'accuracy': cv_accuracy.mean(),
            'f1': cv_f1.mean(),
            'precision': cv_precision.mean(),
            'recall': cv_recall.mean()
        }

    # Преобразуем результаты в DataFrame для удобного отображения
    results_df = pd.DataFrame(results).T

    # Визуализируем результаты
    plt.figure(figsize=(15, 10))

    # Accuracy
    plt.subplot(2, 2, 1)
    sns.barplot(x=results_df.index, y=results_df['accuracy'])
    plt.title('Accuracy')
    plt.xticks(rotation=45)
    plt.ylim(0.8, 1.0)

    # F1-score
    plt.subplot(2, 2, 2)
    sns.barplot(x=results_df.index, y=results_df['f1'])
    plt.title('F1-score')
    plt.xticks(rotation=45)
    plt.ylim(0.7, 1.0)

    # Precision
    plt.subplot(2, 2, 3)
    sns.barplot(x=results_df.index, y=results_df['precision'])
    plt.title('Precision')
    plt.xticks(rotation=45)
    plt.ylim(0.7, 1.0)

    # Recall
    plt.subplot(2, 2, 4)
    sns.barplot(x=results_df.index, y=results_df['recall'])
    plt.title('Recall')
    plt.xticks(rotation=45)
    plt.ylim(0.7, 1.0)

    plt.tight_layout()
    plt.savefig('data/model_comparison.png')

    print("\nРезультаты сравнения моделей:")
    print(results_df)
    print("\nСравнение моделей завершено. График сохранен в data/model_comparison.png")

    return results_df

def main():
    """
    Основная функция для запуска процесса обучения и оценки модели
    """
    print("Начало анализа битых DOM-элементов")

    # Проверяем наличие данных или генерируем их
    if not os.path.exists('data/dom_elements.csv'):
        print("Данные не найдены. Запуск генерации синтетических данных...")
        from data_generator import generate_synthetic_data, save_data
        df = generate_synthetic_data(n_samples=5000)
        save_data(df)

    # Загружаем данные
    df = load_data()

    # Проводим разведочный анализ данных
    explore_data(df)

    # Сравниваем различные модели
    compare_models()

    # Подготавливаем данные для обучения
    X_train, X_test, y_train, y_test = prepare_data(df)

    # Обучаем модель Random Forest
    best_model = train_random_forest(X_train, y_train)

    # Оцениваем качество модели
    evaluate_model(best_model, X_test, y_test, X_train, y_train)

    print("Анализ завершен успешно!")

if __name__ == "__main__":
    main()
