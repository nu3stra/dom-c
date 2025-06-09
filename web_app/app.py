import os
import sys
import pandas as pd
import numpy as np
import joblib
import requests
from bs4 import BeautifulSoup
from flask import Flask, render_template, request, jsonify
from tqdm import tqdm

# Добавляем директорию проекта в путь для импорта
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Импортируем необходимые функции из нашего проекта
from src.data_generator import generate_synthetic_data, save_data

app = Flask(__name__)

# Загружаем модель
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'random_forest_model.pkl'))
try:
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        print(f"Модель успешно загружена из {model_path}")
    else:
        model = None
        print("Модель не найдена. Анализ будет недоступен до обучения модели.")
except Exception as e:
    model = None
    print(f"Ошибка при загрузке модели: {e}")

def extract_dom_elements(url, max_elements=500):
    """
    Извлекает DOM-элементы с веб-страницы
    """
    try:
        # Получаем HTML-контент веб-страницы
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        # Парсим HTML с помощью BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')

        # Список для хранения данных об элементах
        elements_data = []

        # Целевые теги для анализа
        target_tags = ['div', 'a', 'button', 'img', 'input', 'span', 'p', 'h1', 'h2', 'h3', 'ul', 'li', 'form', 'meta']

        # Собираем все элементы указанных тегов
        all_elements = []
        for tag in target_tags:
            all_elements.extend(soup.find_all(tag))

        # Ограничиваем количество элементов
        all_elements = all_elements[:max_elements]

        # Обрабатываем каждый элемент
        for element in all_elements:
            tag = element.name
            attr_count = len(element.attrs)
            has_text = 1 if element.get_text(strip=True) else 0

            # Получаем строковое представление элемента с атрибутами
            element_html = str(element)
            # Создаем короткое представление элемента (первые 100 символов)
            element_short = element_html[:100] + "..." if len(element_html) > 100 else element_html

            # Создаем представление тега с атрибутами (без содержимого)
            element_tag_with_attrs = f"<{tag}"
            for attr_name, attr_value in element.attrs.items():
                if isinstance(attr_value, list):
                    attr_value = " ".join(attr_value)
                element_tag_with_attrs += f' {attr_name}="{attr_value}"'
            element_tag_with_attrs += ">"

            # Определяем глубину вложенности
            depth = 0
            parent = element.parent
            while parent and parent.name != 'html':
                depth += 1
                parent = parent.parent

            # Проверяем наличие пустых атрибутов
            has_empty_attr = 0
            for attr, value in element.attrs.items():
                if value == "" or value is None:
                    has_empty_attr = 1
                    break

            # Определяем видимость (приблизительно, по наличию style="display:none")
            visibility = 0 if 'style' in element.attrs and 'display:none' in element.attrs['style'] else 1

            # Интерактивные элементы
            is_interactive = 1 if tag in ['a', 'button', 'input', 'form'] else 0

            # Проверяем display:none
            has_display_none = 1 if 'style' in element.attrs and 'display:none' in element.attrs['style'] else 0

            # Проверяем отсутствие обязательных атрибутов
            missing_required_attr = 0
            missing_attr_name = None
            if tag == 'img' and ('src' not in element.attrs or not element.attrs['src']):
                missing_required_attr = 1
                missing_attr_name = 'src'
            elif tag == 'a' and ('href' not in element.attrs or not element.attrs['href']):
                missing_required_attr = 1
                missing_attr_name = 'href'
            elif tag == 'input' and ('type' not in element.attrs or not element.attrs['type']):
                missing_required_attr = 1
                missing_attr_name = 'type'

            # Примерное определение конфликта CSS
            css_conflict = 0
            if visibility == 0 and is_interactive == 1:
                css_conflict = 1

            # Определение проблем вложенных структур (упрощенно)
            nested_structure_issue = 0
            if tag == 'a' and element.find('a'):
                nested_structure_issue = 1  # Вложенные ссылки

            # Добавляем данные об элементе
            elements_data.append({
                'tag': tag,
                'attr_count': attr_count,
                'has_text': has_text,
                'depth': depth,
                'has_empty_attr': has_empty_attr,
                'visibility': visibility,
                'is_interactive': is_interactive,
                'has_display_none': has_display_none,
                'missing_required_attr': missing_required_attr,
                'missing_attr_name': missing_attr_name,
                'css_conflict': css_conflict,
                'nested_structure_issue': nested_structure_issue,
                'element_html': element_short,
                'element_tag_with_attrs': element_tag_with_attrs
            })

        # Создаем DataFrame
        elements_df = pd.DataFrame(elements_data)
        return elements_df

    except Exception as e:
        print(f"Ошибка при получении данных с сайта: {e}")
        return pd.DataFrame()

def analyze_website_with_seo(url, model, max_elements=500):
    """
    Анализирует веб-сайт, выявляя битые DOM-элементы и элементы с SEO-проблемами
    """
    # Получаем DOM-элементы с сайта
    elements_df = extract_dom_elements(url, max_elements)

    if elements_df.empty:
        return {"error": "Не удалось получить данные с сайта"}

    # Предсказания модели
    if model is not None:
        # Подготавливаем данные для предсказания
        elements_encoded = pd.get_dummies(elements_df, columns=['tag'])

        # Получаем список столбцов признаков из модели
        if hasattr(model, 'feature_names_in_'):
            model_features = model.feature_names_in_

            # Добавляем отсутствующие столбцы
            for feature in model_features:
                if feature not in elements_encoded.columns:
                    elements_encoded[feature] = 0

            # Убеждаемся, что порядок столбцов соответствует модели
            try:
                elements_encoded = elements_encoded[model_features]

                # Делаем предсказания
                predictions = model.predict(elements_encoded)
                probabilities = model.predict_proba(elements_encoded)[:, 1]

                # Добавляем результаты к исходному DataFrame
                elements_df['predicted_broken'] = predictions
                elements_df['probability_broken'] = probabilities
            except Exception as e:
                elements_df['predicted_broken'] = 0
                elements_df['probability_broken'] = 0
                print(f"Ошибка при предсказании: {e}")
        else:
            elements_df['predicted_broken'] = 0
            elements_df['probability_broken'] = 0
    else:
        # Если модель не загружена, используем эвристики
        elements_df['predicted_broken'] = (
            elements_df['missing_required_attr'] |
            elements_df['has_empty_attr'] |
            elements_df['css_conflict'] |
            elements_df['nested_structure_issue']
        ).astype(int)
        elements_df['probability_broken'] = elements_df['predicted_broken'].astype(float)

    # Добавляем анализ SEO-проблем
    # Инициализируем признаки SEO-проблем
    elements_df['missing_alt_tag'] = 0
    elements_df['empty_heading'] = 0
    elements_df['missing_meta_description'] = 0
    elements_df['incorrect_heading_hierarchy'] = 0
    elements_df['duplicate_h1'] = 0
    elements_df['missing_link_title'] = 0
    elements_df['long_url'] = 0
    elements_df['invalid_internal_link'] = 0

    # Анализируем SEO-проблемы по типам тегов

    # Проблема 1: Отсутствие alt у изображений
    img_mask = elements_df['tag'] == 'img'
    for idx, row in elements_df[img_mask].iterrows():
        html = row['element_html']
        if 'alt=' not in html or 'alt=""' in html:
            elements_df.loc[idx, 'missing_alt_tag'] = 1

    # Проблема 2: Пустые заголовки
    heading_mask = elements_df['tag'].isin(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
    elements_df.loc[heading_mask & (elements_df['has_text'] == 0), 'empty_heading'] = 1

    # Проблема 3: Отсутствие мета-тегов description
    meta_mask = elements_df['tag'] == 'meta'
    meta_description_found = False
    for idx, row in elements_df[meta_mask].iterrows():
        html = row['element_html'].lower()
        if 'name="description"' in html:
            meta_description_found = True
            break

    if not meta_description_found and any(meta_mask):
        elements_df.loc[meta_mask, 'missing_meta_description'] = 1

    # Проблема 4: Неправильная иерархия заголовков
    headings = elements_df[heading_mask].sort_values('depth')
    if not headings.empty:
        heading_levels = {'h1': 1, 'h2': 2, 'h3': 3, 'h4': 4, 'h5': 5, 'h6': 6}
        prev_level = 0
        for idx, row in headings.iterrows():
            current_level = heading_levels[row['tag']]
            if current_level > prev_level + 1 and prev_level > 0:
                elements_df.loc[idx, 'incorrect_heading_hierarchy'] = 1
            prev_level = current_level

    # Проблема 5: Дублирование h1
    h1_count = elements_df[elements_df['tag'] == 'h1'].shape[0]
    if h1_count > 1:
        elements_df.loc[elements_df['tag'] == 'h1', 'duplicate_h1'] = 1

    # Проблема 6: Отсутствие title у ссылок
    link_mask = elements_df['tag'] == 'a'
    for idx, row in elements_df[link_mask].iterrows():
        html = row['element_html']
        if 'title=' not in html:
            elements_df.loc[idx, 'missing_link_title'] = 1

    # Проблема 7: Слишком длинные URL
    for idx, row in elements_df[link_mask].iterrows():
        html = row['element_html']
        href_start = html.find('href="')
        if href_start >= 0:
            href_start += 6  # длина 'href="'
            href_end = html.find('"', href_start)
            if href_end >= 0:
                href = html[href_start:href_end]
                if len(href) > 100:
                    elements_df.loc[idx, 'long_url'] = 1

    # Проблема 8: Невалидные внутренние ссылки
    for idx, row in elements_df[link_mask].iterrows():
        html = row['element_html']
        href_start = html.find('href="')
        if href_start >= 0:
            href_start += 6
            href_end = html.find('"', href_start)
            if href_end >= 0:
                href = html[href_start:href_end]
                if href.startswith('#') and len(href) == 1:
                    elements_df.loc[idx, 'invalid_internal_link'] = 1
                elif href == '' or href == 'javascript:void(0)':
                    elements_df.loc[idx, 'invalid_internal_link'] = 1

    # Создаем общий признак для SEO-проблем
    seo_problems = [
        'missing_alt_tag', 'empty_heading', 'missing_meta_description',
        'incorrect_heading_hierarchy', 'duplicate_h1', 'missing_link_title',
        'long_url', 'invalid_internal_link'
    ]

    elements_df['has_seo_issues'] = elements_df[seo_problems].sum(axis=1) > 0

    # Формируем результаты
    total_elements = len(elements_df)
    broken_elements = elements_df['predicted_broken'].sum()
    seo_issues = elements_df['has_seo_issues'].sum()

    # Процент элементов с проблемами
    broken_percent = (broken_elements / total_elements * 100) if total_elements > 0 else 0
    seo_issues_percent = (seo_issues / total_elements * 100) if total_elements > 0 else 0

    # Подсчет проблем по типам
    broken_types = elements_df[elements_df['predicted_broken'] == 1][
        ['missing_required_attr', 'has_empty_attr', 'css_conflict',
         'has_display_none', 'nested_structure_issue']
    ].sum().to_dict()

    seo_issues_types = elements_df[seo_problems].sum().to_dict()

    # Группировка по тегам
    broken_by_tag = elements_df.groupby('tag')['predicted_broken'].mean().to_dict()
    seo_by_tag = elements_df.groupby('tag')['has_seo_issues'].mean().to_dict()

    # Определяем общий балл (100 - процент проблемных элементов)
    broken_score = max(0, 100 - broken_percent)
    seo_score = max(0, 100 - seo_issues_percent)
    overall_score = (broken_score + seo_score) / 2

    # Список проблемных элементов
    broken_elements_list = []
    for _, element in elements_df[elements_df['predicted_broken'] == 1].sort_values('probability_broken', ascending=False).iterrows():
        reasons = []
        if element['missing_required_attr'] == 1:
            attr_name = element['missing_attr_name']
            if attr_name:
                reasons.append(f"Отсутствует обязательный атрибут '{attr_name}'")
            else:
                reasons.append("Отсутствует обязательный атрибут")
        if element['has_empty_attr'] == 1:
            reasons.append("Пустое значение атрибута")
        if element['css_conflict'] == 1:
            reasons.append("Конфликт CSS-стилей")
        if element['has_display_none'] == 1:
            reasons.append("Элемент скрыт через display:none")
        if element['nested_structure_issue'] == 1:
            reasons.append("Проблемы с вложенной структурой")

        broken_elements_list.append({
            'tag': element['tag'],
            'html': element['element_html'],
            'probability': float(element['probability_broken']),
            'reasons': reasons
        })

    # Список элементов с SEO-проблемами
    seo_issues_list = []
    for _, element in elements_df[elements_df['has_seo_issues'] == True].iterrows():
        issues = []
        for problem, desc in [
            ('missing_alt_tag', 'Отсутствие атрибута alt'),
            ('empty_heading', 'Пустой заголовок'),
            ('missing_meta_description', 'Отсутствие meta description'),
            ('incorrect_heading_hierarchy', 'Неправильная иерархия заголовков'),
            ('duplicate_h1', 'Дублирование заголовков h1'),
            ('missing_link_title', 'Отсутствие атрибута title у ссылки'),
            ('long_url', 'Слишком длинный URL'),
            ('invalid_internal_link', 'Невалидная внутренняя ссылка')
        ]:
            if element[problem] == 1:
                issues.append(desc)

        seo_issues_list.append({
            'tag': element['tag'],
            'html': element['element_html'],
            'issues': issues
        })

    # Формируем итоговый результат
    result = {
        'url': url,
        'total_elements': int(total_elements),
        'broken_elements': int(broken_elements),
        'broken_percent': float(broken_percent),
        'seo_issues': int(seo_issues),
        'seo_issues_percent': float(seo_issues_percent),
        'broken_types': broken_types,
        'seo_issues_types': seo_issues_types,
        'broken_by_tag': broken_by_tag,
        'seo_by_tag': seo_by_tag,
        'broken_score': float(broken_score),
        'seo_score': float(seo_score),
        'overall_score': float(overall_score),
        'broken_elements_list': broken_elements_list[:20],  # Ограничиваем для предотвращения перегрузки
        'seo_issues_list': seo_issues_list[:20]
    }

    return result

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    url = request.form.get('url')

    if not url:
        return jsonify({'error': 'URL не указан'})

    # Проверяем, что URL начинается с http:// или https://
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url

    try:
        result = analyze_website_with_seo(url, model)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
