document.addEventListener('DOMContentLoaded', function() {
    // Элементы интерфейса
    const analyzeForm = document.getElementById('analyze-form');
    const urlInput = document.getElementById('url');
    const analyzeBtn = document.getElementById('analyze-btn');
    const loadingIndicator = document.getElementById('loading');
    const resultsSection = document.getElementById('results');
    const tabButtons = document.querySelectorAll('.tab-btn');
    const tabPanes = document.querySelectorAll('.tab-pane');

    // Переключение вкладок
    tabButtons.forEach(button => {
        button.addEventListener('click', function() {
            // Удаляем активный класс со всех кнопок
            tabButtons.forEach(btn => btn.classList.remove('active'));
            // Добавляем активный класс текущей кнопке
            this.classList.add('active');

            // Получаем идентификатор вкладки
            const tabId = this.dataset.tab;

            // Скрываем все вкладки
            tabPanes.forEach(pane => pane.classList.remove('active'));

            // Показываем нужную вкладку
            document.getElementById(`${tabId}-tab`).classList.add('active');
        });
    });

    // Обработка отправки формы
    analyzeForm.addEventListener('submit', function(e) {
        e.preventDefault();

        const url = urlInput.value.trim();
        if (!url) {
            alert('Пожалуйста, введите URL для анализа');
            return;
        }

        // Показываем индикатор загрузки
        loadingIndicator.classList.remove('hidden');
        resultsSection.classList.add('hidden');
        analyzeBtn.disabled = true;

        // Отправляем запрос на анализ
        const formData = new FormData();
        formData.append('url', url);

        fetch('/analyze', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            // Скрываем индикатор загрузки
            loadingIndicator.classList.add('hidden');
            analyzeBtn.disabled = false;

            // Проверяем наличие ошибок
            if (data.error) {
                alert(`Ошибка анализа: ${data.error}`);
                return;
            }

            // Отображаем результаты
            displayResults(data);
            resultsSection.classList.remove('hidden');

            // Прокручиваем страницу к результатам
            resultsSection.scrollIntoView({ behavior: 'smooth' });
        })
        .catch(error => {
            loadingIndicator.classList.add('hidden');
            analyzeBtn.disabled = false;
            alert(`Произошла ошибка: ${error.message}`);
        });
    });

    // Функция для отображения результатов
    function displayResults(data) {
        // Заполняем основные данные
        document.getElementById('result-url').textContent = data.url;
        document.getElementById('total-elements').textContent = data.total_elements;
        document.getElementById('broken-elements').textContent = data.broken_elements;
        document.getElementById('broken-percent').textContent = data.broken_percent.toFixed(1) + '%';
        document.getElementById('seo-issues').textContent = data.seo_issues;
        document.getElementById('seo-issues-percent').textContent = data.seo_issues_percent.toFixed(1) + '%';

        // Отображаем общие оценки и устанавливаем цвета
        updateScoreDisplay('overall-score', data.overall_score);
        updateScoreDisplay('broken-score', data.broken_score);
        updateScoreDisplay('seo-score', data.seo_score);

        // Создаем графики
        createBrokenTypesChart(data.broken_types);
        createSeoTypesChart(data.seo_issues_types);
        createTagsCharts(data.broken_by_tag, data.seo_by_tag);

        // Отображаем списки проблемных элементов
        displayBrokenElements(data.broken_elements_list);
        displaySeoIssues(data.seo_issues_list);

        // Генерируем рекомендации
        generateRecommendations(data);
    }

    // Обновление отображения оценок
    function updateScoreDisplay(elementId, score) {
        const scoreElement = document.getElementById(elementId);
        const scoreCircle = document.getElementById(elementId + '-circle');

        // Устанавливаем значение
        scoreElement.textContent = Math.round(score);

        // Устанавливаем цвет в зависимости от оценки
        if (score >= 90) {
            scoreCircle.className = 'score-circle good';
        } else if (score >= 50) {
            scoreCircle.className = 'score-circle average';
        } else {
            scoreCircle.className = 'score-circle poor';
        }
    }

    // Создание графика типов битых элементов
    function createBrokenTypesChart(brokenTypes) {
        const labels = {
            'missing_required_attr': 'Отсутствие обязательных атрибутов',
            'has_empty_attr': 'Пустые атрибуты',
            'css_conflict': 'Конфликты CSS',
            'has_display_none': 'Скрытые элементы',
            'nested_structure_issue': 'Проблемы вложенности'
        };

        const data = [];
        const chartLabels = [];

        for (const [key, value] of Object.entries(brokenTypes)) {
            if (value > 0) {
                chartLabels.push(labels[key] || key);
                data.push(value);
            }
        }

        const ctx = document.getElementById('broken-types-chart').getContext('2d');

        if (window.brokenTypesChart) {
            window.brokenTypesChart.destroy();
        }

        window.brokenTypesChart = new Chart(ctx, {
            type: 'pie',
            data: {
                labels: chartLabels,
                datasets: [{
                    data: data,
                    backgroundColor: [
                        '#ea4335',
                        '#fbbc05',
                        '#34a853',
                        '#4285f4',
                        '#9c27b0'
                    ],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'right'
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const label = context.label || '';
                                const value = context.formattedValue;
                                const total = context.dataset.data.reduce((a, b) => a + b, 0);
                                const percentage = Math.round((context.raw / total) * 100);
                                return `${label}: ${value} (${percentage}%)`;
                            }
                        }
                    }
                }
            }
        });
    }

    // Создание графика типов SEO-проблем
    function createSeoTypesChart(seoTypes) {
        const labels = {
            'missing_alt_tag': 'Отсутствие alt у изображений',
            'empty_heading': 'Пустые заголовки',
            'missing_meta_description': 'Отсутствие meta description',
            'incorrect_heading_hierarchy': 'Неправильная иерархия заголовков',
            'duplicate_h1': 'Дублирование h1',
            'missing_link_title': 'Отсутствие title у ссылок',
            'long_url': 'Длинные URL',
            'invalid_internal_link': 'Невалидные ссылки'
        };

        const data = [];
        const chartLabels = [];

        for (const [key, value] of Object.entries(seoTypes)) {
            if (value > 0) {
                chartLabels.push(labels[key] || key);
                data.push(value);
            }
        }

        const ctx = document.getElementById('seo-types-chart').getContext('2d');

        if (window.seoTypesChart) {
            window.seoTypesChart.destroy();
        }

        window.seoTypesChart = new Chart(ctx, {
            type: 'pie',
            data: {
                labels: chartLabels,
                datasets: [{
                    data: data,
                    backgroundColor: [
                        '#ea4335',
                        '#fbbc05',
                        '#34a853',
                        '#4285f4',
                        '#9c27b0',
                        '#ff6d01',
                        '#46bdc6',
                        '#795548'
                    ],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'right'
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const label = context.label || '';
                                const value = context.formattedValue;
                                const total = context.dataset.data.reduce((a, b) => a + b, 0);
                                const percentage = Math.round((context.raw / total) * 100);
                                return `${label}: ${value} (${percentage}%)`;
                            }
                        }
                    }
                }
            }
        });
    }

    // Создание графиков распределения по тегам
    function createTagsCharts(brokenByTag, seoByTag) {
        // Объединяем теги из обоих объектов
        const allTags = new Set([...Object.keys(brokenByTag), ...Object.keys(seoByTag)]);
        const tags = Array.from(allTags);

        // Формируем данные для графиков
        const brokenData = tags.map(tag => (brokenByTag[tag] || 0) * 100);
        const seoData = tags.map(tag => (seoByTag[tag] || 0) * 100);

        // График битых элементов по тегам
        const brokenCtx = document.getElementById('broken-by-tag-chart').getContext('2d');

        if (window.brokenByTagChart) {
            window.brokenByTagChart.destroy();
        }

        window.brokenByTagChart = new Chart(brokenCtx, {
            type: 'bar',
            data: {
                labels: tags,
                datasets: [{
                    label: 'Процент битых элементов',
                    data: brokenData,
                    backgroundColor: 'rgba(234, 67, 53, 0.7)',
                    borderColor: 'rgba(234, 67, 53, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        title: {
                            display: true,
                            text: 'Процент'
                        }
                    }
                }
            }
        });

        // График SEO-проблем по тегам
        const seoCtx = document.getElementById('seo-by-tag-chart').getContext('2d');

        if (window.seoByTagChart) {
            window.seoByTagChart.destroy();
        }

        window.seoByTagChart = new Chart(seoCtx, {
            type: 'bar',
            data: {
                labels: tags,
                datasets: [{
                    label: 'Процент элементов с SEO-проблемами',
                    data: seoData,
                    backgroundColor: 'rgba(66, 133, 244, 0.7)',
                    borderColor: 'rgba(66, 133, 244, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        title: {
                            display: true,
                            text: 'Процент'
                        }
                    }
                }
            }
        });
    }

    // Отображение списка битых элементов
    function displayBrokenElements(elements) {
        const container = document.getElementById('broken-elements-list');
        container.innerHTML = '';

        if (elements.length === 0) {
            container.innerHTML = '<p class="empty-list">Битых элементов не обнаружено.</p>';
            return;
        }

        elements.forEach(element => {
            const item = document.createElement('div');
            item.className = 'element-item';

            const header = document.createElement('div');
            header.className = 'element-header';

            const tag = document.createElement('span');
            tag.className = 'element-tag';
            tag.textContent = element.tag;

            const probability = document.createElement('span');
            probability.className = 'element-probability';
            probability.textContent = `Вероятность проблемы: ${(element.probability * 100).toFixed(1)}%`;

            header.appendChild(tag);
            header.appendChild(probability);

            const html = document.createElement('div');
            html.className = 'element-html';
            html.textContent = element.html;

            const problems = document.createElement('ul');
            problems.className = 'element-problems';

            element.reasons.forEach(reason => {
                const li = document.createElement('li');
                li.textContent = reason;
                problems.appendChild(li);
            });

            item.appendChild(header);
            item.appendChild(html);
            item.appendChild(problems);

            container.appendChild(item);
        });
    }

    // Отображение списка элементов с SEO-проблемами
    function displaySeoIssues(elements) {
        const container = document.getElementById('seo-issues-list');
        container.innerHTML = '';

        if (elements.length === 0) {
            container.innerHTML = '<p class="empty-list">Элементов с SEO-проблемами не обнаружено.</p>';
            return;
        }

        elements.forEach(element => {
            const item = document.createElement('div');
            item.className = 'element-item';

            const header = document.createElement('div');
            header.className = 'element-header';

            const tag = document.createElement('span');
            tag.className = 'element-tag';
            tag.textContent = element.tag;

            header.appendChild(tag);

            const html = document.createElement('div');
            html.className = 'element-html';
            html.textContent = element.html;

            const issues = document.createElement('ul');
            issues.className = 'element-problems';

            element.issues.forEach(issue => {
                const li = document.createElement('li');
                li.textContent = issue;
                issues.appendChild(li);
            });

            item.appendChild(header);
            item.appendChild(html);
            item.appendChild(issues);

            container.appendChild(item);
        });
    }

    // Генерация рекомендаций
    function generateRecommendations(data) {
        const container = document.getElementById('recommendations-list');
        container.innerHTML = '';

        const recommendations = [];

        // Рекомендации по битым элементам
        if (data.broken_elements > 0) {
            if (data.broken_types.missing_required_attr > 0) {
                recommendations.push({
                    title: 'Добавьте отсутствующие обязательные атрибуты',
                    content: `На сайте обнаружено ${data.broken_types.missing_required_attr} элементов с отсутствующими обязательными атрибутами. Добавьте атрибуты src для изображений, href для ссылок и type для полей ввода.`,
                    icon: 'fas fa-exclamation-triangle'
                });
            }

            if (data.broken_types.has_empty_attr > 0) {
                recommendations.push({
                    title: 'Исправьте пустые атрибуты',
                    content: `На сайте обнаружено ${data.broken_types.has_empty_attr} элементов с пустыми атрибутами. Заполните атрибуты значениями или удалите их.`,
                    icon: 'fas fa-times-circle'
                });
            }

            if (data.broken_types.css_conflict > 0) {
                recommendations.push({
                    title: 'Устраните конфликты CSS',
                    content: `На сайте обнаружено ${data.broken_types.css_conflict} элементов с конфликтами CSS-стилей. Проверьте стили, которые могут делать элементы невидимыми или неинтерактивными.`,
                    icon: 'fas fa-code'
                });
            }

            if (data.broken_types.nested_structure_issue > 0) {
                recommendations.push({
                    title: 'Исправьте проблемы вложенности',
                    content: `На сайте обнаружено ${data.broken_types.nested_structure_issue} элементов с проблемами вложенности. Избегайте вложенных ссылок и других нестандартных вложений.`,
                    icon: 'fas fa-layer-group'
                });
            }
        }

        // Рекомендации по SEO
        if (data.seo_issues > 0) {
            if (data.seo_issues_types.missing_alt_tag > 0) {
                recommendations.push({
                    title: 'Добавьте атрибуты alt для изображений',
                    content: `На сайте обнаружено ${data.seo_issues_types.missing_alt_tag} изображений без атрибута alt. Добавьте информативные описания для всех изображений.`,
                    icon: 'fas fa-image'
                });
            }

            if (data.seo_issues_types.empty_heading > 0) {
                recommendations.push({
                    title: 'Заполните пустые заголовки',
                    content: `На сайте обнаружено ${data.seo_issues_types.empty_heading} пустых заголовков. Заполните их содержимым или удалите.`,
                    icon: 'fas fa-heading'
                });
            }

            if (data.seo_issues_types.missing_meta_description > 0) {
                recommendations.push({
                    title: 'Добавьте мета-описание',
                    content: 'На сайте отсутствует мета-тег description. Добавьте описание страницы для улучшения отображения в результатах поиска.',
                    icon: 'fas fa-file-alt'
                });
            }

            if (data.seo_issues_types.duplicate_h1 > 0) {
                recommendations.push({
                    title: 'Устраните дублирование заголовков h1',
                    content: 'На странице обнаружено несколько заголовков h1. Оставьте только один основной заголовок h1.',
                    icon: 'fas fa-clone'
                });
            }

            if (data.seo_issues_types.missing_link_title > 0) {
                recommendations.push({
                    title: 'Добавьте атрибуты title для ссылок',
                    content: `На сайте обнаружено ${data.seo_issues_types.missing_link_title} ссылок без атрибута title. Добавьте информативные подсказки для ссылок.`,
                    icon: 'fas fa-link'
                });
            }
        }

        // Общие рекомендации
        if (recommendations.length === 0) {
            recommendations.push({
                title: 'Отлично! Серьезных проблем не обнаружено',
                content: 'Ваш сайт хорошо структурирован с точки зрения DOM-элементов и SEO. Продолжайте следить за качеством контента и структуры.',
                icon: 'fas fa-check-circle'
            });
        }

        // Отображаем рекомендации
        recommendations.forEach(rec => {
            const item = document.createElement('div');
            item.className = 'recommendation-item';

            const title = document.createElement('div');
            title.className = 'recommendation-title';

            const icon = document.createElement('i');
            icon.className = rec.icon;

            const titleText = document.createElement('span');
            titleText.textContent = rec.title;

            title.appendChild(icon);
            title.appendChild(titleText);

            const content = document.createElement('div');
            content.className = 'recommendation-content';
            content.textContent = rec.content;

            item.appendChild(title);
            item.appendChild(content);

            container.appendChild(item);
        });
    }
});
