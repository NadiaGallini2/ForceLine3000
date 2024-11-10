# Force Line - Система автоматической диспетчеризации заявок 📧

## Описание
**Force Line** — это мощный инструмент, предназначенный для автоматической диспетчеризации заявок на основе сообщений электронной почты. Приложение позволяет загружать, анализировать, классифицировать и экспортировать данные заявок для оптимизации процесса управления заявками. Поддерживает обработку данных в форматах CSV, Excel и XML.

## Возможности
- **Загрузка данных**: Загружайте несколько файлов в форматах CSV или Excel с данными заявок.
- **Анализ данных**: Анализируйте содержимое заявок для извлечения ключевых параметров.
- **Классификация заявок**: Классифицируйте заявки по типам (например, «Техническая» и «Общая»).
- **Экспорт данных**: Экспортируйте обработанные данные в форматы CSV или XML для дальнейшего использования.

## Установка

Для того чтобы запустить приложение локально, выполните следующие шаги:

### 1. Клонировать репозиторий
Сначала клонируйте репозиторий на свой локальный компьютер:

```bash
https://github.com/NadiaGallini2/ForceLine
cd ForceLine
```

### 2. Создание виртуального окружения (по желанию)
Рекомендуется использовать виртуальное окружение для вашего проекта. Вы можете создать его с помощью venv или conda:

Используя venv:
```bash
python3 -m venv new_env
new_env\Scripts\activate
```
Используя conda:
```bash
conda create --name force-line python=3.10
conda activate force-line
```


pandas_profiling

### 3. Установка зависимостей
Установите все необходимые библиотеки Python, выполнив:
```bash
pip install -r requirements.txt

pip install -r requirements.txt --proxy=http://127.0.0.1:10801

pip install pandas_profiling --proxy=http://127.0.0.1:10801

pip install joblib --proxy=http://127.0.0.1:10801

pip install D:\ForceLine3000\pandas_profiling-3.6.6-py2.py3-none-any.whl --proxy=http://127.0.0.1:10801 

pip install sklearn --proxy=http://127.0.0.1:10801
```

## Использование
Запуск приложения Чтобы запустить приложение Streamlit, выполните команду:

```bash
streamlit run app.py
```