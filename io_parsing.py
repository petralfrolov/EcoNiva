import streamlit as st
import pandas as pd
import pdfplumber
from io import BytesIO
from config import FEED_MAP, TARGET_COLS
import re


def _classify_by_code(clean_name: str, logs: list) -> str | None:
    """
    Пытается классифицировать корм по его числовому коду.

    Ищет в названии компонента паттерн вида XXXX.XX.XX.XX.X.XX и
    использует четвертую пару чисел для определения типа корма
    (01 - сенаж, 02 - силос, 07 - корнаж).

    Args:
        clean_name (str): Очищенное название компонента.
        logs (list): Список для записи логов процесса.

    Returns:
        str | None: Название категории корма, если код опознан, иначе None.
    """
    match = re.search(r'\b(\d{4}\.\d{2}\.\d{2}\.(\d{2})\.\d\.\d{2})\b', clean_name)
    if not match:
        return None

    code_full, feed_type_code = match.groups()

    if feed_type_code == '01':
        category = 'Сенаж'
    elif feed_type_code == '02':
        category = 'Силос'
    elif feed_type_code == '07':
        category = 'Корнаж_ЗСК'
    else:
        logs.append(f"Найден код '{code_full}', но тип корма ('{feed_type_code}') не опознан.")
        return None

    logs.append(f"Ингредиент '{clean_name}' классифицирован по коду '{feed_type_code}' как '{category}'.")
    return category


def _aggregate_table(df_raw, logs):
    """
    Агрегирует данные из таблицы рациона, сопоставляя компоненты с категориями.

    Производит очистку данных, сопоставление по словарю `FEED_MAP` и
    через классификатор по коду `_classify_by_code`. Компоненты,
    которые не удалось сопоставить, возвращаются в отдельном списке.

    Args:
        df_raw (pd.DataFrame): Исходный DataFrame, извлеченный из файла.
        logs (list): Список для записи логов процесса.

    Returns:
        tuple: Кортеж из трех элементов:
               - dict: Словарь с агрегированными данными по категориям.
               - list: Обновленный список логов.
               - list: Список словарей с неопознанными компонентами.
    """
    logs.append(f"Обнаружены колонки: {list(df_raw.columns)}")
    ingredient_col = next((c for c in df_raw.columns if 'Ингредиент' in str(c)), None)
    sv_kg_col = next((c for c in df_raw.columns if 'СВ кг' in str(c)), None)
    if not ingredient_col or not sv_kg_col:
        logs.append("ОШИБКА: В таблице не найдены обязательные колонки 'Ингредиенты' или 'СВ кг'.")
        return {}, logs, []

    logs.append(f"Колонка ингредиентов: '{ingredient_col}', колонка данных: '{sv_kg_col}'")
    df = df_raw[[ingredient_col, sv_kg_col]].copy()

    df[sv_kg_col] = pd.to_numeric(df[sv_kg_col].astype(str).str.replace(',', '.'), errors='coerce')
    df.dropna(inplace=True)
    logs.append(f"Найдено {len(df)} строк с числовыми данными.")

    df['clean_name'] = (df[ingredient_col].astype(str).str.split('/', n=1).str[0]
                        .str.replace(r'\s{2,}', ' ', regex=True)
                        .str.strip('.,\\ '))

    aggregated_data = {key: 0.0 for key in FEED_MAP.keys()}
    unclassified = []

    logs.append("\n--- Построчное сопоставление ---")
    for index, row in df.iterrows():
        if 'общие значения' in row['clean_name'].lower():
            logs.append(
                f"Строка {index + 1}: '{row[ingredient_col]}' -> ✗ Общие значения, пропускаем")
            continue

        match_found = False
        for category, names in FEED_MAP.items():
            if any(name.strip().lower() in row['clean_name'].lower() for name in names):
                aggregated_data[category] += float(row[sv_kg_col])
                logs.append(
                    f"Строка {index + 1}: '{row[ingredient_col]}' -> '{row['clean_name']}' -> ✓ '{category}' ({row[sv_kg_col]:.2f} кг)")
                match_found = True
                break

        if not match_found:
            category_from_code = _classify_by_code(row['clean_name'], logs)
            if category_from_code:
                aggregated_data[category_from_code] += float(row[sv_kg_col])
                match_found = True

        if not match_found:
            logs.append(
                f"Строка {index + 1}: '{row[ingredient_col]}' -> '{row['clean_name']}' -> ✗ Категория не найдена")
            unclassified.append({
                "name": row['clean_name'],
                "value": float(row[sv_kg_col])
            })

    logs.append("\n--- Итог агрегации ---")
    logs.append(str({k: round(v, 2) for k, v in aggregated_data.items() if v > 0}))

    if sum(aggregated_data.values()) < 0.01 and not unclassified:
        logs.append("ПРЕДУПРЕЖДЕНИЕ: Сумма всех найденных компонентов равна нулю. Данные не загружены.")
        return {}, logs, []

    return aggregated_data, logs, unclassified


@st.cache_data
def parse_pdf_report(uploaded_file):
    """
    Извлекает данные о рационе из PDF-файла.

    Использует `pdfplumber` для поиска и извлечения таблиц из файла,
    после чего передает их в `_aggregate_table` для обработки.

    Args:
        uploaded_file: Загруженный пользователем PDF-файл.

    Returns:
        tuple: Результат вызова `_aggregate_table`.
    """
    logs = []
    if uploaded_file is None:
        return {}, ["Файл не был загружен."], []

    logs.append(f"Начало обработки файла: {uploaded_file.name}")
    tables = []
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            for _, page in enumerate(pdf.pages):
                extracted = page.extract_tables()
                if extracted:
                    tables.extend(extracted)
        logs.append(f"Найдено таблиц в PDF: {len(tables)}")
    except Exception as e:
        logs.append(f"КРИТИЧЕСКАЯ ОШИБКА при чтении PDF: {e}")
        return {}, logs, []

    if not tables or len(tables[0]) < 2:
        logs.append("ОШИБКА: Не удалось найти подходящую таблицу с данными в PDF.")
        return {}, logs, []

    df = pd.DataFrame(tables[0][1:], columns=tables[0][0])
    df.columns = df.iloc[0]
    df = df[1:].reset_index(drop=True)

    return _aggregate_table(df, logs)


@st.cache_data
def parse_excel_report(uploaded_file):
    """
    Извлекает данные о рационе из Excel-файла, в том числе из плохо
    структурированных файлов с объединенными ячейками и смещенными заголовками.

    Args:
        uploaded_file: Загруженный пользователем Excel-файл.

    Returns:
        tuple: Результат вызова `_aggregate_table`.
    """
    logs = []
    if uploaded_file is None:
        return {}, ["Файл не был загружен."], []

    logs.append(f"Начало обработки Excel: {uploaded_file.name}")
    df_raw = None

    try:
        # Пытаемся прочитать первый лист без заголовков, чтобы проанализировать его структуру
        df_raw = pd.read_excel(uploaded_file, header=None, sheet_name=0)
    except Exception as e:
        logs.append(f"КРИТИЧЕСКАЯ ОШИБКА при чтении Excel: {e}")
        return {}, logs, []

    # --- Начало новой логики для "кривых" файлов ---

    # 1. Ищем строку, где находятся настоящие заголовки (по слову "Ингредиент")
    header_row_index = -1
    for i, row in df_raw.iterrows():
        # Проверяем, есть ли искомое слово в какой-либо ячейке строки
        if any('Ингредиент' in str(cell) for cell in row):
            header_row_index = i
            logs.append(f"Найдена строка с заголовками, её индекс: {header_row_index}")
            break

    if header_row_index == -1:
        logs.append("ОШИБКА: В файле не найдена строка с заголовком 'Ингредиент'.")
        # Попробуем передать "как есть" в старую логику на всякий случай
        return _aggregate_table(df_raw, logs)

    # 2. Ищем конец таблицы данных (по строке "Общее значение")
    footer_row_index = len(df_raw)  # По умолчанию берем до конца
    for i in range(header_row_index, len(df_raw)):
        # Проверяем первую ячейку строки
        cell_value = str(df_raw.iloc[i, 0])
        if 'Общее значение' in cell_value or 'Сводный анализ' in cell_value:
            footer_row_index = i
            logs.append(f"Найдена строка с итогами ('{cell_value}'), её индекс: {footer_row_index}")
            break

    # 3. Вырезаем только нужный кусок DataFrame
    df_clean = df_raw.iloc[header_row_index:footer_row_index].copy()

    # 4. Устанавливаем правильные заголовки
    df_clean.columns = df_clean.iloc[0].astype(str)
    # Удаляем строку, которая теперь стала заголовком
    df_clean = df_clean.iloc[1:].reset_index(drop=True)
    logs.append(f"Установлены новые заголовки: {list(df_clean.columns)}")

    # 5. "Чиним" объединенные ячейки в первом столбце (самое важное)
    # Метод ffill() (forward fill) заполняет пустые ячейки значением из предыдущей непустой.
    ingredient_col_name = df_clean.columns[0]
    df_clean[ingredient_col_name] = df_clean[ingredient_col_name].ffill()
    logs.append(
        f"Применено прямое заполнение (ffill) для столбца '{ingredient_col_name}' для исправления объединенных ячеек.")

    # 6. Удаляем полностью пустые строки, если они есть
    df_clean.dropna(how='all', inplace=True)

    # --- Конец новой логики ---

    # Передаем уже очищенный и структурированный DataFrame в функцию агрегации
    logs.append("Передача очищенного DataFrame в функцию _aggregate_table.")
    return _aggregate_table(df_clean, logs)


@st.cache_data
def parse_any_report(uploaded_file):
    """
    Универсальный парсер, который вызывает нужную функцию в зависимости от расширения файла.

    Args:
        uploaded_file: Загруженный пользователем файл.

    Returns:
        tuple: Результат вызова `parse_pdf_report` или `parse_excel_report`.
    """
    if uploaded_file is None:
        return {}, ["Файл не был загружен."], []
    name = uploaded_file.name.lower()
    if name.endswith(".pdf"):
        return parse_pdf_report(uploaded_file)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return parse_excel_report(uploaded_file)
    return {}, [f"Неподдерживаемый формат: {uploaded_file.name}. Ожидается PDF или Excel."], []


@st.cache_data
def parse_pdf_nutrients(uploaded_file):
    """
    Читает PDF-отчёт с таблицами нутриентов и возвращает агрегированные значения
    по TARGET_COLS в формате: (aggregated_data: dict[str,float], logs: list[str], unclassified: list[dict])

    Логика обработки таблицы совпадает с примером из ноутбука:
      - поиск таблицы со столбцами 'СП' ИЛИ 'Нутриент'
      - переименование столбцов до вида ['Нутриент', ..., 'Содержание', ...]
      - очистка названий и чисел
      - свёртка в wide-строку и отбор TARGET_COLS (отсутствующие -> 0.0)
    """
    logs = []
    if uploaded_file is None:
        return {}, ["Файл не был загружен."], []

    logs.append(f"Начало обработки файла: {uploaded_file.name}")

    # 1) собрать все таблицы из PDF
    tables = []
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            for p_i, page in enumerate(pdf.pages, start=1):
                extracted = page.extract_tables()
                if extracted:
                    logs.append(f"Стр. {p_i}: найдено таблиц: {len(extracted)}")
                    tables.extend(extracted)
    except Exception as e:
        logs.append(f"КРИТИЧЕСКАЯ ОШИБКА при чтении PDF: {e}")
        return {}, logs, []

    if not tables:
        logs.append("ОШИБКА: Таблиц в PDF не найдено.")
        return {}, logs, []

    # 2) найти первую «подходящую» таблицу нутриентов и привести к единому виду
    df = None
    chosen_idx = None
    for i, tbl in enumerate(tables):
        # минимальная валидация
        if not tbl or len(tbl) < 2:
            continue
        # конструируем DataFrame
        try:
            tmp = pd.DataFrame(tbl[1:], columns=tbl[0])
            if tmp.shape[0] == 0:
                continue

            # как в ноутбуке: объявляем заголовком первую строку
            tmp.columns = tmp.iloc[0]
            # ВЕТКА 1: в заголовках есть 'СП' -> переименовываем спец-столбцы
            if 'СП' in tmp.columns:
                cols = list(tmp.columns)
                if len(cols) >= 2:
                    cols[0] = 'Нутриент'
                if len(cols) >= 2:
                    cols[-2] = 'Содержание'
                tmp.columns = cols
                df = tmp.copy()
                chosen_idx = i
                logs.append(f"Выбрана таблица {i}: по признаку наличия столбца 'СП'.")
                break

            # ВЕТКА 2: уже есть 'Нутриент' -> срезаем повтор заголовка и чиним 'Содержани'
            if 'Нутриент' in tmp.columns:
                tmp = tmp[1:].reset_index(drop=True)
                tmp = tmp.rename(columns={'Содержани': 'Содержание'})
                if 'Содержание' in tmp.columns:
                    df = tmp.copy()
                    chosen_idx = i
                    logs.append(f"Выбрана таблица #{i}: по признаку наличия столбца 'Нутриент'.")
                    break
        except Exception as e:
            logs.append(f"Пропускаю таблицу #{i}: ошибка преобразования: {e}")
            continue

    if df is None:
        logs.append("ОШИБКА: Не найдено ни одной таблицы с колонками 'СП' или 'Нутриент'.")
        return {}, logs, []

    if 'Нутриент' not in df.columns or 'Содержание' not in df.columns:
        logs.append(f"ОШИБКА: В выбранной таблице нет обязательных столбцов. Колонки: {list(df.columns)}")
        return {}, logs, []

    logs.append(f"Колонки выбранной таблицы: {list(df.columns)}")

    # 3) очистка названий нутриентов
    names = (
        df['Нутриент'].astype(str)
        .str.split('/', n=1).str[0]
        .str.strip(' .,\u00A0')
    )
    df = df.copy()
    df['Нутриент'] = names

    # 4) очистка чисел
    clean = (
        df['Содержание'].astype(str)
        .str.replace('\u00A0', '', regex=False)  # неразрывные пробелы
        .str.replace(' ', '', regex=False)       # обычные пробелы
        .str.replace(',', '.', regex=False)      # запятая -> точка
        .replace({'': None, '-': None, '—': None})
    )
    df['Содержание'] = pd.to_numeric(clean, errors='coerce')

    before = len(df)
    df = df.dropna(subset=['Содержание'])
    logs.append(f"Удалено пустых строк: {before - len(df)}; осталось: {len(df)}.")

    if df.empty:
        logs.append("ОШИБКА: После очистки не осталось числовых значений.")
        return {}, logs, []

    # 5) свёртка в wide
    s = df.groupby('Нутриент', as_index=True)['Содержание'].sum()
    wide = s.to_frame().T
    wide.columns.name = None
    wide = wide.reset_index(drop=True)
    wide['__source'] = str(getattr(uploaded_file, "name", "pdf"))

    # 6) собрать aggregated_data по TARGET_COLS (отсутствующие -> 0.0)
    aggregated_data = {}
    for col in TARGET_COLS:
        if col in wide.columns:
            try:
                aggregated_data[col] = float(pd.to_numeric(wide[col].iloc[0], errors='coerce') or 0.0)
            except Exception:
                aggregated_data[col] = 0.0
        else:
            aggregated_data[col] = 0.0

    # округлять или нет — оставляем как есть; лог выведем округлённый
    logs.append("\n--- Итог парсинга ---\n" +
                str({k: round(v, 3) for k, v in aggregated_data.items()}))

    # формат возвращаем такой же, как у существующих парсеров: dict, logs, unclassified
    return aggregated_data, logs, []


@st.cache_data
def parse_excel_nutrients(uploaded_file):
    """
    Читает Excel-отчёт с таблицами нутриентов, в том числе разорванными на части,
    и возвращает агрегированные значения по TARGET_COLS.
    """
    logs = []
    if uploaded_file is None:
        return {}, ["Файл не был загружен."], []

    logs.append(f"Начало обработки файла нутриентов Excel: {uploaded_file.name}")

    try:
        # 1. Читаем первый лист без заголовков, чтобы проанализировать его структуру
        df_raw = pd.read_excel(uploaded_file, header=None, sheet_name=0)
    except Exception as e:
        logs.append(f"КРИТИЧЕСКАЯ ОШИБКА при чтении Excel: {e}")
        return {}, logs, []

    # 2. Ищем все строки-заголовки по слову "Нутриент"
    header_indices = df_raw[df_raw.apply(
        lambda row: row.astype(str).str.contains('Нутриент').any(), axis=1
    )].index.tolist()

    if not header_indices:
        logs.append("ОШИБКА: Не найдено ни одной таблицы с заголовком 'Нутриент'.")
        return {}, logs, []

    logs.append(f"Найдены заголовки на строках: {header_indices}")

    # 3. Определяем корректное количество колонок и их имена по первому заголовку
    header_series = df_raw.iloc[header_indices[0]]
    last_valid_col = header_series.last_valid_index()
    column_names = header_series.iloc[:last_valid_col + 1].tolist()
    num_columns = len(column_names)
    logs.append(f"Определены {num_columns} колонок: {column_names}")

    # 4. Собираем и объединяем все части таблицы
    all_data_chunks = []
    for i, start_index in enumerate(header_indices):
        end_index = header_indices[i + 1] if i + 1 < len(header_indices) else len(df_raw)
        # Вырезаем данные, используя определенное количество колонок
        chunk = df_raw.iloc[start_index + 1: end_index, :num_columns]
        all_data_chunks.append(chunk)

    df = pd.concat(all_data_chunks, ignore_index=True)
    df.columns = column_names

    # 5. Базовая очистка объединенной таблицы
    df.dropna(subset=['Нутриент'], inplace=True)
    df = df[df['Нутриент'] != 'Нутриент'].reset_index(drop=True)

    # Переименуем обрезанные колонки для единообразия
    df = df.rename(columns={'Содержан': 'Содержание', 'Единиц': 'Единицы', 'Едини': 'Единицы'})

    if 'Нутриент' not in df.columns or 'Содержание' not in df.columns:
        logs.append(
            f"ОШИБКА: В таблице нет обязательных столбцов 'Нутриент' или 'Содержание'. Найдено: {list(df.columns)}")
        return {}, logs, []

    # 6. Очистка названий нутриентов (аналогично parse_pdf_nutrients)
    df['Нутриент'] = (df['Нутриент'].astype(str)
                      .str.split('/', n=1).str[0]
                      .str.strip(' .,\u00A0'))

    # 7. Очистка числовых значений (аналогично parse_pdf_nutrients)
    clean_values = (df['Содержание'].astype(str)
                    .str.replace(r'[^\d,.-]', '', regex=True)  # Оставляем только цифры, запятую, точку, минус
                    .str.replace(',', '.', regex=False)
                    .replace({'': None, '-': None, '—': None}))
    df['Содержание'] = pd.to_numeric(clean_values, errors='coerce')
    df.dropna(subset=['Содержание'], inplace=True)

    if df.empty:
        logs.append("ОШИБКА: После очистки в таблице не осталось валидных числовых данных.")
        return {}, logs, []

    logs.append(f"После очистки осталось {len(df)} строк с данными.")

    # 8. Сворачиваем таблицу в "широкий" формат
    s = df.groupby('Нутриент')['Содержание'].sum()
    wide = s.to_frame().T

    # 9. Собираем итоговый словарь по TARGET_COLS
    aggregated_data = {}
    for col in TARGET_COLS:
        if col in wide.columns:
            aggregated_data[col] = float(wide[col].iloc[0])
        else:
            aggregated_data[col] = 0.0

    logs.append("\n--- Итог парсинга нутриентов ---\n" + str({k: round(v, 3) for k, v in aggregated_data.items()}))

    return aggregated_data, logs, []
