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
    Извлекает данные о рационе из Excel-файла.

    Использует `pandas` для чтения файла. Пытается найти лист,
    содержащий колонки 'Ингредиент' и 'СВ кг'. Если такой лист не найден,
    используется первый лист. Данные передаются в `_aggregate_table`.

    Args:
        uploaded_file: Загруженный пользователем Excel-файл.

    Returns:
        tuple: Результат вызова `_aggregate_table`.
    """
    logs = []
    if uploaded_file is None:
        return {}, ["Файл не был загружен."], []

    logs.append(f"Начало обработки Excel: {uploaded_file.name}")
    try:
        xls = pd.ExcelFile(uploaded_file)
        candidate = None
        for sheet in xls.sheet_names:
            df_try = xls.parse(sheet)
            cols = [str(c) for c in df_try.columns]
            if any('Ингредиент' in c for c in cols) and any('СВ кг' in c for c in cols):
                candidate = df_try
                logs.append(f"Выбрана вкладка: {sheet}")
                break
        if candidate is None:
            logs.append("Подходящих вкладок не найдено, используем первый лист.")
            candidate = xls.parse(xls.sheet_names[0])
    except Exception as e:
        logs.append(f"КРИТИЧЕСКАЯ ОШИБКА при чтении Excel: {e}")
        return {}, logs, []

    return _aggregate_table(candidate, logs)


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
