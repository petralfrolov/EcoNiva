import streamlit as st
import pandas as pd
import numpy as np
import pdfplumber
import pickle
import statsmodels.api as sm
import plotly.graph_objects as go
from pathlib import Path

# --- КОНФИГУРАЦИЯ И КОНСТАНТЫ (без изменений) ---
st.set_page_config(layout="wide", page_title="Анализатор рациона коров")
# ... (весь блок FEED_MAP, MODEL_PATHS, TARGET_RANGES, FEATURE_TO_COMPONENT_MAP остается без изменений)
FEED_MAP = {
    'Силос': [
        'Силос', 'силос', 'силос_', 'C-C', 'c-c', 'С-С', 'с-с', 'С-С 2242.05.05.02.1.23\\',
        '1603.02.05.02.1.24', '1630.10.05.02.1.23', '1630.14.05.02.1.23', '2221.02.05.02.1.23 Силос',
        '2531.04.05.02.1.24', '2531.05.05.02.1.23', '2531.05.05.02.1.24', '3635.05.05.02.1.24',
        '3638.04.05.02.1.23', '5100.01.05.02.1.24', '5210.07.05.02.1.24', '5701.02.05.02.1.24',
        '5701.03.05.02.1.24', '6101.08.06.02.1.24', '6110.09.05.02.1.24', '6601.40.05.02.1.24',
        'C-C 2261.03.05.02.1.23', 'c-c 4029.04.05.02.1.23', 'С-С 2241.04.05.02.1.24 СР',
        'С-С 2242.01.05.02.1.24', 'С-С 2242.03.05.02.1.24', 'С-С 2242.03.05.02.1.24 \\ 12.03.25',
        'С-С 2242.05.05.02.1.23\\\n11.04.24', 'С-С 2261.04.05.02.1.23 \\ 12.03.25',
        'С-С 2262.04.05.02.1.24', 'С-С 2392.05.05.02.1.24', 'С-С 2392.07.05.02.1.24 \\ 12.03.25',
        'С-с 4029.01.05.02.1.24 Уланово22.04.25', 'С-с 6201.10.05.02.1.23', 'С-с 6203.05.05.02.1.24',
        'Силос 22.02.02.05.02.1.24', 'Силос 2201.03.05.02.1.24', 'Силос 2211.03.05.02.1.24',
        'Силос 2221.03.05.02.1.24', 'Силос 2232.03.05.02.1.24', 'Силос 2291.03.05.02.1.24',
        'Силос 2291.04.05.02.1.24', 'Силос 2371.04.05.02.1.24 -', 'Силос 2371.05.05.02.1.23 -',
        'Силос 2501.05.05.02.1.24 -', 'Силос 3636.07.05.02.1.24 -', 'Силос 3637.05.05.02.1.23 -',
        'Силос 3637.06.05.02.1.23 -', 'Силос 3637.07.05.02.1.24 -', 'с-с 17.04\\5901.01.05.02.1.24 Аристово',
        'с-с 21.03.25\\5204.01.05.02.1.24 Болдасовка', 'с-с 23.06.2025\\4030.01.05.02.1.24 Прудки',
        'с-с 23.06.2025\\5901.01.05.02.1.24.Аристово', 'с-с 25.03.25\\5901.01.05.02.1.24 Аристово',
        'с-с 27.02.25\\4030.01.05.02.1.24 Прудки', 'с-с 4029.03.05.02.1.23', 'с-с 6204.01.05.02.1.24',
        'силос 9202.04.05.02.1.24', 'силос9202.01.05.02.1.24', 'силос_9202.04.05.02.1.24',
        'кукуруза6601.40.05.02.1.24.15.0'
    ],
    'Сенаж': [
        'Сенаж', 'С-Ж', 'с-ж', 'Люцерна', 'люцерна', 'люц_', 'суданка', 'рожь_',
        '1603.03.01.01.1.24', '1603.05.01.01.1.24', '1630.01.12.01.1.24', '1630.02.01.01.1.24',
        '1630.03.01.01.1.24', '1630.07.02.01.1.24', '2531.01.01.01.2.24', '3635.01.01.01.1.23',
        '3635.01.01.01.1.24', '3638.02.01.01.2.24', '3638.02.01.01.3.24 CP', '5001.01.01.01.1.23',
        '5001.02.16.01.1.23', '5100.01.10.01.1.24', '5210.03.03.01.1.24', '5701.01.01.01.1.24',
        '5701.01.01.01.1.25', '5701.04.01.01.3.24', '6110.01.01.01.1.24 люцерна', '6110.06.18.01.1.24',
        '6110.10.18.01.1.24', '6601.07.06.01.1.24', '6601.25.01.01.2.24', 'c-ж 6204.01.12.01.1.24',
        'Люцерна 2501.03.01.01.2.24 -', 'Люцерна 2501.04.08.01.1.24 -', 'Люцерна 3637.01.01.01.1.24 -',
        'Люцерна 3637.03.01.01.2.23 -', 'Люцерна 3637.03.01.01.2.24 -', 'Люцерна 3637.04.01.01.3.24 -',
        'Люцерна2371.01.01.01.2.24 -', 'С-Ж 2241.02.04.01.1.24', 'С-Ж 2241.03.04.01.1.23',
        'С-Ж 2242.01.04.01.1.23\\', 'С-Ж 2242.02.01.01.1.24', 'С-Ж 2242.02.08.01.1.24',
        'С-Ж 2242.02.08.01.1.24 \\ 12.03.25', 'С-Ж 2242.03.08.01.2.23\\', 'С-Ж 2262.01.01.01.1.24',
        'С-Ж 2262.01.01.01.1.24 \\ 12.03.25', 'С-Ж 2262.03.01.01.2.24', 'С-Ж 2262.05.10.01.1.24',
        'С-Ж 2262.05.10.01.1.24 \\ 12.03.25', 'С-Ж 2262.08.08.01.2.24', 'С-Ж 2392.01.01.01.1.24',
        'С-Ж 2392.01.01.01.1.24 \\ 12.03.25', 'С-Ж 2392.04.10.01.2.24', 'С-Ж 2392.04.10.01.2.24 \\ 12.03.25',
        'С-ж 4019.02.09.01.2.24 Косово- клевер', 'С-ж 4019.02.09.01.2.24 Косово1',
        'С-ж 4029.01.01.01.1.25 Уланово', 'С-ж 4032.01.01.01.1.25 Гусево', 'С-ж 5701.01.01.01.1.24 Бушовка',
        'С-ж 6201.07.01.01.1.24', 'С-ж 6202.02.01.01.1.24 03.02.25', 'Сенаж 22.02.01.01.01.1.24',
        'Сенаж 2201.02.01.01.2.24', 'Сенаж 2211.02.01.01.2.24', 'Сенаж 2221.02.08.01.1.24',
        'Сенаж 2232.02.08.01.2.24', 'Сенаж 2291.01.01.01.2.24', 'Сенаж 2291.02.01.01.3.24',
        'с-ж 17.04\\5902.01.01.01.1.24 Сугоново', 'с-ж 21.03.25', 'с-ж 21.03.25\\4030.06.01.01.1.24 Детчино',
        'с-ж 21.03.25\\5110.01.08.01.1.24 Дурово', 'с-ж 21.03.25\\5313.02.01.01.2.24',
        'с-ж 22.08.24\\4030.11.01.01.2.24 Санталовка', 'с-ж 6204.01.12.01.1.24', 'с-ж 6204.06.01.01.2.24',
        'с-ж 8.05.25\\5110.01.08.01.1.24 Дурово', 'с-ж 8.05.25\\5313.03.01.01.3.24 Поливаново',
        'с-ж4029.09.01.01.3.23', 'люц_9202.04.01.01.1.24', 'люцерна6601.29.08.01.1.24.15',
        'люцерна_9202.04.01.01.1.24\\06', 'рожь_9202.02.13.01.1.24',
        'суданка6601.30.06.01.2.24.15.0'
    ],
    'Сено': [
        'Сено из люцерны', 'Сено луговое', 'Сено луговое ЛБ', 'Сено люцерна ЛБ',
        'Сено луговое 2024', 'Сено люцерна ЛБ 2025'
    ],
    'Солома': [
        'Солома', 'солома', 'Солома ЭНА Восточное -', 'солома ячменная', 'Пшеничная солома',
        'Солома пшеничная', 'Солома пшеничная ЛБ', 'Солома-пшеничная ОМ Север', 'Дробленая солома КН Восток',
        '1603.02.15.04.1.24', 'Солома пшеничная 2024', 'солома6601.24.15.04.1.24.15.07'
    ],
    'Кукуруза': [
        'Кукуруза', 'кукуруза', 'Кукуруза влажная, кр',
        'Кукуруза. 70%,сухая средн.\nпомол, ЭНАПКХ',
        'Кукуруза. 70%,сухая, мелк.\nпомол, ЭНАПКХ',
        'Плющенная кукуруза ток', 'плющенная кукуруза',
        'Плющ зерно 6203.06.05.06.1.24', 'кукуруза_плющ_', 'плющенное',
        '1603.01.05.06.1.24', '6601.48.05.06.1.24', 'Кукуруза влажная, кр. 68%',
        'Кукуруза. 65%,сухая средн', 'Кукуруза. 65%,сухая средн. помол',
        'Кукуруза. 70%,сухая средн', 'Кукуруза. 70%,сухая средн. помол, ЭНАПКХ',
        'Кукуруза. 70%,сухая, мелк', 'Кукуруза. 70%,сухая, мелк. помол',
        'Кукуруза. 70%,сухая, мелк. помол, ЭНАПКХ',
        'Плющ зерно 6203.06.05.06.1.24\n03.02.25', 'Плющенная кукуруза ток1',
        'кукуруза_плющ_9202.01.05.06', 'плющенное6601.48.05.06.1.24'
    ],
    'Зерновые_прочие': [
        'Тритикале', 'Ячмень', 'Ячмень. сухой',
        'Ячмень. сухой 53%, средний\nпомол, ЭНАПКХ',
        'Ячмень.61%, сухой, средний\nпомол, ЭНАПКХ',
        'Пшеница', 'пшеница_консерв_СНБ',
        'Пшеница. 55%, сухая.средн', 'Пшеница. 55%, сухая.средн. помол',
        'Пшеница. 55%, сухая.средн. помол, ЭНАПКХ',
        'Тритикале 2501.02.12.01.1.24 -', 'Тритикале 3636.01.12.01.1.24 -',
        'Тритикале 3637.01.12.01.1.23 -', 'Тритикале 3637.02.12.01.1.24 -',
        'Ячмень. сухой 53%, средний', 'Ячмень. сухой 53%, средний помол',
        'Ячмень. сухой 53%, средний помол, ЭНАПКХ', 'Ячмень.60%, сухой, средний',
        'Ячмень.60%, сухой, средний помол', 'Ячмень.61%, сухой, средний'
    ],
    'Корнаж_ЗСК': [
        'Корнаж', 'корнаж', 'карнаж', 'корнаж_', 'ЗСК', 'ЗСК 6202.05.05.07.1.23', 'к-ж', 'к-ж Детч',
        '1603.01.05.07.1.24', '1630.17.05.07.1.24', '5100.01.05.07.1.24', '5701.01.05.07.1.24',
        '5701.01.05.1.23', '5701.04.05.07.1.24', '6601.49.05.07.1.24',
        'ЗСК 6201.05.05.07.1.22', 'ЗСК 6202.05.05.07.1.23\n03.02.2025 (Res#1) MA1',
        'Корнаж 2371.06.05.07.1.23 -', 'Корнаж 2501.08.05.07.1.23 -', 'Корнаж 2501.09.05.07.1.23 -',
        'к-ж 17.04\\5204.01.05.07.1.24 Болдасовка', 'к-ж 23.12.24\\5204.01.05.07.1.24 Болдасовка',
        'к-ж 6204.01.05.07.1.24', 'к-ж 6204.02.05.07.1.23', 'к-ж 8.05.25\\5204.01.05.07.1.24 Болдасовка',
        'к-ж Детч 4030.02.05.07.1.24_04.04.2025', 'к-ж4030.05.05.07.1.24 Детчино 25.06.25',
        'к-ж5701.05.07.1.23 Бушовка', 'карнаж6601.49.05.07.1.24.15.07', 'корнаж_9202.03.05.07.1.23'
    ],
    'Шрот_соевый': [
        'Шрот соевый', 'Шрот соевый. 47%, ЭНАПКХ', 'Шрот соевый. 49%, ЭНАПКХ',
        'Шрот соевый. 51%, ЭНАПКХ', 'Шрот соевый. 52%, ЭНАПКХ'
    ],
    'Шрот_рапсовый': [
        'Шрот рапсовый', 'Шрот Рапсовый', 'Шрот рапсовый. 38%', 'Шрот рапсовый. 38%, ЭНАПКХ',
        'Шрот рапсовый. 41%', 'Шрот рапсовый. 41%, ЭНАПКХ', 'Шрот рапсовый. 42%',
        'Шрот рапсовый. 44%, ЭНАПКХ'
    ],
    'Шрот_подсолнечный': [
        'Шрот подсолнечный', 'Шрот подсолнечный. 38%'
    ],
    'Жмых_рапсовый': [
        'Жмых рапсовый', 'Жмых рапсовый СП', 'Жмых рапсовый СП 39, ЭНАПКХ',
        'Жмых рапсовый. 35%', 'Жмых рапсовый. 36%', 'Жмых рапсовый. 36%, ЭНАПКХ',
        'Жмых рапсовый. 39%', 'Жмых рапсовый. 39%, ЭНАПКХ'
    ],
    'Жмых_льняной': [
        'Жмых льняной', 'Жмых льняной. 36%, ЭНАПКХ'
    ],
    'Побочные_свекловичные': [
        'Жом свекловичный сухой', 'Жом свекловичный сухой. гр',
        'Жом свекловичный сухой. гр, ЭНАПКХ v',
        'Жом свекловичный сухой. гранула, ЭНАПКХ',
        'Патока свекловичная', 'Патока свекловичная,\nсах',
        'Патока свекловичная, сах',
        'Патока свекловичная, сах.49%,\nЭНАПКХ v',
        'Патока свекловичная, сах.49%, ЭНАПКХ\nv',
        'Патока свекловичная, сахар', 'Меласса свекловичная',
        'Жом свекловичный сухой. гр, ЭНАПКХ v2', 'Меласса свекловичная',
        'Патока свекловичная,\nсах.49%, ЭНАПКХ v2', 'Патока свекловичная, сах.49%',
        'Патока свекловичная, сах.49%,\nЭНАПКХ v2', 'Патока свекловичная, сах.49%, ЭНАПКХ',
        'Патока свекловичная, сах.49%, ЭНАПКХ\nv2',
        'Патока свекловичная, сахар\n55%, ЭНАПКХ', 'Патока свекловичная, сахар 55%',
        'Патока свекловичная, сахар 55%, ЭНАПКХ'
    ],
    'Побочные_прочие': [
        'Соевая оболочка ЭНАПКХ', 'Дробина сухая', 'Дрожжи кормовые. СП', 'Пивные дрожжи сухие',
        'Дробина сухая. 27%, пивная', 'Дрожжи кормовые. СП 37%'
    ],
    'Комбикорма': [
        'Комбикорм', 'Комбикорм -', 'Комбикорм 10.3.3 (без пшен.)\n_Сентябрь',
        'Комбикорм 11.3.3 (без пшен.)\n_Сентябрь', 'Комбикорм №', 'КК', 'Комбиком №', 'ККД',
        'Кормосмесь', 'ккз аристово v', 'КК 10 Аристово 11.03.25', 'КК 10 От 03.01.24',
        'КК 10 От 17.09.24 ячмень', 'КК №10', 'КК10', 'ККД10 Аристово', 'Комбиком №10',
        'Комбиком №13', 'Комбикорм -\n11', 'Комбикорм 10.3.3 (без пшен.)',
        'Комбикорм 10.3.3 (без пшен.)\n_Сентябрь 09.09.24', 'Комбикорм 10.3.3_Январь',
        'Комбикорм 11.3.3 (без пшен.)', 'Комбикорм 11.3.3 (без пшен.)\n_Сентябрь 09.09.24',
        'Комбикорм 11.3.3 (без пшен.)\n_Сентябрь 12.03.25', 'Комбикорм 11.3.3_Январь',
        'Комбикорм №11', 'Комбикорм №11 ЭНАС', 'Кормосмесь 10', 'ккз аристово v4'
    ],
    'Премиксы': [
        'Премикс дойный А', 'Премикс дойный Б',
        'Премикс дойный Б. 07.23\nЭНАПКХ', 'Премикс Транзит Б', 'Премикс Транзит Б. 07.23',
        'Премикс дойный А. 12.23 ЭНАПКХ', 'Премикс дойный Б. 07.23',
        'Премикс дойный Б. 07.23 ЭНАПКХ', 'Премикс дойный Б. 09.22'
    ],
    'Жиры': [
        'Жир защищенный', 'Жир защищеный', 'Жир защищеный. 99%,\nфракц.,ЭНАПКХ',
        'Жир защищенный. 99%', 'Жир защищенный. 99%, гидроген. ЭНАПКХ',
        'Жир защищеный. 99%, фракц.,ЭНАПКХ'
    ],
    'Минеральные_добавки': [
        'Соль. ЭНАПКХ', 'Сода. ЭНАПКХ', 'Мел. ЭНАПКХ', 'Поташ. ЭНАПКХ',
        'Кальций пропионат', 'Добавка ЛЕД ЭНАПКХ'
    ]
}
MODEL_PATHS = {
    "Лауриновая": Path(r'C:\Users\Петр\papka bebrapka\ols_Лауриновая.pkl'),
    "Пальмитиновая": Path(r'C:\Users\Петр\papka bebrapka\ols_Пальмитиновая.pkl'),
    "Стеариновая": Path(r'C:\Users\Петр\papka bebrapka\ols_Стеариновая.pkl'),
    "Олеиновая": Path(r'C:\Users\Петр\papka bebrapka\ols_Олеиновая.pkl'),
    "Линолевая": Path(r'C:\Users\Петр\papka bebrapka\ols_Линолевая.pkl'),
    "Линоленовая": Path(r'C:\Users\Петр\papka bebrapka\ols_Линоленовая.pkl'),
}
TARGET_RANGES = {
    "Лауриновая": (2.0, 4.4),
    "Пальмитиновая": (21.0, 32.0),
    "Стеариновая": (8.0, 13.5),
    "Олеиновая": (20.0, 28.0),
    "Линолевая": (2.2, 5.0),
    "Линоленовая": (0.0, 1.5),
}
FEATURE_TO_COMPONENT_MAP = {
    'Sum_conc': ['Кукуруза', 'Зерновые_прочие', 'Комбикорма', 'Корнаж_ЗСК'],
    'Sum_rough': ['Сенаж', 'Сено', 'Солома'],
    'Sum_prot': ['Шрот_соевый', 'Шрот_рапсовый', 'Шрот_подсолнечный', 'Жмых_рапсовый', 'Жмых_льняной'],
    'share_kukur': ['Кукуруза'], 'share_korn': ['Корнаж_ЗСК'], 'share_zern': ['Зерновые_прочие'],
    'share_soloma': ['Солома'], 'share_prot_soy': ['Шрот_соевый'], 'share_prot_raps': ['Шрот_рапсовый'],
    'share_prot_lin': ['Жмых_льняной'], 'Силос': ['Силос'], 'Жиры': ['Жиры'],
    'Побочные_свекловичные': ['Побочные_свекловичные'], 'Побочные_прочие': ['Побочные_прочие']
}


# --- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ (без изменений) ---
@st.cache_data
def parse_pdf_report(uploaded_file):
    logs = []
    if uploaded_file is None: return {}, ["Файл не был загружен."]
    logs.append(f"Начало обработки файла: {uploaded_file.name}")
    tables = []
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            for page_num, page in enumerate(pdf.pages):
                extracted = page.extract_tables()
                if extracted: tables.extend(extracted)
        logs.append(f"Найдено таблиц в PDF: {len(tables)}")
    except Exception as e:
        logs.append(f"КРИТИЧЕСКАЯ ОШИБКА при чтении PDF: {e}")
        return {}, logs
    if not tables or len(tables[0]) < 2:
        logs.append("ОШИБКА: Не удалось найти подходящую таблицу с данными в PDF.")
        return {}, logs
    df = pd.DataFrame(tables[0][1:], columns=tables[0][0])
    df.columns = df.iloc[0]
    df = df[1:].reset_index(drop=True)
    logs.append(f"Обнаружены колонки: {list(df.columns)}")
    ingredient_col = next((c for c in df.columns if 'Ингредиент' in c), None)
    sv_kg_col = next((c for c in df.columns if 'СВ кг' in c), None)
    if not ingredient_col or not sv_kg_col:
        logs.append("ОШИБКА: В таблице не найдены обязательные колонки 'Ингредиенты' или 'СВ кг'.")
        return {}, logs
    logs.append(f"Колонка ингредиентов: '{ingredient_col}', колонка данных: '{sv_kg_col}'")
    df = df[[ingredient_col, sv_kg_col]].copy()
    df[sv_kg_col] = pd.to_numeric(df[sv_kg_col].str.replace(',', '.'), errors='coerce')
    df.dropna(inplace=True)
    logs.append(f"Найдено {len(df)} строк с числовыми данными.")
    df['clean_name'] = (df[ingredient_col].astype(str).str.split('/', n=1).str[0]
                        .str.replace(r'\s{2,}', ' ', regex=True)
                        .str.strip('.,\ '))
    aggregated_data = {key: 0.0 for key in FEED_MAP.keys()}
    logs.append("\n--- Построчное сопоставление (первые 15 строк) ---")
    for index, row in df.head(15).iterrows():
        match_found = False
        for category, names in FEED_MAP.items():
            if any(name.strip().lower() in row['clean_name'].lower() for name in names):
                aggregated_data[category] += row[sv_kg_col]
                logs.append(
                    f"Строка {index + 1}: '{row[ingredient_col]}' -> '{row['clean_name']}' -> ✓ '{category}' ({row[sv_kg_col]:.2f} кг)")
                match_found = True
                break
        if not match_found:
            logs.append(
                f"Строка {index + 1}: '{row[ingredient_col]}' -> '{row['clean_name']}' -> ✗ Категория не найдена")
    for index, row in df.iloc[15:].iterrows():
        for category, names in FEED_MAP.items():
            if any(name.strip().lower() in row['clean_name'].lower() for name in names):
                aggregated_data[category] += row[sv_kg_col]
                break
    logs.append("\n--- Итог агрегации ---")
    logs.append(str({k: round(v, 2) for k, v in aggregated_data.items() if v > 0}))
    if sum(aggregated_data.values()) < 0.01:
        logs.append("ПРЕДУПРЕЖДЕНИЕ: Сумма всех найденных компонентов равна нулю. Данные не загружены.")
        return {}, logs
    return aggregated_data, logs


@st.cache_data
def engineer_features(manual_inputs: dict):
    df = pd.DataFrame([manual_inputs])
    eps = 1e-9
    df['Sum_conc'] = df[['Кукуруза', 'Зерновые_прочие', 'Комбикорма', 'Корнаж_ЗСК']].sum(axis=1)
    df['Sum_rough'] = df[['Сенаж', 'Сено', 'Солома']].sum(axis=1)
    df['Sum_prot_ex'] = df[['Шрот_соевый', 'Шрот_рапсовый', 'Шрот_подсолнечный']].sum(axis=1)
    df['Sum_prot_pr'] = df[['Жмых_рапсовый', 'Жмых_льняной']].sum(axis=1)
    df['Sum_prot'] = df['Sum_prot_ex'] + df['Sum_prot_pr']
    df['share_kukur'] = np.where(df['Sum_conc'] > 0, df['Кукуруза'] / (df['Sum_conc'] + eps), 0.0)
    df['share_korn'] = np.where(df['Sum_conc'] > 0, df['Корнаж_ЗСК'] / (df['Sum_conc'] + eps), 0.0)
    df['share_zern'] = np.where(df['Sum_conc'] > 0, df['Зерновые_прочие'] / (df['Sum_conc'] + eps), 0.0)
    df['share_soloma'] = np.where(df['Sum_rough'] > 0, df['Солома'] / (df['Sum_rough'] + eps), 0.0)
    df['share_prot_soy'] = np.where(df['Sum_prot'] > 0, df['Шрот_соевый'] / (df['Sum_prot'] + eps), 0.0)
    df['share_prot_raps'] = np.where(df['Sum_prot'] > 0, df['Шрот_рапсовый'] / (df['Sum_prot'] + eps), 0.0)
    df['share_prot_lin'] = np.where(df['Sum_prot'] > 0, df['Жмых_льняной'] / (df['Sum_prot'] + eps), 0.0)
    return df


@st.cache_resource
def load_models_and_get_influencers(paths: dict):
    models = {}
    all_model_features = set()
    for acid_name, path in paths.items():
        if not path.exists():
            st.error(f"Файл модели не найден: {path}")
            return None, [], []
        with open(path, "rb") as f:
            model_data = pickle.load(f)
            models[acid_name] = model_data
            all_model_features.update(model_data['features'])
    strong_components = set()
    for feature in all_model_features:
        if feature in FEATURE_TO_COMPONENT_MAP:
            strong_components.update(FEATURE_TO_COMPONENT_MAP[feature])
    all_components = list(FEED_MAP.keys())
    strong_list = sorted([c for c in all_components if c in strong_components])
    weak_list = sorted([c for c in all_components if c not in strong_components])
    return models, strong_list, weak_list

# === ИНТЕРПРЕТАЦИЯ И МЕРЫ ===

def _predict_acid(acid_name: str, inputs_dict: dict) -> float:
    pack = models[acid_name]
    X = engineer_features(inputs_dict)[pack['features']]
    if pack.get('add_constant', True):
        X = sm.add_constant(X, has_constant='add')
    sf = pack['model'].get_prediction(X).summary_frame(alpha=0.05)
    return float(sf['mean'].iloc[0])

def local_sensitivities(acid_name: str, base_inputs: dict, step: float = 0.5) -> dict:
    base = _predict_acid(acid_name, base_inputs)
    sens = {}
    for comp in FEED_MAP.keys():
        x2 = dict(base_inputs)
        x2[comp] = max(0.0, x2[comp] + step)
        p2 = _predict_acid(acid_name, x2)
        sens[comp] = (p2 - base) / step  # п.п. на +1 кг
    return sens  # без сортировки — матрица сама отсортирует строки

def sensitivities_matrix(base_inputs: dict, step: float = 0.5) -> pd.DataFrame:
    acids = list(models.keys())
    comps = list(FEED_MAP.keys())
    data = {acid: {} for acid in acids}
    for acid in acids:
        s = local_sensitivities(acid, base_inputs, step=step)
        for comp in comps:
            data[acid][comp] = s.get(comp, 0.0)
    return pd.DataFrame(data, index=comps)

def get_sens_matrix_cached(base_inputs: dict, step: float = 0.5) -> pd.DataFrame:
    df = st.session_state.get('sens_matrix_last', None)
    if isinstance(df, pd.DataFrame) and not df.empty:
        return df
    df = sensitivities_matrix(base_inputs, step=step)
    st.session_state['sens_matrix_last'] = df
    return df

def _total_sv(inputs: dict) -> float:
    return float(sum(inputs.values()))

def build_measures(preds: dict,
                   sens_df: pd.DataFrame,
                   base_inputs: dict,
                   sv_bounds: tuple[float, float] = (15.0, 30.0),
                   sv_margin: float = 0.5,
                   top_k: int = 3,
                   tol: float = 1e-6) -> dict:
    """
    Возвращает:
      measures: {acid: {"status":..., "inc":[...], "dec":[...]}}
      heavy: {"inc": set(...), "dec": set(...)} — компоненты, актуальные ≥ для 2 кислот

    Направление:
      - mean < lo - tol  → тянуть ВВЕРХ
      - mean > hi + tol  → тянуть ВНИЗ
      - lo - tol ≤ mean ≤ hi + tol → тянуть к ЦЕНТРУ ( (lo+hi)/2 )
    Ограничения:
      - не предлагать уменьшать компонент, если он уже 0;
      - если ΣСВ ≥ sv_max - sv_margin → не предлагать увеличения;
        если ΣСВ ≤ sv_min + sv_margin → не предлагать уменьшения.
    """

    sv = _total_sv(base_inputs)
    sv_min, sv_max = sv_bounds
    allow_inc_global = sv < (sv_max - sv_margin)
    allow_dec_global = sv > (sv_min + sv_margin)

    measures = {}
    all_inc, all_dec = [], []

    for acid, d in preds.items():
        if acid not in sens_df.columns:
            continue
        status = d['status']
        mean = d['mean']
        lo, hi = d['target_min'], d['target_max']
        if '🟢' in status:
            continue  # в норме — мер не даём

        col = sens_df[acid]
        center = 0.5 * (lo + hi)

        # определяем нужное направление
        if mean < lo - tol:
            need_up = True
        elif mean > hi + tol:
            need_up = False
        else:
            # внутри коридора (в т.ч. 🟡 из-за ДИ): тянем к центру
            need_up = mean < center

        # кандидаты по знаку чувствительности
        if need_up:
            # чтобы поднять кислоту: УВЕЛИЧИВАТЬ sens>0; УМЕНЬШАТЬ sens<0
            inc_candidates = col[col > 0].sort_values(ascending=False)
            dec_candidates = col[col < 0].abs().sort_values(ascending=False)
        else:
            # чтобы опустить кислоту: УМЕНЬШАТЬ sens>0; УВЕЛИЧИВАТЬ sens<0
            dec_candidates = col[col > 0].sort_values(ascending=False)
            inc_candidates = col[col < 0].abs().sort_values(ascending=False)

        # применяем глобальные ограничения ΣСВ и нулевые компоненты
        inc_list = []
        if allow_inc_global:
            for comp in inc_candidates.index:
                inc_list.append(comp)
                if len(inc_list) >= top_k: break

        dec_list = []
        if allow_dec_global:
            for comp in dec_candidates.index:
                if base_inputs.get(comp, 0.0) > 0.0:  # нельзя уменьшать ниже нуля
                    dec_list.append(comp)
                if len(dec_list) >= top_k: break

        measures[acid] = {"status": status, "inc": inc_list, "dec": dec_list}
        all_inc.extend(inc_list)
        all_dec.extend(dec_list)

    # компоненты, попавшие в меры ≥ для 2 кислот — выделяем жирным
    from collections import Counter
    inc_count = Counter(all_inc)
    dec_count = Counter(all_dec)
    heavy = {
        "inc": {c for c, n in inc_count.items() if n >= 2},
        "dec": {c for c, n in dec_count.items() if n >= 2},
    }
    return measures, heavy

def get_current_inputs():
    # Всегда отдаёт словарь входов; даже если анализа ещё не было
    if 'current_inputs' in st.session_state:
        return st.session_state.current_inputs
    return {k: st.session_state.get(f"inp_{k}", 0.0) for k in FEED_MAP.keys()}


# --- ИНТЕРФЕЙС ПРИЛОЖЕНИЯ ---

def run_analysis():
    current_inputs = {key: st.session_state[f"inp_{key}"] for key in FEED_MAP.keys()}

    # Сохраняем текущие инпуты и общую сумму в состояние для отображения
    st.session_state.current_inputs = current_inputs
    st.session_state.total_sv = sum(current_inputs.values())

    features_df = engineer_features(current_inputs)

    predictions = {}
    any_deviations = False
    for acid_name, model_data in models.items():
        model = model_data['model']
        model_features = model_data['features']
        if not all(f in features_df.columns for f in model_features):
            st.error(f"Ошибка: для модели '{acid_name}' не хватает признаков.")
            continue
        X = features_df[model_features]
        if model_data.get('add_constant', True): X = sm.add_constant(X, has_constant='add')
        pred_summary = model.get_prediction(X).summary_frame(alpha=0.05)
        mean, ci_low, ci_up = \
            pred_summary['mean'].iloc[0], \
            pred_summary['mean_ci_lower'].iloc[0], \
            pred_summary['mean_ci_upper'].iloc[0]
        mean = max(0.0, float(mean))
        ci_low = max(0.0, float(ci_low))
        ci_up = max(0.0, float(ci_up))
        t_min, t_max = TARGET_RANGES[acid_name]
        color = '#2ca02c'
        status = "🟢 Норма"
        if mean < t_min or mean > t_max:
            status = "🔴 Вне нормы"
            any_deviations = True
            color = '#d62728'
        elif ci_low < t_min or ci_up > t_max:
            status = "🟡 Риск отклонения"
            any_deviations = True
            color = '#ff7f0e'
        predictions[acid_name] = {"mean": mean, "ci_lower": ci_low, "ci_upper": ci_up,
                                  "target": f"{t_min:.1f}%–{t_max:.1f}%", "target_min": t_min, "target_max": t_max,
                                  "status": status, "color": color}

    st.session_state.predictions = predictions
    st.session_state.any_deviations = any_deviations
    st.session_state.analysis_run = True


def reset_app_state():
    for key in FEED_MAP.keys():
        st.session_state[f"inp_{key}"] = 0.0
    st.session_state.analysis_run = False
    st.session_state.logs = []
    st.session_state.last_uploaded_filename = None
    if 'current_inputs' in st.session_state:
        del st.session_state['current_inputs']


# Инициализация состояния
if 'analysis_run' not in st.session_state:
    st.session_state.analysis_run = False
if 'logs' not in st.session_state:
    st.session_state.logs = []
if 'current_inputs' not in st.session_state:
    st.session_state.current_inputs = {k: st.session_state.get(f"inp_{k}", 0.0) for k in FEED_MAP.keys()}
if 'total_sv' not in st.session_state:
    st.session_state.total_sv = float(sum(st.session_state.current_inputs.values()))
models, strong_influencers, weak_influencers = load_models_and_get_influencers(MODEL_PATHS)
if models is None: st.stop()
for key in FEED_MAP.keys():
    if f"inp_{key}" not in st.session_state: st.session_state[f"inp_{key}"] = 0.0

with st.sidebar:
    st.header("⚙️ Параметры рациона")
    uploaded_file = st.file_uploader("Загрузите PDF-отчет", type="pdf")

    if uploaded_file is not None:
        if st.session_state.get('last_uploaded_filename') != uploaded_file.name:
            st.session_state.last_uploaded_filename = uploaded_file.name
            parsed_data, logs = parse_pdf_report(uploaded_file)
            st.session_state.logs = logs
            if parsed_data:
                for key, value in parsed_data.items():
                    st.session_state[f"inp_{key}"] = value
                run_analysis()
                st.rerun()

    st.subheader("Состав рациона (кг СВ):")
    st.button("Сбросить изменения", use_container_width=True, on_click=reset_app_state)
    st.markdown("**Сильно влияющие компоненты**")
    for key in strong_influencers:
        st.number_input(key, min_value=0.0, step=0.1, format="%.2f", key=f"inp_{key}", on_change=run_analysis)

    st.markdown("**Прочие компоненты**")
    for key in weak_influencers:
        st.number_input(key, min_value=0.0, step=0.1, format="%.2f", key=f"inp_{key}", on_change=run_analysis)

# --- ОТОБРАЖЕНИЕ РЕЗУЛЬТАТОВ ---
# st.title("🐄 Аналитический дашборд")

if st.session_state.logs:
    with st.expander("📝 Логи разбора PDF-файла", expanded=False):
        st.code("\n".join(st.session_state.logs), language='text')

if not st.session_state.analysis_run:
    st.info("Введите данные в панели слева или загрузите PDF-отчет для начала анализа.")
else:
    # --- НОВЫЙ БЛОК: РАЦИОН ---
    st.subheader("Рацион")
    st.markdown("""
    <style>
    /* Заголовки */
    h1, h2, h3 { line-height: 1.2; }
    h2 { font-size: 1.6rem !important; }   /* st.subheader */
    h3 { font-size: 1.3rem !important; }

    /* Карточки metric */
    [data-testid="stMetricValue"] { font-size: 2rem; }
    [data-testid="stMetricLabel"] { font-size: 1.1rem; }
    [data-testid="stMetricDelta"] { font-size: 1rem; }

    /* Текст в info/success/warning */
    .block-container p, .stAlert { font-size: 1rem; }

    /* Немного крупнее подписи под графиками и в тултипах Plotly
       (основное для тултипов уже задано в hoverlabel) */
    </style>
    """, unsafe_allow_html=True)
    FONT = 16
    col1, col2 = st.columns([2, 1])

    with col1:
        current_inputs = get_current_inputs()
        pie_data = {k: v for k, v in current_inputs.items() if v > 0}
        if pie_data:
            pie_fig = go.Figure(data=[go.Pie(
                labels=list(pie_data.keys()),
                values=list(pie_data.values()),
                hole=.3,
                textinfo="label+percent"
            )])
            pie_fig.update_traces(textfont_size=FONT)  # подписи сегментов
            pie_fig.update_layout(
                height=400,
                margin=dict(l=30, r=30, t=10, b=30),
                font=dict(size=FONT + 2),  # общий шрифт внутри фигуры
                legend=dict(font=dict(size=FONT)),  # легенда (если понадобится)
                showlegend=False,
                hoverlabel=dict(font=dict(size=FONT))  # всплывающие подсказки
            )
            st.plotly_chart(pie_fig, use_container_width=True)
        else:
            st.info("Нет данных для отображения структуры рациона.")

    with col2:
        total_sv = st.session_state.total_sv
        st.metric(label="Сумма СВ кг/день", value=f"{total_sv:.2f}")

        if total_sv < 15:
            st.error("🔴 Общее количество СВ ниже нормы (< 15 кг). Рекомендуется увеличить объем рациона.")
        elif total_sv > 30:
            st.error("🔴 Общее количество СВ выше нормы (> 30 кг). Рекомендуется снизить объем рациона.")
        else:
            st.success("🟢 Общее количество СВ в норме.")

    st.markdown("---")
    preds = st.session_state.predictions
    st.subheader("Прогноз по жирным кислотам")
    cols = st.columns(len(preds))
    # ... (отображение метрик и рекомендаций без изменений)
    for i, (acid_name, data) in enumerate(preds.items()):
        cols[i].metric(
            label=f"{acid_name} ({data['status']})", value=f"{data['mean']:.2f}%",
            help=f"Цель: {data['target']}. 95% ДИ: {data['ci_lower']:.2f}%–{data['ci_upper']:.2f}%"
        )

    # === РЕКОМЕНДАЦИИ: форматированный Markdown со строками-переносами ===
    if st.session_state.any_deviations:
        df_sens_use = get_sens_matrix_cached(get_current_inputs(), step=0.5)

        measures, heavy = build_measures(
            preds=preds,
            sens_df=df_sens_use,
            base_inputs=get_current_inputs(),
            sv_bounds=(15.0, 30.0),
            sv_margin=0.5,
            top_k=3
        )

        def fmt_list(lst, heavy_set):
            if not lst:
                return "—"
            return ", ".join([f"**{c}**" if c in heavy_set else c for c in lst])

        md_lines = []
        for acid, payload in measures.items():
            lo, hi = TARGET_RANGES[acid]
            inc_str = fmt_list(payload["inc"], heavy["inc"])
            dec_str = fmt_list(payload["dec"], heavy["dec"])
            # ВСЕГДА «увеличить» сначала, затем «уменьшить»
            md_lines.append(
                f"- **{acid}**: {payload['status']}. Возможные меры:  \n"
                f"  - Увеличить: {inc_str}  \n"
                f"  - Уменьшить: {dec_str}"
            )

        # желтый бокс + переносы строк за счёт двойных пробелов + \n
        st.warning("\n".join(md_lines))

    # --- ИСПРАВЛЕННЫЙ БЛОК ВИЗУАЛИЗАЦИИ ---
    fig = go.Figure()

    rows = []
    for acid_name, p in preds.items():
        center = 0.5 * (p['target_min'] + p['target_max'])
        span = max(p['target_max'] - p['target_min'], 1e-9)  # защита от нулевой ширины
        delta_norm = abs(p['mean'] - center) / span  # нормализованное отклонение
        rows.append((acid_name, delta_norm))

    # чем больше |норм. отклонение|, тем выше в списке
    rows.sort(key=lambda x: x[1], reverse=True)

    acid_names = [name for name, _ in rows]
    mean_values = [preds[name]['mean'] for name in acid_names]

    # Сначала добавляем цветные бары
    for acid_name in reversed(acid_names):
        p = preds[acid_name]
        fig.add_trace(go.Bar(
            y=[acid_name],
            x=[p['mean']],
            name=acid_name,
            showlegend=False,
            orientation='h',
            marker_color=p['color'],
            error_x=dict(type='data', symmetric=False, array=[p['ci_upper'] - p['mean']],
                         arrayminus=[p['mean'] - p['ci_lower']], thickness=3, width=6),
            hovertemplate=(
                f"{acid_name}: {p['mean']:.2f}%<br>"
                f"95% ДИ: {p['ci_lower']:.2f}% – {p['ci_upper']:.2f}%<extra></extra>"
            )
            # text=f"{p['mean']:.2f}%",
            # textposition='auto'
        ))

    # Потом добавляем зоны, чтобы они были сверху
    for acid_name in reversed(acid_names):
        p = preds[acid_name]
        fig.add_shape(
            type="rect", xref="x", yref="y",
            x0=p['target_min'], y0=acid_name,
            x1=p['target_max'], y1=acid_name,
            y0shift=-0.45, y1shift=0.45,
            fillcolor="green", opacity=0.4,  # Полупрозрачный цвет
            layer="above",  # Отображаем поверх баров
            line_width=0
        )

    # В самом конце добавляем жирные точки прогноза
    fig.add_trace(go.Scatter(
        y=acid_names,
        x=mean_values,
        mode='markers',
        marker=dict(color='black', size=8, symbol='diamond'),
        showlegend=False,
        hovertemplate=()
    ))

    # --- Легенда (прокси-трейсы) ---
    # Прогноз (бриллиант)
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(symbol="diamond", size=10, color="black"),
        name="Прогноз",
        hoverinfo="skip", showlegend=True
    ))
    # 95% ДИ (линия)
    fig.add_trace(go.Scatter(
        x=[None, None], y=[None, None], mode="lines",
        line=dict(width=3, color="black"),
        name="Доверительный интервал 95%",
        hoverinfo="skip", showlegend=True
    ))
    # Целевой диапазон (зелёная зона)
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(symbol="square", size=12,
                    color="rgba(0,128,0,0.35)",
                    line=dict(color="green", width=1)),
        name="Целевой диапазон",
        hoverinfo="skip", showlegend=True
    ))

    fig.update_layout(
        margin=dict(l=50, r=50, t=10, b=10),
        font=dict(size=FONT + 2),
        hoverlabel=dict(font=dict(size=FONT)),
        title_text="",
        barmode="stack",
        yaxis_title="Жирная кислота",
        xaxis_title="Содержание, %",
        showlegend=True,
        legend=dict(
            font=dict(size=FONT),  # размер шрифта легенды
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="left", x=0,
            bgcolor="rgba(255,255,255,0.6)"
        ),
        height=600,
    )

    # шрифты осей: заголовки и деления
    fig.update_xaxes(title_font=dict(size=FONT + 2), tickfont=dict(size=FONT))
    fig.update_yaxes(title_font=dict(size=FONT + 2), tickfont=dict(size=FONT))

    # Сетка
    fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.12)", zeroline=False)
    fig.update_yaxes(showgrid=False)

    st.plotly_chart(fig, use_container_width=True)

# === ИНТЕРПРЕТАЦИЯ (автоматически) ===
st.markdown("---")
with st.expander("🔍 Интерпретация (Δ п.п. на +1 кг компонента)", expanded=False):
    step_val = 0.5  # можно вынести в слайдер выше по странице, если нужно управлять
    base_inputs = get_current_inputs()
    df_sens = sensitivities_matrix(base_inputs, step=step_val)
    st.session_state['sens_matrix_last'] = df_sens  # кэш на сессию
    st.dataframe(df_sens.round(3), use_container_width=True)
    st.caption("Положительное значение → при увеличении компонента кислота растёт; "
               "отрицательное → падает. Единицы: процентные пункты на +1 кг компонента.")
