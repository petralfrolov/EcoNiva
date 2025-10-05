import streamlit as st
import numpy as np

from config import (
    PAGE_TITLE, FONT, SV_BOUNDS, SV_MARGIN,
    FEED_MAP, MODEL_PATHS, TARGET_RANGES
)
from io_parsing import parse_any_report
from models_math import (
    load_models_and_get_influencers,
    predict_all_acids, sensitivities_matrix, build_measures
)
from visuals import style_css, build_pie_figure, build_acids_bar_figure, build_treemap_figure

# --- Конфигурация страницы ---
st.set_page_config(layout="wide", page_title=PAGE_TITLE)


# --- Хелперы состояния ---
def get_current_inputs():
    if 'current_inputs' in st.session_state:
        return st.session_state.current_inputs
    return {k: st.session_state.get(f"inp_{k}", 0.0) for k in FEED_MAP.keys()}


def reset_app_state():
    for key in FEED_MAP.keys():
        st.session_state[f"inp_{key}"] = 0.0
    st.session_state.analysis_run = False
    st.session_state.logs = []
    st.session_state.last_uploaded_filename = None
    if 'current_inputs' in st.session_state:
        del st.session_state['current_inputs']


def run_analysis():
    current_inputs = {key: st.session_state[f"inp_{key}"] for key in FEED_MAP.keys()}
    st.session_state.current_inputs = current_inputs
    st.session_state.total_sv = sum(current_inputs.values())
    preds, any_deviations = predict_all_acids(models, current_inputs, TARGET_RANGES)
    st.session_state.predictions = preds
    st.session_state.any_deviations = any_deviations
    st.session_state.analysis_run = True


# --- Инициализация состояния ---
if 'analysis_run' not in st.session_state:
    st.session_state.analysis_run = False
if 'logs' not in st.session_state:
    st.session_state.logs = []
if 'current_inputs' not in st.session_state:
    st.session_state.current_inputs = {k: st.session_state.get(f"inp_{k}", 0.0) for k in FEED_MAP.keys()}
if 'total_sv' not in st.session_state:
    st.session_state.total_sv = float(sum(st.session_state.current_inputs.values()))
for key in FEED_MAP.keys():
    if f"inp_{key}" not in st.session_state:
        st.session_state[f"inp_{key}"] = 0.0

# --- Загрузка моделей ---
models, all_components = load_models_and_get_influencers(MODEL_PATHS)
if models is None:
    st.stop()

def render_group(keys: list[str]):
    # сортируем: сначала активные, внутри — по убыванию значения
    items = [(k, float(st.session_state.get(f"inp_{k}", 0.0))) for k in keys]
    # активные вверх, внутри — по алфавиту
    items.sort(key=lambda kv: (kv[1] == 0.0, kv[0].lower()))

    for k, v in items:
        icon = "⚫" if v > 0 else "⚪"
        c1, c2 = st.columns([6, 1])
        with c1:
            st.number_input(
                f"{icon} {k}",
                min_value=0.0,
                step=0.1,
                format="%.2f",
                key=f"inp_{k}",
                on_change=run_analysis
            )
        with c2:
            st.checkbox(
                "🔒", key=f"lock_{k}",
                help="Зафиксировать компонент (автоподбор не изменит этот элемент)."
            )
    st.caption("🔒 — Зафиксировать компонент (автоподбор не изменит этот элемент).")

def unlock_all():
    for k in FEED_MAP.keys():
        st.session_state[f"lock_{k}"] = False

def run_optimizer():
    from models_math import optimize_ration
    base_inputs = get_current_inputs()
    if st.session_state.get("lock_sv_total_cb", False):
        lock_sv = True
    else:
        lock_sv = False
    locks = {k for k in FEED_MAP.keys() if st.session_state.get(f"lock_{k}", False)}
    new_inputs, report = optimize_ration(
        models=models,
        base_inputs=base_inputs,
        target_ranges=TARGET_RANGES,
        locks=locks,
        sv_bounds=SV_BOUNDS,
        lock_sv_total=lock_sv # <-- И здесь: передаем значение в оптимизатор
    )
    # применяем результат
    for k, v in new_inputs.items():
        st.session_state[f"inp_{k}"] = float(v)
    st.session_state["optimizer_report"] = report
    run_analysis()

# --- Сайдбар ---
with st.sidebar:
    st.header("Параметры рациона")
    uploaded_file = st.file_uploader("Загрузите рацион (PDF или Excel)", type=["pdf", "xlsx", "xls"])

    if uploaded_file is not None:
        if st.session_state.get('last_uploaded_filename') != uploaded_file.name:
            st.session_state.last_uploaded_filename = uploaded_file.name
            parsed_data, logs = parse_any_report(uploaded_file)  # <= вот здесь
            st.session_state.logs = logs
            if parsed_data:
                # Проходим по всем известным компонентам
                for component in all_components:
                    # Если компонент отсутствует в отчете или его значение близко к нулю
                    if parsed_data.get(component, 0.0) < 0.001:
                        # Ставим на него замок
                        st.session_state[f"lock_{component}"] = True
                    else:
                        # В противном случае — снимаем замок
                        st.session_state[f"lock_{component}"] = False
                for key, value in parsed_data.items():
                    st.session_state[f"inp_{key}"] = value
                st.session_state["lock_sv_total_cb"] = False
                run_analysis()
                st.rerun()

    st.subheader("Состав рациона (кг СВ):")
    st.button("Сбросить изменения", use_container_width=True, on_click=reset_app_state)
    st.button("Разблокировать всё", use_container_width=True, on_click=unlock_all)
    st.button("Автоподбор", use_container_width=True, on_click=run_optimizer, args=())
    lock_sv_total = st.checkbox(
        "Зафиксировать общее СВ",
        key="lock_sv_total_cb",
        help="Если опция включена, автоподбор будет только перераспределять компоненты, сохраняя их общую сумму."
    )
    render_group(all_components)

# --- Основной экран ---
if st.session_state.logs:
    with st.expander("📝 Логи разбора файла", expanded=False):
        st.code("\n".join(st.session_state.logs), language='text')

if not st.session_state.analysis_run:
    st.info("Введите данные в панели слева или загрузите pdf/xlsx отчёт для начала анализа.")
else:
    st.subheader("Рацион")
    st.markdown(style_css(FONT), unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        current_inputs = get_current_inputs()
        pie_data = {k: v for k, v in current_inputs.items() if v > 0}
        if pie_data:
            st.plotly_chart(build_treemap_figure(pie_data, FONT), use_container_width=True)
        else:
            st.info("Нет данных для отображения структуры рациона.")

    with col2:
        total_sv = st.session_state.total_sv
        st.metric(label="Сумма СВ кг/день", value=f"{total_sv:.2f}")
        if total_sv < SV_BOUNDS[0]:
            st.error(
                f"🔴 Общее количество СВ ниже нормы (< {SV_BOUNDS[0]:.0f} кг). Рекомендуется увеличить объем рациона.")
        elif total_sv > SV_BOUNDS[1]:
            st.error(
                f"🔴 Общее количество СВ выше нормы (> {SV_BOUNDS[1]:.0f} кг). Рекомендуется снизить объем рациона.")
        else:
            st.success("🟢 Общее количество СВ в норме.")

    st.markdown("---")
    preds = st.session_state.predictions
    st.subheader("Прогноз по жирным кислотам")
    cols = st.columns(len(preds))
    for i, (acid_name, data) in enumerate(preds.items()):
        cols[i].metric(
            label=f"{acid_name} ({data['status']})",
            value=f"{data['mean']:.2f}%",
            help=f"Цель: {data['target']}. 95% ДИ: {data['ci_lower']:.2f}%–{data['ci_upper']:.2f}%"
        )

    # --- Рекомендации ---
    if st.session_state.any_deviations:
        base_inputs = get_current_inputs()
        df_sens_use = sensitivities_matrix(models, base_inputs, step=0.5)

        measures, heavy = build_measures(
            preds=preds,
            sens_df=df_sens_use,
            base_inputs=base_inputs,
            sv_bounds=SV_BOUNDS,
            sv_margin=SV_MARGIN,
            top_k=3
        )


        def fmt_list(lst, heavy_set):
            if not lst:
                return "—"
            return ", ".join([f"**{c}**" if c in heavy_set else c for c in lst])


        md_lines = []
        for acid, payload in measures.items():
            inc_str = fmt_list(payload["inc"], heavy["inc"])
            dec_str = fmt_list(payload["dec"], heavy["dec"])
            md_lines.append(
                f"- **{acid}**: {payload['status']}. Возможные меры:  \n"
                f"  - Увеличить: {inc_str}  \n"
                f"  - Уменьшить: {dec_str}"
            )
        st.warning("\n".join(md_lines))

    # --- Большой график ---
    st.plotly_chart(build_acids_bar_figure(preds, FONT), use_container_width=True)

    # --- Интерпретация ---
    st.markdown("---")
    st.subheader("Интерпретация (изменение кислоты в % на +1 кг компонента)")
    step_val = 0.5
    base_inputs = get_current_inputs()
    df_sens = sensitivities_matrix(models, base_inputs, step=step_val)
    st.session_state['sens_matrix_last'] = df_sens

    # --- ЦВЕТА: красный для +, синий для - (белый около нуля) ---
    M = float(np.nanmax(np.abs(df_sens.values))) or 1.0  # защита от деления на 0
    styled = (df_sens
              .style
              .background_gradient(cmap="bwr", vmin=-M, vmax=M, axis=None)
              .format("{:.3f}"))

    st.dataframe(styled, use_container_width=True)
    st.caption(
        "Цвет: красный — положительное влияние (рост кислоты при +1 кг), синий — отрицательное.")
