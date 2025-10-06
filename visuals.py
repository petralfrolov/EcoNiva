import plotly.graph_objects as go


def style_css(font_size=16) -> str:
    """
    Генерирует строку с CSS-стилями для кастомизации интерфейса Streamlit.

    Args:
        font_size (int): Базовый размер шрифта для элементов.

    Returns:
        str: Строка, содержащая тег <style> с CSS-правилами.
    """
    return f"""
    <style>
    h1, h2, h3 {{ line-height: 1.2; }}
    h2 {{ font-size: 1.6rem !important; }}
    h3 {{ font-size: 1.3rem !important; }}

    [data-testid="stMetricValue"] {{ font-size: 2rem; }}
    [data-testid="stMetricLabel"] {{ font-size: 1.1rem; }}
    [data-testid="stMetricDelta"] {{ font-size: 1rem; }}

    .block-container p, .stAlert {{ font-size: 1rem; }}
    </style>
    """


def build_pie_figure(pie_data: dict, font: int = 16) -> go.Figure:
    """
    Создает круговую диаграмму (Pie Chart) для визуализации состава рациона.

    Args:
        pie_data (dict): Словарь с данными, где ключи - названия,
                         а значения - числовые величины.
        font (int): Размер шрифта для текста на диаграмме.

    Returns:
        go.Figure: Объект фигуры Plotly.
    """
    fig = go.Figure(data=[go.Pie(
        labels=list(pie_data.keys()),
        values=list(pie_data.values()),
        hole=.3,
        textinfo="label+percent"
    )])
    fig.update_traces(textfont_size=font)
    fig.update_layout(
        height=400,
        margin=dict(l=30, r=30, t=10, b=30),
        font=dict(size=font + 2),
        legend=dict(font=dict(size=font)),
        showlegend=False,
        hoverlabel=dict(font=dict(size=font))
    )
    return fig


def build_treemap_figure(data: dict, font: int = 16) -> go.Figure:
    """
    Создает древовидную диаграмму (Treemap) для визуализации структуры рациона.

    Этот тип диаграммы эффективно показывает долю каждого компонента
    в общей сумме.

    Args:
        data (dict): Словарь с данными для визуализации.
        font (int): Базовый размер шрифта.

    Returns:
        go.Figure: Объект фигуры Plotly.
    """
    labels = list(data.keys())
    values = list(data.values())

    fig = go.Figure(go.Treemap(
        labels=labels,
        values=values,
        parents=[""] * len(labels),
        textinfo="label+value+percent root",
        marker_colorscale='Greens',
    ))

    fig.update_layout(
        height=400,
        margin=dict(l=10, r=10, t=10, b=10),
        font=dict(size=font),
        hoverlabel=dict(font=dict(size=font))
    )
    return fig


def build_acids_bar_figure(preds: dict, font: int = 16) -> go.Figure:
    """
    Создает сложную столбчатую диаграмму для визуализации прогноза по кислотам.

    Отображает среднее прогнозное значение, 95% доверительный интервал
    и целевой диапазон для каждой жирной кислоты. Кислоты сортируются
    по степени отклонения от нормы.

    Args:
        preds (dict): Словарь с результатами прогнозирования от функции
                      `predict_all_acids`.
        font (int): Базовый размер шрифта.

    Returns:
        go.Figure: Объект фигуры Plotly.
    """
    fig = go.Figure()

    status_priority = {"🔴": 0, "🟡": 1, "🟢": 2}
    rows_to_sort = []
    for acid_name, p in preds.items():
        status_char = p['status'][0]
        priority = status_priority.get(status_char, 2)
        rows_to_sort.append({
            "name": acid_name,
            "priority": priority,
            "value": p['mean']
        })

    rows_to_sort.sort(key=lambda row: (row['priority'], -row['value']))

    # Извлекаем отсортированный список названий
    acid_names = [row['name'] for row in rows_to_sort]
    mean_values = [preds[name]['mean'] for name in acid_names]

    for acid_name in reversed(acid_names):
        p = preds[acid_name]
        fig.add_trace(go.Bar(
            y=[acid_name],
            x=[p['mean']],
            name=acid_name,
            showlegend=False,
            orientation='h',
            marker_color=p['color'],
            error_x=dict(
                type='data',
                symmetric=False,
                array=[p['ci_upper'] - p['mean']],
                arrayminus=[p['mean'] - p['ci_lower']],
                thickness=3, width=6
            ),
            hovertemplate=(
                f"{acid_name}: {p['mean']:.2f}%<br>"
                f"95% ДИ: {p['ci_lower']:.2f}% – {p['ci_upper']:.2f}%<extra></extra>"
            )
        ))

    for acid_name in reversed(acid_names):
        p = preds[acid_name]
        fig.add_shape(
            type="rect", xref="x", yref="y",
            x0=p['target_min'], y0=acid_name,
            x1=p['target_max'], y1=acid_name,
            y0shift=-0.45, y1shift=0.45,
            fillcolor="green", opacity=0.4,
            layer="above",
            line_width=0
        )

    fig.add_trace(go.Scatter(
        y=acid_names,
        x=mean_values,
        mode='markers',
        marker=dict(color='black', size=8, symbol='diamond'),
        showlegend=False,
        hovertemplate=()
    ))

    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(symbol="diamond", size=10, color="black"),
        name="Прогноз",
        hoverinfo="skip", showlegend=True
    ))
    fig.add_trace(go.Scatter(
        x=[None, None], y=[None, None], mode="lines",
        line=dict(width=3, color="black"),
        name="Доверительный интервал 95%",
        hoverinfo="skip", showlegend=True
    ))
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
        font=dict(size=font + 2),
        hoverlabel=dict(font=dict(size=font)),
        title_text="",
        barmode="stack",
        yaxis_title="Жирная кислота",
        xaxis_title="Содержание, %",
        showlegend=True,
        legend=dict(
            font=dict(size=font),
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="left", x=0,
            bgcolor="rgba(255,255,255,0.6)"
        ),
        height=600,
    )
    fig.update_xaxes(title_font=dict(size=font + 2), tickfont=dict(size=font),
                     showgrid=True, gridcolor="rgba(0,0,0,0.12)", zeroline=False)
    fig.update_yaxes(title_font=dict(size=font + 2), tickfont=dict(size=font),
                     showgrid=False)
    return fig
