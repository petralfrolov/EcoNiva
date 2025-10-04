import plotly.graph_objects as go


def style_css(font_size=16) -> str:
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


def build_acids_bar_figure(preds: dict, font: int = 16) -> go.Figure:
    fig = go.Figure()

    # Карта для определения приоритета статуса (0 - самый высокий)
    status_priority = {"🔴": 0, "🟡": 1, "🟢": 2}

    rows_to_sort = []
    for acid_name, p in preds.items():
        # Расчет отклонения от центра (как и раньше)
        center = 0.5 * (p['target_min'] + p['target_max'])
        span = max(p['target_max'] - p['target_min'], 1e-9)
        delta_norm = abs(p['mean'] - center) / span

        # Получаем приоритет из статуса
        status_char = p['status'][0]
        priority = status_priority.get(status_char, 2)  # По умолчанию - низший приоритет

        rows_to_sort.append({
            "name": acid_name,
            "priority": priority,
            "deviation": delta_norm
        })

    # Сортируем сначала по приоритету (0, 1, 2),
    # а затем внутри каждой группы - по убыванию отклонения
    rows_to_sort.sort(key=lambda row: (row['priority'], -row['deviation']))

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
