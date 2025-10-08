"""
CeciAI - Dashboard de Monitoramento
Dashboard interativo para visualização de performance de trading

Autor: CeciAI Team
Data: 2025-10-08

Uso:
    streamlit run dashboard.py --server.port 8050
"""

import asyncio
import os
from datetime import datetime, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# Importar módulos do CeciAI
from backtesting.paper_trading import PaperTradingEngine
from config.capital_management import CapitalManager
from utils.coinapi_client import CoinAPIClient, CoinAPIMode

# Configuração da página
st.set_page_config(
    page_title="CeciAI Trading Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Título
st.title("🤖 CeciAI - Trading Dashboard")
st.markdown("Sistema Inteligente de Trading com IA")

# ==================== FUNÇÕES AUXILIARES ====================


@st.cache_resource
def get_paper_trading_engine():
    """Inicializa Paper Trading Engine (singleton)"""
    return PaperTradingEngine(initial_capital=10000.0)


@st.cache_resource
def get_capital_manager():
    """Inicializa Capital Manager (singleton)"""
    return CapitalManager(initial_capital=10000.0)


def format_currency(value):
    """Formata valor como moeda"""
    return f"${value:,.2f}"


def format_percentage(value):
    """Formata valor como porcentagem"""
    return f"{value:+.2%}"


def create_equity_curve_chart(equity_data):
    """Cria gráfico de curva de equity"""
    if not equity_data:
        return None

    df = pd.DataFrame(equity_data)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["equity"],
            mode="lines",
            name="Equity",
            line=dict(color="green", width=2),
            fill="tozeroy",
        )
    )

    fig.update_layout(
        title="Curva de Equity",
        xaxis_title="Data/Hora",
        yaxis_title="Capital (USD)",
        hovermode="x unified",
        template="plotly_dark",
    )

    return fig


def create_trades_chart(trades):
    """Cria gráfico de trades ganhos vs perdidos"""
    if not trades:
        return None

    winning = [t for t in trades if t.get("pnl", 0) > 0]
    losing = [t for t in trades if t.get("pnl", 0) < 0]

    fig = go.Figure(
        data=[
            go.Bar(name="Ganhos", x=["Trades"], y=[len(winning)], marker_color="green"),
            go.Bar(name="Perdas", x=["Trades"], y=[len(losing)], marker_color="red"),
        ]
    )

    fig.update_layout(
        title="Trades Ganhos vs Perdidos",
        yaxis_title="Número de Trades",
        barmode="group",
        template="plotly_dark",
    )

    return fig


def create_pnl_distribution(trades):
    """Cria gráfico de distribuição de P&L"""
    if not trades:
        return None

    pnls = [t.get("pnl", 0) for t in trades]

    fig = go.Figure(data=[go.Histogram(x=pnls, nbinsx=20, marker_color="blue")])

    fig.update_layout(
        title="Distribuição de P&L",
        xaxis_title="P&L (USD)",
        yaxis_title="Frequência",
        template="plotly_dark",
    )

    return fig


# ==================== SIDEBAR ====================

with st.sidebar:
    st.header("⚙️ Configurações")

    # Seleção de modo
    mode = st.selectbox("Modo de Operação", ["Paper Trading", "Visualização de Dados"])

    # Status do sistema
    st.divider()
    st.subheader("📊 Status do Sistema")

    # Obter status do capital manager
    capital_mgr = get_capital_manager()
    status = capital_mgr.get_status()

    st.metric("Capital Total", format_currency(status["current_capital"]))
    st.metric("Capital Disponível", format_currency(status["available_capital"]))
    st.metric("P&L Total", format_currency(status["total_pnl"]), format_percentage(status["total_pnl_pct"]))

    # Circuit Breaker
    if status["circuit_breaker_active"]:
        st.error("⚠️ Circuit Breaker ATIVO")
    else:
        st.success("✅ Sistema Operacional")

    st.divider()

    # Ações rápidas
    st.subheader("🚀 Ações Rápidas")

    if st.button("🔄 Atualizar Dados", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()

    if st.button("📊 Exportar Relatório", use_container_width=True):
        st.info("Funcionalidade em desenvolvimento")

# ==================== MAIN DASHBOARD ====================

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Overview", "💼 Posições", "📈 Performance", "🔍 Histórico"]
)

# ==================== TAB 1: OVERVIEW ====================

with tab1:
    st.header("Visão Geral do Trading")

    # Métricas principais
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Capital Inicial",
            format_currency(status["initial_capital"]),
        )

    with col2:
        st.metric(
            "Capital Atual",
            format_currency(status["current_capital"]),
            format_currency(status["total_pnl"]),
        )

    with col3:
        st.metric(
            "Posições Abertas",
            status["open_positions"],
        )

    with col4:
        performance = capital_mgr.get_performance_metrics()
        win_rate = performance.get("win_rate", 0) if performance.get("total_trades", 0) > 0 else 0
        st.metric(
            "Win Rate",
            f"{win_rate:.1%}",
        )

    st.divider()

    # Gráficos
    col1, col2 = st.columns(2)

    with col1:
        # Curva de equity (simulada para exemplo)
        equity_data = [
            {"timestamp": datetime.now() - timedelta(days=i), "equity": 10000 + (i * 50)}
            for i in range(30, 0, -1)
        ]
        equity_data.append({"timestamp": datetime.now(), "equity": status["current_capital"]})

        equity_chart = create_equity_curve_chart(equity_data)
        if equity_chart:
            st.plotly_chart(equity_chart, use_container_width=True)

    with col2:
        # Trades ganhos vs perdidos
        closed_positions = capital_mgr.get_closed_positions()
        trades_chart = create_trades_chart(closed_positions)
        if trades_chart:
            st.plotly_chart(trades_chart, use_container_width=True)
        else:
            st.info("Nenhum trade executado ainda")

# ==================== TAB 2: POSIÇÕES ====================

with tab2:
    st.header("Posições Abertas")

    open_positions = capital_mgr.get_open_positions()

    if open_positions:
        for pos in open_positions:
            with st.expander(f"{pos.get('symbol', 'N/A')} - {pos.get('signal', 'N/A')}"):
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.write("**Detalhes da Posição**")
                    st.write(f"Entrada: {format_currency(pos.get('entry_price', 0))}")
                    st.write(f"Quantidade: {format_currency(pos.get('quantity_usd', 0))}")
                    st.write(f"Abertura: {pos.get('opened_at', 'N/A')}")

                with col2:
                    st.write("**Stop Loss & Take Profit**")
                    stop_loss = pos.get("stop_loss", {})
                    if isinstance(stop_loss, dict):
                        st.write(f"Stop Loss: {format_currency(stop_loss.get('price', 0))}")
                    else:
                        st.write(f"Stop Loss: {format_currency(stop_loss)}")

                    tp1 = pos.get("take_profit_1", {})
                    if isinstance(tp1, dict):
                        st.write(f"Take Profit: {format_currency(tp1.get('price', 0))}")
                    else:
                        st.write(f"Take Profit: {format_currency(tp1)}")

                with col3:
                    st.write("**Métricas**")
                    st.write(f"Confiança: {pos.get('confidence', 0):.1%}")
                    st.write(f"Score: {pos.get('opportunity_score', 0)}/100")

    else:
        st.info("Nenhuma posição aberta no momento")

# ==================== TAB 3: PERFORMANCE ====================

with tab3:
    st.header("Análise de Performance")

    performance = capital_mgr.get_performance_metrics()

    if performance.get("total_trades", 0) > 0:
        # Métricas de performance
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total de Trades", performance.get("total_trades", 0))

        with col2:
            st.metric(
                "Trades Ganhos",
                performance.get("winning_trades", 0),
                f"{performance.get('win_rate', 0):.1%}",
            )

        with col3:
            st.metric("Trades Perdidos", performance.get("losing_trades", 0))

        with col4:
            st.metric("Profit Factor", f"{performance.get('profit_factor', 0):.2f}")

        st.divider()

        # Mais métricas
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Ganho Médio", format_currency(performance.get("avg_win", 0)))

        with col2:
            st.metric("Perda Média", format_currency(performance.get("avg_loss", 0)))

        with col3:
            st.metric("Maior Ganho", format_currency(performance.get("largest_win", 0)))

        st.divider()

        # Distribuição de P&L
        closed_positions = capital_mgr.get_closed_positions()
        pnl_chart = create_pnl_distribution(closed_positions)
        if pnl_chart:
            st.plotly_chart(pnl_chart, use_container_width=True)

    else:
        st.info("Nenhum trade executado ainda. Execute alguns trades para ver as métricas de performance.")

# ==================== TAB 4: HISTÓRICO ====================

with tab4:
    st.header("Histórico de Trades")

    closed_positions = capital_mgr.get_closed_positions()

    if closed_positions:
        # Converter para DataFrame
        df = pd.DataFrame(closed_positions)

        # Selecionar colunas relevantes
        columns_to_show = [
            "symbol",
            "signal",
            "entry_price",
            "exit_price",
            "quantity_usd",
            "pnl",
            "pnl_pct",
            "close_reason",
        ]

        # Filtrar colunas existentes
        available_columns = [col for col in columns_to_show if col in df.columns]

        if available_columns:
            display_df = df[available_columns].copy()

            # Formatar valores monetários
            if "entry_price" in display_df.columns:
                display_df["entry_price"] = display_df["entry_price"].apply(lambda x: f"${x:,.2f}")
            if "exit_price" in display_df.columns:
                display_df["exit_price"] = display_df["exit_price"].apply(lambda x: f"${x:,.2f}")
            if "quantity_usd" in display_df.columns:
                display_df["quantity_usd"] = display_df["quantity_usd"].apply(lambda x: f"${x:,.2f}")
            if "pnl" in display_df.columns:
                display_df["pnl"] = display_df["pnl"].apply(lambda x: f"${x:,.2f}")
            if "pnl_pct" in display_df.columns:
                display_df["pnl_pct"] = display_df["pnl_pct"].apply(lambda x: f"{x:+.2%}")

            # Exibir tabela
            st.dataframe(display_df, use_container_width=True, hide_index=True)

            # Opção de download
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"ceci_ai_trades_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
            )
        else:
            st.warning("Não há colunas disponíveis para exibição")

    else:
        st.info("Nenhum trade no histórico ainda")

# ==================== FOOTER ====================

st.divider()
st.markdown(
    """
    <div style='text-align: center'>
        <p>🤖 CeciAI Trading System v1.0.0 | Desenvolvido com ❤️ para trading inteligente</p>
        <p><small>⚠️ Sistema em paper trading. Não use capital real sem testes completos.</small></p>
    </div>
    """,
    unsafe_allow_html=True,
)

