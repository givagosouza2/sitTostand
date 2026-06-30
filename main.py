import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.interpolate import interp1d
from sklearn.cluster import KMeans

# ============================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================
st.set_page_config(
    page_title="Análise Sit-to-Stand",
    page_icon="📊",
    layout="wide"
)

# ============================================================
# FUNÇÕES DE PROCESSAMENTO
# ============================================================
@st.cache_data
def carregar_dados(arquivo):
    try:
        df = pd.read_csv(arquivo, sep=';')
        df.columns = ['tempo_ms', 'X', 'Y', 'Z']
        df['tempo_s'] = df['tempo_ms'] / 1000.0
        return df
    except Exception as e:
        st.error(f"Erro ao carregar arquivo: {e}")
        return None


def interpolar_100hz(df, fs_alvo=100):
    t_original = df['tempo_s'].values
    t_novo = np.arange(t_original[0], t_original[-1], 1.0/fs_alvo)
    df_interp = pd.DataFrame({'tempo_s': t_novo})
    for eixo in ['X', 'Y', 'Z']:
        f_interp = interp1d(t_original, df[eixo].values,
                            kind='linear', fill_value='extrapolate')
        df_interp[eixo] = f_interp(t_novo)
    return df_interp, fs_alvo


def aplicar_detrend(df, eixos=['X', 'Y', 'Z']):
    df_detrend = df.copy()
    for eixo in eixos:
        df_detrend[eixo] = signal.detrend(df[eixo].values, type='linear')
    return df_detrend


def filtrar_sinal(df, fs, fc=8.0, ordem=4, eixos=['X', 'Y', 'Z']):
    nyq = fs / 2.0
    b, a = signal.butter(ordem, fc/nyq, btype='low')
    df_filtrado = df.copy()
    for eixo in eixos:
        df_filtrado[eixo] = signal.filtfilt(b, a, df[eixo].values)
    return df_filtrado


def classificar_kmeans(sinal, n_clusters=5, random_state=42):
    X = sinal.reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X)
    centers = kmeans.cluster_centers_.flatten()
    ordem = np.argsort(centers)
    centers_ordenados = centers[ordem]
    mapa = {antigo: novo for novo, antigo in enumerate(ordem)}
    labels_ordenados = np.array([mapa[l] for l in labels])
    return labels_ordenados, centers_ordenados


def encontrar_baseline(labels, janela_n=50):
    variancias = []
    for i in range(len(labels) - janela_n):
        janela = labels[i:i+janela_n]
        variancias.append(np.var(janela))
    idx_min = np.argmin(variancias)
    return idx_min, idx_min + janela_n


def detectar_inicio_atividade(labels, idx_fim_baseline, estado_baseline, 
                                sequencia_min=5):
    for i in range(idx_fim_baseline, len(labels) - sequencia_min + 1):
        sequencia = labels[i:i+sequencia_min]
        if all(s > estado_baseline for s in sequencia):
            return i
    return None


# ============================================================
# INTERFACE STREAMLIT
# ============================================================
st.title("📊 Análise do Teste Sentar-Levantar")
st.markdown("---")

# --- UPLOAD DO ARQUIVO ---
st.sidebar.header("⚙️ Configurações")
arquivo = st.sidebar.file_uploader(
    "📁 Carregue o arquivo do acelerômetro",
    type=['txt', 'csv']
)

# --- PARÂMETROS ---
st.sidebar.markdown("---")
st.sidebar.subheader("🔧 Parâmetros de Processamento")
fc = st.sidebar.slider("Frequência de corte (Hz)", 1.0, 20.0, 8.0, 0.5)
n_clusters = st.sidebar.slider("Número de clusters (KMeans)", 3, 10, 5, 1)
janela_ms = st.sidebar.slider("Janela baseline (ms)", 200, 1000, 500, 50)
sequencia_min = st.sidebar.slider("Sequência mínima p/ detecção", 3, 10, 5, 1)

# --- NOVO: DURAÇÃO DA JANELA DE ANÁLISE ---
st.sidebar.markdown("---")
st.sidebar.subheader("⏱️ Janela de Análise")
duracao_janela = st.sidebar.slider(
    "Duração após início da atividade (s)",
    min_value=5.0,
    max_value=120.0,
    value=30.0,
    step=1.0,
    help="Define o momento final como (início + duração)"
)

# ============================================================
# PROCESSAMENTO
# ============================================================
if arquivo is not None:
    with st.spinner("⏳ Processando..."):
        # Pipeline
        df = carregar_dados(arquivo)
        df_proc, fs = interpolar_100hz(df, fs_alvo=100)
        df_proc = aplicar_detrend(df_proc)
        df_proc = filtrar_sinal(df_proc, fs, fc=fc)
        
        sinal = df_proc['Y'].values
        tempo = df_proc['tempo_s'].values
        
        # KMeans
        labels, centers = classificar_kmeans(sinal, n_clusters=n_clusters)
        
        # Baseline
        janela_n = int(fs * janela_ms / 1000.0)
        idx_ini, idx_fim = encontrar_baseline(labels, janela_n)
        estado_baseline = int(np.median(labels[idx_ini:idx_fim]))
        
        # Início da atividade
        idx_atividade = detectar_inicio_atividade(
            labels, idx_fim, estado_baseline, sequencia_min
        )
        
        # ========================================================
        # CÁLCULO DO MOMENTO FINAL (NOVO!)
        # ========================================================
        if idx_atividade is not None:
            t_inicio = tempo[idx_atividade]
            t_fim = t_inicio + duracao_janela
            
            # Garante que não ultrapassa o fim do sinal
            t_fim = min(t_fim, tempo[-1])
            
            # Encontra o índice correspondente ao tempo final
            idx_fim_janela = np.searchsorted(tempo, t_fim)
            idx_fim_janela = min(idx_fim_janela, len(tempo) - 1)
            
            # Dados dentro da janela de análise
            mask_janela = (tempo >= t_inicio) & (tempo <= t_fim)
            dados_janela = sinal[mask_janela]
            tempo_janela = tempo[mask_janela]
        
        # ========================================================
        # RESULTADOS
        # ========================================================
        st.success("✅ Análise concluída!")
        
        st.markdown("### 📋 Marcos Temporais Detectados")
        
        if idx_atividade is not None:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("🟢 Início Atividade", f"{t_inicio:.2f} s")
            col2.metric("🔴 Fim da Janela", f"{t_fim:.2f} s")
            col3.metric("⏱️ Duração", f"{duracao_janela:.1f} s")
            col4.metric("📏 Amstras na Janela", f"{len(dados_janela)}")
            
            # ========================================================
            # ESTATÍSTICAS DENTRO DA JANELA DE ANÁLISE
            # ========================================================
            st.markdown("### 📊 Estatísticas na Janela de Análise (Eixo Y)")
            stat1, stat2, stat3, stat4 = st.columns(4)
            stat1.metric("Média", f"{np.mean(dados_janela):.4f} g")
            stat2.metric("Desvio Padrão", f"{np.std(dados_janela):.4f} g")
            stat3.metric("Máximo", f"{np.max(dados_janela):.4f} g")
            stat4.metric("Mínimo", f"{np.min(dados_janela):.4f} g")
        else:
            st.warning("⚠️ Não foi possível detectar o início da atividade.")
        
        # ========================================================
        # GRÁFICO PRINCIPAL
        # ========================================================
        st.markdown("---")
        st.markdown("### 📈 Sinal do Eixo Y com Janela de Análise")
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Sinal completo (cinza claro)
        ax.plot(tempo, sinal, color='lightgray', linewidth=0.5, 
                label='Sinal completo', alpha=0.7)
        
        # Sinal dentro da janela (azul)
        if idx_atividade is not None:
            ax.plot(tempo_janela, dados_janela, color='steelblue', 
                    linewidth=1.5, label='Janela de análise')
        
        # Destaque da baseline
        ax.axvspan(tempo[idx_ini], tempo[idx_fim], 
                   color='green', alpha=0.2, label=f'Baseline ({janela_ms} ms)')
        
        # Destaque da janela de análise (amarelo)
        if idx_atividade is not None:
            ax.axvspan(t_inicio, t_fim, 
                       color='gold', alpha=0.2, 
                       label=f'Janela ({duracao_janela:.0f}s)')
        
        # Linhas verticais dos marcos
        if idx_atividade is not None:
            ax.axvline(t_inicio, color='green', linestyle='--', 
                       linewidth=2, label=f'Início = {t_inicio:.2f} s')
            ax.axvline(t_fim, color='red', linestyle='--', 
                       linewidth=2, label=f'Fim = {t_fim:.2f} s')
        
        ax.set_xlabel('Tempo (s)', fontsize=12)
        ax.set_ylabel('Aceleração Y (g)', fontsize=12)
        ax.set_title('Eixo Y — Detecção de Início e Janela de Análise (+30s)', 
                     fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=9, ncol=2)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        
        # ========================================================
        # ZOOM NA JANELA DE ANÁLISE
        # ========================================================
        if idx_atividade is not None:
            st.markdown("### 🔍 Zoom na Janela de Análise")
            
            fig2, ax2 = plt.subplots(figsize=(14, 5))
            ax2.plot(tempo_janela, dados_janela, color='steelblue', linewidth=1.5)
            ax2.axhline(np.mean(dados_janela), color='orange', 
                        linestyle='--', linewidth=1.5, 
                        label=f'Média = {np.mean(dados_janela):.4f} g')
            ax2.set_xlabel('Tempo (s)', fontsize=12)
            ax2.set_ylabel('Aceleração Y (g)', fontsize=12)
            ax2.set_title(f'Janela de Análise: {t_inicio:.2f} s → {t_fim:.2f} s', 
                         fontsize=13, fontweight='bold')
            ax2.legend(loc='best')
            ax2.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig2)
        
        # ========================================================
        # DOWNLOAD DOS RESULTADOS
        # ========================================================
        if idx_atividade is not None:
            st.markdown("---")
            
            # DataFrame com a janela de análise
            df_janela = pd.DataFrame({
                'tempo_s': tempo_janela,
                'aceleracao_Y_g': dados_janela
            })
            
            csv_janela = df_janela.to_csv(index=False).encode('utf-8')
            csv_resultados = pd.DataFrame({
                'Parametro': ['Inicio atividade (s)', 'Fim da janela (s)', 
                              'Duracao (s)', 'Media baseline (cluster)',
                              'Estado baseline', 'Numero de amostras na janela',
                              'Media janela (g)', 'DP janela (g)',
                              'Max janela (g)', 'Min janela (g)'],
                'Valor': [t_inicio, t_fim, duracao_janela, 
                          centers[estado_baseline], estado_baseline,
                          len(dados_janela), np.mean(dados_janela),
                          np.std(dados_janela), np.max(dados_janela),
                          np.min(dados_janela)]
            }).to_csv(index=False).encode('utf-8')
            
            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                st.download_button(
                    "💾 Baixar dados da janela (CSV)",
                    csv_janela,
                    "janela_analise_30s.csv",
                    "text/csv"
                )
            with col_dl2:
                st.download_button(
                    "📄 Baixar relatório de parâmetros",
                    csv_resultados,
                    "relatorio_sitstand.csv",
                    "text/csv"
                )

else:
    st.info("👈 Faça upload de um arquivo na barra lateral para começar a análise.")
    
    st.markdown("""
    ### 📋 Algoritmo
    
    1. **Processamento do sinal**: interpolação (100 Hz), detrend, filtro passa-baixa
    2. **KMeans (5 clusters)**: classificação do sinal em estados
    3. **Baseline**: janela de menor variabilidade nos clusters
    4. **Início da atividade**: primeira sequência de 5 valores em clusters superiores
    5. **Fim da janela**: **início + 30 segundos** (ajustável na barra lateral)
    
    ### 🎯 Legenda do Gráfico
    
    - 🟢 **Verde**: janela de baseline (repouso)
    - 🟡 **Amarelo**: janela de análise (30s após início)
    - 🟢 **Linha verde tracejada**: momento do início da atividade
    - 🔴 **Linha vermelha tracejada**: momento final (início + 30s)
    """)
