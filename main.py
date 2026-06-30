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
    page_title="Análise Sentar-Levantar (Corrigido)",
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


# ============================================================
# CORREÇÃO: BUSCAR BASELINE NO SINAL (NÃO NOS RÓTULOS)
# ============================================================
def encontrar_baseline_sinal(sinal, fs, janela_ms=500):
    """
    CORREÇÃO: Encontra a janela de 500 ms com MENOR VARIÂNCIA NO SINAL.
    Isso identifica o período de repouso real.
    
    Retorna: índice inicial, índice final, média do sinal, desvio padrão
    """
    janela_n = int(fs * janela_ms / 1000.0)
    
    # Calcula variância do SINAL (não dos rótulos) em janelas deslizantes
    variancias = []
    for i in range(len(sinal) - janela_n):
        janela = sinal[i:i+janela_n]
        variancias.append(np.var(janela))
    
    # Encontra a janela com MENOR variância do sinal
    idx_min = np.argmin(variancias)
    
    # Estatísticas da baseline no sinal
    baseline = sinal[idx_min:idx_min + janela_n]
    media = np.mean(baseline)
    dp = np.std(baseline)
    
    return idx_min, idx_min + janela_n, media, dp


def classificar_kmeans(sinal, n_clusters=5, random_state=42):
    """Classifica o sinal em clusters usando KMeans."""
    X = sinal.reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X)
    centers = kmeans.cluster_centers_.flatten()
    
    # Ordena clusters pelo valor do centro
    ordem = np.argsort(centers)
    centers_ordenados = centers[ordem]
    mapa = {antigo: novo for novo, antigo in enumerate(ordem)}
    labels_ordenados = np.array([mapa[l] for l in labels])
    
    return labels_ordenados, centers_ordenados


def detectar_inicio_atividade(labels, idx_fim_baseline, estado_baseline, 
                                sequencia_min=5):
    """Busca sequência de valores em estados superiores ao baseline."""
    for i in range(idx_fim_baseline, len(labels) - sequencia_min + 1):
        sequencia = labels[i:i+sequencia_min]
        if all(s > estado_baseline for s in sequencia):
            return i
    return None


# ============================================================
# PIPELINE COMPLETO
# ============================================================
def processar_acelerometro(df_original, fs_alvo=100, fc=8.0):
    df_interp, fs = interpolar_100hz(df_original, fs_alvo)
    df_detrend = aplicar_detrend(df_interp)
    df_filtrado = filtrar_sinal(df_detrend, fs, fc=fc)
    return df_filtrado, fs


# ============================================================
# INTERFACE STREAMLIT
# ============================================================
st.title("📊 Análise Sentar-Levantar — Detecção de Baseline Corrigida")
st.markdown("---")

arquivo = st.sidebar.file_uploader(
    "📁 Carregue o arquivo do acelerômetro",
    type=['txt', 'csv']
)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Parâmetros")
fc = st.sidebar.slider("Frequência de corte (Hz)", 1.0, 20.0, 8.0, 0.5)
fs_alvo = st.sidebar.slider("Frequência de amostragem (Hz)", 50, 200, 100, 10)
janela_ms = st.sidebar.slider("Janela baseline (ms)", 200, 1000, 500, 50)
n_clusters = st.sidebar.slider("Número de clusters", 3, 10, 5, 1)
sequencia_min = st.sidebar.slider("Sequência mínima", 3, 10, 5, 1)

# ============================================================
# PROCESSAMENTO
# ============================================================
if arquivo is not None:
    with st.spinner("⏳ Processando..."):
        df_original = carregar_dados(arquivo)
        
        if df_original is not None:
            st.success("✅ Arquivo carregado!")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("📏 Amostras", f"{len(df_original):,}")
            col2.metric("⏱️ Duração", f"{df_original['tempo_s'].iloc[-1]:.2f} s")
            col3.metric("📡 Fs Original", 
                       f"{1000/np.mean(np.diff(df_original['tempo_ms'])):.1f} Hz")
            
            # Processamento
            df_proc, fs = processar_acelerometro(df_original, fs_alvo, fc)
            sinal_y = df_proc['Y'].values
            tempo = df_proc['tempo_s'].values
            
            # CORREÇÃO: Buscar baseline NO SINAL
            idx_ini, idx_fim, media_sinal, dp_sinal = encontrar_baseline_sinal(
                sinal_y, fs, janela_ms
            )
            
            # KMeans para classificar em estados
            labels, centers = classificar_kmeans(sinal_y, n_clusters)
            
            # Identificar qual é o cluster dominante na baseline
            labels_baseline = labels[idx_ini:idx_fim]
            estado_baseline = int(np.bincount(labels_baseline).argmax())
            
            # Detectar início da atividade
            idx_atividade = detectar_inicio_atividade(
                labels, idx_fim, estado_baseline, sequencia_min
            )
            
            # ========================================================
            # RESULTADOS
            # ========================================================
            st.markdown("---")
            st.subheader("📋 Resultados da Detecção")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("📍 Baseline", 
                       f"{tempo[idx_ini]:.2f} - {tempo[idx_fim]:.2f} s")
            col2.metric("📊 Média Baseline", f"{media_sinal:.4f} g")
            col3.metric("📐 DP Baseline", f"{dp_sinal:.4f} g")
            col4.metric("🏷️ Estado Baseline", f"Cluster {estado_baseline}")
            
            if idx_atividade is not None:
                st.metric("🎯 Início Atividade", 
                         f"{tempo[idx_atividade]:.2f} s")
            else:
                st.warning("⚠️ Não foi possível detectar o início da atividade")
            
            # ========================================================
            # GRÁFICO 1: SINAL COM BASELINE DESTACADA
            # ========================================================
            st.markdown("---")
            st.subheader("📈 Sinal do Eixo Y com Baseline Detectada")
            
            fig, ax = plt.subplots(figsize=(14, 6))
            
            # Sinal completo
            ax.plot(tempo, sinal_y, 'b-', linewidth=0.8, alpha=0.7, 
                   label='Sinal Y (filtrado)')
            
            # Destaque da BASELINE (CORREÇÃO: agora no sinal real)
            ax.axvspan(tempo[idx_ini], tempo[idx_fim], 
                      color='green', alpha=0.3, linewidth=2,
                      label=f'Baseline ({janela_ms} ms) - Menor variância')
            
            # Linha da média da baseline
            ax.axhline(media_sinal, color='darkgreen', linestyle='--', 
                      linewidth=2, label=f'Média baseline = {media_sinal:.4f} g')
            
            # Marca o início da atividade
            if idx_atividade is not None:
                ax.axvline(tempo[idx_atividade], color='red', 
                          linestyle=':', linewidth=2.5,
                          label=f'Início Atividade = {tempo[idx_atividade]:.2f} s')
                ax.plot(tempo[idx_atividade], sinal_y[idx_atividade], 
                       'ro', markersize=12, markeredgecolor='black', zorder=5)
            
            ax.set_xlabel('Tempo (s)', fontsize=12)
            ax.set_ylabel('Aceleração Y (g)', fontsize=12)
            ax.set_title('Detecção de Baseline no Sinal Original (Corrigido)', 
                        fontsize=14, fontweight='bold')
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            
            # ========================================================
            # GRÁFICO 2: VARIÂNCIA AO LONGO DO TEMPO
            # ========================================================
            st.subheader("📊 Variância do Sinal (Janelas de 500 ms)")
            
            # Calcula variância em janelas deslizantes para visualização
            janela_n = int(fs * janela_ms / 1000.0)
            variancias = []
            tempos_var = []
            for i in range(len(sinal_y) - janela_n):
                janela = sinal_y[i:i+janela_n]
                variancias.append(np.var(janela))
                tempos_var.append(tempo[i + janela_n//2])
            
            fig2, ax2 = plt.subplots(figsize=(14, 4))
            ax2.plot(tempos_var, variancias, 'k-', linewidth=0.8)
            ax2.axvspan(tempo[idx_ini], tempo[idx_fim], 
                       color='green', alpha=0.3, 
                       label='Baseline selecionada')
            ax2.set_xlabel('Tempo (s)', fontsize=12)
            ax2.set_ylabel('Variância (g²)', fontsize=12)
            ax2.set_title('Variância do Sinal - Identificação da Baseline', 
                         fontsize=13, fontweight='bold')
            ax2.legend(loc='best')
            ax2.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig2)
            
            # ========================================================
            # GRÁFICO 3: ZOOM NA REGIÃO DE INTERESSE
            # ========================================================
            if idx_atividade is not None:
                st.subheader("🔍 Zoom na Transição Baseline → Atividade")
                
                t_zoom_ini = max(tempo[0], tempo[idx_atividade] - 1.5)
                t_zoom_fim = min(tempo[-1], tempo[idx_atividade] + 1.5)
                mask = (tempo >= t_zoom_ini) & (tempo <= t_zoom_fim)
                
                fig3, ax3 = plt.subplots(figsize=(12, 5))
                ax3.plot(tempo[mask], sinal_y[mask], 'b-', linewidth=1.2)
                ax3.axhline(media_sinal, color='darkgreen', linestyle='--', linewidth=2)
                ax3.axvline(tempo[idx_atividade], color='red', 
                           linestyle=':', linewidth=2.5)
                ax3.plot(tempo[idx_atividade], sinal_y[idx_atividade], 
                        'ro', markersize=12, markeredgecolor='black')
                ax3.set_xlabel('Tempo (s)', fontsize=12)
                ax3.set_ylabel('Aceleração Y (g)', fontsize=12)
                ax3.set_title('Zoom na Transição', fontsize=13, fontweight='bold')
                ax3.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig3)
            
            # ========================================================
            # GRÁFICO 4: CLUSTERS AO LONGO DO TEMPO
            # ========================================================
            st.subheader("🏷️ Classificação por Clusters (KMeans)")
            
            fig4, (ax4a, ax4b) = plt.subplots(2, 1, figsize=(14, 6), 
                                               sharex=True,
                                               gridspec_kw={'height_ratios': [2, 1]})
            
            # Sinal colorido por cluster
            cores_cluster = plt.cm.tab10(np.linspace(0, 1, n_clusters))
            for i in range(n_clusters):
                mask = labels == i
                ax4a.scatter(tempo[mask], sinal_y[mask], 
                            c=[cores_cluster[i]], s=3, alpha=0.5,
                            label=f'Cluster {i} (μ={centers[i]:.3f})')
            
            ax4a.axvspan(tempo[idx_ini], tempo[idx_fim], 
                        color='green', alpha=0.2, label='Baseline')
            if idx_atividade is not None:
                ax4a.axvline(tempo[idx_atividade], color='red', 
                            linestyle=':', linewidth=2.5)
            
            ax4a.set_ylabel('Aceleração Y (g)')
            ax4a.legend(loc='best', fontsize=9, ncol=2)
            ax4a.grid(True, alpha=0.3)
            ax4a.set_title('Sinal Classificado por Clusters')
            
            # Rótulos ao longo do tempo
            ax4b.scatter(tempo, labels, c=labels, cmap='tab10', 
                        s=5, alpha=0.7)
            ax4b.axvspan(tempo[idx_ini], tempo[idx_fim], 
                        color='green', alpha=0.2)
            if idx_atividade is not None:
                ax4b.axvline(tempo[idx_atividade], color='red', 
                            linestyle=':', linewidth=2.5)
            ax4b.set_xlabel('Tempo (s)')
            ax4b.set_ylabel('Cluster')
            ax4b.set_yticks(range(n_clusters))
            ax4b.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig4)

else:
    st.info("👈 Faça upload de um arquivo para iniciar a análise.")
    
    st.markdown("""
    ### 🔧 Correção Aplicada
    
    **Problema anterior**: O código buscava a menor variabilidade nos **rótulos do KMeans** 
    (valores discretos 0-4), o que não representa o repouso real.
    
    **Solução**: Agora buscamos a janela de menor variabilidade no **sinal original** 
    (eixo Y filtrado), que identifica corretamente o período de repouso.
    
    ### 📋 Algoritmo Corrigido
    
    1. **Processamento do sinal** (interpolação, detrend, filtro)
    2. **Busca da baseline**: Janela de 500 ms com MENOR VARIÂNCIA NO SINAL
    3. **KMeans**: Classifica o sinal em 5 estados
    4. **Identificação**: Cluster dominante na janela de baseline
    5. **Detecção**: Primeira sequência de 5 valores em clusters superiores
    
    ### 🎯 Interpretação
    
    - 🟢 **Verde**: janela de baseline (repouso real - menor variância)
    - 🔴 **Vermelho**: momento do início da atividade
    """)
