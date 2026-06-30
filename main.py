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
    page_title="Análise Sentar-Levantar (KMeans)",
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
# KMEANS - CLASSIFICAÇÃO EM 5 ESTADOS
# ============================================================
def classificar_kmeans(sinal, n_clusters=5, random_state=42):
    """
    Aplica KMeans para dividir o sinal em n_clusters estados.
    Retorna: rótulos de cada amostra, centros dos clusters ordenados.
    """
    X = sinal.reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X)
    centers = kmeans.cluster_centers_.flatten()
    
    # Ordena os clusters pelo valor do centro (do menor para o maior)
    ordem = np.argsort(centers)
    centers_ordenados = centers[ordem]
    
    # Mapeia os rótulos para a ordem (0 = menor, n-1 = maior)
    mapa = {antigo: novo for novo, antigo in enumerate(ordem)}
    labels_ordenados = np.array([mapa[l] for l in labels])
    
    return labels_ordenados, centers_ordenados


# ============================================================
# DETECÇÃO DA BASELINE (MENOR VARIABILIDADE)
# ============================================================
def encontrar_baseline(labels, janela_n=50):
    """
    Encontra a janela de 50 amostras (500 ms @ 100 Hz) com menor variabilidade
    nos rótulos (menor número de estados distintos e menor variância dos rótulos).
    """
    variancias = []
    for i in range(len(labels) - janela_n):
        janela = labels[i:i+janela_n]
        variancias.append(np.var(janela))
    
    idx_min = np.argmin(variancias)
    return idx_min, idx_min + janela_n


# ============================================================
# DETECÇÃO DO INÍCIO DA ATIVIDADE
# ============================================================
def detectar_inicio_atividade(labels, idx_fim_baseline, estado_baseline, 
                                sequencia_min=5):
    """
    A partir do final da baseline, busca a primeira sequência de 
    'sequencia_min' valores consecutivos em estados SUPERIORES ao baseline.
    """
    for i in range(idx_fim_baseline, len(labels) - sequencia_min + 1):
        sequencia = labels[i:i+sequencia_min]
        # Todos os valores devem ser estritamente maiores que o estado da baseline
        if all(s > estado_baseline for s in sequencia):
            return i
    return None


# ============================================================
# INTERFACE STREAMLIT
# ============================================================
st.title("📊 Análise Sentar-Levantar — KMeans (5 Estados)")
st.markdown("---")

# --- UPLOAD ---
arquivo = st.sidebar.file_uploader(
    "📁 Arquivo do acelerômetro",
    type=['txt', 'csv']
)

# --- PARÂMETROS ---
st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Parâmetros")
fc = st.sidebar.slider("Frequência de corte (Hz)", 1.0, 20.0, 8.0, 0.5)
n_clusters = st.sidebar.slider("Número de clusters (KMeans)", 3, 10, 5, 1)
janela_ms = st.sidebar.slider("Janela baseline (ms)", 200, 1000, 500, 50)
sequencia_min = st.sidebar.slider("Sequência mínima para detecção", 3, 10, 5, 1)

# ============================================================
# PROCESSAMENTO
# ============================================================
if arquivo is not None:
    with st.spinner("⏳ Processando..."):
        # Pipeline
        df = carregar_dados(arquivo)
        df_proc, fs = interpolar_100hz(df)
        df_proc = aplicar_detrend(df_proc)
        df_proc = filtrar_sinal(df_proc, fs, fc=fc)
        
        # Eixo Y
        sinal = df_proc['Y'].values
        tempo = df_proc['tempo_s'].values
        
        # KMeans
        labels, centers = classificar_kmeans(sinal, n_clusters=n_clusters)
        
        # Baseline
        janela_n = int(fs * janela_ms / 1000.0)
        idx_ini, idx_fim = encontrar_baseline(labels, janela_n)
        
        # Estado da baseline = cluster mais frequente na janela
        estado_baseline = int(np.median(labels[idx_ini:idx_fim]))
        
        # Início da atividade
        idx_atividade = detectar_inicio_atividade(
            labels, idx_fim, estado_baseline, sequencia_min
        )
        
        # ========================================================
        # RESULTADOS
        # ========================================================
        st.success("✅ Análise concluída!")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📍 Baseline", f"{tempo[idx_ini]:.2f} - {tempo[idx_fim]:.2f} s")
        col2.metric("🏷️ Estado Baseline", f"Cluster {estado_baseline}")
        col3.metric("🎯 Início Atividade", 
                    f"{tempo[idx_atividade]:.2f} s" if idx_atividade else "N/A")
        col4.metric("📊 Clusters", f"{n_clusters}")
        
        # Centros dos clusters
        st.subheader("🏷️ Centros dos Clusters (ordenados)")
        cols = st.columns(n_clusters)
        for i, c in enumerate(centers):
            cor = "🔴" if i == estado_baseline else "🔵" if i > estado_baseline else "⚪"
            cols[i].metric(f"Cluster {i}", f"{c:.3f} g", 
                          delta=f"{cor} {'baseline' if i == estado_baseline else 'superior' if i > estado_baseline else 'inferior'}")
        
        # ========================================================
        # GRÁFICO PRINCIPAL
        # ========================================================
        st.markdown("---")
        st.subheader("📈 Sinal e Classificação por Clusters")
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Sinal colorido por cluster
        cores_cluster = plt.cm.tab10(np.linspace(0, 1, n_clusters))
        for i in range(n_clusters):
            mask = labels == i
            ax.scatter(tempo[mask], sinal[mask], 
                      c=[cores_cluster[i]], s=3, alpha=0.5,
                      label=f'Cluster {i} (μ={centers[i]:.3f})')
        
        # Destaque da baseline
        ax.axvspan(tempo[idx_ini], tempo[idx_fim], 
                   color='green', alpha=0.15, label=f'Baseline ({janela_ms} ms)')
        
        # Marca o início da atividade
        if idx_atividade is not None:
            ax.axvline(tempo[idx_atividade], color='red', 
                       linestyle='--', linewidth=2.5,
                       label=f'Início Atividade = {tempo[idx_atividade]:.2f} s')
            ax.plot(tempo[idx_atividade], sinal[idx_atividade], 
                    'ro', markersize=12, markeredgecolor='black')
        
        # Linha dos centros dos clusters
        for i, c in enumerate(centers):
            ax.axhline(c, color=cores_cluster[i], linestyle=':', 
                       linewidth=0.8, alpha=0.7)
        
        ax.set_xlabel('Tempo (s)', fontsize=12)
        ax.set_ylabel('Aceleração Y (g)', fontsize=12)
        ax.set_title('Classificação KMeans - Eixo Y', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=9, ncol=2)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        
        # ========================================================
        # GRÁFICO DOS RÓTULOS AO LONGO DO TEMPO
        # ========================================================
        st.subheader("🏷️ Sequência de Estados ao Longo do Tempo")
        
        fig2, ax2 = plt.subplots(figsize=(14, 3))
        scatter = ax2.scatter(tempo, labels, c=labels, cmap='tab10', 
                             s=5, alpha=0.7)
        ax2.axvspan(tempo[idx_ini], tempo[idx_fim], 
                    color='green', alpha=0.2, label='Baseline')
        if idx_atividade is not None:
            ax2.axvline(tempo[idx_atividade], color='red', 
                        linestyle='--', linewidth=2, label='Início Atividade')
        ax2.set_xlabel('Tempo (s)')
        ax2.set_ylabel('Cluster')
        ax2.set_yticks(range(n_clusters))
        ax2.set_title('Sequência de Clusters')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig2)
        
        # ========================================================
        # ZOOM NA REGIÃO DE INTERESSE
        # ========================================================
        if idx_atividade is not None:
            st.subheader("🔍 Zoom na Transição Baseline → Atividade")
            
            t_zoom_ini = max(tempo[0], tempo[idx_atividade] - 1.0)
            t_zoom_fim = min(tempo[-1], tempo[idx_atividade] + 1.0)
            mask = (tempo >= t_zoom_ini) & (tempo <= t_zoom_fim)
            
            fig3, (ax3a, ax3b) = plt.subplots(2, 1, figsize=(12, 6), 
                                               sharex=True,
                                               gridspec_kw={'height_ratios': [2, 1]})
            
            # Sinal
            for i in range(n_clusters):
                mask_c = (labels == i) & mask
                ax3a.scatter(tempo[mask_c], sinal[mask_c], 
                            c=[cores_cluster[i]], s=8, alpha=0.7,
                            label=f'Cluster {i}')
            
            ax3a.axvspan(tempo[idx_ini], tempo[idx_fim], 
                         color='green', alpha=0.15)
            ax3a.axvline(tempo[idx_atividade], color='red', 
                         linestyle='--', linewidth=2)
            ax3a.set_ylabel('Aceleração Y (g)')
            ax3a.legend(loc='best', fontsize=8)
            ax3a.grid(True, alpha=0.3)
            ax3a.set_title('Zoom na Transição')
            
            # Rótulos
            ax3b.scatter(tempo[mask], labels[mask], c=labels[mask], 
                        cmap='tab10', s=15, alpha=0.8)
            ax3b.axvspan(tempo[idx_ini], tempo[idx_fim], 
                         color='green', alpha=0.2)
            ax3b.axvline(tempo[idx_atividade], color='red', 
                         linestyle='--', linewidth=2)
            ax3b.set_xlabel('Tempo (s)')
            ax3b.set_ylabel('Cluster')
            ax3b.set_yticks(range(n_clusters))
            ax3b.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig3)

else:
    st.info("👈 Faça upload de um arquivo para iniciar a análise.")
    
    st.markdown("""
    ### 📋 Algoritmo
    
    1. **Processamento do sinal** (interpolação, detrend, filtro)
    2. **KMeans** com 5 clusters para classificar cada amostra em um estado
    3. **Baseline**: janela de 500 ms com menor variabilidade nos clusters
    4. **Detecção**: primeira sequência de 5 valores consecutivos em clusters 
       **superiores** ao estado da baseline
    
    ### 🎯 Legenda
    
    - 🟢 **Verde**: janela de baseline (repouso)
    - 🔴 **Vermelho**: momento do início da atividade
    - 🔵 **Azul**: clusters superiores ao baseline
    - ⚪ **Cinza**: clusters inferiores ao baseline
    """)
