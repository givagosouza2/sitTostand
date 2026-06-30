import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.interpolate import interp1d
from sklearn.cluster import KMeans

# ============================================================
# CONFIGURAÇÃO
# ============================================================
st.set_page_config(page_title="Sit-to-Stand - Detecção de Atividade", layout="wide")

# ============================================================
# FUNÇÕES DE PROCESSAMENTO DO SINAL
# ============================================================
@st.cache_data
def carregar_dados(arquivo):
    df = pd.read_csv(arquivo, sep=';')
    df.columns = ['tempo_ms', 'X', 'Y', 'Z']
    df['tempo_s'] = df['tempo_ms'] / 1000.0
    return df


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
# PASSO 1: K-MEANS - SEPARAÇÃO EM 5 ESTADOS
# ============================================================
def classificar_kmeans(sinal, n_clusters=5, random_state=42):
    """
    Classifica cada amostra do sinal em um dos n_clusters estados.
    Retorna os rótulos ordenados (0 = menor amplitude, n-1 = maior).
    """
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


# ============================================================
# PASSO 2: BUSCAR JANELA DE MENOR VARIABILIDADE NO SINAL ORIGINAL
# ============================================================
def encontrar_baseline_sinal_original(sinal, fs, janela_ms=500):
    """
    Busca no SINAL ORIGINAL (não nos clusters!) a janela de 500 ms
    com menor variabilidade. Isso representa o período de repouso.
    
    Retorna: (índice_início, índice_fim) da janela no sinal.
    """
    janela_n = int(fs * janela_ms / 1000.0)  # 50 amostras @ 100 Hz
    
    variancias = []
    for i in range(len(sinal) - janela_n):
        janela = sinal[i:i+janela_n]
        variancias.append(np.var(janela))
    
    idx_min = np.argmin(variancias)
    return idx_min, idx_min + janela_n


# ============================================================
# PASSO 3: ESTADO DOMINANTE NA JANELA DE BASELINE
# ============================================================
def estado_dominante(labels, idx_ini, idx_fim):
    """
    Identifica qual cluster (estado) é o mais frequente 
    dentro da janela de baseline.
    """
    labels_janela = labels[idx_ini:idx_fim]
    contagens = np.bincount(labels_janela)
    estado = int(np.argmax(contagens))
    return estado


# ============================================================
# PASSO 4: DETECÇÃO ANTERÓGRADA DO INÍCIO DA ATIVIDADE
# ============================================================
def detectar_inicio_atividade(labels, idx_fim_baseline, estado_baseline, 
                                sequencia_min=5):
    """
    A partir do FINAL da janela de baseline, busca anterogradamente
    (para frente no tempo) a primeira sequência de `sequencia_min`
    amostras consecutivas em estados SUPERIORES ao estado dominante
    da baseline.
    
    Retorna: índice do primeiro ponto da sequência (início da atividade)
    """
    n_labels = len(labels)
    
    for i in range(idx_fim_baseline, n_labels - sequencia_min + 1):
        sequencia = labels[i:i+sequencia_min]
        # Todos os valores devem ser ESTRIAMENTE SUPERIORES ao estado da baseline
        if all(s > estado_baseline for s in sequencia):
            return i  # Retorna o índice do PRIMEIRO ponto da sequência
    
    return None  # Não encontrou


# ============================================================
# INTERFACE STREAMLIT
# ============================================================
st.title("📊 Sit-to-Stand - Detecção Automática do Início da Atividade")
st.markdown("---")

# Sidebar
st.sidebar.header("⚙️ Parâmetros")
arquivo = st.sidebar.file_uploader("📁 Arquivo do acelerômetro", type=['txt', 'csv'])

fc = st.sidebar.slider("Frequência de corte (Hz)", 1.0, 20.0, 8.0, 0.5)
n_clusters = st.sidebar.slider("Número de clusters (KMeans)", 3, 10, 5, 1)
janela_ms = st.sidebar.slider("Janela baseline (ms)", 200, 1000, 500, 50)
sequencia_min = st.sidebar.slider("Sequência mínima p/ detecção", 3, 10, 5, 1)
duracao_janela = st.sidebar.slider("Duração após início (s)", 5.0, 120.0, 30.0, 1.0)

# ============================================================
# PROCESSAMENTO
# ============================================================
if arquivo is not None:
    with st.spinner("⏳ Processando..."):
        # Pipeline de processamento
        df = carregar_dados(arquivo)
        df_proc, fs = interpolar_100hz(df, fs_alvo=100)
        df_proc = aplicar_detrend(df_proc)
        df_proc = filtrar_sinal(df_proc, fs, fc=fc)
        
        sinal_y = df_proc['Y'].values
        tempo = df_proc['tempo_s'].values
        
        # PASSO 1: K-Means
        labels, centers = classificar_kmeans(sinal_y, n_clusters=n_clusters)
        
        # PASSO 2: Janela de menor variabilidade NO SINAL ORIGINAL
        idx_ini, idx_fim = encontrar_baseline_sinal_original(
            sinal_y, fs, janela_ms
        )
        
        # PASSO 3: Estado dominante na janela de baseline
        est_baseline = estado_dominante(labels, idx_ini, idx_fim)
        
        # PASSO 4: Detecção anterógrada do início da atividade
        idx_atividade = detectar_inicio_atividade(
            labels, idx_fim, est_baseline, sequencia_min
        )
        
        # PASSO 5: Marcar 30 segundos após o início
        if idx_atividade is not None:
            t_inicio = tempo[idx_atividade]
            t_fim = min(t_inicio + duracao_janela, tempo[-1])
            idx_fim_janela = np.searchsorted(tempo, t_fim)
            idx_fim_janela = min(idx_fim_janela, len(tempo) - 1)
        
        # ========================================================
        # RESULTADOS
        # ========================================================
        st.success("✅ Análise concluída!")
        
        st.markdown("### 🎯 Marcos Detectados")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("📍 Baseline Início", f"{tempo[idx_ini]:.2f} s")
        col2.metric("📍 Baseline Fim", f"{tempo[idx_fim]:.2f} s")
        col3.metric("🏷️ Estado Baseline", f"Cluster {est_baseline}")
        col4.metric("🎯 Início Atividade", 
                    f"{t_inicio:.2f} s" if idx_atividade is not None else "N/A")
        col5.metric("🏁 Fim Janela (+30s)", 
                    f"{t_fim:.2f} s" if idx_atividade is not None else "N/A")
        
        # ========================================================
        # GRÁFICO 1: SINAL ORIGINAL COM BASELINE E ATIVIDADE
        # ========================================================
        st.markdown("### 📈 Sinal Original (Eixo Y) - Detecção de Marcos")
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Sinal completo
        ax.plot(tempo, sinal_y, 'b-', linewidth=0.6, alpha=0.7, label='Sinal Y')
        
        # Janela de baseline (menor variabilidade no sinal original)
        ax.axvspan(tempo[idx_ini], tempo[idx_fim], 
                   color='green', alpha=0.3, linewidth=2,
                   label=f'Baseline ({janela_ms} ms) - Menor variância')
        
        # Janela de atividade (+30s)
        if idx_atividade is not None:
            ax.axvspan(t_inicio, t_fim, 
                       color='gold', alpha=0.3, linewidth=2,
                       label=f'Janela de Análise (+{duracao_janela:.0f}s)')
            
            # Linha do início da atividade
            ax.axvline(t_inicio, color='red', linestyle='--', linewidth=2.5,
                       label=f'Início Atividade = {t_inicio:.2f} s')
            ax.plot(t_inicio, sinal_y[idx_atividade], 
                    'ro', markersize=12, markeredgecolor='black', zorder=5)
            
            # Linha do fim
            ax.axvline(t_fim, color='purple', linestyle='--', linewidth=2,
                       label=f'Fim (+30s) = {t_fim:.2f} s')
        
        ax.set_xlabel('Tempo (s)', fontsize=12)
        ax.set_ylabel('Aceleração Y (g)', fontsize=12)
        ax.set_title('Detecção Automática - Sinal Original', fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        
        # ========================================================
        # GRÁFICO 2: CLUSTERS AO LONGO DO TEMPO
        # ========================================================
        st.markdown("### 🏷️ Classificação por Clusters (KMeans)")
        
        fig2, (ax2a, ax2b) = plt.subplots(2, 1, figsize=(14, 6), 
                                           sharex=True,
                                           gridspec_kw={'height_ratios': [2, 1]})
        
        # Sinal colorido por cluster
        cores_cluster = plt.cm.tab10(np.linspace(0, 1, n_clusters))
        for i in range(n_clusters):
            mask = labels == i
            ax2a.scatter(tempo[mask], sinal_y[mask], 
                        c=[cores_cluster[i]], s=3, alpha=0.5,
                        label=f'Cluster {i} (μ={centers[i]:.3f})')
        
        ax2a.axvspan(tempo[idx_ini], tempo[idx_fim], 
                     color='green', alpha=0.2)
        if idx_atividade is not None:
            ax2a.axvspan(t_inicio, t_fim, color='gold', alpha=0.2)
            ax2a.axvline(t_inicio, color='red', linestyle='--', linewidth=2)
        
        ax2a.set_ylabel('Aceleração Y (g)')
        ax2a.legend(loc='best', fontsize=8, ncol=2)
        ax2a.grid(True, alpha=0.3)
        ax2a.set_title('Sinal Classificado por Clusters')
        
        # Rótulos ao longo do tempo
        ax2b.scatter(tempo, labels, c=labels, cmap='tab10', s=5, alpha=0.7)
        ax2b.axvspan(tempo[idx_ini], tempo[idx_fim], color='green', alpha=0.2)
        if idx_atividade is not None:
            ax2b.axvspan(t_inicio, t_fim, color='gold', alpha=0.2)
            ax2b.axvline(t_inicio, color='red', linestyle='--', linewidth=2)
        ax2b.set_xlabel('Tempo (s)')
        ax2b.set_ylabel('Cluster')
        ax2b.set_yticks(range(n_clusters))
        ax2b.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig2)
        
        # ========================================================
        # ZOOM NA REGIÃO DE TRANSIÇÃO
        # ========================================================
        if idx_atividade is not None:
            st.markdown("### 🔍 Zoom na Transição Baseline → Atividade")
            
            t_zoom_ini = max(tempo[0], t_inicio - 2.0)
            t_zoom_fim = min(tempo[-1], t_inicio + 3.0)
            mask = (tempo >= t_zoom_ini) & (tempo <= t_zoom_fim)
            
            fig3, (ax3a, ax3b) = plt.subplots(2, 1, figsize=(12, 6), 
                                               sharex=True,
                                               gridspec_kw={'height_ratios': [2, 1]})
            
            # Sinal
            for i in range(n_clusters):
                mask_c = (labels == i) & mask
                ax3a.scatter(tempo[mask_c], sinal_y[mask_c], 
                            c=[cores_cluster[i]], s=8, alpha=0.7,
                            label=f'Cluster {i}')
            
            ax3a.axvspan(tempo[idx_ini], tempo[idx_fim], color='green', alpha=0.2)
            ax3a.axvline(t_inicio, color='red', linestyle='--', linewidth=2.5)
            ax3a.set_ylabel('Aceleração Y (g)')
            ax3a.legend(loc='best', fontsize=8)
            ax3a.grid(True, alpha=0.3)
            ax3a.set_title('Transição Baseline → Atividade')
            
            # Rótulos
            ax3b.scatter(tempo[mask], labels[mask], c=labels[mask], 
                        cmap='tab10', s=15, alpha=0.8)
            ax3b.axvspan(tempo[idx_ini], tempo[idx_fim], color='green', alpha=0.2)
            ax3b.axvline(t_inicio, color='red', linestyle='--', linewidth=2.5)
            ax3b.set_xlabel('Tempo (s)')
            ax3b.set_ylabel('Cluster')
            ax3b.set_yticks(range(n_clusters))
            ax3b.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig3)
        
        # ========================================================
        # DOWNLOAD
        # ========================================================
        if idx_atividade is not None:
            st.markdown("---")
            
            resultado = pd.DataFrame({
                'Parametro': [
                    'Baseline inicio (s)', 'Baseline fim (s)',
                    'Estado dominante baseline (cluster)',
                    'Inicio atividade (s)', 'Fim janela (+30s)',
                    'Duracao janela (s)'
                ],
                'Valor': [
                    tempo[idx_ini], tempo[idx_fim], est_baseline,
                    t_inicio, t_fim, t_fim - t_inicio
                ]
            })
            csv = resultado.to_csv(index=False).encode('utf-8')
            st.download_button("💾 Baixar resultados", csv, 
                               "resultados_sitstand.csv", "text/csv")

else:
    st.info("👈 Faça upload de um arquivo para iniciar a análise.")
    
    st.markdown("""
    ### 📋 Algoritmo Implementado
    
    1. **K-Means**: Separa o sinal em 5 estados (clusters)
    2. **Busca da Baseline**: Janela de 500 ms com **menor variabilidade no SINAL ORIGINAL**
    3. **Estado Dominante**: Cluster mais frequente na janela de baseline
    4. **Detecção Anterógrada**: A partir do fim da baseline, busca a primeira sequência 
       de 5 amostras em estados **superiores** ao dominante da baseline
    5. **Janela Final**: Início da atividade + 30 segundos
    """)
