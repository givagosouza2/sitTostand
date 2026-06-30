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
    page_title="Sit-to-Stand - Detecção de Picos",
    page_icon="📊",
    layout="wide"
)

# ============================================================
# FUNÇÕES DE PROCESSAMENTO DO SINAL
# ============================================================
@st.cache_data
def carregar_dados(arquivo):
    """Carrega o CSV do acelerômetro."""
    try:
        df = pd.read_csv(arquivo, sep=';')
        df.columns = ['tempo_ms', 'X', 'Y', 'Z']
        df['tempo_s'] = df['tempo_ms'] / 1000.0
        return df
    except Exception as e:
        st.error(f"Erro ao carregar arquivo: {e}")
        return None


def interpolar_100hz(df, fs_alvo=100):
    """Interpola para frequência uniforme."""
    t_original = df['tempo_s'].values
    t_novo = np.arange(t_original[0], t_original[-1], 1.0/fs_alvo)
    df_interp = pd.DataFrame({'tempo_s': t_novo})
    for eixo in ['X', 'Y', 'Z']:
        f_interp = interp1d(t_original, df[eixo].values,
                            kind='linear', fill_value='extrapolate')
        df_interp[eixo] = f_interp(t_novo)
    return df_interp, fs_alvo


def aplicar_detrend(df, eixos=['X', 'Y', 'Z']):
    """Remove tendência linear."""
    df_detrend = df.copy()
    for eixo in eixos:
        df_detrend[eixo] = signal.detrend(df[eixo].values, type='linear')
    return df_detrend


def filtrar_sinal(df, fs, fc=8.0, ordem=4, eixos=['X', 'Y', 'Z']):
    """Filtro Butterworth passa-baixa."""
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
    """Classifica o sinal em clusters usando KMeans."""
    X = sinal.reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X)
    centers = kmeans.cluster_centers_.flatten()
    ordem = np.argsort(centers)
    centers_ordenados = centers[ordem]
    mapa = {antigo: novo for novo, antigo in enumerate(ordem)}
    labels_ordenados = np.array([mapa[l] for l in labels])
    return labels_ordenados, centers_ordenados


# ============================================================
# DETECÇÃO DA BASELINE (MENOR VARIABILIDADE)
# ============================================================
def encontrar_baseline(sinal, fs, janela_ms=500):
    """Encontra janela de 500 ms com menor variância no sinal."""
    janela_n = int(fs * janela_ms / 1000.0)
    variancias = []
    for i in range(len(sinal) - janela_n):
        variancias.append(np.var(sinal[i:i+janela_n]))
    idx_min = np.argmin(variancias)
    return idx_min, idx_min + janela_n


# ============================================================
# DETECÇÃO DO INÍCIO DA ATIVIDADE
# ============================================================
def detectar_inicio_atividade(labels, idx_fim_baseline, estado_baseline, 
                                sequencia_min=5):
    """Busca sequência de valores em estados superiores ao baseline."""
    for i in range(idx_fim_baseline, len(labels) - sequencia_min + 1):
        sequencia = labels[i:i+sequencia_min]
        if all(s > estado_baseline for s in sequencia):
            return i
    return None


# ============================================================
# DETECÇÃO DE PICOS NA JANELA DE ANÁLISE
# ============================================================
def detectar_picos(sinal, fs, altura_min=0.1, distancia_min_s=0.5, 
                   prominencia_min=0.05):
    """
    Detecta picos no sinal usando scipy.signal.find_peaks.
    
    Parâmetros:
    -----------
    sinal : array do sinal filtrado
    fs : frequência de amostragem (Hz)
    altura_min : altura mínima do pico (g)
    distancia_min_s : distância mínima entre picos (segundos)
    prominencia_min : proeminência mínima do pico (g)
    
    Retorna:
    --------
    indices_picos : índices dos picos detectados
    propriedades : dicionário com propriedades dos picos
    """
    distancia_min_amostras = int(distancia_min_s * fs)
    
    indices_picos, propriedades = signal.find_peaks(
        sinal,
        height=altura_min,
        distance=distancia_min_amostras,
        prominence=prominencia_min
    )
    
    return indices_picos, propriedades


# ============================================================
# INTERFACE STREAMLIT
# ============================================================
st.title("📊 Sit-to-Stand — Detecção de Picos na Janela de Análise")
st.markdown("---")

# --- UPLOAD DO ARQUIVO ---
st.sidebar.header("⚙️ Configurações")
arquivo = st.sidebar.file_uploader(
    "📁 Carregue o arquivo do acelerômetro",
    type=['txt', 'csv']
)

# --- PARÂMETROS DE PROCESSAMENTO ---
st.sidebar.markdown("---")
st.sidebar.subheader("🔧 Processamento do Sinal")
fc = st.sidebar.slider("Frequência de corte (Hz)", 1.0, 20.0, 8.0, 0.5)
fs_alvo = st.sidebar.slider("Frequência de amostragem (Hz)", 50, 200, 100, 10)

# --- PARÂMETROS DE DETECÇÃO DE BASELINE ---
st.sidebar.markdown("---")
st.sidebar.subheader("📏 Detecção de Baseline")
janela_ms = st.sidebar.slider("Janela baseline (ms)", 200, 1000, 500, 50)
n_clusters = st.sidebar.slider("Número de clusters", 3, 10, 5, 1)
sequencia_min = st.sidebar.slider("Sequência mínima p/ início", 3, 10, 5, 1)
duracao_janela = st.sidebar.slider("Duração da janela (s)", 5.0, 120.0, 30.0, 1.0)

# --- PARÂMETROS DE DETECÇÃO DE PICOS ---
st.sidebar.markdown("---")
st.sidebar.subheader("📈 Detecção de Picos")
altura_min = st.sidebar.slider("Altura mínima do pico (g)", 0.01, 0.5, 0.1, 0.01)
distancia_min_s = st.sidebar.slider("Distância mínima entre picos (s)", 
                                     0.2, 2.0, 0.5, 0.1)
prominencia_min = st.sidebar.slider("Proeminência mínima (g)", 
                                     0.01, 0.3, 0.05, 0.01)

# ============================================================
# PROCESSAMENTO
# ============================================================
if arquivo is not None:
    with st.spinner("⏳ Processando..."):
        # Pipeline completo
        df = carregar_dados(arquivo)
        df_proc, fs = interpolar_100hz(df, fs_alvo=fs_alvo)
        df_proc = aplicar_detrend(df_proc)
        df_proc = filtrar_sinal(df_proc, fs, fc=fc)
        
        sinal = df_proc['Y'].values
        tempo = df_proc['tempo_s'].values
        
        # KMeans
        labels, centers = classificar_kmeans(sinal, n_clusters=n_clusters)
        
        # Baseline
        idx_ini_base, idx_fim_base = encontrar_baseline(sinal, fs, janela_ms)
        estado_baseline = int(np.bincount(labels[idx_ini_base:idx_fim_base]).argmax())
        
        # Início da atividade
        idx_atividade = detectar_inicio_atividade(
            labels, idx_fim_base, estado_baseline, sequencia_min
        )
        
        # Janela de análise
        if idx_atividade is not None:
            t_inicio = tempo[idx_atividade]
            t_fim = min(t_inicio + duracao_janela, tempo[-1])
            
            # Máscara da janela
            mask_janela = (tempo >= t_inicio) & (tempo <= t_fim)
            sinal_janela = sinal[mask_janela]
            tempo_janela = tempo[mask_janela]
            idx_ini_janela = np.where(mask_janela)[0][0]
            idx_fim_janela = np.where(mask_janela)[0][-1]
            
            # DETECÇÃO DE PICOS NA JANELA
            indices_picos_janela, props_picos = detectar_picos(
                sinal_janela, fs,
                altura_min=altura_min,
                distancia_min_s=distancia_min_s,
                prominencia_min=prominencia_min
            )
            
            # Converter índices locais (da janela) para índices globais
            indices_picos_globais = indices_picos_janela + idx_ini_janela
            tempos_picos = tempo[indices_picos_globais]
            valores_picos = sinal[indices_picos_globais]
            
            # ========================================================
            # RESULTADOS - MARCOS TEMPORAIS
            # ========================================================
            st.success("✅ Análise concluída!")
            
            st.markdown("### 🎯 Marcos Temporais")
            col1, col2, col3 = st.columns(3)
            col1.metric("🟢 Início Atividade", f"{t_inicio:.2f} s")
            col2.metric("🔴 Fim da Janela", f"{t_fim:.2f} s")
            col3.metric("⏱️ Duração", f"{duracao_janela:.1f} s")
            
            # ========================================================
            # RESULTADOS - PICOS DETECTADOS
            # ========================================================
            st.markdown("### 📈 Picos Detectados na Janela")
            
            num_picos = len(indices_picos_globais)
            
            col_p1, col_p2, col_p3, col_p4 = st.columns(4)
            col_p1.metric("🔺 Nº de Picos", num_picos)
            
            if num_picos > 0:
                # Repetições = picos - 1 (cada subida completa conta como 1 rep)
                repeticoes = num_picos
                col_p2.metric("🔄 Repetições", repeticoes)
                
                # Cadência (picos por minuto)
                cadencia = (num_picos / duracao_janela) * 60
                col_p3.metric("⚡ Cadência", f"{cadencia:.1f} picos/min")
                
                # Tempo médio entre picos
                if num_picos > 1:
                    intervalos = np.diff(tempos_picos)
                    tempo_medio = np.mean(intervalos)
                    col_p4.metric("⏲️ Intervalo Médio", f"{tempo_medio:.2f} s")
                else:
                    col_p4.metric("⏲️ Intervalo Médio", "N/A")
                
                # ========================================================
                # TABELA DE PICOS
                # ========================================================
                st.markdown("### 📋 Detalhes dos Picos")
                
                df_picos = pd.DataFrame({
                    'Pico #': range(1, num_picos + 1),
                    'Tempo (s)': np.round(tempos_picos, 3),
                    'Aceleração (g)': np.round(valores_picos, 4),
                    'Tempo relativo (s)': np.round(tempos_picos - t_inicio, 3)
                })
                
                # Adicionar intervalos entre picos
                if num_picos > 1:
                    intervalos = np.diff(tempos_picos)
                    df_picos['Intervalo (s)'] = [np.nan] + np.round(intervalos, 3).tolist()
                else:
                    df_picos['Intervalo (s)'] = [np.nan]
                
                st.dataframe(df_picos, use_container_width=True, hide_index=True)
                
                # ========================================================
                # ESTATÍSTICAS DOS PICOS
                # ========================================================
                st.markdown("### 📊 Estatísticas dos Picos")
                
                stat1, stat2, stat3, stat4 = st.columns(4)
                stat1.metric("Aceleração Média", 
                            f"{np.mean(valores_picos):.3f} g")
                stat2.metric("Aceleração Máxima", 
                            f"{np.max(valores_picos):.3f} g")
                stat3.metric("Aceleração Mínima", 
                            f"{np.min(valores_picos):.3f} g")
                stat4.metric("Desvio Padrão", 
                            f"{np.std(valores_picos):.3f} g")
                
                if num_picos > 1:
                    intervalos = np.diff(tempos_picos)
                    stat5, stat6, stat7, stat8 = st.columns(4)
                    stat5.metric("Intervalo Mínimo", 
                                f"{np.min(intervalos):.2f} s")
                    stat6.metric("Intervalo Máximo", 
                                f"{np.max(intervalos):.2f} s")
                    stat7.metric("Intervalo Médio", 
                                f"{np.mean(intervalos):.2f} s")
                    stat8.metric("Variabilidade (DP)", 
                                f"{np.std(intervalos):.2f} s")
                
            else:
                st.warning("⚠️ Nenhum pico foi detectado com os parâmetros atuais. "
                          "Tente reduzir a altura mínima ou a proeminência.")
            
            # ========================================================
            # GRÁFICO 1: SINAL COMPLETO COM MARCOS
            # ========================================================
            st.markdown("---")
            st.markdown("### 📈 Sinal Completo com Marcos Temporais")
            
            fig, ax = plt.subplots(figsize=(14, 5))
            
            # Sinal completo (cinza claro)
            ax.plot(tempo, sinal, color='lightgray', linewidth=0.5, 
                   label='Sinal completo')
            
            # Janela de análise destacada
            ax.axvspan(t_inicio, t_fim, color='gold', alpha=0.2, 
                      label=f'Janela de análise ({duracao_janela:.0f}s)')
            
            # Baseline
            ax.axvspan(tempo[idx_ini_base], tempo[idx_fim_base], 
                      color='green', alpha=0.3, label='Baseline')
            
            # Linhas de marco
            ax.axvline(t_inicio, color='green', linestyle='--', linewidth=2,
                      label=f'Início = {t_inicio:.2f} s')
            ax.axvline(t_fim, color='red', linestyle='--', linewidth=2,
                      label=f'Fim = {t_fim:.2f} s')
            
            ax.set_xlabel('Tempo (s)', fontsize=12)
            ax.set_ylabel('Aceleração Y (g)', fontsize=12)
            ax.set_title('Visão Geral — Sit-to-Stand', fontsize=13, fontweight='bold')
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            
            # ========================================================
            # GRÁFICO 2: JANELA DE ANÁLISE COM PICOS
            # ========================================================
            st.markdown("### 🔍 Janela de Análise com Picos Detectados")
            
            fig2, ax2 = plt.subplots(figsize=(14, 6))
            
            # Sinal da janela
            ax2.plot(tempo_janela, sinal_janela, color='steelblue', 
                    linewidth=1.2, label='Sinal Y (filtrado)')
            
            # Linha zero
            ax2.axhline(0, color='gray', linestyle=':', linewidth=0.8)
            
            # Linha da média
            ax2.axhline(np.mean(sinal_janela), color='orange', 
                       linestyle='--', linewidth=1, alpha=0.7,
                       label=f'Média = {np.mean(sinal_janela):.3f} g')
            
            # Marcar picos
            if num_picos > 0:
                ax2.scatter(tempos_picos, valores_picos, 
                           color='red', s=100, marker='^', 
                           edgecolors='darkred', linewidths=1.5,
                           zorder=5, label=f'Picos detectados (n={num_picos})')
                
                # Numerar os picos
                for i, (t, v) in enumerate(zip(tempos_picos, valores_picos)):
                    ax2.annotate(f'{i+1}', 
                               (t, v), 
                               textcoords="offset points", 
                               xytext=(0, 10), 
                               ha='center', fontsize=9, 
                               fontweight='bold', color='darkred')
                
                # Linhas verticais nos picos
                for t in tempos_picos:
                    ax2.axvline(t, color='red', linestyle=':', 
                               linewidth=0.5, alpha=0.4)
            
            ax2.set_xlabel('Tempo (s)', fontsize=12)
            ax2.set_ylabel('Aceleração Y (g)', fontsize=12)
            ax2.set_title(f'Janela de Análise: {t_inicio:.2f} s → {t_fim:.2f} s', 
                         fontsize=13, fontweight='bold')
            ax2.legend(loc='best', fontsize=10)
            ax2.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig2)
            
            # ========================================================
            # GRÁFICO 3: INTERVALOS ENTRE PICOS
            # ========================================================
            if num_picos > 1:
                st.markdown("### ⏱️ Intervalos Entre Picos Consecutivos")
                
                intervalos = np.diff(tempos_picos)
                
                fig3, ax3 = plt.subplots(figsize=(12, 4))
                barras = ax3.bar(range(1, len(intervalos) + 1), intervalos, 
                                color='steelblue', edgecolor='navy', alpha=0.8)
                
                # Linha da média
                ax3.axhline(np.mean(intervalos), color='red', 
                           linestyle='--', linewidth=2,
                           label=f'Média = {np.mean(intervalos):.2f} s')
                
                # Adicionar valores nas barras
                for i, (barra, val) in enumerate(zip(barras, intervalos)):
                    ax3.text(barra.get_x() + barra.get_width()/2, 
                            barra.get_height() + 0.01,
                            f'{val:.2f}', 
                            ha='center', va='bottom', fontsize=9)
                
                ax3.set_xlabel('Intervalo (entre pico N e N+1)', fontsize=12)
                ax3.set_ylabel('Duração (s)', fontsize=12)
                ax3.set_title('Tempo Entre Picos Consecutivos', 
                             fontsize=13, fontweight='bold')
                ax3.legend(loc='best')
                ax3.grid(True, alpha=0.3, axis='y')
                ax3.set_xticks(range(1, len(intervalos) + 1))
                plt.tight_layout()
                st.pyplot(fig3)
            
            # ========================================================
            # DOWNLOAD
            # ========================================================
            st.markdown("---")
            st.markdown("### 💾 Download dos Resultados")
            
            # Dados da janela
            df_saida_janela = pd.DataFrame({
                'tempo_s': tempo_janela,
                'aceleracao_Y_g': sinal_janela
            })
            
            # Dados dos picos
            if num_picos > 0:
                df_saida_picos = pd.DataFrame({
                    'pico_numero': range(1, num_picos + 1),
                    'tempo_s': tempos_picos,
                    'tempo_relativo_s': tempos_picos - t_inicio,
                    'aceleracao_g': valores_picos
                })
                if num_picos > 1:
                    df_saida_picos['intervalo_antes_s'] = [np.nan] + list(np.diff(tempos_picos))
            
            # Relatório geral
            df_relatorio = pd.DataFrame({
                'Parametro': [
                    'Inicio atividade (s)', 'Fim da janela (s)',
                    'Duracao janela (s)', 'Numero de picos',
                    'Cadencia (picos/min)', 'Acel media picos (g)',
                    'Acel max picos (g)', 'Acel min picos (g)'
                ],
                'Valor': [
                    t_inicio, t_fim, duracao_janela, num_picos,
                    (num_picos / duracao_janela) * 60 if num_picos > 0 else 0,
                    np.mean(valores_picos) if num_picos > 0 else 0,
                    np.max(valores_picos) if num_picos > 0 else 0,
                    np.min(valores_picos) if num_picos > 0 else 0
                ]
            })
            
            col_dl1, col_dl2, col_dl3 = st.columns(3)
            
            with col_dl1:
                st.download_button(
                    "📊 Dados da janela",
                    df_saida_janela.to_csv(index=False).encode('utf-8'),
                    "janela_analise.csv",
                    "text/csv"
                )
            
            with col_dl2:
                if num_picos > 0:
                    st.download_button(
                        "🔺 Dados dos picos",
                        df_saida_picos.to_csv(index=False).encode('utf-8'),
                        "picos_detectados.csv",
                        "text/csv"
                    )
            
            with col_dl3:
                st.download_button(
                    "📄 Relatório geral",
                    df_relatorio.to_csv(index=False).encode('utf-8'),
                    "relatorio_sitstand.csv",
                    "text/csv"
                )

else:
    st.info("👈 Faça upload de um arquivo na barra lateral para começar a análise.")
    
    st.markdown("""
    ### 📋 Pipeline Completo
    
    1. **Processamento do sinal**
       - Interpolação para 100 Hz
       - Detrend linear
       - Filtro Butterworth passa-baixa (8 Hz)
    
    2. **Detecção da baseline**
       - KMeans com 5 clusters
       - Janela de 500 ms com menor variância
       - Estado dominante = cluster mais frequente
    
    3. **Detecção do início da atividade**
       - Sequência de 5 amostras em clusters superiores ao baseline
    
    4. **Janela de análise**
       - Início + 30 segundos (ajustável)
    
    5. **Detecção de picos** 🆕
       - `scipy.signal.find_peaks` com parâmetros ajustáveis
       - Altura mínima, distância mínima e proeminência
    
    ### 📊 Métricas Calculadas
    
    - Número de picos detectados
    - Cadência (picos por minuto)
    - Intervalo médio entre picos
    - Variabilidade dos intervalos (DP)
    - Estatísticas de aceleração nos picos
    
    ### 🎯 Interpretação
    
    No teste sentar-levantar, cada **pico positivo** no eixo Y 
    (aceleração vertical) geralmente corresponde à fase de **extensão** 
    (subida), quando o centro de massa acelera para cima.
    """)
