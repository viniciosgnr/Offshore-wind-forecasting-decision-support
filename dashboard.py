# dashboard.py - VERSÃO FINAL COM AUDITORIA DA IA

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import sqrt
import os
from openai import OpenAI
import joblib

# ===================================================================
# PASSO 1: DEFINIÇÃO DAS FERRAMENTAS DO AGENTE
# ===================================================================
import json # Garanta que este import está no topo do seu arquivo

# Ferramenta 4: Rodar a otimização de estratégia
def run_strategy_optimization_tool():
    """Executa a otimização de estratégia e retorna um resumo em JSON."""
    try:
        # Usa as variáveis que serão salvas no session_state
        cost_matrix, threshold_range, horizon_range, best_strategy = optimize_strategy(
            st.session_state.y_test_inv, st.session_state.all_predictions, st.session_state.cap
        )
        st.session_state.best_strategy_results = best_strategy
        st.success("Agent has successfully completed the strategy optimization!")
        return json.dumps(best_strategy)
    except Exception as e:
        return json.dumps({"error": str(e)})

def run_global_strategy_optimization_tool():
    """
    Executa a otimização GLOBAL 3D (incluindo o P10 Safety Factor) 
    e retorna um resumo em JSON. Esta ferramenta só deve ser usada para o 
    modelo 'CEEMDAN-EWT-TFT-Aggregator (Interp)'.
    """
    try:
        # Chama a função de otimização 3D
        best_strategy = optimize_strategy_3d(
            st.session_state.y_test_inv, st.session_state.all_predictions, st.session_state.cap
        )
        st.session_state.best_strategy_results = best_strategy
        st.success("Agent has successfully completed the GLOBAL 3D strategy optimization!")
        # Retorna o resultado completo, incluindo o safety_factor
        return json.dumps(best_strategy)
    except Exception as e:
        return json.dumps({"error": str(e)})
# ===================================================================

# CÓDIGO CORRIGIDO
def apply_new_parameters_tool(threshold_percent, horizon):
    """Aplica novos parâmetros de estratégia ao session_state e força o rerun do app."""
    try:
        # >>>>> CORREÇÃO: Converte o valor para inteiro antes de salvar <<<<<
        st.session_state['op_threshold_percent'] = int(threshold_percent)
        st.session_state['strategy_horizon'] = int(horizon) # Boa prática fazer o mesmo para o horizonte
        st.toast(f"Agent is applying new parameters: Threshold={threshold_percent}%, Horizon=t+{horizon}.")
        st.rerun()
        return json.dumps({"status": "success", "message": "Parameters applied. App is reloading."})
    except Exception as e:
        return json.dumps({"error": str(e)})


# Ferramenta 6: Gerar um relatório
def generate_summary_report_tool(start_time_step, end_time_step):
    """Gera um relatório de resumo em Markdown para um período."""
    try:
        end_time_step = min(end_time_step, len(st.session_state.y_test_inv) - 1)
        start_time_step = max(0, start_time_step)
        costs_df = calculate_costs(
            st.session_state.y_test_inv[start_time_step:end_time_step],
            {q: p[start_time_step:end_time_step] for q, p in st.session_state.all_predictions.items()},
            st.session_state.operational_threshold,
            st.session_state.strategy_horizon
        )
        ori_results = costs_df.loc['Preditiva (ORI)']
        report = f"""### Operational Summary Report (Steps {start_time_step}-{end_time_step})
- **Total Cost:** ${ori_results['Custo Total (USD)']:,.2f}
- **Blackout Events:** {int(ori_results['Eventos de Blackout'])}
- **Generator Starts:** {int(ori_results['Número de Partidas'])}"""
        return report
    except Exception as e:
        return json.dumps({"error": str(e)})
# ===================================================================



def plot_attention_interpretation(attention_weights, y_test_inv, current_time_step, look_back):
    """
    [VERSION 8 - VISUAL NORMALIZATION]
    Applies Min-Max scaling to the attention vector to visually enhance the differences in importance.
    """
    try:
        # 1. Process attention weights
        avg_attention = np.mean(attention_weights[current_time_step], axis=0)
        attention_vector = avg_attention[-1, :]
        
        # >>>>> NOVA ETAPA: NORMALIZAÇÃO MIN-MAX PARA VISUALIZAÇÃO <<<<<
        min_val = np.min(attention_vector)
        max_val = np.max(attention_vector)
        # Evita divisão por zero se todos os valores forem iguais
        if max_val > min_val:
            normalized_attention = (attention_vector - min_val) / (max_val - min_val)
        else:
            normalized_attention = np.ones_like(attention_vector) * 0.5 # Se todos forem iguais, pinta com cor média

        # 2. Prepare the input data series
        if current_time_step < look_back:
            st.warning("Not enough history to display the input series for this early time step.")
            return None

        start_index = current_time_step - look_back
        end_index = current_time_step
        input_series = y_test_inv[start_index:end_index, 0]

        # 3. Prepare chart axes
        x_labels_past = [f"t-{look_back-i}" for i in range(look_back)]
        
        # 4. Create the figure with subplots
        from plotly.subplots import make_subplots
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.2, 0.8])

        # 5. Add the Heatmap (usando o vetor NORMALIZADO)
        fig.add_trace(go.Heatmap(
            z=[normalized_attention], # <<< MUDANÇA AQUI
            x=x_labels_past, 
            y=['Importance'],
            colorscale='Viridis', 
            showscale=True,
            colorbar=dict(title=dict(text="Normalized Importance", side="right"), thickness=15)), row=1, col=1)

        # 6. Add the Line Chart for the input series
        fig.add_trace(go.Scatter(
            x=x_labels_past, y=input_series, mode='lines+markers', name='Input Series (Actual)',
            line=dict(color='royalblue')
        ), row=2, col=1)

        # 7. Finalize Layout (in English)
        fig.update_layout(
            title_text=f"Attention Analysis: Input for Forecast at Point t={current_time_step}",
            height=450,
            plot_bgcolor='white',
            yaxis2_title="Power (MW)",
            showlegend=False
        )
        fig.update_xaxes(title_text="Past Timesteps (Model Input)", row=2, col=1)
        
        return fig

    except Exception as e:
        print(f"Error plotting attention: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None

# Adicione esta função ao seu arquivo dashboard.py, junto com as outras funções de plotagem.

import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd

import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd

def plot_real_decomposition_3d(components_df, plane_spacing_multiplier=5.0, downsample_factor=1):
    """
    Creates a clear 3D plot of decomposed signal components with significant
    visual separation between each component's plane.

    Args:
        components_df (pd.DataFrame): DataFrame where each column is a component.
        plane_spacing_multiplier (float): A large multiplier to increase the visual
                                          distance between component planes on the Y-axis.
                                          Values between 2.0 and 5.0 are recommended.
        downsample_factor (int): Factor to reduce point density for clarity (1 = no downsampling).
    """
    fig = go.Figure()
    labels = components_df.columns
    colors = px.colors.qualitative.Plotly

    # Downsample for visual clarity if needed
    if downsample_factor > 1:
        plot_df = components_df.iloc[::downsample_factor, :]
    else:
        plot_df = components_df

    num_points = len(plot_df)

    # ==================================================================
    # LÓGICA CORRETA - AUMENTANDO O ESPAÇAMENTO NO EIXO Y
    # ==================================================================
    for i, label in enumerate(labels):
        x_axis_data = plot_df.index
        # O espaçamento é aplicado no eixo Y, que separa os componentes
        y_axis_data = np.full(num_points, i * plane_spacing_multiplier)
        # A amplitude é plotada no eixo Z, como esperado
        z_axis_data = plot_df[label].values

        fig.add_trace(go.Scatter3d(
            x=x_axis_data,
            y=y_axis_data,
            z=z_axis_data,
            mode='lines',
            name=str(label),
            line=dict(width=3, color=colors[i % len(colors)])
        ))
    # ==================================================================

    tick_labels = []
    for i, label in enumerate(labels):
        if isinstance(label, int) or str(label).isdigit():
             if i == len(labels) - 1:
                 tick_labels.append('Res')
             else:
                 tick_labels.append(f'IMF{i+1}')
        else:
             tick_labels.append(str(label))

    fig.update_layout(
        title="Signal Decomposition Visualization (CEEMDAN-EWT)",
        scene=dict(
            xaxis_title='Sample Point',
            yaxis_title='Component',
            zaxis_title='Amplitude',

            # "Estica" o eixo Y para dar espaço aos planos e aos rótulos
            aspectratio=dict(x=1.5, y=4, z=1),

            yaxis=dict(
                # Os ticks acompanham o novo espaçamento
                tickvals=[i * plane_spacing_multiplier for i in range(len(labels))],
                ticktext=tick_labels,
                backgroundcolor="rgba(230, 230, 230, 0.7)",
                gridcolor="white",
            ),
            zaxis=dict(backgroundcolor="rgba(230, 230, 230, 0.7)", gridcolor="white"),
            xaxis=dict(backgroundcolor="rgba(230, 230, 230, 0.7)", gridcolor="white"),
            # Câmera em um ângulo clássico para melhor visualização
            camera=dict(eye=dict(x=-1.8, y=-1.8, z=1.2))
        ),
        legend=dict(x=0.01, y=0.98),
        height=800,
        margin=dict(l=0, r=0, b=0, t=40)
    )
    return fig







def load_decomposition_data(selected_model_name):
    """
    Carrega o DataFrame de componentes decompostas com base no nome do modelo selecionado.
    Retorna o DataFrame se encontrado, ou None caso contrário.
    """
    # Mapeia o nome do modelo no dashboard para o diretório onde o arquivo .gz está salvo
    model_dir_map = {
        "CEEMDAN-EWT-LSTM": "saved_models/proposed_method",
        "CEEMDAN-EWT-GRU": "saved_models/proposed_method_gru", 
        "CEEMDAN-EWT-BiLSTM": "saved_models/proposed_method_bilstm",
        "CEEMDAN-EWT-BiLSTM-Attn": "saved_models/proposed_method_bilstm_att",
        "CEEMDAN-EWT-CNN-BiLSTM": "saved_models/proposed_method_cnn_bilstm",
        "CEEMDAN-EWT-CNN-BiLSTM-Attn": "saved_models/proposed_method_cnn_bilstm_att",
        # O modelo TFT agregador usa um nome de arquivo diferente
        "CEEMDAN-EWT-TFT-Aggregator": "saved_models/proposed_method_tft_aggregator",
        "CEEMDAN-EWT-TFT-Aggregator (Interp)": "saved_models/proposed_method_tft_aggregator_interp"
    }

    if selected_model_name not in model_dir_map:
        return None # Não é um modelo de decomposição conhecido

    model_dir = model_dir_map[selected_model_name]
    
    # Define o nome do arquivo a ser procurado
    # O modelo TFT agregador tem um nome de arquivo diferente
    if "TFT-Aggregator" in selected_model_name:
        file_name = "features_df.gz"
    else:
        file_name = "decomposed_df.gz"
        
    file_path = os.path.join(model_dir, file_name)

    if os.path.exists(file_path):
        try:
            print(f"Carregando dados de decomposição de: {file_path}")
            components_df = joblib.load(file_path)
            return components_df
        except Exception as e:
            st.warning(f"Não foi possível carregar o arquivo de decomposição '{file_path}': {e}")
            return None
    else:
        st.warning(f"Arquivo de decomposição não encontrado em '{file_path}'. Execute o modelo em modo de treinamento primeiro.")
        return None



# ===================================================================
# INICIALIZAÇÃO E FUNÇÕES DO DASHBOARD
# ===================================================================
if "messages" not in st.session_state:
    st.session_state.messages = []
# >>>>> NOVO: Inicializa o estado para o contexto da auditoria <<<<<
if "last_context" not in st.session_state:
    st.session_state.last_context = None

# >>>>> NOVA MUDANÇA: Inicializa o estado para os resultados da otimização <<<<<
if "best_strategy_results" not in st.session_state:
    st.session_state.best_strategy_results = None


# Substitua a função get_llm_driven_proactive_alert pela versão final e definitiva

# ===================================================================
# FUNÇÃO PROACTIVE AGENT (VERSÃO FINAL E ROBUSTA COM LÓGICA EM PYTHON)
# ===================================================================
def get_llm_driven_proactive_alert(api_key, ori_history, kpi_data, threshold):
    """
    Decides if an alert is needed using Python logic, then uses the LLM
    to generate the correctly formatted natural language response.
    """
    # --- PASSO 1: LÓGICA DE DECISÃO EM PYTHON (100% CONFIÁVEL) ---
    is_critical_event = False
    alert_reason = ""

    # Condição 1: Mudança no nível de risco
    if len(ori_history) > 1 and ori_history[-1] != ori_history[-2]:
        is_critical_event = True
        alert_reason = f"Risk level changed from '{ori_history[-2]}' to '{ori_history[-1]}'."

    # Condição 2: Previsão P10 abaixo do limiar
    # Usamos um loop para encontrar a primeira violação e parar.
    if not is_critical_event: # Só checa se a primeira condição não foi atendida
        for horizon in [10, 20, 30]:
            p10_key = f'pred_low_t{horizon}'
            p10_value = kpi_data[p10_key]
            if p10_value < threshold:
                is_critical_event = True
                alert_reason = f"The Worst-Case Forecast for t+{horizon} min ({p10_value:.2f} MW) is below the {threshold:.2f} MW Operational Threshold."
                break # Encontrou a causa do alerta, não precisa checar mais

    # --- PASSO 2: CONSTRUÇÃO DO CONTEXTO E PROMPT PARA O LLM ---
    
    # O contexto de dados permanece o mesmo (totalmente em inglês)
    context_for_llm = f"""
### Continuous Surveillance Data
- Recent Risk History: {' -> '.join(ori_history)}
- CURRENT Risk Level: {ori_history[-1]}
- Worst-Case Forecast (t+10 min): {kpi_data['pred_low_t10']:.2f} MW
- Worst-Case Forecast (t+20 min): {kpi_data['pred_low_t20']:.2f} MW
- Worst-Case Forecast (t+30 min): {kpi_data['pred_low_t30']:.2f} MW
- Operational Threshold: {threshold:.2f} MW
- Median Forecast (t+10 min): {kpi_data['pred_median_t10']:.2f} MW
- Median Forecast (t+20 min): {kpi_data['pred_median_t20']:.2f} MW
- Median Forecast (t+30 min): {kpi_data['pred_median_t30']:.2f} MW
"""
    
    # O System Prompt agora é muito mais simples. Ele não decide, apenas formata.
    if is_critical_event:
        system_prompt = f"""
You are an 'Operational Surveillance Agent'. Your task is to format an alert message.
The critical event has already been identified. Your ONLY job is to present it clearly.

1.  Start the response with "ALERT:".
2.  State the exact reason for the alert, which is: "{alert_reason}"
3.  Conclude with a brief, one-sentence trend analysis based on the provided "Median Forecast" data.
"""
    else: # Não há evento crítico
        system_prompt = """
You are an 'Operational Surveillance Agent'. Your task is to provide a trend summary.
There are no critical events. Your ONLY job is to summarize the forecast trend.

1.  Start the response with "TREND:".
2.  Formulate an initial sentence summarizing the overall trend (RISING, FALLING, or STABLE) by comparing the "Median Forecast" at t+10 and t+30.
3.  Then, list the expected (median) and worst-case trajectory for each horizon (t+10, t+20, t+30) in a bulleted format.
"""

    # --- PASSO 3: CHAMADA AO LLM (AGORA APENAS PARA FORMATAÇÃO) ---
    client = OpenAI(api_key=api_key)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": context_for_llm} # O LLM usa o contexto para a análise de tendência
    ]
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini", 
            messages=messages, 
            temperature=0.1 
        )
        return response.choices[0].message.content
            
    except Exception as e:
        return f"Error in proactive agent analysis: {e}"





# Substitua sua função get_chatbot_response existente por esta

def get_chatbot_response(api_key, user_prompt, context_data):
    client = OpenAI(api_key=api_key)

    # --- 1. GET RECENT HISTORY (No changes needed here) ---
    start_idx_history = max(0, context_data['current_time_step'] - 4)
    end_idx_history = context_data['current_time_step'] + 1
    recent_ori_history = context_data['ori_levels'][start_idx_history:end_idx_history]
    ori_history_string = " -> ".join(recent_ori_history)

    # ==================================================================
    # CORREÇÃO AQUI: Traduzindo a construção da string de contexto
    # ==================================================================
    # Build the context string dynamically in English
    data_context_string = f"""
    ### Current Situation (Real-Time)
    - Recent Risk History (last 5 steps): {ori_history_string}
    - CURRENT Risk Level (ORI): {context_data['risk_level']}
    - Main Recommendation: {context_data['recommendation']}
    - CURRENT Operational Threshold: {context_data['current_kpis']['threshold']:.2f} MW
    - Actual Power (t+10 min): {context_data['current_kpis']['real_power']:.2f} MW
    - Median Forecast (t+10 min): {context_data['current_kpis']['pred_median']:.2f} MW
    - Worst Case Forecast (t+20 min): {context_data['current_kpis']['pred_low_20']:.2f} MW
    - Worst Case Forecast (t+30 min): {context_data['current_kpis']['pred_low_30']:.2f} MW
    """

    # Add optimization context if it exists (also in English)
    if context_data.get("optimization_results"):
        opt_results = context_data["optimization_results"]
        optimization_context_string = f"""
    ### Last Strategy Optimization Results
    - Minimum Cost Found: USD {opt_results['cost']:,.2f}
    - Best Operational Threshold: {opt_results['threshold_percent']:.1f}% of capacity
    - Best Trigger Horizon: t+{opt_results['horizon']} min
    - Blackouts with Optimal Strategy: {opt_results['blackouts']}
    - Generator Starts with Optimal Strategy: {opt_results['starts']}
    """
        data_context_string += optimization_context_string
    else:
        data_context_string += "\n- Strategy optimization has not been run yet."
    # ==================================================================

    # The system prompt also needs to be fully in English
    system_prompt = f"""
    You are a 'Strategic Operational Copilot', an expert AI assistant for offshore wind-connected FPSO operations.
    Your role is to answer the operator's questions clearly, objectively, and strategically.

    1.  **Analyze the 'Current Situation (Real-Time)'** to provide immediate operational recommendations, clearly listing all evaluated data. Use the 'Recent Risk History' to comment on trends.
    2.  **Use the 'Last Strategy Optimization Results'** to provide long-term insights and cost-reduction tips.
    3.  Concisely and directly cite the adjustments suggested by the Last Strategy Optimization Results. If the optimization has not been run, suggest running it.
    4.  Be concise and direct. Base ALL your answers on the provided context data.

    Current data context:
    {data_context_string}
    """

    messages = [{"role": "system", "content": system_prompt}]
    # Append the last 4 messages to maintain conversation context
    messages.extend(context_data["chat_history"][-4:]) 
    messages.append({"role": "user", "content": user_prompt})
    
    try:
        response = client.chat.completions.create(model="gpt-4o-mini", messages=messages, temperature=0.3)
        # Return the response and the context string used to generate it
        return response.choices[0].message.content, data_context_string
    except Exception as e:
        return f"Error contacting the Operational Copilot: {e}", data_context_string


try:
    from myfunctions import ann_quantile_model_with_rri, proposed_method, svr_model,rf_model, lstm_model, bilstm_model, bilstm_att_model,cnn_bilstm_model,cnn_bilstm_att_model,transformer_model,tft_model,proposed_method_bilstm,proposed_method_bilstm_att,proposed_method_cnn_bilstm,proposed_method_cnn_bilstm_att,proposed_method_tft_aggregator,proposed_method_gru,proposed_method_tft_aggregator_interp
except ImportError:
    st.error("O arquivo 'myfunctions.py' não foi encontrado.")
    st.stop()

st.set_page_config(page_title="Dashboard de Previsão Eólica Offshore", page_icon="🌬️", layout="wide")

# ... (todo o resto do seu código de carregamento de dados, modelos e custos permanece IDÊNTICO) ...
@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path, sep=';')
    df['Date'] = pd.to_datetime(df['time_stamp'], format="%Y-%m-%d %H:%M:%S")
    start_date = df['Date'].min()
    df['Month'] = (df['Date'].dt.year - start_date.year) * 12 + df['Date'].dt.month - start_date.month + 1
    df.rename(columns={'power_5_avg': 'LV ActivePower (kW)'}, inplace=True)
    new_data = df[['Month', 'Date', 'LV ActivePower (kW)', 'wind_speed_235_avg']].dropna()
    cap = new_data['LV ActivePower (kW)'].max()
    return new_data, cap

model_functions = {"SVR": svr_model,
                   "Random Forest": rf_model,
                   "ANN Simples": ann_quantile_model_with_rri,
                   "LSTM Simples": lstm_model,
                   "Bi-LSTM": bilstm_model,
                   "Bi-LSTM + Attention": bilstm_att_model,
                   "CNN + Bi-LSTM": cnn_bilstm_model,
                   "CNN + Bi-LSTM + Attn": cnn_bilstm_att_model,
                   "Transformer": transformer_model,
                   "TFT": tft_model,
                   "CEEMDAN-EWT-LSTM": proposed_method,
                   "CEEMDAN-EWT-GRU": proposed_method_gru,
                   "CEEMDAN-EWT-BiLSTM": proposed_method_bilstm,
                   "CEEMDAN-EWT-BiLSTM-Attn": proposed_method_bilstm_att,
                   "CEEMDAN-EWT-CNN-BiLSTM": proposed_method_cnn_bilstm,
                   "CEEMDAN-EWT-CNN-BiLSTM-Attn": proposed_method_cnn_bilstm_att,
                   "CEEMDAN-EWT-TFT-Aggregator": proposed_method_tft_aggregator,
                   "CEEMDAN-EWT-TFT-Aggregator (Interp)": proposed_method_tft_aggregator_interp}

@st.cache_data
def run_model(model_name, data, months, look_back, data_partition, cap, strategy_horizon):
    model_func = model_functions[model_name]
    with st.spinner(f"Carregando modelo '{model_name}' e executando inferência..."):
        # --- CORREÇÃO AQUI ---
        # 1. Recebe a saída completa da função do modelo, seja ela uma tupla de 2 ou 3 elementos.
        model_output = model_func(data, months, look_back, data_partition, cap, strategy_horizon=strategy_horizon, mode='inference')
    
    st.success(f"Inferência do modelo '{model_name}' concluída!")
    
    # 2. Retorna a tupla inteira. O desempacotamento será feito fora da função.
    return model_output


# Substitua a função existente em dashboard.py por esta

@st.cache_data
def optimize_strategy(y_test_inv, all_predictions, cap):
    """
    Executa uma varredura de parâmetros para encontrar a combinação ótima
    de limiar operacional e horizonte de gatilho que minimiza o custo total.
    
    Retorna:
        - Matriz de custos para o mapa de calor.
        - Listas de valores de limiar e horizonte testados.
        - Dicionário com a melhor estratégia encontrada (incluindo custos, blackouts e partidas).
    """
    print("--- EXECUTANDO OTIMIZADOR DE ESTRATÉGIA ---")
    
    threshold_percents = np.arange(0.10, 0.51, 0.02)
    horizon_options = [20, 30]

    cost_matrix = []
    best_strategy = {
        "cost": float('inf'),
        "threshold_percent": None,
        "horizon": None,
        # >>>>> NOVO: Campos para armazenar as métricas <<<<<
        "blackouts": None,
        "starts": None
    }

    for horizon in horizon_options:
        row = []
        for thresh_p in threshold_percents:
            current_threshold = thresh_p * cap
            
            cost_df = calculate_costs(y_test_inv, all_predictions, current_threshold, horizon)
            
            strategy_results = cost_df.loc['Preditiva (ORI)']
            total_cost = strategy_results['Custo Total (USD)']
            row.append(total_cost)
            
            if total_cost < best_strategy["cost"]:
                best_strategy["cost"] = total_cost
                best_strategy["threshold_percent"] = thresh_p * 100
                best_strategy["horizon"] = horizon
                # >>>>> NOVO: Salva as métricas da melhor estratégia <<<<<
                best_strategy["blackouts"] = strategy_results["Eventos de Blackout"]
                best_strategy["starts"] = strategy_results["Número de Partidas"]
        cost_matrix.append(row)
        
    print("--- OTIMIZAÇÃO CONCLUÍDA ---")
    return np.array(cost_matrix), threshold_percents * 100, horizon_options, best_strategy


@st.cache_data
def optimize_strategy_3d(y_test_inv, all_predictions_original, cap):
    """
    Executa uma varredura 3D para encontrar a combinação ótima de limiar, 
    horizonte E P10 Safety Factor.
    """
    print("--- EXECUTANDO OTIMIZADOR DE ESTRATÉGIA 3D (Threshold, Horizon, Safety Factor) ---")
    
    # 1. DEFINIR OS ESPAÇOS DE BUSCA
    threshold_percents = np.arange(0.10, 0.51, 0.05)
    horizon_options = [20, 30]
    safety_factor_options = np.arange(0.50, 1.01, 0.01)

    best_strategy = {
        "cost": float('inf'),
        "threshold_percent": None,
        "horizon": None,
        "safety_factor": None, # Novo campo
        "blackouts": None,
        "starts": None
    }

    # 2. LOOP TRIPLO PARA A BUSCA 3D
    for sf in safety_factor_options:
        all_predictions_adjusted = {q: p.copy() for q, p in all_predictions_original.items()}
        if sf < 1.0:
            all_predictions_adjusted[0.1][:, 1:] *= sf
        
        print(f"\nTesting with Safety Factor: {sf:.2f}")

        for horizon in horizon_options:
            for thresh_p in threshold_percents:
                current_threshold = thresh_p * cap
                cost_df = calculate_costs(y_test_inv, all_predictions_adjusted, current_threshold, horizon)
                strategy_results = cost_df.loc['Preditiva (ORI)']
                total_cost = strategy_results['Custo Total (USD)']
                
                if total_cost < best_strategy["cost"]:
                    best_strategy["cost"] = total_cost
                    best_strategy["threshold_percent"] = thresh_p * 100
                    best_strategy["horizon"] = horizon
                    best_strategy["safety_factor"] = sf
                    best_strategy["blackouts"] = strategy_results["Eventos de Blackout"]
                    best_strategy["starts"] = strategy_results["Número de Partidas"]
    
    print("\n--- OTIMIZAÇÃO 3D CONCLUÍDA ---")
    return best_strategy
# ===================================================================

@st.cache_data
def calculate_costs(y_test_inv, all_predictions, operational_threshold, strategy_horizon):
    COST_PER_HOUR_GENERATOR, COST_PER_STARTUP, COST_PER_BLACKOUT_EVENT = 500, 2000, 100000
    TIME_STEP_HOURS, STARTUP_TIME_STEPS = 10 / 60, 2
    results = {}
    # Estratégia Reativa
    gen_on_reactive = [False] * len(y_test_inv); starts_reactive = 0; blackouts_reactive = 0
    for t in range(1, len(y_test_inv)):
        if y_test_inv[t-1, 0] < operational_threshold and not gen_on_reactive[t-1]:
            starts_reactive += 1
            for s_step in range(STARTUP_TIME_STEPS):
                if t + s_step < len(y_test_inv) and y_test_inv[t + s_step, 0] < operational_threshold: blackouts_reactive += 1
        if t >= STARTUP_TIME_STEPS and y_test_inv[t - STARTUP_TIME_STEPS, 0] < operational_threshold: gen_on_reactive[t] = True
        if gen_on_reactive[t-1] and y_test_inv[t-1, 0] > operational_threshold * 1.1: gen_on_reactive[t] = False
        elif gen_on_reactive[t-1]: gen_on_reactive[t] = True
    cost_fuel_reactive = sum(gen_on_reactive) * TIME_STEP_HOURS * COST_PER_HOUR_GENERATOR; cost_starts_reactive = starts_reactive * COST_PER_STARTUP; cost_blackout_reactive = blackouts_reactive * COST_PER_BLACKOUT_EVENT
    results['Reativa (Sem Previsão)'] = {"Custo Total (USD)": cost_fuel_reactive + cost_starts_reactive + cost_blackout_reactive, "Horas de Gerador Ligado": sum(gen_on_reactive) * TIME_STEP_HOURS, "Número de Partidas": starts_reactive, "Eventos de Blackout": blackouts_reactive}
    # Estratégias Preditivas
    p_low_t20, p_low_t30 = all_predictions[0.1][:, 1], all_predictions[0.1][:, 2]
    trigger_ori = p_low_t20 if strategy_horizon == 20 else p_low_t30
    gen_on_ori = [trigger_ori[t] < operational_threshold for t in range(len(trigger_ori))]
    starts_ori = sum(1 for t in range(1, len(gen_on_ori)) if gen_on_ori[t] and not gen_on_ori[t-1]); blackouts_ori = sum(1 for t in range(STARTUP_TIME_STEPS, len(y_test_inv)) if y_test_inv[t, 0] < operational_threshold and not gen_on_ori[t - STARTUP_TIME_STEPS])
    cost_fuel_ori = sum(gen_on_ori) * TIME_STEP_HOURS * COST_PER_HOUR_GENERATOR; cost_starts_ori = starts_ori * COST_PER_STARTUP; cost_blackout_ori = blackouts_ori * COST_PER_BLACKOUT_EVENT
    results['Preditiva (ORI)'] = {"Custo Total (USD)": cost_fuel_ori + cost_starts_ori + cost_blackout_ori, "Horas de Gerador Ligado": sum(gen_on_ori) * TIME_STEP_HOURS, "Número de Partidas": starts_ori, "Eventos de Blackout": blackouts_ori}
    gen_on_conservative = [(p_low_t20[t] < operational_threshold) or (p_low_t30[t] < operational_threshold) for t in range(len(p_low_t20))]
    starts_conservative = sum(1 for t in range(1, len(gen_on_conservative)) if gen_on_conservative[t] and not gen_on_conservative[t-1]); blackouts_conservative = sum(1 for t in range(STARTUP_TIME_STEPS, len(y_test_inv)) if y_test_inv[t, 0] < operational_threshold and not gen_on_conservative[t - STARTUP_TIME_STEPS])
    cost_fuel_conservative = sum(gen_on_conservative) * TIME_STEP_HOURS * COST_PER_HOUR_GENERATOR; cost_starts_conservative = starts_conservative * COST_PER_STARTUP; cost_blackout_conservative = blackouts_conservative * COST_PER_BLACKOUT_EVENT
    results['Conservadora (ORI Atenção)'] = {"Custo Total (USD)": cost_fuel_conservative + cost_starts_conservative + cost_blackout_conservative, "Horas de Gerador Ligado": sum(gen_on_conservative) * TIME_STEP_HOURS, "Número de Partidas": starts_conservative, "Eventos de Blackout": blackouts_conservative}
    return pd.DataFrame(results).T


# ===================================================================
# FUNÇÕES DE COMPARAÇÃO DE MODELOS
# ===================================================================

# CORREÇÃO NA FUNÇÃO load_multiple_models
def load_multiple_models(model_names, data, months, look_back, data_partition, cap, strategy_horizon, p10_safety_factor):
    """Loads multiple models and applies the safety factor to t+20 and t+30 horizons when applicable."""
    models_results = {}
    progress_bar = st.progress(0)
    for i, model_name in enumerate(model_names):
        model_output = run_model(model_name, data, months, look_back, data_partition, cap, strategy_horizon)
        
        # Unpack predictions safely
        all_predictions, y_test_inv = model_output[0], model_output[1]

        # Apply safety factor ONLY to t+20 and t+30 if it's a proposed model
        if "CEEMDAN-EWT-TFT-Aggregator" in model_name and p10_safety_factor < 1.0:
            all_predictions = all_predictions.copy()
            
            # >>>>> MUDANÇA PRINCIPAL AQUI <<<<<
            # Multiplica apenas as colunas 1 (t+20) e 2 (t+30) do P10
            all_predictions[0.1][:, 1:] = all_predictions[0.1][:, 1:] * p10_safety_factor
            
            print(f"Applied safety factor of {p10_safety_factor} to t+20/t+30 P10 for {model_name}")

        models_results[model_name] = {'predictions': all_predictions, 'y_test': y_test_inv}
        progress_bar.progress((i + 1) / len(model_names))
    progress_bar.empty()
    return models_results

def compare_metrics(models_results, cap):
    """Compara métricas de performance entre modelos"""
    metrics_data = []
    for model_name, results in models_results.items():
        all_predictions = results['predictions']
        y_test_inv = results['y_test']
        for h_idx, h_name in enumerate(['t+10 min', 't+20 min', 't+30 min']):
            pred_median = all_predictions[0.5][:, h_idx]
            y_true = y_test_inv[:, h_idx]
            mape = np.mean(np.abs((y_true - pred_median) / cap)) * 100
            rmse = sqrt(mean_squared_error(y_true, pred_median))
            mae = mean_absolute_error(y_true, pred_median)
            r2 = r2_score(y_true, pred_median)
            metrics_data.append({'Modelo': model_name, 'Horizonte': h_name, 'MAPE (%)': mape, 'RMSE (MW)': rmse, 'MAE (MW)': mae, 'R²': r2})
    return pd.DataFrame(metrics_data)

def compare_costs(models_results, operational_threshold, strategy_horizon):
    """Compara custos operacionais entre modelos"""
    costs_data = []
    for model_name, results in models_results.items():
        cost_df = calculate_costs(results['y_test'], results['predictions'], operational_threshold, strategy_horizon)
        ori_results = cost_df.loc['Preditiva (ORI)']
        costs_data.append({'Modelo': model_name, 'Custo Total (USD)': ori_results['Custo Total (USD)'], 'Horas de Gerador Ligado': ori_results['Horas de Gerador Ligado'], 'Número de Partidas': ori_results['Número de Partidas'], 'Eventos de Blackout': ori_results['Eventos de Blackout']})
    return pd.DataFrame(costs_data)

def plot_metrics_comparison(metrics_df):
    """Creates performance metric comparison charts in English."""
    # The tabs are now just for organization; titles are inside the charts.
    tab1, tab2, tab3, tab4 = st.tabs(["MAPE (%)", "RMSE (MW)", "MAE (MW)", "R²"])
    colors = px.colors.qualitative.Set2
    
    with tab1:
        # Use 'Model' and 'Horizon' for the axes
        fig = px.bar(metrics_df, x='Model', y='MAPE (%)', color='Horizon', barmode='group', 
                     color_discrete_sequence=colors, title="MAPE Comparison by Model and Horizon")
        st.plotly_chart(fig, use_container_width=True)
    with tab2:
        fig = px.bar(metrics_df, x='Model', y='RMSE (MW)', color='Horizon', barmode='group', 
                     color_discrete_sequence=colors, title="RMSE Comparison by Model and Horizon")
        st.plotly_chart(fig, use_container_width=True)
    with tab3:
        fig = px.bar(metrics_df, x='Model', y='MAE (MW)', color='Horizon', barmode='group', 
                     color_discrete_sequence=colors, title="MAE Comparison by Model and Horizon")
        st.plotly_chart(fig, use_container_width=True)
    with tab4:
        fig = px.bar(metrics_df, x='Model', y='R²', color='Horizon', barmode='group', 
                     color_discrete_sequence=colors, title="R² Comparison by Model and Horizon")
        st.plotly_chart(fig, use_container_width=True)

def plot_costs_comparison(costs_df):
    """Creates a cost comparison chart in English."""
    # Use the new English column names for x, y, and text
    fig = px.bar(costs_df, x='Model', y='Total Cost (USD)', 
                 title="Total Cost Comparison by Model", text='Total Cost (USD)')
    fig.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
    st.plotly_chart(fig, use_container_width=True)

def plot_comparative_predictions(models_results, y_test_inv, current_time_step, horizon_idx, horizon_name, cap):
    """Plota previsões de múltiplos modelos sobrepostas"""
    start_idx = max(0, current_time_step - 125)
    end_idx = min(len(y_test_inv), current_time_step + 125)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(start_idx, end_idx)), y=y_test_inv[start_idx:end_idx, horizon_idx], mode='lines', name='Real', line=dict(color='black', width=2)))
    colors = px.colors.qualitative.Set1
    for i, (model_name, results) in enumerate(models_results.items()):
        pred_median = results['predictions'][0.5][:, horizon_idx]
        fig.add_trace(go.Scatter(x=list(range(start_idx, end_idx)), y=pred_median[start_idx:end_idx], mode='lines', name=model_name, line=dict(dash='dash', width=2, color=colors[i % len(colors)])))
    fig.update_layout(title=f"Comparação de Previsões - {horizon_name}", xaxis_title="Amostra de Teste", yaxis_title="Potência (MW)", hovermode='x unified', height=500)
    st.plotly_chart(fig, use_container_width=True)




# ===================================================================
# PASSO 2: SIDEBAR CONTROLADA POR ESTADO E SALVAMENTO DE VARIÁVEIS
# ===================================================================

# --- 1. SIDEBAR ---
st.sidebar.header("Developed by:")
try:
    st.sidebar.image(Image.open('logos/logo_coppe.png'))
    st.sidebar.image(Image.open('logos/logo_lafae.png'))
except FileNotFoundError:
    st.sidebar.warning("Logo files not found.")
st.sidebar.markdown("---")

st.sidebar.header("Operation Mode")
comparison_mode = st.sidebar.checkbox("🔬 Model Comparison Mode")
st.sidebar.markdown("---")
st.sidebar.header("Simulation Parameters")

# Inicializa os valores no session_state se não existirem
if 'op_threshold_percent' not in st.session_state:
    st.session_state.op_threshold_percent = 30
if 'strategy_horizon' not in st.session_state:
    st.session_state.strategy_horizon = 20

# CÓDIGO CORRIGIDO
# >>>>> CORREÇÃO: Garante que o valor passado para o slider seja um inteiro <<<<<
op_threshold_percent = st.sidebar.slider(
    'Operational Threshold (% of Capacity)', 
    min_value=10, 
    max_value=50, 
    value=int(st.session_state.op_threshold_percent), # Converte para int aqui
    step=5, 
    key='op_threshold_percent_widget'
)

st.session_state.op_threshold_percent = op_threshold_percent # Garante a atualização

strategy_horizon = st.sidebar.select_slider("Risk Trigger Strategy:", options=[20, 30], value=st.session_state.strategy_horizon, format_func=lambda x: f"t+{x} min", key='strategy_horizon_widget')
st.session_state.strategy_horizon = strategy_horizon # Garante a atualização

# (O resto da lógica da sidebar permanece a mesma)
safety_factor = 1.0
if not comparison_mode:
    selected_model_name = st.sidebar.selectbox("Select Forecast Model:", options=list(model_functions.keys()), index=list(model_functions.keys()).index("CEEMDAN-EWT-TFT-Aggregator (Interp)"))
    if "CEEMDAN-EWT-TFT-Aggregator" in selected_model_name:
        st.sidebar.markdown("---")
        st.sidebar.subheader("Proposed Model Adjustment")
        safety_factor = st.sidebar.slider('P10 Safety Factor', 0.50, 1.0, 1.0, 0.01)
else:
    # A lógica do modo de comparação vai aqui... (omitida para focar na implementação principal)
    st.warning("Tool-using agent is available in Normal Mode only.")
    st.stop()

# ==================================================================
# >>>>> CORREÇÃO AQUI: DEFINIR OS PARÂMETROS DENTRO DO ESCOPO CORRETO <<<<<
# ==================================================================
look_back = 6
months_to_run = [1, 2, 3, 4, 5, 6]
data_partition = 0.8
horizon = 3  # <-- A VARIÁVEL AGORA ESTÁ DEFINIDA NO ESCOPO CERTO
# ==================================================================


# --- 2. CARREGAMENTO DE DADOS ---
new_data, cap = load_data('dataset/44.csv')
operational_threshold = (op_threshold_percent / 100) * cap

# --- 3. EXECUÇÃO DO MODO NORMAL ---
# (O código de execução do modelo e UI permanece o mesmo, mas adicionamos o salvamento no final)
try:
    model_output = run_model(selected_model_name, new_data, [1,2,3,4,5,6], 6, 0.8, cap, strategy_horizon)
    all_predictions, y_test_inv = model_output[0], model_output[1]
    attention_weights = model_output[2] if len(model_output) >= 3 else None
    if attention_weights is not None: st.sidebar.success("Attention weights loaded!")
    if "CEEMDAN-EWT-TFT-Aggregator" in selected_model_name and safety_factor < 1.0:
        all_predictions = all_predictions.copy()
        all_predictions[0.1][:, 1:] *= safety_factor
    p_low_t20, p_low_t30 = all_predictions[0.1][:, 1], all_predictions[0.1][:, 2]
    ori_levels = []
    if strategy_horizon == 20:
        for i in range(len(p_low_t20)):
            if p_low_t20[i] < operational_threshold: ori_levels.append('High')
            elif p_low_t30[i] < operational_threshold: ori_levels.append('Attention')
            else: ori_levels.append('Low')
    else:
        for p_low in p_low_t30: ori_levels.append('High' if p_low < operational_threshold else 'Low')
except Exception as e:
    st.error(f"An error occurred while running the model: {e}")
    st.stop()

# SALVAR VARIÁVEIS GLOBAIS PARA AS FERRAMENTAS USAREM
st.session_state.all_predictions = all_predictions
st.session_state.y_test_inv = y_test_inv
st.session_state.cap = cap
st.session_state.operational_threshold = operational_threshold
st.session_state.ori_levels = ori_levels
st.session_state.strategy_horizon = strategy_horizon
    
# ==================================================================
# MOVER OS DICIONÁRIOS PARA O ESCOPO CORRETO (APÓS A DEFINIÇÃO DE ori_levels)
# ==================================================================
color_map = {'Low': 'mediumseagreen', 'Attention': 'orange', 'High': 'salmon'}
icon_map = {
    'Low': 'icons/gerador_manter_desligado.png', 
    'Attention': 'icons/gerador_atencao.png', 
    'High': 'icons/gerador_ligar.png'
}
recommendation_text = {
    'Low': "KEEP GENERATOR OFF",
    'Attention': "STANDBY TEAM",
    'High': "CONNECT GENERATOR"
}
recommendation_color = {'Low': 'green', 'Attention': 'orange', 'High': 'red'}
# ==================================================================


# --- 3. TITLE ---
st.title("🌬️ Offshore Wind Farm Operation Simulation Dashboard")
st.markdown("### Decision Support Tool for Backup Generator Connection")
st.markdown("---")

# --- 4. TIME CONTROL AND KPIs ---
st.header("Dataset Simulation")
max_time_step = len(y_test_inv) - 1
current_time_step = st.slider("Navigate in Time (Test Sample)", 0, max_time_step, max_time_step // 2, help="Drag to move forward or backward in time.")
st.subheader("Key Performance Indicators (KPIs)")
kpi_cols = st.columns(4)
real_power_t10 = y_test_inv[current_time_step, 0]
pred_median_t10 = all_predictions[0.5][current_time_step, 0]
pred_median_t20 = all_predictions[0.5][current_time_step, 1]
pred_median_t30 = all_predictions[0.5][current_time_step, 2]
pred_low_t10_kpi = all_predictions[0.1][current_time_step, 0]
pred_low_t20_kpi = all_predictions[0.1][current_time_step, 1]
pred_low_t30_kpi = all_predictions[0.1][current_time_step, 2]
kpi_cols[0].metric("Actual Power (t+10 min)", f"{real_power_t10:.2f} MW")
kpi_cols[1].metric("Median Forecast (t+10 min)", f"{pred_median_t10:.2f} MW")
kpi_cols[2].metric("Worst Case (t+20 min)", f"{pred_low_t20_kpi:.2f} MW", delta=f"{(pred_low_t20_kpi - operational_threshold):.2f} MW vs Threshold", delta_color="normal", help="10% quantile forecast for 20 minutes. Used for the 'High' risk trigger.")
kpi_cols[3].metric("Worst Case (t+30 min)", f"{pred_low_t30_kpi:.2f} MW", delta=f"{(pred_low_t30_kpi - operational_threshold):.2f} MW vs Threshold", delta_color="normal", help="10% quantile forecast for 30 minutes. Used for the 'Attention' risk trigger.")

# --- 5. MAIN RECOMMENDATION ---
st.subheader("Operator Recommendation")

# IMPORTANT: Ensure your 'ori_levels' array now contains 'Low', 'Attention', 'High'
current_risk = ori_levels[current_time_step] 

rec_col1, rec_col2 = st.columns([1, 4])
with rec_col1:
    try: st.image(icon_map[current_risk], width=150)
    except: st.warning("Icon not found.")
with rec_col2:
    st.markdown(f"<h3 style='color:{recommendation_color[current_risk]};'>{recommendation_text[current_risk]}</h3>", unsafe_allow_html=True)

    # ==================================================================
    # 2. IF/ELIF STATEMENTS (NOW IN ENGLISH)
    # ==================================================================
    if current_risk == 'High':
        st.write(f"The worst-case forecast for **20 minutes** ({pred_low_t20_kpi:.2f} MW) is below the safety threshold of **{operational_threshold:.2f} MW**.")
    elif current_risk == 'Attention':
        st.write(f"The 20-min horizon is safe, but the worst-case forecast for **30 minutes** ({p_low_t30[current_time_step]:.2f} MW) is below the threshold. Prepare for a potential action.")
    else: # 'Low'
        st.write("The worst-case forecasts for 20 and 30 minutes are above the safety threshold.")
    # ==================================================================




# --- SECTION 5.1: PROACTIVE OPERATIONAL SURVEILLANCE AGENT ---
st.subheader("📢 Proactive Surveillance Agent Analysis")

# Collects history and ALL necessary KPIs for the agent's analysis
start_idx_history = max(0, current_time_step - 4)
end_idx_history = current_time_step + 1
ori_history_for_agent = ori_levels[start_idx_history:end_idx_history]

kpis_for_agent = {
    "pred_low_t10": pred_low_t10_kpi,
    "pred_low_t20": pred_low_t20_kpi,
    "pred_low_t30": pred_low_t30_kpi,
    # Adds median forecasts for trend analysis
    "pred_median_t10": pred_median_t10,
    "pred_median_t20": pred_median_t20,
    "pred_median_t30": pred_median_t30,
}

# Calls the new LLM-based agent function
with st.spinner("Surveillance agent analyzing the situation..."):
    proactive_analysis = get_llm_driven_proactive_alert(
        st.secrets["OPENAI_API_KEY"],
        ori_history_for_agent,
        kpis_for_agent,
        operational_threshold
    )

# Displays the agent's analysis, visually differentiating alerts from trends
if proactive_analysis:
    # ==================================================================
    # 3. AGENT DISPLAY LOGIC (NOW CORRECTLY MATCHES "ALERT:")
    # ==================================================================
    if proactive_analysis.startswith("ALERT:"):
        st.warning(proactive_analysis)
    elif proactive_analysis.startswith("TREND:"):
        st.info(proactive_analysis)
    else:
        st.write(proactive_analysis)
    # ==================================================================






# --- 6. VISUALIZAÇÃO GRÁFICA DETALHADA ---
st.header("Detailed Graphical Visualization") # Traduzido
VIEW_WINDOW = 250
start_idx = max(0, current_time_step - VIEW_WINDOW // 2); end_idx = min(len(y_test_inv), start_idx + VIEW_WINDOW)
time_axis = np.arange(start_idx, end_idx)

# Traduzido
graphic_tabs = st.tabs([f"Forecast t+{(h+1)*10} min" for h in range(horizon)])

for h, tab in enumerate(graphic_tabs):
    with tab: 
        fig = go.Figure()
        
        # --- 1. Add Dummy Traces for ORI Legend Items (in English) ---
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(symbol='square', size=10, color='lightgreen'), name='Low Risk'))
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(symbol='square', size=10, color='orange'), name='Attention Risk')) # Corrigido para orange
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(symbol='square', size=10, color='salmon'), name='High Risk'))
        
        # ==================================================================
        # CORREÇÃO AQUI: Usar o dicionário 'color_map' que já está em inglês
        # ==================================================================
        shapes = [
            go.layout.Shape(
                type="rect", xref="x", yref="paper",
                x0=i - 0.5, y0=0, x1=i + 0.5, y1=1,
                # Usa o dicionário 'color_map' que espera chaves em inglês ('Low', 'High', etc.)
                fillcolor=color_map.get(ori_levels[i], 'white'), 
                opacity=0.4, layer="below", line_width=0
            ) for i in range(start_idx, end_idx)
        ]
        fig.update_layout(shapes=shapes)
        # ==================================================================
        
        # --- 3. Add Main Data Traces (Unchanged) ---
        fig.add_hline(y=operational_threshold, line_dash="dot", line_color="red", annotation_text="Operational Threshold", annotation_position="bottom right")
        fig.add_trace(go.Scatter(x=time_axis, y=all_predictions[0.9][start_idx:end_idx, h], fill=None, mode='lines', line_color='rgba(0,100,80,0.2)', showlegend=False))
        fig.add_trace(go.Scatter(x=time_axis, y=all_predictions[0.1][start_idx:end_idx, h], fill='tonexty', mode='lines', line_color='rgba(0,100,80,0.2)', name='80% Confidence Interval'))
        fig.add_trace(go.Scatter(x=time_axis, y=y_test_inv[start_idx:end_idx, h], mode='lines', name='Actual', line=dict(color='black', width=2)))
        fig.add_trace(go.Scatter(x=time_axis, y=all_predictions[0.5][start_idx:end_idx, h], mode='lines', name='Median Forecast', line=dict(color='firebrick', dash='dash')))
        fig.add_vline(x=current_time_step, line_dash="dash", line_color="purple", annotation_text="Current Time", annotation_position="top left")
        
        # --- 4. Finalize Layout (Unchanged) ---
        fig.update_layout(
            title_text=f"Forecast and Risk Analysis - Horizon t+{(h+1)*10} min",
            xaxis_title="Test Samples",
            yaxis_title="LV ActivePower (MW)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=600,
            xaxis_range=[start_idx - 0.5, end_idx - 0.5]
        )
        
        st.plotly_chart(fig, use_container_width=True, key=f"graphic_chart_{h}")

# Também traduzi a última linha de texto informativo
st.info("**How to use:** Drag the 'Navigate in Time' slider to center the view. The charts show a 250-sample window (approx. 41 hours) for a clearer analysis.")

st.markdown("---")

# --- INTERPRETABILITY ANALYSIS (TEMPORAL ATTENTION) ---
if attention_weights is not None:
    st.markdown("---")
    st.header("🔍 Interpretability Analysis (Temporal Attention)")
    st.info(
        """
        This analysis shows which **past time steps** (within the `look_back` window) the model considered most important 
        to make the forecast at the **current point selected on the slider**.
        
        - The **heatmap** at the top shows the importance (the brighter the color, the more important).
        - The **line chart** below shows the actual power data that the model "saw" as input.
        
        This helps to understand whether the model is focusing on recent peaks, valleys, or transitions to make its predictions.
        """
    )

    try:
        # Chama a função de plotagem (agora simplificada)
        fig_attention = plot_attention_interpretation(
            attention_weights,
            y_test_inv,
            current_time_step,
            look_back
        )
        
        if fig_attention:
            st.plotly_chart(fig_attention, use_container_width=True)
        else:
            st.warning("Could not generate the attention chart for the selected time step.")
            
    except Exception as e:
        st.error(f"An unexpected error occurred while generating the attention chart: {e}")
        import traceback
        st.code(traceback.format_exc())

 


# --- 7. ANÁLISES DE SUPORTE (ECONÔMICA E DE PERFORMANCE) ---
st.header("Análises de Suporte")
cost_results_df = calculate_costs(y_test_inv, all_predictions, operational_threshold, strategy_horizon)
with st.expander("Ver Análise de Custo-Benefício das Estratégias"):
    st.markdown("#### Comparativo de Custo Total e Performance Operacional")
    st.write("A simulação compara a estratégia preditiva (ORI) com uma estratégia reativa e uma conservadora ao longo de todo o período de teste.")
    st.markdown("##### Premissas de Custo da Simulação")
    cost_assumptions_cols = st.columns(3)
    cost_assumptions_cols[0].info("**Custo por Hora (Gerador):** USD 500.00"); cost_assumptions_cols[1].info("**Custo por Partida:** USD 2,000.00"); cost_assumptions_cols[2].info("**Custo por Evento de Blackout:** USD 100,000.00")
    cost_results_df_display = cost_results_df.copy()
    cost_results_df_display['Custo Total (USD)'] = cost_results_df_display['Custo Total (USD)'].apply(lambda x: f"USD {x:,.2f}")
    st.table(cost_results_df_display)
    fig_cost = go.Figure(data=[go.Bar(x=cost_results_df.index, y=cost_results_df['Custo Total (USD)'], text=cost_results_df_display['Custo Total (USD)'], textposition='auto')])
    fig_cost.update_layout(title_text='Custo Total por Estratégia', yaxis_title='Custo (USD)'); st.plotly_chart(fig_cost, use_container_width=True)

with st.expander("Ver Métricas de Performance do Modelo"):
    st.markdown(f"**Modelo Utilizado:** `{selected_model_name}`")
    y_pred_median = all_predictions[0.5]
    st.markdown("##### Métricas Gerais por Horizonte")
    metric_data = []
    for h in range(horizon):
        y_true_h, y_pred_h = y_test_inv[:, h], y_pred_median[:, h]
        mape = np.mean(np.abs((y_true_h - y_pred_h) / cap)) * 100 if cap > 0 else 0
        rmse = sqrt(mean_squared_error(y_true_h, y_pred_h))
        mae = mean_absolute_error(y_true_h, y_pred_h)
        r2 = r2_score(y_true_h, y_pred_h)
        metric_data.append({"Horizonte": f"t+{(h+1)*10} min", "MAPE (%)": f"{mape:.4f}", "RMSE (MW)": f"{rmse:.4f}", "MAE (MW)": f"{mae:.4f}", "R²": f"{r2:.4f}"})
    metric_df = pd.DataFrame(metric_data)
    st.table(metric_df.set_index("Horizonte"))
    st.markdown("##### Regression Grafics (Real vs. Predicted)")
    regression_tabs = st.tabs([f"Horizon t+{(h+1)*10} min" for h in range(horizon)])
    for h, tab in enumerate(regression_tabs):
        with tab:
            y_true_h, y_pred_h = y_test_inv[:, h], y_pred_median[:, h]
            fig_reg = px.scatter(x=y_true_h, y=y_pred_h, labels={'x': 'Real Values (MW)', 'y': 'Predicted Values (MW)'}, title=f'Real vs. Predicted (t+{(h+1)*10} min)', trendline='ols', trendline_color_override='red', hover_data={'R²': [f'{r2_score(y_true_h, y_pred_h):.3f}'] * len(y_true_h)})
            fig_reg.update_layout(height=450)
            st.plotly_chart(fig_reg, use_container_width=True)

# >>>>> NOVA SEÇÃO PARA O GRÁFICO DE DECOMPOSIÇÃO (LÓGICA SIMPLIFICADA) <<<<<

# Verifica se o modelo selecionado é um dos que usam decomposição
if "CEEMDAN" in selected_model_name:
    # Carrega os dados de decomposição diretamente do arquivo .gz
    decomposed_components = load_decomposition_data(selected_model_name)

    # Se os dados foram carregados com sucesso, exibe o gráfico
    if decomposed_components is not None:
        #st.markdown("---")
        st.header("🔬 Análise da Decomposição de Componentes do Modelo")
        
        with st.expander("Ver Gráfico 3D da Decomposição Real", expanded=True):
            st.info(
                f"O gráfico abaixo mostra as **{len(decomposed_components.columns)} componentes reais** (IMFs + Resíduo) que o modelo `{selected_model_name}` "
                "utilizou como entrada. Estes dados foram carregados diretamente do arquivo de decomposição pré-processado."
            )
            
            # Usa a função de plotagem que já criamos, sem precisar de nenhuma alteração nela
            fig_decomposition = plot_real_decomposition_3d(decomposed_components)
            st.plotly_chart(fig_decomposition, use_container_width=True)




# ===================================================================
# >>>>> OTIMIZADOR NA UI PELA VERSÃO COM LÓGICA CONDICIONAL <<<<<
# ===================================================================
st.markdown("---")
st.header("🛠️ Operational Strategy Optimizer")

# Verifica se o modelo selecionado é o proposto para usar o otimizador 3D
is_proposed_interp_model = "CEEMDAN-EWT-TFT-Aggregator (Interp)" in selected_model_name

if is_proposed_interp_model:
    st.write("""
    For the proposed model, this tool performs an exhaustive **3D simulation** to find the optimal combination of 
    **Operational Threshold**, **Trigger Horizon**, and **P10 Safety Factor** that results in the lowest total cost.
    """)
    if st.button("Run Global Strategy Optimization", key="optimize_button_3d"):
        with st.spinner("Running 3D simulations for multiple strategies... (This may take a few minutes)"):
            # CHAMA A NOVA FUNÇÃO DE OTIMIZAÇÃO 3D
            best_strategy = optimize_strategy_3d(y_test_inv, all_predictions, cap)
            st.session_state.best_strategy_results = best_strategy
        
        st.subheader("Best Global Strategy Found")
        col1, col2, col3 = st.columns(3)
        col1.metric("Minimum Total Cost", f"USD {best_strategy['cost']:,.2f}")
        col2.metric("Best Operational Threshold", f"{best_strategy['threshold_percent']:.1f}%")
        col3.metric("Best Trigger Horizon", f"t+{best_strategy['horizon']} min")
        st.metric("Best P10 Safety Factor", f"{best_strategy['safety_factor']:.2f}")
        st.info(f"**Blackouts:** {best_strategy['blackouts']} | **Generator Starts:** {best_strategy['starts']}")
        st.success(f"The global lowest-cost strategy involves a threshold of {best_strategy['threshold_percent']:.1f}%, a trigger at t+{best_strategy['horizon']} minutes, and a P10 Safety Factor of {best_strategy['safety_factor']:.2f}.")

else: # Para todos os outros modelos, usa o otimizador 2D com heatmap
    st.write("""
    This tool performs a simulation to find the combination of **Operational Threshold** and **Trigger Horizon** 
    that results in the lowest total cost for the selected model.
    """)
    if st.button("Run Strategy Optimization", key="optimize_button_2d"):
        with st.spinner("Running simulations... (This may take a moment)"):
            # CHAMA A FUNÇÃO DE OTIMIZAÇÃO 2D ORIGINAL
            cost_matrix, threshold_range, horizon_range, best_strategy = optimize_strategy(y_test_inv, all_predictions, cap)
            st.session_state.best_strategy_results = best_strategy
        
        st.subheader("Best Strategy Found")
        col1, col2, col3 = st.columns(3)
        col1.metric("Minimum Total Cost", f"USD {best_strategy['cost']:,.2f}")
        col2.metric("Best Operational Threshold", f"{best_strategy['threshold_percent']:.1f}%")
        col3.metric("Best Trigger Horizon", f"t+{best_strategy['horizon']} min")
        st.info(f"**Blackouts:** {best_strategy['blackouts']} | **Generator Starts:** {best_strategy['starts']}")
        
        st.subheader("Total Cost Heatmap (USD)")
        fig_heatmap = px.imshow(
            cost_matrix,
            labels=dict(x="Operational Threshold (%)", y="Trigger Horizon (min)", color="Total Cost (USD)"),
            x=[f"{t:.1f}" for t in threshold_range],
            y=horizon_range,
            color_continuous_scale='RdYlGn_r'
        )
        fig_heatmap.update_yaxes(type='category')
        fig_heatmap.update_layout(title=f'Simulated Total Cost per Strategy for Model "{selected_model_name}"')
        st.plotly_chart(fig_heatmap, use_container_width=True)
        st.info("The green areas represent parameter combinations that lead to a lower total operational cost.")
# ===================================================================



# ===================================================================
# SEÇÃO DO COPILOTO COM CONTEXTO (RAG) FINAL E APRIMORADO
# ===================================================================
st.markdown("---")
st.header("🤖 Operational Copilot: Interactive Analysis")

# Inicializa o histórico de chat se não existir
if "messages" not in st.session_state:
    st.session_state.messages = []

# Exibe o histórico de chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input do usuário
if prompt := st.chat_input("Ask a question or give a command..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Agent is thinking..."):
            client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

            # ==================================================================
            # >>>>> MUDANÇA PRINCIPAL AQUI: CONSTRUÇÃO DO CONTEXTO APRIMORADO <<<<<
            # ==================================================================
            current_risk = st.session_state.ori_levels[current_time_step]
            
            # Coleta os valores de previsão do estado da aplicação
            # (Essas variáveis já são calculadas na seção de KPIs do seu dashboard)
            pred_median_t10 = all_predictions[0.5][current_time_step, 0]
            pred_median_t20 = all_predictions[0.5][current_time_step, 1]
            pred_median_t30 = all_predictions[0.5][current_time_step, 2]
            pred_p10_t10 = all_predictions[0.1][current_time_step, 0]
            pred_p10_t20 = all_predictions[0.1][current_time_step, 1]
            pred_p10_t30 = all_predictions[0.1][current_time_step, 2]

            data_context_string = f"""### Current Situation (Real-Time)
- Selected Model: {selected_model_name}
- Current Time Step: {current_time_step}
- Recent Risk History: {' -> '.join(st.session_state.ori_levels[max(0, current_time_step-4):current_time_step+1])}
- CURRENT Risk Level (ORI): {current_risk}
- CURRENT Operational Threshold: {st.session_state.operational_threshold:.2f} MW
- CURRENT P10 Safety Factor (applied): {safety_factor:.2f}

### Forecast Data (at current time step)
- Horizon t+10 min: Median Forecast: {pred_median_t10:.2f} MW, Worst-Case (P10): {pred_p10_t10:.2f} MW
- Horizon t+20 min: Median Forecast: {pred_median_t20:.2f} MW, Worst-Case (P10): {pred_p10_t20:.2f} MW
- Horizon t+30 min: Median Forecast: {pred_median_t30:.2f} MW, Worst-Case (P10): {pred_p10_t30:.2f} MW
"""
            # Adiciona o contexto da otimização, se existir
            if st.session_state.get("best_strategy_results"):
                opt_results = st.session_state["best_strategy_results"]
                sf_info = f", Best P10 Safety Factor: {opt_results.get('safety_factor', 'N/A'):.2f}" if 'safety_factor' in opt_results else ""
                data_context_string += f"""
### Last Strategy Optimization Results
- Minimum Cost Found: USD {opt_results['cost']:,.2f}
- Best Threshold: {opt_results['threshold_percent']:.1f}%
- Best Horizon: t+{opt_results['horizon']} min{sf_info}"""
            else:
                data_context_string += "\n- Strategy optimization has not been run yet."
            # ==================================================================

            # 2. Descrever as ferramentas (Nenhuma mudança aqui)
            tools = [
                {"type": "function", "function": {"name": "apply_new_parameters_tool", "description": "Applies a new operational strategy by setting the threshold and horizon.", "parameters": {"type": "object", "properties": {"threshold_percent": {"type": "number"}, "horizon": {"type": "integer"}}, "required": ["threshold_percent", "horizon"]}}},
                {"type": "function", "function": {"name": "generate_summary_report_tool", "description": "Generates a summary report of operational costs and events for a given time period.", "parameters": {"type": "object", "properties": {"start_time_step": {"type": "integer"}, "end_time_step": {"type": "integer"}}, "required": ["start_time_step", "end_time_step"]}}}
            ]
            if "CEEMDAN-EWT-TFT-Aggregator (Interp)" in selected_model_name:
                tools.append({"type": "function", "function": {"name": "run_global_strategy_optimization_tool", "description": "Runs a full 3D simulation (including safety factor) to find the absolute best operational strategy. Use this for the proposed model.", "parameters": {"type": "object", "properties": {}}}})
            else:
                tools.append({"type": "function", "function": {"name": "run_strategy_optimization_tool", "description": "Runs a 2D simulation to find the optimal operational threshold and trigger horizon. Use this for benchmark models.", "parameters": {"type": "object", "properties": {}}}})

            # 3. Criar o System Prompt Híbrido (Nenhuma mudança aqui)
            system_prompt = f"""You are a 'Strategic Operational Copilot', an expert AI assistant for offshore wind-connected FPSO operations. Your primary goal is to assist the operator.
You have two ways to respond:
1.  **Use a Tool:** If the user's request directly maps to one of the available tools, you MUST call that tool. IMPORTANT: If the selected model is 'CEEMDAN-EWT-TFT-Aggregator (Interp)', you MUST use the 'run_global_strategy_optimization_tool'. For all other models, use the standard 'run_strategy_optimization_tool'.
2.  **Answer Directly:** If the user asks a general question (e.g., 'what is the status?', 'summarize the situation'), you MUST analyze the provided 'Current data context' and formulate a comprehensive, data-driven answer. DO NOT call a tool for general questions.
Always be concise and base ALL your responses and tool decisions on the provided context data.
Current data context:
{data_context_string}
"""
            
            # 4. Loop de Execução (Nenhuma mudança aqui)
            messages_for_api = [{"role": "system", "content": system_prompt}] + st.session_state.messages
            
            response = client.chat.completions.create(model="gpt-4o-mini", messages=messages_for_api, tools=tools, tool_choice="auto")
            response_message = response.choices[0].message
            tool_calls = response_message.tool_calls

            if tool_calls:
                messages_for_api.append(response_message)
                
                available_functions = {
                    "run_strategy_optimization_tool": run_strategy_optimization_tool,
                    "run_global_strategy_optimization_tool": run_global_strategy_optimization_tool,
                    "apply_new_parameters_tool": apply_new_parameters_tool,
                    "generate_summary_report_tool": generate_summary_report_tool
                }
                
                for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    function_to_call = available_functions[function_name]
                    function_args = json.loads(tool_call.function.arguments)
                    
                    with st.spinner(f"Agent is using tool: `{function_name}`..."):
                        function_response = function_to_call(**function_args)
                    
                    messages_for_api.append({"tool_call_id": tool_call.id, "role": "tool", "name": function_name, "content": function_response})
                
                with st.spinner("Agent is processing tool results..."):
                    second_response = client.chat.completions.create(model="gpt-4o-mini", messages=messages_for_api, tools=tools, tool_choice="auto")
                    final_response_content = second_response.choices[0].message.content
                    st.markdown(final_response_content)
                    st.session_state.messages.append({"role": "assistant", "content": final_response_content})
            else:
                final_response_content = response_message.content
                st.markdown(final_response_content)
                st.session_state.messages.append({"role": "assistant", "content": final_response_content})



# >>>>> NOVA SEÇÃO: Auditoria da Resposta da IA <<<<<
if st.session_state.last_context:
    with st.expander("View Last Copilot Analysis Context (Audit)"): # Traduzido
        st.markdown("The data below was sent to the AI to generate the most recent response.") # Traduzido
        st.code(st.session_state.last_context, language='markdown')
