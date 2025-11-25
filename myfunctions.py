#!/usr/bin/env python
# coding: utf-8

# In[1]:


from PyEMD import CEEMDAN
import math
import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn import metrics
import numpy
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_docs as tfdocs


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from math import sqrt




# Função para criar janelas de tempo
def create_dataset(data, look_back):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i+look_back, 0])
        y.append(data[i+look_back, 0])
    return np.array(X), np.array(y)


# In[2]:

import tensorflow as tf
from tensorflow.keras.layers import Layer

# Classe de Atenção
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], 1),
            initializer='random_normal',
            trainable=True
        )
        self.b = self.add_weight(
            name='attention_bias',
            shape=(input_shape[1], 1),
            initializer='zeros',
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        e = tf.math.tanh(tf.matmul(inputs, self.W) + self.b)  # (batch_size, time_steps, 1)
        a = tf.nn.softmax(e, axis=1)                          # (batch_size, time_steps, 1)
        output = inputs * a                                   # broadcasting
        return tf.reduce_sum(output, axis=1)                  # (batch_size, features)


#----#



##SVR

# Em myfunctions.py

def svr_model(new_data, months, look_back, data_partition, cap, horizon=3, strategy_horizon=20, mode='train'):
    """
    Modelo SVR (v3 - Versão de Análise)
    Implementação que demonstra o comportamento determinístico do SVR.
    Treina três modelos idênticos para simular a estrutura quantílica,
    resultando em previsões de ponto e um intervalo de confiança nulo,
    o que é um resultado experimental válido para análise comparativa.
    """
    # --- Imports ---
    import numpy as np
    import pandas as pd
    from math import sqrt
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.svm import SVR
    from sklearn.multioutput import MultiOutputRegressor
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    import random
    import os
    import joblib

    # --- Reprodutibilidade e Configurações ---
    random.seed(42)
    np.random.seed(42)
    MODEL_DIR = "saved_models/svr_model"

    # --- Preparação dos Dados ---
    df = new_data.loc[new_data['Month'].isin(months)].reset_index(drop=True).dropna()
    values = df['LV ActivePower (kW)'].values.reshape(-1, 1)
    train_size = int(len(values) * data_partition)
    train, test = values[:train_size], values[train_size:]

    def create_dataset_multi(dataset, look_back, horizon):
        X, Y = [], []
        for j in range(len(dataset) - look_back - horizon + 1):
            X.append(dataset[j:j + look_back, 0])
            Y.append(dataset[j + look_back:j + look_back + horizon, 0])
        return np.array(X), np.array(Y)

    trainX, trainY = create_dataset_multi(train, look_back, horizon)
    testX, testY = create_dataset_multi(test, look_back, horizon)
    y_test_inv = testY

    all_predictions = {}
    quantiles = [0.1, 0.5, 0.9]

    if mode == 'train':
        print("--- EXECUTANDO SVR EM MODO DE TREINAMENTO (ANÁLISE DETERMINÍSTICA) ---")
        sc_X = StandardScaler().fit(trainX)
        sc_y = StandardScaler().fit(trainY)
        X_train, y_train = sc_X.transform(trainX), sc_y.transform(trainY)
        X_test = sc_X.transform(testX)

        # Treina um único modelo SVR, pois os outros seriam idênticos
        print("Treinando modelo SVR (previsão de ponto)...")
        base_svr = SVR(kernel='rbf', C=0.6997192627226148, epsilon=0.0010906250648233518, gamma='auto')
        model = MultiOutputRegressor(base_svr)
        model.fit(X_train, y_train)
        
        # Faz a previsão
        y_pred_scaled = model.predict(X_test)
        y_pred_inv = sc_y.inverse_transform(y_pred_scaled)
        
        # Atribui a mesma previsão de ponto para todos os "quantis"
        # para manter a estrutura de dados compatível com o dashboard.
        print("Atribuindo previsão de ponto para todos os quantis (0.1, 0.5, 0.9)...")
        for q in quantiles:
            all_predictions[q] = np.clip(y_pred_inv, 0, cap)

        # Salva o modelo e os scalers
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(model, os.path.join(MODEL_DIR, "model.gz"))
        joblib.dump(sc_X, os.path.join(MODEL_DIR, "scaler_X.gz"))
        joblib.dump(sc_y, os.path.join(MODEL_DIR, "scaler_y.gz"))
        print("Modelo SVR e scalers salvos com sucesso.")

    elif mode == 'inference':
        print("--- EXECUTANDO SVR EM MODO DE INFERÊNCIA ---")
        if not os.path.exists(MODEL_DIR):
            raise FileNotFoundError(f"Diretório de modelos SVR '{MODEL_DIR}' não encontrado.")

        # Carrega o único modelo e os scalers
        model = joblib.load(os.path.join(MODEL_DIR, "model.gz"))
        sc_X = joblib.load(os.path.join(MODEL_DIR, "scaler_X.gz"))
        sc_y = joblib.load(os.path.join(MODEL_DIR, "scaler_y.gz"))
        
        X_test = sc_X.transform(testX)
        y_pred_scaled = model.predict(X_test)
        y_pred_inv = sc_y.inverse_transform(y_pred_scaled)
        
        # Replica a previsão de ponto para todos os quantis
        for q in quantiles:
            all_predictions[q] = np.clip(y_pred_inv, 0, cap)

    else:
        raise ValueError("O parâmetro 'mode' deve ser 'train' ou 'inference'.")

    # A correção de cruzamento de quantis não tem efeito aqui, mas a mantemos por consistência estrutural.
    all_predictions[0.5] = np.maximum(all_predictions[0.1], all_predictions[0.5])
    all_predictions[0.9] = np.maximum(all_predictions[0.5], all_predictions[0.9])

    if mode == 'train':
        # A previsão da mediana é a própria previsão de ponto do SVR
        y_pred_median = all_predictions[0.5]
        print("\n===== Métricas Completas SVR (Previsão de Ponto) =====")
        for h in range(horizon):
            y_true_h, y_pred_h = y_test_inv[:, h], y_pred_median[:, h]
            mape = np.mean(np.abs((y_true_h - y_pred_h) / cap)) * 100 if cap > 0 else 0
            rmse = sqrt(mean_squared_error(y_true_h, y_pred_h))
            mae = mean_absolute_error(y_true_h, y_pred_h)
            r2 = r2_score(y_true_h, y_pred_h)
            print(f"Horizonte t+{(h+1)*10}: MAPE={mape:.4f}%, RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")

        # A lógica de risco ORI ainda pode ser calculada, mas será binária (Alto ou Baixo),
        # pois o quantil 0.1 é igual à mediana. Não haverá o estado de "Atenção".
        operational_threshold = 0.3 * cap 
        p_low_t20 = all_predictions[0.1][:, 1]
        p_low_t30 = all_predictions[0.1][:, 2]
        ori_levels = []
        # Com SVR, o risco de "Atenção" não existirá, pois p_low_t20 == p_median_t20
        for i in range(len(p_low_t20)):
            if p_low_t20[i] < operational_threshold or p_low_t30[i] < operational_threshold:
                ori_levels.append('Alto')
            else:
                ori_levels.append('Baixo')

        color_map = {'Baixo': 'lightgreen', 'Alto': 'salmon'} # Apenas 2 estados de risco
        num_samples_to_plot = 300
        time_axis = np.arange(min(num_samples_to_plot, len(y_test_inv)))

        for h in range(horizon):
            fig, ax = plt.subplots(figsize=(18, 8))
            for i in range(len(time_axis)):
                ax.axvspan(i, i + 1, facecolor=color_map.get(ori_levels[i], 'white'), alpha=0.5, zorder=0)
            ax.axhline(y=operational_threshold, color='red', linestyle=':', linewidth=2.5, label=f'Limiar Operacional ({operational_threshold:.2f} kW)', zorder=4)
            
            # O fill_between não mostrará nada, o que é o resultado esperado
            ax.fill_between(time_axis, all_predictions[0.1][:len(time_axis), h], all_predictions[0.9][:len(time_axis), h], color='cornflowerblue', alpha=0.6, label='Intervalo de Confiança (N/A para SVR padrão)', zorder=1)
            
            ax.plot(time_axis, y_test_inv[:len(time_axis), h], label="Real", color='black', linewidth=2, zorder=3)
            ax.plot(time_axis, all_predictions[0.5][:len(time_axis), h], label="Previsão de Ponto (SVR)", color='firebrick', linestyle='--', linewidth=2.5, zorder=2)
            
            ax.set_title(f'Previsão de Ponto SVR com Risco Binário - Horizonte t+{(h+1)*10} min', fontsize=16)
            ax.set_ylabel('LV ActivePower (kW)', fontsize=12)
            ax.set_xlabel('Amostras de Teste', fontsize=12)
            
            risk_patches = [Patch(facecolor=color, alpha=0.5, label=f'Risco {level}') for level, color in color_map.items()]
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles=handles + risk_patches, loc='best', fontsize=12)
            
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
            ax.set_xlim(0, len(time_axis))
            ax.set_ylim(bottom=max(0, y_test_inv.min() - (0.05 * cap)))
            plt.tight_layout()
            plt.show()
            plt.close(fig)

    return all_predictions, y_test_inv



    

#--------------#


# Em myfunctions.py

# A função principal `svr_model` permanece inalterada.
# A função `svr_model_optuna` é ajustada para otimizar, treinar e avaliar, mas sem salvar.

def svr_model_optuna(new_data, months, look_back, data_partition, cap, horizon=3, n_trials=30):
    """
    Otimiza, treina e avalia um modelo SVR determinístico.
    - DESCobre os hiperparâmetros ótimos com Optuna e TimeSeriesSplit.
    - TREINA um modelo final com os melhores parâmetros em todo o conjunto de treino.
    - AVALIA e plota os resultados no conjunto de teste.
    - NÃO salva o modelo ou os scalers no disco.
    """
    # --- Imports ---
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import TimeSeriesSplit
    from math import sqrt
    import optuna
    import random

    # --- Reprodutibilidade ---
    random.seed(42)
    np.random.seed(42)

    # --- Preparação dos Dados ---
    print("--- PREPARANDO DADOS PARA OTIMIZAÇÃO E TREINO DO SVR ---")
    df = new_data.loc[new_data['Month'].isin(months)].reset_index(drop=True).dropna()
    values = df['LV ActivePower (kW)'].values.reshape(-1, 1)

    # Divisão em treino (para otimização) e teste (para avaliação final)
    train_size = int(len(values) * data_partition)
    train_values, test_values = values[:train_size], values[train_size:]

    def create_dataset_multi(dataset, look_back, horizon):
        X, Y = [], []
        for j in range(len(dataset) - look_back - horizon + 1):
            X.append(dataset[j:j + look_back, 0])
            Y.append(dataset[j + look_back:j + look_back + horizon, 0])
        return np.array(X), np.array(Y)

    # Janelas para o conjunto de treino completo (será usado na validação cruzada)
    trainX_full, trainY_full = create_dataset_multi(train_values, look_back, horizon)
    
    # Janelas para o conjunto de teste final (mantido separado)
    testX, testY = create_dataset_multi(test_values, look_back, horizon)
    print(f"Dados preparados: {trainX_full.shape[0]} amostras de treino, {testX.shape[0]} amostras de teste.")

    # --- Função Objetivo do Optuna ---
    def objective(trial):
        C = trial.suggest_float("C", 1e-1, 1e3, log=True)
        epsilon = trial.suggest_float("epsilon", 1e-3, 1.0, log=True)
        gamma = trial.suggest_categorical("gamma", ["scale", "auto"])

        tscv = TimeSeriesSplit(n_splits=3)
        mape_scores = []

        for train_idx, val_idx in tscv.split(trainX_full):
            X_tr, X_val = trainX_full[train_idx], trainX_full[val_idx]
            y_tr, y_val = trainY_full[train_idx], trainY_full[val_idx]

            sc_X_fold = StandardScaler().fit(X_tr)
            sc_y_fold = StandardScaler().fit(y_tr)

            X_tr_scaled = sc_X_fold.transform(X_tr)
            y_tr_scaled = sc_y_fold.transform(y_tr)
            X_val_scaled = sc_X_fold.transform(X_val)

            base_model = SVR(kernel="rbf", C=C, epsilon=epsilon, gamma=gamma)
            model = MultiOutputRegressor(base_model)
            model.fit(X_tr_scaled, y_tr_scaled)

            preds_scaled = model.predict(X_val_scaled)
            preds_inv = sc_y_fold.inverse_transform(preds_scaled)

            mape_cap = np.mean(np.abs(y_val - preds_inv) / cap) * 100
            mape_scores.append(mape_cap)

        return np.mean(mape_scores)

    # --- Execução da Otimização ---
    print("\n--- INICIANDO BUSCA DE HIPERPARÂMETROS COM OPTUNA ---")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True,n_jobs=-1)

    best_params = study.best_params
    print("\n" + "="*50)
    print("OTIMIZAÇÃO CONCLUÍDA")
    print("="*50)
    print(f"Melhor resultado (MAPE médio na validação): {study.best_value:.4f}%")
    print("\nHiperparâmetros Ótimos Encontrados:")
    for key, value in study.best_params.items():
        print(f"  - {key}: {value}")
    print("="*50)

    # --- Treinamento Final com os Melhores Parâmetros ---
    print("\n--- TREINANDO MODELO SVR FINAL COM OS MELHORES PARÂMETROS ---")
    
    # Scalers finais, ajustados em todo o conjunto de treino
    sc_X_final = StandardScaler().fit(trainX_full)
    sc_y_final = StandardScaler().fit(trainY_full)

    X_train_scaled = sc_X_final.transform(trainX_full)
    y_train_scaled = sc_y_final.transform(trainY_full)
    X_test_scaled = sc_X_final.transform(testX)

    final_base_model = SVR(kernel="rbf", **best_params)
    final_model = MultiOutputRegressor(final_base_model)
    final_model.fit(X_train_scaled, y_train_scaled)
    print("Treinamento final concluído.")

    # --- Avaliação Final no Conjunto de Teste ---
    y_pred_test = sc_y_final.inverse_transform(final_model.predict(X_test_scaled))
    y_test_inv = testY # testY já está na escala original

    metrics_data = []
    print("\n===== MÉTRICAS FINAIS NO CONJUNTO DE TESTE =====")
    for h in range(horizon):
        mape = np.mean(np.abs((y_test_inv[:, h] - y_pred_test[:, h]) / cap)) * 100
        rmse = sqrt(mean_squared_error(y_test_inv[:, h], y_pred_test[:, h]))
        mae = mean_absolute_error(y_test_inv[:, h], y_pred_test[:, h])
        r2 = r2_score(y_test_inv[:, h], y_pred_test[:, h])
        metrics_data.append({'Horizonte': f't+{(h+1)*10} min', 'MAPE (%)': mape, 'RMSE (MW)': rmse, 'MAE (MW)': mae, 'R²': r2})
        print(f"Horizonte t+{(h+1)*10}: MAPE={mape:.4f}%, RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")

    # --- Plotagem dos Resultados Finais ---
    print("\n--- GERANDO GRÁFICOS DE RESULTADOS ---")
    for h in range(horizon):
        plt.figure(figsize=(15, 6))
        plt.plot(y_test_inv[:, h], label=f"Real t+{(h+1)*10}", color="blue", alpha=0.8)
        plt.plot(y_pred_test[:, h], label=f"Previsto (Otimizado) t+{(h+1)*10}", color="red", linestyle="--")
        plt.title(f"Previsão SVR Otimizado vs. Real - Horizonte t+{(h+1)*10} min")
        plt.xlabel("Amostras de Teste")
        plt.ylabel("LV ActivePower (kW)")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

    return final_model, best_params, pd.DataFrame(metrics_data)





#------------#

# myfunctions.py (sem lambda, usando funções nomeadas)



# =================================
# 1. FUNÇÃO DE PERDA QUANTÍLICA (Inalterada)
# =================================
def quantile_loss(q, y_true, y_pred):
    e = y_true - y_pred
    return tf.keras.backend.mean(tf.keras.backend.maximum(q * e, (q - 1) * e), axis=-1)

# =======================================================================
# 2. FUNÇÕES DE PERDA NOMEADAS (Substituindo a lambda)
# =======================================================================
# Criamos funções específicas para cada quantil. O Keras pode salvá-las e carregá-las com segurança.
def quantile_loss_p10(y_true, y_pred):
    return quantile_loss(0.1, y_true, y_pred)

def quantile_loss_p50(y_true, y_pred):
    return quantile_loss(0.5, y_true, y_pred)

def quantile_loss_p90(y_true, y_pred):
    return quantile_loss(0.9, y_true, y_pred)

# Dicionário para facilitar o acesso a essas funções
loss_functions = {
    0.1: quantile_loss_p10,
    0.5: quantile_loss_p50,
    0.9: quantile_loss_p90
}

# =======================================================================
# 3. FUNÇÃO ÚNICA PARA TREINAMENTO E INFERÊNCIA
# =======================================================================


def ann_quantile_model_with_rri(new_data, months, look_back, data_partition, cap, horizon=3, strategy_horizon=20, mode='train'):
    """
    Modelo ANN Quantílico (v2) com correção de cruzamento de quantis (quantile crossing).
    Versão 100% completa com métricas e plotagem.
    """
    # --- Imports ---
    import tensorflow as tf
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from tensorflow.keras.optimizers import Adam
    from math import sqrt
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense,Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    import random
    from matplotlib.patches import Patch
    import os
    import joblib
    
    # --- Reprodutibilidade e Configurações ---
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)
    MODEL_DIR = "saved_models/ann_quantile"

    # --- Funções de Perda (essenciais para carregar o modelo) ---
    def quantile_loss_p10(y_true, y_pred):
        e = y_true - y_pred
        return tf.keras.backend.mean(tf.keras.backend.maximum(0.1 * e, (0.1 - 1) * e), axis=-1)
    def quantile_loss_p50(y_true, y_pred):
        e = y_true - y_pred
        return tf.keras.backend.mean(tf.keras.backend.maximum(0.5 * e, (0.5 - 1) * e), axis=-1)
    def quantile_loss_p90(y_true, y_pred):
        e = y_true - y_pred
        return tf.keras.backend.mean(tf.keras.backend.maximum(0.9 * e, (0.9 - 1) * e), axis=-1)
    loss_functions = {0.1: quantile_loss_p10, 0.5: quantile_loss_p50, 0.9: quantile_loss_p90}

    # --- Preparação dos Dados ---
    df = new_data.loc[new_data['Month'].isin(months)].reset_index(drop=True).dropna()
    values = df['LV ActivePower (kW)'].values.reshape(-1, 1)
    train_size = int(len(values) * data_partition)
    train, test = values[:train_size], values[train_size:]

    def create_dataset_multi(dataset, look_back, horizon):
        X, Y = [], []
        for j in range(len(dataset) - look_back - horizon + 1):
            X.append(dataset[j:j + look_back, 0])
            Y.append(dataset[j + look_back:j + look_back + horizon, 0])
        return np.array(X), np.array(Y)

    trainX, trainY = create_dataset_multi(train, look_back, horizon)
    testX, testY = create_dataset_multi(test, look_back, horizon)
    y_test_inv = testY

    all_predictions = {}
    quantiles = [0.1, 0.5, 0.9]

    if mode == 'train':
        print("--- EXECUTANDO ANN EM MODO DE TREINAMENTO ---")
        sc_X = StandardScaler().fit(trainX)
        sc_y = StandardScaler().fit(trainY)
        X_train, y_train = sc_X.transform(trainX), sc_y.transform(trainY)
        X_test = sc_X.transform(testX)

        for q in quantiles:
            print(f"Treinando modelo para o quantil: {q}")
            model = Sequential([Dense(256, activation='relu', input_shape=(look_back,)),
                                Dropout(0.17050673083465284),
                                Dense(32, activation='relu'), 
                                Dense(horizon)])
            model.compile(optimizer=Adam(learning_rate=0.00010533245345497477), loss=loss_functions[q])
            early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
            model.fit(X_train, y_train, epochs=5000, batch_size=64, validation_split=0.2, shuffle=False, verbose=0, callbacks=[early_stop])
            y_pred_scaled = model.predict(X_test)
            y_pred_inv = sc_y.inverse_transform(y_pred_scaled)
            all_predictions[q] = np.clip(y_pred_inv, 0, cap)
            os.makedirs(MODEL_DIR, exist_ok=True)
            model_path = os.path.join(MODEL_DIR, f"model_q{str(q).replace('.', '')}.keras")
            model.save(model_path)
            print(f"Modelo salvo em: {model_path}")

        joblib.dump(sc_X, os.path.join(MODEL_DIR, "scaler_X.gz"))
        joblib.dump(sc_y, os.path.join(MODEL_DIR, "scaler_y.gz"))
        print("Scalers salvos com sucesso.")

    elif mode == 'inference':
        print("--- EXECUTANDO ANN EM MODO DE INFERÊNCIA ---")
        if not os.path.exists(MODEL_DIR):
            raise FileNotFoundError(f"Diretório de modelos '{MODEL_DIR}' não encontrado. Execute a função em modo 'train' primeiro.")

        sc_X = joblib.load(os.path.join(MODEL_DIR, "scaler_X.gz"))
        sc_y = joblib.load(os.path.join(MODEL_DIR, "scaler_y.gz"))
        X_test = sc_X.transform(testX)
        custom_objects = {fn.__name__: fn for fn in loss_functions.values()}

        for q in quantiles:
            model_path = os.path.join(MODEL_DIR, f"model_q{str(q).replace('.', '')}.keras")
            model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
            y_pred_scaled = model.predict(X_test)
            y_pred_inv = sc_y.inverse_transform(y_pred_scaled)
            all_predictions[q] = np.clip(y_pred_inv, 0, cap)

    else:
        raise ValueError("O parâmetro 'mode' deve ser 'train' ou 'inference'.")

    # ##################################################################
    # #################### CORREÇÃO DE QUANTILE CROSSING ###############
    # ##################################################################
    print("Aplicando correção de cruzamento de quantis...")
    all_predictions[0.5] = np.maximum(all_predictions[0.1], all_predictions[0.5])
    all_predictions[0.9] = np.maximum(all_predictions[0.5], all_predictions[0.9])
    # ##################################################################
    # ##################################################################

    if mode == 'train':
        y_pred_median = all_predictions[0.5]
        print("\n===== Métricas Completas (baseadas na previsão da mediana q=0.5) =====")
        for h in range(horizon):
            y_true_h, y_pred_h = y_test_inv[:, h], y_pred_median[:, h]
            mape = np.mean(np.abs((y_true_h - y_pred_h) / cap)) * 100
            rmse = sqrt(mean_squared_error(y_true_h, y_pred_h))
            mae = mean_absolute_error(y_true_h, y_pred_h)
            r2 = r2_score(y_true_h, y_pred_h)
            print(f"Horizonte t+{(h+1)*10}: MAPE={mape:.4f}%, RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")

        operational_threshold = 0.3 * cap 
        p_low_t20 = all_predictions[0.1][:, 1]
        p_low_t30 = all_predictions[0.1][:, 2]
        ori_levels = []
        if strategy_horizon == 20:
            for i in range(len(p_low_t20)):
                if p_low_t20[i] < operational_threshold: ori_levels.append('Alto')
                elif p_low_t30[i] < operational_threshold: ori_levels.append('Atenção')
                else: ori_levels.append('Baixo')
        else:
            for p_low in p_low_t30:
                ori_levels.append('Alto' if p_low < operational_threshold else 'Baixo')

        color_map = {'Baixo': 'lightgreen', 'Atenção': 'gold', 'Alto': 'salmon'}
        
        num_samples_to_plot = 300
        time_axis = np.arange(min(num_samples_to_plot, len(y_test_inv)))

        for h in range(horizon):
            fig, ax = plt.subplots(figsize=(18, 8))
            for i in range(len(time_axis)):
                ax.axvspan(i, i + 1, facecolor=color_map.get(ori_levels[i], 'white'), alpha=0.5, zorder=0)
            ax.axhline(y=operational_threshold, color='red', linestyle=':', linewidth=2.5, label=f'Limiar Operacional ({operational_threshold:.2f} kW)', zorder=4)
            ax.fill_between(time_axis, all_predictions[0.1][:len(time_axis), h], all_predictions[0.9][:len(time_axis), h], color='cornflowerblue', alpha=0.6, label='Intervalo de Confiança (80%)', zorder=1)
            ax.plot(time_axis, y_test_inv[:len(time_axis), h], label="Real", color='black', linewidth=2, zorder=3)
            ax.plot(time_axis, all_predictions[0.5][:len(time_axis), h], label="Previsão Mediana", color='firebrick', linestyle='--', linewidth=2.5, zorder=2)
            ax.set_title(f'Previsão Probabilística com ORI - Horizonte t+{(h+1)*10} min', fontsize=16)
            ax.set_ylabel('LV ActivePower (kW)', fontsize=12)
            ax.set_xlabel('Amostras de Teste', fontsize=12)
            risk_patches = [Patch(facecolor=color, alpha=0.5, label=f'Risco {level}') for level, color in color_map.items() if level in set(ori_levels)]
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles=handles + risk_patches, loc='best', fontsize=12)
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
            ax.set_xlim(0, len(time_axis))
            ax.set_ylim(bottom=max(0, y_test_inv.min() - (0.05 * cap)))
            plt.tight_layout()
            plt.show()
            plt.close(fig) # Adicionado para garantir que a memória da figura seja liberada

    return all_predictions, y_test_inv



#-----#

# Em myfunctions.py

# A função principal `ann_quantile_model_with_rri` permanece inalterada.
# A função `ann_model_optuna` é ajustada para otimizar, treinar e avaliar, mas sem salvar.

def ann_model_optuna(new_data, months, look_back, data_partition, cap, horizon=3, n_trials=30):
    """
    Otimiza, treina e avalia um modelo ANN determinístico.
    - DESCobre os hiperparâmetros ótimos com Optuna e TimeSeriesSplit.
    - TREINA um modelo final com os melhores parâmetros em todo o conjunto de treino.
    - AVALIA e plota os resultados no conjunto de teste.
    - NÃO salva o modelo ou os scalers no disco.
    """
    # --- Imports ---
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.optimizers import Adam
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from math import sqrt
    import optuna
    import random

    # --- Reprodutibilidade ---
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)

    # --- Preparação dos Dados ---
    print("--- PREPARANDO DADOS PARA OTIMIZAÇÃO E TREINO DA ANN ---")
    df = new_data.loc[new_data['Month'].isin(months)].reset_index(drop=True).dropna()
    values = df['LV ActivePower (kW)'].values.reshape(-1, 1)

    train_size = int(len(values) * data_partition)
    train_values, test_values = values[:train_size], values[train_size:]

    def create_dataset_multi(dataset, look_back, horizon):
        X, Y = [], []
        for j in range(len(dataset) - look_back - horizon + 1):
            X.append(dataset[j:j + look_back, 0])
            Y.append(dataset[j + look_back:j + look_back + horizon, 0])
        return np.array(X), np.array(Y)

    trainX_full, trainY_full = create_dataset_multi(train_values, look_back, horizon)
    testX, testY = create_dataset_multi(test_values, look_back, horizon)
    print(f"Dados preparados: {trainX_full.shape[0]} amostras de treino, {testX.shape[0]} amostras de teste.")

    # --- Função Objetivo do Optuna ---
    def objective(trial):
        neurons_layer1 = trial.suggest_categorical('neurons_layer1', [32, 64, 128, 256])
        neurons_layer2 = trial.suggest_categorical('neurons_layer2', [16, 32, 64, 128])
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
        
        tscv = TimeSeriesSplit(n_splits=3)
        mape_scores = []

        for train_idx, val_idx in tscv.split(trainX_full):
            X_tr, X_val = trainX_full[train_idx], trainX_full[val_idx]
            y_tr, y_val = trainY_full[train_idx], trainY_full[val_idx]

            sc_X_fold = StandardScaler().fit(X_tr)
            sc_y_fold = StandardScaler().fit(y_tr)
            X_tr_scaled = sc_X_fold.transform(X_tr)
            y_tr_scaled = sc_y_fold.transform(y_tr)
            X_val_scaled = sc_X_fold.transform(X_val)

            model = Sequential([
                Dense(neurons_layer1, activation='relu', input_shape=(look_back,)),
                Dropout(dropout_rate),
                Dense(neurons_layer2, activation='relu'),
                Dense(horizon)
            ])
            model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')
            early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=0)
            model.fit(X_tr_scaled, y_tr_scaled, batch_size=batch_size, epochs=5000,
                      validation_data=(X_val_scaled, sc_y_fold.transform(y_val)),
                      shuffle=False, verbose=0, callbacks=[early_stop])

            preds_scaled = model.predict(X_val_scaled, verbose=0)
            preds_inv = sc_y_fold.inverse_transform(preds_scaled)
            mape_cap = np.mean(np.abs(y_val - preds_inv) / cap) * 100
            mape_scores.append(mape_cap)
            tf.keras.backend.clear_session()
        return np.mean(mape_scores)

    # --- Execução da Otimização ---
    print("\n--- INICIANDO BUSCA DE HIPERPARÂMETROS PARA ANN COM OPTUNA ---")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    print("\n" + "="*50)
    print("OTIMIZAÇÃO DA ANN CONCLUÍDA")
    print("="*50)
    print(f"Melhor resultado (MAPE médio na validação): {study.best_value:.4f}%")
    print("\nHiperparâmetros Ótimos Encontrados:")
    for key, value in study.best_params.items():
        print(f"  - {key}: {value}")
    print("="*50)

    # --- Treinamento Final com os Melhores Parâmetros ---
    print("\n--- TREINANDO MODELO ANN FINAL COM OS MELHORES PARÂMETROS ---")
    sc_X_final = StandardScaler().fit(trainX_full)
    sc_y_final = StandardScaler().fit(trainY_full)

    X_train_scaled = sc_X_final.transform(trainX_full)
    y_train_scaled = sc_y_final.transform(trainY_full)
    X_test_scaled = sc_X_final.transform(testX)

    final_model = Sequential([
        Dense(best_params['neurons_layer1'], activation='relu', input_shape=(look_back,)),
        Dropout(best_params['dropout_rate']),
        Dense(best_params['neurons_layer2'], activation='relu'),
        Dense(horizon)
    ])
    final_model.compile(optimizer=Adam(learning_rate=best_params['learning_rate']), loss='mean_squared_error')
    
    # Usamos uma fração do treino para early stopping, como na função quantílica
    early_stop_final = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)
    final_model.fit(X_train_scaled, y_train_scaled, 
                    batch_size=best_params['batch_size'], 
                    epochs=5000,
                    validation_split=0.2, # Usando 20% do treino para validação final
                    shuffle=False, verbose=0, callbacks=[early_stop_final])
    print("Treinamento final concluído.")

    # --- Avaliação Final no Conjunto de Teste ---
    y_pred_test = sc_y_final.inverse_transform(final_model.predict(X_test_scaled, verbose=0))
    y_test_inv = testY

    metrics_data = []
    print("\n===== MÉTRICAS FINAIS NO CONJUNTO DE TESTE =====")
    for h in range(horizon):
        mape = np.mean(np.abs((y_test_inv[:, h] - y_pred_test[:, h]) / cap)) * 100
        rmse = sqrt(mean_squared_error(y_test_inv[:, h], y_pred_test[:, h]))
        mae = mean_absolute_error(y_test_inv[:, h], y_pred_test[:, h])
        r2 = r2_score(y_test_inv[:, h], y_pred_test[:, h])
        metrics_data.append({'Horizonte': f't+{(h+1)*10} min', 'MAPE (%)': mape, 'RMSE (MW)': rmse, 'MAE (MW)': mae, 'R²': r2})
        print(f"Horizonte t+{(h+1)*10}: MAPE={mape:.4f}%, RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")

    # --- Plotagem dos Resultados Finais ---
    print("\n--- GERANDO GRÁFICOS DE RESULTADOS ---")
    for h in range(horizon):
        plt.figure(figsize=(15, 6))
        plt.plot(y_test_inv[:, h], label=f"Real t+{(h+1)*10}", color="blue", alpha=0.8)
        plt.plot(y_pred_test[:, h], label=f"Previsto (Otimizado) t+{(h+1)*10}", color="red", linestyle="--")
        plt.title(f"Previsão ANN Otimizada vs. Real - Horizonte t+{(h+1)*10} min")
        plt.xlabel("Amostras de Teste")
        plt.ylabel("LV ActivePower (kW)")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

    return final_model, best_params, pd.DataFrame(metrics_data)



# RF

def rf_model(new_data, months, look_back, data_partition, cap, horizon=3, strategy_horizon=20, mode='train'):
    """
    Modelo Random Forest adaptado para previsão quantílica, seguindo o padrão do dashboard.
    - Opera em modos 'train' e 'inference'.
    - Gera previsões quantílicas calculando os percentis das previsões de cada árvore no ensemble.
    - Aplica correção de cruzamento de quantis.
    - Salva/carrega o modelo e os scalers para integração com o dashboard.
    - Gera métricas e gráficos de ORI no modo 'train'.
    """
    # --- Imports ---
    import numpy as np
    import pandas as pd
    from math import sqrt
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    import random
    import os
    import joblib

    # --- Reprodutibilidade e Configurações ---
    random.seed(42)
    np.random.seed(42)
    MODEL_DIR = "saved_models/rf_model" # Diretório específico para este modelo

    # --- Preparação dos Dados ---
    df = new_data.loc[new_data['Month'].isin(months)].reset_index(drop=True).dropna()
    values = df['LV ActivePower (kW)'].values.reshape(-1, 1)
    train_size = int(len(values) * data_partition)
    train, test = values[:train_size], values[train_size:]

    def create_dataset_multi(dataset, look_back, horizon):
        X, Y = [], []
        for j in range(len(dataset) - look_back - horizon + 1):
            X.append(dataset[j:j + look_back, 0])
            Y.append(dataset[j + look_back:j + look_back + horizon, 0])
        return np.array(X), np.array(Y)

    trainX, trainY = create_dataset_multi(train, look_back, horizon)
    testX, testY = create_dataset_multi(test, look_back, horizon)
    y_test_inv = testY

    all_predictions = {}
    quantiles = [0.1, 0.5, 0.9]

    if mode == 'train':
        print("--- EXECUTANDO RANDOM FOREST EM MODO DE TREINAMENTO ---")
        sc_X = StandardScaler().fit(trainX)
        sc_y = StandardScaler().fit(trainY)
        X_train, y_train = sc_X.transform(trainX), sc_y.transform(trainY)
        X_test = sc_X.transform(testX)

        # Treinamos um ÚNICO modelo Random Forest
        print("Treinando modelo Random Forest...")
        # n_jobs=-1 usa todos os processadores disponíveis, acelerando o treinamento
        model = RandomForestRegressor(n_estimators=315,max_depth=7,min_samples_split=12,min_samples_leaf=7, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)

        # Salva o modelo e os scalers
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(model, os.path.join(MODEL_DIR, "model.gz"))
        joblib.dump(sc_X, os.path.join(MODEL_DIR, "scaler_X.gz"))
        joblib.dump(sc_y, os.path.join(MODEL_DIR, "scaler_y.gz"))
        print("Modelo Random Forest e scalers salvos com sucesso.")

    elif mode == 'inference':
        print("--- EXECUTANDO RANDOM FOREST EM MODO DE INFERÊNCIA ---")
        if not os.path.exists(MODEL_DIR):
            raise FileNotFoundError(f"Diretório de modelos RF '{MODEL_DIR}' não encontrado.")

        model = joblib.load(os.path.join(MODEL_DIR, "model.gz"))
        sc_X = joblib.load(os.path.join(MODEL_DIR, "scaler_X.gz"))
        sc_y = joblib.load(os.path.join(MODEL_DIR, "scaler_y.gz"))
        X_test = sc_X.transform(testX)

    else:
        raise ValueError("O parâmetro 'mode' deve ser 'train' ou 'inference'.")

    # ##################################################################
    # ####### GERAÇÃO DAS PREVISÕES QUATÍLICAS (A MÁGICA DO RF) ########
    # ##################################################################
    print("Gerando previsões quantílicas a partir do ensemble Random Forest...")
    # 1. Pega as previsões de cada árvore individualmente
    # O resultado é uma lista de previsões, onde cada item é a previsão de uma árvore
    tree_preds_scaled = np.array([tree.predict(X_test) for tree in model.estimators_])

    # 2. Calcula os quantis sobre as previsões das árvores para cada amostra
    # np.quantile(array, q, axis=0) calcula o quantil ao longo do eixo das árvores
    for q in quantiles:
        quantile_preds_scaled = np.quantile(tree_preds_scaled, q, axis=0)
        # Inverte a escala e armazena
        y_pred_inv = sc_y.inverse_transform(quantile_preds_scaled)
        all_predictions[q] = np.clip(y_pred_inv, 0, cap)
    # ##################################################################
    # ##################################################################

    # Correção de cruzamento de quantis (boa prática, embora menos provável no RF)
    print("Aplicando correção de cruzamento de quantis...")
    all_predictions[0.5] = np.maximum(all_predictions[0.1], all_predictions[0.5])
    all_predictions[0.9] = np.maximum(all_predictions[0.5], all_predictions[0.9])

    if mode == 'train':
        y_pred_median = all_predictions[0.5]
        print("\n===== Métricas Completas RF (baseadas na previsão da mediana q=0.5) =====")
        for h in range(horizon):
            y_true_h, y_pred_h = y_test_inv[:, h], y_pred_median[:, h]
            mape = np.mean(np.abs((y_true_h - y_pred_h) / cap)) * 100 if cap > 0 else 0
            rmse = sqrt(mean_squared_error(y_true_h, y_pred_h))
            mae = mean_absolute_error(y_true_h, y_pred_h)
            r2 = r2_score(y_true_h, y_pred_h)
            print(f"Horizonte t+{(h+1)*10}: MAPE={mape:.4f}%, RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")

        # Lógica de Risco ORI (idêntica aos outros modelos)
        operational_threshold = 0.3 * cap 
        p_low_t20 = all_predictions[0.1][:, 1]
        p_low_t30 = all_predictions[0.1][:, 2]
        ori_levels = []
        if strategy_horizon == 20:
            for i in range(len(p_low_t20)):
                if p_low_t20[i] < operational_threshold: ori_levels.append('Alto')
                elif p_low_t30[i] < operational_threshold: ori_levels.append('Atenção')
                else: ori_levels.append('Baixo')
        else:
            for p_low in p_low_t30:
                ori_levels.append('Alto' if p_low < operational_threshold else 'Baixo')

        color_map = {'Baixo': 'lightgreen', 'Atenção': 'gold', 'Alto': 'salmon'}
        num_samples_to_plot = 300
        time_axis = np.arange(min(num_samples_to_plot, len(y_test_inv)))

        for h in range(horizon):
            fig, ax = plt.subplots(figsize=(18, 8))
            for i in range(len(time_axis)):
                ax.axvspan(i, i + 1, facecolor=color_map.get(ori_levels[i], 'white'), alpha=0.5, zorder=0)
            ax.axhline(y=operational_threshold, color='red', linestyle=':', linewidth=2.5, label=f'Limiar Operacional ({operational_threshold:.2f} kW)', zorder=4)
            ax.fill_between(time_axis, all_predictions[0.1][:len(time_axis), h], all_predictions[0.9][:len(time_axis), h], color='cornflowerblue', alpha=0.6, label='Intervalo de Confiança (80%)', zorder=1)
            ax.plot(time_axis, y_test_inv[:len(time_axis), h], label="Real", color='black', linewidth=2, zorder=3)
            ax.plot(time_axis, all_predictions[0.5][:len(time_axis), h], label="Previsão Mediana", color='firebrick', linestyle='--', linewidth=2.5, zorder=2)
            ax.set_title(f'Previsão Quantílica Random Forest com ORI - Horizonte t+{(h+1)*10} min', fontsize=16)
            ax.set_ylabel('LV ActivePower (kW)', fontsize=12)
            ax.set_xlabel('Amostras de Teste', fontsize=12)
            risk_patches = [Patch(facecolor=color, alpha=0.5, label=f'Risco {level}') for level, color in color_map.items() if level in set(ori_levels)]
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles=handles + risk_patches, loc='best', fontsize=12)
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
            ax.set_xlim(0, len(time_axis))
            ax.set_ylim(bottom=max(0, y_test_inv.min() - (0.05 * cap)))
            plt.tight_layout()
            plt.show()
            plt.close(fig)

    return all_predictions, y_test_inv



#----#



# Em myfunctions.py

# A função principal `rf_model` permanece inalterada.
# A função `rf_model_optuna` é ajustada para otimizar, treinar e avaliar, mas sem salvar.

def rf_model_optuna(new_data, months, look_back, data_partition, cap, horizon=3, n_trials=30):
    """
    Otimiza, treina e avalia um modelo Random Forest determinístico.
    - DESCobre os hiperparâmetros ótimos com Optuna e TimeSeriesSplit.
    - TREINA um modelo final com os melhores parâmetros em todo o conjunto de treino.
    - AVALIA e plota os resultados no conjunto de teste.
    - NÃO salva o modelo ou os scalers no disco.
    """
    # --- Imports ---
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import TimeSeriesSplit
    from math import sqrt
    import optuna
    import random

    # --- Reprodutibilidade ---
    random.seed(42)
    np.random.seed(42)

    # --- Preparação dos Dados ---
    print("--- PREPARANDO DADOS PARA OTIMIZAÇÃO E TREINO DO RANDOM FOREST ---")
    df = new_data.loc[new_data['Month'].isin(months)].reset_index(drop=True).dropna()
    values = df['LV ActivePower (kW)'].values.reshape(-1, 1)

    train_size = int(len(values) * data_partition)
    train_values, test_values = values[:train_size], values[train_size:]

    def create_dataset_multi(dataset, look_back, horizon):
        X, Y = [], []
        for j in range(len(dataset) - look_back - horizon + 1):
            X.append(dataset[j:j + look_back, 0])
            Y.append(dataset[j + look_back:j + look_back + horizon, 0])
        return np.array(X), np.array(Y)

    trainX_full, trainY_full = create_dataset_multi(train_values, look_back, horizon)
    testX, testY = create_dataset_multi(test_values, look_back, horizon)
    print(f"Dados preparados: {trainX_full.shape[0]} amostras de treino, {testX.shape[0]} amostras de teste.")

    # --- Função Objetivo do Optuna ---
    def objective(trial):
        # Espaço de busca para os hiperparâmetros do Random Forest
        n_estimators = trial.suggest_int('n_estimators', 50, 400)
        max_depth = trial.suggest_int('max_depth', 5, 50)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 14)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
        
        tscv = TimeSeriesSplit(n_splits=3)
        mape_scores = []

        for train_idx, val_idx in tscv.split(trainX_full):
            X_tr, X_val = trainX_full[train_idx], trainX_full[val_idx]
            y_tr, y_val = trainY_full[train_idx], trainY_full[val_idx]

            sc_X_fold = StandardScaler().fit(X_tr)
            sc_y_fold = StandardScaler().fit(y_tr)

            X_tr_scaled = sc_X_fold.transform(X_tr)
            y_tr_scaled = sc_y_fold.transform(y_tr)
            X_val_scaled = sc_X_fold.transform(X_val)

            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_tr_scaled, y_tr_scaled)

            preds_scaled = model.predict(X_val_scaled)
            preds_inv = sc_y_fold.inverse_transform(preds_scaled)

            mape_cap = np.mean(np.abs(y_val - preds_inv) / cap) * 100
            mape_scores.append(mape_cap)

        return np.mean(mape_scores)

    # --- Execução da Otimização ---
    print("\n--- INICIANDO BUSCA DE HIPERPARÂMETROS PARA RANDOM FOREST COM OPTUNA ---")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    print("\n" + "="*50)
    print("OTIMIZAÇÃO DO RANDOM FOREST CONCLUÍDA")
    print("="*50)
    print(f"Melhor resultado (MAPE médio na validação): {study.best_value:.4f}%")
    print("\nHiperparâmetros Ótimos Encontrados:")
    for key, value in study.best_params.items():
        print(f"  - {key}: {value}")
    print("="*50)

    # --- Treinamento Final com os Melhores Parâmetros ---
    print("\n--- TREINANDO MODELO RANDOM FOREST FINAL COM OS MELHORES PARÂMETROS ---")
    
    sc_X_final = StandardScaler().fit(trainX_full)
    sc_y_final = StandardScaler().fit(trainY_full)

    X_train_scaled = sc_X_final.transform(trainX_full)
    y_train_scaled = sc_y_final.transform(trainY_full)
    X_test_scaled = sc_X_final.transform(testX)

    final_model = RandomForestRegressor(
        random_state=42,
        n_jobs=-1,
        **best_params
    )
    final_model.fit(X_train_scaled, y_train_scaled)
    print("Treinamento final concluído.")

    # --- Avaliação Final no Conjunto de Teste ---
    # A previsão do Random Forest é sempre determinística (ponto)
    y_pred_test = sc_y_final.inverse_transform(final_model.predict(X_test_scaled))
    y_test_inv = testY

    metrics_data = []
    print("\n===== MÉTRICAS FINAIS NO CONJUNTO DE TESTE =====")
    for h in range(horizon):
        mape = np.mean(np.abs((y_test_inv[:, h] - y_pred_test[:, h]) / cap)) * 100
        rmse = sqrt(mean_squared_error(y_test_inv[:, h], y_pred_test[:, h]))
        mae = mean_absolute_error(y_test_inv[:, h], y_pred_test[:, h])
        r2 = r2_score(y_test_inv[:, h], y_pred_test[:, h])
        metrics_data.append({'Horizonte': f't+{(h+1)*10} min', 'MAPE (%)': mape, 'RMSE (MW)': rmse, 'MAE (MW)': mae, 'R²': r2})
        print(f"Horizonte t+{(h+1)*10}: MAPE={mape:.4f}%, RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")

    # --- Plotagem dos Resultados Finais ---
    print("\n--- GERANDO GRÁFICOS DE RESULTADOS ---")
    for h in range(horizon):
        plt.figure(figsize=(15, 6))
        plt.plot(y_test_inv[:, h], label=f"Real t+{(h+1)*10}", color="blue", alpha=0.8)
        plt.plot(y_pred_test[:, h], label=f"Previsto (Otimizado) t+{(h+1)*10}", color="red", linestyle="--")
        plt.title(f"Previsão Random Forest Otimizado vs. Real - Horizonte t+{(h+1)*10} min")
        plt.xlabel("Amostras de Teste")
        plt.ylabel("LV ActivePower (kW)")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

    return final_model, best_params, pd.DataFrame(metrics_data)




# In[5]:


# Em myfunctions.py

def lstm_model(new_data, months, look_back, data_partition, cap, horizon=3, strategy_horizon=20, mode='train'):
    """
    Modelo LSTM adaptado para previsão quantílica, seguindo o padrão do dashboard.
    - Opera em modos 'train' e 'inference'.
    - Utiliza funções de perda quantílica para treinar modelos para os quantis 0.1, 0.5 e 0.9.
    - Aplica correção de cruzamento de quantis.
    - Salva/carrega modelos e scalers para integração com o dashboard.
    - Gera métricas e gráficos de ORI no modo 'train'.
    """
    # --- Imports ---
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from math import sqrt
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.layers import Dense, LSTM, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    import random
    import os
    import joblib

    # --- Reprodutibilidade e Configurações ---
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)
    MODEL_DIR = "saved_models/lstm_model" # Diretório específico para este modelo

    # --- Funções de Perda (essenciais para carregar o modelo) ---
    # Reutilizando as funções de perda nomeadas já definidas no arquivo
    # (quantile_loss_p10, quantile_loss_p50, quantile_loss_p90, loss_functions)

    # --- Preparação dos Dados ---
    df = new_data.loc[new_data['Month'].isin(months)].reset_index(drop=True).dropna()
    values = df['LV ActivePower (kW)'].values.reshape(-1, 1)
    train_size = int(len(values) * data_partition)
    train, test = values[:train_size], values[train_size:]

    def create_dataset_multi(dataset, look_back, horizon):
        X, Y = [], []
        for j in range(len(dataset) - look_back - horizon + 1):
            X.append(dataset[j:j + look_back, 0])
            Y.append(dataset[j + look_back:j + look_back + horizon, 0])
        return np.array(X), np.array(Y)

    trainX, trainY = create_dataset_multi(train, look_back, horizon)
    testX, testY = create_dataset_multi(test, look_back, horizon)
    y_test_inv = testY

    all_predictions = {}
    quantiles = [0.1, 0.5, 0.9]

    if mode == 'train':
        print("--- EXECUTANDO LSTM EM MODO DE TREINAMENTO ---")
        sc_X = StandardScaler().fit(trainX)
        sc_y = StandardScaler().fit(trainY)
        X_train_scaled, y_train_scaled = sc_X.transform(trainX), sc_y.transform(trainY)
        X_test_scaled = sc_X.transform(testX)

        # Reshape para o formato 3D do LSTM
        X_train = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
        X_test = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))

        for q in quantiles:
            print(f"Treinando modelo LSTM para o quantil: {q}")
            
            model = Sequential([
                LSTM(units=128, input_shape=(look_back, 1)),
                Dropout(0.22760734633943003),
                Dense(horizon)
            ])
            model.compile(optimizer=Adam(learning_rate=0.0009238947460667959), loss=loss_functions[q])
            
            early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
            model.fit(X_train, y_train_scaled, epochs=5000, batch_size=128, validation_split=0.1, shuffle=False, verbose=0, callbacks=[early_stop])
            
            y_pred_scaled = model.predict(X_test)
            y_pred_inv = sc_y.inverse_transform(y_pred_scaled)
            all_predictions[q] = np.clip(y_pred_inv, 0, cap)

            os.makedirs(MODEL_DIR, exist_ok=True)
            model_path = os.path.join(MODEL_DIR, f"model_q{str(q).replace('.', '')}.keras")
            model.save(model_path)
            print(f"Modelo LSTM salvo em: {model_path}")

        joblib.dump(sc_X, os.path.join(MODEL_DIR, "scaler_X.gz"))
        joblib.dump(sc_y, os.path.join(MODEL_DIR, "scaler_y.gz"))
        print("Scalers para LSTM salvos com sucesso.")

    elif mode == 'inference':
        print("--- EXECUTANDO LSTM EM MODO DE INFERÊNCIA ---")
        if not os.path.exists(MODEL_DIR):
            raise FileNotFoundError(f"Diretório de modelos LSTM '{MODEL_DIR}' não encontrado.")

        sc_X = joblib.load(os.path.join(MODEL_DIR, "scaler_X.gz"))
        sc_y = joblib.load(os.path.join(MODEL_DIR, "scaler_y.gz"))
        X_test_scaled = sc_X.transform(testX)
        X_test = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))
        
        custom_objects = {fn.__name__: fn for fn in loss_functions.values()}

        for q in quantiles:
            model_path = os.path.join(MODEL_DIR, f"model_q{str(q).replace('.', '')}.keras")
            model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
            
            y_pred_scaled = model.predict(X_test)
            y_pred_inv = sc_y.inverse_transform(y_pred_scaled)
            all_predictions[q] = np.clip(y_pred_inv, 0, cap)

    else:
        raise ValueError("O parâmetro 'mode' deve ser 'train' ou 'inference'.")

    # Correção de cruzamento de quantis
    print("Aplicando correção de cruzamento de quantis...")
    all_predictions[0.5] = np.maximum(all_predictions[0.1], all_predictions[0.5])
    all_predictions[0.9] = np.maximum(all_predictions[0.5], all_predictions[0.9])

    if mode == 'train':
        y_pred_median = all_predictions[0.5]
        print("\n===== Métricas Completas LSTM (baseadas na previsão da mediana q=0.5) =====")
        for h in range(horizon):
            y_true_h, y_pred_h = y_test_inv[:, h], y_pred_median[:, h]
            mape = np.mean(np.abs((y_true_h - y_pred_h) / cap)) * 100 if cap > 0 else 0
            rmse = sqrt(mean_squared_error(y_true_h, y_pred_h))
            mae = mean_absolute_error(y_true_h, y_pred_h)
            r2 = r2_score(y_true_h, y_pred_h)
            print(f"Horizonte t+{(h+1)*10}: MAPE={mape:.4f}%, RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")

        # Lógica de Risco ORI
        operational_threshold = 0.3 * cap 
        p_low_t20 = all_predictions[0.1][:, 1]
        p_low_t30 = all_predictions[0.1][:, 2]
        ori_levels = []
        if strategy_horizon == 20:
            for i in range(len(p_low_t20)):
                if p_low_t20[i] < operational_threshold: ori_levels.append('Alto')
                elif p_low_t30[i] < operational_threshold: ori_levels.append('Atenção')
                else: ori_levels.append('Baixo')
        else:
            for p_low in p_low_t30:
                ori_levels.append('Alto' if p_low < operational_threshold else 'Baixo')

        color_map = {'Baixo': 'lightgreen', 'Atenção': 'gold', 'Alto': 'salmon'}
        num_samples_to_plot = 300
        time_axis = np.arange(min(num_samples_to_plot, len(y_test_inv)))

        for h in range(horizon):
            fig, ax = plt.subplots(figsize=(18, 8))
            for i in range(len(time_axis)):
                ax.axvspan(i, i + 1, facecolor=color_map.get(ori_levels[i], 'white'), alpha=0.5, zorder=0)
            ax.axhline(y=operational_threshold, color='red', linestyle=':', linewidth=2.5, label=f'Limiar Operacional ({operational_threshold:.2f} kW)', zorder=4)
            ax.fill_between(time_axis, all_predictions[0.1][:len(time_axis), h], all_predictions[0.9][:len(time_axis), h], color='cornflowerblue', alpha=0.6, label='Intervalo de Confiança (80%)', zorder=1)
            ax.plot(time_axis, y_test_inv[:len(time_axis), h], label="Real", color='black', linewidth=2, zorder=3)
            ax.plot(time_axis, all_predictions[0.5][:len(time_axis), h], label="Previsão Mediana", color='firebrick', linestyle='--', linewidth=2.5, zorder=2)
            ax.set_title(f'Previsão Quantílica LSTM com ORI - Horizonte t+{(h+1)*10} min', fontsize=16)
            ax.set_ylabel('LV ActivePower (kW)', fontsize=12)
            ax.set_xlabel('Amostras de Teste', fontsize=12)
            risk_patches = [Patch(facecolor=color, alpha=0.5, label=f'Risco {level}') for level, color in color_map.items() if level in set(ori_levels)]
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles=handles + risk_patches, loc='best', fontsize=12)
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
            ax.set_xlim(0, len(time_axis))
            ax.set_ylim(bottom=max(0, y_test_inv.min() - (0.05 * cap)))
            plt.tight_layout()
            plt.show()
            plt.close(fig)

    return all_predictions, y_test_inv





#---#
# Em myfunctions.py

# A função principal `lstm_model` permanece inalterada.
# A função `lstm_model_with_optuna` é ajustada para otimizar, treinar e avaliar, mas sem salvar.

def lstm_model_with_optuna(new_data, months, look_back, data_partition, cap, horizon=3, n_trials=30):
    """
    Otimiza, treina e avalia um modelo LSTM determinístico.
    - DESCobre os hiperparâmetros ótimos com Optuna e TimeSeriesSplit.
    - TREINA um modelo final com os melhores parâmetros em todo o conjunto de treino.
    - AVALIA e plota os resultados no conjunto de teste.
    - NÃO salva o modelo ou os scalers no disco.
    """
    # --- Imports ---
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.optimizers import Adam
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from math import sqrt
    import optuna
    import random

    # --- Reprodutibilidade ---
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)

    # --- Preparação dos Dados ---
    print("--- PREPARANDO DADOS PARA OTIMIZAÇÃO E TREINO DO LSTM ---")
    df = new_data.loc[new_data['Month'].isin(months)].reset_index(drop=True).dropna()
    values = df['LV ActivePower (kW)'].values.reshape(-1, 1)

    train_size = int(len(values) * data_partition)
    train_values, test_values = values[:train_size], values[train_size:]

    def create_dataset_multi(dataset, look_back, horizon):
        X, Y = [], []
        for j in range(len(dataset) - look_back - horizon + 1):
            X.append(dataset[j:j + look_back, 0])
            Y.append(dataset[j + look_back:j + look_back + horizon, 0])
        return np.array(X), np.array(Y)

    trainX_full, trainY_full = create_dataset_multi(train_values, look_back, horizon)
    testX, testY = create_dataset_multi(test_values, look_back, horizon)
    print(f"Dados preparados: {trainX_full.shape[0]} amostras de treino, {testX.shape[0]} amostras de teste.")

    # --- Função Objetivo do Optuna ---
    def objective(trial):
        # Espaço de busca para a arquitetura LSTM
        lstm_units = trial.suggest_categorical('lstm_units', [32, 64, 128, 256])
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])

        tscv = TimeSeriesSplit(n_splits=3)
        mape_scores = []

        for train_idx, val_idx in tscv.split(trainX_full):
            X_tr_raw, X_val_raw = trainX_full[train_idx], trainX_full[val_idx]
            y_tr_raw, y_val_raw = trainY_full[train_idx], trainY_full[val_idx]

            sc_X_fold = StandardScaler().fit(X_tr_raw)
            sc_y_fold = StandardScaler().fit(y_tr_raw)

            X_tr = sc_X_fold.transform(X_tr_raw).reshape(-1, look_back, 1)
            y_tr = sc_y_fold.transform(y_tr_raw)
            X_val = sc_X_fold.transform(X_val_raw).reshape(-1, look_back, 1)
            y_val = sc_y_fold.transform(y_val_raw)

            model = Sequential([
                LSTM(units=lstm_units, input_shape=(look_back, 1)),
                Dropout(dropout_rate),
                Dense(horizon)
            ])
            model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')
            early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=0)
            model.fit(X_tr, y_tr, batch_size=batch_size, epochs=5000,
                      validation_data=(X_val, y_val),
                      shuffle=False, verbose=0, callbacks=[early_stop])

            preds_scaled = model.predict(X_val, verbose=0)
            preds_inv = sc_y_fold.inverse_transform(preds_scaled)
            mape_cap = np.mean(np.abs(y_val_raw - preds_inv) / cap) * 100
            mape_scores.append(mape_cap)
            tf.keras.backend.clear_session()

        return np.mean(mape_scores)

    # --- Execução da Otimização ---
    print("\n--- INICIANDO BUSCA DE HIPERPARÂMETROS PARA LSTM COM OPTUNA ---")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    print("\n" + "="*50)
    print("OTIMIZAÇÃO DO LSTM CONCLUÍDA")
    print("="*50)
    print(f"Melhor resultado (MAPE médio na validação): {study.best_value:.4f}%")
    print("\nHiperparâmetros Ótimos Encontrados:")
    for key, value in study.best_params.items():
        print(f"  - {key}: {value}")
    print("="*50)

    # --- Treinamento Final com os Melhores Parâmetros ---
    print("\n--- TREINANDO MODELO LSTM FINAL COM OS MELHORES PARÂMETROS ---")
    sc_X_final = StandardScaler().fit(trainX_full)
    sc_y_final = StandardScaler().fit(trainY_full)

    X_train_scaled = sc_X_final.transform(trainX_full).reshape(-1, look_back, 1)
    y_train_scaled = sc_y_final.transform(trainY_full)
    X_test_scaled = sc_X_final.transform(testX).reshape(-1, look_back, 1)

    final_model = Sequential([
        LSTM(units=best_params['lstm_units'], input_shape=(look_back, 1)),
        Dropout(best_params['dropout_rate']),
        Dense(horizon)
    ])
    final_model.compile(optimizer=Adam(learning_rate=best_params['learning_rate']), loss='mean_squared_error')
    
    early_stop_final = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)
    final_model.fit(X_train_scaled, y_train_scaled,
                    batch_size=best_params['batch_size'],
                    epochs=5000,
                    validation_split=0.1,
                    shuffle=False, verbose=0, callbacks=[early_stop_final])
    print("Treinamento final concluído.")

    # --- Avaliação Final no Conjunto de Teste ---
    y_pred_test = sc_y_final.inverse_transform(final_model.predict(X_test_scaled, verbose=0))
    y_test_inv = testY

    metrics_data = []
    print("\n===== MÉTRICAS FINAIS NO CONJUNTO DE TESTE =====")
    for h in range(horizon):
        mape = np.mean(np.abs((y_test_inv[:, h] - y_pred_test[:, h]) / cap)) * 100
        rmse = sqrt(mean_squared_error(y_test_inv[:, h], y_pred_test[:, h]))
        mae = mean_absolute_error(y_test_inv[:, h], y_pred_test[:, h])
        r2 = r2_score(y_test_inv[:, h], y_pred_test[:, h])
        metrics_data.append({'Horizonte': f't+{(h+1)*10} min', 'MAPE (%)': mape, 'RMSE (MW)': rmse, 'MAE (MW)': mae, 'R²': r2})
        print(f"Horizonte t+{(h+1)*10}: MAPE={mape:.4f}%, RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")

    # --- Plotagem dos Resultados Finais ---
    print("\n--- GERANDO GRÁFICOS DE RESULTADOS ---")
    for h in range(horizon):
        plt.figure(figsize=(15, 6))
        plt.plot(y_test_inv[:, h], label=f"Real t+{(h+1)*10}", color="blue", alpha=0.8)
        plt.plot(y_pred_test[:, h], label=f"Previsto (Otimizado) t+{(h+1)*10}", color="red", linestyle="--")
        plt.title(f"Previsão LSTM Otimizado vs. Real - Horizonte t+{(h+1)*10} min")
        plt.xlabel("Amostras de Teste")
        plt.ylabel("LV ActivePower (kW)")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

    return final_model, best_params, pd.DataFrame(metrics_data)




    
#-------------#

## BILSTM MULTI-HORIZON (direto)
# Em myfunctions.py

def bilstm_model(new_data, months, look_back, data_partition, cap, horizon=3, strategy_horizon=20, mode='train'):
    """
    Modelo Bi-LSTM adaptado para previsão quantílica, seguindo o padrão do dashboard.
    - Opera em modos 'train' e 'inference'.
    - Utiliza funções de perda quantílica para treinar modelos para os quantis 0.1, 0.5 e 0.9.
    - Aplica correção de cruzamento de quantis.
    - Salva/carrega modelos e scalers para integração com o dashboard.
    - Gera métricas e gráficos de ORI no modo 'train'.
    """
    # --- Imports ---
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from math import sqrt
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional
    from tensorflow.keras.callbacks import EarlyStopping
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    import random
    import os
    import joblib

    # --- Reprodutibilidade e Configurações ---
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)
    MODEL_DIR = "saved_models/bilstm_model" # Diretório específico para este modelo

    # --- Funções de Perda (essenciais para carregar o modelo) ---
    # Reutilizando as funções de perda nomeadas já definidas no arquivo
    # (quantile_loss_p10, quantile_loss_p50, quantile_loss_p90, loss_functions)

    # --- Preparação dos Dados ---
    df = new_data.loc[new_data['Month'].isin(months)].reset_index(drop=True).dropna()
    values = df['LV ActivePower (kW)'].values.reshape(-1, 1)
    train_size = int(len(values) * data_partition)
    train, test = values[:train_size], values[train_size:]

    def create_dataset_multi(dataset, look_back, horizon):
        X, Y = [], []
        for j in range(len(dataset) - look_back - horizon + 1):
            X.append(dataset[j:j + look_back, 0])
            Y.append(dataset[j + look_back:j + look_back + horizon, 0])
        return np.array(X), np.array(Y)

    trainX, trainY = create_dataset_multi(train, look_back, horizon)
    testX, testY = create_dataset_multi(test, look_back, horizon)
    y_test_inv = testY

    all_predictions = {}
    quantiles = [0.1, 0.5, 0.9]

    if mode == 'train':
        print("--- EXECUTANDO BI-LSTM EM MODO DE TREINamento ---")
        sc_X = StandardScaler().fit(trainX)
        sc_y = StandardScaler().fit(trainY)
        X_train_scaled, y_train_scaled = sc_X.transform(trainX), sc_y.transform(trainY)
        X_test_scaled = sc_X.transform(testX)

        # Reshape para o formato 3D do Bi-LSTM
        X_train = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
        X_test = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))

        for q in quantiles:
            print(f"Treinando modelo Bi-LSTM para o quantil: {q}")
            
            model = Sequential([
                Bidirectional(LSTM(units=256, input_shape=(look_back, 1))),
                Dropout(0.37797400205329756),
                Dense(horizon)
            ])
            model.compile(optimizer=Adam(learning_rate=0.0009463754587277058), loss=loss_functions[q])
            
            early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
            model.fit(X_train, y_train_scaled, epochs=5000, batch_size=128, validation_split=0.1, shuffle=False, verbose=0, callbacks=[early_stop])
            
            y_pred_scaled = model.predict(X_test)
            y_pred_inv = sc_y.inverse_transform(y_pred_scaled)
            all_predictions[q] = np.clip(y_pred_inv, 0, cap)

            os.makedirs(MODEL_DIR, exist_ok=True)
            model_path = os.path.join(MODEL_DIR, f"model_q{str(q).replace('.', '')}.keras")
            model.save(model_path)
            print(f"Modelo Bi-LSTM salvo em: {model_path}")

        joblib.dump(sc_X, os.path.join(MODEL_DIR, "scaler_X.gz"))
        joblib.dump(sc_y, os.path.join(MODEL_DIR, "scaler_y.gz"))
        print("Scalers para Bi-LSTM salvos com sucesso.")

    elif mode == 'inference':
        print("--- EXECUTANDO BI-LSTM EM MODO DE INFERÊNCIA ---")
        if not os.path.exists(MODEL_DIR):
            raise FileNotFoundError(f"Diretório de modelos Bi-LSTM '{MODEL_DIR}' não encontrado.")

        sc_X = joblib.load(os.path.join(MODEL_DIR, "scaler_X.gz"))
        sc_y = joblib.load(os.path.join(MODEL_DIR, "scaler_y.gz"))
        X_test_scaled = sc_X.transform(testX)
        X_test = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))
        
        custom_objects = {fn.__name__: fn for fn in loss_functions.values()}

        for q in quantiles:
            model_path = os.path.join(MODEL_DIR, f"model_q{str(q).replace('.', '')}.keras")
            model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
            
            y_pred_scaled = model.predict(X_test)
            y_pred_inv = sc_y.inverse_transform(y_pred_scaled)
            all_predictions[q] = np.clip(y_pred_inv, 0, cap)

    else:
        raise ValueError("O parâmetro 'mode' deve ser 'train' ou 'inference'.")

    # Correção de cruzamento de quantis
    print("Aplicando correção de cruzamento de quantis...")
    all_predictions[0.5] = np.maximum(all_predictions[0.1], all_predictions[0.5])
    all_predictions[0.9] = np.maximum(all_predictions[0.5], all_predictions[0.9])

    if mode == 'train':
        y_pred_median = all_predictions[0.5]
        print("\n===== Métricas Completas Bi-LSTM (baseadas na previsão da mediana q=0.5) =====")
        for h in range(horizon):
            y_true_h, y_pred_h = y_test_inv[:, h], y_pred_median[:, h]
            mape = np.mean(np.abs((y_true_h - y_pred_h) / cap)) * 100 if cap > 0 else 0
            rmse = sqrt(mean_squared_error(y_true_h, y_pred_h))
            mae = mean_absolute_error(y_true_h, y_pred_h)
            r2 = r2_score(y_true_h, y_pred_h)
            print(f"Horizonte t+{(h+1)*10}: MAPE={mape:.4f}%, RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")

        # Lógica de Risco ORI
        operational_threshold = 0.3 * cap 
        p_low_t20 = all_predictions[0.1][:, 1]
        p_low_t30 = all_predictions[0.1][:, 2]
        ori_levels = []
        if strategy_horizon == 20:
            for i in range(len(p_low_t20)):
                if p_low_t20[i] < operational_threshold: ori_levels.append('Alto')
                elif p_low_t30[i] < operational_threshold: ori_levels.append('Atenção')
                else: ori_levels.append('Baixo')
        else:
            for p_low in p_low_t30:
                ori_levels.append('Alto' if p_low < operational_threshold else 'Baixo')

        color_map = {'Baixo': 'lightgreen', 'Atenção': 'gold', 'Alto': 'salmon'}
        num_samples_to_plot = 300
        time_axis = np.arange(min(num_samples_to_plot, len(y_test_inv)))

        for h in range(horizon):
            fig, ax = plt.subplots(figsize=(18, 8))
            for i in range(len(time_axis)):
                ax.axvspan(i, i + 1, facecolor=color_map.get(ori_levels[i], 'white'), alpha=0.5, zorder=0)
            ax.axhline(y=operational_threshold, color='red', linestyle=':', linewidth=2.5, label=f'Limiar Operacional ({operational_threshold:.2f} kW)', zorder=4)
            ax.fill_between(time_axis, all_predictions[0.1][:len(time_axis), h], all_predictions[0.9][:len(time_axis), h], color='cornflowerblue', alpha=0.6, label='Intervalo de Confiança (80%)', zorder=1)
            ax.plot(time_axis, y_test_inv[:len(time_axis), h], label="Real", color='black', linewidth=2, zorder=3)
            ax.plot(time_axis, all_predictions[0.5][:len(time_axis), h], label="Previsão Mediana", color='firebrick', linestyle='--', linewidth=2.5, zorder=2)
            ax.set_title(f'Previsão Quantílica Bi-LSTM com ORI - Horizonte t+{(h+1)*10} min', fontsize=16)
            ax.set_ylabel('LV ActivePower (kW)', fontsize=12)
            ax.set_xlabel('Amostras de Teste', fontsize=12)
            risk_patches = [Patch(facecolor=color, alpha=0.5, label=f'Risco {level}') for level, color in color_map.items() if level in set(ori_levels)]
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles=handles + risk_patches, loc='best', fontsize=12)
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
            ax.set_xlim(0, len(time_axis))
            ax.set_ylim(bottom=max(0, y_test_inv.min() - (0.05 * cap)))
            plt.tight_layout()
            plt.show()
            plt.close(fig)

    return all_predictions, y_test_inv


#-----#
# Em myfunctions.py

# A função principal `bilstm_model` permanece inalterada.
# A função `bilstm_model_with_optuna` é ajustada para otimizar, treinar e avaliar, mas sem salvar.

def bilstm_model_with_optuna(new_data, months, look_back, data_partition, cap, horizon=3, n_trials=30):
    """
    Otimiza, treina e avalia um modelo Bi-LSTM determinístico.
    - DESCobre os hiperparâmetros ótimos com Optuna e TimeSeriesSplit.
    - TREINA um modelo final com os melhores parâmetros em todo o conjunto de treino.
    - AVALIA e plota os resultados no conjunto de teste.
    - NÃO salva o modelo ou os scalers no disco.
    """
    # --- Imports ---
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.optimizers import Adam
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from math import sqrt
    import optuna
    import random

    # --- Reprodutibilidade ---
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)

    # --- Preparação dos Dados ---
    print("--- PREPARANDO DADOS PARA OTIMIZAÇÃO E TREINO DO BI-LSTM ---")
    df = new_data.loc[new_data['Month'].isin(months)].reset_index(drop=True).dropna()
    values = df['LV ActivePower (kW)'].values.reshape(-1, 1)

    train_size = int(len(values) * data_partition)
    train_values, test_values = values[:train_size], values[train_size:]

    def create_dataset_multi(dataset, look_back, horizon):
        X, Y = [], []
        for j in range(len(dataset) - look_back - horizon + 1):
            X.append(dataset[j:j + look_back, 0])
            Y.append(dataset[j + look_back:j + look_back + horizon, 0])
        return np.array(X), np.array(Y)

    trainX_full, trainY_full = create_dataset_multi(train_values, look_back, horizon)
    testX, testY = create_dataset_multi(test_values, look_back, horizon)
    print(f"Dados preparados: {trainX_full.shape[0]} amostras de treino, {testX.shape[0]} amostras de teste.")

    # --- Função Objetivo do Optuna ---
    def objective(trial):
        lstm_units = trial.suggest_categorical('lstm_units', [32, 64, 128, 256])
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])

        tscv = TimeSeriesSplit(n_splits=3)
        mape_scores = []

        for train_idx, val_idx in tscv.split(trainX_full):
            X_tr_raw, X_val_raw = trainX_full[train_idx], trainX_full[val_idx]
            y_tr_raw, y_val_raw = trainY_full[train_idx], trainY_full[val_idx]

            sc_X_fold = StandardScaler().fit(X_tr_raw)
            sc_y_fold = StandardScaler().fit(y_tr_raw)

            X_tr = sc_X_fold.transform(X_tr_raw).reshape(-1, look_back, 1)
            y_tr = sc_y_fold.transform(y_tr_raw)
            X_val = sc_X_fold.transform(X_val_raw).reshape(-1, look_back, 1)
            y_val = sc_y_fold.transform(y_val_raw)

            model = Sequential([
                Bidirectional(LSTM(units=lstm_units, input_shape=(look_back, 1))),
                Dropout(dropout_rate),
                Dense(horizon)
            ])
            model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')
            early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=0)
            model.fit(X_tr, y_tr, batch_size=batch_size, epochs=5000,
                      validation_data=(X_val, y_val),
                      shuffle=False, verbose=0, callbacks=[early_stop])

            preds_scaled = model.predict(X_val, verbose=0)
            preds_inv = sc_y_fold.inverse_transform(preds_scaled)
            mape_cap = np.mean(np.abs(y_val_raw - preds_inv) / cap) * 100
            mape_scores.append(mape_cap)
            tf.keras.backend.clear_session()

        return np.mean(mape_scores)

    # --- Execução da Otimização ---
    print("\n--- INICIANDO BUSCA DE HIPERPARÂMETROS PARA BI-LSTM COM OPTUNA ---")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    print("\n" + "="*50)
    print("OTIMIZAÇÃO DO BI-LSTM CONCLUÍDA")
    print("="*50)
    print(f"Melhor resultado (MAPE médio na validação): {study.best_value:.4f}%")
    print("\nHiperparâmetros Ótimos Encontrados:")
    for key, value in study.best_params.items():
        print(f"  - {key}: {value}")
    print("="*50)

    # --- Treinamento Final com os Melhores Parâmetros ---
    print("\n--- TREINANDO MODELO BI-LSTM FINAL COM OS MELHORES PARÂMETROS ---")
    sc_X_final = StandardScaler().fit(trainX_full)
    sc_y_final = StandardScaler().fit(trainY_full)

    X_train_scaled = sc_X_final.transform(trainX_full).reshape(-1, look_back, 1)
    y_train_scaled = sc_y_final.transform(trainY_full)
    X_test_scaled = sc_X_final.transform(testX).reshape(-1, look_back, 1)

    final_model = Sequential([
        Bidirectional(LSTM(units=best_params['lstm_units'], input_shape=(look_back, 1))),
        Dropout(best_params['dropout_rate']),
        Dense(horizon)
    ])
    final_model.compile(optimizer=Adam(learning_rate=best_params['learning_rate']), loss='mean_squared_error')
    
    early_stop_final = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)
    final_model.fit(X_train_scaled, y_train_scaled,
                    batch_size=best_params['batch_size'],
                    epochs=5000,
                    validation_split=0.1,
                    shuffle=False, verbose=0, callbacks=[early_stop_final])
    print("Treinamento final concluído.")

    # --- Avaliação Final no Conjunto de Teste ---
    y_pred_test = sc_y_final.inverse_transform(final_model.predict(X_test_scaled, verbose=0))
    y_test_inv = testY

    metrics_data = []
    print("\n===== MÉTRICAS FINAIS NO CONJUNTO DE TESTE =====")
    for h in range(horizon):
        mape = np.mean(np.abs((y_test_inv[:, h] - y_pred_test[:, h]) / cap)) * 100
        rmse = sqrt(mean_squared_error(y_test_inv[:, h], y_pred_test[:, h]))
        mae = mean_absolute_error(y_test_inv[:, h], y_pred_test[:, h])
        r2 = r2_score(y_test_inv[:, h], y_pred_test[:, h])
        metrics_data.append({'Horizonte': f't+{(h+1)*10} min', 'MAPE (%)': mape, 'RMSE (MW)': rmse, 'MAE (MW)': mae, 'R²': r2})
        print(f"Horizonte t+{(h+1)*10}: MAPE={mape:.4f}%, RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")

    # --- Plotagem dos Resultados Finais ---
    print("\n--- GERANDO GRÁFICOS DE RESULTADOS ---")
    for h in range(horizon):
        plt.figure(figsize=(15, 6))
        plt.plot(y_test_inv[:, h], label=f"Real t+{(h+1)*10}", color="blue", alpha=0.8)
        plt.plot(y_pred_test[:, h], label=f"Previsto (Otimizado) t+{(h+1)*10}", color="red", linestyle="--")
        plt.title(f"Previsão Bi-LSTM Otimizado vs. Real - Horizonte t+{(h+1)*10} min")
        plt.xlabel("Amostras de Teste")
        plt.ylabel("LV ActivePower (kW)")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

    return final_model, best_params, pd.DataFrame(metrics_data)



#----#

# Em myfunctions.py

def bilstm_att_model(new_data, months, look_back, data_partition, cap, horizon=3, strategy_horizon=20, mode='train'):
    """
    Modelo Bi-LSTM com Attention adaptado para previsão quantílica, seguindo o padrão do dashboard.
    - Utiliza a camada de atenção oficial do Keras (tf.keras.layers.Attention) para maior eficiência e robustez.
    - Opera em modos 'train' e 'inference'.
    - Utiliza funções de perda quantílica para treinar modelos para os quantis 0.1, 0.5 e 0.9.
    - Aplica correção de cruzamento de quantis.
    - Salva/carrega modelos e scalers para integração com o dashboard.
    - Gera métricas e gráficos de ORI no modo 'train'.
    """
    # --- Imports ---
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from math import sqrt
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, Bidirectional, Attention
    from tensorflow.keras.callbacks import EarlyStopping
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    import random
    import os
    import joblib

    # --- Reprodutibilidade e Configurações ---
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)
    MODEL_DIR = "saved_models/bilstm_att_model" # Diretório específico

    # --- Funções de Perda (essenciais para carregar o modelo) ---
    # Reutilizando as funções de perda nomeadas já definidas no arquivo

    # --- Preparação dos Dados ---
    df = new_data.loc[new_data['Month'].isin(months)].reset_index(drop=True).dropna()
    values = df['LV ActivePower (kW)'].values.reshape(-1, 1)
    train_size = int(len(values) * data_partition)
    train, test = values[:train_size], values[train_size:]

    def create_dataset_multi(dataset, look_back, horizon):
        X, Y = [], []
        for j in range(len(dataset) - look_back - horizon + 1):
            X.append(dataset[j:j + look_back, 0])
            Y.append(dataset[j + look_back:j + look_back + horizon, 0])
        return np.array(X), np.array(Y)

    trainX, trainY = create_dataset_multi(train, look_back, horizon)
    testX, testY = create_dataset_multi(test, look_back, horizon)
    y_test_inv = testY

    all_predictions = {}
    quantiles = [0.1, 0.5, 0.9]

    if mode == 'train':
        print("--- EXECUTANDO BI-LSTM+ATTENTION EM MODO DE TREINAMENTO ---")
        sc_X = StandardScaler().fit(trainX)
        sc_y = StandardScaler().fit(trainY)
        X_train_scaled, y_train_scaled = sc_X.transform(trainX), sc_y.transform(trainY)
        X_test_scaled = sc_X.transform(testX)

        X_train = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
        X_test = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))

        for q in quantiles:
            print(f"Treinando modelo Bi-LSTM+Attention para o quantil: {q}")
            
            # Arquitetura com a camada de atenção oficial do Keras
            input_layer = Input(shape=(look_back, 1))
            # return_sequences=True é crucial para a camada de atenção funcionar
            lstm_out = Bidirectional(LSTM(units=32, return_sequences=True))(input_layer)
            
            # A camada de atenção do Keras espera uma lista [query, value]
            # No self-attention, query e value são a mesma coisa: a saída do LSTM
            attention_out = Attention()([lstm_out, lstm_out])
            
            # A saída da atenção precisa ser achatada ou agregada.
            # Uma camada LSTM com return_sequences=False pode fazer essa agregação.
            # Vamos usar uma segunda LSTM para processar a sequência ponderada pela atenção.
            lstm_agg = LSTM(units=32, return_sequences=False)(attention_out)
            
            drop = Dropout(0.2)(lstm_agg)
            output_layer = Dense(horizon)(drop)
            
            model = Model(inputs=input_layer, outputs=output_layer)
            model.compile(optimizer='adam', loss=loss_functions[q])
            
            early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
            model.fit(X_train, y_train_scaled, epochs=5000, batch_size=64, validation_split=0.1, shuffle=False, verbose=0, callbacks=[early_stop])
            
            y_pred_scaled = model.predict(X_test)
            y_pred_inv = sc_y.inverse_transform(y_pred_scaled)
            all_predictions[q] = np.clip(y_pred_inv, 0, cap)

            os.makedirs(MODEL_DIR, exist_ok=True)
            model_path = os.path.join(MODEL_DIR, f"model_q{str(q).replace('.', '')}.keras")
            model.save(model_path)
            print(f"Modelo Bi-LSTM+Attention salvo em: {model_path}")

        joblib.dump(sc_X, os.path.join(MODEL_DIR, "scaler_X.gz"))
        joblib.dump(sc_y, os.path.join(MODEL_DIR, "scaler_y.gz"))
        print("Scalers para Bi-LSTM+Attention salvos com sucesso.")

    elif mode == 'inference':
        print("--- EXECUTANDO BI-LSTM+ATTENTION EM MODO DE INFERÊNCIA ---")
        if not os.path.exists(MODEL_DIR):
            raise FileNotFoundError(f"Diretório de modelos Bi-LSTM+Attention '{MODEL_DIR}' não encontrado.")

        sc_X = joblib.load(os.path.join(MODEL_DIR, "scaler_X.gz"))
        sc_y = joblib.load(os.path.join(MODEL_DIR, "scaler_y.gz"))
        X_test_scaled = sc_X.transform(testX)
        X_test = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))
        
        custom_objects = {fn.__name__: fn for fn in loss_functions.values()}

        for q in quantiles:
            model_path = os.path.join(MODEL_DIR, f"model_q{str(q).replace('.', '')}.keras")
            model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
            
            y_pred_scaled = model.predict(X_test)
            y_pred_inv = sc_y.inverse_transform(y_pred_scaled)
            all_predictions[q] = np.clip(y_pred_inv, 0, cap)

    else:
        raise ValueError("O parâmetro 'mode' deve ser 'train' ou 'inference'.")

    # Correção de cruzamento de quantis
    print("Aplicando correção de cruzamento de quantis...")
    all_predictions[0.5] = np.maximum(all_predictions[0.1], all_predictions[0.5])
    all_predictions[0.9] = np.maximum(all_predictions[0.5], all_predictions[0.9])

    if mode == 'train':
        y_pred_median = all_predictions[0.5]
        print("\n===== Métricas Completas Bi-LSTM+Attention (baseadas na mediana q=0.5) =====")
        for h in range(horizon):
            y_true_h, y_pred_h = y_test_inv[:, h], y_pred_median[:, h]
            mape = np.mean(np.abs((y_true_h - y_pred_h) / cap)) * 100 if cap > 0 else 0
            rmse = sqrt(mean_squared_error(y_true_h, y_pred_h))
            mae = mean_absolute_error(y_true_h, y_pred_h)
            r2 = r2_score(y_true_h, y_pred_h)
            print(f"Horizonte t+{(h+1)*10}: MAPE={mape:.4f}%, RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")

        # Lógica de Risco ORI
        operational_threshold = 0.3 * cap 
        p_low_t20 = all_predictions[0.1][:, 1]
        p_low_t30 = all_predictions[0.1][:, 2]
        ori_levels = []
        if strategy_horizon == 20:
            for i in range(len(p_low_t20)):
                if p_low_t20[i] < operational_threshold: ori_levels.append('Alto')
                elif p_low_t30[i] < operational_threshold: ori_levels.append('Atenção')
                else: ori_levels.append('Baixo')
        else:
            for p_low in p_low_t30:
                ori_levels.append('Alto' if p_low < operational_threshold else 'Baixo')

        color_map = {'Baixo': 'lightgreen', 'Atenção': 'gold', 'Alto': 'salmon'}
        num_samples_to_plot = 300
        time_axis = np.arange(min(num_samples_to_plot, len(y_test_inv)))

        for h in range(horizon):
            fig, ax = plt.subplots(figsize=(18, 8))
            for i in range(len(time_axis)):
                ax.axvspan(i, i + 1, facecolor=color_map.get(ori_levels[i], 'white'), alpha=0.5, zorder=0)
            ax.axhline(y=operational_threshold, color='red', linestyle=':', linewidth=2.5, label=f'Limiar Operacional ({operational_threshold:.2f} kW)', zorder=4)
            ax.fill_between(time_axis, all_predictions[0.1][:len(time_axis), h], all_predictions[0.9][:len(time_axis), h], color='cornflowerblue', alpha=0.6, label='Intervalo de Confiança (80%)', zorder=1)
            ax.plot(time_axis, y_test_inv[:len(time_axis), h], label="Real", color='black', linewidth=2, zorder=3)
            ax.plot(time_axis, all_predictions[0.5][:len(time_axis), h], label="Previsão Mediana", color='firebrick', linestyle='--', linewidth=2.5, zorder=2)
            ax.set_title(f'Previsão Quantílica Bi-LSTM+Attention com ORI - Horizonte t+{(h+1)*10} min', fontsize=16)
            ax.set_ylabel('LV ActivePower (kW)', fontsize=12)
            ax.set_xlabel('Amostras de Teste', fontsize=12)
            risk_patches = [Patch(facecolor=color, alpha=0.5, label=f'Risco {level}') for level, color in color_map.items() if level in set(ori_levels)]
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles=handles + risk_patches, loc='best', fontsize=12)
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
            ax.set_xlim(0, len(time_axis))
            ax.set_ylim(bottom=max(0, y_test_inv.min() - (0.05 * cap)))
            plt.tight_layout()
            plt.show()
            plt.close(fig)

    return all_predictions, y_test_inv




#-------#
# Em myfunctions.py

# A função principal `bilstm_att_model` permanece inalterada.
# A função `bilstm_att_model_with_optuna` é ajustada para otimizar, treinar e avaliar, mas sem salvar.

def bilstm_att_model_with_optuna(new_data, months, look_back, data_partition, cap, horizon=3, n_trials=30):
    """
    Otimiza, treina e avalia um modelo Bi-LSTM com Attention determinístico.
    - DESCobre os hiperparâmetros ótimos com Optuna e TimeSeriesSplit.
    - TREINA um modelo final com os melhores parâmetros em todo o conjunto de treino.
    - AVALIA e plota os resultados no conjunto de teste.
    - NÃO salva o modelo ou os scalers no disco.
    """
    # --- Imports ---
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, Bidirectional, Attention
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.optimizers import Adam
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from math import sqrt
    import optuna
    import random

    # --- Reprodutibilidade ---
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)

    # --- Preparação dos Dados ---
    print("--- PREPARANDO DADOS PARA OTIMIZAÇÃO E TREINO DO BI-LSTM+ATTENTION ---")
    df = new_data.loc[new_data['Month'].isin(months)].reset_index(drop=True).dropna()
    values = df['LV ActivePower (kW)'].values.reshape(-1, 1)

    train_size = int(len(values) * data_partition)
    train_values, test_values = values[:train_size], values[train_size:]

    def create_dataset_multi(dataset, look_back, horizon):
        X, Y = [], []
        for j in range(len(dataset) - look_back - horizon + 1):
            X.append(dataset[j:j + look_back, 0])
            Y.append(dataset[j + look_back:j + look_back + horizon, 0])
        return np.array(X), np.array(Y)

    trainX_full, trainY_full = create_dataset_multi(train_values, look_back, horizon)
    testX, testY = create_dataset_multi(test_values, look_back, horizon)
    print(f"Dados preparados: {trainX_full.shape[0]} amostras de treino, {testX.shape[0]} amostras de teste.")

    # --- Função Objetivo do Optuna ---
    def objective(trial):
        # Espaço de busca para a arquitetura Bi-LSTM + Attention
        bilstm_units = trial.suggest_categorical('bilstm_units', [32, 64, 128])
        agg_lstm_units = trial.suggest_categorical('agg_lstm_units', [16, 32, 64])
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])

        tscv = TimeSeriesSplit(n_splits=3)
        mape_scores = []

        for train_idx, val_idx in tscv.split(trainX_full):
            X_tr_raw, X_val_raw = trainX_full[train_idx], trainX_full[val_idx]
            y_tr_raw, y_val_raw = trainY_full[train_idx], trainY_full[val_idx]

            sc_X_fold = StandardScaler().fit(X_tr_raw)
            sc_y_fold = StandardScaler().fit(y_tr_raw)

            X_tr = sc_X_fold.transform(X_tr_raw).reshape(-1, look_back, 1)
            y_tr = sc_y_fold.transform(y_tr_raw)
            X_val = sc_X_fold.transform(X_val_raw).reshape(-1, look_back, 1)
            y_val = sc_y_fold.transform(y_val_raw)

            # Arquitetura usando API Funcional, espelhando o modelo principal
            input_layer = Input(shape=(look_back, 1))
            lstm_out = Bidirectional(LSTM(units=bilstm_units, return_sequences=True))(input_layer)
            attention_out = Attention()([lstm_out, lstm_out])
            lstm_agg = LSTM(units=agg_lstm_units, return_sequences=False)(attention_out)
            drop = Dropout(dropout_rate)(lstm_agg)
            output_layer = Dense(horizon)(drop)
            model = Model(inputs=input_layer, outputs=output_layer)
            
            model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')
            early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=0)
            model.fit(X_tr, y_tr, batch_size=batch_size, epochs=5000,
                      validation_data=(X_val, y_val),
                      shuffle=False, verbose=0, callbacks=[early_stop])

            preds_scaled = model.predict(X_val, verbose=0)
            preds_inv = sc_y_fold.inverse_transform(preds_scaled)
            mape_cap = np.mean(np.abs(y_val_raw - preds_inv) / cap) * 100
            mape_scores.append(mape_cap)
            tf.keras.backend.clear_session()

        return np.mean(mape_scores)

    # --- Execução da Otimização ---
    print("\n--- INICIANDO BUSCA DE HIPERPARÂMETROS PARA BI-LSTM+ATTENTION COM OPTUNA ---")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    print("\n" + "="*50)
    print("OTIMIZAÇÃO DO BI-LSTM+ATTENTION CONCLUÍDA")
    print("="*50)
    print(f"Melhor resultado (MAPE médio na validação): {study.best_value:.4f}%")
    print("\nHiperparâmetros Ótimos Encontrados:")
    for key, value in study.best_params.items():
        print(f"  - {key}: {value}")
    print("="*50)

    # --- Treinamento Final com os Melhores Parâmetros ---
    print("\n--- TREINANDO MODELO BI-LSTM+ATTENTION FINAL COM OS MELHORES PARÂMETROS ---")
    sc_X_final = StandardScaler().fit(trainX_full)
    sc_y_final = StandardScaler().fit(trainY_full)

    X_train_scaled = sc_X_final.transform(trainX_full).reshape(-1, look_back, 1)
    y_train_scaled = sc_y_final.transform(trainY_full)
    X_test_scaled = sc_X_final.transform(testX).reshape(-1, look_back, 1)

    # Constrói o modelo final com a arquitetura e os parâmetros ótimos
    input_layer = Input(shape=(look_back, 1))
    lstm_out = Bidirectional(LSTM(units=best_params['bilstm_units'], return_sequences=True))(input_layer)
    attention_out = Attention()([lstm_out, lstm_out])
    lstm_agg = LSTM(units=best_params['agg_lstm_units'], return_sequences=False)(attention_out)
    drop = Dropout(best_params['dropout_rate'])(lstm_agg)
    output_layer = Dense(horizon)(drop)
    final_model = Model(inputs=input_layer, outputs=output_layer)
    
    final_model.compile(optimizer=Adam(learning_rate=best_params['learning_rate']), loss='mean_squared_error')
    
    early_stop_final = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)
    final_model.fit(X_train_scaled, y_train_scaled,
                    batch_size=best_params['batch_size'],
                    epochs=5000,
                    validation_split=0.1,
                    shuffle=False, verbose=0, callbacks=[early_stop_final])
    print("Treinamento final concluído.")

    # --- Avaliação Final no Conjunto de Teste ---
    y_pred_test = sc_y_final.inverse_transform(final_model.predict(X_test_scaled, verbose=0))
    y_test_inv = testY

    metrics_data = []
    print("\n===== MÉTRICAS FINAIS NO CONJUNTO DE TESTE =====")
    for h in range(horizon):
        mape = np.mean(np.abs((y_test_inv[:, h] - y_pred_test[:, h]) / cap)) * 100
        rmse = sqrt(mean_squared_error(y_test_inv[:, h], y_pred_test[:, h]))
        mae = mean_absolute_error(y_test_inv[:, h], y_pred_test[:, h])
        r2 = r2_score(y_test_inv[:, h], y_pred_test[:, h])
        metrics_data.append({'Horizonte': f't+{(h+1)*10} min', 'MAPE (%)': mape, 'RMSE (MW)': rmse, 'MAE (MW)': mae, 'R²': r2})
        print(f"Horizonte t+{(h+1)*10}: MAPE={mape:.4f}%, RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")

    # --- Plotagem dos Resultados Finais ---
    print("\n--- GERANDO GRÁFICOS DE RESULTADOS ---")
    for h in range(horizon):
        plt.figure(figsize=(15, 6))
        plt.plot(y_test_inv[:, h], label=f"Real t+{(h+1)*10}", color="blue", alpha=0.8)
        plt.plot(y_pred_test[:, h], label=f"Previsto (Otimizado) t+{(h+1)*10}", color="red", linestyle="--")
        plt.title(f"Previsão Bi-LSTM+Attention Otimizado vs. Real - Horizonte t+{(h+1)*10} min")
        plt.xlabel("Amostras de Teste")
        plt.ylabel("LV ActivePower (kW)")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

    return final_model, best_params, pd.DataFrame(metrics_data)





# ===============================
# Modelo CNN + BILSTM Multi-horizonte
# ===============================
# Em myfunctions.py

def cnn_bilstm_model(new_data, months, look_back, data_partition, cap, horizon=3, strategy_horizon=20, mode='train'):
    """
    Modelo CNN-BiLSTM adaptado para previsão quantílica, seguindo o padrão do dashboard.
    - Opera em modos 'train' e 'inference'.
    - Utiliza funções de perda quantílica para treinar modelos para os quantis 0.1, 0.5 e 0.9.
    - Aplica correção de cruzamento de quantis.
    - Salva/carrega modelos e scalers para integração com o dashboard.
    - Gera métricas e gráficos de ORI no modo 'train'.
    """
    # --- Imports ---
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from math import sqrt
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Conv1D, Bidirectional, LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    import random
    import os
    import joblib

    # --- Reprodutibilidade e Configurações ---
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)
    MODEL_DIR = "saved_models/cnn_bilstm_model" # Diretório específico

    # --- Funções de Perda (essenciais para carregar o modelo) ---
    # Reutilizando as funções de perda nomeadas já definidas no arquivo

    # --- Preparação dos Dados ---
    df = new_data.loc[new_data['Month'].isin(months)].reset_index(drop=True).dropna()
    values = df['LV ActivePower (kW)'].values.reshape(-1, 1)
    train_size = int(len(values) * data_partition)
    train, test = values[:train_size], values[train_size:]

    def create_dataset_multi(dataset, look_back, horizon):
        X, Y = [], []
        for j in range(len(dataset) - look_back - horizon + 1):
            X.append(dataset[j:j + look_back, 0])
            Y.append(dataset[j + look_back:j + look_back + horizon, 0])
        return np.array(X), np.array(Y)

    trainX, trainY = create_dataset_multi(train, look_back, horizon)
    testX, testY = create_dataset_multi(test, look_back, horizon)
    y_test_inv = testY

    all_predictions = {}
    quantiles = [0.1, 0.5, 0.9]

    if mode == 'train':
        print("--- EXECUTANDO CNN-BI-LSTM EM MODO DE TREINAMENTO ---")
        sc_X = StandardScaler().fit(trainX)
        sc_y = StandardScaler().fit(trainY)
        X_train_scaled, y_train_scaled = sc_X.transform(trainX), sc_y.transform(trainY)
        X_test_scaled = sc_X.transform(testX)

        X_train = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
        X_test = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))

        for q in quantiles:
            print(f"Treinando modelo CNN-Bi-LSTM para o quantil: {q}")
            
            input_layer = Input(shape=(look_back, 1))
            conv_layer = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(input_layer)
            # A camada BiLSTM precisa de return_sequences=False, pois é a última camada recorrente
            lstm_out = Bidirectional(LSTM(units=128, return_sequences=False))(conv_layer)
            drop_layer = Dropout(0.34611204605260304)(lstm_out)
            output_layer = Dense(horizon)(drop_layer)
            
            model = Model(inputs=input_layer, outputs=output_layer)
            model.compile(optimizer=Adam(learning_rate=0.0006626788668346496), loss=loss_functions[q])
            
            early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
            model.fit(X_train, y_train_scaled, epochs=5000, batch_size=128, validation_split=0.1, shuffle=False, verbose=0, callbacks=[early_stop])
            
            y_pred_scaled = model.predict(X_test)
            y_pred_inv = sc_y.inverse_transform(y_pred_scaled)
            all_predictions[q] = np.clip(y_pred_inv, 0, cap)

            os.makedirs(MODEL_DIR, exist_ok=True)
            model_path = os.path.join(MODEL_DIR, f"model_q{str(q).replace('.', '')}.keras")
            model.save(model_path)
            print(f"Modelo CNN-Bi-LSTM salvo em: {model_path}")

        joblib.dump(sc_X, os.path.join(MODEL_DIR, "scaler_X.gz"))
        joblib.dump(sc_y, os.path.join(MODEL_DIR, "scaler_y.gz"))
        print("Scalers para CNN-Bi-LSTM salvos com sucesso.")

    elif mode == 'inference':
        print("--- EXECUTANDO CNN-BI-LSTM EM MODO DE INFERÊNCIA ---")
        if not os.path.exists(MODEL_DIR):
            raise FileNotFoundError(f"Diretório de modelos CNN-Bi-LSTM '{MODEL_DIR}' não encontrado.")

        sc_X = joblib.load(os.path.join(MODEL_DIR, "scaler_X.gz"))
        sc_y = joblib.load(os.path.join(MODEL_DIR, "scaler_y.gz"))
        X_test_scaled = sc_X.transform(testX)
        X_test = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))
        
        custom_objects = {fn.__name__: fn for fn in loss_functions.values()}

        for q in quantiles:
            model_path = os.path.join(MODEL_DIR, f"model_q{str(q).replace('.', '')}.keras")
            model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
            
            y_pred_scaled = model.predict(X_test)
            y_pred_inv = sc_y.inverse_transform(y_pred_scaled)
            all_predictions[q] = np.clip(y_pred_inv, 0, cap)

    else:
        raise ValueError("O parâmetro 'mode' deve ser 'train' ou 'inference'.")

    # Correção de cruzamento de quantis
    print("Aplicando correção de cruzamento de quantis...")
    all_predictions[0.5] = np.maximum(all_predictions[0.1], all_predictions[0.5])
    all_predictions[0.9] = np.maximum(all_predictions[0.5], all_predictions[0.9])

    if mode == 'train':
        y_pred_median = all_predictions[0.5]
        print("\n===== Métricas Completas CNN-Bi-LSTM (baseadas na mediana q=0.5) =====")
        for h in range(horizon):
            y_true_h, y_pred_h = y_test_inv[:, h], y_pred_median[:, h]
            mape = np.mean(np.abs((y_true_h - y_pred_h) / cap)) * 100 if cap > 0 else 0
            rmse = sqrt(mean_squared_error(y_true_h, y_pred_h))
            mae = mean_absolute_error(y_true_h, y_pred_h)
            r2 = r2_score(y_true_h, y_pred_h)
            print(f"Horizonte t+{(h+1)*10}: MAPE={mape:.4f}%, RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")

        # Lógica de Risco ORI
        operational_threshold = 0.3 * cap 
        p_low_t20 = all_predictions[0.1][:, 1]
        p_low_t30 = all_predictions[0.1][:, 2]
        ori_levels = []
        if strategy_horizon == 20:
            for i in range(len(p_low_t20)):
                if p_low_t20[i] < operational_threshold: ori_levels.append('Alto')
                elif p_low_t30[i] < operational_threshold: ori_levels.append('Atenção')
                else: ori_levels.append('Baixo')
        else:
            for p_low in p_low_t30:
                ori_levels.append('Alto' if p_low < operational_threshold else 'Baixo')

        color_map = {'Baixo': 'lightgreen', 'Atenção': 'gold', 'Alto': 'salmon'}
        num_samples_to_plot = 300
        time_axis = np.arange(min(num_samples_to_plot, len(y_test_inv)))

        for h in range(horizon):
            fig, ax = plt.subplots(figsize=(18, 8))
            for i in range(len(time_axis)):
                ax.axvspan(i, i + 1, facecolor=color_map.get(ori_levels[i], 'white'), alpha=0.5, zorder=0)
            ax.axhline(y=operational_threshold, color='red', linestyle=':', linewidth=2.5, label=f'Limiar Operacional ({operational_threshold:.2f} kW)', zorder=4)
            ax.fill_between(time_axis, all_predictions[0.1][:len(time_axis), h], all_predictions[0.9][:len(time_axis), h], color='cornflowerblue', alpha=0.6, label='Intervalo de Confiança (80%)', zorder=1)
            ax.plot(time_axis, y_test_inv[:len(time_axis), h], label="Real", color='black', linewidth=2, zorder=3)
            ax.plot(time_axis, all_predictions[0.5][:len(time_axis), h], label="Previsão Mediana", color='firebrick', linestyle='--', linewidth=2.5, zorder=2)
            ax.set_title(f'Previsão Quantílica CNN-Bi-LSTM com ORI - Horizonte t+{(h+1)*10} min', fontsize=16)
            ax.set_ylabel('LV ActivePower (kW)', fontsize=12)
            ax.set_xlabel('Amostras de Teste', fontsize=12)
            risk_patches = [Patch(facecolor=color, alpha=0.5, label=f'Risco {level}') for level, color in color_map.items() if level in set(ori_levels)]
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles=handles + risk_patches, loc='best', fontsize=12)
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
            ax.set_xlim(0, len(time_axis))
            ax.set_ylim(bottom=max(0, y_test_inv.min() - (0.05 * cap)))
            plt.tight_layout()
            plt.show()
            plt.close(fig)

    return all_predictions, y_test_inv



# Em myfunctions.py

# A função principal `cnn_bilstm_model` permanece inalterada.
# A função `cnn_bilstm_model_with_optuna` é ajustada para otimizar, treinar e avaliar, mas sem salvar.

def cnn_bilstm_model_with_optuna(new_data, months, look_back, data_partition, cap, horizon=3, n_trials=30):
    """
    Otimiza, treina e avalia um modelo CNN-BiLSTM determinístico.
    - DESCobre os hiperparâmetros ótimos com Optuna e TimeSeriesSplit.
    - TREINA um modelo final com os melhores parâmetros em todo o conjunto de treino.
    - AVALIA e plota os resultados no conjunto de teste.
    - NÃO salva o modelo ou os scalers no disco.
    """
    # --- Imports ---
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Conv1D, Bidirectional, LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.optimizers import Adam
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from math import sqrt
    import optuna
    import random

    # --- Reprodutibilidade ---
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)

    # --- Preparação dos Dados ---
    print("--- PREPARANDO DADOS PARA OTIMIZAÇÃO E TREINO DO CNN-BI-LSTM ---")
    df = new_data.loc[new_data['Month'].isin(months)].reset_index(drop=True).dropna()
    values = df['LV ActivePower (kW)'].values.reshape(-1, 1)

    train_size = int(len(values) * data_partition)
    train_values, test_values = values[:train_size], values[train_size:]

    def create_dataset_multi(dataset, look_back, horizon):
        X, Y = [], []
        for j in range(len(dataset) - look_back - horizon + 1):
            X.append(dataset[j:j + look_back, 0])
            Y.append(dataset[j + look_back:j + look_back + horizon, 0])
        return np.array(X), np.array(Y)

    trainX_full, trainY_full = create_dataset_multi(train_values, look_back, horizon)
    testX, testY = create_dataset_multi(test_values, look_back, horizon)
    print(f"Dados preparados: {trainX_full.shape[0]} amostras de treino, {testX.shape[0]} amostras de teste.")

    # --- Função Objetivo do Optuna ---
    def objective(trial):
        # Espaço de busca para a arquitetura CNN-BiLSTM
        cnn_filters = trial.suggest_categorical('cnn_filters', [32, 64, 128])
        kernel_size = trial.suggest_categorical('kernel_size', [2, 3, 5])
        lstm_units = trial.suggest_categorical('lstm_units', [32, 64, 128])
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])

        tscv = TimeSeriesSplit(n_splits=3)
        mape_scores = []

        for train_idx, val_idx in tscv.split(trainX_full):
            X_tr_raw, X_val_raw = trainX_full[train_idx], trainX_full[val_idx]
            y_tr_raw, y_val_raw = trainY_full[train_idx], trainY_full[val_idx]

            sc_X_fold = StandardScaler().fit(X_tr_raw)
            sc_y_fold = StandardScaler().fit(y_tr_raw)

            X_tr = sc_X_fold.transform(X_tr_raw).reshape(-1, look_back, 1)
            y_tr = sc_y_fold.transform(y_tr_raw)
            X_val = sc_X_fold.transform(X_val_raw).reshape(-1, look_back, 1)
            y_val = sc_y_fold.transform(y_val_raw)

            input_layer = Input(shape=(look_back, 1))
            conv_layer = Conv1D(filters=cnn_filters, kernel_size=kernel_size, activation='relu', padding='same')(input_layer)
            lstm_out = Bidirectional(LSTM(units=lstm_units, return_sequences=False))(conv_layer)
            drop_layer = Dropout(dropout_rate)(lstm_out)
            output_layer = Dense(horizon)(drop_layer)
            model = Model(inputs=input_layer, outputs=output_layer)
            
            model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')
            early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=0)
            model.fit(X_tr, y_tr, batch_size=batch_size, epochs=5000,
                      validation_data=(X_val, y_val),
                      shuffle=False, verbose=0, callbacks=[early_stop])

            preds_scaled = model.predict(X_val, verbose=0)
            preds_inv = sc_y_fold.inverse_transform(preds_scaled)
            mape_cap = np.mean(np.abs(y_val_raw - preds_inv) / cap) * 100
            mape_scores.append(mape_cap)
            tf.keras.backend.clear_session()

        return np.mean(mape_scores)

    # --- Execução da Otimização ---
    print("\n--- INICIANDO BUSCA DE HIPERPARÂMETROS PARA CNN-BI-LSTM COM OPTUNA ---")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    print("\n" + "="*50)
    print("OTIMIZAÇÃO DO CNN-BI-LSTM CONCLUÍDA")
    print("="*50)
    print(f"Melhor resultado (MAPE médio na validação): {study.best_value:.4f}%")
    print("\nHiperparâmetros Ótimos Encontrados:")
    for key, value in study.best_params.items():
        print(f"  - {key}: {value}")
    print("="*50)

    # --- Treinamento Final com os Melhores Parâmetros ---
    print("\n--- TREINANDO MODELO CNN-BI-LSTM FINAL COM OS MELHORES PARÂMETROS ---")
    sc_X_final = StandardScaler().fit(trainX_full)
    sc_y_final = StandardScaler().fit(trainY_full)

    X_train_scaled = sc_X_final.transform(trainX_full).reshape(-1, look_back, 1)
    y_train_scaled = sc_y_final.transform(trainY_full)
    X_test_scaled = sc_X_final.transform(testX).reshape(-1, look_back, 1)

    input_layer = Input(shape=(look_back, 1))
    conv_layer = Conv1D(filters=best_params['cnn_filters'], kernel_size=best_params['kernel_size'], activation='relu', padding='same')(input_layer)
    lstm_out = Bidirectional(LSTM(units=best_params['lstm_units'], return_sequences=False))(conv_layer)
    drop_layer = Dropout(best_params['dropout_rate'])(lstm_out)
    output_layer = Dense(horizon)(drop_layer)
    final_model = Model(inputs=input_layer, outputs=output_layer)
    
    final_model.compile(optimizer=Adam(learning_rate=best_params['learning_rate']), loss='mean_squared_error')
    
    early_stop_final = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)
    final_model.fit(X_train_scaled, y_train_scaled,
                    batch_size=best_params['batch_size'],
                    epochs=5000,
                    validation_split=0.1,
                    shuffle=False, verbose=0, callbacks=[early_stop_final])
    print("Treinamento final concluído.")

    # --- Avaliação Final no Conjunto de Teste ---
    y_pred_test = sc_y_final.inverse_transform(final_model.predict(X_test_scaled, verbose=0))
    y_test_inv = testY

    metrics_data = []
    print("\n===== MÉTRICAS FINAIS NO CONJUNTO DE TESTE =====")
    for h in range(horizon):
        mape = np.mean(np.abs((y_test_inv[:, h] - y_pred_test[:, h]) / cap)) * 100
        rmse = sqrt(mean_squared_error(y_test_inv[:, h], y_pred_test[:, h]))
        mae = mean_absolute_error(y_test_inv[:, h], y_pred_test[:, h])
        r2 = r2_score(y_test_inv[:, h], y_pred_test[:, h])
        metrics_data.append({'Horizonte': f't+{(h+1)*10} min', 'MAPE (%)': mape, 'RMSE (MW)': rmse, 'MAE (MW)': mae, 'R²': r2})
        print(f"Horizonte t+{(h+1)*10}: MAPE={mape:.4f}%, RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")

    # --- Plotagem dos Resultados Finais ---
    print("\n--- GERANDO GRÁFICOS DE RESULTADOS ---")
    for h in range(horizon):
        plt.figure(figsize=(15, 6))
        plt.plot(y_test_inv[:, h], label=f"Real t+{(h+1)*10}", color="blue", alpha=0.8)
        plt.plot(y_pred_test[:, h], label=f"Previsto (Otimizado) t+{(h+1)*10}", color="red", linestyle="--")
        plt.title(f"Previsão CNN-BiLSTM Otimizado vs. Real - Horizonte t+{(h+1)*10} min")
        plt.xlabel("Amostras de Teste")
        plt.ylabel("LV ActivePower (kW)")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

    return final_model, best_params, pd.DataFrame(metrics_data)




#-------#

# Em myfunctions.py

def cnn_bilstm_att_model(new_data, months, look_back, data_partition, cap, horizon=3, strategy_horizon=20, mode='train'):
    """
    Modelo CNN-BiLSTM com Attention adaptado para previsão quantílica, seguindo o padrão do dashboard.
    - Utiliza a camada de atenção oficial do Keras (tf.keras.layers.Attention).
    - Opera em modos 'train' e 'inference'.
    - Utiliza funções de perda quantílica para treinar modelos para os quantis 0.1, 0.5 e 0.9.
    - Aplica correção de cruzamento de quantis.
    - Salva/carrega modelos e scalers para integração com o dashboard.
    - Gera métricas e gráficos de ORI no modo 'train'.
    """
    # --- Imports ---
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from math import sqrt
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Conv1D, Bidirectional, LSTM, Dense, Dropout, Attention
    from tensorflow.keras.callbacks import EarlyStopping
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    import random
    import os
    import joblib

    # --- Reprodutibilidade e Configurações ---
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)
    MODEL_DIR = "saved_models/cnn_bilstm_att_model" # Diretório específico

    # --- Funções de Perda (essenciais para carregar o modelo) ---
    # Reutilizando as funções de perda nomeadas já definidas no arquivo

    # --- Preparação dos Dados ---
    df = new_data.loc[new_data['Month'].isin(months)].reset_index(drop=True).dropna()
    values = df['LV ActivePower (kW)'].values.reshape(-1, 1)
    train_size = int(len(values) * data_partition)
    train, test = values[:train_size], values[train_size:]

    def create_dataset_multi(dataset, look_back, horizon):
        X, Y = [], []
        for j in range(len(dataset) - look_back - horizon + 1):
            X.append(dataset[j:j + look_back, 0])
            Y.append(dataset[j + look_back:j + look_back + horizon, 0])
        return np.array(X), np.array(Y)

    trainX, trainY = create_dataset_multi(train, look_back, horizon)
    testX, testY = create_dataset_multi(test, look_back, horizon)
    y_test_inv = testY

    all_predictions = {}
    quantiles = [0.1, 0.5, 0.9]

    if mode == 'train':
        print("--- EXECUTANDO CNN-BI-LSTM+ATTENTION EM MODO DE TREINAMENTO ---")
        sc_X = StandardScaler().fit(trainX)
        sc_y = StandardScaler().fit(trainY)
        X_train_scaled, y_train_scaled = sc_X.transform(trainX), sc_y.transform(trainY)
        X_test_scaled = sc_X.transform(testX)

        X_train = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
        X_test = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))

        for q in quantiles:
            print(f"Treinando modelo CNN-Bi-LSTM+Attention para o quantil: {q}")
            
            input_layer = Input(shape=(look_back, 1))
            conv_layer = Conv1D(filters=128, kernel_size=5, activation='relu', padding='same')(input_layer)
            # A BiLSTM precisa retornar sequências para a camada de atenção
            lstm_out = Bidirectional(LSTM(units=64, return_sequences=True))(conv_layer)
            # A camada de atenção pondera a saída da BiLSTM
            attention_out = Attention()([lstm_out, lstm_out])
            # Uma segunda LSTM agrega a sequência ponderada pela atenção
            lstm_agg = LSTM(units=32, return_sequences=False)(attention_out)
            drop_layer = Dropout(0.2329937583942957)(lstm_agg)
            output_layer = Dense(horizon)(drop_layer)
            
            model = Model(inputs=input_layer, outputs=output_layer)
            model.compile(optimizer=Adam(learning_rate=0.00010174940696950422), loss=loss_functions[q])
            
            early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
            model.fit(X_train, y_train_scaled, epochs=5000, batch_size=128, validation_split=0.1, shuffle=False, verbose=0, callbacks=[early_stop])
            
            y_pred_scaled = model.predict(X_test)
            y_pred_inv = sc_y.inverse_transform(y_pred_scaled)
            all_predictions[q] = np.clip(y_pred_inv, 0, cap)

            os.makedirs(MODEL_DIR, exist_ok=True)
            model_path = os.path.join(MODEL_DIR, f"model_q{str(q).replace('.', '')}.keras")
            model.save(model_path)
            print(f"Modelo CNN-Bi-LSTM+Attention salvo em: {model_path}")

        joblib.dump(sc_X, os.path.join(MODEL_DIR, "scaler_X.gz"))
        joblib.dump(sc_y, os.path.join(MODEL_DIR, "scaler_y.gz"))
        print("Scalers para CNN-Bi-LSTM+Attention salvos com sucesso.")

    elif mode == 'inference':
        print("--- EXECUTANDO CNN-BI-LSTM+ATTENTION EM MODO DE INFERÊNCIA ---")
        if not os.path.exists(MODEL_DIR):
            raise FileNotFoundError(f"Diretório de modelos CNN-Bi-LSTM+Attention '{MODEL_DIR}' não encontrado.")

        sc_X = joblib.load(os.path.join(MODEL_DIR, "scaler_X.gz"))
        sc_y = joblib.load(os.path.join(MODEL_DIR, "scaler_y.gz"))
        X_test_scaled = sc_X.transform(testX)
        X_test = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))
        
        custom_objects = {fn.__name__: fn for fn in loss_functions.values()}

        for q in quantiles:
            model_path = os.path.join(MODEL_DIR, f"model_q{str(q).replace('.', '')}.keras")
            model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
            
            y_pred_scaled = model.predict(X_test)
            y_pred_inv = sc_y.inverse_transform(y_pred_scaled)
            all_predictions[q] = np.clip(y_pred_inv, 0, cap)

    else:
        raise ValueError("O parâmetro 'mode' deve ser 'train' ou 'inference'.")

    # Correção de cruzamento de quantis
    print("Aplicando correção de cruzamento de quantis...")
    all_predictions[0.5] = np.maximum(all_predictions[0.1], all_predictions[0.5])
    all_predictions[0.9] = np.maximum(all_predictions[0.5], all_predictions[0.9])

    if mode == 'train':
        y_pred_median = all_predictions[0.5]
        print("\n===== Métricas Completas CNN-Bi-LSTM+Attention (baseadas na mediana q=0.5) =====")
        for h in range(horizon):
            y_true_h, y_pred_h = y_test_inv[:, h], y_pred_median[:, h]
            mape = np.mean(np.abs((y_true_h - y_pred_h) / cap)) * 100 if cap > 0 else 0
            rmse = sqrt(mean_squared_error(y_true_h, y_pred_h))
            mae = mean_absolute_error(y_true_h, y_pred_h)
            r2 = r2_score(y_true_h, y_pred_h)
            print(f"Horizonte t+{(h+1)*10}: MAPE={mape:.4f}%, RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")

        # Lógica de Risco ORI
        operational_threshold = 0.3 * cap 
        p_low_t20 = all_predictions[0.1][:, 1]
        p_low_t30 = all_predictions[0.1][:, 2]
        ori_levels = []
        if strategy_horizon == 20:
            for i in range(len(p_low_t20)):
                if p_low_t20[i] < operational_threshold: ori_levels.append('Alto')
                elif p_low_t30[i] < operational_threshold: ori_levels.append('Atenção')
                else: ori_levels.append('Baixo')
        else:
            for p_low in p_low_t30:
                ori_levels.append('Alto' if p_low < operational_threshold else 'Baixo')

        color_map = {'Baixo': 'lightgreen', 'Atenção': 'gold', 'Alto': 'salmon'}
        num_samples_to_plot = 300
        time_axis = np.arange(min(num_samples_to_plot, len(y_test_inv)))

        for h in range(horizon):
            fig, ax = plt.subplots(figsize=(18, 8))
            for i in range(len(time_axis)):
                ax.axvspan(i, i + 1, facecolor=color_map.get(ori_levels[i], 'white'), alpha=0.5, zorder=0)
            ax.axhline(y=operational_threshold, color='red', linestyle=':', linewidth=2.5, label=f'Limiar Operacional ({operational_threshold:.2f} kW)', zorder=4)
            ax.fill_between(time_axis, all_predictions[0.1][:len(time_axis), h], all_predictions[0.9][:len(time_axis), h], color='cornflowerblue', alpha=0.6, label='Intervalo de Confiança (80%)', zorder=1)
            ax.plot(time_axis, y_test_inv[:len(time_axis), h], label="Real", color='black', linewidth=2, zorder=3)
            ax.plot(time_axis, all_predictions[0.5][:len(time_axis), h], label="Previsão Mediana", color='firebrick', linestyle='--', linewidth=2.5, zorder=2)
            ax.set_title(f'Previsão Quantílica CNN-Bi-LSTM+Attention com ORI - Horizonte t+{(h+1)*10} min', fontsize=16)
            ax.set_ylabel('LV ActivePower (kW)', fontsize=12)
            ax.set_xlabel('Amostras de Teste', fontsize=12)
            risk_patches = [Patch(facecolor=color, alpha=0.5, label=f'Risco {level}') for level, color in color_map.items() if level in set(ori_levels)]
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles=handles + risk_patches, loc='best', fontsize=12)
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
            ax.set_xlim(0, len(time_axis))
            ax.set_ylim(bottom=max(0, y_test_inv.min() - (0.05 * cap)))
            plt.tight_layout()
            plt.show()
            plt.close(fig)

    return all_predictions, y_test_inv


    
#------#

# Em myfunctions.py

# A função principal `cnn_bilstm_att_model` permanece inalterada.
# A função `cnn_bilstm_att_model_with_optuna` é ajustada para otimizar, treinar e avaliar, mas sem salvar.

def cnn_bilstm_att_model_with_optuna(new_data, months, look_back, data_partition, cap, horizon=3, n_trials=30):
    """
    Otimiza, treina e avalia um modelo CNN-BiLSTM-Attention determinístico.
    - DESCobre os hiperparâmetros ótimos com Optuna e TimeSeriesSplit.
    - TREINA um modelo final com os melhores parâmetros em todo o conjunto de treino.
    - AVALIA e plota os resultados no conjunto de teste.
    - NÃO salva o modelo ou os scalers no disco.
    """
    # --- Imports ---
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Conv1D, Bidirectional, LSTM, Dense, Dropout, Attention
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.optimizers import Adam
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from math import sqrt
    import optuna
    import random

    # --- Reprodutibilidade ---
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)

    # --- Preparação dos Dados ---
    print("--- PREPARANDO DADOS PARA OTIMIZAÇÃO E TREINO DO CNN-BI-LSTM-ATTENTION ---")
    df = new_data.loc[new_data['Month'].isin(months)].reset_index(drop=True).dropna()
    values = df['LV ActivePower (kW)'].values.reshape(-1, 1)

    train_size = int(len(values) * data_partition)
    train_values, test_values = values[:train_size], values[train_size:]

    def create_dataset_multi(dataset, look_back, horizon):
        X, Y = [], []
        for j in range(len(dataset) - look_back - horizon + 1):
            X.append(dataset[j:j + look_back, 0])
            Y.append(dataset[j + look_back:j + look_back + horizon, 0])
        return np.array(X), np.array(Y)

    trainX_full, trainY_full = create_dataset_multi(train_values, look_back, horizon)
    testX, testY = create_dataset_multi(test_values, look_back, horizon)
    print(f"Dados preparados: {trainX_full.shape[0]} amostras de treino, {testX.shape[0]} amostras de teste.")

    # --- Função Objetivo do Optuna ---
    def objective(trial):
        # Espaço de busca para a arquitetura completa
        cnn_filters = trial.suggest_categorical('cnn_filters', [32, 64, 128])
        kernel_size = trial.suggest_categorical('kernel_size', [2, 3, 5])
        bilstm_units = trial.suggest_categorical('bilstm_units', [32, 64, 128])
        agg_lstm_units = trial.suggest_categorical('agg_lstm_units', [16, 32, 64])
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])

        tscv = TimeSeriesSplit(n_splits=3)
        mape_scores = []

        for train_idx, val_idx in tscv.split(trainX_full):
            X_tr_raw, X_val_raw = trainX_full[train_idx], trainX_full[val_idx]
            y_tr_raw, y_val_raw = trainY_full[train_idx], trainY_full[val_idx]

            sc_X_fold = StandardScaler().fit(X_tr_raw)
            sc_y_fold = StandardScaler().fit(y_tr_raw)

            X_tr = sc_X_fold.transform(X_tr_raw).reshape(-1, look_back, 1)
            y_tr = sc_y_fold.transform(y_tr_raw)
            X_val = sc_X_fold.transform(X_val_raw).reshape(-1, look_back, 1)
            y_val = sc_y_fold.transform(y_val_raw)

            # Arquitetura completa usando a API Funcional
            input_layer = Input(shape=(look_back, 1))
            conv_layer = Conv1D(filters=cnn_filters, kernel_size=kernel_size, activation='relu', padding='same')(input_layer)
            lstm_out = Bidirectional(LSTM(units=bilstm_units, return_sequences=True))(conv_layer)
            attention_out = Attention()([lstm_out, lstm_out])
            lstm_agg = LSTM(units=agg_lstm_units, return_sequences=False)(attention_out)
            drop_layer = Dropout(dropout_rate)(lstm_agg)
            output_layer = Dense(horizon)(drop_layer)
            model = Model(inputs=input_layer, outputs=output_layer)
            
            model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')
            early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=0)
            model.fit(X_tr, y_tr, batch_size=batch_size, epochs=5000,
                      validation_data=(X_val, y_val),
                      shuffle=False, verbose=0, callbacks=[early_stop])

            preds_scaled = model.predict(X_val, verbose=0)
            preds_inv = sc_y_fold.inverse_transform(preds_scaled)
            mape_cap = np.mean(np.abs(y_val_raw - preds_inv) / cap) * 100
            mape_scores.append(mape_cap)
            tf.keras.backend.clear_session()

        return np.mean(mape_scores)

    # --- Execução da Otimização ---
    print("\n--- INICIANDO BUSCA DE HIPERPARÂMETROS PARA CNN-BI-LSTM-ATTENTION COM OPTUNA ---")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    print("\n" + "="*50)
    print("OTIMIZAÇÃO DO CNN-BI-LSTM-ATTENTION CONCLUÍDA")
    print("="*50)
    print(f"Melhor resultado (MAPE médio na validação): {study.best_value:.4f}%")
    print("\nHiperparâmetros Ótimos Encontrados:")
    for key, value in study.best_params.items():
        print(f"  - {key}: {value}")
    print("="*50)

    # --- Treinamento Final com os Melhores Parâmetros ---
    print("\n--- TREINANDO MODELO CNN-BI-LSTM-ATTENTION FINAL COM OS MELHORES PARÂMETROS ---")
    sc_X_final = StandardScaler().fit(trainX_full)
    sc_y_final = StandardScaler().fit(trainY_full)

    X_train_scaled = sc_X_final.transform(trainX_full).reshape(-1, look_back, 1)
    y_train_scaled = sc_y_final.transform(trainY_full)
    X_test_scaled = sc_X_final.transform(testX).reshape(-1, look_back, 1)

    # Constrói o modelo final com a arquitetura e os parâmetros ótimos
    input_layer = Input(shape=(look_back, 1))
    conv_layer = Conv1D(filters=best_params['cnn_filters'], kernel_size=best_params['kernel_size'], activation='relu', padding='same')(input_layer)
    lstm_out = Bidirectional(LSTM(units=best_params['bilstm_units'], return_sequences=True))(conv_layer)
    attention_out = Attention()([lstm_out, lstm_out])
    lstm_agg = LSTM(units=best_params['agg_lstm_units'], return_sequences=False)(attention_out)
    drop_layer = Dropout(best_params['dropout_rate'])(lstm_agg)
    output_layer = Dense(horizon)(drop_layer)
    final_model = Model(inputs=input_layer, outputs=output_layer)
    
    final_model.compile(optimizer=Adam(learning_rate=best_params['learning_rate']), loss='mean_squared_error')
    
    early_stop_final = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)
    final_model.fit(X_train_scaled, y_train_scaled,
                    batch_size=best_params['batch_size'],
                    epochs=5000,
                    validation_split=0.1,
                    shuffle=False, verbose=0, callbacks=[early_stop_final])
    print("Treinamento final concluído.")

    # --- Avaliação Final no Conjunto de Teste ---
    y_pred_test = sc_y_final.inverse_transform(final_model.predict(X_test_scaled, verbose=0))
    y_test_inv = testY

    metrics_data = []
    print("\n===== MÉTRICAS FINAIS NO CONJUNTO DE TESTE =====")
    for h in range(horizon):
        mape = np.mean(np.abs((y_test_inv[:, h] - y_pred_test[:, h]) / cap)) * 100
        rmse = sqrt(mean_squared_error(y_test_inv[:, h], y_pred_test[:, h]))
        mae = mean_absolute_error(y_test_inv[:, h], y_pred_test[:, h])
        r2 = r2_score(y_test_inv[:, h], y_pred_test[:, h])
        metrics_data.append({'Horizonte': f't+{(h+1)*10} min', 'MAPE (%)': mape, 'RMSE (MW)': rmse, 'MAE (MW)': mae, 'R²': r2})
        print(f"Horizonte t+{(h+1)*10}: MAPE={mape:.4f}%, RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")

    # --- Plotagem dos Resultados Finais ---
    print("\n--- GERANDO GRÁFICOS DE RESULTADOS ---")
    for h in range(horizon):
        plt.figure(figsize=(15, 6))
        plt.plot(y_test_inv[:, h], label=f"Real t+{(h+1)*10}", color="blue", alpha=0.8)
        plt.plot(y_pred_test[:, h], label=f"Previsto (Otimizado) t+{(h+1)*10}", color="red", linestyle="--")
        plt.title(f"Previsão CNN-BiLSTM-Attention Otimizado vs. Real - Horizonte t+{(h+1)*10} min")
        plt.xlabel("Amostras de Teste")
        plt.ylabel("LV ActivePower (kW)")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

    return final_model, best_params, pd.DataFrame(metrics_data)



#-----#
#Transformer

def transformer_model(new_data, months, look_back, data_partition, cap, horizon=3, strategy_horizon=20, mode='train'):
    """
    Modelo Transformer (v3) com correção de serialização para a camada PositionalEncoding.
    """
    # --- Imports (sem alterações) ---
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from math import sqrt
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
    from tensorflow.keras.callbacks import EarlyStopping
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    import random
    import os
    import joblib

    # --- Reprodutibilidade e Configurações (sem alterações) ---
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)
    MODEL_DIR = "saved_models/transformer_model"

    # --- Funções de Perda (sem alterações) ---
    # Reutilizando as funções de perda nomeadas já definidas no arquivo

    # --- Arquitetura do Transformer (COM CORREÇÃO) ---
    
    # 1. Camada de Codificação Posicional Senoidal (CORRIGIDA)
    class PositionalEncoding(tf.keras.layers.Layer):
        # CORREÇÃO 1: Adicionar **kwargs e passá-los para o super()
        def __init__(self, position, d_model, **kwargs):
            super(PositionalEncoding, self).__init__(**kwargs)
            self.position = position
            self.d_model = d_model
            self.pos_encoding = self.positional_encoding(position, d_model)

        def get_angles(self, position, i, d_model):
            angles = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
            return position * angles

        def positional_encoding(self, position, d_model):
            angle_rads = self.get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)
            angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
            angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
            pos_encoding = angle_rads[np.newaxis, ...]
            return tf.cast(pos_encoding, dtype=tf.float32)

        def call(self, inputs):
            return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

        # CORREÇÃO 2: Implementar get_config para salvar corretamente
        def get_config(self):
            config = super(PositionalEncoding, self).get_config()
            config.update({
                'position': self.position,
                'd_model': self.d_model,
            })
            return config

    # O restante da arquitetura e da função permanece o mesmo
    def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
        x = LayerNormalization(epsilon=1e-6)(inputs)
        x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
        x = Dropout(dropout)(x)
        res = x + inputs
        x = LayerNormalization(epsilon=1e-6)(res)
        x = Dense(ff_dim, activation="relu")(x)
        x = Dropout(dropout)(x)
        x = Dense(inputs.shape[-1])(x)
        return x + res

    def build_transformer_model(input_shape, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout=0, mlp_dropout=0):
        inputs = Input(shape=input_shape)
        d_model = head_size * num_heads
        x = Dense(d_model)(inputs)
        x = PositionalEncoding(input_shape[0], d_model)(x)
        for _ in range(num_transformer_blocks):
            x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)
        x = GlobalAveragePooling1D(data_format="channels_first")(x)
        for dim in mlp_units:
            x = Dense(dim, activation="relu")(x)
            x = Dropout(mlp_dropout)(x)
        outputs = Dense(horizon)(x)
        return Model(inputs, outputs)

    # --- Preparação dos Dados (sem alterações) ---
    df = new_data.loc[new_data['Month'].isin(months)].reset_index(drop=True).dropna()
    values = df['LV ActivePower (kW)'].values.reshape(-1, 1)
    train_size = int(len(values) * data_partition)
    train, test = values[:train_size], values[train_size:]

    def create_dataset_multi(dataset, look_back, horizon):
        X, Y = [], []
        for j in range(len(dataset) - look_back - horizon + 1):
            X.append(dataset[j:j + look_back, 0])
            Y.append(dataset[j + look_back:j + look_back + horizon, 0])
        return np.array(X), np.array(Y)

    trainX, trainY = create_dataset_multi(train, look_back, horizon)
    testX, testY = create_dataset_multi(test, look_back, horizon)
    y_test_inv = testY

    all_predictions = {}
    quantiles = [0.1, 0.5, 0.9]

    if mode == 'train':
        print("--- EXECUTANDO TRANSFORMER (v3) EM MODO DE TREINAMENTO ---")
        sc_X = StandardScaler().fit(trainX)
        sc_y = StandardScaler().fit(trainY)
        X_train_scaled, y_train_scaled = sc_X.transform(trainX), sc_y.transform(trainY)
        X_test_scaled = sc_X.transform(testX)
        X_train = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
        X_test = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))

        for q in quantiles:
            print(f"Treinando modelo Transformer (v3) para o quantil: {q}")
            model = build_transformer_model(
                input_shape=(look_back, 1), head_size=32, num_heads=4, ff_dim=16,
                num_transformer_blocks=1, mlp_units=[64], mlp_dropout=0.4392687065771505, dropout=0.3743992732647771,
            )
            model.compile(optimizer=Adam(learning_rate=0.000634654116306671), loss=loss_functions[q])
            early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
            model.fit(X_train, y_train_scaled, epochs=5000, batch_size=32, validation_split=0.2, shuffle=False, verbose=0, callbacks=[early_stop])
            y_pred_scaled = model.predict(X_test)
            y_pred_inv = sc_y.inverse_transform(y_pred_scaled)
            all_predictions[q] = np.clip(y_pred_inv, 0, cap)
            os.makedirs(MODEL_DIR, exist_ok=True)
            model_path = os.path.join(MODEL_DIR, f"model_q{str(q).replace('.', '')}.keras")
            model.save(model_path)
            print(f"Modelo Transformer (v3) salvo em: {model_path}")

        joblib.dump(sc_X, os.path.join(MODEL_DIR, "scaler_X.gz"))
        joblib.dump(sc_y, os.path.join(MODEL_DIR, "scaler_y.gz"))
        print("Scalers para Transformer (v3) salvos com sucesso.")

    elif mode == 'inference':
        print("--- EXECUTANDO TRANSFORMER (v3) EM MODO DE INFERÊNCIA ---")
        if not os.path.exists(MODEL_DIR):
            raise FileNotFoundError(f"Diretório de modelos Transformer '{MODEL_DIR}' não encontrado.")

        sc_X = joblib.load(os.path.join(MODEL_DIR, "scaler_X.gz"))
        sc_y = joblib.load(os.path.join(MODEL_DIR, "scaler_y.gz"))
        X_test_scaled = sc_X.transform(testX)
        X_test = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))
        
        custom_objects = {fn.__name__: fn for fn in loss_functions.values()}
        custom_objects['PositionalEncoding'] = PositionalEncoding

        for q in quantiles:
            model_path = os.path.join(MODEL_DIR, f"model_q{str(q).replace('.', '')}.keras")
            model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
            y_pred_scaled = model.predict(X_test)
            y_pred_inv = sc_y.inverse_transform(y_pred_scaled)
            all_predictions[q] = np.clip(y_pred_inv, 0, cap)

    else:
        raise ValueError("O parâmetro 'mode' deve ser 'train' ou 'inference'.")

    # Correção de cruzamento de quantis
    print("Aplicando correção de cruzamento de quantis...")
    all_predictions[0.5] = np.maximum(all_predictions[0.1], all_predictions[0.5])
    all_predictions[0.9] = np.maximum(all_predictions[0.5], all_predictions[0.9])

    if mode == 'train':
        # O restante do código para métricas e plotagem permanece o mesmo
        y_pred_median = all_predictions[0.5]
        print("\n===== Métricas Completas Transformer (v3) (baseadas na mediana q=0.5) =====")
        for h in range(horizon):
            y_true_h, y_pred_h = y_test_inv[:, h], y_pred_median[:, h]
            mape = np.mean(np.abs((y_true_h - y_pred_h) / cap)) * 100 if cap > 0 else 0
            rmse = sqrt(mean_squared_error(y_true_h, y_pred_h))
            mae = mean_absolute_error(y_true_h, y_pred_h)
            r2 = r2_score(y_true_h, y_pred_h)
            print(f"Horizonte t+{(h+1)*10}: MAPE={mape:.4f}%, RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")

        # Lógica de Risco ORI
        operational_threshold = 0.3 * cap 
        p_low_t20 = all_predictions[0.1][:, 1]
        p_low_t30 = all_predictions[0.1][:, 2]
        ori_levels = []
        if strategy_horizon == 20:
            for i in range(len(p_low_t20)):
                if p_low_t20[i] < operational_threshold: ori_levels.append('Alto')
                elif p_low_t30[i] < operational_threshold: ori_levels.append('Atenção')
                else: ori_levels.append('Baixo')
        else:
            for p_low in p_low_t30:
                ori_levels.append('Alto' if p_low < operational_threshold else 'Baixo')

        color_map = {'Baixo': 'lightgreen', 'Atenção': 'gold', 'Alto': 'salmon'}
        num_samples_to_plot = 300
        time_axis = np.arange(min(num_samples_to_plot, len(y_test_inv)))

        for h in range(horizon):
            fig, ax = plt.subplots(figsize=(18, 8))
            for i in range(len(time_axis)):
                ax.axvspan(i, i + 1, facecolor=color_map.get(ori_levels[i], 'white'), alpha=0.5, zorder=0)
            ax.axhline(y=operational_threshold, color='red', linestyle=':', linewidth=2.5, label=f'Limiar Operacional ({operational_threshold:.2f} kW)', zorder=4)
            ax.fill_between(time_axis, all_predictions[0.1][:len(time_axis), h], all_predictions[0.9][:len(time_axis), h], color='cornflowerblue', alpha=0.6, label='Intervalo de Confiança (80%)', zorder=1)
            ax.plot(time_axis, y_test_inv[:len(time_axis), h], label="Real", color='black', linewidth=2, zorder=3)
            ax.plot(time_axis, all_predictions[0.5][:len(time_axis), h], label="Previsão Mediana", color='firebrick', linestyle='--', linewidth=2.5, zorder=2)
            ax.set_title(f'Previsão Quantílica Transformer (v3) com ORI - Horizonte t+{(h+1)*10} min', fontsize=16)
            ax.set_ylabel('LV ActivePower (kW)', fontsize=12)
            ax.set_xlabel('Amostras de Teste', fontsize=12)
            risk_patches = [Patch(facecolor=color, alpha=0.5, label=f'Risco {level}') for level, color in color_map.items() if level in set(ori_levels)]
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles=handles + risk_patches, loc='best', fontsize=12)
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
            ax.set_xlim(0, len(time_axis))
            ax.set_ylim(bottom=max(0, y_test_inv.min() - (0.05 * cap)))
            plt.tight_layout()
            plt.show()
            plt.close(fig)

    return all_predictions, y_test_inv



# Em myfunctions.py

# A função principal `transformer_model` permanece inalterada.
# Esta é a nova função de otimização para o Transformer.

def transformer_model_with_optuna(new_data, months, look_back, data_partition, cap, horizon=3, n_trials=30):
    """
    Otimiza, treina e avalia um modelo Transformer determinístico.
    - DESCobre os hiperparâmetros ótimos com Optuna e TimeSeriesSplit.
    - TREINA um modelo final com os melhores parâmetros em todo o conjunto de treino.
    - AVALIA e plota os resultados no conjunto de teste.
    - NÃO salva o modelo ou os scalers no disco.
    """
    # --- Imports ---
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.optimizers import Adam
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from math import sqrt
    import optuna
    import random

    # --- Reprodutibilidade ---
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)

    # --- Arquitetura do Transformer (copiada da função principal) ---
    class PositionalEncoding(tf.keras.layers.Layer):
        def __init__(self, position, d_model, **kwargs):
            super(PositionalEncoding, self).__init__(**kwargs)
            self.position = position
            self.d_model = d_model
            self.pos_encoding = self.positional_encoding(position, d_model)
        def get_angles(self, position, i, d_model):
            angles = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
            return position * angles
        def positional_encoding(self, position, d_model):
            angle_rads = self.get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)
            angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
            angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
            pos_encoding = angle_rads[np.newaxis, ...]
            return tf.cast(pos_encoding, dtype=tf.float32)
        def call(self, inputs):
            return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]
        def get_config(self):
            config = super(PositionalEncoding, self).get_config()
            config.update({'position': self.position, 'd_model': self.d_model})
            return config

    def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
        x = LayerNormalization(epsilon=1e-6)(inputs)
        x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
        x = Dropout(dropout)(x)
        res = x + inputs
        x = LayerNormalization(epsilon=1e-6)(res)
        x = Dense(ff_dim, activation="relu")(x)
        x = Dropout(dropout)(x)
        x = Dense(inputs.shape[-1])(x)
        return x + res

    def build_transformer_model(input_shape, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout=0, mlp_dropout=0):
        inputs = Input(shape=input_shape)
        d_model = head_size * num_heads
        x = Dense(d_model)(inputs)
        x = PositionalEncoding(input_shape[0], d_model)(x)
        for _ in range(num_transformer_blocks):
            x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)
        x = GlobalAveragePooling1D(data_format="channels_first")(x)
        for dim in mlp_units:
            x = Dense(dim, activation="relu")(x)
            x = Dropout(mlp_dropout)(x)
        outputs = Dense(horizon)(x)
        return Model(inputs, outputs)

    # --- Preparação dos Dados ---
    print("--- PREPARANDO DADOS PARA OTIMIZAÇÃO E TREINO DO TRANSFORMER ---")
    df = new_data.loc[new_data['Month'].isin(months)].reset_index(drop=True).dropna()
    values = df['LV ActivePower (kW)'].values.reshape(-1, 1)
    train_size = int(len(values) * data_partition)
    train_values, test_values = values[:train_size], values[train_size:]

    def create_dataset_multi(dataset, look_back, horizon):
        X, Y = [], []
        for j in range(len(dataset) - look_back - horizon + 1):
            X.append(dataset[j:j + look_back, 0])
            Y.append(dataset[j + look_back:j + look_back + horizon, 0])
        return np.array(X), np.array(Y)

    trainX_full, trainY_full = create_dataset_multi(train_values, look_back, horizon)
    testX, testY = create_dataset_multi(test_values, look_back, horizon)
    print(f"Dados preparados: {trainX_full.shape[0]} amostras de treino, {testX.shape[0]} amostras de teste.")

    # --- Função Objetivo do Optuna ---
    def objective(trial):
        # Espaço de busca para os hiperparâmetros do Transformer
        head_size = trial.suggest_categorical('head_size', [32, 64, 128])
        num_heads = trial.suggest_categorical('num_heads', [2, 4, 8])
        ff_dim = trial.suggest_categorical('ff_dim', [4, 8, 16])
        num_transformer_blocks = trial.suggest_int('num_transformer_blocks', 1, 4)
        mlp_units = trial.suggest_categorical('mlp_units', [64, 128])
        dropout = trial.suggest_float('dropout', 0.1, 0.4)
        mlp_dropout = trial.suggest_float('mlp_dropout', 0.1, 0.5)
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [32, 64])

        tscv = TimeSeriesSplit(n_splits=3)
        mape_scores = []

        for train_idx, val_idx in tscv.split(trainX_full):
            X_tr_raw, X_val_raw = trainX_full[train_idx], trainX_full[val_idx]
            y_tr_raw, y_val_raw = trainY_full[train_idx], trainY_full[val_idx]

            sc_X_fold = StandardScaler().fit(X_tr_raw)
            sc_y_fold = StandardScaler().fit(y_tr_raw)
            X_tr = sc_X_fold.transform(X_tr_raw).reshape(-1, look_back, 1)
            y_tr = sc_y_fold.transform(y_tr_raw)
            X_val = sc_X_fold.transform(X_val_raw).reshape(-1, look_back, 1)
            y_val = sc_y_fold.transform(y_val_raw)

            model = build_transformer_model(
                input_shape=(look_back, 1), head_size=head_size, num_heads=num_heads, ff_dim=ff_dim,
                num_transformer_blocks=num_transformer_blocks, mlp_units=[mlp_units], 
                dropout=dropout, mlp_dropout=mlp_dropout
            )
            model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')
            early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=0)
            model.fit(X_tr, y_tr, batch_size=batch_size, epochs=5000,
                      validation_data=(X_val, y_val),
                      shuffle=False, verbose=0, callbacks=[early_stop])

            preds_scaled = model.predict(X_val, verbose=0)
            preds_inv = sc_y_fold.inverse_transform(preds_scaled)
            mape_cap = np.mean(np.abs(y_val_raw - preds_inv) / cap) * 100
            mape_scores.append(mape_cap)
            tf.keras.backend.clear_session()

        return np.mean(mape_scores)

    # --- Execução da Otimização ---
    print("\n--- INICIANDO BUSCA DE HIPERPARÂMETROS PARA TRANSFORMER COM OPTUNA ---")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    print("\n" + "="*50)
    print("OTIMIZAÇÃO DO TRANSFORMER CONCLUÍDA")
    print("="*50)
    print(f"Melhor resultado (MAPE médio na validação): {study.best_value:.4f}%")
    print("\nHiperparâmetros Ótimos Encontrados:")
    for key, value in study.best_params.items():
        print(f"  - {key}: {value}")
    print("="*50)

    # --- Treinamento Final com os Melhores Parâmetros ---
    print("\n--- TREINANDO MODELO TRANSFORMER FINAL COM OS MELHORES PARÂMETROS ---")
    sc_X_final = StandardScaler().fit(trainX_full)
    sc_y_final = StandardScaler().fit(trainY_full)
    X_train_scaled = sc_X_final.transform(trainX_full).reshape(-1, look_back, 1)
    y_train_scaled = sc_y_final.transform(trainY_full)
    X_test_scaled = sc_X_final.transform(testX).reshape(-1, look_back, 1)

    final_model = build_transformer_model(
        input_shape=(look_back, 1), head_size=best_params['head_size'], num_heads=best_params['num_heads'], 
        ff_dim=best_params['ff_dim'], num_transformer_blocks=best_params['num_transformer_blocks'],
        mlp_units=[best_params['mlp_units']], mlp_dropout=best_params['mlp_dropout'], dropout=best_params['dropout']
    )
    final_model.compile(optimizer=Adam(learning_rate=best_params['learning_rate']), loss='mean_squared_error')
    early_stop_final = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)
    final_model.fit(X_train_scaled, y_train_scaled,
                    batch_size=best_params['batch_size'],
                    epochs=5000,
                    validation_split=0.2,
                    shuffle=False, verbose=0, callbacks=[early_stop_final])
    print("Treinamento final concluído.")

    # --- Avaliação Final no Conjunto de Teste ---
    y_pred_test = sc_y_final.inverse_transform(final_model.predict(X_test_scaled, verbose=0))
    y_test_inv = testY

    metrics_data = []
    print("\n===== MÉTRICAS FINAIS NO CONJUNTO DE TESTE =====")
    for h in range(horizon):
        mape = np.mean(np.abs((y_test_inv[:, h] - y_pred_test[:, h]) / cap)) * 100
        rmse = sqrt(mean_squared_error(y_test_inv[:, h], y_pred_test[:, h]))
        mae = mean_absolute_error(y_test_inv[:, h], y_pred_test[:, h])
        r2 = r2_score(y_test_inv[:, h], y_pred_test[:, h])
        metrics_data.append({'Horizonte': f't+{(h+1)*10} min', 'MAPE (%)': mape, 'RMSE (MW)': rmse, 'MAE (MW)': mae, 'R²': r2})
        print(f"Horizonte t+{(h+1)*10}: MAPE={mape:.4f}%, RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")

    # --- Plotagem dos Resultados Finais ---
    print("\n--- GERANDO GRÁFICOS DE RESULTADOS ---")
    for h in range(horizon):
        plt.figure(figsize=(15, 6))
        plt.plot(y_test_inv[:, h], label=f"Real t+{(h+1)*10}", color="blue", alpha=0.8)
        plt.plot(y_pred_test[:, h], label=f"Previsto (Otimizado) t+{(h+1)*10}", color="red", linestyle="--")
        plt.title(f"Previsão Transformer Otimizado vs. Real - Horizonte t+{(h+1)*10} min")
        plt.xlabel("Amostras de Teste")
        plt.ylabel("LV ActivePower (kW)")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

    return final_model, best_params, pd.DataFrame(metrics_data)


# Em myfunctions.py
# Em myfunctions.py

def tft_model(new_data, months, look_back, data_partition, cap, horizon=3, strategy_horizon=20, mode='train'):
    """
    Modelo Temporal Fusion Transformer (TFT) Multivariado (Versão Final Estável).
    Focada na performance e integração com o dashboard, removendo a complexidade da interpretabilidade.
    """
    # --- Imports ---
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from math import sqrt
    from tensorflow.keras.optimizers import Adam
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    import random
    import os
    import joblib

    # --- Reprodutibilidade e Configurações ---
    tf.random.set_seed(42)
    np.random.seed(42)
    random.seed(42)
    MODEL_DIR = "saved_models/tft_model"
    
    # --- Blocos de Construção do TFT ---
    quantiles = [0.1, 0.5, 0.9]
    
    def quantile_loss(y_true, y_pred):
        y_true_expanded = tf.expand_dims(y_true, axis=-1)
        error = y_true_expanded - y_pred
        q_tensor = tf.constant(quantiles, dtype=tf.float32)
        loss = tf.maximum(q_tensor * error, (q_tensor - 1) * error)
        return tf.reduce_mean(loss)

    class GatedLinearUnit(layers.Layer):
        def __init__(self, units, **kwargs): super().__init__(**kwargs); self.linear = layers.Dense(units); self.sigmoid = layers.Dense(units, activation="sigmoid")
        def call(self, inputs): return self.linear(inputs) * self.sigmoid(inputs)

    class GatedResidualNetwork(layers.Layer):
        def __init__(self, units, dropout_rate, **kwargs):
            super().__init__(**kwargs); self.units = units; self.elu_dense = layers.Dense(units, activation="elu"); self.linear_dense = layers.Dense(units); self.dropout = layers.Dropout(dropout_rate); self.gated_linear_unit = GatedLinearUnit(units); self.layer_norm = layers.LayerNormalization(); self.project = layers.Dense(units)
        def call(self, inputs):
            x = self.elu_dense(inputs); x = self.linear_dense(x); x = self.dropout(x)
            if inputs.shape[-1] != self.units: inputs = self.project(inputs)
            x = inputs + self.gated_linear_unit(x); x = self.layer_norm(x); return x

    # --- Preparação dos Dados ---
    df = new_data.loc[new_data['Month'].isin(months)].reset_index(drop=True).dropna()
    features_df = df[['LV ActivePower (kW)', 'wind_speed_235_avg']]
    target_df = df[['LV ActivePower (kW)']]

    train_size = int(len(features_df) * data_partition)
    train_features_df, test_features_df = features_df.iloc[:train_size], features_df.iloc[train_size:]
    train_target_df, test_target_df = target_df.iloc[:train_size], target_df.iloc[train_size:]

    scaler_X = StandardScaler().fit(train_features_df)
    scaler_Y = StandardScaler().fit(train_target_df)

    train_features_scaled = scaler_X.transform(train_features_df)
    test_features_scaled = scaler_X.transform(test_features_df)
    train_target_scaled = scaler_Y.transform(train_target_df)
    test_target_scaled = scaler_Y.transform(test_target_df)

    def create_dataset_multivariate(features, target, look_back, horizon):
        X, Y = [], []
        for j in range(len(features) - look_back - horizon + 1):
            X.append(features[j:j + look_back, :])
            Y.append(target[j + look_back:j + look_back + horizon, 0])
        return np.array(X), np.array(Y)

    trainX, trainY = create_dataset_multivariate(train_features_scaled, train_target_scaled, look_back, horizon)
    testX, testY = create_dataset_multivariate(test_features_scaled, test_target_scaled, look_back, horizon)
    y_test_inv = scaler_Y.inverse_transform(testY)

    all_predictions = {}

    if mode == 'train':
        print("--- EXECUTANDO TFT EM MODO DE TREINAMENTO ---")
        hidden_units, dropout_rate, num_heads, num_features = 128, 0.28961796649204674, 4, trainX.shape[2]
        input_shape = (look_back, num_features)
        
        inputs = layers.Input(shape=input_shape)
        x = GatedResidualNetwork(hidden_units, dropout_rate)(inputs)
        x = GatedResidualNetwork(hidden_units, dropout_rate)(x)
        attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=hidden_units, dropout=dropout_rate)(x, x)
        x = layers.LayerNormalization()(x + attention_output)
        x = GatedResidualNetwork(hidden_units, dropout_rate)(x)
        x = layers.Flatten()(x)
        outputs = layers.Dense(horizon * len(quantiles))(x)
        outputs = layers.Reshape((horizon, len(quantiles)))(outputs)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(loss=quantile_loss, optimizer=Adam(learning_rate=0.0007802553471258404))
        
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
        model.fit(trainX, trainY, epochs=5000, batch_size=32, validation_split=0.2, callbacks=[early_stop], verbose=0)

        os.makedirs(MODEL_DIR, exist_ok=True)
        model.save(os.path.join(MODEL_DIR, "tft_model.keras"))
        joblib.dump(scaler_X, os.path.join(MODEL_DIR, "scaler_X.gz"))
        joblib.dump(scaler_Y, os.path.join(MODEL_DIR, "scaler_Y.gz"))
        print("Modelo TFT e scalers salvos com sucesso.")

    elif mode == 'inference':
        print("--- EXECUTANDO TFT EM MODO DE INFERÊNCIA ---")
        if not os.path.exists(MODEL_DIR):
            raise FileNotFoundError(f"Diretório de modelos TFT '{MODEL_DIR}' não encontrado.")
        
        custom_objects = {"GatedLinearUnit": GatedLinearUnit, "GatedResidualNetwork": GatedResidualNetwork, "quantile_loss": quantile_loss}
        model = keras.models.load_model(os.path.join(MODEL_DIR, "tft_model.keras"), custom_objects=custom_objects)
        scaler_X = joblib.load(os.path.join(MODEL_DIR, "scaler_X.gz"))
        scaler_Y = joblib.load(os.path.join(MODEL_DIR, "scaler_Y.gz"))

    else:
        raise ValueError("O parâmetro 'mode' deve ser 'train' ou 'inference'.")

    raw_predictions_scaled = model.predict(testX)
    for i, q in enumerate(quantiles):
        preds_q_scaled = raw_predictions_scaled[:, :, i]
        preds_q_unscaled = scaler_Y.inverse_transform(preds_q_scaled)
        all_predictions[q] = np.clip(preds_q_unscaled, 0, cap)

    print("Aplicando correção de cruzamento de quantis...")
    all_predictions[0.5] = np.maximum(all_predictions[0.1], all_predictions[0.5])
    all_predictions[0.9] = np.maximum(all_predictions[0.5], all_predictions[0.9])

    if mode == 'train':
        y_pred_median = all_predictions[0.5]
        print("\n===== Métricas Completas TFT (baseadas na mediana q=0.5) =====")
        for h in range(horizon):
            y_true_h, y_pred_h = y_test_inv[:, h], y_pred_median[:, h]
            mape = np.mean(np.abs((y_true_h - y_pred_h) / cap)) * 100 if cap > 0 else 0
            rmse = sqrt(mean_squared_error(y_true_h, y_pred_h))
            mae = mean_absolute_error(y_true_h, y_pred_h)
            r2 = r2_score(y_true_h, y_pred_h)
            print(f"Horizonte t+{(h+1)*10}: MAPE={mape:.4f}%, RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")

        # Lógica de Risco ORI
        operational_threshold = 0.3 * cap 
        p_low_t20 = all_predictions[0.1][:, 1]
        p_low_t30 = all_predictions[0.1][:, 2]
        ori_levels = []
        if strategy_horizon == 20:
            for i in range(len(p_low_t20)):
                if p_low_t20[i] < operational_threshold: ori_levels.append('Alto')
                elif p_low_t30[i] < operational_threshold: ori_levels.append('Atenção')
                else: ori_levels.append('Baixo')
        else:
            for p_low in p_low_t30:
                ori_levels.append('Alto' if p_low < operational_threshold else 'Baixo')

        color_map = {'Baixo': 'lightgreen', 'Atenção': 'gold', 'Alto': 'salmon'}
        num_samples_to_plot = 300
        time_axis = np.arange(min(num_samples_to_plot, len(y_test_inv)))

        for h in range(horizon):
            fig, ax = plt.subplots(figsize=(18, 8))
            for i in range(len(time_axis)):
                ax.axvspan(i, i + 1, facecolor=color_map.get(ori_levels[i], 'white'), alpha=0.5, zorder=0)
            ax.axhline(y=operational_threshold, color='red', linestyle=':', linewidth=2.5, label=f'Limiar Operacional ({operational_threshold:.2f} kW)', zorder=4)
            ax.fill_between(time_axis, all_predictions[0.1][:len(time_axis), h], all_predictions[0.9][:len(time_axis), h], color='cornflowerblue', alpha=0.6, label='Intervalo de Confiança (80%)', zorder=1)
            ax.plot(time_axis, y_test_inv[:len(time_axis), h], label="Real", color='black', linewidth=2, zorder=3)
            ax.plot(time_axis, all_predictions[0.5][:len(time_axis), h], label="Previsão Mediana", color='firebrick', linestyle='--', linewidth=2.5, zorder=2)
            ax.set_title(f'Previsão Quantílica TFT com ORI - Horizonte t+{(h+1)*10} min', fontsize=16)
            ax.set_ylabel('LV ActivePower (kW)', fontsize=12)
            ax.set_xlabel('Amostras de Teste', fontsize=12)
            risk_patches = [Patch(facecolor=color, alpha=0.5, label=f'Risco {level}') for level, color in color_map.items() if level in set(ori_levels)]
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles=handles + risk_patches, loc='best', fontsize=12)
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
            ax.set_xlim(0, len(time_axis))
            ax.set_ylim(bottom=max(0, y_test_inv.min() - (0.05 * cap)))
            plt.tight_layout()
            plt.show()
            plt.close(fig)

    return all_predictions, y_test_inv



# Em myfunctions.py

# A função principal `tft_model` permanece inalterada.
# Esta é a nova função de otimização para o TFT.

def tft_model_with_optuna(new_data, months, look_back, data_partition, cap, horizon=3, n_trials=30):
    """
    Otimiza, treina e avalia um modelo TFT determinístico (previsão de ponto).
    - DESCobre os hiperparâmetros ótimos com Optuna e TimeSeriesSplit.
    - TREINA um modelo final com os melhores parâmetros em todo o conjunto de treino.
    - AVALIA e plota os resultados no conjunto de teste.
    - NÃO salva o modelo ou os scalers no disco.
    """
    # --- Imports ---
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.optimizers import Adam
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from math import sqrt
    import optuna
    import random

    # --- Reprodutibilidade ---
    tf.random.set_seed(42)
    np.random.seed(42)
    random.seed(42)

    # --- Blocos de Construção do TFT (copiados da função principal) ---
    class GatedLinearUnit(layers.Layer):
        def __init__(self, units, **kwargs): super().__init__(**kwargs); self.linear = layers.Dense(units); self.sigmoid = layers.Dense(units, activation="sigmoid")
        def call(self, inputs): return self.linear(inputs) * self.sigmoid(inputs)
        def get_config(self):
            config = super().get_config()
            config.update({'units': self.linear.units})
            return config

    class GatedResidualNetwork(layers.Layer):
        def __init__(self, units, dropout_rate, **kwargs):
            super().__init__(**kwargs); self.units = units; self.dropout_rate = dropout_rate; self.elu_dense = layers.Dense(units, activation="elu"); self.linear_dense = layers.Dense(units); self.dropout = layers.Dropout(dropout_rate); self.gated_linear_unit = GatedLinearUnit(units); self.layer_norm = layers.LayerNormalization(); self.project = layers.Dense(units)
        def call(self, inputs):
            x = self.elu_dense(inputs); x = self.linear_dense(x); x = self.dropout(x)
            if inputs.shape[-1] != self.units: inputs = self.project(inputs)
            x = inputs + self.gated_linear_unit(x); x = self.layer_norm(x); return x
        def get_config(self):
            config = super().get_config()
            config.update({'units': self.units, 'dropout_rate': self.dropout_rate})
            return config

    def build_tft_model(input_shape, hidden_units, dropout_rate, num_heads, horizon):
        inputs = layers.Input(shape=input_shape)
        x = GatedResidualNetwork(hidden_units, dropout_rate)(inputs)
        x = GatedResidualNetwork(hidden_units, dropout_rate)(x)
        attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=hidden_units, dropout=dropout_rate)(x, x)
        x = layers.LayerNormalization()(x + attention_output)
        x = GatedResidualNetwork(hidden_units, dropout_rate)(x)
        x = layers.Flatten()(x)
        # Saída determinística (previsão de ponto) para o horizonte
        outputs = layers.Dense(horizon)(x)
        return keras.Model(inputs=inputs, outputs=outputs)

    # --- Preparação dos Dados Multivariados ---
    print("--- PREPARANDO DADOS PARA OTIMIZAÇÃO E TREINO DO TFT ---")
    df = new_data.loc[new_data['Month'].isin(months)].reset_index(drop=True).dropna()
    features_df = df[['LV ActivePower (kW)', 'wind_speed_235_avg']]
    target_df = df[['LV ActivePower (kW)']]

    train_size = int(len(features_df) * data_partition)
    train_features_df, test_features_df = features_df.iloc[:train_size], features_df.iloc[train_size:]
    train_target_df, test_target_df = target_df.iloc[:train_size], target_df.iloc[train_size:]

    def create_dataset_multivariate(features, target, look_back, horizon):
        X, Y = [], []
        for j in range(len(features) - look_back - horizon + 1):
            X.append(features[j:j + look_back, :])
            Y.append(target[j + look_back:j + look_back + horizon, 0])
        return np.array(X), np.array(Y)

    # Scalers ajustados apenas no conjunto de treino completo
    scaler_X_full = StandardScaler().fit(train_features_df)
    scaler_Y_full = StandardScaler().fit(train_target_df)

    train_features_scaled = scaler_X_full.transform(train_features_df)
    train_target_scaled = scaler_Y_full.transform(train_target_df)
    
    trainX_full, trainY_full = create_dataset_multivariate(train_features_scaled, train_target_scaled, look_back, horizon)
    
    test_features_scaled = scaler_X_full.transform(test_features_df)
    test_target_scaled = scaler_Y_full.transform(test_target_df)
    testX, testY = create_dataset_multivariate(test_features_scaled, test_target_scaled, look_back, horizon)
    
    print(f"Dados preparados: {trainX_full.shape[0]} amostras de treino, {testX.shape[0]} amostras de teste.")

    # --- Função Objetivo do Optuna ---
    def objective(trial):
        hidden_units = trial.suggest_categorical('hidden_units', [32, 64, 128])
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.4)
        num_heads = trial.suggest_categorical('num_heads', [2, 4, 8])
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [32, 64])

        tscv = TimeSeriesSplit(n_splits=3)
        mape_scores = []

        for train_idx, val_idx in tscv.split(trainX_full):
            X_tr, X_val = trainX_full[train_idx], trainX_full[val_idx]
            y_tr, y_val = trainY_full[train_idx], trainY_full[val_idx]

            model = build_tft_model(
                input_shape=(look_back, X_tr.shape[2]),
                hidden_units=hidden_units,
                dropout_rate=dropout_rate,
                num_heads=num_heads,
                horizon=horizon
            )
            model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')
            early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=0)
            model.fit(X_tr, y_tr, batch_size=batch_size, epochs=5000,
                      validation_data=(X_val, y_val),
                      shuffle=False, verbose=0, callbacks=[early_stop])

            preds_scaled = model.predict(X_val, verbose=0)
            preds_inv = scaler_Y_full.inverse_transform(preds_scaled)
            y_val_inv = scaler_Y_full.inverse_transform(y_val)
            
            mape_cap = np.mean(np.abs(y_val_inv - preds_inv) / cap) * 100
            mape_scores.append(mape_cap)
            tf.keras.backend.clear_session()

        return np.mean(mape_scores)

    # --- Execução da Otimização ---
    print("\n--- INICIANDO BUSCA DE HIPERPARÂMETROS PARA TFT COM OPTUNA ---")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    print("\n" + "="*50)
    print("OTIMIZAÇÃO DO TFT CONCLUÍDA")
    print("="*50)
    print(f"Melhor resultado (MAPE médio na validação): {study.best_value:.4f}%")
    print("\nHiperparâmetros Ótimos Encontrados:")
    for key, value in study.best_params.items():
        print(f"  - {key}: {value}")
    print("="*50)

    # --- Treinamento Final com os Melhores Parâmetros ---
    print("\n--- TREINANDO MODELO TFT FINAL COM OS MELHORES PARÂMETROS ---")
    
    final_model = build_tft_model(
        input_shape=(look_back, trainX_full.shape[2]),
        hidden_units=best_params['hidden_units'],
        dropout_rate=best_params['dropout_rate'],
        num_heads=best_params['num_heads'],
        horizon=horizon
    )
    final_model.compile(optimizer=Adam(learning_rate=best_params['learning_rate']), loss='mean_squared_error')
    early_stop_final = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)
    final_model.fit(trainX_full, trainY_full,
                    batch_size=best_params['batch_size'],
                    epochs=5000,
                    validation_split=0.2,
                    shuffle=False, verbose=0, callbacks=[early_stop_final])
    print("Treinamento final concluído.")

    # --- Avaliação Final no Conjunto de Teste ---
    y_pred_test = scaler_Y_full.inverse_transform(final_model.predict(testX, verbose=0))
    y_test_inv = scaler_Y_full.inverse_transform(testY)

    metrics_data = []
    print("\n===== MÉTRICAS FINAIS NO CONJUNTO DE TESTE =====")
    for h in range(horizon):
        mape = np.mean(np.abs((y_test_inv[:, h] - y_pred_test[:, h]) / cap)) * 100
        rmse = sqrt(mean_squared_error(y_test_inv[:, h], y_pred_test[:, h]))
        mae = mean_absolute_error(y_test_inv[:, h], y_pred_test[:, h])
        r2 = r2_score(y_test_inv[:, h], y_pred_test[:, h])
        metrics_data.append({'Horizonte': f't+{(h+1)*10} min', 'MAPE (%)': mape, 'RMSE (MW)': rmse, 'MAE (MW)': mae, 'R²': r2})
        print(f"Horizonte t+{(h+1)*10}: MAPE={mape:.4f}%, RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")

    # --- Plotagem dos Resultados Finais ---
    print("\n--- GERANDO GRÁFICOS DE RESULTADOS ---")
    for h in range(horizon):
        plt.figure(figsize=(15, 6))
        plt.plot(y_test_inv[:, h], label=f"Real t+{(h+1)*10}", color="blue", alpha=0.8)
        plt.plot(y_pred_test[:, h], label=f"Previsto (Otimizado) t+{(h+1)*10}", color="red", linestyle="--")
        plt.title(f"Previsão TFT Otimizado vs. Real - Horizonte t+{(h+1)*10} min")
        plt.xlabel("Amostras de Teste")
        plt.ylabel("LV ActivePower (kW)")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

    return final_model, best_params, pd.DataFrame(metrics_data)













# In[6]:

##HYBRID EMD LSTM

def emd_lstm(new_data,i,look_back,data_partition,cap):

    x=i
    data1=new_data.loc[new_data['Month'].isin(x)]
    data1=data1.reset_index(drop=True)
    data1=data1.dropna()
    
    datas=data1['LV ActivePower (kW)']
    datas_wind=pd.DataFrame(datas)
    dfs=datas
    s = dfs.values

    from PyEMD import EMD
    import ewtpy
    
    emd = EMD()

    IMFs = emd(s)

    full_imf=pd.DataFrame(IMFs)
    data_decomp=full_imf.T
    


    pred_test=[]
    test_ori=[]
    pred_train=[]
    train_ori=[]

    epoch=100
    batch_size=64
    neuron=128
    lr=0.001
    optimizer='Adam'

    for col in data_decomp:

        datasetss2=pd.DataFrame(data_decomp[col])
        datasets=datasetss2.values
        train_size = int(len(datasets) * data_partition)
        test_size = len(datasets) - train_size
        train, test = datasets[0:train_size], datasets[train_size:len(datasets)]

        trainX, trainY = create_dataset(train, look_back)
        testX, testY = create_dataset(test, look_back)
        X_train=pd.DataFrame(trainX)
        Y_train=pd.DataFrame(trainY)
        X_test=pd.DataFrame(testX)
        Y_test=pd.DataFrame(testY)
        sc_X = StandardScaler()
        sc_y = StandardScaler()
        X= sc_X.fit_transform(X_train)
        y= sc_y.fit_transform(Y_train)
        X1= sc_X.fit_transform(X_test)
        y1= sc_y.fit_transform(Y_test)
        y=y.ravel()
        y1=y1.ravel()  

        import numpy
        trainX = numpy.reshape(X, (X.shape[0],X.shape[1],1))
        testX = numpy.reshape(X1, (X1.shape[0],X1.shape[1],1))
    
        

        numpy.random.seed(1234)
        import tensorflow as tf
        tf.random.set_seed(1234)

    
        import os 
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Dropout, Activation, LSTM


        neuron=128
        model = Sequential()
        model.add(LSTM(units = neuron,input_shape=(trainX.shape[1], trainX.shape[2])))
        model.add(Dense(1))
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(loss='mse',optimizer=optimizer)


        # Fitting the RNN to the Training set
        model.fit(trainX, y, epochs = epoch, batch_size = batch_size,verbose=0)

        # make predictions
        y_pred_train = model.predict(trainX)
        y_pred_test = model.predict(testX)

        # make predictions

        y_pred_test= numpy.array(y_pred_test).ravel()
        y_pred_test=pd.DataFrame(y_pred_test)
        y1=pd.DataFrame(y1)
        y=pd.DataFrame(y)
        y_pred_train= numpy.array(y_pred_train).ravel()
        y_pred_train=pd.DataFrame(y_pred_train)

        y_test= sc_y.inverse_transform (y1)
        y_train= sc_y.inverse_transform (y)

        y_pred_test1= sc_y.inverse_transform (y_pred_test)
        y_pred_train1= sc_y.inverse_transform (y_pred_train)


        pred_test.append(y_pred_test1)
        test_ori.append(y_test)
        pred_train.append(y_pred_train1)
        train_ori.append(y_train)


    result_pred_test= pd.DataFrame.from_records(pred_test)
    result_pred_train= pd.DataFrame.from_records(pred_train)


    a=result_pred_test.sum(axis = 0, skipna = True) 
    b=result_pred_train.sum(axis = 0, skipna = True) 


    dataframe=pd.DataFrame(dfs)
    dataset=dataframe.values

    train_size = int(len(dataset) * data_partition)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size], dataset[train_size:len(dataset)]

    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    X_train=pd.DataFrame(trainX)
    Y_train=pd.DataFrame(trainY)
    X_test=pd.DataFrame(testX)
    Y_test=pd.DataFrame(testY)

    sc_X = StandardScaler()
    sc_y = StandardScaler() 
    X= sc_X.fit_transform(X_train)
    y= sc_y.fit_transform(Y_train)
    X1= sc_X.fit_transform(X_test)
    y1= sc_y.fit_transform(Y_test)
    y=y.ravel()
    y1=y1.ravel()


    trainX = numpy.reshape(X, (X.shape[0], 1, X.shape[1]))
    testX = numpy.reshape(X1, (X1.shape[0], 1, X1.shape[1]))

    numpy.random.seed(1234)
    tf.random.set_seed(1234)
    
    y1=pd.DataFrame(y1)
    y=pd.DataFrame(y)

    y_test= sc_y.inverse_transform (y1)
    y_train= sc_y.inverse_transform (y)

    a= pd.DataFrame(a)    
    y_test= pd.DataFrame(y_test)    

    #summarize the fit of the model
    mape=numpy.mean((numpy.abs(y_test-a))/cap)*100
    rmse= sqrt(mean_squared_error(y_test,a))
    mae=metrics.mean_absolute_error(y_test,a)

    
    print('MAPE',mape)
    print('RMSE',rmse)
    print('MAE',mae)

    # ======== Gráfico ========
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label='Real', color='blue')
    plt.plot(a, label='Predito (EMD LSTM)', color='orange',linestyle='--')
    plt.title('Previsão com EMD LSTM - Dados de Teste')
    plt.xlabel('Amostras')
    plt.ylabel('LV ActivePower (kW)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# In[7]:


##HYBRID EEMD LSTM

def eemd_lstm(new_data,i,look_back,data_partition,cap):

    x=i
    data1=new_data.loc[new_data['Month'].isin(x)]
    data1=data1.reset_index(drop=True)
    data1=data1.dropna()
    
    datas=data1['LV ActivePower (kW)']
    datas_wind=pd.DataFrame(datas)
    dfs=datas
    s = dfs.values

    from PyEMD import EEMD
    import ewtpy
    
    emd = EEMD(noise_width=0.02)
    emd.noise_seed(12345)

    IMFs = emd(s)

    full_imf=pd.DataFrame(IMFs)
    data_decomp=full_imf.T
    


    pred_test=[]
    test_ori=[]
    pred_train=[]
    train_ori=[]

    epoch=100
    batch_size=64
    neuron=128
    lr=0.001
    optimizer='Adam'

    for col in data_decomp:

        datasetss2=pd.DataFrame(data_decomp[col])
        datasets=datasetss2.values
        train_size = int(len(datasets) * data_partition)
        test_size = len(datasets) - train_size
        train, test = datasets[0:train_size], datasets[train_size:len(datasets)]

        trainX, trainY = create_dataset(train, look_back)
        testX, testY = create_dataset(test, look_back)
        X_train=pd.DataFrame(trainX)
        Y_train=pd.DataFrame(trainY)
        X_test=pd.DataFrame(testX)
        Y_test=pd.DataFrame(testY)
        sc_X = StandardScaler()
        sc_y = StandardScaler()
        X= sc_X.fit_transform(X_train)
        y= sc_y.fit_transform(Y_train)
        X1= sc_X.fit_transform(X_test)
        y1= sc_y.fit_transform(Y_test)
        y=y.ravel()
        y1=y1.ravel()  

        import numpy
        trainX = numpy.reshape(X, (X.shape[0],X.shape[1],1))
        testX = numpy.reshape(X1, (X1.shape[0],X1.shape[1],1))
    
        

        numpy.random.seed(1234)
        import tensorflow as tf
        tf.random.set_seed(1234)

    
        import os 
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Dropout, Activation, LSTM

        neuron=128
        model = Sequential()
        model.add(LSTM(units = neuron,input_shape=(trainX.shape[1], trainX.shape[2])))
        model.add(Dense(1))
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(loss='mse',optimizer=optimizer)


        # Fitting the RNN to the Training set
        model.fit(trainX, y, epochs = epoch, batch_size = batch_size,verbose=0)

        # make predictions
        y_pred_train = model.predict(trainX)
        y_pred_test = model.predict(testX)

        # make predictions

        y_pred_test= numpy.array(y_pred_test).ravel()
        y_pred_test=pd.DataFrame(y_pred_test)
        y1=pd.DataFrame(y1)
        y=pd.DataFrame(y)
        y_pred_train= numpy.array(y_pred_train).ravel()
        y_pred_train=pd.DataFrame(y_pred_train)

        y_test= sc_y.inverse_transform (y1)
        y_train= sc_y.inverse_transform (y)

        y_pred_test1= sc_y.inverse_transform (y_pred_test)
        y_pred_train1= sc_y.inverse_transform (y_pred_train)


        pred_test.append(y_pred_test1)
        test_ori.append(y_test)
        pred_train.append(y_pred_train1)
        train_ori.append(y_train)


    result_pred_test= pd.DataFrame.from_records(pred_test)
    result_pred_train= pd.DataFrame.from_records(pred_train)


    a=result_pred_test.sum(axis = 0, skipna = True) 
    b=result_pred_train.sum(axis = 0, skipna = True) 


    dataframe=pd.DataFrame(dfs)
    dataset=dataframe.values

    train_size = int(len(dataset) * data_partition)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size], dataset[train_size:len(dataset)]

    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    X_train=pd.DataFrame(trainX)
    Y_train=pd.DataFrame(trainY)
    X_test=pd.DataFrame(testX)
    Y_test=pd.DataFrame(testY)

    sc_X = StandardScaler()
    sc_y = StandardScaler() 
    X= sc_X.fit_transform(X_train)
    y= sc_y.fit_transform(Y_train)
    X1= sc_X.fit_transform(X_test)
    y1= sc_y.fit_transform(Y_test)
    y=y.ravel()
    y1=y1.ravel()


    trainX = numpy.reshape(X, (X.shape[0], 1, X.shape[1]))
    testX = numpy.reshape(X1, (X1.shape[0], 1, X1.shape[1]))

    numpy.random.seed(1234)
    tf.random.set_seed(1234)
    
    y1=pd.DataFrame(y1)
    y=pd.DataFrame(y)

    y_test= sc_y.inverse_transform (y1)
    y_train= sc_y.inverse_transform (y)

    a= pd.DataFrame(a)    
    y_test= pd.DataFrame(y_test)    

    #summarize the fit of the model
    mape=numpy.mean((numpy.abs(y_test-a))/cap)*100
    rmse= sqrt(mean_squared_error(y_test,a))
    mae=metrics.mean_absolute_error(y_test,a)

    
    print('MAPE',mape)
    print('RMSE',rmse)
    print('MAE',mae)

    # ======== Gráfico ========
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label='Real', color='blue')
    plt.plot(a, label='Predito (EEMD LSTM)', color='orange', linestyle='--')
    plt.title('Previsão com EEMD LSTM - Dados de Teste')
    plt.xlabel('Amostras')
    plt.ylabel('LV ActivePower (kW)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# In[8]:


##HYBRID CEEMDAN LSTM

def ceemdan_lstm(new_data,i,look_back,data_partition,cap):

    x=i
    data1=new_data.loc[new_data['Month'].isin(x)]
    data1=data1.reset_index(drop=True)
    data1=data1.dropna()
    
    datas=data1['LV ActivePower (kW)']
    datas_wind=pd.DataFrame(datas)
    dfs=datas
    s = dfs.values

    from PyEMD import CEEMDAN
    
    emd = CEEMDAN(epsilon=0.05)
    emd.noise_seed(12345)

    IMFs = emd(s)

    full_imf=pd.DataFrame(IMFs)
    data_decomp=full_imf.T
    


    pred_test=[]
    test_ori=[]
    pred_train=[]
    train_ori=[]

    epoch=100
    batch_size=64
    neuron=128
    lr=0.001
    optimizer='Adam'

    for col in data_decomp:

        datasetss2=pd.DataFrame(data_decomp[col])
        datasets=datasetss2.values
        train_size = int(len(datasets) * data_partition)
        test_size = len(datasets) - train_size
        train, test = datasets[0:train_size], datasets[train_size:len(datasets)]

        trainX, trainY = create_dataset(train, look_back)
        testX, testY = create_dataset(test, look_back)
        X_train=pd.DataFrame(trainX)
        Y_train=pd.DataFrame(trainY)
        X_test=pd.DataFrame(testX)
        Y_test=pd.DataFrame(testY)
        sc_X = StandardScaler()
        sc_y = StandardScaler()
        X= sc_X.fit_transform(X_train)
        y= sc_y.fit_transform(Y_train)
        X1= sc_X.fit_transform(X_test)
        y1= sc_y.fit_transform(Y_test)
        y=y.ravel()
        y1=y1.ravel()  

        import numpy
        trainX = numpy.reshape(X, (X.shape[0],X.shape[1],1))
        testX = numpy.reshape(X1, (X1.shape[0],X1.shape[1],1))
    
        

        numpy.random.seed(1234)
        import tensorflow as tf
        tf.random.set_seed(1234)

    
        import os 
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Dropout, Activation, LSTM


        neuron=128
        model = Sequential()
        model.add(LSTM(units = neuron,input_shape=(trainX.shape[1], trainX.shape[2])))
        model.add(Dense(1))
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(loss='mse',optimizer=optimizer)


        # Fitting the RNN to the Training set
        model.fit(trainX, y, epochs = epoch, batch_size = batch_size,verbose=0)

        # make predictions
        y_pred_train = model.predict(trainX)
        y_pred_test = model.predict(testX)

        # make predictions

        y_pred_test= numpy.array(y_pred_test).ravel()
        y_pred_test=pd.DataFrame(y_pred_test)
        y1=pd.DataFrame(y1)
        y=pd.DataFrame(y)
        y_pred_train= numpy.array(y_pred_train).ravel()
        y_pred_train=pd.DataFrame(y_pred_train)

        y_test= sc_y.inverse_transform (y1)
        y_train= sc_y.inverse_transform (y)

        y_pred_test1= sc_y.inverse_transform (y_pred_test)
        y_pred_train1= sc_y.inverse_transform (y_pred_train)


        pred_test.append(y_pred_test1)
        test_ori.append(y_test)
        pred_train.append(y_pred_train1)
        train_ori.append(y_train)


    result_pred_test= pd.DataFrame.from_records(pred_test)
    result_pred_train= pd.DataFrame.from_records(pred_train)


    a=result_pred_test.sum(axis = 0, skipna = True) 
    b=result_pred_train.sum(axis = 0, skipna = True) 


    dataframe=pd.DataFrame(dfs)
    dataset=dataframe.values

    train_size = int(len(dataset) * data_partition)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size], dataset[train_size:len(dataset)]

    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    X_train=pd.DataFrame(trainX)
    Y_train=pd.DataFrame(trainY)
    X_test=pd.DataFrame(testX)
    Y_test=pd.DataFrame(testY)

    sc_X = StandardScaler()
    sc_y = StandardScaler() 
    X= sc_X.fit_transform(X_train)
    y= sc_y.fit_transform(Y_train)
    X1= sc_X.fit_transform(X_test)
    y1= sc_y.fit_transform(Y_test)
    y=y.ravel()
    y1=y1.ravel()


    trainX = numpy.reshape(X, (X.shape[0], 1, X.shape[1]))
    testX = numpy.reshape(X1, (X1.shape[0], 1, X1.shape[1]))

    numpy.random.seed(1234)
    tf.random.set_seed(1234)
    
    y1=pd.DataFrame(y1)
    y=pd.DataFrame(y)

    y_test= sc_y.inverse_transform (y1)
    y_train= sc_y.inverse_transform (y)

    a= pd.DataFrame(a)    
    y_test= pd.DataFrame(y_test)    

    #summarize the fit of the model
    mape=numpy.mean((numpy.abs(y_test-a))/cap)*100
    rmse= sqrt(mean_squared_error(y_test,a))
    mae=metrics.mean_absolute_error(y_test,a)

    
    print('MAPE',mape)
    print('RMSE',rmse)
    print('MAE',mae)

    # ======== Gráfico ========
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label='Real', color='blue')
    plt.plot(a, label='Predito (CEEMDAN LSTM)', color='orange',linestyle='--')
    plt.title('Previsão com CEEMDAN LSTM - Dados de Teste')
    plt.xlabell('Amostras')
    plt.ylabel('LV ActivePower (kW)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
                
##Proposed Method Hybrid CEEMDAN-EWT LSTM

'''def proposed_method(new_data,i,look_back,data_partition,cap):

    x=i
    data1=new_data.loc[new_data['Month'].isin(x)]
    data1=data1.reset_index(drop=True)
    data1=data1.dropna()
    
    datas=data1['LV ActivePower (kW)']
    datas_wind=pd.DataFrame(datas)
    dfs=datas
    s = dfs.values

    from PyEMD import EMD,EEMD,CEEMDAN
    import numpy

    emd = CEEMDAN(epsilon=0.05)
    emd.noise_seed(12345)

    IMFs = emd(s)

    full_imf=pd.DataFrame(IMFs)
    ceemdan1=full_imf.T
    
    imf1=ceemdan1.iloc[:,0]
    imf_dataps=numpy.array(imf1)
    imf_datasetss= imf_dataps.reshape(-1,1)
    imf_new_datasets=pd.DataFrame(imf_datasetss)

    import ewtpy

    ewt,  mfb ,boundaries = ewtpy.EWT1D(imf1, N =3)
    df_ewt=pd.DataFrame(ewt)

    df_ewt.drop(df_ewt.columns[2],axis=1,inplace=True)
    denoised=df_ewt.sum(axis = 1, skipna = True) 
    ceemdan_without_imf1=ceemdan1.iloc[:,1:]
    new_ceemdan=pd.concat([denoised,ceemdan_without_imf1],axis=1)    
    

    pred_test=[]
    test_ori=[]
    pred_train=[]
    train_ori=[]

    epoch=5000
    batch_size=64
    lr=0.001
    optimizer='Adam'

    for col in new_ceemdan:

        datasetss2=pd.DataFrame(new_ceemdan[col])
        datasets=datasetss2.values
        train_size = int(len(datasets) * data_partition)
        test_size = len(datasets) - train_size
        train, test = datasets[0:train_size], datasets[train_size:len(datasets)]

        trainX, trainY = create_dataset(train, look_back)
        testX, testY = create_dataset(test, look_back)
        X_train=pd.DataFrame(trainX)
        Y_train=pd.DataFrame(trainY)
        X_test=pd.DataFrame(testX)
        Y_test=pd.DataFrame(testY)
        sc_X = StandardScaler()
        sc_y = StandardScaler()
        X= sc_X.fit_transform(X_train)
        y= sc_y.fit_transform(Y_train)
        X1= sc_X.fit_transform(X_test)
        y1= sc_y.fit_transform(Y_test)
        y=y.ravel()
        y1=y1.ravel()  

        import numpy

        trainX = numpy.reshape(X, (X.shape[0], look_back,1))
        testX = numpy.reshape(X1, (X1.shape[0], look_back,1))

        numpy.random.seed(1234)
        import tensorflow as tf
        tf.random.set_seed(1234)

        
    
        import os 
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Dropout, Activation, LSTM
        from tensorflow.keras.callbacks import EarlyStopping


        neuron=32
        model = Sequential()
        model.add(LSTM(units = neuron,input_shape=(look_back, 1)))
        model.add(Dense(1))
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(loss='mse',optimizer=optimizer)

        # EarlyStopping configurado para parar se 'loss' não melhorar por 20 épocas
        early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

        model.fit(trainX, y, epochs = epoch, batch_size = batch_size,validation_split=0.1,shuffle=False,verbose=0,callbacks=[early_stop])

         # make predictions
        y_pred_train = model.predict(trainX)
        y_pred_test = model.predict(testX)
        
        y_pred_test= numpy.array(y_pred_test).ravel()
        y_pred_test=pd.DataFrame(y_pred_test)
        y1=pd.DataFrame(y1)
        y=pd.DataFrame(y)
        y_pred_train= numpy.array(y_pred_train).ravel()
        y_pred_train=pd.DataFrame(y_pred_train)

        y_test= sc_y.inverse_transform (y1)
        y_train= sc_y.inverse_transform (y)

        y_pred_test1= sc_y.inverse_transform (y_pred_test)
        y_pred_train1= sc_y.inverse_transform (y_pred_train)


        pred_test.append(y_pred_test1)
        test_ori.append(y_test)
        pred_train.append(y_pred_train1)
        train_ori.append(y_train)


    result_pred_test= pd.DataFrame.from_records(pred_test)
    result_pred_train= pd.DataFrame.from_records(pred_train)


    a=result_pred_test.sum(axis = 0, skipna = True) 
    b=result_pred_train.sum(axis = 0, skipna = True) 


    dataframe=pd.DataFrame(dfs)
    dataset=dataframe.values

    train_size = int(len(dataset) * data_partition)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size], dataset[train_size:len(dataset)]

    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    X_train=pd.DataFrame(trainX)
    Y_train=pd.DataFrame(trainY)
    X_test=pd.DataFrame(testX)
    Y_test=pd.DataFrame(testY)

    sc_X = StandardScaler()
    sc_y = StandardScaler() 
    X= sc_X.fit_transform(X_train)
    y= sc_y.fit_transform(Y_train)
    X1= sc_X.fit_transform(X_test)
    y1= sc_y.fit_transform(Y_test)
    y=y.ravel()
    y1=y1.ravel()

    import numpy

    #trainX = numpy.reshape(X, (X.shape[0], 1, X.shape[1]))
    #testX = numpy.reshape(X1, (X1.shape[0], 1, X1.shape[1]))

    numpy.random.seed(1234)
    import tensorflow as tf

    y1=pd.DataFrame(y1)
    y=pd.DataFrame(y)

    y_test= sc_y.inverse_transform (y1)
    y_train= sc_y.inverse_transform (y)


    a= pd.DataFrame(a)    
    y_test= pd.DataFrame(y_test)    


    #summarize the fit of the model
    mape=numpy.mean((numpy.abs(y_test-a))/cap)*100
    rmse= sqrt(mean_squared_error(y_test,a))
    mae=metrics.mean_absolute_error(y_test,a)

    
    print('MAPE',mape)
    print('RMSE',rmse)
    print('MAE',mae)

    # ======== Gráfico ========
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label='Real', color='blue')
    plt.plot(a, label='Predito (CEEMDAN EWT LSTM)', color='orange',linestyle='--')
    plt.title('Previsão com CEEMDAN EWT LSTM - Dados de Teste')
    plt.xlabel('Amostras')
    plt.ylabel('LV ActivePower (kW)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

## Proposed Method Hybrid CEEMDAN-EWT LSTM'''




'''# ---------- Hybrid CEEMDAN-EWT-LSTM Multi-Horizonte ----------
def proposed_method(new_data, i, look_back, data_partition, cap, horizon=3):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from math import sqrt
    from sklearn.preprocessing import StandardScaler
    from sklearn import metrics
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.optimizers import Adam
    from PyEMD import CEEMDAN
    import ewtpy
    import tensorflow as tf
    import random, os

    # ---------- Reprodutibilidade ----------
    random.seed(1234)
    np.random.seed(1234)
    tf.random.set_seed(1234)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # ---------- Função de janelas multi-horizonte ----------
    def create_dataset(dataset, look_back=1, horizon=3):
        dataX, dataY = [], []
        for j in range(len(dataset) - look_back - horizon + 1):
            X = dataset[j:(j + look_back), 0]
            Y = dataset[(j + look_back):(j + look_back + horizon), 0]
            dataX.append(X)
            dataY.append(Y)
        return np.array(dataX), np.array(dataY)

    # ---------- Seleção dos dados ----------
    data1 = new_data.loc[new_data['Month'].isin(i)].reset_index(drop=True).dropna()
    dfs = data1['LV ActivePower (kW)']
    s = dfs.values

    # ---------- CEEMDAN ----------
    emd = CEEMDAN(epsilon=0.05)
    emd.noise_seed(12345)
    IMFs = emd(s)
    ceemdan1 = pd.DataFrame(IMFs).T

    # ---------- EWT no 1º IMF ----------
    imf1 = ceemdan1.iloc[:, 0]
    ewt, mfb, boundaries = ewtpy.EWT1D(imf1, N=3)
    df_ewt = pd.DataFrame(ewt)
    df_ewt.drop(df_ewt.columns[2], axis=1, inplace=True)
    denoised = df_ewt.sum(axis=1, skipna=True)
    ceemdan_without_imf1 = ceemdan1.iloc[:, 1:]
    new_ceemdan = pd.concat([denoised, ceemdan_without_imf1], axis=1)

    # ---------- Previsão por componente ----------
    preds, trues = [], []
    epoch, batch_size, neuron = 5000, 64, 32

    for col in new_ceemdan:
        datasets = new_ceemdan[[col]].values
        train_size = int(len(datasets) * data_partition)
        train, test = datasets[:train_size], datasets[train_size:]

        trainX, trainY = create_dataset(train, look_back, horizon)
        testX,  testY  = create_dataset(test, look_back, horizon)

        sc_X, sc_y = StandardScaler(), StandardScaler()
        X_train = sc_X.fit_transform(trainX)
        X_test  = sc_X.transform(testX)
        y_train = sc_y.fit_transform(trainY)
        y_test_scaled = sc_y.transform(testY)

        # reshape para (samples, timesteps, features)
        X_train = X_train.reshape((X_train.shape[0], look_back, 1))
        X_test  = X_test.reshape((X_test.shape[0], look_back, 1))

        # ---------- Modelo LSTM ----------
        model = Sequential([
            LSTM(units=neuron, input_shape=(look_back, 1), return_sequences=False),
            Dropout(0.2),
            Dense(horizon)   # saída multi-horizonte
        ])
        model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))

        early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

        model.fit(
            X_train, y_train,
            epochs=epoch,
            batch_size=batch_size,
            verbose=0,
            shuffle=False,
            validation_split=0.1,
            callbacks=[early_stop]
        )

        # ---------- Previsões ----------
        y_pred_test = model.predict(X_test, verbose=0)
        y_pred_test_inv = sc_y.inverse_transform(y_pred_test)
        y_test_inv = sc_y.inverse_transform(y_test_scaled)

        preds.append(y_pred_test_inv)
        trues.append(y_test_inv)

    # ---------- Reconstrução ----------
    preds = np.array(preds)   # (componentes, samples, horizon)
    trues = np.array(trues)

    y_pred_final = np.sum(preds, axis=0)   # soma dos componentes
    y_true_final = np.sum(trues, axis=0)

    # ---------- Métricas ----------
    mape_h, rmse_h, mae_h = [], [], []
    for step in range(horizon):
        mape_h.append(np.mean(np.abs(y_true_final[:, step] - y_pred_final[:, step]) / cap) * 100)
        rmse_h.append(sqrt(metrics.mean_squared_error(y_true_final[:, step], y_pred_final[:, step])))
        mae_h.append(metrics.mean_absolute_error(y_true_final[:, step], y_pred_final[:, step]))

    print("MAPE por horizonte:", mape_h)
    print("RMSE por horizonte:", rmse_h)
    print("MAE por horizonte:", mae_h)
    print("MAPE médio:", np.mean(mape_h))

    # ---------- Gráficos ----------
    for step in range(horizon):
        plt.figure(figsize=(12, 6))
        plt.plot(y_true_final[:, step], label=f"Real t+{step+1}", color='blue')
        plt.plot(y_pred_final[:, step], label=f"Predito t+{step+1}", color='orange', linestyle='--')
        plt.title(f"Previsão CEEMDAN-EWT-LSTM - Horizonte t+{step+1}")
        plt.xlabel("Amostras")
        plt.ylabel("LV ActivePower (kW)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return {
        "MAPE por horizonte": mape_h,
        "RMSE por horizonte": rmse_h,
        "MAE por horizonte": mae_h,
        "MAPE médio": np.mean(mape_h)
    }
'''

# Em myfunctions.py
# Em myfunctions.py

def proposed_method(new_data, months, look_back, data_partition, cap, horizon=3, strategy_horizon=20, mode='train'):
    """
    Modelo híbrido CEEMDAN-EWT-LSTM (v2) com fluxo de pré-processamento corrigido e adaptado para o dashboard.
    - Corrige o vazamento de dados na etapa de normalização.
    - Implementa modos 'train' e 'inference' e salvamento/carregamento de todos os artefatos.
    """
    # --- Imports ---
    import numpy as np
    import pandas as pd
    from math import sqrt
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.optimizers import Adam
    from PyEMD import CEEMDAN
    import ewtpy
    import tensorflow as tf
    import random
    import os
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    import joblib
    import gc

    # --- Reprodutibilidade e Configurações ---
    random.seed(1234)
    np.random.seed(1234)
    tf.random.set_seed(1234)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    MODEL_DIR = "saved_models/proposed_method"

    # --- Funções Auxiliares ---
    # Reutilizando as funções de perda nomeadas já definidas no arquivo

    def create_dataset(dataset, look_back, horizon):
        dataX, dataY = [], []
        for j in range(len(dataset) - look_back - horizon + 1):
            X = dataset[j:(j + look_back), 0]
            Y = dataset[(j + look_back):(j + look_back + horizon), 0]
            dataX.append(X)
            dataY.append(Y)
        return np.array(dataX), np.array(dataY)

    # --- Decomposição e Preparação ---
    data1 = new_data.loc[new_data['Month'].isin(months)].reset_index(drop=True).dropna()
    
    if mode == 'train':
        print("--- EXECUTANDO proposed_method EM MODO DE TREINAMENTO ---")
        print("Iniciando decomposição do sinal (CEEMDAN + EWT)...")
        s = data1['LV ActivePower (kW)'].values
        emd = CEEMDAN(epsilon=0.05); emd.noise_seed(12345)
        IMFs = emd(s)
        ceemdan1 = pd.DataFrame(IMFs).T
        imf1 = ceemdan1.iloc[:, 0].values
        ewt, mfb, boundaries = ewtpy.EWT1D(imf1, N=3)
        df_ewt = pd.DataFrame(ewt)
        if df_ewt.shape[1] > 2: df_ewt.drop(df_ewt.columns[2], axis=1, inplace=True)
        denoised = df_ewt.sum(axis=1)
        ceemdan_without_imf1 = ceemdan1.iloc[:, 1:]
        ceemdan_without_imf1.columns = [f"imf_{c}" for c in ceemdan_without_imf1.columns]
        denoised.name = "imf_denoised"
        decomposed_df = pd.concat([denoised, ceemdan_without_imf1], axis=1)
        
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(decomposed_df, os.path.join(MODEL_DIR, "decomposed_df.gz"))
        print("Decomposição concluída e salva.")
        
    elif mode == 'inference':
        print("--- EXECUTANDO proposed_method EM MODO DE INFERÊNCIA ---")
        if not os.path.exists(os.path.join(MODEL_DIR, "decomposed_df.gz")):
            raise FileNotFoundError("Arquivo de decomposição 'decomposed_df.gz' não encontrado.")
        decomposed_df = joblib.load(os.path.join(MODEL_DIR, "decomposed_df.gz"))
        print("Decomposição carregada.")
        
    else:
        raise ValueError("O parâmetro 'mode' deve ser 'train' ou 'inference'.")

    component_names = decomposed_df.columns.tolist()
    quantiles = [0.1, 0.5, 0.9]
    all_predictions = {}

    for q in quantiles:
        imf_predictions_for_quantile_q = []
        quantile_dir = os.path.join(MODEL_DIR, f"q{str(q).replace('.', '')}")

        for col_name in component_names:
            tf.keras.backend.clear_session()
            datasets = decomposed_df[[col_name]].values
            train_size = int(len(datasets) * data_partition)
            train, test = datasets[:train_size], datasets[train_size:]
            trainX, trainY = create_dataset(train, look_back, horizon)
            testX, _ = create_dataset(test, look_back, horizon)

            if trainX.shape[0] == 0: continue

            if mode == 'train':
                print(f"--- Q{q} | Treinando LSTM para: {col_name} ---")
                
                # CORREÇÃO: Ajustar scalers APENAS no treino
                sc_X, sc_y = StandardScaler(), StandardScaler()
                sc_X.fit(trainX)
                sc_y.fit(trainY)

                # Usar transform para todos
                X_train = sc_X.transform(trainX)
                y_train = sc_y.transform(trainY)
                X_test = sc_X.transform(testX)

                X_train = X_train.reshape((X_train.shape[0], look_back, 1))
                X_test = X_test.reshape((X_test.shape[0], look_back, 1))

                model = Sequential([LSTM(units=32, input_shape=(look_back, 1)), Dropout(0.2), Dense(horizon)])
                model.compile(loss=loss_functions[q], optimizer=Adam(learning_rate=0.001))
                early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
                model.fit(X_train, y_train, epochs=5000, batch_size=64, verbose=0, shuffle=False, validation_split=0.1, callbacks=[early_stop])

                y_pred_scaled = model.predict(X_test, verbose=0)
                y_pred_inv = sc_y.inverse_transform(y_pred_scaled)
                
                os.makedirs(quantile_dir, exist_ok=True)
                model.save(os.path.join(quantile_dir, f"model_{col_name}.keras"))
                joblib.dump(sc_X, os.path.join(quantile_dir, f"scaler_X_{col_name}.gz"))
                joblib.dump(sc_y, os.path.join(quantile_dir, f"scaler_y_{col_name}.gz"))

            elif mode == 'inference':
                # print(f"--- Q{q} | Inferência para: {col_name} ---")
                sc_X = joblib.load(os.path.join(quantile_dir, f"scaler_X_{col_name}.gz"))
                sc_y = joblib.load(os.path.join(quantile_dir, f"scaler_y_{col_name}.gz"))
                custom_objects = {fn.__name__: fn for fn in loss_functions.values()}
                model = tf.keras.models.load_model(os.path.join(quantile_dir, f"model_{col_name}.keras"), custom_objects=custom_objects)

                X_test = sc_X.transform(testX)
                X_test = X_test.reshape((X_test.shape[0], look_back, 1))
                y_pred_scaled = model.predict(X_test, verbose=0)
                y_pred_inv = sc_y.inverse_transform(y_pred_scaled)
            
            imf_predictions_for_quantile_q.append(y_pred_inv)
            gc.collect()

        reconstructed_prediction = np.sum(np.array(imf_predictions_for_quantile_q), axis=0)
        all_predictions[q] = np.clip(reconstructed_prediction, 0, cap)

    # Preparação dos dados reais para validação
    original_values = data1[['LV ActivePower (kW)']].values
    train_size_orig = int(len(original_values) * data_partition)
    _, test_orig = original_values[:train_size_orig], original_values[train_size_orig:]
    _, y_test_inv = create_dataset(test_orig, look_back, horizon)

    # Correção de cruzamento de quantis
    print("Aplicando correção de cruzamento de quantis...")
    all_predictions[0.5] = np.maximum(all_predictions[0.1], all_predictions[0.5])
    all_predictions[0.9] = np.maximum(all_predictions[0.5], all_predictions[0.9])

    if mode == 'train':
        # Seção de métricas e gráficos (código omitido por brevidade, mas é o mesmo que você já tem)
        y_pred_median = all_predictions[0.5]
        print("\n===== Métricas Completas (CEEMDAN-EWT-LSTM) =====")
        for h in range(horizon):
            y_true_h, y_pred_h = y_test_inv[:, h], y_pred_median[:, h]
            mape = np.mean(np.abs((y_true_h - y_pred_h) / cap)) * 100 if cap > 0 else 0
            rmse = sqrt(mean_squared_error(y_true_h, y_pred_h))
            mae = mean_absolute_error(y_true_h, y_pred_h)
            r2 = r2_score(y_true_h, y_pred_h)
            print(f"Horizonte t+{(h+1)*10}: MAPE={mape:.4f}%, RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")

        # Lógica de Risco ORI e plotagem
        operational_threshold = 0.3 * cap 
        p_low_t20 = all_predictions[0.1][:, 1]
        p_low_t30 = all_predictions[0.1][:, 2]
        ori_levels = []
        if strategy_horizon == 20:
            for i in range(len(p_low_t20)):
                if p_low_t20[i] < operational_threshold: ori_levels.append('Alto')
                elif p_low_t30[i] < operational_threshold: ori_levels.append('Atenção')
                else: ori_levels.append('Baixo')
        else:
            for p_low in p_low_t30:
                ori_levels.append('Alto' if p_low < operational_threshold else 'Baixo')

        color_map = {'Baixo': 'lightgreen', 'Atenção': 'gold', 'Alto': 'salmon'}
        num_samples_to_plot = 300
        time_axis = np.arange(min(num_samples_to_plot, len(y_test_inv)))

        for h in range(horizon):
            fig, ax = plt.subplots(figsize=(18, 8))
            for i in range(len(time_axis)):
                ax.axvspan(i, i + 1, facecolor=color_map.get(ori_levels[i], 'white'), alpha=0.5, zorder=0)
            ax.axhline(y=operational_threshold, color='red', linestyle=':', linewidth=2.5, label=f'Limiar Operacional ({operational_threshold:.2f} kW)', zorder=4)
            ax.fill_between(time_axis, all_predictions[0.1][:len(time_axis), h], all_predictions[0.9][:len(time_axis), h], color='cornflowerblue', alpha=0.6, label='Intervalo de Confiança (80%)', zorder=1)
            ax.plot(time_axis, y_test_inv[:len(time_axis), h], label="Real", color='black', linewidth=2, zorder=3)
            ax.plot(time_axis, all_predictions[0.5][:len(time_axis), h], label="Previsão Mediana", color='firebrick', linestyle='--', linewidth=2.5, zorder=2)
            ax.set_title(f'Previsão Híbrida (CEEMDAN-EWT-LSTM) com ORI - Horizonte t+{(h+1)*10} min', fontsize=16)
            ax.set_ylabel('LV ActivePower (kW)', fontsize=12)
            ax.set_xlabel('Amostras de Teste', fontsize=12)
            risk_patches = [Patch(facecolor=color, alpha=0.5, label=f'Risco {level}') for level, color in color_map.items() if level in set(ori_levels)]
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles=handles + risk_patches, loc='best', fontsize=12)
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
            ax.set_xlim(0, len(time_axis))
            ax.set_ylim(bottom=max(0, y_test_inv.min() - (0.05 * cap)))
            plt.tight_layout()
            plt.show()
            plt.close(fig)

    return all_predictions, y_test_inv

# Em myfunctions.py

# Adicione esta nova função. Ela é uma cópia da sua `proposed_method`, mas com GRU.
def proposed_method_gru(new_data, months, look_back, data_partition, cap, horizon=3, strategy_horizon=20, mode='train'):
    """
    Modelo híbrido CEEMDAN-EWT-GRU, adaptado do CEEMDAN-EWT-LSTM.
    - Utiliza GRU em vez de LSTM para cada componente.
    - Mantém todas as boas práticas do modelo original.
    """
    # --- Imports ---
    import numpy as np
    import pandas as pd
    from math import sqrt
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from tensorflow.keras.models import Sequential
    # >>>>> MUDANÇA 1: Importar a camada GRU <<<<<
    from tensorflow.keras.layers import GRU, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.optimizers import Adam
    from PyEMD import CEEMDAN
    import ewtpy
    import tensorflow as tf
    import random
    import os
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    import joblib
    import gc

    # --- Reprodutibilidade e Configurações ---
    random.seed(1234)
    np.random.seed(1234)
    tf.random.set_seed(1234)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    # >>>>> MUDANÇA 2: Novo diretório para salvar os modelos GRU <<<<<
    MODEL_DIR = "saved_models/proposed_method_gru"

    # --- Funções Auxiliares (Inalteradas) ---
    # (Supondo que `loss_functions` está definido no escopo global do seu arquivo, como antes)
    def create_dataset(dataset, look_back, horizon):
        dataX, dataY = [], []
        for j in range(len(dataset) - look_back - horizon + 1):
            X = dataset[j:(j + look_back), 0]
            Y = dataset[(j + look_back):(j + look_back + horizon), 0]
            dataX.append(X)
            dataY.append(Y)
        return np.array(dataX), np.array(dataY)

    # --- Decomposição e Preparação (Inalteradas) ---
    data1 = new_data.loc[new_data['Month'].isin(months)].reset_index(drop=True).dropna()
    
    if mode == 'train':
        print("--- EXECUTANDO proposed_method_gru EM MODO DE TREINAMENTO ---")
        print("Iniciando decomposição do sinal (CEEMDAN + EWT)...")
        s = data1['LV ActivePower (kW)'].values
        emd = CEEMDAN(epsilon=0.05); emd.noise_seed(12345)
        IMFs = emd(s)
        ceemdan1 = pd.DataFrame(IMFs).T
        imf1 = ceemdan1.iloc[:, 0].values
        ewt, mfb, boundaries = ewtpy.EWT1D(imf1, N=3)
        df_ewt = pd.DataFrame(ewt)
        if df_ewt.shape[1] > 2: df_ewt.drop(df_ewt.columns[2], axis=1, inplace=True)
        denoised = df_ewt.sum(axis=1)
        ceemdan_without_imf1 = ceemdan1.iloc[:, 1:]
        ceemdan_without_imf1.columns = [f"imf_{c}" for c in ceemdan_without_imf1.columns]
        denoised.name = "imf_denoised"
        decomposed_df = pd.concat([denoised, ceemdan_without_imf1], axis=1)
        
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(decomposed_df, os.path.join(MODEL_DIR, "decomposed_df.gz"))
        print("Decomposição concluída e salva.")
        
    elif mode == 'inference':
        print("--- EXECUTANDO proposed_method_gru EM MODO DE INFERÊNCIA ---")
        decomposed_path = os.path.join(MODEL_DIR, "decomposed_df.gz")
        if not os.path.exists(decomposed_path):
            raise FileNotFoundError(f"Arquivo de decomposição '{decomposed_path}' não encontrado.")
        decomposed_df = joblib.load(decomposed_path)
        print("Decomposição carregada.")
        
    else:
        raise ValueError("O parâmetro 'mode' deve ser 'train' ou 'inference'.")

    component_names = decomposed_df.columns.tolist()
    quantiles = [0.1, 0.5, 0.9]
    all_predictions = {}

    for q in quantiles:
        imf_predictions_for_quantile_q = []
        quantile_dir = os.path.join(MODEL_DIR, f"q{str(q).replace('.', '')}")

        for col_name in component_names:
            tf.keras.backend.clear_session()
            datasets = decomposed_df[[col_name]].values
            train_size = int(len(datasets) * data_partition)
            train, test = datasets[:train_size], datasets[train_size:]
            trainX, trainY = create_dataset(train, look_back, horizon)
            testX, _ = create_dataset(test, look_back, horizon)

            if trainX.shape[0] == 0: continue

            if mode == 'train':
                # >>>>> MUDANÇA 3: Atualizar print <<<<<
                print(f"--- Q{q} | Treinando GRU para: {col_name} ---")
                
                sc_X, sc_y = StandardScaler(), StandardScaler()
                sc_X.fit(trainX)
                sc_y.fit(trainY)

                X_train = sc_X.transform(trainX)
                y_train = sc_y.transform(trainY)
                X_test = sc_X.transform(testX)

                X_train = X_train.reshape((X_train.shape[0], look_back, 1))
                X_test = X_test.reshape((X_test.shape[0], look_back, 1))

                # >>>>> MUDANÇA 4: Substituir LSTM por GRU <<<<<
                model = Sequential([
                    GRU(units=32, input_shape=(look_back, 1)), 
                    Dropout(0.2), 
                    Dense(horizon)
                ])
                model.compile(loss=loss_functions[q], optimizer=Adam(learning_rate=0.001))
                early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
                model.fit(X_train, y_train, epochs=5000, batch_size=64, verbose=0, shuffle=False, validation_split=0.1, callbacks=[early_stop])

                y_pred_scaled = model.predict(X_test, verbose=0)
                y_pred_inv = sc_y.inverse_transform(y_pred_scaled)
                
                os.makedirs(quantile_dir, exist_ok=True)
                model.save(os.path.join(quantile_dir, f"model_{col_name}.keras"))
                joblib.dump(sc_X, os.path.join(quantile_dir, f"scaler_X_{col_name}.gz"))
                joblib.dump(sc_y, os.path.join(quantile_dir, f"scaler_y_{col_name}.gz"))

            elif mode == 'inference':
                sc_X = joblib.load(os.path.join(quantile_dir, f"scaler_X_{col_name}.gz"))
                sc_y = joblib.load(os.path.join(quantile_dir, f"scaler_y_{col_name}.gz"))
                custom_objects = {fn.__name__: fn for fn in loss_functions.values()}
                model = tf.keras.models.load_model(os.path.join(quantile_dir, f"model_{col_name}.keras"), custom_objects=custom_objects)

                X_test = sc_X.transform(testX)
                X_test = X_test.reshape((X_test.shape[0], look_back, 1))
                y_pred_scaled = model.predict(X_test, verbose=0)
                y_pred_inv = sc_y.inverse_transform(y_pred_scaled)
            
            imf_predictions_for_quantile_q.append(y_pred_inv)
            gc.collect()

        reconstructed_prediction = np.sum(np.array(imf_predictions_for_quantile_q), axis=0)
        all_predictions[q] = np.clip(reconstructed_prediction, 0, cap)

    # Preparação dos dados reais para validação
    original_values = data1[['LV ActivePower (kW)']].values
    train_size_orig = int(len(original_values) * data_partition)
    _, test_orig = original_values[:train_size_orig], original_values[train_size_orig:]
    _, y_test_inv = create_dataset(test_orig, look_back, horizon)

    # Correção de cruzamento de quantis
    print("Aplicando correção de cruzamento de quantis...")
    all_predictions[0.5] = np.maximum(all_predictions[0.1], all_predictions[0.5])
    all_predictions[0.9] = np.maximum(all_predictions[0.5], all_predictions[0.9])

    if mode == 'train':
        y_pred_median = all_predictions[0.5]
        # >>>>> MUDANÇA 5: Atualizar título das métricas <<<<<
        print("\n===== Métricas Completas (CEEMDAN-EWT-GRU) =====")
        for h in range(horizon):
            y_true_h, y_pred_h = y_test_inv[:, h], y_pred_median[:, h]
            mape = np.mean(np.abs((y_true_h - y_pred_h) / cap)) * 100 if cap > 0 else 0
            rmse = sqrt(mean_squared_error(y_true_h, y_pred_h))
            mae = mean_absolute_error(y_true_h, y_pred_h)
            r2 = r2_score(y_true_h, y_pred_h)
            print(f"Horizonte t+{(h+1)*10}: MAPE={mape:.4f}%, RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")

        # Lógica de Risco ORI e plotagem
        operational_threshold = 0.3 * cap 
        p_low_t20 = all_predictions[0.1][:, 1]
        p_low_t30 = all_predictions[0.1][:, 2]
        ori_levels = []
        if strategy_horizon == 20:
            for i in range(len(p_low_t20)):
                if p_low_t20[i] < operational_threshold: ori_levels.append('Alto')
                elif p_low_t30[i] < operational_threshold: ori_levels.append('Atenção')
                else: ori_levels.append('Baixo')
        else:
            for p_low in p_low_t30:
                ori_levels.append('Alto' if p_low < operational_threshold else 'Baixo')

        color_map = {'Baixo': 'lightgreen', 'Atenção': 'gold', 'Alto': 'salmon'}
        num_samples_to_plot = 300
        time_axis = np.arange(min(num_samples_to_plot, len(y_test_inv)))

        for h in range(horizon):
            fig, ax = plt.subplots(figsize=(18, 8))
            for i in range(len(time_axis)):
                ax.axvspan(i, i + 1, facecolor=color_map.get(ori_levels[i], 'white'), alpha=0.5, zorder=0)
            ax.axhline(y=operational_threshold, color='red', linestyle=':', linewidth=2.5, label=f'Limiar Operacional ({operational_threshold:.2f} kW)', zorder=4)
            ax.fill_between(time_axis, all_predictions[0.1][:len(time_axis), h], all_predictions[0.9][:len(time_axis), h], color='cornflowerblue', alpha=0.6, label='Intervalo de Confiança (80%)', zorder=1)
            ax.plot(time_axis, y_test_inv[:len(time_axis), h], label="Real", color='black', linewidth=2, zorder=3)
            ax.plot(time_axis, all_predictions[0.5][:len(time_axis), h], label="Previsão Mediana", color='firebrick', linestyle='--', linewidth=2.5, zorder=2)
            # >>>>> MUDANÇA 6: Atualizar título do gráfico <<<<<
            ax.set_title(f'Previsão Híbrida (CEEMDAN-EWT-GRU) com ORI - Horizonte t+{(h+1)*10} min', fontsize=16)
            ax.set_ylabel('LV ActivePower (kW)', fontsize=12)
            ax.set_xlabel('Amostras de Teste', fontsize=12)
            risk_patches = [Patch(facecolor=color, alpha=0.5, label=f'Risco {level}') for level, color in color_map.items() if level in set(ori_levels)]
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles=handles + risk_patches, loc='best', fontsize=12)
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
            ax.set_xlim(0, len(time_axis))
            ax.set_ylim(bottom=max(0, y_test_inv.min() - (0.05 * cap)))
            plt.tight_layout()
            plt.show()
            plt.close(fig)

    return all_predictions, y_test_inv



# ====================================================================================
# FUNÇÃO COMPLETA PARA O MODELO HÍBRIDO (CEEMDAN-EWT)-TFT AGREGADOR -
# ====================================================================================
# Adicione esta nova função ao seu myfunctions.py

def proposed_method_tft_aggregator(new_data, months, look_back, data_partition, cap, horizon=3, strategy_horizon=20, mode='train'):
    """
    Modelo Híbrido (CEEMDAN-EWT)-TFT Agregador (v1), adaptado para o padrão do dashboard.
    - Decompõe o sinal e usa os componentes como features multivariadas para um único TFT.
    - Herda todas as boas práticas: sem data leakage, modos de treino/inferência, etc.
    """
    # --- Imports ---
    import numpy as np
    import pandas as pd
    from math import sqrt
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from tensorflow.keras import layers, Model, callbacks
    from PyEMD import CEEMDAN
    from tensorflow.keras.optimizers import Adam
    import ewtpy
    import tensorflow as tf
    import random
    import os
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    import joblib

    # --- Reprodutibilidade e Configurações ---
    random.seed(1234)
    np.random.seed(1234)
    tf.random.set_seed(1234)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    MODEL_DIR = "saved_models/proposed_method_tft_aggregator"

    # --- Blocos de Construção do TFT (Inalterados) ---
    quantiles = [0.1, 0.5, 0.9]
    def quantile_loss(y_true, y_pred):
        y_true_expanded = tf.expand_dims(y_true, axis=-1); error = y_true_expanded - y_pred
        q_tensor = tf.constant(quantiles, dtype=tf.float32); loss = tf.maximum(q_tensor * error, (q_tensor - 1) * error)
        return tf.reduce_mean(loss)
    class GatedLinearUnit(layers.Layer):
        def __init__(self, units, **kwargs): super().__init__(**kwargs); self.linear = layers.Dense(units); self.sigmoid = layers.Dense(units, activation="sigmoid")
        def call(self, inputs): return self.linear(inputs) * self.sigmoid(inputs)
    class GatedResidualNetwork(layers.Layer):
        def __init__(self, units, dropout_rate, **kwargs):
            super().__init__(**kwargs); self.units = units; self.elu_dense = layers.Dense(units, activation="elu"); self.linear_dense = layers.Dense(units); self.dropout = layers.Dropout(dropout_rate); self.gated_linear_unit = GatedLinearUnit(units); self.layer_norm = layers.LayerNormalization(); self.project = layers.Dense(units)
        def call(self, inputs):
            x = self.elu_dense(inputs); x = self.linear_dense(x); x = self.dropout(x)
            if inputs.shape[-1] != self.units: inputs = self.project(inputs)
            x = inputs + self.gated_linear_unit(x); x = self.layer_norm(x); return x

    # --- Decomposição e Preparação dos Dados ---
    data1 = new_data.loc[new_data['Month'].isin(months)].reset_index(drop=True).dropna()
    
    if mode == 'train':
        print("--- EXECUTANDO (CEEMDAN-EWT)-TFT EM MODO DE TREINAMENTO ---")
        print("Iniciando decomposição do sinal (CEEMDAN + EWT)...")
        s = data1['LV ActivePower (kW)'].values
        emd = CEEMDAN(epsilon=0.05); emd.noise_seed(12345)
        IMFs = emd(s)
        ceemdan1 = pd.DataFrame(IMFs).T
        imf1 = ceemdan1.iloc[:, 0].values
        ewt, mfb, boundaries = ewtpy.EWT1D(imf1, N=3)
        df_ewt = pd.DataFrame(ewt)
        if df_ewt.shape[1] > 2: df_ewt.drop(df_ewt.columns[2], axis=1, inplace=True)
        denoised = df_ewt.sum(axis=1, skipna=True)
        ceemdan_without_imf1 = ceemdan1.iloc[:, 1:]
        features_df = pd.concat([pd.Series(denoised, name='imf_denoised'), ceemdan_without_imf1], axis=1)
        features_df.columns = features_df.columns.astype(str)
        
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(features_df, os.path.join(MODEL_DIR, "features_df.gz"))
        print("Decomposição concluída e features salvas.")
        
    elif mode == 'inference':
        print("--- EXECUTANDO (CEEMDAN-EWT)-TFT EM MODO DE INFERÊNCIA ---")
        features_path = os.path.join(MODEL_DIR, "features_df.gz")
        if not os.path.exists(features_path):
            raise FileNotFoundError(f"Arquivo de features decompostas '{features_path}' não encontrado.")
        features_df = joblib.load(features_path)
        print("Features decompostas carregadas.")
        
    else:
        raise ValueError("O parâmetro 'mode' deve ser 'train' ou 'inference'.")

    target_df = data1[['LV ActivePower (kW)']]
    train_size = int(len(features_df) * data_partition)
    train_features_df, test_features_df = features_df.iloc[:train_size], features_df.iloc[train_size:]
    train_target_df, test_target_df = target_df.iloc[:train_size], target_df.iloc[train_size:]

    def create_dataset_multivariate(features, target, look_back, horizon):
        X, Y = [], []
        for j in range(len(features) - look_back - horizon + 1):
            X.append(features[j:j + look_back, :])
            Y.append(target[j + look_back:j + look_back + horizon, 0])
        return np.array(X), np.array(Y)

    if mode == 'train':
        scaler_X = StandardScaler().fit(train_features_df)
        scaler_Y = StandardScaler().fit(train_target_df)
        joblib.dump(scaler_X, os.path.join(MODEL_DIR, "scaler_X.gz"))
        joblib.dump(scaler_Y, os.path.join(MODEL_DIR, "scaler_Y.gz"))
    else: # inference
        scaler_X = joblib.load(os.path.join(MODEL_DIR, "scaler_X.gz"))
        scaler_Y = joblib.load(os.path.join(MODEL_DIR, "scaler_Y.gz"))

    train_features_scaled = scaler_X.transform(train_features_df)
    test_features_scaled = scaler_X.transform(test_features_df)
    train_target_scaled = scaler_Y.transform(train_target_df)
    test_target_scaled = scaler_Y.transform(test_target_df)
    
    trainX, trainY = create_dataset_multivariate(train_features_scaled, train_target_scaled, look_back, horizon)
    testX, testY = create_dataset_multivariate(test_features_scaled, test_target_scaled, look_back, horizon)
    y_test_inv = scaler_Y.inverse_transform(testY)

    if mode == 'train':
        print("Construindo e treinando o modelo (CEEMDAN-EWT)-TFT...")
        hidden_units, dropout_rate, num_heads, num_features = 32, 0.26010475808037614, 4, trainX.shape[2]
        input_shape = (look_back, num_features)
        inputs = layers.Input(shape=input_shape)
        x = GatedResidualNetwork(hidden_units, dropout_rate)(inputs); x = GatedResidualNetwork(hidden_units, dropout_rate)(x)
        attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=hidden_units, dropout=dropout_rate)(x, x)
        attention_output = layers.Dropout(dropout_rate)(attention_output); x = layers.LayerNormalization()(x + attention_output)
        x = GatedResidualNetwork(hidden_units, dropout_rate)(x); x = layers.Flatten()(x)
        outputs = layers.Dense(horizon * len(quantiles))(x)
        outputs = layers.Reshape((horizon, len(quantiles)))(outputs)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss=quantile_loss, optimizer=Adam(learning_rate=0.0018087787786653733))
        early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
        model.fit(trainX, trainY, epochs=5000, batch_size=64, validation_split=0.2, callbacks=[early_stop], verbose=0)
        model.save(os.path.join(MODEL_DIR, "tft_aggregator_model.keras"))
        print("Modelo TFT Agregador salvo com sucesso.")
    
    elif mode == 'inference':
        custom_objects = {"GatedLinearUnit": GatedLinearUnit, "GatedResidualNetwork": GatedResidualNetwork, "quantile_loss": quantile_loss}
        model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "tft_aggregator_model.keras"), custom_objects=custom_objects)

    # Previsão (comum a ambos os modos)
    raw_predictions_scaled = model.predict(testX)
    all_predictions = {}
    for i_q, q in enumerate(quantiles):
        preds_q_scaled = raw_predictions_scaled[:, :, i_q]
        preds_q_unscaled = scaler_Y.inverse_transform(preds_q_scaled)
        all_predictions[q] = np.clip(preds_q_unscaled, 0, cap)

    # Correção de cruzamento de quantis
    print("Aplicando correção de cruzamento de quantis...")
    all_predictions[0.5] = np.maximum(all_predictions[0.1], all_predictions[0.5])
    all_predictions[0.9] = np.maximum(all_predictions[0.5], all_predictions[0.9])

    if mode == 'train':
        # Seção de métricas e gráficos (sem alterações, apenas o título)
        y_pred_median = all_predictions[0.5]
        print("\n===== Métricas Completas (CEEMDAN-EWT-TFT) =====")
        for h in range(horizon):
            y_true_h, y_pred_h = y_test_inv[:, h], y_pred_median[:, h]
            mape = np.mean(np.abs((y_true_h - y_pred_h) / cap)) * 100 if cap > 0 else 0
            rmse = sqrt(mean_squared_error(y_true_h, y_pred_h))
            mae = mean_absolute_error(y_true_h, y_pred_h)
            r2 = r2_score(y_true_h, y_pred_h)
            print(f"Horizonte t+{(h+1)*10}: MAPE={mape:.4f}%, RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")

        # Lógica de Risco ORI e plotagem
        operational_threshold = 0.3 * cap 
        p_low_t20 = all_predictions[0.1][:, 1]
        p_low_t30 = all_predictions[0.1][:, 2]
        ori_levels = []
        if strategy_horizon == 20:
            for i in range(len(p_low_t20)):
                if p_low_t20[i] < operational_threshold: ori_levels.append('Alto')
                elif p_low_t30[i] < operational_threshold: ori_levels.append('Atenção')
                else: ori_levels.append('Baixo')
        else:
            for p_low in p_low_t30:
                ori_levels.append('Alto' if p_low < operational_threshold else 'Baixo')

        color_map = {'Baixo': 'lightgreen', 'Atenção': 'gold', 'Alto': 'salmon'}
        num_samples_to_plot = 300
        time_axis = np.arange(min(num_samples_to_plot, len(y_test_inv)))

        for h in range(horizon):
            fig, ax = plt.subplots(figsize=(18, 8))
            for i in range(len(time_axis)):
                ax.axvspan(i, i + 1, facecolor=color_map.get(ori_levels[i], 'white'), alpha=0.5, zorder=0)
            ax.axhline(y=operational_threshold, color='red', linestyle=':', linewidth=2.5, label=f'Limiar Operacional ({operational_threshold:.2f} kW)', zorder=4)
            ax.fill_between(time_axis, all_predictions[0.1][:len(time_axis), h], all_predictions[0.9][:len(time_axis), h], color='cornflowerblue', alpha=0.6, label='Intervalo de Confiança (80%)', zorder=1)
            ax.plot(time_axis, y_test_inv[:len(time_axis), h], label="Real", color='black', linewidth=2, zorder=3)
            ax.plot(time_axis, all_predictions[0.5][:len(time_axis), h], label="Previsão Mediana", color='firebrick', linestyle='--', linewidth=2.5, zorder=2)
            ax.set_title(f'Previsão Híbrida (CEEMDAN-EWT)-TFT Agregador com ORI - Horizonte t+{(h+1)*10} min', fontsize=16)
            ax.set_ylabel('LV ActivePower (kW)', fontsize=12)
            ax.set_xlabel('Amostras de Teste', fontsize=12)
            risk_patches = [Patch(facecolor=color, alpha=0.5, label=f'Risco {level}') for level, color in color_map.items() if level in set(ori_levels)]
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles=handles + risk_patches, loc='best', fontsize=12)
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
            ax.set_xlim(0, len(time_axis))
            ax.set_ylim(bottom=max(0, y_test_inv.min() - (0.05 * cap)))
            plt.tight_layout()
            plt.show()
            plt.close(fig)

    return all_predictions, y_test_inv

# Em myfunctions.py
# Adicione esta NOVA função 100% COMPLETA ao seu arquivo.

def proposed_method_tft_aggregator_interp(new_data, months, look_back, data_partition, cap, horizon=3, strategy_horizon=20, mode='train'):
    """
    [VERSÃO TFT-BALANCED]
    Uma versão que segue uma estrutura clássica de Encoder-Decoder com atenção.
    - Usa UM Gated Residual Network (GRN) para processar as features de entrada (Encoder).
    - Aplica a camada de Atenção.
    - Usa UM Gated Residual Network (GRN) para processar a saída da atenção (Decoder).
    - O objetivo é testar uma arquitetura balanceada, mais simples que a original mas mais robusta que as versões 'Lite'.
    - Salva artefatos em 'proposed_method_tft_balanced'.
    """
    # --- Imports e Configurações ---
    import numpy as np
    import pandas as pd
    from math import sqrt
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from tensorflow.keras import layers, Model, callbacks
    from PyEMD import CEEMDAN
    from tensorflow.keras.optimizers import Adam
    import ewtpy
    import tensorflow as tf
    import random
    import os
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    import joblib

    random.seed(1234)
    np.random.seed(1234)
    tf.random.set_seed(1234)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    MODEL_DIR = "saved_models/proposed_method_tft_aggregator_interp"

    # --- Blocos de Construção do TFT ---
    quantiles = [0.1, 0.5, 0.9]
    def quantile_loss(y_true, y_pred):
        y_true_expanded = tf.expand_dims(y_true, axis=-1)
        error = y_true_expanded - y_pred
        q_tensor = tf.constant(quantiles, dtype=tf.float32)
        loss = tf.maximum(q_tensor * error, (q_tensor - 1) * error)
        return tf.reduce_mean(loss)

    class GatedLinearUnit(layers.Layer):
        def __init__(self, units, **kwargs):
            super().__init__(**kwargs)
            self.linear = layers.Dense(units)
            self.sigmoid = layers.Dense(units, activation="sigmoid")
        def call(self, inputs):
            return self.linear(inputs) * self.sigmoid(inputs)
        def get_config(self):
            config = super().get_config()
            config.update({'units': self.linear.units})
            return config

    class GatedResidualNetwork(layers.Layer):
        def __init__(self, units, dropout_rate, **kwargs):
            super().__init__(**kwargs)
            self.units = units
            self.dropout_rate = dropout_rate
            self.elu_dense = layers.Dense(units, activation="elu")
            self.linear_dense = layers.Dense(units)
            self.dropout = layers.Dropout(dropout_rate)
            self.gated_linear_unit = GatedLinearUnit(units)
            self.layer_norm = layers.LayerNormalization()
            self.project = layers.Dense(units)
        def call(self, inputs):
            x = self.elu_dense(inputs)
            x = self.linear_dense(x)
            x = self.dropout(x)
            if inputs.shape[-1] != self.units:
                inputs = self.project(inputs)
            x = inputs + self.gated_linear_unit(x)
            x = self.layer_norm(x)
            return x
        def get_config(self):
            config = super().get_config()
            config.update({'units': self.units, 'dropout_rate': self.dropout_rate})
            return config

    # --- Decomposição e Preparação dos Dados ---
    data1 = new_data.loc[new_data['Month'].isin(months)].reset_index(drop=True).dropna()
    if mode == 'train':
        print(f"--- EXECUTANDO {os.path.basename(MODEL_DIR)} EM MODO DE TREINAMENTO ---")
        s = data1['LV ActivePower (kW)'].values
        emd = CEEMDAN(epsilon=0.05)
        emd.noise_seed(12345)
        IMFs = emd(s)
        ceemdan1 = pd.DataFrame(IMFs).T
        imf1 = ceemdan1.iloc[:, 0].values
        ewt, _, _ = ewtpy.EWT1D(imf1, N=3)
        df_ewt = pd.DataFrame(ewt)
        if df_ewt.shape[1] > 2:
            df_ewt.drop(df_ewt.columns[2], axis=1, inplace=True)
        denoised = df_ewt.sum(axis=1, skipna=True)
        ceemdan_without_imf1 = ceemdan1.iloc[:, 1:]
        features_df = pd.concat([pd.Series(denoised, name='imf_denoised'), ceemdan_without_imf1], axis=1)
        features_df.columns = features_df.columns.astype(str)
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(features_df, os.path.join(MODEL_DIR, "features_df.gz"))
    else:
        print(f"--- EXECUTANDO {os.path.basename(MODEL_DIR)} EM MODO DE INFERÊNCIA ---")
        features_path = os.path.join(MODEL_DIR, "features_df.gz")
        if not os.path.exists(features_path):
            raise FileNotFoundError(f"Arquivo de features '{features_path}' não encontrado.")
        features_df = joblib.load(features_path)
    
    target_df = data1[['LV ActivePower (kW)']]
    train_size = int(len(features_df) * data_partition)
    train_features_df, test_features_df = features_df.iloc[:train_size], features_df.iloc[train_size:]
    train_target_df, test_target_df = target_df.iloc[:train_size], target_df.iloc[train_size:]

    def create_dataset_multivariate(features, target, look_back, horizon):
        X, Y = [], []
        for j in range(len(features) - look_back - horizon + 1):
            X.append(features[j:j + look_back, :])
            Y.append(target[j + look_back:j + look_back + horizon, 0])
        return np.array(X), np.array(Y)

    if mode == 'train':
        scaler_X = StandardScaler().fit(train_features_df)
        scaler_Y = StandardScaler().fit(train_target_df)
        joblib.dump(scaler_X, os.path.join(MODEL_DIR, "scaler_X.gz"))
        joblib.dump(scaler_Y, os.path.join(MODEL_DIR, "scaler_Y.gz"))
    else:
        scaler_X = joblib.load(os.path.join(MODEL_DIR, "scaler_X.gz"))
        scaler_Y = joblib.load(os.path.join(MODEL_DIR, "scaler_Y.gz"))

    train_features_scaled = scaler_X.transform(train_features_df)
    test_features_scaled = scaler_X.transform(test_features_df)
    train_target_scaled = scaler_Y.transform(train_target_df)
    test_target_scaled = scaler_Y.transform(test_target_df)

    trainX, trainY = create_dataset_multivariate(train_features_scaled, train_target_scaled, look_back, horizon)
    testX, testY = create_dataset_multivariate(test_features_scaled, test_target_scaled, look_back, horizon)
    y_test_inv = scaler_Y.inverse_transform(testY)

    # ===================================================================
    # >>>>> ARQUITETURA DO MODELO TFT-BALANCED <<<<<
    # ===================================================================
    hidden_units, dropout_rate, num_heads, num_features = 32, 0.30594407849772465, 2, trainX.shape[2]
    input_shape = (look_back, num_features)
    inputs = layers.Input(shape=input_shape)

    # 1. Camada de Encoder: UM GRN para processar as features de entrada
    x = GatedResidualNetwork(hidden_units, dropout_rate)(inputs)
    
    # 2. Camada de Atenção
    attention_layer = layers.MultiHeadAttention(num_heads=num_heads, key_dim=hidden_units, dropout=dropout_rate)
    attention_output, attention_scores = attention_layer(x, x, return_attention_scores=True)
    
    # 3. Conexão Residual e Normalização
    attention_output = layers.Dropout(dropout_rate)(attention_output)
    x = layers.LayerNormalization()(x + attention_output)
    
    # 4. Camada de Decoder: UM GRN para processar o contexto da atenção
    x = GatedResidualNetwork(hidden_units, dropout_rate)(x)
    
    # 5. Camada de Saída
    x = layers.Flatten()(x)
    outputs = layers.Dense(horizon * len(quantiles))(x)
    outputs = layers.Reshape((horizon, len(quantiles)))(outputs)
    
    model = Model(inputs=inputs, outputs=outputs)
    # ===================================================================
    # FIM DA ARQUITETURA TFT-BALANCED
    # ===================================================================

    if mode == 'train':
        print("Construindo e treinando o modelo (CEEMDAN-EWT)-TFT-Balanced...")
        # Reutilizando os hiperparâmetros otimizados do TFT original
        model.compile(loss=quantile_loss, optimizer=Adam(learning_rate=0.0024005989556429123))
        early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
        model.fit(trainX, trainY, epochs=5000, batch_size=64, validation_split=0.2, callbacks=[early_stop], verbose=0)
        model.save(os.path.join(MODEL_DIR, "tft_aggregator_interp_model.keras"))
        print("Modelo TFT-Balanced salvo com sucesso.")
    
    elif mode == 'inference':
        custom_objects = {"GatedLinearUnit": GatedLinearUnit, "GatedResidualNetwork": GatedResidualNetwork, "quantile_loss": quantile_loss}
        model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "tft_aggregator_interp_model.keras"), custom_objects=custom_objects)

    # --- Previsão e Extração dos Pesos ---
    attention_model = Model(inputs=inputs, outputs=attention_scores)
    raw_predictions_scaled = model.predict(testX)
    attention_weights = attention_model.predict(testX)
    
    all_predictions = {}
    for i_q, q in enumerate(quantiles):
        preds_q_scaled = raw_predictions_scaled[:, :, i_q]
        preds_q_unscaled = scaler_Y.inverse_transform(preds_q_scaled)
        all_predictions[q] = np.clip(preds_q_unscaled, 0, cap)

    all_predictions[0.5] = np.maximum(all_predictions[0.1], all_predictions[0.5])
    all_predictions[0.9] = np.maximum(all_predictions[0.5], all_predictions[0.9])

    # --- Seção de Treinamento e Plotagem ---
    if mode == 'train':
        y_pred_median = all_predictions[0.5]
        print(f"\n===== Métricas Completas ({os.path.basename(MODEL_DIR)}) =====")
        for h in range(horizon):
            y_true_h, y_pred_h = y_test_inv[:, h], y_pred_median[:, h]
            mape = np.mean(np.abs((y_true_h - y_pred_h) / cap)) * 100 if cap > 0 else 0
            rmse = sqrt(mean_squared_error(y_true_h, y_pred_h))
            mae = mean_absolute_error(y_true_h, y_pred_h)
            r2 = r2_score(y_true_h, y_pred_h)
            print(f"Horizonte t+{(h+1)*10}: MAPE={mape:.4f}%, RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")

        operational_threshold = 0.3 * cap 
        p_low_t20 = all_predictions[0.1][:, 1]
        p_low_t30 = all_predictions[0.1][:, 2]
        ori_levels = []
        if strategy_horizon == 20:
            for i in range(len(p_low_t20)):
                if p_low_t20[i] < operational_threshold: ori_levels.append('Alto')
                elif p_low_t30[i] < operational_threshold: ori_levels.append('Atenção')
                else: ori_levels.append('Baixo')
        else:
            for p_low in p_low_t30:
                ori_levels.append('Alto' if p_low < operational_threshold else 'Baixo')

        color_map = {'Baixo': 'lightgreen', 'Atenção': 'gold', 'Alto': 'salmon'}
        num_samples_to_plot = 300
        time_axis = np.arange(min(num_samples_to_plot, len(y_test_inv)))

        for h in range(horizon):
            fig, ax = plt.subplots(figsize=(18, 8))
            for i in range(len(time_axis)):
                ax.axvspan(i, i + 1, facecolor=color_map.get(ori_levels[i], 'white'), alpha=0.5, zorder=0)
            ax.axhline(y=operational_threshold, color='red', linestyle=':', linewidth=2.5, label=f'Limiar Operacional ({operational_threshold:.2f} kW)', zorder=4)
            ax.fill_between(time_axis, all_predictions[0.1][:len(time_axis), h], all_predictions[0.9][:len(time_axis), h], color='cornflowerblue', alpha=0.6, label='Intervalo de Confiança (80%)', zorder=1)
            ax.plot(time_axis, y_test_inv[:len(time_axis), h], label="Real", color='black', linewidth=2, zorder=3)
            ax.plot(time_axis, all_predictions[0.5][:len(time_axis), h], label="Previsão Mediana", color='firebrick', linestyle='--', linewidth=2.5, zorder=2)
            ax.set_title(f'Previsão Híbrida (TFT-Balanced) com ORI - Horizonte t+{(h+1)*10} min', fontsize=16)
            ax.set_ylabel('LV ActivePower (kW)', fontsize=12)
            ax.set_xlabel('Amostras de Teste', fontsize=12)
            risk_patches = [Patch(facecolor=color, alpha=0.5, label=f'Risco {level}') for level, color in color_map.items() if level in set(ori_levels)]
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles=handles + risk_patches, loc='best', fontsize=12)
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
            ax.set_xlim(0, len(time_axis))
            ax.set_ylim(bottom=max(0, y_test_inv.min() - (0.05 * cap)))
            plt.tight_layout()
            plt.show()
            plt.close(fig)

    return all_predictions, y_test_inv, attention_weights, testX



# ====================================================================================
# NOVA FUNÇÃO COMPLETA: MODELO HÍBRIDO (CEEMDAN-EWT)-TFT AGREGADOR com VSN (v2)
# ====================================================================================
# Adicione esta nova função completa ao seu myfunctions.py

def proposed_method_tft_aggregator_vsn(new_data, months, look_back, data_partition, cap, horizon=3, strategy_horizon=20, mode='train'):
    """
    Modelo Híbrido (CEEMDAN-EWT)-TFT Agregador com Variable Selection Network (VSN).
    - Adiciona uma camada de seleção de features (VSN) antes do corpo principal do TFT.
    - O VSN aprende a ponderar a importância de cada IMF de entrada.
    - Permite a extração dos pesos das variáveis para análise de interpretabilidade.
    - Mantém todas as boas práticas: sem data leakage, modos de treino/inferência, etc.
    """
    # --- Imports ---
    import numpy as np
    import pandas as pd
    from math import sqrt
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from tensorflow.keras import layers, Model, callbacks
    from PyEMD import CEEMDAN
    from tensorflow.keras.optimizers import Adam
    import ewtpy
    import tensorflow as tf
    import random
    import os
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    import joblib

    # --- Reprodutibilidade e Configurações ---
    random.seed(1234)
    np.random.seed(1234)
    tf.random.set_seed(1234)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    # Novo diretório para o modelo com VSN
    MODEL_DIR = "saved_models/proposed_method_tft_aggregator_vsn"

    # --- Blocos de Construção do TFT (com get_config para salvamento) ---
    quantiles = [0.1, 0.5, 0.9]
    def quantile_loss(y_true, y_pred):
        y_true_expanded = tf.expand_dims(y_true, axis=-1); error = y_true_expanded - y_pred
        q_tensor = tf.constant(quantiles, dtype=tf.float32); loss = tf.maximum(q_tensor * error, (q_tensor - 1) * error)
        return tf.reduce_mean(loss)

    class GatedLinearUnit(layers.Layer):
        def __init__(self, units, **kwargs):
            super().__init__(**kwargs)
            self.units = units
            self.linear = layers.Dense(units)
            self.sigmoid = layers.Dense(units, activation="sigmoid")
        def call(self, inputs): return self.linear(inputs) * self.sigmoid(inputs)
        def get_config(self):
            config = super().get_config()
            config.update({'units': self.units})
            return config

    class GatedResidualNetwork(layers.Layer):
        def __init__(self, units, dropout_rate, **kwargs):
            super().__init__(**kwargs)
            self.units = units
            self.dropout_rate = dropout_rate
            self.elu_dense = layers.Dense(units, activation="elu")
            self.linear_dense = layers.Dense(units)
            self.dropout = layers.Dropout(dropout_rate)
            self.gated_linear_unit = GatedLinearUnit(units)
            self.layer_norm = layers.LayerNormalization()
            self.project = layers.Dense(units)
        def call(self, inputs):
            x = self.elu_dense(inputs); x = self.linear_dense(x); x = self.dropout(x)
            if inputs.shape[-1] != self.units: inputs = self.project(inputs)
            x = inputs + self.gated_linear_unit(x); x = self.layer_norm(x); return x
        def get_config(self):
            config = super().get_config()
            config.update({'units': self.units, 'dropout_rate': self.dropout_rate})
            return config

    # --- Decomposição e Preparação dos Dados (Inalterado) ---
    data1 = new_data.loc[new_data['Month'].isin(months)].reset_index(drop=True).dropna()
    
    if mode == 'train':
        print("--- EXECUTANDO (CEEMDAN-EWT)-TFT-VSN EM MODO DE TREINAMENTO ---")
        print("Iniciando decomposição do sinal (CEEMDAN + EWT)...")
        s = data1['LV ActivePower (kW)'].values
        emd = CEEMDAN(epsilon=0.05); emd.noise_seed(12345)
        IMFs = emd(s)
        ceemdan1 = pd.DataFrame(IMFs).T
        imf1 = ceemdan1.iloc[:, 0].values
        ewt, mfb, boundaries = ewtpy.EWT1D(imf1, N=3)
        df_ewt = pd.DataFrame(ewt)
        if df_ewt.shape[1] > 2: df_ewt.drop(df_ewt.columns[2], axis=1, inplace=True)
        denoised = df_ewt.sum(axis=1, skipna=True)
        ceemdan_without_imf1 = ceemdan1.iloc[:, 1:]
        features_df = pd.concat([pd.Series(denoised, name='imf_denoised'), ceemdan_without_imf1], axis=1)
        features_df.columns = features_df.columns.astype(str)
        
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(features_df, os.path.join(MODEL_DIR, "features_df.gz"))
        print("Decomposição concluída e features salvas.")
        
    elif mode == 'inference':
        print("--- EXECUTANDO (CEEMDAN-EWT)-TFT-VSN EM MODO DE INFERÊNCIA ---")
        features_path = os.path.join(MODEL_DIR, "features_df.gz")
        if not os.path.exists(features_path):
            raise FileNotFoundError(f"Arquivo de features decompostas '{features_path}' não encontrado.")
        features_df = joblib.load(features_path)
        print("Features decompostas carregadas.")
        
    else:
        raise ValueError("O parâmetro 'mode' deve ser 'train' ou 'inference'.")

    target_df = data1[['LV ActivePower (kW)']]
    train_size = int(len(features_df) * data_partition)
    train_features_df, test_features_df = features_df.iloc[:train_size], features_df.iloc[train_size:]
    train_target_df, test_target_df = target_df.iloc[:train_size], target_df.iloc[train_size:]

    def create_dataset_multivariate(features, target, look_back, horizon):
        X, Y = [], []
        for j in range(len(features) - look_back - horizon + 1):
            X.append(features[j:j + look_back, :])
            Y.append(target[j + look_back:j + look_back + horizon, 0])
        return np.array(X), np.array(Y)

    if mode == 'train':
        scaler_X = StandardScaler().fit(train_features_df)
        scaler_Y = StandardScaler().fit(train_target_df)
        joblib.dump(scaler_X, os.path.join(MODEL_DIR, "scaler_X.gz"))
        joblib.dump(scaler_Y, os.path.join(MODEL_DIR, "scaler_Y.gz"))
    else: # inference
        scaler_X = joblib.load(os.path.join(MODEL_DIR, "scaler_X.gz"))
        scaler_Y = joblib.load(os.path.join(MODEL_DIR, "scaler_Y.gz"))

    train_features_scaled = scaler_X.transform(train_features_df)
    test_features_scaled = scaler_X.transform(test_features_df)
    train_target_scaled = scaler_Y.transform(train_target_df)
    
    trainX, trainY = create_dataset_multivariate(train_features_scaled, train_target_scaled, look_back, horizon)
    testX, testY = create_dataset_multivariate(test_features_scaled, scaler_Y.transform(test_target_df), look_back, horizon)
    y_test_inv = scaler_Y.inverse_transform(testY)

    # --- Construção do Modelo com VSN ---
    if mode == 'train':
        print("Construindo e treinando o modelo (CEEMDAN-EWT)-TFT com VSN...")
        hidden_units, dropout_rate, num_heads = 64, 0.2, 8
        num_features = trainX.shape[2]
        input_shape = (look_back, num_features)
        
        inputs = layers.Input(shape=input_shape)
        
        flattened_features = layers.Flatten()(inputs)
        vsn_grn = GatedResidualNetwork(units=num_features, dropout_rate=dropout_rate, name="VSN_GRN")(flattened_features)
        variable_weights = layers.Activation("softmax", name="variable_weights")(vsn_grn)
        variable_weights_reshaped = layers.Reshape((1, num_features))(variable_weights)
        
        weighted_features = layers.Multiply(name="weighted_features")([inputs, variable_weights_reshaped])

        x = GatedResidualNetwork(hidden_units, dropout_rate, name="GRN_1")(weighted_features)
        x = GatedResidualNetwork(hidden_units, dropout_rate, name="GRN_2")(x)
        attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=hidden_units, dropout=dropout_rate, name="attention")(x, x)
        attention_output = layers.Dropout(dropout_rate)(attention_output)
        x = layers.LayerNormalization(name="add_norm_1")(x + attention_output)
        x = GatedResidualNetwork(hidden_units, dropout_rate, name="GRN_3")(x)
        x = layers.Flatten()(x)
        
        outputs = layers.Dense(horizon * len(quantiles))(x)
        outputs = layers.Reshape((horizon, len(quantiles)))(outputs)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss=quantile_loss, optimizer=Adam(learning_rate=0.001))
        
        early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
        model.fit(trainX, trainY, epochs=5000, batch_size=64, validation_split=0.2, callbacks=[early_stop], verbose=0)
        
        model.save(os.path.join(MODEL_DIR, "tft_vsn_model.keras"))
        print("Modelo TFT-VSN Agregador salvo com sucesso.")
    
    elif mode == 'inference':
        custom_objects = {"GatedLinearUnit": GatedLinearUnit, "GatedResidualNetwork": GatedResidualNetwork, "quantile_loss": quantile_loss}
        model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "tft_vsn_model.keras"), custom_objects=custom_objects)

    # --- Previsão e Pós-processamento ---
    raw_predictions_scaled = model.predict(testX)
    all_predictions = {}
    for i_q, q in enumerate(quantiles):
        preds_q_scaled = raw_predictions_scaled[:, :, i_q]
        preds_q_unscaled = scaler_Y.inverse_transform(preds_q_scaled)
        all_predictions[q] = np.clip(preds_q_unscaled, 0, cap)

    print("Aplicando correção de cruzamento de quantis...")
    all_predictions[0.5] = np.maximum(all_predictions[0.1], all_predictions[0.5])
    all_predictions[0.9] = np.maximum(all_predictions[0.5], all_predictions[0.9])

    # --- Análise de Interpretabilidade e Métricas (Apenas no modo 'train') ---
    if mode == 'train':
        print("\n--- Análise de Interpretabilidade (Pesos do VSN) ---")
        interpretability_model = Model(inputs=model.inputs, outputs=model.get_layer("variable_weights").output)
        weights = interpretability_model.predict(testX)
        mean_weights = np.mean(weights, axis=0)
        
        plt.figure(figsize=(12, 7))
        feature_names = features_df.columns
        plt.bar(feature_names, mean_weights)
        plt.title('Importância Média das Features (IMFs) - Aprendida pelo VSN', fontsize=16)
        plt.ylabel('Peso Médio (Importância)', fontsize=12)
        plt.xlabel('Componente (IMF)', fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

        y_pred_median = all_predictions[0.5]
        print("\n===== Métricas Completas (CEEMDAN-EWT-TFT-VSN) =====")
        for h in range(horizon):
            y_true_h, y_pred_h = y_test_inv[:, h], y_pred_median[:, h]
            mape = np.mean(np.abs((y_true_h - y_pred_h) / cap)) * 100 if cap > 0 else 0
            rmse = sqrt(mean_squared_error(y_true_h, y_pred_h))
            mae = mean_absolute_error(y_true_h, y_pred_h)
            r2 = r2_score(y_true_h, y_pred_h)
            print(f"Horizonte t+{(h+1)*10}: MAPE={mape:.4f}%, RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")

        # --- Plotagem Final com ORI (Apenas no modo 'train') ---
        operational_threshold = 0.3 * cap 
        p_low_t20 = all_predictions[0.1][:, 1]
        p_low_t30 = all_predictions[0.1][:, 2]
        ori_levels = []
        if strategy_horizon == 20:
            for i in range(len(p_low_t20)):
                if p_low_t20[i] < operational_threshold: ori_levels.append('Alto')
                elif p_low_t30[i] < operational_threshold: ori_levels.append('Atenção')
                else: ori_levels.append('Baixo')
        else:
            for p_low in p_low_t30:
                ori_levels.append('Alto' if p_low < operational_threshold else 'Baixo')

        color_map = {'Baixo': 'lightgreen', 'Atenção': 'gold', 'Alto': 'salmon'}
        num_samples_to_plot = 300
        time_axis = np.arange(min(num_samples_to_plot, len(y_test_inv)))

        for h in range(horizon):
            fig, ax = plt.subplots(figsize=(18, 8))
            for i in range(len(time_axis)):
                ax.axvspan(i, i + 1, facecolor=color_map.get(ori_levels[i], 'white'), alpha=0.5, zorder=0)
            ax.axhline(y=operational_threshold, color='red', linestyle=':', linewidth=2.5, label=f'Limiar Operacional ({operational_threshold:.2f} kW)', zorder=4)
            ax.fill_between(time_axis, all_predictions[0.1][:len(time_axis), h], all_predictions[0.9][:len(time_axis), h], color='cornflowerblue', alpha=0.6, label='Intervalo de Confiança (80%)', zorder=1)
            ax.plot(time_axis, y_test_inv[:len(time_axis), h], label="Real", color='black', linewidth=2, zorder=3)
            ax.plot(time_axis, all_predictions[0.5][:len(time_axis), h], label="Previsão Mediana", color='firebrick', linestyle='--', linewidth=2.5, zorder=2)
            ax.set_title(f'Previsão Híbrida (TFT-VSN) com ORI - Horizonte t+{(h+1)*10} min', fontsize=16)
            ax.set_ylabel('LV ActivePower (kW)', fontsize=12)
            ax.set_xlabel('Amostras de Teste', fontsize=12)
            risk_patches = [Patch(facecolor=color, alpha=0.5, label=f'Risco {level}') for level, color in color_map.items() if level in set(ori_levels)]
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles=handles + risk_patches, loc='best', fontsize=12)
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
            ax.set_xlim(0, len(time_axis))
            ax.set_ylim(bottom=max(0, y_test_inv.min() - (0.05 * cap)))
            plt.tight_layout()
            plt.show()
            plt.close(fig)

    return all_predictions, y_test_inv




# Em myfunctions.py

# A função principal `proposed_method_tft_aggregator` permanece inalterada.
# Esta é a nova função de otimização para o modelo proposto.

def proposed_method_tft_aggregator_with_optuna(new_data, months, look_back, data_partition, cap, horizon=3, n_trials=100):
    """
    Otimiza, treina e avalia um modelo CEEMDAN-EWT-TFT Aggregator determinístico.
    - DESCobre os hiperparâmetros ótimos para a arquitetura TFT com Optuna e TimeSeriesSplit.
    - TREINA um modelo final com os melhores parâmetros em todo o conjunto de treino.
    - AVALIA e plota os resultados no conjunto de teste.
    - NÃO salva o modelo ou os scalers no disco.
    """
    # --- Imports ---
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.optimizers import Adam
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from math import sqrt
    from PyEMD import CEEMDAN
    import ewtpy
    import optuna
    import random

    # --- Reprodutibilidade ---
    random.seed(1234)
    np.random.seed(1234)
    tf.random.set_seed(1234)

    # --- Blocos de Construção do TFT (para o modelo determinístico) ---
    class GatedLinearUnit(layers.Layer):
        def __init__(self, units, **kwargs): super().__init__(**kwargs); self.linear = layers.Dense(units); self.sigmoid = layers.Dense(units, activation="sigmoid")
        def call(self, inputs): return self.linear(inputs) * self.sigmoid(inputs)
        def get_config(self):
            config = super().get_config()
            config.update({'units': self.linear.units})
            return config

    class GatedResidualNetwork(layers.Layer):
        def __init__(self, units, dropout_rate, **kwargs):
            super().__init__(**kwargs); self.units = units; self.dropout_rate = dropout_rate; self.elu_dense = layers.Dense(units, activation="elu"); self.linear_dense = layers.Dense(units); self.dropout = layers.Dropout(dropout_rate); self.gated_linear_unit = GatedLinearUnit(units); self.layer_norm = layers.LayerNormalization(); self.project = layers.Dense(units)
        def call(self, inputs):
            x = self.elu_dense(inputs); x = self.linear_dense(x); x = self.dropout(x)
            if inputs.shape[-1] != self.units: inputs = self.project(inputs)
            x = inputs + self.gated_linear_unit(x); x = self.layer_norm(x); return x
        def get_config(self):
            config = super().get_config()
            config.update({'units': self.units, 'dropout_rate': self.dropout_rate})
            return config

    def build_tft_model(input_shape, hidden_units, dropout_rate, num_heads, horizon):
        inputs = layers.Input(shape=input_shape)
        x = GatedResidualNetwork(hidden_units, dropout_rate)(inputs)
        x = GatedResidualNetwork(hidden_units, dropout_rate)(x)
        attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=hidden_units, dropout=dropout_rate)(x, x)
        x = layers.LayerNormalization()(x + attention_output)
        x = GatedResidualNetwork(hidden_units, dropout_rate)(x)
        x = layers.Flatten()(x)
        outputs = layers.Dense(horizon)(x)
        return keras.Model(inputs=inputs, outputs=outputs)

    # --- Decomposição e Preparação dos Dados ---
    print("--- PREPARANDO DADOS PARA OTIMIZAÇÃO E TREINO DO CEEMDAN-EWT-TFT ---")
    data1 = new_data.loc[new_data['Month'].isin(months)].reset_index(drop=True).dropna()
    
    # 1. Decomposição do sinal completo
    s = data1['LV ActivePower (kW)'].values
    emd = CEEMDAN(epsilon=0.05); emd.noise_seed(12345)
    IMFs = emd(s)
    ceemdan1 = pd.DataFrame(IMFs).T
    imf1 = ceemdan1.iloc[:, 0].values
    ewt, _, _ = ewtpy.EWT1D(imf1, N=3)
    df_ewt = pd.DataFrame(ewt)
    if df_ewt.shape[1] > 2: df_ewt.drop(df_ewt.columns[2], axis=1, inplace=True)
    denoised = df_ewt.sum(axis=1, skipna=True)
    ceemdan_without_imf1 = ceemdan1.iloc[:, 1:]
    features_df = pd.concat([pd.Series(denoised, name='imf_denoised'), ceemdan_without_imf1], axis=1)
    features_df.columns = features_df.columns.astype(str)
    
    target_df = data1[['LV ActivePower (kW)']]
    
    # 2. Divisão temporal (Treino/Teste)
    train_size = int(len(features_df) * data_partition)
    train_features_df, test_features_df = features_df.iloc[:train_size], features_df.iloc[train_size:]
    train_target_df, test_target_df = target_df.iloc[:train_size], target_df.iloc[train_size:]

    def create_dataset_multivariate(features, target, look_back, horizon):
        X, Y = [], []
        for j in range(len(features) - look_back - horizon + 1):
            X.append(features[j:j + look_back, :])
            Y.append(target[j + look_back:j + look_back + horizon, 0])
        return np.array(X), np.array(Y)

    # 3. Escalonamento e criação de janelas
    scaler_X_full = StandardScaler().fit(train_features_df)
    scaler_Y_full = StandardScaler().fit(train_target_df)
    
    train_features_scaled = scaler_X_full.transform(train_features_df)
    train_target_scaled = scaler_Y_full.transform(train_target_df)
    trainX_full, trainY_full = create_dataset_multivariate(train_features_scaled, train_target_scaled, look_back, horizon)
    
    test_features_scaled = scaler_X_full.transform(test_features_df)
    test_target_scaled = scaler_Y_full.transform(test_target_df)
    testX, testY = create_dataset_multivariate(test_features_scaled, test_target_scaled, look_back, horizon)
    
    print(f"Dados preparados: {trainX_full.shape[0]} amostras de treino, {testX.shape[0]} amostras de teste.")

    # --- Função Objetivo do Optuna ---
    def objective(trial):
        hidden_units = trial.suggest_categorical('hidden_units', [32, 64, 128])
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.4)
        num_heads = trial.suggest_categorical('num_heads', [2, 4, 8])
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [32, 64])

        tscv = TimeSeriesSplit(n_splits=3)
        mape_scores = []

        for train_idx, val_idx in tscv.split(trainX_full):
            X_tr, X_val = trainX_full[train_idx], trainX_full[val_idx]
            y_tr, y_val = trainY_full[train_idx], trainY_full[val_idx]

            model = build_tft_model(
                input_shape=(look_back, X_tr.shape[2]),
                hidden_units=hidden_units, dropout_rate=dropout_rate,
                num_heads=num_heads, horizon=horizon
            )
            model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')
            early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=0)
            model.fit(X_tr, y_tr, batch_size=batch_size, epochs=5000,
                      validation_data=(X_val, y_val),
                      shuffle=False, verbose=0, callbacks=[early_stop])

            preds_scaled = model.predict(X_val, verbose=0)
            preds_inv = scaler_Y_full.inverse_transform(preds_scaled)
            y_val_inv = scaler_Y_full.inverse_transform(y_val)
            
            mape_cap = np.mean(np.abs(y_val_inv - preds_inv) / cap) * 100
            mape_scores.append(mape_cap)
            tf.keras.backend.clear_session()

        return np.mean(mape_scores)

    # --- Execução da Otimização ---
    print("\n--- INICIANDO BUSCA DE HIPERPARÂMETROS PARA CEEMDAN-EWT-TFT COM OPTUNA ---")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    print("\n" + "="*50)
    print("OTIMIZAÇÃO DO CEEMDAN-EWT-TFT CONCLUÍDA")
    print("="*50)
    print(f"Melhor resultado (MAPE médio na validação): {study.best_value:.4f}%")
    print("\nHiperparâmetros Ótimos Encontrados:")
    for key, value in study.best_params.items():
        print(f"  - {key}: {value}")
    print("="*50)

    # --- Treinamento Final com os Melhores Parâmetros ---
    print("\n--- TREINANDO MODELO CEEMDAN-EWT-TFT FINAL COM OS MELHORES PARÂMETROS ---")
    
    final_model = build_tft_model(
        input_shape=(look_back, trainX_full.shape[2]),
        hidden_units=best_params['hidden_units'],
        dropout_rate=best_params['dropout_rate'],
        num_heads=best_params['num_heads'],
        horizon=horizon
    )
    final_model.compile(optimizer=Adam(learning_rate=best_params['learning_rate']), loss='mean_squared_error')
    early_stop_final = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)
    final_model.fit(trainX_full, trainY_full,
                    batch_size=best_params['batch_size'],
                    epochs=5000,
                    validation_split=0.1,
                    shuffle=False, verbose=0, callbacks=[early_stop_final])
    print("Treinamento final concluído.")

    # --- Avaliação Final no Conjunto de Teste ---
    y_pred_test = scaler_Y_full.inverse_transform(final_model.predict(testX, verbose=0))
    y_test_inv = scaler_Y_full.inverse_transform(testY)

    metrics_data = []
    print("\n===== MÉTRICAS FINAIS NO CONJUNTO DE TESTE =====")
    for h in range(horizon):
        mape = np.mean(np.abs((y_test_inv[:, h] - y_pred_test[:, h]) / cap)) * 100
        rmse = sqrt(mean_squared_error(y_test_inv[:, h], y_pred_test[:, h]))
        mae = mean_absolute_error(y_test_inv[:, h], y_pred_test[:, h])
        r2 = r2_score(y_test_inv[:, h], y_pred_test[:, h])
        metrics_data.append({'Horizonte': f't+{(h+1)*10} min', 'MAPE (%)': mape, 'RMSE (MW)': rmse, 'MAE (MW)': mae, 'R²': r2})
        print(f"Horizonte t+{(h+1)*10}: MAPE={mape:.4f}%, RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")

    # --- Plotagem dos Resultados Finais ---
    print("\n--- GERANDO GRÁFICOS DE RESULTADOS ---")
    for h in range(horizon):
        plt.figure(figsize=(15, 6))
        plt.plot(y_test_inv[:, h], label=f"Real t+{(h+1)*10}", color="blue", alpha=0.8)
        plt.plot(y_pred_test[:, h], label=f"Previsto (Otimizado) t+{(h+1)*10}", color="red", linestyle="--")
        plt.title(f"Previsão CEEMDAN-EWT-TFT Otimizado vs. Real - Horizonte t+{(h+1)*10} min")
        plt.xlabel("Amostras de Teste")
        plt.ylabel("LV ActivePower (kW)")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

    return final_model, best_params, pd.DataFrame(metrics_data)


def proposed_method_tft_balanced_with_optuna(new_data, months, look_back, data_partition, cap, horizon=3, n_trials=60):
    """
    [VERSÃO OTIMIZADA DO TFT-BALANCED]
    Otimiza, treina e avalia um modelo CEEMDAN-EWT-TFT com a arquitetura "Balanced".
    - DESCobre os hiperparâmetros ótimos para a arquitetura (1 GRN -> Atenção -> 1 GRN) com Optuna.
    - TREINA um modelo final com os melhores parâmetros.
    - AVALIA e plota os resultados no conjunto de teste.
    - NÃO salva o modelo ou os scalers no disco (função de laboratório).
    """
    # --- Imports ---
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.optimizers import Adam
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from math import sqrt
    from PyEMD import CEEMDAN
    import ewtpy
    import optuna
    import random

    # --- Reprodutibilidade ---
    random.seed(1234)
    np.random.seed(1234)
    tf.random.set_seed(1234)

    # ===================================================================
    # >>>>> MUDANÇA PRINCIPAL: ARQUITETURA DO MODELO <<<<<
    # ===================================================================
    class GatedLinearUnit(layers.Layer):
        def __init__(self, units, **kwargs): super().__init__(**kwargs); self.linear = layers.Dense(units); self.sigmoid = layers.Dense(units, activation="sigmoid")
        def call(self, inputs): return self.linear(inputs) * self.sigmoid(inputs)
        def get_config(self):
            config = super().get_config(); config.update({'units': self.linear.units}); return config

    class GatedResidualNetwork(layers.Layer):
        def __init__(self, units, dropout_rate, **kwargs):
            super().__init__(**kwargs); self.units = units; self.dropout_rate = dropout_rate; self.elu_dense = layers.Dense(units, activation="elu"); self.linear_dense = layers.Dense(units); self.dropout = layers.Dropout(dropout_rate); self.gated_linear_unit = GatedLinearUnit(units); self.layer_norm = layers.LayerNormalization(); self.project = layers.Dense(units)
        def call(self, inputs):
            x = self.elu_dense(inputs); x = self.linear_dense(x); x = self.dropout(x)
            if inputs.shape[-1] != self.units: inputs = self.project(inputs)
            x = inputs + self.gated_linear_unit(x); x = self.layer_norm(x); return x
        def get_config(self):
            config = super().get_config(); config.update({'units': self.units, 'dropout_rate': self.dropout_rate}); return config

    def build_tft_model_balanced(input_shape, hidden_units, dropout_rate, num_heads, horizon):
        """ Constrói a arquitetura TFT-Balanced (1 GRN -> Atenção -> 1 GRN). """
        inputs = layers.Input(shape=input_shape)
        
        # 1. Camada de Encoder: UM GRN para processar as features de entrada
        x = GatedResidualNetwork(hidden_units, dropout_rate)(inputs)
        
        # 2. Camada de Atenção
        attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=hidden_units, dropout=dropout_rate)(x, x)
        
        # 3. Conexão Residual e Normalização
        x = layers.LayerNormalization()(x + attention_output)
        
        # 4. Camada de Decoder: UM GRN para processar o contexto da atenção
        x = GatedResidualNetwork(hidden_units, dropout_rate)(x)
        
        # 5. Camada de Saída
        x = layers.Flatten()(x)
        outputs = layers.Dense(horizon)(x)
        return keras.Model(inputs=inputs, outputs=outputs)
    # ===================================================================
    # FIM DA MUDANÇA
    # ===================================================================

    # --- Decomposição e Preparação dos Dados (sem alterações) ---
    print("--- PREPARANDO DADOS PARA OTIMIZAÇÃO E TREINO DO CEEMDAN-EWT-TFT-BALANCED ---")
    data1 = new_data.loc[new_data['Month'].isin(months)].reset_index(drop=True).dropna()
    
    s = data1['LV ActivePower (kW)'].values
    emd = CEEMDAN(epsilon=0.05); emd.noise_seed(12345)
    IMFs = emd(s)
    ceemdan1 = pd.DataFrame(IMFs).T
    imf1 = ceemdan1.iloc[:, 0].values
    ewt, _, _ = ewtpy.EWT1D(imf1, N=3)
    df_ewt = pd.DataFrame(ewt)
    if df_ewt.shape[1] > 2: df_ewt.drop(df_ewt.columns[2], axis=1, inplace=True)
    denoised = df_ewt.sum(axis=1, skipna=True)
    ceemdan_without_imf1 = ceemdan1.iloc[:, 1:]
    features_df = pd.concat([pd.Series(denoised, name='imf_denoised'), ceemdan_without_imf1], axis=1)
    features_df.columns = features_df.columns.astype(str)
    
    target_df = data1[['LV ActivePower (kW)']]
    
    train_size = int(len(features_df) * data_partition)
    train_features_df, test_features_df = features_df.iloc[:train_size], features_df.iloc[train_size:]
    train_target_df, test_target_df = target_df.iloc[:train_size], target_df.iloc[train_size:]

    def create_dataset_multivariate(features, target, look_back, horizon):
        X, Y = [], []
        for j in range(len(features) - look_back - horizon + 1):
            X.append(features[j:j + look_back, :]); Y.append(target[j + look_back:j + look_back + horizon, 0])
        return np.array(X), np.array(Y)

    scaler_X_full = StandardScaler().fit(train_features_df)
    scaler_Y_full = StandardScaler().fit(train_target_df)
    
    train_features_scaled = scaler_X_full.transform(train_features_df)
    train_target_scaled = scaler_Y_full.transform(train_target_df)
    trainX_full, trainY_full = create_dataset_multivariate(train_features_scaled, train_target_scaled, look_back, horizon)
    
    test_features_scaled = scaler_X_full.transform(test_features_df)
    test_target_scaled = scaler_Y_full.transform(test_target_df)
    testX, testY = create_dataset_multivariate(test_features_scaled, test_target_scaled, look_back, horizon)
    
    print(f"Dados preparados: {trainX_full.shape[0]} amostras de treino, {testX.shape[0]} amostras de teste.")

    # --- Função Objetivo do Optuna (sem alterações, exceto a chamada do build) ---
    def objective(trial):
        hidden_units = trial.suggest_categorical('hidden_units', [32, 64, 128])
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.4)
        num_heads = trial.suggest_categorical('num_heads', [2, 4, 8])
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [32, 64])

        tscv = TimeSeriesSplit(n_splits=3)
        mape_scores = []

        for train_idx, val_idx in tscv.split(trainX_full):
            X_tr, X_val = trainX_full[train_idx], trainX_full[val_idx]
            y_tr, y_val = trainY_full[train_idx], trainY_full[val_idx]

            # >>>>> USA A NOVA FUNÇÃO DE BUILD <<<<<
            model = build_tft_model_balanced(
                input_shape=(look_back, X_tr.shape[2]),
                hidden_units=hidden_units, dropout_rate=dropout_rate,
                num_heads=num_heads, horizon=horizon
            )
            model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')
            early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=0)
            model.fit(X_tr, y_tr, batch_size=batch_size, epochs=5000,
                      validation_data=(X_val, y_val),
                      shuffle=False, verbose=0, callbacks=[early_stop])

            preds_scaled = model.predict(X_val, verbose=0)
            preds_inv = scaler_Y_full.inverse_transform(preds_scaled)
            y_val_inv = scaler_Y_full.inverse_transform(y_val)
            
            mape_cap = np.mean(np.abs(y_val_inv - preds_inv) / cap) * 100
            mape_scores.append(mape_cap)
            tf.keras.backend.clear_session()

        return np.mean(mape_scores)

    # --- Execução da Otimização (sem alterações) ---
    print("\n--- INICIANDO BUSCA DE HIPERPARÂMETROS PARA CEEMDAN-EWT-TFT-BALANCED COM OPTUNA ---")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    print("\n" + "="*50)
    print("OTIMIZAÇÃO DO CEEMDAN-EWT-TFT-BALANCED CONCLUÍDA")
    print("="*50)
    print(f"Melhor resultado (MAPE médio na validação): {study.best_value:.4f}%")
    print("\nHiperparâmetros Ótimos Encontrados:")
    for key, value in study.best_params.items():
        print(f"  - {key}: {value}")
    print("="*50)

    # --- Treinamento Final com os Melhores Parâmetros (sem alterações, exceto a chamada do build) ---
    print("\n--- TREINANDO MODELO CEEMDAN-EWT-TFT-BALANCED FINAL COM OS MELHORES PARÂMETROS ---")
    
    # >>>>> USA A NOVA FUNÇÃO DE BUILD <<<<<
    final_model = build_tft_model_balanced(
        input_shape=(look_back, trainX_full.shape[2]),
        hidden_units=best_params['hidden_units'],
        dropout_rate=best_params['dropout_rate'],
        num_heads=best_params['num_heads'],
        horizon=horizon
    )
    final_model.compile(optimizer=Adam(learning_rate=best_params['learning_rate']), loss='mean_squared_error')
    early_stop_final = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)
    final_model.fit(trainX_full, trainY_full,
                    batch_size=best_params['batch_size'],
                    epochs=5000,
                    validation_split=0.1,
                    shuffle=False, verbose=0, callbacks=[early_stop_final])
    print("Treinamento final concluído.")

    # --- Avaliação Final e Plotagem (sem alterações) ---
    y_pred_test = scaler_Y_full.inverse_transform(final_model.predict(testX, verbose=0))
    y_test_inv = scaler_Y_full.transform(testY)

    metrics_data = []
    print("\n===== MÉTRICAS FINAIS NO CONJUNTO DE TESTE =====")
    for h in range(horizon):
        mape = np.mean(np.abs((y_test_inv[:, h] - y_pred_test[:, h]) / cap)) * 100
        rmse = sqrt(mean_squared_error(y_test_inv[:, h], y_pred_test[:, h]))
        mae = mean_absolute_error(y_test_inv[:, h], y_pred_test[:, h])
        r2 = r2_score(y_test_inv[:, h], y_pred_test[:, h])
        metrics_data.append({'Horizonte': f't+{(h+1)*10} min', 'MAPE (%)': mape, 'RMSE (MW)': rmse, 'MAE (MW)': mae, 'R²': r2})
        print(f"Horizonte t+{(h+1)*10}: MAPE={mape:.4f}%, RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")

    print("\n--- GERANDO GRÁFICOS DE RESULTADOS ---")
    for h in range(horizon):
        plt.figure(figsize=(15, 6))
        plt.plot(y_test_inv[:, h], label=f"Real t+{(h+1)*10}", color="blue", alpha=0.8)
        plt.plot(y_pred_test[:, h], label=f"Previsto (Otimizado) t+{(h+1)*10}", color="red", linestyle="--")
        plt.title(f"Previsão CEEMDAN-EWT-TFT-Balanced Otimizado vs. Real - Horizonte t+{(h+1)*10} min")
        plt.xlabel("Amostras de Teste")
        plt.ylabel("LV ActivePower (kW)")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

    return final_model, best_params, pd.DataFrame(metrics_data)





#---#
# Adicione esta nova função ao seu myfunctions.py

def proposed_method_bilstm(new_data, months, look_back, data_partition, cap, horizon=3, strategy_horizon=20, mode='train'):
    """
    Modelo híbrido CEEMDAN-EWT-BiLSTM (v1), baseado no padrão do proposed_method.
    - Utiliza uma camada Bidirectional(LSTM) em vez de LSTM simples.
    - Herda todas as boas práticas: sem data leakage na normalização, modos de treino/inferência, etc.
    """
    # --- Imports (sem alterações) ---
    import numpy as np
    import pandas as pd
    from math import sqrt
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    # >>>>> MUDANÇA 1: Importar a camada Bidirectional <<<<<
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.optimizers import Adam
    from PyEMD import CEEMDAN
    import ewtpy
    import tensorflow as tf
    import random
    import os
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    import joblib
    import gc

    # --- Reprodutibilidade e Configurações ---
    random.seed(1234)
    np.random.seed(1234)
    tf.random.set_seed(1234)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    # >>>>> MUDANÇA 2: Novo diretório para salvar os modelos <<<<<
    MODEL_DIR = "saved_models/proposed_method_bilstm"

    # --- Funções Auxiliares (sem alterações) ---
    def create_dataset(dataset, look_back, horizon):
        dataX, dataY = [], []
        for j in range(len(dataset) - look_back - horizon + 1):
            X = dataset[j:(j + look_back), 0]
            Y = dataset[(j + look_back):(j + look_back + horizon), 0]
            dataX.append(X)
            dataY.append(Y)
        return np.array(dataX), np.array(dataY)

    # --- Decomposição e Preparação (sem alterações) ---
    data1 = new_data.loc[new_data['Month'].isin(months)].reset_index(drop=True).dropna()
    
    if mode == 'train':
        print("--- EXECUTANDO proposed_method_bilstm EM MODO DE TREINAMENTO ---")
        print("Iniciando decomposição do sinal (CEEMDAN + EWT)...")
        s = data1['LV ActivePower (kW)'].values
        emd = CEEMDAN(epsilon=0.05); emd.noise_seed(12345)
        IMFs = emd(s)
        ceemdan1 = pd.DataFrame(IMFs).T
        imf1 = ceemdan1.iloc[:, 0].values
        ewt, mfb, boundaries = ewtpy.EWT1D(imf1, N=3)
        df_ewt = pd.DataFrame(ewt)
        if df_ewt.shape[1] > 2: df_ewt.drop(df_ewt.columns[2], axis=1, inplace=True)
        denoised = df_ewt.sum(axis=1)
        ceemdan_without_imf1 = ceemdan1.iloc[:, 1:]
        ceemdan_without_imf1.columns = [f"imf_{c}" for c in ceemdan_without_imf1.columns]
        denoised.name = "imf_denoised"
        decomposed_df = pd.concat([denoised, ceemdan_without_imf1], axis=1)
        
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(decomposed_df, os.path.join(MODEL_DIR, "decomposed_df.gz"))
        print("Decomposição concluída e salva.")
        
    elif mode == 'inference':
        print("--- EXECUTANDO proposed_method_bilstm EM MODO DE INFERÊNCIA ---")
        decomposed_path = os.path.join(MODEL_DIR, "decomposed_df.gz")
        if not os.path.exists(decomposed_path):
            raise FileNotFoundError(f"Arquivo de decomposição '{decomposed_path}' não encontrado.")
        decomposed_df = joblib.load(decomposed_path)
        print("Decomposição carregada.")
        
    else:
        raise ValueError("O parâmetro 'mode' deve ser 'train' ou 'inference'.")

    component_names = decomposed_df.columns.tolist()
    quantiles = [0.1, 0.5, 0.9]
    all_predictions = {}

    for q in quantiles:
        imf_predictions_for_quantile_q = []
        quantile_dir = os.path.join(MODEL_DIR, f"q{str(q).replace('.', '')}")

        for col_name in component_names:
            tf.keras.backend.clear_session()
            datasets = decomposed_df[[col_name]].values
            train_size = int(len(datasets) * data_partition)
            train, test = datasets[:train_size], datasets[train_size:]
            trainX, trainY = create_dataset(train, look_back, horizon)
            testX, _ = create_dataset(test, look_back, horizon)

            if trainX.shape[0] == 0: continue

            if mode == 'train':
                print(f"--- Q{q} | Treinando BiLSTM para: {col_name} ---")
                
                sc_X, sc_y = StandardScaler(), StandardScaler()
                sc_X.fit(trainX)
                sc_y.fit(trainY)

                X_train = sc_X.transform(trainX)
                y_train = sc_y.transform(trainY)
                X_test = sc_X.transform(testX)

                X_train = X_train.reshape((X_train.shape[0], look_back, 1))
                X_test = X_test.reshape((X_test.shape[0], look_back, 1))

                # >>>>> MUDANÇA 3: Substituir LSTM por Bidirectional(LSTM(...)) <<<<<
                model = Sequential([
                    Bidirectional(LSTM(units=32, input_shape=(look_back, 1))),
                    Dropout(0.2),
                    Dense(horizon)
                ])
                model.compile(loss=loss_functions[q], optimizer=Adam(learning_rate=0.001))
                early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
                model.fit(X_train, y_train, epochs=5000, batch_size=64, verbose=0, shuffle=False, validation_split=0.1, callbacks=[early_stop])

                y_pred_scaled = model.predict(X_test, verbose=0)
                y_pred_inv = sc_y.inverse_transform(y_pred_scaled)
                
                os.makedirs(quantile_dir, exist_ok=True)
                model.save(os.path.join(quantile_dir, f"model_{col_name}.keras"))
                joblib.dump(sc_X, os.path.join(quantile_dir, f"scaler_X_{col_name}.gz"))
                joblib.dump(sc_y, os.path.join(quantile_dir, f"scaler_y_{col_name}.gz"))

            elif mode == 'inference':
                sc_X = joblib.load(os.path.join(quantile_dir, f"scaler_X_{col_name}.gz"))
                sc_y = joblib.load(os.path.join(quantile_dir, f"scaler_y_{col_name}.gz"))
                custom_objects = {fn.__name__: fn for fn in loss_functions.values()}
                model = tf.keras.models.load_model(os.path.join(quantile_dir, f"model_{col_name}.keras"), custom_objects=custom_objects)

                X_test = sc_X.transform(testX)
                X_test = X_test.reshape((X_test.shape[0], look_back, 1))
                y_pred_scaled = model.predict(X_test, verbose=0)
                y_pred_inv = sc_y.inverse_transform(y_pred_scaled)
            
            imf_predictions_for_quantile_q.append(y_pred_inv)
            gc.collect()

        reconstructed_prediction = np.sum(np.array(imf_predictions_for_quantile_q), axis=0)
        all_predictions[q] = np.clip(reconstructed_prediction, 0, cap)

    # Preparação dos dados reais para validação (sem alterações)
    original_values = data1[['LV ActivePower (kW)']].values
    train_size_orig = int(len(original_values) * data_partition)
    _, test_orig = original_values[:train_size_orig], original_values[train_size_orig:]
    _, y_test_inv = create_dataset(test_orig, look_back, horizon)

    # Correção de cruzamento de quantis (sem alterações)
    print("Aplicando correção de cruzamento de quantis...")
    all_predictions[0.5] = np.maximum(all_predictions[0.1], all_predictions[0.5])
    all_predictions[0.9] = np.maximum(all_predictions[0.5], all_predictions[0.9])

    if mode == 'train':
        # Seção de métricas e gráficos (sem alterações, apenas o título)
        y_pred_median = all_predictions[0.5]
        print("\n===== Métricas Completas (CEEMDAN-EWT-BiLSTM) =====")
        for h in range(horizon):
            y_true_h, y_pred_h = y_test_inv[:, h], y_pred_median[:, h]
            mape = np.mean(np.abs((y_true_h - y_pred_h) / cap)) * 100 if cap > 0 else 0
            rmse = sqrt(mean_squared_error(y_true_h, y_pred_h))
            mae = mean_absolute_error(y_true_h, y_pred_h)
            r2 = r2_score(y_true_h, y_pred_h)
            print(f"Horizonte t+{(h+1)*10}: MAPE={mape:.4f}%, RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")

        # Lógica de Risco ORI e plotagem
        operational_threshold = 0.3 * cap 
        p_low_t20 = all_predictions[0.1][:, 1]
        p_low_t30 = all_predictions[0.1][:, 2]
        ori_levels = []
        if strategy_horizon == 20:
            for i in range(len(p_low_t20)):
                if p_low_t20[i] < operational_threshold: ori_levels.append('Alto')
                elif p_low_t30[i] < operational_threshold: ori_levels.append('Atenção')
                else: ori_levels.append('Baixo')
        else:
            for p_low in p_low_t30:
                ori_levels.append('Alto' if p_low < operational_threshold else 'Baixo')

        color_map = {'Baixo': 'lightgreen', 'Atenção': 'gold', 'Alto': 'salmon'}
        num_samples_to_plot = 300
        time_axis = np.arange(min(num_samples_to_plot, len(y_test_inv)))

        for h in range(horizon):
            fig, ax = plt.subplots(figsize=(18, 8))
            for i in range(len(time_axis)):
                ax.axvspan(i, i + 1, facecolor=color_map.get(ori_levels[i], 'white'), alpha=0.5, zorder=0)
            ax.axhline(y=operational_threshold, color='red', linestyle=':', linewidth=2.5, label=f'Limiar Operacional ({operational_threshold:.2f} kW)', zorder=4)
            ax.fill_between(time_axis, all_predictions[0.1][:len(time_axis), h], all_predictions[0.9][:len(time_axis), h], color='cornflowerblue', alpha=0.6, label='Intervalo de Confiança (80%)', zorder=1)
            ax.plot(time_axis, y_test_inv[:len(time_axis), h], label="Real", color='black', linewidth=2, zorder=3)
            ax.plot(time_axis, all_predictions[0.5][:len(time_axis), h], label="Previsão Mediana", color='firebrick', linestyle='--', linewidth=2.5, zorder=2)
            # >>>>> MUDANÇA 4: Título do gráfico atualizado <<<<<
            ax.set_title(f'Previsão Híbrida (CEEMDAN-EWT-BiLSTM) com ORI - Horizonte t+{(h+1)*10} min', fontsize=16)
            ax.set_ylabel('LV ActivePower (kW)', fontsize=12)
            ax.set_xlabel('Amostras de Teste', fontsize=12)
            risk_patches = [Patch(facecolor=color, alpha=0.5, label=f'Risco {level}') for level, color in color_map.items() if level in set(ori_levels)]
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles=handles + risk_patches, loc='best', fontsize=12)
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
            ax.set_xlim(0, len(time_axis))
            ax.set_ylim(bottom=max(0, y_test_inv.min() - (0.05 * cap)))
            plt.tight_layout()
            plt.show()
            plt.close(fig)

    return all_predictions, y_test_inv



#--------#

# ---------- Hybrid CEEMDAN-EWT BiLSTM + Attention (Multi-horizonte) ----------
# Adicione esta nova função ao seu myfunctions.py

# Substitua a função existente em myfunctions.py por esta versão corrigida

def proposed_method_bilstm_att(new_data, months, look_back, data_partition, cap, horizon=3, strategy_horizon=20, mode='train'):
    """
    Modelo híbrido CEEMDAN-EWT-BiLSTM-Attention (v2 - CORRIGIDO), baseado no padrão do proposed_method.
    - Utiliza a API Funcional para implementar corretamente a camada de Atenção.
    - Herda todas as boas práticas: sem data leakage, modos de treino/inferência, etc.
    """
    # --- Imports ---
    import numpy as np
    import pandas as pd
    from math import sqrt
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    # >>>>> MUDANÇA 1: Importar Input e Model para a API Funcional <<<<<
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Attention, Input
    from tensorflow.keras.models import Model
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.optimizers import Adam
    from PyEMD import CEEMDAN
    import ewtpy
    import tensorflow as tf
    import random
    import os
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    import joblib
    import gc

    # --- Reprodutibilidade e Configurações ---
    random.seed(1234)
    np.random.seed(1234)
    tf.random.set_seed(1234)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    MODEL_DIR = "saved_models/proposed_method_bilstm_att"

    # --- Funções Auxiliares (sem alterações) ---
    def create_dataset(dataset, look_back, horizon):
        dataX, dataY = [], []
        for j in range(len(dataset) - look_back - horizon + 1):
            X = dataset[j:(j + look_back), 0]
            Y = dataset[(j + look_back):(j + look_back + horizon), 0]
            dataX.append(X)
            dataY.append(Y)
        return np.array(dataX), np.array(dataY)

    # --- Decomposição e Preparação (sem alterações) ---
    data1 = new_data.loc[new_data['Month'].isin(months)].reset_index(drop=True).dropna()
    
    if mode == 'train':
        print("--- EXECUTANDO proposed_method_bilstm_att EM MODO DE TREINAMENTO ---")
        print("Iniciando decomposição do sinal (CEEMDAN + EWT)...")
        s = data1['LV ActivePower (kW)'].values
        emd = CEEMDAN(epsilon=0.05); emd.noise_seed(12345)
        IMFs = emd(s)
        ceemdan1 = pd.DataFrame(IMFs).T
        imf1 = ceemdan1.iloc[:, 0].values
        ewt, mfb, boundaries = ewtpy.EWT1D(imf1, N=3)
        df_ewt = pd.DataFrame(ewt)
        if df_ewt.shape[1] > 2: df_ewt.drop(df_ewt.columns[2], axis=1, inplace=True)
        denoised = df_ewt.sum(axis=1)
        ceemdan_without_imf1 = ceemdan1.iloc[:, 1:]
        ceemdan_without_imf1.columns = [f"imf_{c}" for c in ceemdan_without_imf1.columns]
        denoised.name = "imf_denoised"
        decomposed_df = pd.concat([denoised, ceemdan_without_imf1], axis=1)
        
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(decomposed_df, os.path.join(MODEL_DIR, "decomposed_df.gz"))
        print("Decomposição concluída e salva.")
        
    elif mode == 'inference':
        print("--- EXECUTANDO proposed_method_bilstm_att EM MODO DE INFERÊNCIA ---")
        decomposed_path = os.path.join(MODEL_DIR, "decomposed_df.gz")
        if not os.path.exists(decomposed_path):
            raise FileNotFoundError(f"Arquivo de decomposição '{decomposed_path}' não encontrado.")
        decomposed_df = joblib.load(decomposed_path)
        print("Decomposição carregada.")
        
    else:
        raise ValueError("O parâmetro 'mode' deve ser 'train' ou 'inference'.")

    component_names = decomposed_df.columns.tolist()
    quantiles = [0.1, 0.5, 0.9]
    all_predictions = {}

    for q in quantiles:
        imf_predictions_for_quantile_q = []
        quantile_dir = os.path.join(MODEL_DIR, f"q{str(q).replace('.', '')}")

        for col_name in component_names:
            tf.keras.backend.clear_session()
            datasets = decomposed_df[[col_name]].values
            train_size = int(len(datasets) * data_partition)
            train, test = datasets[:train_size], datasets[train_size:]
            trainX, trainY = create_dataset(train, look_back, horizon)
            testX, _ = create_dataset(test, look_back, horizon)

            if trainX.shape[0] == 0: continue

            if mode == 'train':
                print(f"--- Q{q} | Treinando BiLSTM+Attn para: {col_name} ---")
                
                sc_X, sc_y = StandardScaler(), StandardScaler()
                sc_X.fit(trainX)
                sc_y.fit(trainY)

                X_train = sc_X.transform(trainX)
                y_train = sc_y.transform(trainY)
                X_test = sc_X.transform(testX)

                X_train = X_train.reshape((X_train.shape[0], look_back, 1))
                X_test = X_test.reshape((X_test.shape[0], look_back, 1))

                # >>>>> MUDANÇA 2: Usar a API Funcional para a arquitetura correta <<<<<
                input_layer = Input(shape=(look_back, 1))
                lstm_out = Bidirectional(LSTM(units=32, return_sequences=True))(input_layer)
                attention_out = Attention()([lstm_out, lstm_out])
                lstm_agg = LSTM(units=32, return_sequences=False)(attention_out)
                drop = Dropout(0.2)(lstm_agg)
                output_layer = Dense(horizon)(drop)
                model = Model(inputs=input_layer, outputs=output_layer)
                
                model.compile(loss=loss_functions[q], optimizer=Adam(learning_rate=0.001))
                early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
                model.fit(X_train, y_train, epochs=5000, batch_size=64, verbose=0, shuffle=False, validation_split=0.1, callbacks=[early_stop])

                y_pred_scaled = model.predict(X_test, verbose=0)
                y_pred_inv = sc_y.inverse_transform(y_pred_scaled)
                
                os.makedirs(quantile_dir, exist_ok=True)
                model.save(os.path.join(quantile_dir, f"model_{col_name}.keras"))
                joblib.dump(sc_X, os.path.join(quantile_dir, f"scaler_X_{col_name}.gz"))
                joblib.dump(sc_y, os.path.join(quantile_dir, f"scaler_y_{col_name}.gz"))

            elif mode == 'inference':
                sc_X = joblib.load(os.path.join(quantile_dir, f"scaler_X_{col_name}.gz"))
                sc_y = joblib.load(os.path.join(quantile_dir, f"scaler_y_{col_name}.gz"))
                custom_objects = {fn.__name__: fn for fn in loss_functions.values()}
                model = tf.keras.models.load_model(os.path.join(quantile_dir, f"model_{col_name}.keras"), custom_objects=custom_objects)

                X_test = sc_X.transform(testX)
                X_test = X_test.reshape((X_test.shape[0], look_back, 1))
                y_pred_scaled = model.predict(X_test, verbose=0)
                y_pred_inv = sc_y.inverse_transform(y_pred_scaled)
            
            imf_predictions_for_quantile_q.append(y_pred_inv)
            gc.collect()

        reconstructed_prediction = np.sum(np.array(imf_predictions_for_quantile_q), axis=0)
        all_predictions[q] = np.clip(reconstructed_prediction, 0, cap)

    # Preparação dos dados reais para validação (sem alterações)
    original_values = data1[['LV ActivePower (kW)']].values
    train_size_orig = int(len(original_values) * data_partition)
    _, test_orig = original_values[:train_size_orig], original_values[train_size_orig:]
    _, y_test_inv = create_dataset(test_orig, look_back, horizon)

    # Correção de cruzamento de quantis (sem alterações)
    print("Aplicando correção de cruzamento de quantis...")
    all_predictions[0.5] = np.maximum(all_predictions[0.1], all_predictions[0.5])
    all_predictions[0.9] = np.maximum(all_predictions[0.5], all_predictions[0.9])

    if mode == 'train':
        # Seção de métricas e gráficos (sem alterações)
        y_pred_median = all_predictions[0.5]
        print("\n===== Métricas Completas (CEEMDAN-EWT-BiLSTM-Attn) =====")
        for h in range(horizon):
            y_true_h, y_pred_h = y_test_inv[:, h], y_pred_median[:, h]
            mape = np.mean(np.abs((y_true_h - y_pred_h) / cap)) * 100 if cap > 0 else 0
            rmse = sqrt(mean_squared_error(y_true_h, y_pred_h))
            mae = mean_absolute_error(y_true_h, y_pred_h)
            r2 = r2_score(y_true_h, y_pred_h)
            print(f"Horizonte t+{(h+1)*10}: MAPE={mape:.4f}%, RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")

        # Lógica de Risco ORI e plotagem
        operational_threshold = 0.3 * cap 
        p_low_t20 = all_predictions[0.1][:, 1]
        p_low_t30 = all_predictions[0.1][:, 2]
        ori_levels = []
        if strategy_horizon == 20:
            for i in range(len(p_low_t20)):
                if p_low_t20[i] < operational_threshold: ori_levels.append('Alto')
                elif p_low_t30[i] < operational_threshold: ori_levels.append('Atenção')
                else: ori_levels.append('Baixo')
        else:
            for p_low in p_low_t30:
                ori_levels.append('Alto' if p_low < operational_threshold else 'Baixo')

        color_map = {'Baixo': 'lightgreen', 'Atenção': 'gold', 'Alto': 'salmon'}
        num_samples_to_plot = 300
        time_axis = np.arange(min(num_samples_to_plot, len(y_test_inv)))

        for h in range(horizon):
            fig, ax = plt.subplots(figsize=(18, 8))
            for i in range(len(time_axis)):
                ax.axvspan(i, i + 1, facecolor=color_map.get(ori_levels[i], 'white'), alpha=0.5, zorder=0)
            ax.axhline(y=operational_threshold, color='red', linestyle=':', linewidth=2.5, label=f'Limiar Operacional ({operational_threshold:.2f} kW)', zorder=4)
            ax.fill_between(time_axis, all_predictions[0.1][:len(time_axis), h], all_predictions[0.9][:len(time_axis), h], color='cornflowerblue', alpha=0.6, label='Intervalo de Confiança (80%)', zorder=1)
            ax.plot(time_axis, y_test_inv[:len(time_axis), h], label="Real", color='black', linewidth=2, zorder=3)
            ax.plot(time_axis, all_predictions[0.5][:len(time_axis), h], label="Previsão Mediana", color='firebrick', linestyle='--', linewidth=2.5, zorder=2)
            ax.set_title(f'Previsão Híbrida (CEEMDAN-EWT-BiLSTM-Attn) com ORI - Horizonte t+{(h+1)*10} min', fontsize=16)
            ax.set_ylabel('LV ActivePower (kW)', fontsize=12)
            ax.set_xlabel('Amostras de Teste', fontsize=12)
            risk_patches = [Patch(facecolor=color, alpha=0.5, label=f'Risco {level}') for level, color in color_map.items() if level in set(ori_levels)]
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles=handles + risk_patches, loc='best', fontsize=12)
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
            ax.set_xlim(0, len(time_axis))
            ax.set_ylim(bottom=max(0, y_test_inv.min() - (0.05 * cap)))
            plt.tight_layout()
            plt.show()
            plt.close(fig)

    return all_predictions, y_test_inv





#---#
# Adicione esta nova função ao seu myfunctions.py

# Substitua a função existente em myfunctions.py por esta versão corrigida

def proposed_method_cnn_bilstm_att(new_data, months, look_back, data_partition, cap, horizon=3, strategy_horizon=20, mode='train'):
    """
    Modelo híbrido CEEMDAN-EWT + CNN-BiLSTM-Attention (v2 - CORRIGIDO), baseado no padrão do proposed_method.
    - Utiliza a API Funcional para implementar corretamente a camada de Atenção após a CNN e o BiLSTM.
    - Herda todas as boas práticas: sem data leakage, modos de treino/inferência, etc.
    """
    # --- Imports ---
    import numpy as np
    import pandas as pd
    from math import sqrt
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    # >>>>> MUDANÇA 1: Importar Input e Model para a API Funcional <<<<<
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Conv1D, Attention, Input
    from tensorflow.keras.models import Model
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.optimizers import Adam
    from PyEMD import CEEMDAN
    import ewtpy
    import tensorflow as tf
    import random
    import os
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    import joblib
    import gc

    # --- Reprodutibilidade e Configurações ---
    random.seed(1234)
    np.random.seed(1234)
    tf.random.set_seed(1234)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    MODEL_DIR = "saved_models/proposed_method_cnn_bilstm_att"

    # --- Funções Auxiliares (sem alterações) ---
    def create_dataset(dataset, look_back, horizon):
        dataX, dataY = [], []
        for j in range(len(dataset) - look_back - horizon + 1):
            X = dataset[j:(j + look_back), 0]
            Y = dataset[(j + look_back):(j + look_back + horizon), 0]
            dataX.append(X)
            dataY.append(Y)
        return np.array(dataX), np.array(dataY)

    # --- Decomposição e Preparação (sem alterações) ---
    data1 = new_data.loc[new_data['Month'].isin(months)].reset_index(drop=True).dropna()
    
    if mode == 'train':
        print("--- EXECUTANDO proposed_method_cnn_bilstm_att EM MODO DE TREINAMENTO ---")
        print("Iniciando decomposição do sinal (CEEMDAN + EWT)...")
        s = data1['LV ActivePower (kW)'].values
        emd = CEEMDAN(epsilon=0.05); emd.noise_seed(12345)
        IMFs = emd(s)
        ceemdan1 = pd.DataFrame(IMFs).T
        imf1 = ceemdan1.iloc[:, 0].values
        ewt, mfb, boundaries = ewtpy.EWT1D(imf1, N=3)
        df_ewt = pd.DataFrame(ewt)
        if df_ewt.shape[1] > 2: df_ewt.drop(df_ewt.columns[2], axis=1, inplace=True)
        denoised = df_ewt.sum(axis=1)
        ceemdan_without_imf1 = ceemdan1.iloc[:, 1:]
        ceemdan_without_imf1.columns = [f"imf_{c}" for c in ceemdan_without_imf1.columns]
        denoised.name = "imf_denoised"
        decomposed_df = pd.concat([denoised, ceemdan_without_imf1], axis=1)
        
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(decomposed_df, os.path.join(MODEL_DIR, "decomposed_df.gz"))
        print("Decomposição concluída e salva.")
        
    elif mode == 'inference':
        print("--- EXECUTANDO proposed_method_cnn_bilstm_att EM MODO DE INFERÊNCIA ---")
        decomposed_path = os.path.join(MODEL_DIR, "decomposed_df.gz")
        if not os.path.exists(decomposed_path):
            raise FileNotFoundError(f"Arquivo de decomposição '{decomposed_path}' não encontrado.")
        decomposed_df = joblib.load(decomposed_path)
        print("Decomposição carregada.")
        
    else:
        raise ValueError("O parâmetro 'mode' deve ser 'train' ou 'inference'.")

    component_names = decomposed_df.columns.tolist()
    quantiles = [0.1, 0.5, 0.9]
    all_predictions = {}

    for q in quantiles:
        imf_predictions_for_quantile_q = []
        quantile_dir = os.path.join(MODEL_DIR, f"q{str(q).replace('.', '')}")

        for col_name in component_names:
            tf.keras.backend.clear_session()
            datasets = decomposed_df[[col_name]].values
            train_size = int(len(datasets) * data_partition)
            train, test = datasets[:train_size], datasets[train_size:]
            trainX, trainY = create_dataset(train, look_back, horizon)
            testX, _ = create_dataset(test, look_back, horizon)

            if trainX.shape[0] == 0: continue

            if mode == 'train':
                print(f"--- Q{q} | Treinando CNN-BiLSTM-Attn para: {col_name} ---")
                
                sc_X, sc_y = StandardScaler(), StandardScaler()
                sc_X.fit(trainX)
                sc_y.fit(trainY)

                X_train = sc_X.transform(trainX)
                y_train = sc_y.transform(trainY)
                X_test = sc_X.transform(testX)

                X_train = X_train.reshape((X_train.shape[0], look_back, 1))
                X_test = X_test.reshape((X_test.shape[0], look_back, 1))

                # >>>>> MUDANÇA 2: Usar a API Funcional para a arquitetura correta <<<<<
                input_layer = Input(shape=(look_back, 1))
                conv_out = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(input_layer)
                lstm_out = Bidirectional(LSTM(units=32, return_sequences=True))(conv_out)
                attention_out = Attention()([lstm_out, lstm_out])
                lstm_agg = LSTM(units=32, return_sequences=False)(attention_out)
                drop = Dropout(0.2)(lstm_agg)
                output_layer = Dense(horizon)(drop)
                model = Model(inputs=input_layer, outputs=output_layer)
                
                model.compile(loss=loss_functions[q], optimizer=Adam(learning_rate=0.001))
                early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
                model.fit(X_train, y_train, epochs=5000, batch_size=64, verbose=0, shuffle=False, validation_split=0.1, callbacks=[early_stop])

                y_pred_scaled = model.predict(X_test, verbose=0)
                y_pred_inv = sc_y.inverse_transform(y_pred_scaled)
                
                os.makedirs(quantile_dir, exist_ok=True)
                model.save(os.path.join(quantile_dir, f"model_{col_name}.keras"))
                joblib.dump(sc_X, os.path.join(quantile_dir, f"scaler_X_{col_name}.gz"))
                joblib.dump(sc_y, os.path.join(quantile_dir, f"scaler_y_{col_name}.gz"))

            elif mode == 'inference':
                sc_X = joblib.load(os.path.join(quantile_dir, f"scaler_X_{col_name}.gz"))
                sc_y = joblib.load(os.path.join(quantile_dir, f"scaler_y_{col_name}.gz"))
                custom_objects = {fn.__name__: fn for fn in loss_functions.values()}
                model = tf.keras.models.load_model(os.path.join(quantile_dir, f"model_{col_name}.keras"), custom_objects=custom_objects)

                X_test = sc_X.transform(testX)
                X_test = X_test.reshape((X_test.shape[0], look_back, 1))
                y_pred_scaled = model.predict(X_test, verbose=0)
                y_pred_inv = sc_y.inverse_transform(y_pred_scaled)
            
            imf_predictions_for_quantile_q.append(y_pred_inv)
            gc.collect()

        reconstructed_prediction = np.sum(np.array(imf_predictions_for_quantile_q), axis=0)
        all_predictions[q] = np.clip(reconstructed_prediction, 0, cap)

    # Preparação dos dados reais para validação (sem alterações)
    original_values = data1[['LV ActivePower (kW)']].values
    train_size_orig = int(len(original_values) * data_partition)
    _, test_orig = original_values[:train_size_orig], original_values[train_size_orig:]
    _, y_test_inv = create_dataset(test_orig, look_back, horizon)

    # Correção de cruzamento de quantis (sem alterações)
    print("Aplicando correção de cruzamento de quantis...")
    all_predictions[0.5] = np.maximum(all_predictions[0.1], all_predictions[0.5])
    all_predictions[0.9] = np.maximum(all_predictions[0.5], all_predictions[0.9])

    if mode == 'train':
        # Seção de métricas e gráficos (sem alterações)
        y_pred_median = all_predictions[0.5]
        print("\n===== Métricas Completas (CEEMDAN-EWT-CNN-BiLSTM-Attn) =====")
        for h in range(horizon):
            y_true_h, y_pred_h = y_test_inv[:, h], y_pred_median[:, h]
            mape = np.mean(np.abs((y_true_h - y_pred_h) / cap)) * 100 if cap > 0 else 0
            rmse = sqrt(mean_squared_error(y_true_h, y_pred_h))
            mae = mean_absolute_error(y_true_h, y_pred_h)
            r2 = r2_score(y_true_h, y_pred_h)
            print(f"Horizonte t+{(h+1)*10}: MAPE={mape:.4f}%, RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")

        # Lógica de Risco ORI e plotagem
        operational_threshold = 0.3 * cap 
        p_low_t20 = all_predictions[0.1][:, 1]
        p_low_t30 = all_predictions[0.1][:, 2]
        ori_levels = []
        if strategy_horizon == 20:
            for i in range(len(p_low_t20)):
                if p_low_t20[i] < operational_threshold: ori_levels.append('Alto')
                elif p_low_t30[i] < operational_threshold: ori_levels.append('Atenção')
                else: ori_levels.append('Baixo')
        else:
            for p_low in p_low_t30:
                ori_levels.append('Alto' if p_low < operational_threshold else 'Baixo')

        color_map = {'Baixo': 'lightgreen', 'Atenção': 'gold', 'Alto': 'salmon'}
        num_samples_to_plot = 300
        time_axis = np.arange(min(num_samples_to_plot, len(y_test_inv)))

        for h in range(horizon):
            fig, ax = plt.subplots(figsize=(18, 8))
            for i in range(len(time_axis)):
                ax.axvspan(i, i + 1, facecolor=color_map.get(ori_levels[i], 'white'), alpha=0.5, zorder=0)
            ax.axhline(y=operational_threshold, color='red', linestyle=':', linewidth=2.5, label=f'Limiar Operacional ({operational_threshold:.2f} kW)', zorder=4)
            ax.fill_between(time_axis, all_predictions[0.1][:len(time_axis), h], all_predictions[0.9][:len(time_axis), h], color='cornflowerblue', alpha=0.6, label='Intervalo de Confiança (80%)', zorder=1)
            ax.plot(time_axis, y_test_inv[:len(time_axis), h], label="Real", color='black', linewidth=2, zorder=3)
            ax.plot(time_axis, all_predictions[0.5][:len(time_axis), h], label="Previsão Mediana", color='firebrick', linestyle='--', linewidth=2.5, zorder=2)
            ax.set_title(f'Previsão Híbrida (CEEMDAN-EWT-CNN-BiLSTM-Attn) com ORI - Horizonte t+{(h+1)*10} min', fontsize=16)
            ax.set_ylabel('LV ActivePower (kW)', fontsize=12)
            ax.set_xlabel('Amostras de Teste', fontsize=12)
            risk_patches = [Patch(facecolor=color, alpha=0.5, label=f'Risco {level}') for level, color in color_map.items() if level in set(ori_levels)]
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles=handles + risk_patches, loc='best', fontsize=12)
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
            ax.set_xlim(0, len(time_axis))
            ax.set_ylim(bottom=max(0, y_test_inv.min() - (0.05 * cap)))
            plt.tight_layout()
            plt.show()
            plt.close(fig)

    return all_predictions, y_test_inv






#----#
# Adicione esta nova função ao seu myfunctions.py

def proposed_method_cnn_bilstm(new_data, months, look_back, data_partition, cap, horizon=3, strategy_horizon=20, mode='train'):
    """
    Modelo híbrido CEEMDAN-EWT + CNN-BiLSTM (v1), baseado no padrão do proposed_method.
    - Utiliza uma camada Conv1D antes da camada Bidirectional(LSTM).
    - Herda todas as boas práticas: sem data leakage, modos de treino/inferência, etc.
    """
    # --- Imports ---
    import numpy as np
    import pandas as pd
    from math import sqrt
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    # >>>>> MUDANÇA 1: Importar a camada Conv1D <<<<<
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Conv1D
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.optimizers import Adam
    from PyEMD import CEEMDAN
    import ewtpy
    import tensorflow as tf
    import random
    import os
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    import joblib
    import gc

    # --- Reprodutibilidade e Configurações ---
    random.seed(1234)
    np.random.seed(1234)
    tf.random.set_seed(1234)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    # >>>>> MUDANÇA 2: Novo diretório para salvar os modelos <<<<<
    MODEL_DIR = "saved_models/proposed_method_cnn_bilstm"

    # --- Funções Auxiliares (sem alterações) ---
    def create_dataset(dataset, look_back, horizon):
        dataX, dataY = [], []
        for j in range(len(dataset) - look_back - horizon + 1):
            X = dataset[j:(j + look_back), 0]
            Y = dataset[(j + look_back):(j + look_back + horizon), 0]
            dataX.append(X)
            dataY.append(Y)
        return np.array(dataX), np.array(dataY)

    # --- Decomposição e Preparação (sem alterações) ---
    data1 = new_data.loc[new_data['Month'].isin(months)].reset_index(drop=True).dropna()
    
    if mode == 'train':
        print("--- EXECUTANDO proposed_method_cnn_bilstm EM MODO DE TREINAMENTO ---")
        print("Iniciando decomposição do sinal (CEEMDAN + EWT)...")
        s = data1['LV ActivePower (kW)'].values
        emd = CEEMDAN(epsilon=0.05); emd.noise_seed(12345)
        IMFs = emd(s)
        ceemdan1 = pd.DataFrame(IMFs).T
        imf1 = ceemdan1.iloc[:, 0].values
        ewt, mfb, boundaries = ewtpy.EWT1D(imf1, N=3)
        df_ewt = pd.DataFrame(ewt)
        if df_ewt.shape[1] > 2: df_ewt.drop(df_ewt.columns[2], axis=1, inplace=True)
        denoised = df_ewt.sum(axis=1)
        ceemdan_without_imf1 = ceemdan1.iloc[:, 1:]
        ceemdan_without_imf1.columns = [f"imf_{c}" for c in ceemdan_without_imf1.columns]
        denoised.name = "imf_denoised"
        decomposed_df = pd.concat([denoised, ceemdan_without_imf1], axis=1)
        
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(decomposed_df, os.path.join(MODEL_DIR, "decomposed_df.gz"))
        print("Decomposição concluída e salva.")
        
    elif mode == 'inference':
        print("--- EXECUTANDO proposed_method_cnn_bilstm EM MODO DE INFERÊNCIA ---")
        decomposed_path = os.path.join(MODEL_DIR, "decomposed_df.gz")
        if not os.path.exists(decomposed_path):
            raise FileNotFoundError(f"Arquivo de decomposição '{decomposed_path}' não encontrado.")
        decomposed_df = joblib.load(decomposed_path)
        print("Decomposição carregada.")
        
    else:
        raise ValueError("O parâmetro 'mode' deve ser 'train' ou 'inference'.")

    component_names = decomposed_df.columns.tolist()
    quantiles = [0.1, 0.5, 0.9]
    all_predictions = {}

    for q in quantiles:
        imf_predictions_for_quantile_q = []
        quantile_dir = os.path.join(MODEL_DIR, f"q{str(q).replace('.', '')}")

        for col_name in component_names:
            tf.keras.backend.clear_session()
            datasets = decomposed_df[[col_name]].values
            train_size = int(len(datasets) * data_partition)
            train, test = datasets[:train_size], datasets[train_size:]
            trainX, trainY = create_dataset(train, look_back, horizon)
            testX, _ = create_dataset(test, look_back, horizon)

            if trainX.shape[0] == 0: continue

            if mode == 'train':
                print(f"--- Q{q} | Treinando CNN-BiLSTM para: {col_name} ---")
                
                sc_X, sc_y = StandardScaler(), StandardScaler()
                sc_X.fit(trainX)
                sc_y.fit(trainY)

                X_train = sc_X.transform(trainX)
                y_train = sc_y.transform(trainY)
                X_test = sc_X.transform(testX)

                X_train = X_train.reshape((X_train.shape[0], look_back, 1))
                X_test = X_test.reshape((X_test.shape[0], look_back, 1))

                # >>>>> MUDANÇA 3: Arquitetura CNN + BiLSTM <<<<<
                model = Sequential([
                    Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', input_shape=(look_back, 1)),
                    Bidirectional(LSTM(units=32)),
                    Dropout(0.2),
                    Dense(horizon)
                ])
                model.compile(loss=loss_functions[q], optimizer=Adam(learning_rate=0.001))
                early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
                model.fit(X_train, y_train, epochs=5000, batch_size=64, verbose=0, shuffle=False, validation_split=0.1, callbacks=[early_stop])

                y_pred_scaled = model.predict(X_test, verbose=0)
                y_pred_inv = sc_y.inverse_transform(y_pred_scaled)
                
                os.makedirs(quantile_dir, exist_ok=True)
                model.save(os.path.join(quantile_dir, f"model_{col_name}.keras"))
                joblib.dump(sc_X, os.path.join(quantile_dir, f"scaler_X_{col_name}.gz"))
                joblib.dump(sc_y, os.path.join(quantile_dir, f"scaler_y_{col_name}.gz"))

            elif mode == 'inference':
                sc_X = joblib.load(os.path.join(quantile_dir, f"scaler_X_{col_name}.gz"))
                sc_y = joblib.load(os.path.join(quantile_dir, f"scaler_y_{col_name}.gz"))
                custom_objects = {fn.__name__: fn for fn in loss_functions.values()}
                model = tf.keras.models.load_model(os.path.join(quantile_dir, f"model_{col_name}.keras"), custom_objects=custom_objects)

                X_test = sc_X.transform(testX)
                X_test = X_test.reshape((X_test.shape[0], look_back, 1))
                y_pred_scaled = model.predict(X_test, verbose=0)
                y_pred_inv = sc_y.inverse_transform(y_pred_scaled)
            
            imf_predictions_for_quantile_q.append(y_pred_inv)
            gc.collect()

        reconstructed_prediction = np.sum(np.array(imf_predictions_for_quantile_q), axis=0)
        all_predictions[q] = np.clip(reconstructed_prediction, 0, cap)

    # Preparação dos dados reais para validação (sem alterações)
    original_values = data1[['LV ActivePower (kW)']].values
    train_size_orig = int(len(original_values) * data_partition)
    _, test_orig = original_values[:train_size_orig], original_values[train_size_orig:]
    _, y_test_inv = create_dataset(test_orig, look_back, horizon)

    # Correção de cruzamento de quantis (sem alterações)
    print("Aplicando correção de cruzamento de quantis...")
    all_predictions[0.5] = np.maximum(all_predictions[0.1], all_predictions[0.5])
    all_predictions[0.9] = np.maximum(all_predictions[0.5], all_predictions[0.9])

    if mode == 'train':
        # Seção de métricas e gráficos (sem alterações, apenas o título)
        y_pred_median = all_predictions[0.5]
        print("\n===== Métricas Completas (CEEMDAN-EWT-CNN-BiLSTM) =====")
        for h in range(horizon):
            y_true_h, y_pred_h = y_test_inv[:, h], y_pred_median[:, h]
            mape = np.mean(np.abs((y_true_h - y_pred_h) / cap)) * 100 if cap > 0 else 0
            rmse = sqrt(mean_squared_error(y_true_h, y_pred_h))
            mae = mean_absolute_error(y_true_h, y_pred_h)
            r2 = r2_score(y_true_h, y_pred_h)
            print(f"Horizonte t+{(h+1)*10}: MAPE={mape:.4f}%, RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")

        # Lógica de Risco ORI e plotagem
        operational_threshold = 0.3 * cap 
        p_low_t20 = all_predictions[0.1][:, 1]
        p_low_t30 = all_predictions[0.1][:, 2]
        ori_levels = []
        if strategy_horizon == 20:
            for i in range(len(p_low_t20)):
                if p_low_t20[i] < operational_threshold: ori_levels.append('Alto')
                elif p_low_t30[i] < operational_threshold: ori_levels.append('Atenção')
                else: ori_levels.append('Baixo')
        else:
            for p_low in p_low_t30:
                ori_levels.append('Alto' if p_low < operational_threshold else 'Baixo')

        color_map = {'Baixo': 'lightgreen', 'Atenção': 'gold', 'Alto': 'salmon'}
        num_samples_to_plot = 300
        time_axis = np.arange(min(num_samples_to_plot, len(y_test_inv)))

        for h in range(horizon):
            fig, ax = plt.subplots(figsize=(18, 8))
            for i in range(len(time_axis)):
                ax.axvspan(i, i + 1, facecolor=color_map.get(ori_levels[i], 'white'), alpha=0.5, zorder=0)
            ax.axhline(y=operational_threshold, color='red', linestyle=':', linewidth=2.5, label=f'Limiar Operacional ({operational_threshold:.2f} kW)', zorder=4)
            ax.fill_between(time_axis, all_predictions[0.1][:len(time_axis), h], all_predictions[0.9][:len(time_axis), h], color='cornflowerblue', alpha=0.6, label='Intervalo de Confiança (80%)', zorder=1)
            ax.plot(time_axis, y_test_inv[:len(time_axis), h], label="Real", color='black', linewidth=2, zorder=3)
            ax.plot(time_axis, all_predictions[0.5][:len(time_axis), h], label="Previsão Mediana", color='firebrick', linestyle='--', linewidth=2.5, zorder=2)
            # >>>>> MUDANÇA 4: Título do gráfico atualizado <<<<<
            ax.set_title(f'Previsão Híbrida (CEEMDAN-EWT-CNN-BiLSTM) com ORI - Horizonte t+{(h+1)*10} min', fontsize=16)
            ax.set_ylabel('LV ActivePower (kW)', fontsize=12)
            ax.set_xlabel('Amostras de Teste', fontsize=12)
            risk_patches = [Patch(facecolor=color, alpha=0.5, label=f'Risco {level}') for level, color in color_map.items() if level in set(ori_levels)]
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles=handles + risk_patches, loc='best', fontsize=12)
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
            ax.set_xlim(0, len(time_axis))
            ax.set_ylim(bottom=max(0, y_test_inv.min() - (0.05 * cap)))
            plt.tight_layout()
            plt.show()
            plt.close(fig)

    return all_predictions, y_test_inv




#-----#
## Hybrid CEEMDAN-EWT LSTM with Optuna (multi-horizon)
def proposed_method_lstm_optuna(new_data, i, look_back, data_partition, cap, horizon=3):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from math import sqrt
    from sklearn.preprocessing import StandardScaler
    from sklearn import metrics
    from sklearn.model_selection import TimeSeriesSplit
    import optuna
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.optimizers import Adam
    from PyEMD import CEEMDAN
    import ewtpy
    import tensorflow as tf
    import random, os

    # --- Dataset helper ---
    def create_dataset(dataset, look_back=1, horizon=1):
        dataX, dataY = [], []
        for j in range(len(dataset) - look_back - horizon + 1):
            dataX.append(dataset[j:(j+look_back), 0])
            dataY.append(dataset[j+look_back:j+look_back+horizon, 0])
        return np.array(dataX), np.array(dataY)

    # Reprodutibilidade
    random.seed(1234)
    np.random.seed(1234)
    tf.random.set_seed(1234)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # --- Seleção dos dados ---
    data1=new_data.loc[new_data['Month'].isin(i)].reset_index(drop=True).dropna()
    s = data1['LV ActivePower (kW)'].values

    # --- CEEMDAN ---
    emd = CEEMDAN(epsilon=0.05)
    emd.noise_seed(12345)
    IMFs = emd(s)
    ceemdan1=pd.DataFrame(IMFs).T

    # --- EWT no 1º IMF ---
    imf1=ceemdan1.iloc[:,0]
    ewt, _, _ = ewtpy.EWT1D(imf1, N=3)
    df_ewt=pd.DataFrame(ewt).drop(2, axis=1)
    denoised=df_ewt.sum(axis=1)
    ceemdan_without_imf1=ceemdan1.iloc[:,1:]
    final_df=pd.concat([denoised, ceemdan_without_imf1],axis=1)
    num_imfs = final_df.shape[1]

    # ===============================
    # OPTUNA OBJECTIVE FUNCTION
    # ===============================
    def objective(trial):
        lstm_units = trial.suggest_categorical("lstm_units", [16, 32, 64, 128])
        dropout_rate = trial.suggest_float("dropout", 0.1, 0.5)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

        tscv = TimeSeriesSplit(n_splits=3)
        val_scores = []

        for train_index, val_index in tscv.split(final_df):
            preds, trues = [], []

            for col in range(num_imfs):
                series = final_df.iloc[:, col].values.reshape(-1, 1)
                sc = StandardScaler()
                series = sc.fit_transform(series)

                X, y = create_dataset(series, look_back, horizon)
                X = X.reshape((X.shape[0], look_back, 1))

                if val_index[-1] >= len(X):
                    continue

                X_train, X_val = X[train_index], X[val_index]
                y_train, y_val = y[train_index], y[val_index]

                model = Sequential([
                    LSTM(units=lstm_units, return_sequences=False, input_shape=(look_back,1)),
                    Dropout(dropout_rate),
                    Dense(horizon)
                ])
                model.compile(optimizer=Adam(learning_rate=lr), loss="mse")

                early_stop = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)

                model.fit(X_train, y_train, epochs=5000, batch_size=batch_size,
                          validation_data=(X_val, y_val), verbose=0, shuffle=False, callbacks=[early_stop])

                y_val_pred = model.predict(X_val, verbose=0)
                y_val_pred_inv = sc.inverse_transform(y_val_pred)
                y_val_inv = sc.inverse_transform(y_val)

                preds.append(y_val_pred_inv)
                trues.append(y_val_inv)

            if len(preds)==0: 
                continue

            preds = np.array(preds)   # (comp, samples, horizon)
            trues = np.array(trues)

            y_pred_final = np.sum(preds, axis=0)
            y_true_final = np.sum(trues, axis=0)

            # ===== MAPE médio dos horizontes =====
            mape_horizons = []
            for step in range(horizon):
                mape_h = np.mean(np.abs(y_true_final[:, step] - y_pred_final[:, step]) / cap) * 100
                mape_horizons.append(mape_h)
            val_scores.append(np.mean(mape_horizons))

        return np.mean(val_scores) if val_scores else np.inf

    # ===============================
    # OPTUNA SEARCH
    # ===============================
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30, show_progress_bar=True)
    best_params = study.best_params
    print("Melhores parâmetros:", best_params)

    # ===============================
    # RETRAIN FINAL MODEL
    # ===============================
    preds, trues = [], []

    for col in range(num_imfs):
        series = final_df.iloc[:, col].values.reshape(-1, 1)
        sc = StandardScaler()
        series = sc.fit_transform(series)

        X, y = create_dataset(series, look_back, horizon)
        X = X.reshape((X.shape[0], look_back, 1))

        train_size = int(len(X)*data_partition)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        model = Sequential([
            LSTM(units=best_params['lstm_units'], return_sequences=False, input_shape=(look_back,1)),
            Dropout(best_params['dropout']),
            Dense(horizon)
        ])
        model.compile(optimizer=Adam(learning_rate=best_params['lr']), loss="mse")

        early_stop = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)

        model.fit(X_train, y_train, epochs=5000, batch_size=best_params['batch_size'],
                  verbose=0, shuffle=False, validation_split=0.1, callbacks=[early_stop])

        pred = model.predict(X_test, verbose=0)
        pred_inv = sc.inverse_transform(pred)
        y_test_inv = sc.inverse_transform(y_test)

        preds.append(pred_inv)
        trues.append(y_test_inv)

    preds = np.array(preds)
    trues = np.array(trues)

    final_pred = np.sum(preds, axis=0)
    final_true = np.sum(trues, axis=0)

    # ===== Métricas por horizonte =====
    mape_horizons, rmse_horizons, mae_horizons = [], [], []
    for step in range(horizon):
        mape_h = np.mean(np.abs(final_true[:, step] - final_pred[:, step]) / cap) * 100
        rmse_h = sqrt(metrics.mean_squared_error(final_true[:, step], final_pred[:, step]))
        mae_h = metrics.mean_absolute_error(final_true[:, step], final_pred[:, step])
        mape_horizons.append(mape_h)
        rmse_horizons.append(rmse_h)
        mae_horizons.append(mae_h)
        print(f"H{step+1}: MAPE={mape_h:.4f} | RMSE={rmse_h:.4f} | MAE={mae_h:.4f}")

    print("MAPE médio:", np.mean(mape_horizons))

    # --- Plot separado para cada horizonte ---
    import matplotlib.pyplot as plt
    for step in range(horizon):
        plt.figure(figsize=(12,6))
        plt.plot(final_true[:, step], label=f"Real t+{step+1}", color="blue")
        plt.plot(final_pred[:, step], label=f"Previsto t+{step+1}", color="orange", linestyle="--")
        plt.title(f"Previsão CEEMDAN+EWT LSTM Optuna - Horizonte t+{step+1}")
        plt.xlabel("Amostras")
        plt.ylabel("LV ActivePower (kW)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()





#------#
## Hybrid CEEMDAN-EWT BILSTM with Optuna (multi-horizon)
def proposed_method_bilstm_optuna(new_data, i, look_back, data_partition, cap, horizon=3):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from math import sqrt
    from sklearn.preprocessing import StandardScaler
    from sklearn import metrics
    from sklearn.model_selection import TimeSeriesSplit
    import optuna
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.optimizers import Adam
    from PyEMD import CEEMDAN
    import ewtpy
    import tensorflow as tf
    import random, os

    # --- Dataset helper ---
    def create_dataset(dataset, look_back=1, horizon=1):
        dataX, dataY = [], []
        for j in range(len(dataset) - look_back - horizon + 1):
            dataX.append(dataset[j:(j+look_back), 0])
            dataY.append(dataset[j+look_back:j+look_back+horizon, 0])
        return np.array(dataX), np.array(dataY)

    # Reprodutibilidade
    random.seed(1234)
    np.random.seed(1234)
    tf.random.set_seed(1234)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # --- Seleção dos dados ---
    data1=new_data.loc[new_data['Month'].isin(i)].reset_index(drop=True).dropna()
    s = data1['LV ActivePower (kW)'].values

    # --- CEEMDAN ---
    emd = CEEMDAN(epsilon=0.05)
    emd.noise_seed(12345)
    IMFs = emd(s)
    ceemdan1=pd.DataFrame(IMFs).T

    # --- EWT no 1º IMF ---
    imf1=ceemdan1.iloc[:,0]
    ewt, _, _ = ewtpy.EWT1D(imf1, N=3)
    df_ewt=pd.DataFrame(ewt).drop(2, axis=1)
    denoised=df_ewt.sum(axis=1)
    ceemdan_without_imf1=ceemdan1.iloc[:,1:]
    final_df=pd.concat([denoised, ceemdan_without_imf1],axis=1)
    num_imfs = final_df.shape[1]

    # ===============================
    # OPTUNA OBJECTIVE FUNCTION
    # ===============================
    def objective(trial):
        lstm_units = trial.suggest_categorical("lstm_units", [16, 32, 64, 128])
        dropout_rate = trial.suggest_float("dropout", 0.1, 0.5)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

        tscv = TimeSeriesSplit(n_splits=3)
        val_scores = []

        for train_index, val_index in tscv.split(final_df):
            preds, trues = [], []

            for col in range(num_imfs):
                series = final_df.iloc[:, col].values.reshape(-1, 1)
                sc = StandardScaler()
                series = sc.fit_transform(series)

                X, y = create_dataset(series, look_back, horizon)
                X = X.reshape((X.shape[0], look_back, 1))

                if val_index[-1] >= len(X):
                    continue

                X_train, X_val = X[train_index], X[val_index]
                y_train, y_val = y[train_index], y[val_index]

                model = Sequential([
                    Bidirectional(LSTM(units=lstm_units, return_sequences=False),input_shape=(look_back,1)),
                    Dropout(dropout_rate),
                    Dense(horizon)
                ])
                model.compile(optimizer=Adam(learning_rate=lr), loss="mse")

                early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

                model.fit(X_train, y_train, epochs=5000, batch_size=batch_size,
                          validation_data=(X_val, y_val), verbose=0, shuffle=False, callbacks=[early_stop])

                y_val_pred = model.predict(X_val, verbose=0)
                y_val_pred_inv = sc.inverse_transform(y_val_pred)
                y_val_inv = sc.inverse_transform(y_val)

                preds.append(y_val_pred_inv)
                trues.append(y_val_inv)

            if len(preds)==0: 
                continue

            preds = np.array(preds)   # (comp, samples, horizon)
            trues = np.array(trues)

            y_pred_final = np.sum(preds, axis=0)
            y_true_final = np.sum(trues, axis=0)

            # ===== MAPE médio dos horizontes =====
            mape_horizons = []
            for step in range(horizon):
                mape_h = np.mean(np.abs(y_true_final[:, step] - y_pred_final[:, step]) / cap) * 100
                mape_horizons.append(mape_h)
            val_scores.append(np.mean(mape_horizons))

        return np.mean(val_scores) if val_scores else np.inf

    # ===============================
    # OPTUNA SEARCH
    # ===============================
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50, show_progress_bar=True)
    best_params = study.best_params
    print("Melhores parâmetros:", best_params)

    # ===============================
    # RETRAIN FINAL MODEL
    # ===============================
    preds, trues = [], []

    for col in range(num_imfs):
        series = final_df.iloc[:, col].values.reshape(-1, 1)
        sc = StandardScaler()
        series = sc.fit_transform(series)

        X, y = create_dataset(series, look_back, horizon)
        X = X.reshape((X.shape[0], look_back, 1))

        train_size = int(len(X)*data_partition)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        model = Sequential([
            Bidirectional(LSTM(units=best_params['lstm_units'], return_sequences=False), input_shape=(look_back,1)),
            Dropout(best_params['dropout']),
            Dense(horizon)
        ])
        model.compile(optimizer=Adam(learning_rate=best_params['lr']), loss="mse")

        early_stop = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)

        model.fit(X_train, y_train, epochs=5000, batch_size=best_params['batch_size'],
                  verbose=0, shuffle=False, validation_split=0.1, callbacks=[early_stop])

        pred = model.predict(X_test, verbose=0)
        pred_inv = sc.inverse_transform(pred)
        y_test_inv = sc.inverse_transform(y_test)

        preds.append(pred_inv)
        trues.append(y_test_inv)

    preds = np.array(preds)
    trues = np.array(trues)

    final_pred = np.sum(preds, axis=0)
    final_true = np.sum(trues, axis=0)

    # ===== Métricas por horizonte =====
    mape_horizons, rmse_horizons, mae_horizons = [], [], []
    for step in range(horizon):
        mape_h = np.mean(np.abs(final_true[:, step] - final_pred[:, step]) / cap) * 100
        rmse_h = sqrt(metrics.mean_squared_error(final_true[:, step], final_pred[:, step]))
        mae_h = metrics.mean_absolute_error(final_true[:, step], final_pred[:, step])
        mape_horizons.append(mape_h)
        rmse_horizons.append(rmse_h)
        mae_horizons.append(mae_h)
        print(f"H{step+1}: MAPE={mape_h:.4f} | RMSE={rmse_h:.4f} | MAE={mae_h:.4f}")

    print("MAPE médio:", np.mean(mape_horizons))

    # --- Plot separado para cada horizonte ---
    import matplotlib.pyplot as plt
    for step in range(horizon):
        plt.figure(figsize=(12,6))
        plt.plot(final_true[:, step], label=f"Real t+{step+1}", color="blue")
        plt.plot(final_pred[:, step], label=f"Previsto t+{step+1}", color="orange", linestyle="--")
        plt.title(f"Previsão CEEMDAN+EWT BILSTM Optuna - Horizonte t+{step+1}")
        plt.xlabel("Amostras")
        plt.ylabel("LV ActivePower (kW)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()




#---#
## Hybrid CEEMDAN-EWT BiLSTM + Attention with Optuna (3-step ahead)
def proposed_method_bilstm_att_optuna(new_data, i, look_back, data_partition, cap, horizon=3):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from math import sqrt
    from sklearn.preprocessing import StandardScaler
    from sklearn import metrics
    from sklearn.model_selection import TimeSeriesSplit
    import optuna
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.optimizers import Adam
    from PyEMD import CEEMDAN
    import ewtpy
    import tensorflow as tf
    import random, os

    # --- Attention Layer ---
    class AttentionLayer(tf.keras.layers.Layer):
        def __init__(self, **kwargs):
            super(AttentionLayer, self).__init__(**kwargs)

        def build(self, input_shape):
            self.W = self.add_weight(
                name='attention_weight',
                shape=(input_shape[-1], 1),
                initializer='random_normal',
                trainable=True
            )
            self.b = self.add_weight(
                name='attention_bias',
                shape=(input_shape[1], 1),
                initializer='zeros',
                trainable=True
            )
            super(AttentionLayer, self).build(input_shape)

        def call(self, inputs):
            e = tf.math.tanh(tf.matmul(inputs, self.W) + self.b)  
            a = tf.nn.softmax(e, axis=1)                         
            output = inputs * a                                   
            return tf.reduce_sum(output, axis=1)                  

    # --- Dataset helper (multi-step) ---
    def create_dataset(dataset, look_back=1, horizon=1):
        dataX, dataY = [], []
        for j in range(len(dataset) - look_back - horizon + 1):
            dataX.append(dataset[j:(j+look_back), 0])
            dataY.append(dataset[j+look_back:j+look_back+horizon, 0])
        return np.array(dataX), np.array(dataY)

    # Reprodutibilidade
    random.seed(1234)
    np.random.seed(1234)
    tf.random.set_seed(1234)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # --- Seleção dos dados ---
    data1=new_data.loc[new_data['Month'].isin(i)].reset_index(drop=True).dropna()
    s = data1['LV ActivePower (kW)'].values

    # --- CEEMDAN ---
    emd = CEEMDAN(epsilon=0.05)
    emd.noise_seed(12345)
    IMFs = emd(s)
    ceemdan1=pd.DataFrame(IMFs).T

    # --- EWT no 1º IMF ---
    imf1=ceemdan1.iloc[:,0]
    ewt, _, _ = ewtpy.EWT1D(imf1, N=3)
    df_ewt=pd.DataFrame(ewt).drop(2, axis=1)
    denoised=df_ewt.sum(axis=1)
    ceemdan_without_imf1=ceemdan1.iloc[:,1:]
    final_df=pd.concat([denoised, ceemdan_without_imf1],axis=1)
    num_imfs = final_df.shape[1]

    # ===============================
    # OPTUNA OBJECTIVE FUNCTION
    # ===============================
    def objective(trial):
        lstm_units = trial.suggest_categorical("lstm_units", [16, 32, 64, 128])
        dropout_rate = trial.suggest_float("dropout", 0.1, 0.5)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

        tscv = TimeSeriesSplit(n_splits=3)
        val_scores = []

        for train_index, val_index in tscv.split(final_df):
            pred_ensemble = []
            y_true_sum = None

            for col in range(num_imfs):
                series = final_df.iloc[:, col].values.reshape(-1, 1)
                sc = StandardScaler()
                series = sc.fit_transform(series)

                X, y = create_dataset(series, look_back, horizon)
                X = X.reshape((X.shape[0], look_back, 1))

                if val_index[-1] >= len(X):
                    continue

                X_train, X_val = X[train_index], X[val_index]
                y_train, y_val = y[train_index], y[val_index]

                model = Sequential([
                    Bidirectional(LSTM(units=lstm_units, return_sequences=True, input_shape=(look_back,1))),
                    AttentionLayer(),
                    Dropout(dropout_rate),
                    Dense(horizon)   # saída multi-step
                ])
                model.compile(optimizer=Adam(learning_rate=lr), loss="mse")

                early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

                model.fit(X_train, y_train, epochs=5000, batch_size=batch_size,
                          validation_data=(X_val, y_val), verbose=0, shuffle=False, callbacks=[early_stop])

                y_val_pred = model.predict(X_val, verbose=0)
                y_val_pred_inv = sc.inverse_transform(y_val_pred)
                y_val_inv = sc.inverse_transform(y_val)

                # ===== MAPE médio dos horizontes =====
                mape_horizons = []
                for step in range(horizon):
                    mape_h = np.mean(np.abs(y_val_inv[:, step] - y_val_pred_inv[:, step]) / cap) * 100
                    mape_horizons.append(mape_h)
                val_scores.append(np.mean(mape_horizons))

        return np.mean(val_scores) if val_scores else np.inf

    # ===============================
    # OPTUNA SEARCH
    # ===============================
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30, show_progress_bar=True)
    best_params = study.best_params
    print("Melhores parâmetros:", best_params)

    # ===============================
    # RETRAIN FINAL MODEL
    # ===============================
    pred_test_all = []
    y_true_all = []

    for col in range(num_imfs):
        series = final_df.iloc[:, col].values.reshape(-1, 1)
        sc = StandardScaler()
        series = sc.fit_transform(series)

        X, y = create_dataset(series, look_back, horizon)
        X = X.reshape((X.shape[0], look_back, 1))

        train_size = int(len(X)*data_partition)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        model = Sequential([
            Bidirectional(LSTM(units=best_params['lstm_units'], return_sequences=True, input_shape=(look_back,1))),
            AttentionLayer(),
            Dropout(best_params['dropout']),
            Dense(horizon)
        ])
        model.compile(optimizer=Adam(learning_rate=best_params['lr']), loss="mse")

        early_stop = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)

        model.fit(X_train, y_train, epochs=5000, batch_size=best_params['batch_size'],
                  verbose=0, shuffle=False, validation_split=0.1, callbacks=[early_stop])

        pred = model.predict(X_test, verbose=0)
        pred_inv = sc.inverse_transform(pred)
        y_test_inv = sc.inverse_transform(y_test)

        pred_test_all.append(pred_inv)
        y_true_all.append(y_test_inv)

    # Reconstrução
    final_pred = np.sum(pred_test_all, axis=0)
    final_true = np.sum(y_true_all, axis=0)

    # --- Métricas ---
    for step in range(horizon):
        mape_h = np.mean(np.abs(final_true[:, step] - final_pred[:, step]) / cap) * 100
        rmse_h = sqrt(metrics.mean_squared_error(final_true[:, step], final_pred[:, step]))
        mae_h  = metrics.mean_absolute_error(final_true[:, step], final_pred[:, step])
        print(f"Horizonte {step+1} -> MAPE: {mape_h:.4f} | RMSE: {rmse_h:.4f} | MAE: {mae_h:.4f}")

    # --- Plot por horizonte ---
    plt.figure(figsize=(12,6))
    for step in range(horizon):
        plt.plot(final_true[:, step], label=f"Real t+{step+1}")
        plt.plot(final_pred[:, step], linestyle="--", label=f"Predito t+{step+1}")
    plt.title("CEEMDAN+EWT BiLSTM+Attention com Optuna (3 passos à frente)")
    plt.xlabel("Amostras")
    plt.ylabel("LV ActivePower (kW)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()




#-------#

# Hybrid CEEMDAN-EWT CNN-BiLSTM + Attention with Optuna (3-step ahead)
def proposed_method_cnn_bilstm_att_optuna(new_data, i, look_back, data_partition, cap):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from math import sqrt
    from sklearn.preprocessing import StandardScaler
    from sklearn import metrics
    from sklearn.model_selection import TimeSeriesSplit
    import optuna
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Conv1D
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.optimizers import Adam
    from PyEMD import CEEMDAN
    import ewtpy
    import tensorflow as tf
    import random, os

    # --- Attention Layer ---
    class AttentionLayer(tf.keras.layers.Layer):
        def __init__(self, **kwargs):
            super(AttentionLayer, self).__init__(**kwargs)

        def build(self, input_shape):
            self.W = self.add_weight(
                name='attention_weight',
                shape=(input_shape[-1], 1),
                initializer='random_normal',
                trainable=True
            )
            self.b = self.add_weight(
                name='attention_bias',
                shape=(input_shape[1], 1),
                initializer='zeros',
                trainable=True
            )
            super(AttentionLayer, self).build(input_shape)

        def call(self, inputs):
            e = tf.math.tanh(tf.matmul(inputs, self.W) + self.b)  
            a = tf.nn.softmax(e, axis=1)                         
            output = inputs * a                                   
            return tf.reduce_sum(output, axis=1)                  

    # --- Dataset helper (multi-step = 3) ---
    def create_dataset(dataset, look_back=1, horizon=3):
        dataX, dataY = [], []
        for j in range(len(dataset)-look_back-horizon+1):
            dataX.append(dataset[j:(j+look_back), 0])
            dataY.append(dataset[j+look_back:j+look_back+horizon, 0])
        return np.array(dataX), np.array(dataY)

    # Reprodutibilidade
    random.seed(1234)
    np.random.seed(1234)
    tf.random.set_seed(1234)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # --- Seleção dos dados ---
    data1=new_data.loc[new_data['Month'].isin(i)].reset_index(drop=True).dropna()
    s = data1['LV ActivePower (kW)'].values

    # --- CEEMDAN ---
    emd = CEEMDAN(epsilon=0.05)
    emd.noise_seed(12345)
    IMFs = emd(s)
    ceemdan1=pd.DataFrame(IMFs).T

    # --- EWT no 1º IMF ---
    imf1=ceemdan1.iloc[:,0]
    ewt, _, _ = ewtpy.EWT1D(imf1, N=3)
    df_ewt=pd.DataFrame(ewt).drop(2, axis=1)
    denoised=df_ewt.sum(axis=1)
    ceemdan_without_imf1=ceemdan1.iloc[:,1:]
    final_df=pd.concat([denoised, ceemdan_without_imf1],axis=1)
    num_imfs = final_df.shape[1]

    # ===============================
    # OPTUNA OBJECTIVE FUNCTION
    # ===============================
    def objective(trial):
        filters = trial.suggest_categorical("filters", [16, 32, 64])
        kernel_size = trial.suggest_int("kernel_size", 2, 5)
        lstm_units = trial.suggest_categorical("lstm_units", [32, 64, 128])
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

        dropout_rate = 0.2   # fixo
        batch_size   = 32    # fixo

        tscv = TimeSeriesSplit(n_splits=3)
        val_scores = []

        for train_index, val_index in tscv.split(final_df):
            preds_h = [[] for _ in range(3)]   # armazenar previsões por horizonte
            trues_h = [[] for _ in range(3)]

            for col in range(num_imfs):
                series = final_df.iloc[:, col].values.reshape(-1, 1)
                sc = StandardScaler()
                series = sc.fit_transform(series)

                X, y = create_dataset(series, look_back, horizon=3)
                X = X.reshape((X.shape[0], look_back, 1))

                if val_index[-1] >= len(X):
                    continue

                X_train, X_val = X[train_index], X[val_index]
                y_train, y_val = y[train_index], y[val_index]

                model = Sequential([
                    Conv1D(filters=filters, kernel_size=kernel_size, activation="relu", padding="same", input_shape=(look_back,1)),
                    Bidirectional(LSTM(units=lstm_units, return_sequences=True)),
                    AttentionLayer(),
                    Dropout(dropout_rate),
                    Dense(3)  # 3 horizontes
                ])
                model.compile(optimizer=Adam(learning_rate=lr), loss="mse")

                early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

                model.fit(X_train, y_train, epochs=5000, batch_size=batch_size,
                          validation_data=(X_val, y_val), verbose=0, shuffle=False, callbacks=[early_stop])

                pred = model.predict(X_val, verbose=0)
                pred_inv = sc.inverse_transform(pred)  
                y_val_inv = sc.inverse_transform(y_val)

                for h in range(3):
                    preds_h[h].append(pred_inv[:,h])
                    trues_h[h].append(y_val_inv[:,h])

            if len(preds_h[0])==0:
                continue

            mape_h = []
            for h in range(3):
                summed_pred = np.sum(preds_h[h], axis=0)
                summed_true = np.sum(trues_h[h], axis=0)
                mape_h.append(np.mean(np.abs(summed_true - summed_pred) / cap) * 100)

            val_scores.append(np.mean(mape_h))  # média dos 3 horizontes

        return np.mean(val_scores) if val_scores else np.inf

    # ===============================
    # OPTUNA SEARCH
    # ===============================
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50, show_progress_bar=True)
    best_params = study.best_params
    print("Melhores parâmetros:", best_params)

    # ===============================
    # RETRAIN FINAL MODEL
    # ===============================
    preds_h_final = [[] for _ in range(3)]
    trues_h_final = [[] for _ in range(3)]

    for col in range(num_imfs):
        series = final_df.iloc[:, col].values.reshape(-1, 1)
        sc = StandardScaler()
        series = sc.fit_transform(series)

        X, y = create_dataset(series, look_back, horizon=3)
        X = X.reshape((X.shape[0], look_back, 1))

        train_size = int(len(X)*data_partition)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        model = Sequential([
            Conv1D(filters=best_params['filters'], kernel_size=best_params['kernel_size'], activation="relu", padding="same", input_shape=(look_back,1)),
            Bidirectional(LSTM(units=best_params['lstm_units'], return_sequences=True)),
            AttentionLayer(),
            Dropout(0.2),
            Dense(3)
        ])
        model.compile(optimizer=Adam(learning_rate=best_params['lr']), loss="mse")

        early_stop = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)

        model.fit(X_train, y_train, epochs=5000, batch_size=32,
                  verbose=0, shuffle=False, validation_split=0.1, callbacks=[early_stop])

        pred = model.predict(X_test, verbose=0)
        pred_inv = sc.inverse_transform(pred)
        y_test_inv = sc.inverse_transform(y_test)

        for h in range(3):
            preds_h_final[h].append(pred_inv[:,h])
            trues_h_final[h].append(y_test_inv[:,h])

    # --- Avaliação final ---
    for h in range(3):
        final_pred = np.sum(preds_h_final[h], axis=0)
        final_true = np.sum(trues_h_final[h], axis=0)

        mape = np.mean(np.abs(final_true - final_pred)/cap)*100
        rmse = sqrt(metrics.mean_squared_error(final_true, final_pred))
        mae  = metrics.mean_absolute_error(final_true, final_pred)

        print(f"Horizonte t+{h+1} | MAPE: {mape:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f}")

        plt.figure(figsize=(12,4))
        plt.plot(final_true, label="Real", color="blue")
        plt.plot(final_pred, label=f"Predito (t+{h+1})", color="orange", linestyle="--")
        plt.title(f"CEEMDAN+EWT CNN-BiLSTM+Attention com Optuna - Horizonte t+{h+1}")
        plt.xlabel("Amostras")
        plt.ylabel("LV ActivePower (kW)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
#------#







# In[10]:


##HYBRID EEMD BO LSTM

def eemd_bo_lstm(new_data,i,look_back,data_partition,cap):

    x=i
    data1=new_data.loc[new_data['month'].isin(x)]
    data1=data1.reset_index(drop=True)
    data1=data1.dropna()
    
    datas=data1['LV ActivePower (kW)']
    datas_wind=pd.DataFrame(datas)
    dfs=datas
    s = dfs.values

    from PyEMD import EEMD
    import ewtpy
    
    emd = EEMD(noise_width=0.02)
    emd.noise_seed(12345)

    IMFs = emd(s)

    full_imf=pd.DataFrame(IMFs)
    data_decomp=full_imf.T
    


    pred_test=[]
    test_ori=[]
    pred_train=[]
    train_ori=[]

    epoch=100
    batch_size=64
    neuron=128
    lr=0.001
    optimizer='Adam'

    for col in data_decomp:

        datasetss2=pd.DataFrame(data_decomp[col])
        datasets=datasetss2.values
        train_size = int(len(datasets) * data_partition)
        test_size = len(datasets) - train_size
        train, test = datasets[0:train_size], datasets[train_size:len(datasets)]

        trainX, trainY = create_dataset(train, look_back)
        testX, testY = create_dataset(test, look_back)
        X_train=pd.DataFrame(trainX)
        Y_train=pd.DataFrame(trainY)
        X_test=pd.DataFrame(testX)
        Y_test=pd.DataFrame(testY)
        sc_X = StandardScaler()
        sc_y = StandardScaler()
        X= sc_X.fit_transform(X_train)
        y= sc_y.fit_transform(Y_train)
        X1= sc_X.fit_transform(X_test)
        y1= sc_y.fit_transform(Y_test)
        y=y.ravel()
        y1=y1.ravel()  

        import numpy
        trainX = numpy.reshape(X, (X.shape[0],X.shape[1],1))
        testX = numpy.reshape(X1, (X1.shape[0],X1.shape[1],1))
    
        

        numpy.random.seed(1234)
        import tensorflow as tf
        tf.random.set_seed(1234)

    
        import os 
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        from keras.models import Sequential
        from keras.layers.core import Dense, Dropout, Activation
        from sklearn.metrics import mean_squared_error
        from keras.models import Sequential
        from keras.layers import LSTM, Dense
        from keras.callbacks import EarlyStopping
        from bayes_opt import BayesianOptimization
        # Define the LSTM model
        def create_model(units,learning_rate):
            model = Sequential()
            model.add(LSTM(units,input_shape=(trainX.shape[1], trainX.shape[2])))
            model.add(Dense(1))
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            model.compile(loss='mse',optimizer=optimizer)
            return model
        
        # Define the objective function for Bayesian Optimization
        def lstm_cv(units,learning_rate):
            model = create_model(int(units),learning_rate)
            es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=10)
            history = model.fit(trainX, y, epochs50100, batch_size=32, validation_split=0.2, verbose=0, callbacks=[es])
            y_pred = model.predict(testX)
            mse = mean_squared_error(y1, y_pred)
            return -mse

        # Define the parameter space for Bayesian Optimization
        pbounds = {'units': (50,200),'learning_rate': (0.001, 0.01)}

        # Run Bayesian Optimization
        lstm_bo = BayesianOptimization(f=lstm_cv, pbounds=pbounds, random_state=42)
        lstm_bo.maximize(init_points=5, n_iter=10, acq='ei')

        # Print the optimal hyperparameters
        print(lstm_bo.max)
    
        opt_lr = lstm_bo.max['params']['learning_rate']
        opt_unit= lstm_bo.max['params']['units']
        opt_units=int(opt_unit)

        neuron=opt_units
        model = Sequential()
        model.add(LSTM(units = neuron,input_shape=(trainX.shape[1], trainX.shape[2])))
        model.add(Dense(1))
        optimizer = tf.keras.optimizers.Adam(learning_rate=opt_lr)
        model.compile(loss='mse',optimizer=optimizer)


        # Fitting the RNN to the Training set
        model.fit(trainX, y, epochs = epoch, batch_size = batch_size,verbose=0)

        # make predictions
        y_pred_train = model.predict(trainX)
        y_pred_test = model.predict(testX)

        # make predictions

        y_pred_test= numpy.array(y_pred_test).ravel()
        y_pred_test=pd.DataFrame(y_pred_test)
        y1=pd.DataFrame(y1)
        y=pd.DataFrame(y)
        y_pred_train= numpy.array(y_pred_train).ravel()
        y_pred_train=pd.DataFrame(y_pred_train)

        y_test= sc_y.inverse_transform (y1)
        y_train= sc_y.inverse_transform (y)

        y_pred_test1= sc_y.inverse_transform (y_pred_test)
        y_pred_train1= sc_y.inverse_transform (y_pred_train)


        pred_test.append(y_pred_test1)
        test_ori.append(y_test)
        pred_train.append(y_pred_train1)
        train_ori.append(y_train)


    result_pred_test= pd.DataFrame.from_records(pred_test)
    result_pred_train= pd.DataFrame.from_records(pred_train)


    a=result_pred_test.sum(axis = 0, skipna = True) 
    b=result_pred_train.sum(axis = 0, skipna = True) 


    dataframe=pd.DataFrame(dfs)
    dataset=dataframe.values

    train_size = int(len(dataset) * data_partition)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size], dataset[train_size:len(dataset)]

    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    X_train=pd.DataFrame(trainX)
    Y_train=pd.DataFrame(trainY)
    X_test=pd.DataFrame(testX)
    Y_test=pd.DataFrame(testY)

    sc_X = StandardScaler()
    sc_y = StandardScaler() 
    X= sc_X.fit_transform(X_train)
    y= sc_y.fit_transform(Y_train)
    X1= sc_X.fit_transform(X_test)
    y1= sc_y.fit_transform(Y_test)
    y=y.ravel()
    y1=y1.ravel()


    trainX = numpy.reshape(X, (X.shape[0], 1, X.shape[1]))
    testX = numpy.reshape(X1, (X1.shape[0], 1, X1.shape[1]))

    numpy.random.seed(1234)
    tf.random.set_seed(1234)
    
    y1=pd.DataFrame(y1)
    y=pd.DataFrame(y)

    y_test= sc_y.inverse_transform (y1)
    y_train= sc_y.inverse_transform (y)

    a= pd.DataFrame(a)    
    y_test= pd.DataFrame(y_test)    

    #summarize the fit of the model
    mape=numpy.mean((numpy.abs(y_test-a))/cap)*100
    rmse= sqrt(mean_squared_error(y_test,a))
    mae=metrics.mean_absolute_error(y_test,a)

    
    print('MAPE',mape.to_string())
    print('RMSE',rmse)
    print('MAE',mae)


# In[ ]:


##HYBRID EMD ENN

def emd_enn(new_data,i,look_back,data_partition,cap):

    x=i
    data1=new_data.loc[new_data['month'].isin(x)]
    data1=data1.reset_index(drop=True)
    data1=data1.dropna()
    
    datas=data1['LV ActivePower (kW)']
    datas_wind=pd.DataFrame(datas)
    dfs=datas
    s = dfs.values

    from PyEMD import EMD
    import ewtpy
    
    emd = EMD()

    IMFs = emd(s)

    full_imf=pd.DataFrame(IMFs)
    data_decomp=full_imf.T
    


    pred_test=[]
    test_ori=[]
    pred_train=[]
    train_ori=[]

    epoch=100
    batch_size=64
    neuron=128
    lr=0.001
    optimizer='Adam'

    for col in data_decomp:

        datasetss2=pd.DataFrame(data_decomp[col])
        datasets=datasetss2.values
        train_size = int(len(datasets) * data_partition)
        test_size = len(datasets) - train_size
        train, test = datasets[0:train_size], datasets[train_size:len(datasets)]

        trainX, trainY = create_dataset(train, look_back)
        testX, testY = create_dataset(test, look_back)
        X_train=pd.DataFrame(trainX)
        Y_train=pd.DataFrame(trainY)
        X_test=pd.DataFrame(testX)
        Y_test=pd.DataFrame(testY)
        sc_X = StandardScaler()
        sc_y = StandardScaler()
        X= sc_X.fit_transform(X_train)
        y= sc_y.fit_transform(Y_train)
        X1= sc_X.fit_transform(X_test)
        y1= sc_y.fit_transform(Y_test)
        y=y.ravel()
        y1=y1.ravel()  

        import numpy
        trainX = numpy.reshape(X, (X.shape[0],X.shape[1],1))
        testX = numpy.reshape(X1, (X1.shape[0],X1.shape[1],1))
    
        

        numpy.random.seed(1234)
        import tensorflow as tf
        tf.random.set_seed(1234)

    
        import os 
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        from keras.models import Sequential
        from keras.layers.core import Dense, Dropout, Activation
        from keras.layers import LSTM, Dense,SimpleRNN
        from keras.callbacks import EarlyStopping



        model = Sequential()
        model.add(SimpleRNN(units=neuron,input_shape=(trainX.shape[1], trainX.shape[2])))
        model.add(Dense(1))
        model.compile(loss='mse')

        model.fit(trainX, y, epochs = epoch,batch_size = batch_size,verbose=0)

        # make predictions
        y_pred_train = model.predict(trainX)
        y_pred_test = model.predict(testX)

        # make predictions

        y_pred_test= numpy.array(y_pred_test).ravel()
        y_pred_test=pd.DataFrame(y_pred_test)
        y1=pd.DataFrame(y1)
        y=pd.DataFrame(y)
        y_pred_train= numpy.array(y_pred_train).ravel()
        y_pred_train=pd.DataFrame(y_pred_train)

        y_test= sc_y.inverse_transform (y1)
        y_train= sc_y.inverse_transform (y)

        y_pred_test1= sc_y.inverse_transform (y_pred_test)
        y_pred_train1= sc_y.inverse_transform (y_pred_train)


        pred_test.append(y_pred_test1)
        test_ori.append(y_test)
        pred_train.append(y_pred_train1)
        train_ori.append(y_train)


    result_pred_test= pd.DataFrame.from_records(pred_test)
    result_pred_train= pd.DataFrame.from_records(pred_train)


    a=result_pred_test.sum(axis = 0, skipna = True) 
    b=result_pred_train.sum(axis = 0, skipna = True) 


    dataframe=pd.DataFrame(dfs)
    dataset=dataframe.values

    train_size = int(len(dataset) * data_partition)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size], dataset[train_size:len(dataset)]

    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    X_train=pd.DataFrame(trainX)
    Y_train=pd.DataFrame(trainY)
    X_test=pd.DataFrame(testX)
    Y_test=pd.DataFrame(testY)

    sc_X = StandardScaler()
    sc_y = StandardScaler() 
    X= sc_X.fit_transform(X_train)
    y= sc_y.fit_transform(Y_train)
    X1= sc_X.fit_transform(X_test)
    y1= sc_y.fit_transform(Y_test)
    y=y.ravel()
    y1=y1.ravel()


    trainX = numpy.reshape(X, (X.shape[0], 1, X.shape[1]))
    testX = numpy.reshape(X1, (X1.shape[0], 1, X1.shape[1]))

    numpy.random.seed(1234)
    tf.random.set_seed(1234)
    
    y1=pd.DataFrame(y1)
    y=pd.DataFrame(y)

    y_test= sc_y.inverse_transform (y1)
    y_train= sc_y.inverse_transform (y)

    a= pd.DataFrame(a)    
    y_test= pd.DataFrame(y_test)    

    #summarize the fit of the model
    mape=numpy.mean((numpy.abs(y_test-a))/cap)*100
    rmse= sqrt(mean_squared_error(y_test,a))
    mae=metrics.mean_absolute_error(y_test,a)

    
    print('MAPE',mape.to_string())
    print('RMSE',rmse)
    print('MAE',mae)


# In[ ]:




