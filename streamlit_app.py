
import streamlit as st
import pandas as pd
import io
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
import pmdarima as pm

# --- Configuración de la página ---
st.set_page_config(
    page_title="HE360° - Plataforma de Gestión Empresarial",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📈 Plataforma Horizonte Empresarial 360°")

# --- CONSTANTES GLOBALES (replicadas del notebook original) ---

# Financiero
UMBRAL_LIQUIDEZ_MINIMA = 0.0      # $0 COP, límite de seguridad para Alertas de Liquidez.
TASA_IVA_GENERAL = 0.19           # 19% IVA General en Colombia.
TASA_IMPUESTO_RENTA = 0.35        # 35% Impuesto de Renta para sociedades.
TASA_RETENCION_FUENTE = 0.025     # 2.5% Tasa de Retención en la Fuente Estándar (sugerida).
TASA_ROTACION_CARTERA_DIAS = 30   # 30 días, estándar para KPI de Días de Cartera.
HORIZONTE_PROYECCION = 12 # Meses a proyectar para Prophet.
NUM_SIMULACIONES = 5000   # Número de escenarios para Monte Carlo.

# Sostenibilidad
FE_ELECT = 0.112      # KgCO2e/kWh (Factor de Emisión Eléctrica)
FE_ACPM = 3.179       # Kg CO2e/L (Factor de Emisión Diésel/ACPM)
LMP_DBO5 = 70.0       # mg/L (Límite Máximo Permitido para DBO5)
C_NAT = 2.0           # mg/L (Concentración Natural de DBO5)
IMPUESTO_CARBONO = 27399.14 # COP/tonelada CO2eq (Tarifa 2025)
COSTO_UNITARIO_AGUA = 3950 # COP/m3 (Aproximación para Acuamar)
COSTO_DISPOSICION_BASE = 380      # COP/Kg (Costo promedio por disposición final)
FACTOR_RECICLAJE_BASE = 0.20      # COP/Kg (Ingreso o ahorro por material reciclable)

# Riesgos y Cumplimiento
COSTO_PROMEDIO_INCIDENTE = 5000000        # COP
COSTO_PROMEDIO_NO_CONFORMIDAD = 2000000   # COP
COSTO_PROMEDIO_SANCION_LEGAL = 10000000   # COP
COSTO_POR_PUNTO_REDUCIDO = 5_000_000 # COP (inversión para reducir la gravedad en 1 punto)
FACTOR_AHORRO_POR_GRAVEDAD_PUNTO = 0.05 # Asumimos que 1 punto de gravedad_incidente reducido puede generar un 5% de ahorro en el costo anual total de incidentes.

# RRHH
COSTO_PROMEDIO_CONTRATACION = 3000000     # COP
COSTO_PROMEDIO_CAPACITACION = 1500000     # COP

# Comercial
PERIODO_FRECUENCIA_MONETARIA = '12 months'  # Define el período para cálculos monetarios (ej. 12 meses)
METRICA_MONETARIA_BASE = 'Margen_Generado' # Métrica base para análisis monetario (ej. Margen Generado)
UMBRAL_CLASE_A = 0.20  # Clientes/Productos que representan el 20% superior de la métrica
UMBRAL_CLASE_B = 0.50  # Clientes/Productos que representan el siguiente 30% (hasta el 50% acumulado)
FRECUENCIA_AGREGACION_BASE = 'W' # Frecuencia de agregación para datos (ej. Semanal 'W', Mensual 'M', Diario 'D')

# Productividad
HORIZONTE_PREDICCION_MESES_PROD = 36 # 3 años para SARIMA de productividad

# Gobernanza
PONDERACIONES_MODULOS = {
    'Financiero': 0.25,
    'Sostenibilidad': 0.20,
    'Operativo_RRHH': 0.20,
    'Riesgos_Cumplimiento': 0.20,
    'Comercial': 0.15
}

# Verificar que las ponderaciones sumen 1.0
if sum(PONDERACIONES_MODULOS.values()) != 1.0:
    total_ponderaciones = sum(PONDERACIONES_MODULOS.values())
    PONDERACIONES_MODULOS = {k: v / total_ponderaciones for k, v in PONDERACIONES_MODULOS.items()}

KPIs_Comercial = {
    'clientes_activos': {'peso': 0.25, 'optimo': 50, 'peor': 10, 'sentido': 'mayor_es_mejor'},
    'AOV': {'peso': 0.25, 'optimo': 1000000.0, 'peor': 500000.0, 'sentido': 'mayor_es_mejor'},
    'frecuencia_compra_promedio': {'peso': 0.25, 'optimo': 30, 'peor': 10, 'sentido': 'mayor_es_mejor'},
    'cac_promedio': {'peso': 0.25, 'optimo': 100000.0, 'peor': 500000.0, 'sentido': 'menor_es_mejor'}
}
# Ajustar pesos de KPIs comerciales si no suman 1.0
if sum(kpi['peso'] for kpi in KPIs_Comercial.values()) != 1.0:
    total_pesos_kpi_comercial = sum(kpi['peso'] for kpi in KPIs_Comercial.values())
    for kpi_name, kpi_params in KPIs_Comercial.items():
        kpi_params['peso'] /= total_pesos_kpi_comercial

NORMALIZACION_RANGOS = {
    'Probabilidad_Fallo_Liquidez': {'optimo': 0.0, 'peor': 0.10, 'sentido': 'menor_es_mejor'}, # 0% óptimo, 10% peor
    'Crecimiento_OPEX_Proyectado': {'optimo': -5.0, 'peor': 20.0, 'sentido': 'menor_es_mejor'}, # -5% óptimo (reducción), 20% peor (alto crecimiento)
    'Emisiones_GEI_Proyectadas': {'optimo': 0.0, 'peor': 500.0, 'sentido': 'menor_es_mejor'}, # 0 tCO2e óptimo, 500 tCO2e peor
    'Costo_Ambiental_Proyectado': {'optimo': 0.0, 'peor': 50_000_000.0, 'sentido': 'menor_es_mejor'}, # 0 COP óptimo, 50M COP peor
    'Costo_Oculto_Total': {'optimo': 0.0, 'peor': 50_000_000.0, 'sentido': 'menor_es_mejor'}, # 0 COP óptimo, 50M COP peor
    'Tasa_Utilizacion': {'optimo': 100.0, 'peor': 50.0, 'sentido': 'mayor_es_mejor'}, # 100% óptimo, 50% peor
    'Probabilidad_Fallo_Legal': {'optimo': 0.0, 'peor': 1.0, 'sentido': 'menor_es_mejor'}, # 0 óptimo, 1 (100%) peor
    'Costo_Total_Riesgo_MC': {'optimo': 0.0, 'peor': 100_000_000.0, 'sentido': 'menor_es_mejor'} # 0 COP óptimo, 100M COP peor
}

def normalizar_kpi(valor, rango_optimo, rango_peor, sentido):
    if sentido == 'menor_es_mejor':
        if valor <= rango_optimo: return 100.0
        if valor >= rango_peor: return 0.0
        return 100.0 - ((valor - rango_optimo) / (rango_peor - rango_optimo)) * 100.0
    elif sentido == 'mayor_es_mejor':
        if valor >= rango_optimo: return 100.0
        if valor <= rango_peor: return 0.0
        return ((valor - rango_peor) / (rango_optimo - rango_peor)) * 100.0
    return 0.0


# --- Funciones de Carga de Datos Crudos ---
@st.cache_data
def load_raw_data(uploaded_file, file_type):
    if uploaded_file is not None:
        try:
            if file_type == "csv":
                df = pd.read_csv(uploaded_file)
            elif file_type == "xlsx":
                df = pd.read_excel(uploaded_file)
            elif file_type == "parquet":
                df = pd.read_parquet(uploaded_file)
            return df
        except Exception as e:
            st.error(f"Error al cargar el archivo: {e}")
            return None
    return None

# --- Función de Carga y Preprocesamiento de Datos ---
def load_and_preprocess_data(raw_dfs):
    processed_dfs = {}

    # Define default columns for each DataFrame if not loaded
    default_columns_map = {
        'df_fin': ['FECHA', 'Efectivo_Inicial', 'Cobros_Clientes', 'Pagos_Proveedores', 'Ventas_Gravadas', 'Gastos_Gravados', 'Base_Retencion', 'Gasto_Nomina'],
        'df_com': ['FECHA_TRANSACCION', 'Cliente_ID', 'Producto_ID', 'Volumen_Vendido', 'Precio_Unitario_Real', 'Costo_Venta_Unitario', 'Gasto_Marketing'],
        'df_sost': ['Fecha_Reporte', 'Ubicacion_Planta', 'ID_Camion', 'Consumo_Combustible_ACPM', 'Consumo_Electrico_kWh', 'Descarga_DBO5_Kg', 'Consumo_Agua_Total_m3', 'Kg_Residuos_Solidos', 'Costo_unitario_Electrico_COP_kWh', 'Costo_unitario_Combustible_COP_L', 'Costo_Multa_Ambiental', 'Incumplimientos_Vencidos', 'Volumen_Produccion_Kg', 'Indicador_Turismo'],
        'df_riesgos': ['FECHA', 'Planta_ID', 'Horas_Trabajadas_Periodo', 'Gravedad_Incidente', 'Dias_Perdidos', 'Tipo_Evento', 'Costo_Total_Evento', 'Norma_Cumplir', 'Estado_Checklist'],
        'df_rrhh': ['FECHA', 'ANIO', 'Costo_Total_Nomina', 'FTEs_Promedio', 'Tasa_Rotacion_Anual', 'Inversion_Capacitacion'],
        'df_prod_final': ['FECHA', 'Planta_ID', 'Horas_Disponibles', 'Horas_Operadas', 'Unidades_Producidas', 'Unidades_Defectuosas', 'Costo_Mano_Obra_Hora', 'Costo_Reproceso_Unit']
    }

    for df_key, df_raw in raw_dfs.items():
        if df_raw is None or df_raw.empty:
            processed_dfs[df_key] = pd.DataFrame(columns=default_columns_map.get(df_key, []))
            if processed_dfs[df_key].empty:
                processed_dfs[df_key] = pd.DataFrame()
        else:
            df_processed = df_raw.copy()

            # Apply initial preprocessing based on df_key
            if df_key == 'df_fin':
                df_processed['FECHA'] = pd.to_datetime(df_processed['FECHA'], errors='coerce')
                df_processed = df_processed.dropna(subset=['FECHA']).sort_values('FECHA').set_index('FECHA').resample('ME').last()
                df_processed = df_processed.dropna(how='all')
            elif df_key == 'df_com':
                df_processed['FECHA_TRANSACCION'] = pd.to_datetime(df_processed['FECHA_TRANSACCION'], errors='coerce')
            elif df_key == 'df_sost':
                df_processed['Fecha_Reporte'] = pd.to_datetime(df_processed['Fecha_Reporte'], errors='coerce')
            elif df_key == 'df_riesgos':
                df_processed['FECHA'] = pd.to_datetime(df_processed['FECHA'], errors='coerce')
            elif df_key == 'df_rrhh':
                df_processed['FECHA'] = pd.to_datetime(df_processed['FECHA'], errors='coerce')
            elif df_key == 'df_prod_final':
                df_processed['FECHA'] = pd.to_datetime(df_processed['FECHA'], errors='coerce')

            processed_dfs[df_key] = df_processed

    return processed_dfs

# --- Helper function for Prophet (from original notebook) ---
def fit_and_predict_prophet(series, horizon):
    """Aplica Prophet a una serie de tiempo y devuelve las predicciones."""
    df_prophet = series.reset_index().rename(columns={series.index.name: 'ds', series.name: 'y'})

    if df_prophet.empty or len(df_prophet) < 2: # Prophet requires at least 2 data points
        return pd.Series(0, index=pd.date_range(start=pd.Timestamp.now(), periods=horizon, freq='ME')), \
               pd.Series(0, index=pd.date_range(start=pd.Timestamp.now(), periods=horizon, freq='ME'))

    m = Prophet(daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=True)
    m.fit(df_prophet)

    future = m.make_future_dataframe(periods=horizon, freq='ME')
    forecast = m.predict(future)

    # Only need future predictions
    last_historical_date = df_prophet['ds'].max()
    forecast_future = forecast[forecast['ds'] > last_historical_date].set_index('ds')

    std_dev = (forecast_future['yhat_upper'] - forecast_future['yhat_lower']) / 2

    return forecast_future['yhat'], std_dev.rename('std_dev')

# --- Función para calcular todos los KPIs críticos ---

def calculate_all_kpis(processed_dfs):
    kpis = {}

    # --- Financiero ---
    df_fin = processed_dfs.get('df_fin')
    if df_fin is not None and not df_fin.empty:
        # Ensure df_fin_mensual is available for financial calculations
        df_fin_mensual = df_fin.copy()
        df_fin_mensual['FECHA'] = pd.to_datetime(df_fin_mensual.index)

        # KPI: Probabilidad de Fallo de Liquidez
        try:
            pred_cobros, std_cobros = fit_and_predict_prophet(df_fin_mensual['Cobros_Clientes'], HORIZONTE_PROYECCION)
            pred_pagos, _ = fit_and_predict_prophet(df_fin_mensual['Pagos_Proveedores'], HORIZONTE_PROYECCION)

            # Create df_proj with predictions
            df_proj = pd.DataFrame({
                'Cobros_Proyectados': pred_cobros.values,
                'Pagos_Proyectados': pred_pagos.values,
            }, index=pred_cobros.index)

            efectivo_inicial_actual = df_fin_mensual['Efectivo_Inicial'].iloc[-1] if not df_fin_mensual['Efectivo_Inicial'].empty else 0

            # Monte Carlo simulation logic
            num_sims = NUM_SIMULACIONES
            sigma_cobros = std_cobros.mean() # Using mean std_dev as proxy for volatility
            pagos_proyectados_media = pred_pagos.values

            resultados_acumulados = []
            for i in range(num_sims):
                cobros_simulados = pred_cobros.values + np.random.normal(0, sigma_cobros, size=HORIZONTE_PROYECCION)
                cobros_simulados[cobros_simulados < 0] = 0
                flujo_neto_simulado = cobros_simulados - pagos_proyectados_media
                flujo_acumulado_simulado = np.cumsum(flujo_neto_simulado) + efectivo_inicial_actual
                resultados_acumulados.append(flujo_acumulado_simulado)

            df_sims = pd.DataFrame(np.array(resultados_acumulados).T, index=pred_cobros.index)
            kpis['probabilidad_fallo_liquidez'] = (df_sims.min(axis=0) < UMBRAL_LIQUIDEZ_MINIMA).sum() / num_sims
        except Exception as e:
            st.warning(f"Error calculando probabilidad de fallo de liquidez: {e}")
            kpis['probabilidad_fallo_liquidez'] = 0.0

        # KPI: Crecimiento OPEX Proyectado
        try:
            if not pred_pagos.empty:
                primer_pago_proyectado = pred_pagos.iloc[0]
                ultimo_pago_proyectado = pred_pagos.iloc[-1]
                kpis['crecimiento_opex_proyectado'] = ((ultimo_pago_proyectado - primer_pago_proyectado) / primer_pago_proyectado) * 100
            else:
                kpis['crecimiento_opex_proyectado'] = 0.0
        except Exception as e:
            st.warning(f"Error calculando crecimiento OPEX proyectado: {e}")
            kpis['crecimiento_opex_proyectado'] = 0.0
    else:
        kpis['probabilidad_fallo_liquidez'] = 0.0
        kpis['crecimiento_opex_proyectado'] = 0.0

    # --- Sostenibilidad ---
    df_sost = processed_dfs.get('df_sost')
    if df_sost is not None and not df_sost.empty:
        # Replicate df_mensual creation from original notebook
        df_sost_copy = df_sost.copy()
        df_sost_copy['Fecha_Reporte'] = pd.to_datetime(df_sost_copy['Fecha_Reporte'])
        df_sost_copy = df_sost_copy.sort_values('Fecha_Reporte')
        df_sost_copy['Costo_unitario_Agua_COP_m3'] = COSTO_UNITARIO_AGUA # Fill with constant

        df_mensual = df_sost_copy.set_index('Fecha_Reporte').resample('ME').agg({
            'Consumo_Combustible_ACPM': 'sum',
            'Consumo_Electrico_kWh': 'sum',
            'Descarga_DBO5_Kg': 'sum',
            'Consumo_Agua_Total_m3': 'sum',
            'Kg_Residuos_Solidos': 'sum',
            'Volumen_Produccion_Kg': 'sum',
            'Costo_unitario_Electrico_COP_kWh': 'last',
            'Costo_unitario_Combustible_COP_L': 'last',
            'Costo_unitario_Agua_COP_m3': 'last',
            'Costo_Multa_Ambiental': 'sum',
            'Incumplimientos_Vencidos': 'sum',
        }).reset_index()
        df_mensual[['Costo_unitario_Electrico_COP_kWh', 'Costo_unitario_Combustible_COP_L', 'Costo_unitario_Agua_COP_m3']] = \
            df_mensual[['Costo_unitario_Electrico_COP_kWh', 'Costo_unitario_Combustible_COP_L', 'Costo_unitario_Agua_COP_m3']].fillna(method='ffill')
        df_mensual = df_mensual.dropna(subset=['Consumo_Electrico_kWh', 'Consumo_Agua_Total_m3'])

        # Huella de Carbono
        df_mensual['Huella_Carbono_Scope1_KgCO2e'] = df_mensual['Consumo_Combustible_ACPM'] * FE_ACPM
        df_mensual['Huella_Carbono_Scope2_KgCO2e'] = df_mensual['Consumo_Electrico_kWh'] * FE_ELECT
        df_mensual['Huella_Carbono_Total_KgCO2e'] = df_mensual['Huella_Carbono_Scope1_KgCO2e'] + df_mensual['Huella_Carbono_Scope2_KgCO2e']

        # Huella Hídrica
        df_mensual['Huella_Hidrica_Gris_m3'] = (df_mensual['Descarga_DBO5_Kg'] * 1000) / (LMP_DBO5 - C_NAT) if (LMP_DBO5 - C_NAT) != 0 else 0

        # KPI: Emisiones GEI Proyectadas (tCO2e)
        try:
            last_date_sost = df_mensual['Fecha_Reporte'].max()
            pred_emisiones, _ = fit_and_predict_prophet(df_mensual['Huella_Carbono_Total_KgCO2e'], HORIZONTE_PROYECCION)
            kpis['emisiones_gei_proyectadas'] = pred_emisiones.sum() / 1000 # Convert to tons
        except Exception as e:
            st.warning(f"Error calculando emisiones GEI proyectadas: {e}")
            kpis['emisiones_gei_proyectadas'] = 0.0

        # KPI: Costo Ambiental Proyectado (por Huella Hídrica)
        try:
            pred_huella_gris, _ = fit_and_predict_prophet(df_mensual['Huella_Hidrica_Gris_m3'], HORIZONTE_PROYECCION)
            # Use a proxy for multa/treatment cost if available, otherwise default
            if df_mensual['Costo_Multa_Ambiental'].sum() > 0 and df_mensual['Huella_Hidrica_Gris_m3'].sum() > 0:
                costo_multa_base_por_m3 = df_mensual['Costo_Multa_Ambiental'].sum() / df_mensual['Huella_Hidrica_Gris_m3'].sum()
                kpis['costo_ambiental_proyectado'] = pred_huella_gris.sum() * costo_multa_base_por_m3
            else:
                kpis['costo_ambiental_proyectado'] = 0.0 # Default if no historical cost data
        except Exception as e:
            st.warning(f"Error calculando costo ambiental proyectado: {e}")
            kpis['costo_ambiental_proyectado'] = 0.0
    else:
        kpis['emisiones_gei_proyectadas'] = 0.0
        kpis['costo_ambiental_proyectado'] = 0.0

    # --- Operativo / Productividad ---
    df_prod_final = processed_dfs.get('df_prod_final')
    if df_prod_final is not None and not df_prod_final.empty:
        df_prod_final['FECHA'] = pd.to_datetime(df_prod_final['FECHA'], errors='coerce')
        df_prod_final = df_prod_final.dropna(subset=['FECHA']).sort_values('FECHA').reset_index(drop=True)

        # Fill NaNs with 0 as in notebook
        columns_to_fill_na = ['Horas_Disponibles', 'Horas_Operadas', 'Unidades_Producidas', 'Unidades_Defectuosas', 'Costo_Mano_Obra_Hora', 'Costo_Reproceso_Unit']
        for col in columns_to_fill_na:
            if col in df_prod_final.columns: df_prod_final[col] = df_prod_final[col].fillna(0)

        # Calculate intermediate metrics
        df_prod_final['Tiempo_Muerto_Absoluto'] = df_prod_final['Horas_Disponibles'] - df_prod_final['Horas_Operadas']
        df_prod_final['Tiempo_Muerto_Absoluto'] = df_prod_final['Tiempo_Muerto_Absoluto'].clip(lower=0)
        df_prod_final['Tasa_de_Utilizacion'] = np.where(df_prod_final['Horas_Disponibles'] > 0, (df_prod_final['Horas_Operadas'] / df_prod_final['Horas_Disponibles']) * 100, 0)
        df_prod_final['Tasa_de_Calidad'] = np.where(df_prod_final['Unidades_Producidas'] > 0, (1 - (df_prod_final['Unidades_Defectuosas'] / df_prod_final['Unidades_Producidas'])) * 100, 100)
        df_prod_final['Costo_por_Tiempos_Muertos'] = df_prod_final['Tiempo_Muerto_Absoluto'] * df_prod_final['Costo_Mano_Obra_Hora']
        df_prod_final['Costo_por_Defectos_Reproceso'] = df_prod_final['Unidades_Defectuosas'] * df_prod_final['Costo_Reproceso_Unit']
        df_prod_final['Costo_Oculto_Total'] = df_prod_final['Costo_por_Tiempos_Muertos'] + df_prod_final['Costo_por_Defectos_Reproceso']

        # Aggregate to monthly
        df_prod_mensual = df_prod_final.set_index('FECHA').resample('ME').agg({
            'Costo_Oculto_Total': 'sum',
            'Tasa_de_Utilizacion': 'mean',
            'Tasa_de_Calidad': 'mean' # For potential KPI or related calc
        })
        df_prod_mensual['Costo_Oculto_Total'] = df_prod_mensual['Costo_Oculto_Total'].fillna(0)
        df_prod_mensual['Tasa_de_Utilizacion'] = df_prod_mensual['Tasa_de_Utilizacion'].fillna(df_prod_mensual['Tasa_de_Utilizacion'].mean() if not df_prod_mensual['Tasa_de_Utilizacion'].empty else 0)

        # KPI: Costo Oculto Total (proyectado con SARIMA)
        try:
            ts_costo_oculto = df_prod_mensual['Costo_Oculto_Total'].fillna(0)
            if not ts_costo_oculto.empty and len(ts_costo_oculto) > 24: # SARIMA needs sufficient data
                model_sarima_costo = pm.auto_arima(ts_costo_oculto, seasonal=True, m=12,
                                                   stepwise=True, suppress_warnings=True, trace=False,
                                                   error_action="ignore", D=1)
                forecast_costo, _ = model_sarima_costo.predict(n_periods=HORIZONTE_PREDICCION_MESES_PROD, return_conf_int=True, alpha=0.05)
                kpis['costo_oculto_total'] = np.sum(forecast_costo) # Sum of projected costs
            else:
                kpis['costo_oculto_total'] = df_prod_mensual['Costo_Oculto_Total'].mean() if not df_prod_mensual.empty else 0.0
        except Exception as e:
            st.warning(f"Error calculando costo oculto total proyectado: {e}")
            kpis['costo_oculto_total'] = 0.0

        # KPI: Tasa de Utilización (último periodo)
        try:
            kpis['tasa_utilizacion'] = df_prod_final['Tasa_de_Utilizacion'].iloc[-1] if not df_prod_final.empty else 0.0
        except Exception as e:
            st.warning(f"Error calculando tasa de utilización: {e}")
            kpis['tasa_utilizacion'] = 0.0
    else:
        kpis['costo_oculto_total'] = 0.0
        kpis['tasa_utilizacion'] = 0.0

    # --- Riesgos y Cumplimiento ---
    df_riesgos = processed_dfs.get('df_riesgos')
    if df_riesgos is not None and not df_riesgos.empty:
        df_riesgos_copy = df_riesgos.copy()
        df_riesgos_copy['FECHA'] = pd.to_datetime(df_riesgos_copy['FECHA'], errors='coerce')
        df_riesgos_copy = df_riesgos_copy.dropna(subset=['FECHA'])

        # KPI: Probabilidad de Fallo Legal
        try:
            kpis['probabilidad_fallo_legal'] = (df_riesgos_copy['Estado_Checklist'] == 'Vencido').any().astype(int) # 1 if any 'Vencido', 0 otherwise
            # More granular: (df_riesgos_copy['Incumplimientos_Vencidos'] > 0).mean() if column exists
            if 'Incumplimientos_Vencidos' in df_riesgos_copy.columns:
                 kpis['probabilidad_fallo_legal'] = (df_riesgos_copy['Incumplimientos_Vencidos'] > 0).mean()
            else:
                 kpis['probabilidad_fallo_legal'] = (df_riesgos_copy['Estado_Checklist'] == 'Vencido').mean()

        except Exception as e:
            st.warning(f"Error calculando probabilidad de fallo legal: {e}")
            kpis['probabilidad_fallo_legal'] = 0.0

        # KPI: Costo Total de Riesgo (Monte Carlo de incidentes)
        try:
            df_riesgos_mensual = df_riesgos_copy.set_index('FECHA').resample('ME').agg({
                'Costo_Total_Evento': 'sum',
                'Gravedad_Incidente': 'sum',
                'Horas_Trabajadas_Periodo': 'sum',
                'Dias_Perdidos': 'sum'
            }).reset_index()
            ts_costo_incidentes = df_riesgos_mensual.set_index('FECHA')['Costo_Total_Evento'].fillna(0)

            if not ts_costo_incidentes.empty and len(ts_costo_incidentes) > 24: # SARIMA needs sufficient data
                model_arima_full = pm.auto_arima(ts_costo_incidentes, seasonal=True, m=12,
                                                 stepwise=True, suppress_warnings=True, trace=False,
                                                 error_action="ignore", D=1)
                forecast_result, conf_int = model_arima_full.predict(n_periods=HORIZONTE_PROYECCION, return_conf_int=True, alpha=0.05)
                forecast_series = pd.Series(forecast_result)

                # Monte Carlo based on SARIMA forecast
                mean_forecast_mc = forecast_series.values
                std_historica_costos = ts_costo_incidentes.std()
                resultados_montecarlo = []
                for _ in range(NUM_SIMULACIONES):
                    costos_simulados_mensuales = np.random.normal(mean_forecast_mc, std_historica_costos, HORIZONTE_PROYECCION)
                    costos_simulados_mensuales[costos_simulados_mensuales < 0] = 0
                    resultados_montecarlo.append(np.sum(costos_simulados_mensuales))
                kpis['costo_total_riesgo_mc'] = np.mean(resultados_montecarlo)
            else:
                kpis['costo_total_riesgo_mc'] = df_riesgos_mensual['Costo_Total_Evento'].mean() if not df_riesgos_mensual.empty else 0.0

        except Exception as e:
            st.warning(f"Error calculando costo total de riesgo MC: {e}")
            kpis['costo_total_riesgo_mc'] = 0.0

    else:
        kpis['probabilidad_fallo_legal'] = 0.0
        kpis['costo_total_riesgo_mc'] = 0.0

    # --- Comercial ---
    df_com = processed_dfs.get('df_com')
    if df_com is not None and not df_com.empty:
        df_com_copy = df_com.copy()
        df_com_copy['FECHA_TRANSACCION'] = pd.to_datetime(df_com_copy['FECHA_TRANSACCION'], errors='coerce').dropna()
        df_com_copy['Margen_Generado'] = (df_com_copy['Precio_Unitario_Real'] - df_com_copy['Costo_Venta_Unitario']) * df_com_copy['Volumen_Vendido']
        df_com_copy['Margen_Generado'] = df_com_copy['Margen_Generado'].replace([np.inf, -np.inf], np.nan).fillna(0)

        # KPI: Clientes Activos (en el último mes de datos)
        try:
            last_month = df_com_copy['FECHA_TRANSACCION'].max().to_period('M')
            df_com_last_month = df_com_copy[df_com_copy['FECHA_TRANSACCION'].dt.to_period('M') == last_month]
            kpis['clientes_activos'] = df_com_last_month['Cliente_ID'].nunique()
        except Exception as e:
            st.warning(f"Error calculando clientes activos: {e}")
            kpis['clientes_activos'] = 0

        # KPI: Valor Promedio de Transacción (AOV)
        try:
            kpis['aov'] = df_com_copy['Margen_Generado'].mean()
        except Exception as e:
            st.warning(f"Error calculando AOV: {e}")
            kpis['aov'] = 0.0

        # KPI: Frecuencia de Compra Promedio
        try:
            transacciones_por_cliente = df_com_copy.groupby('Cliente_ID').size()
            kpis['frecuencia_compra_promedio'] = transacciones_por_cliente.mean()
        except Exception as e:
            st.warning(f"Error calculando frecuencia de compra promedio: {e}")
            kpis['frecuencia_compra_promedio'] = 0.0

        # KPI: Costo de Adquisición de Cliente (CAC)
        try:
            df_com_monthly = df_com_copy.groupby(df_com_copy['FECHA_TRANSACCION'].dt.to_period('M')).agg(
                total_gasto_marketing=('Gasto_Marketing', 'sum'),
                num_clientes_unicos=('Cliente_ID', 'nunique')
            ).reset_index()
            df_com_monthly['CAC'] = df_com_monthly['total_gasto_marketing'] / df_com_monthly['num_clientes_unicos']
            df_com_monthly['CAC'] = df_com_monthly['CAC'].replace([np.inf, -np.inf], np.nan).fillna(0)
            kpis['cac_promedio'] = df_com_monthly['CAC'].mean()
        except Exception as e:
            st.warning(f"Error calculando CAC promedio: {e}")
            kpis['cac_promedio'] = 0.0

    else:
        kpis['clientes_activos'] = 0
        kpis['aov'] = 0.0
        kpis['frecuencia_compra_promedio'] = 0.0
        kpis['cac_promedio'] = 0.0

    # --- RRHH (No direct KPIs for Governance in original notebook, but derived values available for simulation) ---
    df_rrhh = processed_dfs.get('df_rrhh')
    if df_rrhh is not None and not df_rrhh.empty:
        df_rrhh_copy = df_rrhh.copy()
        df_rrhh_copy['FECHA'] = pd.to_datetime(df_rrhh_copy['FECHA'], errors='coerce')
        df_rrhh_copy = df_rrhh_copy.dropna(subset=['FECHA'])

        # Simulate impact of a reduction in Tasa_Rotacion_Anual for governance context
        # Use existing calculation from notebook 7a6b71b7 for values
        try:
            base_tasa_rotacion_anual = df_rrhh_copy['Tasa_Rotacion_Anual'].mean()
            base_ftes_promedio = df_rrhh_copy['FTEs_Promedio'].mean()
            base_empleados_rotados = (base_ftes_promedio * base_tasa_rotacion_anual) / 100
            costo_rotacion_base = base_empleados_rotados * (COSTO_PROMEDIO_CONTRATACION + COSTO_PROMEDIO_CAPACITACION)
            kpis['costo_rotacion_rrhh_anual'] = costo_rotacion_base
        except Exception as e:
            st.warning(f"Error calculando costo de rotación RRHH: {e}")
            kpis['costo_rotacion_rrhh_anual'] = 0.0
    else:
        kpis['costo_rotacion_rrhh_anual'] = 0.0

    return kpis

# --- Inicialización de st.session_state para raw DataFrames ---
# 'dfs' will hold the raw (unpreprocessed) dataframes as they are uploaded.
if 'dfs' not in st.session_state:
    st.session_state['dfs'] = {
        'df_fin': None,
        'df_com': None,
        'df_sost': None,
        'df_riesgos': None,
        'df_rrhh': None,
        'df_prod_final': None
    }

# --- Barra Lateral para Carga de Datos ---
st.sidebar.header("⚙️ Carga de Datos por Módulo")

module_info = {
    "Financiero": 'df_fin',
    "Comercial": 'df_com',
    "Sostenibilidad": 'df_sost',
    "Riesgos_HSEQ": 'df_riesgos',
    "RRHH": 'df_rrhh',
    "Productividad": 'df_prod_final',
}

for module_name, df_key in module_info.items():
    st.sidebar.subheader(f"Módulo {module_name}")
    uploaded_file = st.sidebar.file_uploader(f"Cargar datos para {module_name} (CSV, XLSX, Parquet)", type=["csv", "xlsx", "parquet"], key=f"{df_key}_uploader")

    if uploaded_file:
        file_type = uploaded_file.name.split('.')[-1]
        st.session_state['dfs'][df_key] = load_raw_data(uploaded_file, file_type)
        if st.session_state['dfs'][df_key] is not None:
            st.sidebar.success(f"Datos de {module_name} cargados.")
        else:
            st.sidebar.error(f"Error al cargar el archivo de {module_name}.")
    else:
        if f'{df_key}_uploader' in st.session_state and st.session_state[f'{df_key}_uploader'] is None and st.session_state['dfs'][df_key] is not None:
            st.session_state['dfs'][df_key] = None

# --- Procesar datos una vez que se han cargado ---
# This will run once when the app starts or when data is uploaded
# and store the processed dataframes in st.session_state for global access.
if 'processed_dfs' not in st.session_state:
    st.session_state['processed_dfs'] = load_and_preprocess_data(st.session_state['dfs'])
    st.session_state['kpis'] = calculate_all_kpis(st.session_state['processed_dfs'])
else:
    # Re-run preprocessing and KPI calculation if raw data changes
    current_raw_dfs_checksum = hash(frozenset([(k, df.shape if df is not None else None) for k, df in st.session_state['dfs'].items()]))
    if 'last_raw_dfs_checksum' not in st.session_state or st.session_state['last_raw_dfs_checksum'] != current_raw_dfs_checksum:
        st.session_state['processed_dfs'] = load_and_preprocess_data(st.session_state['dfs'])
        st.session_state['kpis'] = calculate_all_kpis(st.session_state['processed_dfs'])
        st.session_state['last_raw_dfs_checksum'] = current_raw_dfs_checksum


# --- Contenido Principal de la Aplicación con Pestañas ---
st.header("Módulos Analíticos")

tab_names = ["General", "Financiero", "Comercial", "Sostenibilidad", "Riesgos y Cumplimiento", "RRHH", "Productividad"]
tabs = st.tabs(tab_names)

with tabs[0]: # General Tab
    st.subheader("Visión General (Módulo de Gobernanza)")
    st.write("Aquí se mostrará un resumen ejecutivo y el Health Score Global de la empresa.")

    # Display KPI values from session_state
    st.subheader("Valores de KPIs Calculados:")
    if 'kpis' in st.session_state:
        for kpi_name, kpi_value in st.session_state['kpis'].items():
            st.write(f"- **{kpi_name}**: {kpi_value:,.2f}")
    else:
        st.info("Los KPIs aún no se han calculado. Cargue los datos para proceder.")

    # Display confirmation of processed data for verification
    st.subheader("Estado de los Datos Procesados:")
    for df_key, df_processed in st.session_state['processed_dfs'].items():
        if df_processed is not None and not df_processed.empty:
            st.success(f"✅ Datos procesados de **{df_key}** listos (Filas: {df_processed.shape[0]}, Columnas: {df_processed.shape[1]})者に表示されます。")
        else:
            st.warning(f"⚠️ No hay datos procesados o están vacíos para **{df_key}**.")



with tabs[1]: # Financiero Tab
    st.subheader("Módulo Financiero")
    if st.session_state['processed_dfs']['df_fin'] is not None and not st.session_state['processed_dfs']['df_fin'].empty:
        st.success("✅ Datos financieros procesados listos para análisis.")
        st.write("Aquí se integrarán las funciones del Módulo Financiero.")
        st.dataframe(st.session_state['processed_dfs']['df_fin'].head())
    else:
        st.info("Cargue el archivo de datos financieros para habilitar este módulo.")

with tabs[2]: # Comercial Tab
    st.subheader("Módulo Comercial")
    if st.session_state['processed_dfs']['df_com'] is not None and not st.session_state['processed_dfs']['df_com'].empty:
        st.success("✅ Datos comerciales procesados listos para análisis.")
        st.write("Aquí se integrarán las funciones del Módulo Comercial.")
        st.dataframe(st.session_state['processed_dfs']['df_com'].head())
    else:
        st.info("Cargue el archivo de datos comerciales para habilitar este módulo.")

with tabs[3]: # Sostenibilidad Tab
    st.subheader("Módulo Sostenibilidad")
    if st.session_state['processed_dfs']['df_sost'] is not None and not st.session_state['processed_dfs']['df_sost'].empty:
        st.success("✅ Datos de sostenibilidad procesados listos para análisis.")
        st.write("Aquí se integrarán las funciones del Módulo Sostenibilidad.")
        st.dataframe(st.session_state['processed_dfs']['df_sost'].head())
    else:
        st.info("Cargue el archivo de datos de sostenibilidad para habilitar este módulo.")

with tabs[4]: # Riesgos y Cumplimiento Tab
    st.subheader("Módulo Riesgos y Cumplimiento")
    if st.session_state['processed_dfs']['df_riesgos'] is not None and not st.session_state['processed_dfs']['df_riesgos'].empty:
        st.success("✅ Datos de riesgos HSEQ procesados listos para análisis.")
        st.write("Aquí se integrarán las funciones del Módulo Riesgos y Cumplimiento.")
        st.dataframe(st.session_state['processed_dfs']['df_riesgos'].head())
    else:
        st.info("Cargue el archivo de datos de riesgos HSEQ para habilitar este módulo.")

with tabs[5]: # RRHH Tab
    st.subheader("Módulo RRHH")
    if st.session_state['processed_dfs']['df_rrhh'] is not None and not st.session_state['processed_dfs']['df_rrhh'].empty:
        st.success("✅ Datos de RRHH procesados listos para análisis.")
        st.write("Aquí se integrarán las funciones del Módulo RRHH.")
        st.dataframe(st.session_state['processed_dfs']['df_rrhh'].head())
    else:
        st.info("Cargue el archivo de datos de RRHH para habilitar este módulo.")

with tabs[6]: # Productividad Tab
    st.subheader("Módulo Productividad")
    if st.session_state['processed_dfs']['df_prod_final'] is not None and not st.session_state['processed_dfs']['df_prod_final'].empty:
        st.success("✅ Datos de productividad procesados listos para análisis.")
        st.write("Aquí se integrarán las funciones del Módulo Productividad.")
        st.dataframe(st.session_state['processed_dfs']['df_prod_final'].head())
    else:
        st.info("Cargue el archivo de datos de productividad para habilitar este módulo.")
