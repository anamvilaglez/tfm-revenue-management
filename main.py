import functions_framework
import joblib
import pandas as pd
import numpy as np

# Cargar modelo y variables al iniciar la instancia
modelo = joblib.load('modelo_ocupacion.pkl')
variables = joblib.load('variables_modelo.pkl')

@functions_framework.http
def predecir(request):
    # 1. Leer los datos de la petición (enviados desde n8n)
    body = request.get_json(silent=True)
    if not body or 'data' not in body:
        return {'error': 'Falta el campo data'}, 400

    datos = body['data']
    if isinstance(datos, dict):
        datos = [datos]

    df_n8n = pd.DataFrame(datos)
    
    # Validar que n8n envía las columnas mínimas (AHORA EXIGE "Fecha")
    if 'Fecha' not in df_n8n.columns or 'ocupacion_actual' not in df_n8n.columns:
        return {'error': 'Los datos deben contener Fecha y ocupacion_actual'}, 400

    # 2. Formatear fechas y establecer el horizonte dinámico desde n8n
    df_n8n['Fecha'] = pd.to_datetime(df_n8n['Fecha'])
    hoy = pd.to_datetime("today").normalize()
    
    # Extraer el horizonte del JSON de n8n (por defecto 14 si no se envía)
    try:
        dias_horizonte = int(body.get('horizonte', 14))
    except ValueError:
        return {'error': 'El campo horizonte debe ser un número entero'}, 400

    fecha_fin = hoy + pd.Timedelta(days=dias_horizonte)

    # 3. Filtrar solo los días futuros dentro del horizonte
    mask = (df_n8n['Fecha'] > hoy) & (df_n8n['Fecha'] <= fecha_fin)
    df_calendario = df_n8n.loc[mask].copy()

    if df_calendario.empty:
        return {'resultados': [], 'mensaje': 'No hay fechas en el horizonte proyectado'}

    # Ordenar cronológicamente
    df_calendario = df_calendario.sort_values('Fecha')

    # 4. Calcular días vista dinámicamente según la fecha de hoy
    df_calendario['dias_vista'] = (df_calendario['Fecha'] - hoy).dt.days

    # 5. Rellenar columnas faltantes exigidas por el modelo
    for col in variables:
        if col not in df_calendario.columns:
            df_calendario[col] = 0

    # 6. Ejecutar el modelo XGBoost para predecir el pickup
    X_predict = df_calendario[variables]
    pred_pickup = modelo.predict(X_predict)

    # 7. Calcular resultados finales asegurando límites lógicos
    df_calendario['pickup_predicho'] = np.maximum(0, pred_pickup) 
    df_calendario['ocup_final_predicha'] = df_calendario['ocupacion_actual'] + df_calendario['pickup_predicho']
    df_calendario['ocup_final_predicha'] = np.minimum(100, df_calendario['ocup_final_predicha'])

    # 8. Dar formato a la salida para n8n
    df_calendario['Fecha'] = df_calendario['Fecha'].dt.strftime('%Y-%m-%d')
    columnas_salida = ['Fecha', 'dias_vista', 'ocupacion_actual', 'pickup_predicho', 'ocup_final_predicha']
    df_salida = df_calendario[columnas_salida]

    # 9. Retornar el JSON estructurado
    return {
        'resultados': df_salida.to_dict(orient='records'),
        'n_filas': len(df_salida),
        'horizonte_aplicado': dias_horizonte
    }