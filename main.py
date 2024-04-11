import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
from sklearn.metrics import r2_score


# Página 1
def pagina_1():
    # Función para leer el archivo CSV
    def leer_csv(archivo):
        try:
            df = pd.read_csv(archivo)
            return df
        except Exception as e:
            st.error(f"Error al leer el archivo CSV: {e}")

    # Carga del archivo CSV
    archivo_cargado = st.file_uploader("Elige un archivo CSV", type="csv")

    # Mostrar el contenido del archivo CSV si se ha cargado

    train_df = pd.read_csv("Kaggle_Training_Dataset_v3.csv", low_memory=False)

    columnas_a_eliminar = ['sku','potential_issue','pieces_past_due','perf_6_month_avg','perf_12_month_avg',
                           'local_bo_qty','deck_risk','oe_constraint','ppap_risk','stop_auto_buy','rev_stop','went_on_backorder']
    train_df = train_df.drop(columnas_a_eliminar, axis=1)
    train_df.dropna(inplace=True)
    train_df.drop(train_df[train_df['forecast_3_month'] == 0].index, inplace=True)
    from sklearn.linear_model import ElasticNet

    if archivo_cargado is not None:
        test_df = leer_csv(archivo_cargado)
        if test_df is not None:
            test_df = test_df.drop(columnas_a_eliminar, axis=1)
            test_df.dropna(inplace=True)
            test_df.drop(test_df[test_df['forecast_3_month'] == 0].index, inplace=True)
            st.write(test_df)
        target_columns = ['forecast_3_month', 'forecast_6_month', 'forecast_9_month']
        models = {}  # diccionario para almacenar los modelos de las 3 columnas de forecast
        predictions = {}  # diccionario para almacenar las predicciones de los modelos de las 3 columnas de forecast

        for column in target_columns:
               # Definir X e Y para el conjunto de entrenamiento
                x_train = train_df.drop([column], axis=1).values  # Variables predictoras
                y_train = train_df[column].values  # Variable objetivo
                # Definir X para el conjunto de prueba
                x_test = test_df.drop(column, axis=1).values
                y_test = test_df[column].values
                # Crear y entrenar el modelo
                model_rl = ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=10000)
                model_rl.fit(x_train, y_train)
                # Realizar predicciones para el conjunto de prueba
                y_pred = model_rl.predict(x_test)
                # Almacenar el modelo y las predicciones en los diccionarios
                models[column] = model_rl
                predictions[column] = y_pred
                # Almacenar el modelo en la caché de Streamlit
                st.session_state['model_rl'] = model_rl

        st.write("¡Modelo entrenado!")
            # Ejemplo: muestra las predicciones para 'forecast_3_month'
        st.write("Predicciones para los 3 , 6 y 9 meses")
        predictions = pd.DataFrame(predictions, columns=["forecast_3_month", "forecast_6_month", "forecast_9_month"])
        st.dataframe(predictions)  # Accede a la primera columna de las predicciones

        if "width" not in st.session_state:
            ancho_predeterminado = 800  # Puedes cambiar este valor por defecto
            st.session_state["width"] = int(ancho_predeterminado)
        ancho = int(st.session_state["width"])
        # Create a correlation matrix for the numeric columns
        corr_matrix = train_df.corr(method='spearman')  # You can use 'pearson' for Pearson correlation
        # Create a heatmap using plotly.express
        fig = px.imshow(corr_matrix,
                        title="Matriz de correlación",
                        color_continuous_scale="YlGnBu",
                        text_auto=True)  # Enables annotations
        # Display the heatmap using Streamlit
        fig.update_layout(width=ancho, height=ancho * 0.8)
        st.plotly_chart(fig)

        for column in target_columns:
            # Obtener las predicciones correspondientes para el intervalo actual
            y_pred_column = predictions[column]

            # Calcular la pendiente y el intercepto de la línea de regresión
            slope, intercept = np.polyfit(y_test, y_pred_column, 1)

            # Trazar el gráfico de dispersión y la línea de regresión
            fig, ax = plt.subplots(figsize=(15, 10))

            # Agregar puntos de predicción y valores reales
            ax.scatter(y_test, y_pred_column, color='blue', label='Predicción')
            ax.scatter(y_test, y_test, color='red', label='Actual')

            # Etiquetas y título
            ax.set_xlabel('Actual')
            ax.set_ylabel('Predicción')
            ax.set_title(f'Gráfico de Dispersión y Línea de Regresión para {column}')

            # Línea de regresión
            ax.plot(y_test, slope * y_test + intercept, color='red', linestyle='--', linewidth=2)

            # Mostrar la gráfica en Streamlit
            st.pyplot(fig)

        # Recorrer las columnas objetivo
        for column in target_columns:
            # Obtener las predicciones y calcular los residuos
            y_pred_column = predictions[column]
            residuos = y_test - y_pred_column

            # Crear la figura
            fig, ax = plt.subplots(figsize=(8, 6))

            # Agregar puntos y línea de referencia
            ax.scatter(y_pred_column, residuos, alpha=0.5)
            ax.axhline(y=0, color='r', linestyle='--', linewidth=2)

            # Etiquetas y título
            ax.set_xlabel('Predicciones')
            ax.set_ylabel('Residuos')
            ax.set_title(f'Gráfica de Residuos vs. Predicciones para {column}')

            # Mostrar la gráfica en Streamlit
            st.pyplot(fig)

def pagina_2():

    # Se recibe la imagen y el modelo, devuelve la predicción
    def model_prediction(input_Data):

        # changing the input_data to numpy array
        input_data_as_numpy_array = np.asarray(input_Data)

        # reshape the array as we are predicting for one instance
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
        # Cargar el modelo desde la caché de Streamlit
        model_rl = st.session_state['model_rl']
        prediction = model_rl.predict(input_data_reshaped)
        return prediction

    def main():

        # Título
        html_temp = """
        <h1 style="color:#181082;text-align:center;">SISTEMA DE PREDICCION DE COMPRA DE MATERIA PRIMA </h1>
        </div>
        """
        st.markdown(html_temp, unsafe_allow_html=True)

        # Lecctura de datos
        # Datos = st.text_input("Ingrese los valores : N P K Temp Hum pH lluvia:")
        dato1 = st.text_input('Inventario actual')
        if dato1.strip():  # Si la cadena tiene contenido (ignorando espacios en blanco)
            try:
                dato1_numeric = int(dato1)
                # Aquí puedes usar dato1_numeric en tus operaciones posteriores
            except ValueError:
                st.error("Por favor ingrese un valor numérico para 'Inventario actual'")
        else:
            st.error("Por favor ingrese un valor para 'Inventario actual'")
        dato2 = st.text_input('Tiempo de espera del producto (Horas)')
        if dato2.strip():  # Si la cadena tiene contenido (ignorando espacios en blanco)
            try:
                dato22 = int(dato2)
                # Aquí puedes usar dato1_numeric en tus operaciones posteriores
            except ValueError:
                st.error("Por favor ingrese un valor numérico para 'Tiempo de espera del producto'")
        else:
            st.error("Por favor ingrese un valor para 'Tiempo de espera del producto'")
        dato3 = st.text_input('Cantidad en transito')
        if dato3.strip():  # Si la cadena tiene contenido (ignorando espacios en blanco)
            try:
                dato33 = int(dato3)
                # Aquí puedes usar dato1_numeric en tus operaciones posteriores
            except ValueError:
                st.error("Por favor ingrese un valor numérico para 'Cantidad en transito'")
        else:
            st.error("Por favor ingrese un valor para 'Cantidad en transito'")
        dato4 = st.text_input('Ventas de 1 mes')
        if dato4.strip():  # Si la cadena tiene contenido (ignorando espacios en blanco)
            try:
                dato44 = int(dato4)
                # Aquí puedes usar dato1_numeric en tus operaciones posteriores
            except ValueError:
                st.error("Por favor ingrese un valor numérico para 'Ventas de 1 mes'")
        else:
            st.error("Por favor ingrese un valor para 'Ventas de 1 mes'")

        dato5 = st.text_input('Ventas de 3 meses')
        if dato5.strip():  # Si la cadena tiene contenido (ignorando espacios en blanco)
            try:
                dato55 = int(dato5)
                # Aquí puedes usar dato1_numeric en tus operaciones posteriores
            except ValueError:
                st.error("Por favor ingrese un valor numérico para 'Ventas de 3 meses'")
        else:
            st.error("Por favor ingrese un valor para 'Ventas de 3 meses'")
        dato6 = st.text_input('Ventas de 6 meses')
        if dato6.strip():  # Si la cadena tiene contenido (ignorando espacios en blanco)
            try:
                dato66 = int(dato6)
                # Aquí puedes usar dato1_numeric en tus operaciones posteriores
            except ValueError:
                st.error("Por favor ingrese un valor numérico para 'Ventas de 6 meses'")
        else:
            st.error("Por favor ingrese un valor para 'Ventas de 6 meses'")

        dato7 = st.text_input('Ventas de 9 meses')
        if dato7.strip():  # Si la cadena tiene contenido (ignorando espacios en blanco)
            try:
                dato77 = int(dato7)
                # Aquí puedes usar dato1_numeric en tus operaciones posteriores
            except ValueError:
                st.error("Por favor ingrese un valor numérico para 'Ventas de 9 meses'")
        else:
            st.error("Por favor ingrese un valor para 'Ventas de 9 meses'")
        dato8 = st.text_input('Inventario minimo')
        if dato8.strip():  # Si la cadena tiene contenido (ignorando espacios en blanco)
            try:
                dato88 = int(dato8)
                # Aquí puedes usar dato1_numeric en tus operaciones posteriores
            except ValueError:
                st.error("Por favor ingrese un valor numérico para 'Inventario minimo'")
        else:
            st.error("Por favor ingrese un valor para 'Inventario minimo'")

        # code for Prediction
        diagnosis = ''

        # creating a button for Prediction

        if st.button('Result'):
            diagnosis = model_prediction(
                [dato1_numeric,dato22,dato33,dato44,dato55,dato66,dato77,dato88])

            valor = str(diagnosis)
            print("Valor a pasar a st.success:", valor)  # Verificar el valor antes de pasarlo a st.success()
            st.success("Resultado exitoso")
            st.text("La predicción a 3 meses es:" + str(diagnosis[0][0]))  # Acceder al primer elemento de la primera fila
            st.text("La predicción a 6 meses es:" + str(diagnosis[0][1]))  # Acceder al segundo elemento de la primera fila
            st.text("La predicción a 9 meses es:" + str(diagnosis[0][2]))


opciones = ["Opción 1", "Opción 2"]

# Seleccionar la opción del menú
opcion_seleccionada = st.sidebar.selectbox("Menú", opciones)

# Mostrar la página correspondiente a la opción seleccionada
if opcion_seleccionada == "Opción 1":
  pagina_1()
elif opcion_seleccionada == "Opción 2":
  pagina_2()
else:
  st.error("Opción no válida")