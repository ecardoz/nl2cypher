from flask import Flask, request
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Cargar el modelo de lenguaje natural entrenado
model = tf.keras.models.load_model('modelo_de_lenguaje_natural.h5')

app = Flask(__name__)

# Definir una ruta que acepte una consulta en lenguaje natural como entrada
@app.route('/convertir_a_cypher', methods=['POST'])
def convertir_a_cypher():
    # Obtener la consulta en lenguaje natural del cuerpo de la solicitud
    consulta = request.json['consulta']

    # Preprocesar la consulta
    consulta_preprocesada = preprocesar_consulta(consulta)

    # Usar el modelo de lenguaje natural para generar la consulta en Cypher
    consulta_cypher = generar_consulta_cypher(consulta_preprocesada)

    # Devolver la consulta en Cypher como respuesta
    return {'consulta_cypher': consulta_cypher}

# Función para preprocesar la consulta en lenguaje natural
def preprocesar_consulta(consulta, tokenizer, max_length):
    """
    Preprocesa la consulta y la convierte en una representación numérica
    que puede ser procesada por Tensorflow.

    Args:
        consulta (str): La consulta en lenguaje natural.
        tokenizer (Tokenizer): El tokenizer que se usará para convertir la consulta
            en una representación numérica.
        max_length (int): La longitud máxima de la secuencia numérica de la consulta.

    Returns:
        numpy array: Una matriz de forma (1, max_length) que representa la consulta
        preprocesada.

    """
    # Tokenizar la consulta
    consulta_tokens = tokenizer.texts_to_sequences([consulta])
    
    # Paddear la secuencia numérica para que tenga longitud fija
    consulta_padded = pad_sequences(consulta_tokens, maxlen=max_length, padding='post', truncating='post')
    
    return np.array(consulta_padded)

# Función para generar la consulta en Cypher usando el modelo de lenguaje natural
def generar_consulta_cypher(modelo, preprocesador, consulta):
    """
    Genera una consulta Cypher a partir de la salida del modelo de Tensorflow.

    Args:
        modelo (tensorflow.keras.Model): El modelo que se utilizará para generar la consulta Cypher.
        preprocesador (callable): La función que se utilizará para preprocesar la consulta antes de enviarla
            al modelo.
        consulta (str): La consulta en lenguaje natural.

    Returns:
        str: Una cadena que representa la consulta Cypher generada por el modelo.

    """
    # Preprocesar la consulta
    consulta_procesada = preprocesador(consulta)
    
    # Generar la consulta Cypher a partir de la salida del modelo
    output = modelo.predict(consulta_procesada)
    output = np.argmax(output, axis=-1)
    cypher = ''.join([preprocesador.get_word_index()[w] for w in output[0] if w != 0])
    
    return cypher
