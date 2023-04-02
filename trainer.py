import tensorflow as tf

def obtener_datos_entrenamiento():
    # Leer los datos de entrenamiento de un archivo CSV
    datos = pd.read_csv('datos_entrenamiento.csv')
    
    # Devolver los datos como un diccionario con dos claves: 'texto' y 'cypher'
    return {'texto': datos['texto'].tolist(), 'cypher': datos['cypher'].tolist()}

def obtener_datos_prueba():
    # Leer los datos de prueba de un archivo CSV
    datos = pd.read_csv('datos_prueba.csv')
    
    # Devolver los datos como un diccionario con dos claves: 'texto' y 'cypher'
    return {'texto': datos['texto'].tolist(), 'cypher': datos['cypher'].tolist()}

# Obtener los datos de entrenamiento y prueba
datos_entrenamiento = obtener_datos_entrenamiento()
datos_prueba = obtener_datos_prueba()

# Preprocesar los datos
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(datos_entrenamiento['texto'])
datos_entrenamiento_tokens = tokenizer.texts_to_sequences(datos_entrenamiento['texto'])
datos_prueba_tokens = tokenizer.texts_to_sequences(datos_prueba['texto'])

# Añadir padding a las secuencias de tokens
MAXLEN = 50
datos_entrenamiento_tokens_padded = tf.keras.preprocessing.sequence.pad_sequences(datos_entrenamiento_tokens, maxlen=MAXLEN, padding='post')
datos_prueba_tokens_padded = tf.keras.preprocessing.sequence.pad_sequences(datos_prueba_tokens, maxlen=MAXLEN, padding='post')

# Definir el modelo
model = tf.keras.models.Sequential([
  tf.keras.layers.Embedding(len(tokenizer.word_index) + 1, 128, input_length=MAXLEN),
  tf.keras.layers.LSTM(128),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(len(tokenizer.word_index) + 1, activation='softmax')
])

# Compilar el modelo
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Entrenar el modelo
model.fit(datos_entrenamiento_tokens_padded, datos_entrenamiento['cypher'], epochs=10, validation_data=(datos_prueba_tokens_padded, datos_prueba['cypher']))

# Evaluar el rendimiento del modelo
metricas = model.evaluate(datos_prueba_tokens_padded, datos_prueba['cypher'])
print('Precisión:', metricas[1])

