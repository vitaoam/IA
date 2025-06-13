import os
import zipfile
import random
import shutil
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# Configurações
IMAGE_WIDTH = 150
IMAGE_HEIGHT = 150
IMAGE_CHANNELS = 3
BATCH_SIZE = 32
EPOCHS = 15

def download_dataset():
    """
    Download do dataset Dogs vs Cats da Microsoft
    """
    url = "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip"
    
    print("Baixando dataset...")
    dataset_path = tf.keras.utils.get_file(
        "kagglecatsanddogs_5340.zip",
        url,
        extract=True,
        cache_dir="."
    )
    
    print(f"Dataset baixado para: {dataset_path}")
    return os.path.dirname(dataset_path)

def create_dataset_structure():
    """
    Cria estrutura de pastas para o dataset
    """
    base_dir = 'cats_vs_dogs_dataset'
    
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
        
        # Criar diretórios de treino, validação e teste
        for split in ['train', 'validation', 'test']:
            split_dir = os.path.join(base_dir, split)
            os.mkdir(split_dir)
            
            # Criar subdiretórios para gatos e cachorros
            os.mkdir(os.path.join(split_dir, 'cats'))
            os.mkdir(os.path.join(split_dir, 'dogs'))
    
    return base_dir

def prepare_dataset(dataset_dir):
    """
    Prepara o dataset dividindo em treino, validação e teste
    """
    base_dir = create_dataset_structure()
    
    # Caminho para o diretório PetImages
    pet_images_dir = os.path.join(dataset_dir, 'PetImages')
    
    if not os.path.exists(pet_images_dir):
        print("Extraindo dataset...")
        zip_path = os.path.join(dataset_dir, 'kagglecatsanddogs_5340.zip')
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dataset_dir)
    
    # Diretórios de origem
    cats_source_dir = os.path.join(pet_images_dir, 'Cat')
    dogs_source_dir = os.path.join(pet_images_dir, 'Dog')
    
    # Filtrar imagens corrompidas
    def filter_images(source_dir):
        images = []
        for img in os.listdir(source_dir):
            try:
                img_path = os.path.join(source_dir, img)
                with open(img_path, 'rb') as f:
                    if b'JFIF' in f.peek(10):
                        images.append(img)
            except:
                continue
        return images
    
    cat_images = filter_images(cats_source_dir)
    dog_images = filter_images(dogs_source_dir)
    
    # Embaralhar imagens
    random.seed(42)
    random.shuffle(cat_images)
    random.shuffle(dog_images)
    
    # Dividir dataset (70% treino, 15% validação, 15% teste)
    def split_data(images):
        n_train = int(len(images) * 0.7)
        n_val = int(len(images) * 0.15)
        return {
            'train': images[:n_train],
            'validation': images[n_train:n_train+n_val],
            'test': images[n_train+n_val:]
        }
    
    cat_splits = split_data(cat_images)
    dog_splits = split_data(dog_images)
    
    # Copiar imagens para diretórios organizados
    for split in ['train', 'validation', 'test']:
        # Gatos
        for img in cat_splits[split]:
            src = os.path.join(cats_source_dir, img)
            dst = os.path.join(base_dir, split, 'cats', img)
            shutil.copyfile(src, dst)
        
        # Cachorros
        for img in dog_splits[split]:
            src = os.path.join(dogs_source_dir, img)
            dst = os.path.join(base_dir, split, 'dogs', img)
            shutil.copyfile(src, dst)
    
    print(f"Dataset preparado:")
    print(f"Treino: {len(cat_splits['train'])} gatos, {len(dog_splits['train'])} cachorros")
    print(f"Validação: {len(cat_splits['validation'])} gatos, {len(dog_splits['validation'])} cachorros")
    print(f"Teste: {len(cat_splits['test'])} gatos, {len(dog_splits['test'])} cachorros")
    
    return base_dir

def create_data_generators(base_dir):
    """
    Cria geradores de dados com aumento de dados
    """
    # Aumento de dados para treino
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Apenas normalização para validação e teste
    val_test_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        os.path.join(base_dir, 'train'),
        target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )
    
    validation_generator = val_test_datagen.flow_from_directory(
        os.path.join(base_dir, 'validation'),
        target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )
    
    test_generator = val_test_datagen.flow_from_directory(
        os.path.join(base_dir, 'test'),
        target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )
    
    return train_generator, validation_generator, test_generator

def build_model():
    """
    Constrói o modelo CNN
    """
    model = Sequential([
        # Primeiro bloco convolucional
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        
        # Segundo bloco convolucional
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        
        # Terceiro bloco convolucional
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        
        # Quarto bloco convolucional
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        
        # Camadas densas
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    print("Arquitetura do modelo:")
    model.summary()
    
    return model

def train_model(model, train_generator, validation_generator):
    """
    Treina o modelo
    """
    # Callbacks
    checkpoint = ModelCheckpoint(
        'melhor_modelo.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )
    
    print("Iniciando treinamento...")
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE,
        callbacks=[checkpoint, early_stopping]
    )
    
    return history

def evaluate_model(model, test_generator):
    """
    Avalia o modelo no conjunto de teste
    """
    print("Avaliando modelo...")
    test_loss, test_accuracy = model.evaluate(test_generator)
    
    # Predições para matriz de confusão
    predictions = model.predict(test_generator)
    y_pred = (predictions > 0.5).astype(int).flatten()
    y_true = test_generator.classes
    
    # Relatório de classificação
    print(f"\nAcurácia no teste: {test_accuracy:.4f}")
    print(f"Perda no teste: {test_loss:.4f}")
    
    print("\nRelatório de Classificação:")
    print(classification_report(y_true, y_pred, target_names=['Gato', 'Cachorro']))
    
    # Matriz de confusão
    cm = confusion_matrix(y_true, y_pred)
    print("\nMatriz de Confusão:")
    print(f"Gatos corretos: {cm[0][0]}, Gatos como cachorros: {cm[0][1]}")
    print(f"Cachorros como gatos: {cm[1][0]}, Cachorros corretos: {cm[1][1]}")
    
    return test_loss, test_accuracy

def plot_training_history(history):
    """
    Plota histórico de treinamento
    """
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Acurácia do Modelo')
    plt.xlabel('Época')
    plt.ylabel('Acurácia')
    plt.legend(['Treino', 'Validação'])
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Perda do Modelo')
    plt.xlabel('Época')
    plt.ylabel('Perda')
    plt.legend(['Treino', 'Validação'])
    
    plt.tight_layout()
    plt.savefig('historico_treinamento.png')
    plt.show()

def predict_new_image(model, image_path):
    """
    Faz predição em uma nova imagem
    """
    img = tf.keras.preprocessing.image.load_img(
        image_path, target_size=(IMAGE_HEIGHT, IMAGE_WIDTH)
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)
    confidence = prediction[0][0]
    
    if confidence > 0.5:
        return f"Cachorro (confiança: {confidence:.2%})"
    else:
        return f"Gato (confiança: {1-confidence:.2%})"

def main():
    """
    Função principal
    """
    print("=== CLASSIFICAÇÃO GATOS VS CACHORROS COM CNN ===\n")
    
    try:
        # 1. Download e preparação do dataset
        dataset_dir = download_dataset()
        base_dir = prepare_dataset(dataset_dir)
        
        # 2. Criação dos geradores de dados
        train_gen, val_gen, test_gen = create_data_generators(base_dir)
        
        # 3. Construção do modelo
        model = build_model()
        
        # 4. Treinamento
        history = train_model(model, train_gen, val_gen)
        
        # 5. Avaliação
        evaluate_model(model, test_gen)
        
        # 6. Plotar resultados
        plot_training_history(history)
        
    except Exception as e:
        print(f"Erro durante execução: {e}")

if __name__ == "__main__":
    main()
