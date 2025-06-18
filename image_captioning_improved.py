import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, BatchNormalization
from tensorflow.keras.layers import add
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
import pickle
import time
from datetime import datetime

class Config:
    def __init__(self):
        # Thông số dữ liệu
        self.data_dir = "data"
        self.image_dir = os.path.join(self.data_dir, "image/Flicker8k_Dataset")
        self.caption_file = os.path.join(self.data_dir, "text/Flickr8k.token.vietnamese.txt")
        self.train_file = os.path.join(self.data_dir, "text/Flickr_8k.trainImages.txt")
        
        # Thông số cho việc thử nghiệm
        self.max_samples = 200  # Tăng số lượng mẫu
        self.max_length = 40   # Tăng độ dài tối đa của caption
        self.vocab_size = 8000  # Tăng kích thước từ điển
        
        # Thông số mô hình cải tiến
        self.embedding_dim = 300  # Tăng kích thước embedding
        self.units = 512  # Tăng số units
        self.batch_size = 64  # Tăng batch size
        self.epochs = 20  # Tăng số epochs
        self.dropout_rate = 0.5  # Tăng dropout
        
        # Thông số cho việc lưu mô hình
        self.model_dir = "models/image_captioning_improved"
        os.makedirs(self.model_dir, exist_ok=True)

class ImageCaptioningModel:
    def __init__(self, config):
        self.config = config
        self.tokenizer = None
        self.max_length = None
        self.vocab_size = None
        self.model = None
        
    def load_captions(self):
        """Đọc và xử lý caption từ file"""
        with open(self.config.train_file, 'r') as f:
            train_images = set(f.read().splitlines()[:self.config.max_samples])
        
        captions = {}
        with open(self.config.caption_file, 'r', encoding='utf-8') as f:
            for line in f:
                img_id, caption = line.strip().split('\t')
                img_name = img_id.split('#')[0]
                if img_name in train_images:
                    if img_name not in captions:
                        captions[img_name] = []
                    # Thêm start và end token
                    caption = 'startseq ' + caption + ' endseq'
                    captions[img_name].append(caption)
        
        return captions
    
    def create_tokenizer(self, captions):
        """Tạo và huấn luyện tokenizer"""
        all_captions = []
        for img_captions in captions.values():
            all_captions.extend(img_captions)
            
        tokenizer = Tokenizer(num_words=self.config.vocab_size, oov_token="<unk>")
        tokenizer.fit_on_texts(all_captions)
        
        # Lưu tokenizer
        with open(os.path.join(self.config.model_dir, 'tokenizer.pkl'), 'wb') as f:
            pickle.dump(tokenizer, f)
            
        self.tokenizer = tokenizer
        self.vocab_size = min(self.config.vocab_size, len(tokenizer.word_index) + 1)
        return tokenizer
    
    def load_image_features(self, image_names):
        """Trích xuất đặc trưng từ ảnh sử dụng VGG16"""
        base_model = VGG16(weights='imagenet')
        model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)
        
        features = {}
        total = len(image_names)
        for i, img_name in enumerate(image_names, 1):
            if i % 10 == 0:
                print(f'Processing image {i}/{total}')
                
            img_path = os.path.join(self.config.image_dir, img_name)
            img = load_img(img_path, target_size=(224, 224))
            img = img_to_array(img)
            img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
            img = tf.keras.applications.vgg16.preprocess_input(img)
            
            feature = model.predict(img, verbose=0)
            features[img_name] = feature
            
        return features
    
    def create_sequences(self, captions, features):
        """Tạo cặp dữ liệu (X, y) cho việc huấn luyện"""
        X1, X2, y = [], [], []
        
        for img_name, img_captions in captions.items():
            feature = features[img_name]
            
            for caption in img_captions:
                seq = self.tokenizer.texts_to_sequences([caption])[0]
                
                for i in range(1, len(seq)):
                    in_seq = seq[:i]
                    out_seq = seq[i]
                    
                    in_seq = pad_sequences([in_seq], maxlen=self.config.max_length)[0]
                    out_seq = tf.keras.utils.to_categorical([out_seq], num_classes=self.vocab_size)[0]
                    
                    X1.append(feature[0])
                    X2.append(in_seq)
                    y.append(out_seq)
        
        return np.array(X1), np.array(X2), np.array(y)
    
    def define_model(self):
        """Định nghĩa kiến trúc mô hình cải tiến"""
        # 1. Feature Extractor (input 1)
        inputs1 = Input(shape=(4096,))
        fe1 = Dropout(self.config.dropout_rate)(inputs1)
        fe2 = Dense(self.config.units, activation='relu')(fe1)
        fe3 = BatchNormalization()(fe2)
        
        # 2. Sequence Model (input 2)
        inputs2 = Input(shape=(self.config.max_length,))
        se1 = Embedding(self.vocab_size, self.config.embedding_dim, mask_zero=True)(inputs2)
        se2 = Dropout(self.config.dropout_rate)(se1)
        se3 = LSTM(self.config.units, return_sequences=True)(se2)
        se4 = LSTM(self.config.units)(se3)
        se5 = BatchNormalization()(se4)
        
        # 3. Decoder
        decoder1 = add([fe3, se5])
        decoder2 = Dense(self.config.units, activation='relu')(decoder1)
        decoder3 = Dropout(self.config.dropout_rate)(decoder2)
        decoder4 = BatchNormalization()(decoder3)
        outputs = Dense(self.vocab_size, activation='softmax')(decoder4)
        
        model = Model(inputs=[inputs1, inputs2], outputs=outputs)
        
        # Compile với learning rate thấp hơn
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        
        self.model = model
        return model
    
    def train(self):
        """Huấn luyện mô hình"""
        print("1. Loading data...")
        captions = self.load_captions()
        
        print("2. Creating tokenizer...")
        self.create_tokenizer(captions)
        
        print("3. Extracting image features...")
        features = self.load_image_features(list(captions.keys()))
        
        print("4. Creating sequences...")
        X1, X2, y = self.create_sequences(captions, features)
        
        # Chia tập validation
        indices = np.arange(len(X1))
        np.random.shuffle(indices)
        train_size = int(0.8 * len(indices))
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        X1_train, X1_val = X1[train_indices], X1[val_indices]
        X2_train, X2_val = X2[train_indices], X2[val_indices]
        y_train, y_val = y[train_indices], y[val_indices]
        
        print("5. Defining model...")
        self.define_model()
        
        print("6. Training model...")
        # Callbacks
        callbacks = [
            ModelCheckpoint(
                os.path.join(self.config.model_dir, 'model-ep{epoch:03d}-val_acc{val_accuracy:.3f}.keras'),
                monitor='val_accuracy',
                save_best_only=True,
                mode='max'
            ),
            EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_accuracy',
                factor=0.2,
                patience=3,
                min_lr=0.00001
            ),
            TensorBoard(
                log_dir=os.path.join(self.config.model_dir, 'logs'),
                histogram_freq=1
            )
        ]
        
        # Huấn luyện
        history = self.model.fit(
            [X1_train, X2_train],
            y_train,
            validation_data=([X1_val, X2_val], y_val),
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Lưu mô hình cuối cùng
        self.model.save(os.path.join(self.config.model_dir, 'final_model.keras'))
        
        # Lưu lịch sử huấn luyện
        with open(os.path.join(self.config.model_dir, 'training_history.pkl'), 'wb') as f:
            pickle.dump(history.history, f)
        
        return history

def main():
    # Khởi tạo config
    config = Config()
    
    # Tạo và huấn luyện mô hình
    model = ImageCaptioningModel(config)
    history = model.train()
    
    # In kết quả cuối cùng
    print("\nTraining Results:")
    print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")

if __name__ == "__main__":
    main() 