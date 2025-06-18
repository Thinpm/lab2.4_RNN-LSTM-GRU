# Copy toàn bộ nội dung từ image_captioning_improved.py

# Thay đổi tên class và kiến trúc mô hình
class Config:
    def __init__(self):
        # ... giữ nguyên các thông số khác ...
        self.model_dir = "models/image_captioning_rnn"

    def define_model(self):
        """Định nghĩa kiến trúc mô hình với RNN"""
        # 1. Feature Extractor (input 1)
        inputs1 = Input(shape=(4096,))
        fe1 = Dropout(self.config.dropout_rate)(inputs1)
        fe2 = Dense(self.config.units, activation='relu')(fe1)
        fe3 = BatchNormalization()(fe2)
        
        # 2. Sequence Model (input 2) - Sử dụng SimpleRNN
        inputs2 = Input(shape=(self.config.max_length,))
        se1 = Embedding(self.vocab_size, self.config.embedding_dim, mask_zero=True)(inputs2)
        se2 = Dropout(self.config.dropout_rate)(se1)
        se3 = SimpleRNN(self.config.units, return_sequences=True)(se2)
        se4 = SimpleRNN(self.config.units)(se3)
        se5 = BatchNormalization()(se4)
        
        # 3. Decoder
        decoder1 = add([fe3, se5])
        decoder2 = Dense(self.config.units, activation='relu')(decoder1)
        decoder3 = Dropout(self.config.dropout_rate)(decoder2)
        decoder4 = BatchNormalization()(decoder3)
        outputs = Dense(self.vocab_size, activation='softmax')(decoder4)
        
        model = Model(inputs=[inputs1, inputs2], outputs=outputs)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        
        self.model = model
        return model 