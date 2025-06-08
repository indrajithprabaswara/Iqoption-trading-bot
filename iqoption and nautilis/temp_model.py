class ModelEnsemble:
    def __init__(self, input_dim, ensemble_size=3):
        self.models = []
        self.input_dim = input_dim
        self.ensemble_size = ensemble_size
        self.best_model = None
        self.best_accuracy = 0
        self.scaler = StandardScaler()
        
    def build_single_model(self):
        inputs = Input(shape=(self.input_dim, 1))
        
        # First LSTM layer
        x = LSTM(64, return_sequences=True, 
                kernel_regularizer=l2(0.001))(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        # Second LSTM layer
        x = LSTM(32)(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        # Dense layers
        x = Dense(32, activation='relu',
                 kernel_regularizer=l2(0.001))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.1)(x)
        
        # Output layer
        outputs = Dense(1, activation='sigmoid')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        # Use optimizer with learning rate scheduling
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.001,
            decay_steps=1000,
            decay_rate=0.9
        )
        
        optimizer = Adam(learning_rate=lr_schedule,
                       clipnorm=1.0,
                       clipvalue=0.5)
        
        model.compile(optimizer=optimizer,
                     loss='binary_crossentropy',
                     metrics=['accuracy'])
        
        return model
        
    def train_ensemble(self, X, y):
        print(f"Training ensemble with data shape: X={X.shape}, y={y.shape}")
        
        if len(X) != len(y):
            print(f"Error: X and y lengths don't match. X: {len(X)}, y: {len(y)}")
            return
            
        if len(X) < 20:
            print("Not enough data for training")
            return
            
        # Normalize features
        X = self.scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        
        # Configure cross-validation
        n_splits = min(3, len(X) // 20)
        if n_splits < 2:
            n_splits = 2
            
        tscv = TimeSeriesSplit(n_splits=n_splits, gap=2)
        
        # Train ensemble
        for i in range(self.ensemble_size):
            model = self.build_single_model()
            accuracies = []
            
            try:
                for train_idx, val_idx in tscv.split(X):
                    if len(train_idx) < 10 or len(val_idx) < 10:
                        continue
                        
                    X_train = X[train_idx]
                    y_train = y[train_idx]
                    X_val = X[val_idx]
                    y_val = y[val_idx]
                    
                    # Early stopping callback
                    early_stopping = EarlyStopping(
                        monitor='val_accuracy',
                        patience=5,
                        restore_best_weights=True
                    )
                    
                    # Train the model
                    model.fit(
                        X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=50,
                        batch_size=32,
                        callbacks=[early_stopping],
                        verbose=1
                    )
                    
                    # Evaluate model
                    _, accuracy = model.evaluate(X_val, y_val, verbose=0)
                    accuracies.append(accuracy)
                
                if accuracies:
                    avg_accuracy = np.mean(accuracies)
                    print(f"Model {i+1} average accuracy: {avg_accuracy:.4f}")
                    self.models.append((model, avg_accuracy))
                    
                    if avg_accuracy > self.best_accuracy:
                        self.best_accuracy = avg_accuracy
                        self.best_model = model
                
            except Exception as e:
                print(f"Error training model {i+1}: {str(e)}")
                continue
        
        # Sort models by accuracy
        if self.models:
            self.models.sort(key=lambda x: x[1], reverse=True)
            print(f"Ensemble training complete. Best accuracy: {self.best_accuracy:.4f}")
        else:
            print("Warning: No models were successfully trained")
    
    def predict(self, X):
        if not self.models:
            print("No trained models available")
            return None, 0.0

        try:
            # Normalize input data
            X_norm = self.scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
            
            predictions = []
            weights = []
            confidences = []
            
            # Get predictions from top 3 models
            for i, (model, accuracy) in enumerate(self.models[:3]):
                pred = model.predict(X_norm, verbose=0)
                predictions.append(pred)
                weights.append(accuracy)
                
                # Calculate prediction confidence
                confidence = np.mean([abs(p - 0.5) * 2 for p in pred])
                confidences.append(confidence)
            
            # Combine predictions with weighted average
            combined_weights = [w * c for w, c in zip(weights, confidences)]
            weighted_pred = np.average(predictions, weights=combined_weights, axis=0)
            
            # Calculate overall confidence
            avg_confidence = np.mean(confidences)
            if avg_confidence < 0.6:
                print(f"Warning: Low prediction confidence ({avg_confidence:.2f})")
            
            return weighted_pred, avg_confidence
            
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            return None, 0.0
