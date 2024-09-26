import pandas as pd
import tensorflow as tf
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
from sklearn.model_selection import StratifiedShuffleSplit

class DataProcessor:
    def __init__(self, csv_file_path):
        self.df = pd.read_csv(csv_file_path)
        self.label_map = {'positive': 2, 'neutral': 1, 'negative': 0}
        self.df['Pseudo_Labels'] = self.df['Pseudo_Labels'].map(self.label_map)
    
    def split_data(self):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for train_index, test_index in sss.split(self.df, self.df['Pseudo_Labels']):
            self.train_df = self.df.iloc[train_index]
            self.test_df = self.df.iloc[test_index]
        print(f"Train size: {len(self.train_df)}, Test size: {len(self.test_df)}")
        
        sss_val = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for train_index, val_index in sss_val.split(self.train_df, self.train_df['Pseudo_Labels']):
            self.train_split_df = self.train_df.iloc[train_index]
            self.val_df = self.train_df.iloc[val_index]
        print(f"Training size: {len(self.train_split_df)}, Validation size: {len(self.val_df)}")

        return self.train_split_df, self.val_df, self.test_df

class Tokenizer:
    def __init__(self):
        self.tokenizer = DistilBertTokenizer.from_pretrained('artifacts/model')
    
    def tokenize_data(self, data):
        return self.tokenizer(data, padding=True, truncation=True, return_tensors="tf", clean_up_tokenization_spaces=False)

class DataLoader:
    def __init__(self, train_df, val_df, test_df, tokenizer):
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.tokenizer = tokenizer

    def create_datasets(self, batch_size=16):
        train_encodings = self.tokenizer.tokenize_data(self.train_df['Cleaned_Reviews'].tolist())
        val_encodings = self.tokenizer.tokenize_data(self.val_df['Cleaned_Reviews'].tolist())
        test_encodings = self.tokenizer.tokenize_data(self.test_df['Cleaned_Reviews'].tolist())

        train_dataset = tf.data.Dataset.from_tensor_slices((
            dict(train_encodings),
            self.train_df['Pseudo_Labels'].values
        )).shuffle(10000).batch(batch_size)

        val_dataset = tf.data.Dataset.from_tensor_slices((
            dict(val_encodings),
            self.val_df['Pseudo_Labels'].values
        )).batch(batch_size)

        test_dataset = tf.data.Dataset.from_tensor_slices((
            dict(test_encodings),
            self.test_df['Pseudo_Labels'].values
        )).batch(batch_size)

        return train_dataset, val_dataset, test_dataset

class ModelTrainer:
    def __init__(self, num_labels=3, learning_rate=5e-5):
        self.model = TFDistilBertForSequenceClassification.from_pretrained('artifacts/model', num_labels=num_labels)
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )
        self.early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

    def train(self, train_dataset, val_dataset, epochs=10):
        history = self.model.fit(train_dataset, validation_data=val_dataset, epochs=epochs, callbacks=[self.early_stopping])
        return history

    def evaluate(self, test_dataset):
        eval_loss, eval_acc = self.model.evaluate(test_dataset)
        print(f"Test Accuracy: {eval_acc}")
        return eval_acc

    def save_model(self, model_dir='artifacts/model/'):
        self.model.save_pretrained(model_dir)

class Experiment:
    def __init__(self, csv_file_path):
        self.data_processor = DataProcessor(csv_file_path)
        self.tokenizer = Tokenizer()
    
    def run(self):
        # Step 1: Data Splitting
        train_df, val_df, test_df = self.data_processor.split_data()
        
        # Step 2: Tokenization
        data_loader = DataLoader(train_df, val_df, test_df, self.tokenizer)
        train_dataset, val_dataset, test_dataset = data_loader.create_datasets()

        # Step 3: Model Training
        model_trainer = ModelTrainer()
        model_trainer.train(train_dataset, val_dataset)

        # Step 4: Model Evaluation
        model_trainer.evaluate(test_dataset)

        # Step 5: Save Model
        model_trainer.save_model()

if __name__ == '__main__':
    experiment = Experiment('artifacts/data_cleaned.csv')
    experiment.run()