# Creado por: Lucy
import pandas as pd
import torch
from src.data_processing import MarketDataProcessor, MarketDataset
from src.feature_engineering import FeatureEngineer
from src.model import LSTMPredictor, ModelTrainer
from src.backtesting import BacktestStrategy
from src.visualization import MarketVisualizer
from torch.utils.data import DataLoader

def load_excel_files():
    df_300 = pd.read_excel('data/raw/crash300.xlsx')
    df_500 = pd.read_excel('data/raw/crash500.xlsx')
    df_500_2 = pd.read_excel('data/raw/crash500_2.xlsx')
    return df_300, df_500, df_500_2

def main():
    # 1. Cargar y procesar datos
    df_300, df_500, df_500_2 = load_excel_files()
    
    date_cols = ['date', 'time']
    df_500 = df_500.drop(columns=date_cols)
    
    data_processor = MarketDataProcessor(window_size=60)
    
    # Procesar cada dataset
    processed_300 = data_processor.load_data(df_500)
    
    # 2. Feature Engineering
    engineer = FeatureEngineer()
    features_300 = engineer.create_technical_features(processed_300)
    
    # 3. Preparar datos para el modelo
    train_size = int(0.8 * len(features_300))
    train_data = features_300[:train_size]
    test_data = features_300[train_size:]
    
    dataset = MarketDataset(train_data, window_size=60)
    train_loader = DataLoader(
        dataset, 
        batch_size=128,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )
    
    test_dataset = MarketDataset(test_data, window_size=60) 
    test_loader = DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=0
    )
    
    # 4. Configurar y entrenar modelo
    input_dim = features_300.shape[1]
    model = LSTMPredictor(input_dim=input_dim, hidden_dim=32)
    trainer = ModelTrainer(model)
    
    # Entrenamiento
    num_epochs = 10
    for epoch in range(num_epochs):
        train_loss = trainer.train_epoch(train_loader)
        test_loss = trainer.validate(test_loader)
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')
    
    # 5. Backtesting
    backtest = BacktestStrategy(test_data)
    results = backtest.run_backtest()
    print("\nResultados del Backtest (Testing):")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
    
    # 6. Visualización
    visualizer = MarketVisualizer()
    signals = backtest.generate_signals()
    visualizer.plot_price_and_signals(test_data, signals)

if __name__ == "__main__":
    main()