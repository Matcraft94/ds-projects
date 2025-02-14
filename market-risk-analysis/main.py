# Creado por: Lucy
import os
import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from src.data_processing import MarketDataProcessor, MarketDataset
from src.feature_engineering import FeatureEngineer
from src.model import LSTMPredictor, ModelTrainer
from src.backtesting import BacktestStrategy
from src.visualization import MarketVisualizer
from torch.utils.data import DataLoader

def load_excel_files():
    current_file = os.path.abspath(__file__)
    project_dir = os.path.dirname(current_file)
    df_300 = pd.read_excel(os.path.join(project_dir, 'data/raw/crash300.xlsx'))
    df_500 = pd.read_excel(os.path.join(project_dir, 'data/raw/crash500.xlsx'))
    df_500_2 = pd.read_excel(os.path.join(project_dir, 'data/raw/crash500_2.xlsx'))
    return df_300, df_500, df_500_2

def main():
    # 1. Cargar y procesar datos
    df_300, df_500, df_500_2 = load_excel_files()
    
    date_cols = ['date', 'time']
    df_500 = df_500.drop(columns=date_cols)
    
    data_processor = MarketDataProcessor(window_size=60)
    processed_300 = data_processor.load_data(df_500)
    
    # 2. Feature Engineering
    engineer = FeatureEngineer()
    features_300 = engineer.create_technical_features(processed_300)
    print("Final shape, after dropna:", features_300.shape)
    
    # 3. Verificar tamaño mínimo de datos
    window_size = 60
    min_samples = window_size * 3
    
    if len(features_300) < min_samples:
        raise ValueError(f"Insufficient data: {len(features_300)} samples, need at least {min_samples}")
    
    # 4. Separar datos de validación final (20%)
    train_size = int(0.8 * len(features_300))
    train_val_data = features_300[:train_size]
    final_validation_data = features_300[train_size:]
    
    print(f"Tamaño de datos de entrenamiento+validación: {len(train_val_data)}")
    print(f"Tamaño de datos de validación final: {len(final_validation_data)}")
    
    # 5. Calcular parámetros para TimeSeriesSplit
    total_samples = len(train_val_data)
    min_samples_per_fold = window_size * 3  # Mínimo de muestras necesarias por fold
    
    # Calcular número óptimo de splits
    n_splits = min(3, total_samples // (min_samples_per_fold * 2))  # Asegurar al menos el doble del mínimo por fold
    if n_splits < 2:
        n_splits = 2  # Mínimo 2 splits
    
    # Calcular tamaño de test para cada fold
    test_size = total_samples // (n_splits + 1)  # +1 para asegurar suficientes datos de entrenamiento
    
    print(f"Número de splits: {n_splits}")
    print(f"Tamaño de test por fold: {test_size}")
    
    # 6. Configurar Time Series Split
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
    
    # 7. Parámetros del modelo
    batch_size = min(128, test_size // 2)  # Ajustar batch_size según el tamaño de test
    num_epochs = 10
    patience = 3
    input_dim = len(features_300.select_dtypes(include=[np.number]).columns)
    hidden_dim = 32
    
    print(f"Batch size: {batch_size}")
    print(f"Input dimension: {input_dim}")
    
    # 7. Entrenamiento con Time Series Split
    fold_scores = []
    best_model = None
    best_val_loss = float('inf')
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(train_val_data)):
        print(f'\nFold {fold + 1}/{n_splits}')
        
        # Preparar datos para este fold
        train_fold = train_val_data.iloc[train_idx]
        val_fold = train_val_data.iloc[val_idx]
        
        print(f"Tamaño del conjunto de entrenamiento para fold {fold + 1}: {len(train_fold)}")
        print(f"Tamaño del conjunto de validación para fold {fold + 1}: {len(val_fold)}")
        
        # Verificar tamaños mínimos
        if len(train_fold) <= window_size or len(val_fold) <= window_size:
            print(f"Saltando fold {fold + 1} debido a datos insuficientes")
            continue
        
        # Crear datasets
        train_dataset = MarketDataset(train_fold, window_size)
        val_dataset = MarketDataset(val_fold, window_size)
        
        print(f"Tamaño del dataset de entrenamiento: {len(train_dataset)}")
        print(f"Tamaño del dataset de validación: {len(val_dataset)}")
        
        # Ajustar batch_size si es necesario
        actual_batch_size = min(batch_size, len(train_dataset), len(val_dataset))
        
        # Crear dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=actual_batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=actual_batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=True
        )
        
        # Inicializar modelo
        model = LSTMPredictor(input_dim=input_dim, hidden_dim=hidden_dim)
        trainer = ModelTrainer(model)
        
        # Entrenamiento con early stopping
        best_fold_loss = float('inf')
        counter = 0
        
        for epoch in range(num_epochs):
            train_loss = trainer.train_epoch(train_loader)
            val_loss = trainer.validate(val_loader)
            
            print(f'Epoch {epoch + 1}/{num_epochs}:')
            print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            
            if val_loss < best_fold_loss:
                best_fold_loss = val_loss
                counter = 0
                torch.save(model.state_dict(), f'model_fold_{fold + 1}.pth')
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), 'best_model.pth')
                    best_model = model
            else:
                counter += 1
                if counter >= patience:
                    print("Early stopping triggered")
                    break
        
        fold_scores.append(best_fold_loss)
        print(f'Fold {fold + 1} - Best Validation Loss: {best_fold_loss:.4f}')
    
    if not fold_scores:
        raise ValueError("No se completó ningún fold exitosamente")
    
    print("\nResultados Time Series Cross Validation:")
    print(f'Media de Validation Loss: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}')
    
    # 8. Evaluación final
    if len(final_validation_data) > window_size:
        final_dataset = MarketDataset(final_validation_data, window_size)
        print(f"Tamaño del dataset de validación final: {len(final_dataset)}")
        
        final_loader = DataLoader(
            final_dataset,
            batch_size=min(batch_size, len(final_dataset)),
            shuffle=False,
            num_workers=0,
            drop_last=True
        )
        
        # Cargar mejor modelo
        best_model.load_state_dict(torch.load('best_model.pth'))
        trainer = ModelTrainer(best_model)
        final_loss = trainer.validate(final_loader)
        
        print(f"\nPérdida en Conjunto de Validación Final: {final_loss:.4f}")
        
        # 9. Backtesting
        backtest = BacktestStrategy(final_validation_data)
        results = backtest.run_backtest()
        
        print("\nResultados del Backtest (Validación Final):")
        for metric, value in results.items():
            print(f"{metric}: {value:.4f}")
        
        # 10. Visualización
        visualizer = MarketVisualizer()
        signals = backtest.generate_signals()
        visualizer.plot_price_and_signals(final_validation_data, signals)
    else:
        print("Conjunto de validación final demasiado pequeño para evaluación")

if __name__ == "__main__":
    main()