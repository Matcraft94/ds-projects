# Creado por: Lucy
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import os

os.makedirs('data/processed', exist_ok=True)

class MarketVisualizer:
    @staticmethod
    def plot_price_and_signals(data: pd.DataFrame, signals: pd.Series) -> None:
        # fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
        fig = make_subplots(rows=2, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.1,
                    row_heights=[0.7, 0.3])
        
        # Precio y señales
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close']
            ),
            row=1, col=1
        )
        
        # Volatilidad
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['volatility'],
                name='Volatilidad'
            ),
            row=2, col=1
        )
        
        # Señales
        buy_points = data[signals == 1].index
        sell_points = data[signals == -1].index
        
        fig.add_trace(
            go.Scatter(
                x=buy_points,
                y=data.loc[buy_points, 'close'],
                mode='markers',
                name='Compra',
                marker=dict(color='green', size=10)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=sell_points,
                y=data.loc[sell_points, 'close'],
                mode='markers',
                name='Venta',
                marker=dict(color='red', size=10)
            ),
            row=1, col=1
        )
        
        fig.update_layout(height=800, title='Análisis de Mercado y Señales')
        fig.show()
        fig.write_image("data/processed/plot.png")
        

# Ejemplo de uso:
if __name__ == "__main__":
    # Inicializar procesador de datos
    data_processor = MarketDataProcessor()
    df_300 = data_processor.load_data('data/raw/crash300.csv')
    
    # Crear características
    feature_engineer = FeatureEngineer()
    df_300 = feature_engineer.create_technical_features(df_300)
    
    # Preparar datos para el modelo
    dataset = MarketDataset(df_300, window_size=60)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Inicializar y entrenar modelo
    model = LSTMPredictor(input_dim=df_300.shape[1], hidden_dim=64)
    trainer = ModelTrainer(model)
    
    # Ejecutar backtest
    backtest = BacktestStrategy(df_300)
    results = backtest.run_backtest()
    
    # Visualizar resultados
    visualizer = MarketVisualizer()
    signals = backtest.generate_signals()
    visualizer.plot_price_and_signals(df_300, signals)