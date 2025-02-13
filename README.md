# Portfolio de Proyectos de Ciencia de Datos

## Sobre Mí
Soy un Ingeniero Matemático especializado en Computación Científica y Modelado de Ecuador, actualmente trabajando como Analista Cuantitativo en el Centro de Evaluacion e Innovacion Educativa (CEIE), Universidad UTE. Mi enfoque se centra en el uso del modelado matemático, aprendizaje automático y análisis de datos para resolver problemas complejos.

## Resumen de Proyectos

### [Actuarial Loss Prediction](./actuarial-loss-prediction/)
Sistema de predicción de pérdidas actuariales utilizando XGBoost y análisis estadístico avanzado.
- Implementación de modelo XGBoost con optimización de hiperparámetros
- Procesamiento avanzado de texto para descripción de reclamaciones
- Análisis exhaustivo por segmentos de valor
- Pipeline completo de preprocesamiento y modelado
- RMSE de 23,669 en conjunto de prueba

### [Neural PDEs Solver](./neural-pdes-solver/)
Implementación de Redes Neuronales Basadas en Física (PINNs) para resolver ecuaciones diferenciales parciales directas e inversas.
- Solución del sistema de péndulo doble con descomposición de dominio y conservación de energía (MSE < 1e-3)
- Implementación de métodos NLLSQ y VarPro para problemas inversos de PDEs con análisis comparativo
- Sistema de manejo de memoria GPU para optimización de rendimiento
- Arquitectura de red con bloques residuales y activaciones adaptativas
- Visualización avanzada de soluciones y análisis de error mediante contornos y distribuciones

### [Market Risk Analysis Platform](./market-risk-analysis/)
Plataforma integral para evaluación de riesgos de mercado utilizando técnicas de aprendizaje automático y redes neuronales LSTM.
- **Desempeño del Modelo**:
  - Crash 300: Loss de entrenamiento 0.1250 → 0.0833 (validación final)
  - Crash 500: Loss de entrenamiento 0.0473 → 0.0388 (validación final)
  - Media de validation loss: 0.3201 ± 0.2505 (Crash 300) y 0.1763 ± 0.2162 (Crash 500)
- **Métricas de Trading**:
  - Crash 300: Retorno total -14.70%, Ratio de Sharpe -1.4005, Drawdown máximo -14.76%
  - Crash 500: Retorno total -17.63%, Ratio de Sharpe -2.0967, Drawdown máximo -17.68%
- **Eficiencia del Pipeline**:
  - Preprocesamiento robusto: reducción de ruido (11,623 → 10,607 muestras en Crash 300)
  - Convergencia en 10 épocas con early stopping
  - Tasa de éxito en operaciones: 43.78% (Crash 300) y 42.62% (Crash 500)

### [Educational Assessment ML Pipeline](./educational-assessment/)
Sistema automatizado para análisis de datos educativos y predicción del rendimiento estudiantil.
- Sistema de análisis psicométrico automatizado
- Implementación de IRT con redes neuronales
- Dashboard de visualización interactivo
- Seguimiento de métricas de rendimiento

### [Geospatial Anomaly Detection](./geospatial-analysis/)
Sistema para detección de anomalías geográficas mediante análisis espacial avanzado.
- Implementación de análisis espacial con H3
- Modelado probabilístico con Pyro
- Backend API REST con Django
- Visualizaciones de mapas interactivos

### [Mathematical Research Tools](./math-research-tools/)
Implementación de soluciones numéricas para problemas de investigación matemática.
- Solucionador numérico de ecuaciones biharmónicas
- Herramientas de visualización de soluciones
- Documentación técnica detallada
- Implementaciones de papers de investigación

### [Academic Performance Prediction](./academic-performance-prediction/)
Enfoque de aprendizaje automático para la detección temprana de deserción estudiantil usando LightGBM.
- Implementación de LightGBM acelerada por GPU
- Ingeniería de características exhaustiva
- Análisis de factores socioeconómicos
- 88% de precisión global en predicción de deserción

## Tecnologías Utilizadas
- **Lenguajes**: Python, R, MATLAB
- **Frameworks ML/DL**: PyTorch, TensorFlow, scikit-learn
- **Análisis de Datos**: Pandas, NumPy, SciPy
- **Visualización**: Matplotlib, Seaborn, Plotly
- **Bases de Datos**: PostgreSQL, Neo4j
- **Otros**: Git, LaTeX, Django

## Estructura del Repositorio
```
.
├── neural-pdes-solver/
├── market-risk-analysis/
├── educational-assessment/
├── geospatial-analysis/
├── math-research-tools/
├── academic-performance-prediction/
└── README.md
```

## Publicaciones Seleccionadas
- A biharmonic equation with discontinuous nonlinearities. *Eduardo Arias, Marco Calahorrano, Alfonso Castro. Electron. J. Differential Equations, 2024*

## Contacto
- LinkedIn: [Perfil](https://www.linkedin.com/in/eduardo-arias-3e0/)
- Correo: mat.eduardo.arias@outlook.com

## Licencia
Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.
