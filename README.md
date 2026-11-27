# RoboGuard: Robot Arm Health Monitor and Grip-Failure Predictor

Programming for AI, Lab 14 mini project. Analyses UR3 CobotOps sensor data and
trains a Random Forest to predict `grip_lost` events.

## Dataset
Tyrovolas, M., Aliev, K., Antonelli, D., & Stylios, C. (2024). UR3 CobotOps.
UCI Machine Learning Repository. https://doi.org/10.24432/C5J891 (CC BY 4.0)

## Structure
- `core/loader.py` DataLoader (validation, column-name fix, timestamp parsing)
- `core/cleaning.py` BaseProcessor and Cleaner (inheritance)
- `core/analysis.py` Analyzer (statistics, correlation, IQR outliers)
- `core/model.py` GripModel (grouped split, Random Forest, metrics)
- `core/pipeline.py` reusable functions that use all classes
- `tests/` pytest tests

## Run
```bash
pip install -r requirements.txt
python -m core.pipeline      # backend pipeline
pytest                       # tests
streamlit run app.py         # frontend (once app.py is added)
```

## AI usage declaration
(Write here which AI tools you used and for what.)
