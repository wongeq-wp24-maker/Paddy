# Paddy Yield Prediction — Screenshot Style V5

This version is rebuilt against the supplied reference screenshots and ABCCC (2).ipynb.

Visible additions:
- Data Understanding > Class Distribution: large bar chart + pie chart + class summary
- Data Understanding > Correlation: compact screenshot-style heatmap + full 33-input correlation heatmap
- Model Performance: notebook model-comparison table + grouped comparison chart
- Reference-like sidebar, red active tabs, dark model-performance banner, CV table and training-visualisation tabs

## Streamlit Cloud
Keep these files in one repository folder and set **Main file path** to `app.py`.

## V6 visual fix
- Forces a light Streamlit theme so dropdowns, tables and controls remain readable even when the computer/browser uses dark mode.
- Hides Streamlit's floating top toolbar/header so it no longer covers the navigation tabs.
- Makes dropdowns white with visible borders, hover emphasis, and a red primary action style matching the reference dashboard.
- Upload the hidden `.streamlit/config.toml` folder together with the other files to GitHub.
