# Backtest by MoPhongDeals

Script backtest cuối cho model `fs03_lags_cycle` reproduce từ `research10_1788533519`.

Model mặc định:

```text
D:\SCJ999\srateries\RandomForest\RF_MLFlow\models\fs03_fixed_params_1788583848_bundle.joblib
```

Chạy test 2024-2026 với threshold mặc định `0.57`:

```powershell
cd D:\SCJ999\srateries\RandomForest\RF_MLFlow\backtest_by_Mophong
python Backtest_RF_FS03_Fixed.py --years 2024-2026
```

Thử threshold khác:

```powershell
python Backtest_RF_FS03_Fixed.py --years 2024-2026 --threshold 0.56
python Backtest_RF_FS03_Fixed.py --years 2024-2026 --threshold 0.58
```

Output:

- `outputs\yearly_fs03_fixed_t*_*.csv`
- `outputs\deals_fs03_fixed_t*_<year>.csv`

Rule cố định:

- Buy only.
- Entry tại open nến tiếp theo.
- RR = 1.
- 1R = 6 giá.
- Engine mô phỏng: `method.MoPhongDeals.MoPhongDeals`.
- `EXPIRATED=5`, `MAX_RUNNING_DEALS=12`, `MIN_DEAL_GAP=0`.
