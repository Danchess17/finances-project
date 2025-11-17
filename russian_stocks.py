from main import LiquidityAdjustedVaRAnalyzer

analyzer = LiquidityAdjustedVaRAnalyzer()

results = analyzer.run_analysis(
    tickers=['SBER.ME', 'GAZP.ME', 'LKOH.ME', 'ROSN.ME', 'VTBR.ME'],
    weights=[0.3, 0.25, 0.2, 0.15, 0.1],
    portfolio_value=5000000,
    start_date='2020-01-01',  # Период до санкций
    end_date='2022-02-01',    # До начала специальной операции
    confidence_level=0.95,
    time_horizon=10
)