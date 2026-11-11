JavaScript
let activeSymbol = "BTCUSDT";
let tvWidget = null;

function loadTradingViewChart(symbol) {
  if (tvWidget) {
    // Mevcut grafiği temizle veya sembolünü değiştir
    tvWidget.chart().setSymbol(`BINANCE:${symbol}.P`);
    return;
  }

  tvWidget = new TradingView.widget({
    "autosize": true,
    "symbol": `BINANCE:${symbol}.P`,
    "interval": "15",
    "theme": "dark",
    "container_id": "tv_chart_container"
  });
}
