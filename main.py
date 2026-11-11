import ccxt

print("MULTI COIN AI TRADING BOT")
print("--------------------------")

exchange = ccxt.binance()

coins = [
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "BNB/USDT",
    "XRP/USDT",
    "DOGE/USDT",
    "LINK/USDT",
    "AVAX/USDT",
    "ADA/USDT",
]

print("\nCANLI COIN FIYATLARI\n")

for coin in coins:
    try:
        ticker = exchange.fetch_ticker(coin)
        price = ticker["last"]
        print(f"{coin}: {price} USDT")
    except Exception as error:
        print(f"{coin}: Fiyat alinamadi - {error}")

print("\nFiyat taramasi tamamlandi.")
