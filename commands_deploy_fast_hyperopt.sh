sudo apt-get update
sudo apt install -y python3-pip python3-venv python3-dev python3-pandas git curl
git clone https://github.com/freqtrade/freqtrade.git
cd freqtrade
git checkout stable
./setup.sh -i

source ./.env/bin/activate
cd ..
git clone https://github.com/djienne/3EMA_SRSI_ATR.git
cd 3EMA_SRSI_ATR/
freqtrade hyperopt --strategy EMA3SRSI --timeframe 5m --config ./user_data/config_backtest.json --hyperopt-loss SortinoHyperOptLossDaily --analyze-per-epoch --min-trades 500 --timerange 20170817- --spaces buy -j -1 -e 3000
