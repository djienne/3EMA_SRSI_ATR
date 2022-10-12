sudo apt-get update
sudo apt install -y python3-pip python3-venv python3-dev python3-pandas git curl
git clone https://github.com/freqtrade/freqtrade.git
cd freqtrade
git checkout stable
./setup.sh -i
source ./.env/bin/activate
source /.env/bin/activate
