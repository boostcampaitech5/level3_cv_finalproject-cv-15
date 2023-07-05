#!/bin/bash

echo "--------------------------"
echo "install packages for build"
echo "--------------------------"
apt-get update -y
apt-get install curl -y
apt-get install gcc make -y
apt-get install -y make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev liblzma-dev wget curl llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev

echo "------------------------------------"
echo "install screen and env bashrc update"
echo "------------------------------------"
apt-get install screen -y
if !(ls -al ~/ | grep ".screenrc"); then
	echo 'ck 5000' >> ~/.screenrc
	echo 'vbell off' >> ~/.screenrc
	echo 'defscrollback 10000' >> ~/.screenrc
	echo 'termcapinfo xterm* ti@:te@' >> ~/.screenrc
	echo 'startup_message off' >> ~/.screenrc
	echo 'hardstatus on' >> ~/.screenrc
	echo 'hardstatus alwayslastline' >> ~/.screenrc
	echo 'hardstatus string "%{.bW}%-w%{.rW}%n*%t%{-}%+w %= %c ${USER}@%H"' >> ~/.screenrc
	echo 'bindkey -k k1 select 0' >> ~/.screenrc
	echo 'bindkey -k k2 select 1' >> ~/.screenrc
	echo 'bindkey -k k3 select 2' >> ~/.screenrc
fi

echo "------------------------------------------"
echo "install ko-language pack and bashrc update"
echo "------------------------------------------"
apt-get install language-pack-ko -y
if !(grep -qc "LANG" ~/.bashrc); then
	echo 'export LANG="ko_KR.UTF-8"' >> ~/.bashrc
fi

echo "-------------------------------"
echo "install pyenv and bashrc update"
echo "-------------------------------"
git clone https://github.com/pyenv/pyenv.git ~/.pyenv
if !(grep -qc "PYENV_ROOT" ~/.bashrc); then
	echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
    echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
    echo 'eval "$(pyenv init -)"' >> ~/.bashrc
fi

echo "--------------------"
echo "install python3.11.4"
echo "--------------------"
source ~/.bashrc
pyenv install 3.11.4
pyenv global 3.11.4

echo "--------------------------------"
echo "install poetry and bashrc update"
echo "--------------------------------"
curl -sSL https://install.python-poetry.org | python3 -
if !(grep -qc "/opt/ml/.local/bin" ~/.bashrc); then
	echo 'export PATH="/opt/ml/.local/bin:$PATH"' >> ~/.bashrc
fi
