# MRR

## Need to install

[Python 3](https://www.python.org/downloads/)
[Git](https://git-scm.com/)

## OS

Recommend for using Linux / macOS

## Download this repository

```bash
git clone https://github.com/nimiusrd/MRR.git
```

## Update this repository

```bash
git fetch origin main
git merge origin main
```

## 仮想環境 / virtual env
ref: https://docs.python.org/3/library/venv.html

### 作成 / make

```bash
python3 -m venv .venv
```

### 有効化 / activate

Run this command before execution.

```bash
. .venv/bin/activate
```

## Install modules

When you use M1 Mac,

```bash
brew install openblas
pip install cython pybind11 pythran numpy
OPENBLAS=$(brew --prefix openblas) CFLAGS="-falign-functions=8 ${CFLAGS}" pip install --no-use-pep517 scipy scikit-learn
```

This program uses numpy, matplotlib, and so on.
Need to install modules from pip.

```bash
pip3 install -r requirements.txt
```

## 実行 / execute

### 最適化 / optimize

configuration file: `src/config/base.py`

```bash
python main.py
```

### シミュレーション / simulate

引数には`src/config/simulate`にあるシミュレーションをするものを指定する．
`src/config/simulate/12_DE6.py` をシミュレーションしたい場合は以下のように実行する．

```bash
python simulator.py 12_DE6
```
![out](https://user-images.githubusercontent.com/13166203/160993029-90c707a8-0ad9-4ba2-a518-b1912d1687f9.jpg)
