# Praca inżynierska / B.Eng. Thesis

**Tytuł**:  
Wykrywanie i rozpoznawanie samochodowej sygnalizacji świetlnej na podstawie
obserwacji wizualnej z zastosowaniem sieci splotowych

**Title**:  
Detection and recognition of automotive traffic lights from visual observation
using convolutional networks

**Autor/Author**:  
Michał Dziewulski s19682

## Opis PL

Kod znajduje się w katalogu `trafficlightdetection`. Zawiera on cały moduł
stworzony w ramach projektu. W podkatalogu `metrics` jest kod pochodzący z
https://github.com/rafaelpadilla/review_object_detection_metrics zmodyfikowany
na potrzeby projektu.

### Konfiguracja środowiska

W celu skonfigurowania środowiska i pobrania zbioru danych należy wywołać
skrypt `setup.sh`. Wymaga on zainstalowanego programu `conda` oraz
skonfigurowania dostępu do API Kaggle. Wywołanie:

```bash
./setup.sh
```

Po uruchomieniu należy aktywować środowisko:  

```bash
conda activate tl-detection
```

Następnie należy zainstalować paczkę (`-e` dla trybu deweloperskiego):

```bash
pip install -e .
```

### Trenowanie sieci

Do trenowania sieci służy skrypt `train.py`. Po skonfigurowaniu środowiska
wywołać w celu zapoznania się z parametrami:

```bash
python3 -m trafficlightdetection.train --help
```

Konfiguracja wykorzystana w pracy znajduje się w pliku `configuration.yml`.

### Dokonywanie predykcji

Detekcja na pojedynczych zdjęciach jest możliwa dzięki skryptowi `predict.py`.
Wywołanie poprzez:

```bash
python3 -m trafficlightdetection.predict --help
```

Model wykorzystany w pracy znajduje się w katalogu: 
`models/lightning_logs/version_2/checkpoints/epoch=5-step=38069.ckpt`
