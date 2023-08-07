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

## Opis EN

The code is in the `trafficlightdetection` directory. It contains the entire module 
created within the project. In the `metrics` subdirectory there is code from
https://github.com/rafaelpadilla/review_object_detection_metrics modified for the project.

### Environment setup

In order to set up the environment and download the dependencies, call
`setup.sh` script. Call:

```bash
./setup.sh
```

### Trenowanie sieci

Do trenowania sieci służy skrypt `train.py`. Po skonfigurowaniu środowiska
wywołać w celu zapoznania się z parametrami:

### Network training

The `train.py` script is used to train the network. After setting up the environment
call to see the parameters:

```bash
python3 -m trafficlightdetection.train --help
```

Konfiguracja wykorzystana w pracy znajduje się w pliku `configuration.yml`.

The configuration used in the work is in the `configuration.yml` file.

### Dokonywanie predykcji

Detekcja na pojedynczych zdjęciach jest możliwa dzięki skryptowi `predict.py`.
Wywołanie poprzez:

### Making predictions

Detection on individual photos is possible thanks to the `predict.py` script.
Call via:

```bash
python3 -m trafficlightdetection.predict --help
```

Model wykorzystany w pracy znajduje się w katalogu: 
The model used in the work can be found in the catalog:
`models/lightning_logs/version_2/checkpoints/epoch=5-step=38069.ckpt`
