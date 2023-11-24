# RSVbaseline

## Build

```
docker build . -t corr
```
## Train

Положите тренировочные данные в ./data/train/era5 и ./data/train/wrf

Запустити докер-контейнер

```
docker run -it --rm --name corr2 \
	-v ./data/train:/home/data/train \
	-v ./logs/.:/home/logs \
	--gpus device=0 --ipc=host corr
```

Запустите обучение внутри докера.

```
python experiments/constantBaseline/main.py 
```

Экспорт весов модели будет осуществлён в ./logs/constantBaseline

## Inference
Положите тестовые данные в ./data/wrf . Создайте директорию ./data/wrf_corr

Запустите докер-контейнер

```
docker run -it --rm --name corr2 \
	-v ./data/wrf:/home/data/wrf \
	-v ./data/wrf_corr:/home/data/wrf_corr \
	-v ./logs/.:/home/logs \
	--gpus device=0 --ipc=host corr
```

Запустите инференс внутри контейнера

```
python experiments/constantBaseline/test.py 
```

Введите batch size (зависит от VRAM вашей карты - сколько влезет), номер эксперимента и номер эпохи внутри экспепримента.

Экспорт предсказанных значений будет осуществлён в ./data/wrf_corr