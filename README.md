# RSVbaseline
Для запуска бейзлайна коррекции прогнозов численной модели wrf в директории `./experiments/constantBaseline/` запускаем файл `main.py`

```sh
cd /path/to/project/dir/experiments/constantBaseline
python main.py
```

Для удачного запуска в переменных `wrf_folder` и `wrf_folder` укажите актуальные директории с данными WRF и ERA5. 
Также в файле `correction/config.py` укажите актуальный путь к проекту:
```python
__C.GLOBAL.BASE_DIR = 'path to project'
```
## Docker

Dillinger is very easy to install and deploy in a Docker container.
Проект можно запустить внутри Docker контейнера. В проекте есть Dockerfile для сброки небходимого образа.

```sh
cd /path/to/project
docker build ./ -t baseline_wrf_correction 
docker run -it --rm --name correction_baseline \
	-v /path/to/wrf/files/.:/home/wrf_data \
	-v /path/to/era5/files/.:/home/era_data \
	-v $(pwd)/logs/.:/home/logs \
	--gpus device=0 --ipc=host golikov_wrf_correction
```

Оказавшись таким образом внутри контейнера запускаем скрипт с обучением:
```sh
python main.py
```

## Полезные ссылки

Для более подробного ознакомления, с какими данными мы работаем, можно пройти по следующим ссылкам:

- [ERA5] - Ссылка на официальный сайт ERA5 (нужен VPN) 
- [WRF] - Ссылка на официальный сайт WRF 



   [ERA5]: <https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=overview>
   [WRF]: <https://www.mmm.ucar.edu/models/wrf>

