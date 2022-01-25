# Quickstart
Классификация слов из ecog разделена на 2 этапа: 1) Востановление мел-спектрограммы звука из ecog 2) Классификация слов по востановленной мел-спектрограмме.


Что нужно для запуска:
1. Скачать данные по пациентам и указать путь до файлов [тут](https://github.com/pet67/ossadtchi-ml-test-bench-speech/blob/master/library/patients.json#L5-L10).
2. Скачать разметку на слова и положить рядом с файлами пациентов. [patient 1](https://drive.google.com/file/d/1R-k8F_ce8PNPX4RZ9XbGHDRq4cbcVzIW/view?usp=sharing) [patient 2](https://drive.google.com/file/d/1luJYLok_JQifALgHd96ifZ6shh3tr18k/view?usp=sharing)
3. Запустить [этот](https://github.com/pet67/ossadtchi-ml-test-bench-speech/blob/master/run_example.sh) скрипт.


Результат выполнения: json файл с логом обучения и с финальной метрикой качества
