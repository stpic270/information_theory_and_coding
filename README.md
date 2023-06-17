# Данный репозиторий содержит несколько автоенкодеров для сжатия изображения

## Запуск тестов

Чтобы сжать изображение необходимо указать следующие параметры, пример:

#### python src/inference.py -poi dataset/inference_images/baboon.png  -mp Weights/pretrained_for_inference_weights/Mobilenet -b 4 

##### -poi - параметр --path_one_image - путь к изображению, которое вы хотите сжать

-mp - параметр model_path - путь к весам модели, которая должна содержать папки Encoder, Decoder с весами. Веса модели Mobilenet лежат в репозитории, Uresnet можно подргузить по ссылке - ...

-b параметр --B - степень сжатия, данный параметр должен соответствовать весу моделей, на котором его тренировали. Например, в папке Weights/pretrained_for_inference_weights/Mobilenet лежат следующие веса: 

![image](https://github.com/stpic270/information_theory_and_coding/assets/58371161/b58e72b4-c74e-4a8c-8b83-a096bc15a27f)

Значит и -b парамтр может быть [2,4,8]

Чтобы расжать изображение необходимо запустить:

#### python src/inference.py -poi dataset/inference_images/baboon.png  -mp Weights/pretrained_for_inference_weights/Mobilenet -b 4 -oe False

-oe - параметр --only_encode сигнализирует расжать или сжать изображение 

При сжатии и расжатии изображения в конце выводятся путь к либо сжатому файлу, либо расжатому. Пример:

![image](https://github.com/stpic270/information_theory_and_coding/assets/58371161/77266c35-3792-49cf-b399-ba8935f287fd)

Можно также построить графики для сравнения результатов сжатия между jpeg и моделями. Пример запуска:

#### python src/inference.py -ibg Weights/pretrained_for_inference_weights/Mobilenet

-ibg параметр --is_build_graph - путь к весам, такой же как и при сжатии.расжатии одного изображения

В результате будут построены графики PSNR/BPP, для jpeg и выбранной модели. Также будут сохранены результаты сжатия jpeg в папке dataset/inference_images/jpg_compress.

## Запуск тренировки модели

Для начало нужно поместить изображения в папки dataset/Test_images и dataset/Training по классам. Пример:

![image](https://github.com/stpic270/information_theory_and_coding/assets/58371161/a670d6a0-4af5-4f17-9082-d586855a0310)

Пример запуска тренировки модели:

#### python src/train.py -m Uresnet --B 2 -wp Weights/Uresnet -see 8 -e 10 -pe 5 

-m - параметр --model - необходимо выбрать модель. Варианты [Uresnet, Mobilenet, VGG]

--B - степень квантования во время обучения

-wp - параметр --weights_path - путь где будут сохранены веса автоенкодера. Пример сохраненных весов модели в папке Weights:

![image](https://github.com/stpic270/information_theory_and_coding/assets/58371161/aa90ea1e-f622-4718-b7f1-45aa289ddbff)

-see - параметр --save_every_epoch - как часто сохранять, в примере веса сохраняются каждую 8-ую эпоху, начиная с 1
-e - параметр --epochs - количество эпох
-pe - параметр --plot_every - как часто сохранять тестовые картинки во время обучения, в примере тестовые изображения сохраняются каждые 5 эпох, начиная с 1. Изображения сохраняются в папку dataset/images_during_training. Пример:

![image](https://github.com/stpic270/information_theory_and_coding/assets/58371161/85077b54-bcad-4337-aaa1-12f71271d0f6)











