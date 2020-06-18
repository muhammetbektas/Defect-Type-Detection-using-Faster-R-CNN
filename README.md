# Defect-Type-Detection-using-Faster-R-CNN
### Metal Levhalar Üzerindeki Hata Tiplerinin Tespiti <br/>
Biz bu projeyi “Anaconda Prompt” üzerinden “Virtual Environment” oluşturarak gerçekleştirdik. Öncelikle Anaconda’ nın bilgisayarımızda kurulu olması gerekmektedir. <br/>
Bu çalışmada Tensorflow v1.15 (CPU) kullandık. Buna uygun python sürümlerine [buradan](https://www.tensorflow.org/install/source#tested_build_configurations) bakabilirsiniz. Tensorflow GPU kullanmak için CUDA ve cuDNN indirmemiz gerekmektedir. Ekran kartınıza uygun versiyonları öğrenmek için [buraya](https://developer.nvidia.com/cuda-gpus) göz atabilirsiniz.
1. İlk olarak https://github.com/tensorflow/models adresindeki dosyayı .zip formatında indirelim.(TensorFlow Object Detection API repository)<br/>
2. C:/ dizininde “tensorflow1” isimli bir dosya açın ve indirdiğimiz “models_master.zip” dosyasını buraya çıkartalım ve ismini “models” olarak değiştirelim.<br/>
3. Bu linkten “faster_rcnn_ inception_ v2_coco” eğitilmiş modelini indirelim. Bu dosyayı C:\tensorflow1\models\research\object_detection klasörüne çıkartalım.<br/>
4. Github repomdaki “object_detection.zip” adlı dosyayı indirin ve ..\object_detection klasörü içerisine çıkartalım.<br/>
5. Bu işlemleri yaptıktan sonra şimdi “Virtual Environment” oluşturalım. Anaconda Prompt’ a aşağıdaki komutu yazalım ve bunu aktif hale getirelim.<br/>
`C:\>conda create -n tensorflow1 pip python=3.7`
`C:\>conda activate tensorflow1`
6. Kullanacağımız kütüphaneleri sırası ile indirelim. Tüm kütüphanelerin düzgün yüklendiğinden emin olun.<br/>
`(tensorflow1) C:\>pip install --ignore-installed --upgrade tensorflow==1.15`
`(tensorflow1) C:\>conda install -c anaconda protobuf`
`(tensorflow1) C:\>pip install pillow lxml cython jupyter matplotlib pandas opencv-python`
7 . Pythonpath’ i belirtmemiz gerekiyor. “Virtual Environment” i aktif hale getirdiğimizde bu komutuda her zaman çalıştırmanız gerekiyor.<br/>
`(tensorflow1) C:\>set PYTHONPATH=C:\tensorflow1\models;C:\tensorflow1\models\research;C:\tensorflow1\models\research\slim`
8. tf_slim’ i kullanacağımız için bunu da yüklemeliyiz.<br/>
`(tensorflow1) C:\>pip install --upgrade tf_slim`
9. Şimdi protobufları derlememiz gerekiyor. İlk önce komut satırında aşağıdaki komutu çalıştırın ve dizini ayarlayın. Daha sonra diğer komutu çalıştıralım.<br/>
`(tensorflow1)C:\>cd C:\tensorflow1\models\research`
`protoc --python_out=. .\object_detection\protos\anchor_generator.proto .\object_detection\protos\argmax_matcher.proto .\object_detection\protos\bipartite_matcher.proto .\object_detection\protos\box_coder.proto .\object_detection\protos\box_predictor.proto .\object_detection\protos\eval.proto .\object_detection\protos\faster_rcnn.proto .\object_detection\protos\faster_rcnn_box_coder.proto .\object_detection\protos\grid_anchor_generator.proto .\object_detection\protos\hyperparams.proto .\object_detection\protos\image_resizer.proto .\object_detection\protos\input_reader.proto .\object_detection\protos\losses.proto .\object_detection\protos\matcher.proto .\object_detection\protos\mean_stddev_box_coder.proto .\object_detection\protos\model.proto .\object_detection\protos\optimizer.proto .\object_detection\protos\pipeline.proto .\object_detection\protos\post_processing.proto .\object_detection\protos\preprocessor.proto .\object_detection\protos\region_similarity_calculator.proto .\object_detection\protos\square_box_coder.proto .\object_detection\protos\ssd.proto .\object_detection\protos\ssd_anchor_generator.proto .\object_detection\protos\string_int_label_map.proto .\object_detection\protos\train.proto .\object_detection\protos\keypoint_box_coder.proto`
10. Devamında kurulum dosyalarını çalıştıralım.<br/>
`(tensorflow1) C:\tensorflow1\models\research>python setup.py build`
`tensorflow1) C:\tensorflow1\models\research>python setup.py install`
11. Şimdi verileri Labelimg programı ile etiketleyelim. İlk önce aşağıdaki komutu çalıştıralım ve daha sonra command prompt a “labelimg” yazalım. <br/>
`(tensorflow1) C:\>pip install labelimg`
`(tensorflow1) C:\>labelimg`

12. Fotoğrafları bu şekilde hem eğitim hemde test kümesi için etiketleyelim. Xml dosyaları ile birlikte fotoğrafları C:\tensorflow1\models\research\object_detection\images dizininde train ve test dosyaları oluşturarak buraya atalım.
Xml dosyalarını .csv formatına dönüştürmek için aşağıdaki komutu çalıştıralım.<br/>
`(tensorflow1) C:\tensorflow1\models\research\object_detection>python xml_to_csv.py`
13. Bundan sonra “object_detection” klasörü içerisinde ki “generate_tfrecord.py” isimli dosyayı bir text editörü ile açalım. Sınıfları kendi verimize göre düzenleyelim.<br/>
`def class_text_to_int(row_label):
    if row_label == 'DefectTypeA':
        return 1
    elif row_label == 'DefectTypeB':
        return 2
    elif row_label == 'DefectTypeC':
        return 3
    else:
        None`
14. “Record” dosyalarını oluşturmak için aşağıdaki komutları çalıştıralım. Öncesinde dizini şu şekilde ayarlayalım:<br/> `C:\tensorflow1\models\research\object_detection`
`python generate_tfrecord.py --csv_input=images\train_labels.csv --image_dir=images\train --output_path=train.record`
`python generate_tfrecord.py --csv_input=images\test_labels.csv --image_dir=images\test --output_path=test.record`
15. ..\object_detection\training dosyası içerisindeki “labelmap.pptxt” dosyasını bir editör yardımı ile açalım ve kendi sınıflarımıza göre düzenleme yapalım.<br/>
`item {
  id: 1
  name: 'DefectTypeA'
}
item {
  id: 2
  name: 'DefectTypeB'
}
item {
  id: 3
  name: 'DefectTypeC'}`
16. Faster_rcnn_inception_v2.. klasörü içerisinde ki “pipeline.config” dosyasını ve ..\object_detection\samples\configs dosya konumunda ki “faster_rcnn_inception_v2_pets.config” dosyasını ..\object_detection\training dosyası içerisine kopyalayalım.<br/>
17. Daha sonra “faster_rcnn_inception_v2_pets.config” dosyasını bir text editörü ile açalım. Bazı düzenlemeler yapmalıyız.<br/>
#(9. satır) num_classes değerini 3 olarak değiştirelim.<br/>
#(110. satır) faster_rcnn_v2_coco modeliniz nerede ise dosya yolunu doğru bir şekilde ayarlayın. fine_tune_checkpoint :<br/> `"C:/tensorflow1/models/research/object_detection/faster_rcnn_inception_v2_coco_2018_01_28/model.ckpt"`
#(126 ve 128. satır) `input_path : "C:/tensorflow1/models/research/object_detection/train.record"`
`label_map_path: "C:/tensorflow1/models/research/object_detection/training/labelmap.pbtxt"` Bu şekilde görünmelidir.<br/>
#(132. satır) Test kümemizde kaç adet fotoğraf varsa belirtiniz.(num_examples)<br/>
#(140 ve 142. satır) `input_path : "C:/tensorflow1/models/research/object_detection/test.record"`
`label_map_path: "C:/tensorflow1/models/research/object_detection/training/labelmap.pbtxt"` Bu şekilde gözükmelidir ve ya sizin dosya konumlarınıza göre değişebilir.<br/>
18. “inference_graph” dosyası içerisindeki her şeyi ve “training” dosyası içerisindeki kontrol noktalarını silelim. Daha sonra dizini ..\object_detection olacak şekilde ayarlayıp aşağıdaki komutu çalıştıralım ve modelimizi eğitmeye başlayalım.<br/>
`python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/faster_rcnn_inception_v2_pets.config`

Kayıp değeri kalıcı bir şekilde 0.05 in altına düştüğünde “anaconda prompt” u kapatabiliriz.<br/>
19. “inference_graph” dosyası oluşturalım. XXXX yazan yere son kontrol noktasına ait değeri girelim. ..\object_detection\training dosyasına bakabilirsiniz.<br/>
`python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/faster_rcnn_inception_v2_pets.config --trained_checkpoint_prefix training/model.ckpt-XXXX --output_directory inference_graph`
20. Modeli test etmek için komut satırına “idle” yazalım. Açılan pencerede sol üst köşeden “Object_detection_image.py” dosyasını seçelim. “num_classes” değişkenine sınıf sayımızı girelim ve test etmek istediğimiz resmin dosya yolunu aşağıda belirtelim. F5 ile çalıştırabiliriz.<br/>

Şimdi sonuçları görelim.<br/>



