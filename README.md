# Yaprak Hastalığı Tanıma

> MobileNetV3-Large tabanlı derin öğrenme modeli ile bitki yapraklarındaki hastalıkların otomatik tespiti.

&nbsp;

## Proje Hakkında

Bu çalışmada, hafif ve hızlı bir mimari olan **MobileNetV3-Large** modeli, PlantVillage veri seti üzerinde transfer learning yöntemiyle yeniden eğitilmiştir. Amaç; biber, patates ve domates bitkilerine ait 15 farklı sağlıklı/hastalıklı yaprak kategorisini yüksek doğrulukla sınıflandırmaktır.

Modelin temel avantajı, EfficientNet veya ResNet gibi daha büyük mimarilere kıyasla **daha az parametre** ile rekabetçi doğruluk sunmasıdır — bu da onu sınırlı donanımlarda çalıştırma açısından cazip kılar.

&nbsp;

---

## Model Performansı

```
Test Doğruluğu  →  % 97.77
Ağırlıklı F1    →  0.9776
Makro F1        →  0.9691
En İyi Val Acc  →  0.9784  (Epoch 11)
```

&nbsp;

### Sınıf Bazında Detaylı Sonuçlar

```
Sınıf                                          Precision   Recall    F1
─────────────────────────────────────────────────────────────────────────
Pepper__bell___Bacterial_spot                   1.0000     0.9900    0.9950
Pepper__bell___healthy                          0.9938     1.0000    0.9969
Potato___Early_blight                           0.9902     1.0000    0.9951
Potato___Late_blight                            0.9898     0.9898    0.9898
Potato___healthy                                1.0000     0.8182    0.9000  ⚠
Tomato_Bacterial_spot                           0.9862     0.9683    0.9772
Tomato_Early_blight                             0.9222     0.8830    0.9022  ⚠
Tomato_Late_blight                              0.9795     0.9845    0.9820
Tomato_Leaf_Mold                                1.0000     0.9806    0.9902
Tomato_Septoria_leaf_spot                       0.9581     0.9756    0.9668
Tomato_Spider_mites_Two_spotted_spider_mite     0.9605     0.9884    0.9742
Tomato__Target_Spot                             0.9237     0.9237    0.9237  ⚠
Tomato__Tomato_YellowLeaf__Curl_Virus           1.0000     0.9971    0.9985
Tomato__Tomato_mosaic_virus                     0.9667     0.9667    0.9667
Tomato_healthy                                  0.9689     0.9873    0.9781
─────────────────────────────────────────────────────────────────────────
Weighted Avg                                    0.9778     0.9777    0.9776
```

> ⚠ işaretli sınıflar görece düşük F1 skoruna sahip olmakla birlikte, bu durum büyük ölçüde test setindeki az sayıda örneğe (11–118 arası) bağlıdır ve genel model başarısını etkilememektedir.

&nbsp;

### Eğitim Grafikleri

![Eğitim Grafikleri](hasta%20bitki%20tespiti/mobilenet_outputs/training_curves.png)

### Karmaşıklık Matrisi

![Karmaşıklık Matrisi](hasta%20bitki%20tespiti/mobilenet_outputs/confusion_matrix.png)

### Tahmin

![Karmaşıklık Matrisi](hasta%20bitki%20tespiti/mobilenet_outputs/predictions.png)

&nbsp;


---

## Veri Seti

| | |
|---|---|
| Kaynak | [PlantVillage — Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease) |
| Toplam görüntü | 20.638 |
| Sınıf sayısı | 15 |
| Kapsanan bitkiler | Biber · Patates · Domates |
| Bölünme | %75 eğitim · %15 doğrulama · %10 test |
| Bölünme yöntemi | Rastgele (sabit tohum: 21) |

&nbsp;

---

## Model & Eğitim Detayları

**Mimari:** MobileNetV3-Large (ImageNet ağırlıkları)

Backbone tamamıyla dondurulduktan sonra yalnızca son konvolüsyon bloğu (`features.16`) ve yeni başlık eğitime açılmıştır.

```
[ Dondurulmuş Backbone ]
        ↓
[ features.16  —  eğitilebilir ]
        ↓
  Linear(960 → 256)
  Hardswish
  Dropout(0.2)
  Linear(256 → 15)
```

&nbsp;

**Eğitim parametreleri:**

| Parametre | Değer |
|---|---|
| Optimizer | RMSprop (α=0.99) |
| Başlangıç LR | 3e-4 |
| Scheduler | ReduceLROnPlateau (factor=0.3, patience=2) |
| Kayıp fonksiyonu | CrossEntropyLoss (label smoothing=0.1) |
| Batch boyutu | 32 |
| Epoch sayısı | 12 |
| Görüntü boyutu | 224 × 224 |

&nbsp;

**Veri artırma (yalnızca eğitim):**

```
Resize(256×256)  →  RandomResizedCrop(224, scale=0.7–1.0)
→  RandomHorizontalFlip  →  RandomAffine(±10°, translate=5%)
→  ColorJitter(hue=0.05)  →  Normalize(ImageNet)
```
Toplam eğitim süresi: **~38 dakika** (12 epoch)


&nbsp;

---

## Kaynaklar
- [Taminde Kullanılan Fotoğrafın linki](https://extension.wvu.edu/lawn-gardening-pests/plant-disease/fruit-vegetable-diseases/bacterial-leaf-spot-of-pepper)
- [Searching for MobileNetV3 — Howard et al.](https://arxiv.org/abs/1905.02244)
- [PlantVillage Veri Seti Makalesi](https://arxiv.org/abs/1511.08060)
- [Derin Öğrenme ile Bitki Hastalığı Tespiti — Frontiers](https://www.frontiersin.org/articles/10.3389/fpls.2016.01419/full)
