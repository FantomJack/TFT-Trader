# TFT-Trader
Bakalárska práce, Ondrej Bealan, FEI STU.

## Predpoklady
V rámci práce odporúčame pre rovnaké prostredie používať Python 3.10.12.
Pre spustenie programu je potrebné mať nainštalované knižnice v [requirements.txt](./requirements.txt). Môžete to spraviť nasledujúcim spôsobom:

### 1. Najskôr nainštalujte všetky balíky (okrem PyTorch):
 `pip install -r requirements.txt`

### 2. Následne nainštalujte PyTorch s podporou CUDA 11.7:
`pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2+cu117 --extra-index-url https://download.pytorch.org/whl/cu117`
 
## Spracovanie dát
Dáta na trénovanie môžete predspracovať pomocou `python3 src/data_processing.py`. Je možné, že pri tom musíte nachádzať v priečinku src.
Dáta z [data/](./data/) .txt súborov sa spracujú, pridá sa percentuálny denný nárast ceny, vypočítajú sa technické indikátory a pridajú sa kalendárne dáta.
Predspracované dáta sa ukladajú do [csv_data/](./csv_data/), po tickeroch do .csv súborov v priečinkoch 1,2,3,4 podľa časového intervalu, ktorý dataset obsahuje.
 
## Trénovanie
Modely TFT a MLP môžete natrénovať pomocou príkazov
`python3 src/MLP.py` a `python3 src/TFT.py`.
Trénovanie trvá dlhú dobu, pri slabom výkone aj dni, preto odporúčame trénovanie spúšťať v tvare `nohup python3 TFT.py &> output.log &`.
Ak chcete výstup prečítať, stačí potom použiť `tail -f output.log`.
Ak vám na výstupe vôbec nezáleží, môžete ho namiesto `output.log` zahodiť do `/dev/null`.
Natrénované modely sa uložia do [checkpoints/](./checkpoints/).

## Dopredné testovanie
Pre dopredné testovanie prejdite do priečinka [src/forward_testing/](./src/forward_testing/). V ňom sa nachádzajú súbory [mlp_ft.py](./src/forward_testing/mlp_ft.py) a [tft_ft.py](./src/forward_testing/tft_ft.py).
Tie odporúčame spúšťať rovnako ako pri trénovaní pomocou `nohup`.
Po ich spustení sa načítajú modely z [checkpoints/](./checkpoints/) a otestujú sa na 25 tickeroch, na ktorých boli trénované. (Ale na testovacom datasete.)
Výsledky testov sa uložia do [./forward_testing/](./forward_testing/) v koreňovom priečinku.

## Štatistické porovnanie modelov
Štatistické porovnanie modelov je implementované v súbore [src/model_compare.py](./src/model_compare.py). Výsledky celkového testovania sú vypísané do konzoly. Výsledky testovania po skupinách a kombináciách parametrov (trénovacieho intervalu, pomeru rozdelenia dát na trénovacie a validačné a skupiny hyperparametrov) sa uložia do priečinku [./model_compare/](./model_compare/) v koreňovom priečinku.

