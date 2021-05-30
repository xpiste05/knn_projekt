# KNN - License plate recognition
2021

# Autoři
Anh Le Hoang (xlehoa00)

Pištělák Radek (xpiste05)

Šorm Jan (xsormj00)

## Vstup

Cesta ke vstupnímu datasetu se nastavuje v souboru *inputParser.py*. Do proměnné *datasetDir* je potřeba zadat cestu ke složce a v proměnné *valuesFile* se doplní název souboru. Ten obsahuje na každém řádku trojici hodnot (cesta k obrázku, skutečná hodnota a příznak 1 pro trénovací sadu nebo 0 pro testovací) oddělenou středníkem - filePath;label;train. 

## Spuštění

### Přepínače

- -b    velikost batche
- -e    počet epoch
- -o    cesta ke složce pro uložení výstupu
- -r    zapnutí transformace obrázku (zarovnání a ořezání)
- -bl   použití baseline modelu pro rozpoznávání
- -a    náhodné pootočení snímků při trénování
- -t    testovací mód
- -m    cesta k modelu pro testování

### Trénování

```
python main.py -b BATCHSIZE -e EPOCHS -o OUTPUTFOLDER [-r] [-bl] [-a]
```

### Testování

```
python main.py -b BATCHSIZE -m MODEL -t [-r] [-bl] 
```
