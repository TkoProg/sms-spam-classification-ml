# SMS Spam Detection using ANN

Ovo je studentski projekat iz mašinskog učenja / NLP-a.  
Cilj projekta je klasifikovati SMS poruku kao:

- `ham` - normalna poruka
- `spam` - neželjena / spam poruka

Za rješavanje problema korišten je UCI SMS Spam Collection dataset i jednostavan ANN model napravljen pomoću TensorFlow/Keras biblioteke.

## Cilj projekta

Glavni cilj projekta je napraviti model koji na osnovu teksta SMS poruke može predvidjeti da li je poruka spam ili nije.

Primjer:

```text
Congratulations! You won a free ticket. Call now.
```

Model bi ovakvu poruku trebao klasifikovati kao:

```text
spam
```

## Dataset

Korišten je UCI SMS Spam Collection dataset.

Dataset sadrži dvije kolone:

- `label` - klasa poruke (`ham` ili `spam`)
- `message` - tekst SMS poruke

Ukupan broj poruka u datasetu:

| Klasa | Broj poruka |
|---|---:|
| ham | 4825 |
| spam | 747 |

Dataset nije balansiran jer ima mnogo više `ham` poruka nego `spam` poruka. Zbog toga accuracy nije jedina dovoljna metrika, nego se koriste i precision, recall, F1-score i confusion matrix.

## Korištene tehnologije

- Python
- Pandas
- NumPy
- Matplotlib
- scikit-learn
- TensorFlow / Keras
- Google Colab / Jupyter Notebook

## Koraci u projektu

Projekat je urađen kroz sljedeće korake:

1. Učitavanje dataseta
2. Analiza raspodjele klasa
3. Pretvaranje labela u brojeve:
   - `ham` -> `0`
   - `spam` -> `1`
4. Podjela podataka na trening i test skup
5. Tokenizacija teksta
6. Pretvaranje SMS poruka u sekvence brojeva
7. Padding sekvenci na istu dužinu
8. Kreiranje ANN modela
9. Treniranje modela
10. Evaluacija rezultata

## Preprocessing

Neuronska mreža ne može direktno raditi sa tekstom, pa se tekst prvo mora pretvoriti u brojeve.

Korišten je `Tokenizer`, koji svakoj riječi dodjeljuje broj. Nakon toga se SMS poruke pretvaraju u sekvence brojeva.

Pošto SMS poruke nisu iste dužine, korišten je padding. Svaka poruka je podešena na dužinu od 50 tokena.

Korištene vrijednosti:

```python
VOCAB_SIZE = 10000
OOV_TOKEN = "<OOV>"
MAX_LENGTH = 50
PADDING_TYPE = "post"
TRUNC_TYPE = "post"
```

## Podjela podataka

Podaci su podijeljeni na:

- 80% trening skup
- 20% test skup

Korišten je `stratify=y` kako bi omjer `ham` i `spam` poruka ostao približno isti u trening i test skupu.

Broj poruka:

| Skup | Broj poruka |
|---|---:|
| Trening skup | 4457 |
| Test skup | 1115 |

## ANN model

Model se sastoji od sljedećih slojeva:

```python
model = Sequential([
    Embedding(
        input_dim=VOCAB_SIZE,
        output_dim=EMBEDDING_DIM
    ),

    GlobalAveragePooling1D(),

    Dense(24, activation="relu"),

    Dropout(0.3),

    Dense(1, activation="sigmoid")
])
```

Objašnjenje slojeva:

- `Embedding` pretvara riječi u vektore brojeva.
- `GlobalAveragePooling1D` pravi jednu zajedničku reprezentaciju cijele SMS poruke.
- `Dense` sloj uči razliku između spam i ham poruka.
- `Dropout` smanjuje rizik od overfittinga.
- `Sigmoid` izlaz daje vrijednost između 0 i 1.

Vrijednost bliža `0` znači `ham`, a vrijednost bliža `1` znači `spam`.

## Treniranje modela

Model je kompajliran sa:

```python
model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)
```

Korišteni hiperparametri:

```python
EPOCHS = 15
BATCH_SIZE = 32
```

Tokom treniranja korišten je i `validation_split=0.2`, što znači da se 20% trening podataka koristi za validaciju tokom treniranja.

## Rezultati

Model je ostvario dobre rezultate na test skupu.

Classification report:

```text
              precision    recall  f1-score   support

         ham       0.98      1.00      0.99       966
        spam       0.99      0.90      0.94       149

    accuracy                           0.98      1115
   macro avg       0.98      0.95      0.97      1115
weighted avg       0.98      0.98      0.98      1115
```

Accuracy modela je oko 98%.

Za spam klasu model ima:

- precision: 0.99
- recall: 0.90
- F1-score: 0.94

To znači da model veoma rijetko pogrešno označi normalnu poruku kao spam, ali ipak propusti dio stvarnih spam poruka.

## Zaključak

Model uspješno klasifikuje SMS poruke kao `spam` ili `ham`.

Pošto dataset nije balansiran, posebno je važno gledati rezultate za spam klasu, a ne samo ukupni accuracy. Model ima vrlo dobar precision za spam, što znači da kada označi poruku kao spam, uglavnom je u pravu. Recall za spam je 0.90, što znači da model pronalazi većinu spam poruka, ali ne baš sve.

Ovaj projekat pokazuje osnovni NLP pipeline:

```text
SMS tekst -> Tokenizer -> sekvence brojeva -> padding -> ANN model -> spam/ham
```

## Moguća poboljšanja

Neka moguća poboljšanja projekta su:

- podešavanje decision threshold vrijednosti
- poređenje sa klasičnim ML metodama kao što su Naive Bayes ili Logistic Regression
- korištenje TF-IDF reprezentacije
- testiranje drugih arhitektura neuronskih mreža
- pravljenje jednostavne web aplikacije za unos SMS poruke i prikaz predikcije

## Pokretanje projekta

Projekat se može pokrenuti u Google Colab-u ili lokalno u Jupyter Notebook okruženju.

Potrebne biblioteke:

```text
pandas
numpy
matplotlib
scikit-learn
tensorflow
```

Notebook automatski preuzima dataset pomoću komande:

```python
!wget -q https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip
!unzip -q -o sms+spam+collection.zip
```

Nakon toga je dovoljno pokrenuti ćelije redom.

## Autor

Tamir Oladejo
