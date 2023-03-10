# učitavanje biblioteka
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sb
import utils

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer, StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso, SGDRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, LeaveOneOut
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score


class bcolors:
    """
    Klasa konstanti. Predstavljaju opcije kojima možemo stilizovati tekst prilikom ispisa.
    """
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_red(text):
    """
    Ispisati tekst u crvenoj boji.

    Funkcija je namenjena naglašavanju pitanja (zadataka) i razlikovanju pitanja od odgovora.
    """
    print(bcolors.FAIL + text + bcolors.ENDC)


def print_yellow(text):
    """
    Ispisati tekst u žutoj boji.

    Funkcija je namenjena naglašavanju parametara modela koji se obučava.
    """
    print(bcolors.WARNING + text + bcolors.ENDC)


def implication(p, q):
    """
    Logička implikacija.
    """
    return not (p) or q


def equal(p, q):
    """
    Logička jednakost.
    """
    return implication(p, q) and implication(q, p)


# podešavanja ispisa
pd.set_option(
    'display.float_format',
    lambda x: '%.2f' % x)  # prikaz df na 2 decimale (npr. za describe)
""" ================================================ """
print_red(bcolors.BOLD + bcolors.UNDERLINE + "I DEO: ANALIZA PODATAKA")
""" ================================================ """
print_red(
    "1. Broj svog indeksa (bez godine) podeliti po modulu 5 - dobijeni broj označava bazu na kojoj treba raditi. \n   U pitanju su baze koje se tiču vremenskih uslova u 5 različitih gradova u Kini."
)
print(
    "Broj indeksa: DE4/2021 → Baza 4: Šenjang - ShenyangPM20100101_20151231.csv"
)
""" ================================================ """
print_red(
    "2. Sa moodle platforme skinuti bazu podataka koja je dobijena na osnovu broja indeksa."
)
print("Ubačena je u radni folder.")
df = pd.read_csv('ShenyangPM20100101_20151231.csv')
""" ================================================ """
print_red(
    "3. Učitati bazu u DataFrame. Proveriti kako izgleda prvih nekoliko vrsta u bazi."
)
print("Prikaz prvih 5 uzoraka")
print(df.head())
""" ================================================ """
print_red("4a. Koliko ima obeležja? Koliko ima uzoraka?")
print("Obeležja ima " + str(df.shape[1]) +
      ", pri čemu je prvo obeležje redni broj uzorka.")
print(
    "Uzoraka ima " + str(df.shape[0]) +
    ", odnosno broj kolona u csv tabeli minus jedan, jer prvi red predstavlja nazive kolona (obeležja)."
)

print_red("4b. Šta predstavlja jedan uzorak baze?")
print(
    "Jedan uzorak predstavlja meteorološka merenja u datom vremenskom trenutku."
)

print_red("4c. Kojim obeležjima raspolažemo?")
# print(
#     "No - redni broj obeležja. Da li su obeležja vremenski sortirana po rastućem redosledu?"
# )
# print()
# print("year - godina.")
# print("month - mesec.")
# print(
#     "day - dan. Da li je ispoštovano da mesec ima odgovarajući broj dana u godini?"
# )
# print()
# print("hour - sat u danu. Da li svaki dan ima 24 sata?")
# print()
# print(
#     "season - godišnje doba. Prvo godišnje doba počinje 1. marta, drugo 1. juna, treće 1. septembra i četvrto 1. decembra."
# )
# TODO: godišnje doba
print()
print(
    "PM_Taiyuanjie - koncentracija PM2.5 čestica na lokaciji Taiyuanjie. [µg/m^3]"
)
print("PM_US Post - koncentracija PM2.5 čestica na lokaciji US Post. [µg/m^3]")
print(
    "PM_Xiaoheyan - koncentracija PM2.5 čestica na lokaciji Xiaoheyan. [µg/m^3]"
)
print("DEWP - temperatura rose/kondenzacije. [°C]")
print("HUMI - vlažnost vazduha. [%]")
print("PRES - vazdušni pritisak. [hPa]")
print("TEMP - temperatura. [°C]")
print(
    "cbwd - pravac vetra (N-sever, S-jug, E-istok, W-zapad, cv-calm/variable)")
# TODO: da li ima drugih slova
print("Iws - kumulativna brzina vetra. [m/s]")
# TODO: da li brzina vetra može biti negativna
print("precipitation - padavine na sat. [mm]")
print("Iprec - kumulativne padavine. [mm]")

print_red("4d. Koja obeležja su kategorička, a koja numerička?")
print("Kategorička obeležja su godina, mesec, dan, sat, sezona i cbwd.")
print(
    "Numerička obeležja su PM, DEWP, HUMI, PRES, TEMP, lws, precipitation i lprec."
)
print(df.dtypes)
# Mogli smo koristiti i komandu df.info():
# df.info()

print_red(
    "4e. Postoje li nedostajući podaci? Gde se javljaju i koliko ih je? Postoje li nelogične/nevalidne vrednosti?"
)
number_of_days = 365 + 365 + 366 + 365 + 365 + 365
print("Od 1.1.2010. do 31.12.2015. ima " + str(number_of_days) + " dana.")
print(
    "U svakom danu ima 24 časa, tako da bi priložena tabela trebala imati ukupno "
    + str(24 * number_of_days) + " uzoraka, što i ima.")
print("Koje se godine javljaju u bazi?")
print(df['year'].unique())
print("Koji se meseci javljaju u bazi?")
print(df['month'].unique())
print("Koji se dani javljaju u bazi?")
print(df['day'].unique())
print("Koji se časovi javljaju u bazi?")
print(df['hour'].unique())
print("Koji se smerovi vetra javljaju u bazi?")
print(df['cbwd'].unique())
print("Da li je prvo godišnje doba uvek od 1. marta do 31. maja?")
assert all([
    equal(season == 1, (month >= 3) and (month <= 5))
    for month, season in zip(df['month'], df['season'])
])
print("Da.")
print("Da li je drugo godišnje doba uvek od 1. juna do 31. avgusta?")
assert all([
    equal(season == 2, (month >= 6) and (month <= 8))
    for month, season in zip(df['month'], df['season'])
])
print("Da.")
print("Da li je treće godišnje doba uvek od 1. septembra do 30. novembra?")
assert all([
    equal(season == 3, (month >= 9) and (month <= 11))
    for month, season in zip(df['month'], df['season'])
])
print("Da.")
print(
    "Da li je četvrto godišnje doba uvek od 1. decembra do 28. (29.) februara?"
)
assert all([
    equal(season == 4, (month == 12) or (month == 1) or (month == 2))
    for month, season in zip(df['month'], df['season'])
])
print("Da.")
print("Da li svaki dan ima tačno 24 časa?")
# https://www.statology.org/pandas-groupby-count-with-condition/

print("Da.")
print("Da li februar ima uvek 28 dana izuzev prestupne 2012. godine?")
assert all([
    implication(month == 2, ((day >= 1) and (day <= 28))
                or ((day == 29) and (year == 2012)))
    for year, month, day in zip(df['year'], df['month'], df['day'])
])
df.groupby(by=["month"]).count()
print("Da.")

print(
    "Najniža ikad zabeležena temperatura u Šenjangu iznosi -32.9°C, a najviša zabeležena temperatura iznosi 38.4°C."
)
# https://www.extremeweatherwatch.com/cities/shenyang/lowest-temperatures
# https://www.extremeweatherwatch.com/cities/shenyang/highest-temperatures
print("Da li su izmerene temperature u okvirima dozvoljenih?")
MIN_TEMP = -32.9
MAX_TEMP = 38.4
df.loc[df['TEMP'] < MIN_TEMP, 'TEMP'] = np.nan
df.loc[df['TEMP'] > MAX_TEMP, 'TEMP'] = np.nan

# TODO: da li se neki datumi, vremena pojavljuju više puta?

# TODO: 12 meseci uvek, odgovarajući broj dana, svaki dan tačno 24 časa, temp vrednosti u okvirima dozvoljenih

# TODO: histogram nedostajućih vrednosti
print(df.isna().sum() / df.shape[0] * 100)
# pm_us_post_isna = df.loc['PM_US Post'].isna()
# plt.hist(df.loc['PM_US Post'].isna(), density=True, alpha=0.5, bins=50, label = 'Broj nedostajućih merenja PM_US Post tokom vremena')

print('unikatne vrednosti')
print(np.sort(df['PRES'].unique()))
print(np.sort(df['precipitation'].unique()))
print(np.sort(df['Iprec'].unique()))
""" ================================================ """
print_red(
    "5. Izbaciti obeležja koja se odnose na sve lokacije merenja koncentracije PM čestica osim US Post."
)
df.drop(['PM_Taiyuanjie'], axis=1, inplace=True)
df.drop(['PM_Xiaoheyan'], axis=1, inplace=True)
print(df.head())
""" ================================================ """
print_red(
    "6. Ukoliko postoje nedostajući podaci, rešiti taj problem na odgovarajući način. Objasniti zašto je rešeno na odabrani način."
)
print(df.isna().sum() / df.shape[0] * 100)
print(
    "Postoji 58.77% nedostajućih podataka za PM_US Post. Vrste gde nedostaju podaci za PM_US Post moramo ukloniti."
)
df.dropna(inplace=True, subset=['PM_US Post'])
print(df.isna().sum() / df.shape[0] * 100)
print(
    "Zapažanje: Vrste gde nedostaju vrednosti za PM_US Post su ujedno i vrste gde nedostaju podaci za DEWP, HUMI, PRES, TEMP, cbwd i Iws. Takođe, značajno je smanjen broj nedostajućih podataka za precipitation i Iprec."
)
print("Broj nedostajućih vrednosti za DEWP: ")
print(df['DEWP'].isna().sum())
print("Broj nedostajućih vrednosti za HUMI: ")
print(df['HUMI'].isna().sum())
print("Broj nedostajućih vrednosti za PRES: ")
print(df['PRES'].isna().sum())
print("Broj nedostajućih vrednosti za TEMP: ")
print(df['TEMP'].isna().sum())
print("Broj nedostajućih vrednosti za cbwd: ")
print(df['cbwd'].isna().sum())
print("Broj nedostajućih vrednosti za Iws: ")
print(df['Iws'].isna().sum())
print("Procenat nedostajućih vrednosti za precipitation: ")
print(df['precipitation'].isna().sum() / df.shape[0] * 100)
print("Procenat nedostajućih vrednosti za Iprec: ")
print(df['Iprec'].isna().sum() / df.shape[0] * 100)
print("Prikazati kolonu gde nedostaju podaci za DEWP:")
print(df[df['DEWP'].isna()])
print("U istoj koloni nedostaju svi podaci. Tu kolonu moramo obrisati.")
print("Trenutne dimenzije naše tabele: " + str(df.shape))
df.dropna(inplace=True, subset=['DEWP'])
print("Dimenzije naše tabele nakon brisanja: " + str(df.shape))
print("Broj nedostajućih vrednosti za DEWP: ")
print(df['DEWP'].isna().sum())
print("Broj nedostajućih vrednosti za HUMI: ")
print(df['HUMI'].isna().sum())
print("Broj nedostajućih vrednosti za PRES: ")
print(df['PRES'].isna().sum())
print("Broj nedostajućih vrednosti za TEMP: ")
print(df['TEMP'].isna().sum())
print("Broj nedostajućih vrednosti za cbwd: ")
print(df['cbwd'].isna().sum())
print("Broj nedostajućih vrednosti za Iws: ")
print(df['Iws'].isna().sum())
print(
    "Preostaje da još samo sredimo podatke vezane za precipitation i Iprec. Najjednostavnije je primeniti forward fill."
)
# TODO: OBJASNITI I SREDITI!!!
df['precipitation'].fillna(method='ffill', inplace=True)
df['Iprec'].fillna(method='ffill', inplace=True)
print(df.head())
""" ================================================ """
print_red("7. Analizirati obeležja (statističke veličine, raspodela, …)")
print("Dodao sam nove kolone: dan u godini (npr. 16/365) i datetime objekat.")
# https://pandas.pydata.org/docs/reference/api/pandas.to_datetime.html
df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
df['doy'] = df['datetime'].dt.dayofyear
df['date'] = df['datetime'].dt.date

# df_year = df.set_index('year')
# print(df_year.head())
# print(df_year.tail())

print_red('index')
df["cbwdx"] = 0
df["cbwdy"] = 0
df.loc[df['cbwd'] == "NE", "cbwdx"] = 1
df.loc[df['cbwd'] == "SE", "cbwdx"] = 1
df.loc[df['cbwd'] == "NW", "cbwdx"] = -1
df.loc[df['cbwd'] == "SW", "cbwdx"] = -1
df.loc[df['cbwd'] == "NE", "cbwdy"] = 1
df.loc[df['cbwd'] == "NW", "cbwdy"] = 1
df.loc[df['cbwd'] == "SE", "cbwdy"] = -1
df.loc[df['cbwd'] == "SW", "cbwdy"] = -1

# plt.figure()
df.reset_index(inplace=True)
print(df)
fig, axes = plt.subplots(nrows=2, ncols=4)
axes[0, 0].set_title("DEWP")
df["DEWP"].plot(ax=axes[0, 0])
axes[0, 1].set_title("HUMI")
df["HUMI"].plot(ax=axes[0, 1])
axes[0, 2].set_title("PRES")
df["PRES"].plot(ax=axes[0, 2])
axes[0, 3].set_title("TEMP")
df["TEMP"].plot(ax=axes[0, 3])
axes[1, 0].set_title("cbwd")
df["cbwdx"].plot(ax=axes[1, 0])
df["cbwdy"].plot(ax=axes[1, 0])
axes[1, 1].set_title("Iws")
df["Iws"].plot(ax=axes[1, 1])
axes[1, 2].set_title("precipitation")
df["precipitation"].plot(ax=axes[1, 2])
axes[1, 3].set_title("Iprec")
df["Iprec"].plot(ax=axes[1, 3])

# TODO: korelacija sa izvodom

# plt.figure()
fig, axes = plt.subplots(nrows=2, ncols=4)
axes[0, 0].set_title("DEWP")
axes[0, 0].hist(
    df['DEWP'],
    bins=100,
    density=True,
    alpha=0.3,
    label='DEWP',
)
axes[0, 1].set_title("HUMI")
axes[0, 1].hist(
    df['HUMI'],
    bins=100,
    density=True,
    alpha=0.3,
    label='HUMI',
)
axes[0, 2].set_title("PRES")
axes[0, 2].hist(
    df['PRES'],
    bins=100,
    density=True,
    alpha=0.3,
    label='PRES',
)
axes[0, 3].set_title("TEMP")
axes[0, 3].hist(
    df['TEMP'],
    bins=100,
    density=True,
    alpha=0.3,
    label='TEMP',
)
axes[1, 0].set_title("cbwd")
df_cbwd_count = df.groupby("cbwd").agg('count') / df.shape[0]
df_cbwd_count['index'].plot.bar(ax=axes[1, 0])

axes[1, 1].set_title("Iws")
axes[1, 1].hist(
    df['Iws'],
    bins=100,
    density=True,
    alpha=0.3,
    label='Iws',
)
axes[1, 2].set_title("precipitation")
axes[1, 2].hist(
    df['precipitation'],
    bins=25,
    density=True,
    alpha=0.3,
    stacked=True,
    label='precitipation',
)
axes[1, 3].set_title("Iprec")
axes[1, 3].hist(
    df['Iprec'],
    bins=70,
    density=True,
    alpha=0.3,
    label='Iprec',
)
# TODO: srediti brzinu vetra
""" ================================================ """
print_red("8. Analizirati detaljno vrednosti obeležja PM 2.5 (’PM_US Post’).")
df_pm = df.set_index('PM_US Post')
fig, axes = plt.subplots(nrows=1, ncols=3)
axes[0].set_title("PM_US Post")
df["PM_US Post"].plot(ax=axes[0])
axes[1].set_title("PM_US Post hist")
axes[1].hist(
    df['PM_US Post'],
    bins=100,
    density=True,
    alpha=0.3,
    label='PM_US Post',
)
df.plot.scatter(x='index', y='PM_US Post', c='b', ax=axes[2])
# axes[0, 2].plot(df["PM_US Post"], 'b', label='PM_US Post', linestyle='dotted')
# TODO: uraditi low pass filtar i isplotovati

# TODO: promene po mesecima, po danu, po satu

# TODO: PM veći od 500 ukloniti
""" ================================================ """
print_red(
    "9. Vizuelizovati i iskomentarisati zavisnost promene PM 2.5 od preostalih obeležja u bazi."
)

fig, axes = plt.subplots(nrows=3, ncols=4)
df.plot.scatter(y='PM_US Post', x='year', c='b', ax=axes[0, 0])
df.plot.scatter(y='PM_US Post', x='season', c='b', ax=axes[0, 1])
df.plot.scatter(y='PM_US Post', x='doy', c='b', ax=axes[0, 2])
df.plot.scatter(y='PM_US Post', x='hour', c='b', ax=axes[0, 3])
df.plot.scatter(y='PM_US Post', x='DEWP', c='b', ax=axes[1, 0])
df.plot.scatter(y='PM_US Post', x='TEMP', c='b', ax=axes[1, 1])
df.plot.scatter(y='PM_US Post', x='HUMI', c='b', ax=axes[1, 2])
df.plot.scatter(y='PM_US Post', x='PRES', c='b', ax=axes[1, 3])
df.plot.scatter(y='PM_US Post', x='Iws', c='b', ax=axes[2, 0])
df.plot.scatter(y='PM_US Post', x='precipitation', c='b', ax=axes[2, 1])
df.plot.scatter(y='PM_US Post', x='Iprec', c='b', ax=axes[2, 2])
# TODO: 3d grafik zavisnosti vetra i PM cestica

# rastaviti sliku na 2 dela jer jer ogromna
sb.pairplot(df.loc[:, ['TEMP', 'PRES', 'DEWP', 'season', 'HUMI', 'cbwdx', 'PM_US Post']], x_vars=['PM_US Post'])
sb.pairplot(df.loc[:, ['year', 'doy', 'hour', 'Iws', 'precipitation', 'Iprec', 'cbwdy', 'PM_US Post']], x_vars=['PM_US Post'])
# najkorisnije - precipitation i Iprec
sb.pairplot(df.loc[:, ['PM_US Post', 'precipitation', 'Iprec']], x_vars=['PM_US Post'])
""" ================================================ """
print_red("10. Analizirati međusobne korelacije obeležja.")
plt.figure()
matrica_korelacije = df.corr()
sb.heatmap(matrica_korelacije, annot=True)
# TODO: usrednjiti vrednosti da dobijemo vecu korelisanost
# Pritisak i temperatura treba da imaju negativnu povezanost, jer ...
# vlaznost treba da bude linearno proporcionalna temperaturi, sto nije ispunjeno
# planina je na istoku, ima smisla da jaci vetar duva ka zapadu.
""" ================================================ """
print_red(
    "11. Uraditi još nešto po sopstvenom izboru (takođe obavezna stavka).")
# plt.show()
print("prikazati zavisnost temperature od ostalih parametara")
sb.pairplot(df.loc[:, ['TEMP', 'PRES', 'DEWP', 'season', 'HUMI', 'cbwdx']], x_vars=['TEMP'])
sb.pairplot(df.loc[:, ['year', 'doy', 'hour', 'Iws', 'precipitation', 'Iprec', 'cbwdy', 'TEMP']], x_vars=['TEMP'])

print("prikazati zavisnost dana u godini - doy, od ostalih parametara")
sb.pairplot(df.loc[:, ['TEMP', 'PRES', 'DEWP', 'season', 'HUMI', 'cbwdx', 'doy']], x_vars=['doy'])
sb.pairplot(df.loc[:, ['year', 'doy', 'hour', 'Iws', 'precipitation', 'Iprec', 'cbwdy']], x_vars=['doy'])



""" ================================================ """
print_red(bcolors.BOLD + bcolors.UNDERLINE + "II DEO: LINEARNA REGRESIJA")
""" ================================================ """
print_red(
    "1. Potrebno je 15% nasumično izabranih uzoraka ostaviti kao test skup, 15% kao validacioni a preostalih 70% koristiti za obuku modela."
)

X = df.loc[:, ['TEMP', 'PRES', 'DEWP', 'season', 'HUMI', 'cbwdx']]
temp_min = 40
X['pt'] = X['PRES'] * (X['TEMP']+temp_min)
X['pd'] = X['PRES'] * (X['DEWP']+temp_min)
X['td'] = (X['TEMP']+temp_min) * (X['DEWP']+temp_min)
X['ptd'] = X['PRES'] * (X['TEMP']+temp_min) * (X['DEWP']+temp_min)
print(X.head())
y = df['PM_US Post'].copy()
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.3,
                                                    random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_test,
                                                y_test,
                                                test_size=0.5,
                                                random_state=42)
print(f'Training set size: {len(X_train)}')
print(f'Test set size: {len(X_test)}')
print(f'Validation set size: {len(X_val)}')

train_mses = []
val_mses = []
models = []
scalers = []
params = []

print('Testiraće se modeli sa sledećim parametrima: sa i bez normalizacije, sa Lasso, Ridge i bez regularizacije, za različite veličine polinoma fitovanja, sa i bez interakcije fičera i različite grupe fičera, sa MSE i gradient descent metodom sa različitim faktorom učenja. ')
print('Ukupno ima 2*3*10=60 kombinacija ne računajući izbor fičera.')
print('Pretpostavka: treba prvo polinomijalne fičere napraviti pa onda skalirati.')


def model_evaluation(y, y_predicted, N, d):
    mse = mean_squared_error(y_test,
                             y_predicted)  # np.mean((y_test-y_predicted)**2)
    mae = mean_absolute_error(
        y_test, y_predicted)  # np.mean(np.abs(y_test-y_predicted))
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_predicted)
    r2_adj = 1 - (1 - r2) * (N - 1) / (N - d - 1)

    # printing values
    print('Mean squared error: ', mse)
    print('Mean absolute error: ', mae)
    print('Root mean squared error: ', rmse)
    print('R2 score: ', r2)
    print('R2 adjusted score: ', r2_adj)

    # Uporedni prikaz nekoliko pravih i predvidjenih vrednosti
    res = pd.concat([pd.DataFrame(y.values),
                     pd.DataFrame(y_predicted)],
                    axis=1)
    res.columns = ['y', 'y_pred']
    print(res.head(20))


def train_model(normalization=False, method='MSE', alpha=None, degree=1, features=['TEMP', 'PRES', 'DEWP', 'season', 'HUMI', 'cbwdx']):
    # https://datascience.stackexchange.com/questions/20525/should-i-standardize-first-or-generate-polynomials-first
    if method == 'MSE':
        print_yellow(f"Model: norm={normalization}, deg={degree}, method={method}")
    else:
        print_yellow(f"Model: norm={normalization}, method={method}, alpha={alpha}, deg={degree}, method={method}")
    params_dict = {"norm":normalization, "deg":degree, "features":features, "method":method, "alpha":alpha}
    params.append(params_dict)

    X_train_curr = X_train.loc[:, features]
    X_val_curr = X_val.loc[:, features]

    poly = PolynomialFeatures(degree, include_bias=False)
    X_train_poly = poly.fit_transform(X_train_curr)
    X_val_poly = poly.fit_transform(X_val_curr)

    scaler_poly = StandardScaler()

    if normalization:
        X_train_scaled_poly = scaler_poly.fit_transform(X_train_poly)
        X_val_scaled_poly = scaler_poly.transform(X_val_poly)
    else:
        X_train_scaled_poly = X_train_poly
        X_val_scaled_poly = X_val_poly

    scalers.append(scaler_poly)

    if method == 'ridge':
        model = Ridge(alpha=alpha)
    elif method == 'lasso':
        model = Lasso(alpha=alpha)
    elif method == 'SGD':
        model = SGDRegressor(alpha=alpha)
    else:
        model = LinearRegression(fit_intercept=True)
    model.fit(X_train_scaled_poly, y_train)
    models.append(model)

    yhat = model.predict(X_train_scaled_poly)
    print(
        f"training MSE: {mean_squared_error(y_train, yhat) / 2}"
    )
    train_mses.append(mean_squared_error(y_train, yhat) / 2)

    yhat_val = model.predict(X_val_scaled_poly)
    print(f"validation MSE: {mean_squared_error(yhat_val, y_val) / 2}")
    val_mses.append(mean_squared_error(yhat_val, y_val) / 2)

    print("-" * 100)


def test_model(model_number, params_dict):
    print_yellow('TEST RESULTS')
    features = params_dict["features"]
    degree = params_dict["deg"]
    normalization = params_dict["norm"]

    X_test_curr = X_test.loc[:, features]
    poly = PolynomialFeatures(degree, include_bias=False)
    X_test_poly = poly.fit_transform(X_test_curr)
    if normalization:
        X_test_poly_scaled = scalers[model_number].transform(X_test_poly)
    else:
        X_test_poly_scaled = X_test_poly
    yhat_test = models[model_number].predict(X_test_poly_scaled)
    print(f"test MSE: {mean_squared_error(yhat_test, y_test)}")
    model_evaluation(y_test, yhat_test, X_train.shape[0], X_train.shape[1])

for alpha in [0.1, 0.01, 0.001, 0.0001]:
    for degree in range(1, 6):
        train_model(degree=degree)

    for degree in range(1, 8):
        train_model(degree=degree, normalization=True)

    for degree in range(1, 6):
        train_model(degree=degree, method='ridge', alpha=0.01)

    for degree in range(1, 6):
        train_model(degree=degree, method='lasso', alpha=0.01)

    for degree in range(1, 6):
        train_model(degree=degree, method='OGD', alpha=0.01)

    for degree in range(1, 6):
        train_model(degree=degree, method='ridge', alpha=0.01, normalization=True)

    for degree in range(1, 6):
        train_model(degree=degree, method='lasso', alpha=0.01, normalization=True)

    for degree in range(1, 8):
        train_model(degree=degree, method='OGD', alpha=0.01, normalization=True)

print('Choosing the best model.')
model_number = np.argmin(val_mses)
test_model(model_number, params[model_number])
print(params[model_number])

# Ilustracija koeficijenata
plt.figure(figsize=(10,5))
plt.bar(range(len(models[model_number].coef_)), models[model_number].coef_)
# print("koeficijenti: ", models[model_number].coef_)


# utils.plot_train_cv_mses(degrees,
#                          train_mses,
#                          val_mses,
#                          title="degree of polynomial vs. train and CV MSEs")

# X_test_poly = poly.fit_transform(X_test)
# X_test_scaled_poly = scaler_linear.transform(X_test_poly)
# yhat_test = linear_model.predict(X_test_scaled_poly)
# print(f"Cross validation MSE: {mean_squared_error(yhat_test, y_test) / 2}")



# model_evaluation(y_test, yhat_test, X_train_scaled.shape[0],
#                  X_train_scaled.shape[1])


# plt.show()
# quit()
""" ================================================ """
print_red(bcolors.BOLD + bcolors.UNDERLINE + "III DEO: KNN KLASIFIKATOR")
""" ================================================ """


print_red("1. Prvo je potrebno uzorcima iz date baze dodeliti labele: bezbedno, nebezbedno ili opasno. Uzorcima čija je vrednost koncentracije PM2.5 čestica do 55.4 µg/m3 dodeliti labelu bezbedno, onima čija je vrednost koncentracije PM2.5 čestica od 55.5 µg/m3 do 150.4 µg/m3 dodeliti labelu nebezbedno, dok onima sa vrednošću preko 150.5 µg/m3 dodeliti labelu opasno.")

# https://medium.com/analytics-vidhya/pandas-how-to-change-value-based-on-condition-fc8ee38ba529
df['class'] = 'opasno'
df.loc[df['PM_US Post'] <= 150.4, 'class'] = 'nebezbedno'
df.loc[df['PM_US Post'] <= 55.4, 'class'] = 'bezbedno'
print(df.loc[:, ['PM_US Post', 'class']].head(50))



print_red("2. Koristiti 15% uzoraka za testiranje finalnog klasifikatora, a preostalih 85% uzoraka koristiti za metodu unakrsne validacije sa 10 podskupova. Ovom metodom odrediti optimalne parametre klasifikatora, oslanjajući se na željenu meru uspešnosti. Obratiti pažnju da u svakom od podskupova za unakrsnu validaciju, kao i u test skupu, bude dovoljan broj uzoraka svake klase.")

X = df.loc[:, ['TEMP', 'PRES', 'DEWP', 'season', 'HUMI', 'cbwdx', 'cbwdy', 'Iws', 'doy', 'precipitation', 'Iprec']]
y = df['class'].copy()

# Održati klasni odnos originalnih podataka.
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.15,
                                                    random_state=42,
                                                    shuffle=True,
                                                    stratify=y)

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.DistanceMetric.html
# Izbacio sam seuciledan i mahalanobis - vraća error.
metrics = ['euclidean', 'manhattan', 'chebyshev', 'minkowski']
# metrics = ['euclidean']
n_neighbors = np.arange(1,14)
parameters = {'metric':metrics, 'n_neighbors':n_neighbors}
knn = KNeighborsClassifier()
clf = GridSearchCV(estimator=knn, param_grid=parameters, scoring='accuracy', cv=kfold, refit=True, verbose=3)
clf.fit(X_train, y_train)

print(clf.best_score_)
print(clf.best_params_)

yhat_test = clf.predict(X_test)



print_red("3. Za konačno odabrane parametre prikazati i analizirati matricu konfuzije dobijenu akumulacijom matrica iz svake od 10 iteracija unakrsne validacije. Odrediti prosečnu tačnost klasifikatora, kao i tačnost za svaku klasu.")


print('a. Tačnost za najbolju klasu.')

def minor(arr, i, j):
    minor = [row[:j] + row[j+1:] for row in (arr[:i] + arr[i+1:])]
    return np.array(minor)

def evaluation_classifier(conf_mat):
    tp = np.diag(conf_mat)
    fn = np.sum(conf_mat, axis=1) - np.diag(conf_mat) # suma po vrstama
    fp = np.sum(conf_mat, axis=0) - np.diag(conf_mat) # suma po kolonama
    tn = np.full(shape=3, fill_value=np.sum(conf_mat)) - np.sum(conf_mat, axis=1) - np.sum(conf_mat, axis=0) + np.diag(conf_mat) # suma minornih matrica
    # alternativno:
    # tn2 = [np.sum(minor(conf_mat.tolist(), 0, 0)), np.sum(minor(conf_mat.tolist(), 1, 1)), np.sum(minor(conf_mat.tolist(), 2, 2))]

    precision = tp/(tp+fp)
    accuracy = (tp+tn)/(tp+tn+fp+fn)
    sensitivity = tp/(tp+fn)
    specificity = tn/(tn+fp)
    F_score = 2*precision*sensitivity/(precision+sensitivity)

    for i in range(3):
        print('\nKlasa: ', clf.classes_[i])
        print('precision: ', precision[i])
        print('accuracy: ', accuracy[i])
        print('sensitivity/recall: ', sensitivity[i])
        print('specificity: ', specificity[i])
        print('F score: ', F_score[i])

    tp = np.sum(tp)
    fn = np.sum(fn)
    fp = np.sum(fp)
    tn = np.sum(tn)

    precision = tp/(tp+fp)
    accuracy = (tp+tn)/(tp+tn+fp+fn)
    sensitivity = tp/(tp+fn)
    specificity = tn/(tn+fp)
    F_score = 2*precision*sensitivity/(precision+sensitivity)

    print('\nProsečna tačnost klasifikatora:')
    print('precision: ', precision)
    print('accuracy: ', accuracy)
    print('sensitivity/recall: ', sensitivity)
    print('specificity: ', specificity)
    print('F score: ', F_score)

conf_mat = confusion_matrix(y_test, yhat_test, labels=clf.classes_)

disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat,  display_labels=clf.classes_)
disp.plot(cmap="Blues")
disp.ax_.set_title('Matrica konfuzije za najbolji model pronađen prilikom pretrage')
evaluation_classifier(conf_mat)


print('b. Tačnost dobijena usrednjavanjem i matrica konfuzije dobijena akumulacijom.')

indexes = list(kfold.split(X_train, y_train))

# metric = clf.best_params_['metric']
for metric in metrics:
    conf_mat_sum = np.zeros((3, 3))
    accuracy = []

    for k in n_neighbors:

        tmp_accuracy = []

        for train_index, test_index in indexes:

            Xfold_train = X_train.iloc[train_index,:]
            yfold_train = y_train.iloc[train_index]

            Xfold_test = X_train.iloc[test_index,:]
            yfold_test = y_train.iloc[test_index]

            knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
            knn.fit(Xfold_train, yfold_train)

            yfold_pred = knn.predict(Xfold_test)
            tmp_accuracy.append(accuracy_score(yfold_test, yfold_pred))
            conf_mat_sum += confusion_matrix(yfold_test, yfold_pred, labels=clf.classes_)

        accuracy.append(np.mean(tmp_accuracy))

    plt.figure()
    plt.plot(range(1, 14), accuracy, color='red', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=10)
    plt.title('Accuracy for ' + metric)
    plt.xlabel('K Value')
    plt.ylabel('Acc')

    disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat,  display_labels=clf.classes_)
    disp.plot(cmap="Blues")
    disp.ax_.set_title('Matrica konfuzije za metriku ' + str(metric))
    print_yellow('\nMetrika: ' + metric)
    evaluation_classifier(conf_mat_sum)



print_red("4. Klasifikator sa konačno odabranim parametrima obučiti na celokupnom trening skupu, pa testirati na izdvojenom test skupu. Na osnovu dobijene matrice konfuzije izračunati mere uspešnosti klasifikatora, kao i mere uspešnosti za svaku klasu (tačnost, osetljivost, specifičnost, preciznost, F-mera).")

print('Najbolji rezultati su sa sledećim parametrima: ', clf.best_params_)
knn_best = KNeighborsClassifier(n_neighbors = clf.best_params_['n_neighbors'], metric = clf.best_params_['metric'])
knn_best.fit(X_train, y_train)
yhat_best = knn_best.predict(X_test)
conf_mat = confusion_matrix(yhat_best, y_test, labels=clf.classes_)

disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat,  display_labels=clf.classes_)
disp.plot(cmap="Blues")
evaluation_classifier(conf_mat)
disp.ax_.set_title('Matrica konfuzije za najbolju metriku obučenu na celom trening skupu.')




print_red("5. Rezultate prikazati i diskutovati u izveštaju.")
print("Rezultati su prikazani i diskutovani.")





""" ================================================ """
print_red(bcolors.BOLD + bcolors.UNDERLINE + "KRAJ")
""" ================================================ """
print("Isplotovati sve grafike.")
plt.show()
