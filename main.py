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
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


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
print(
    "No - redni broj obeležja. Da li su obeležja vremenski sortirana po rastućem redosledu?"
)
print()
print("year - godina.")
print("month - mesec.")
print(
    "day - dan. Da li je ispoštovano da mesec ima odgovarajući broj dana u godini?"
)
print()
print("hour - sat u danu. Da li svaki dan ima 24 sata?")
print()
print(
    "season - godišnje doba. Prvo godišnje doba počinje 1. marta, drugo 1. juna, treće 1. septembra i četvrto 1. decembra."
)
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
# TODO: da li postoje

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





""" ================================================ """
print_red(bcolors.BOLD + bcolors.UNDERLINE + "II DEO: ANALIZA PODATAKA")
""" ================================================ """
print_red(
    "1. Potrebno je 15% nasumično izabranih uzoraka ostaviti kao test skup, 15% kao validacioni a preostalih 70% koristiti za obuku modela."
)

X = df.loc[:, ['TEMP', 'PRES', 'DEWP', 'season', 'HUMI', 'cbwdx']]
X['pt'] = X['PRES'] * X['TEMP']
X['pd'] = X['PRES'] * X['DEWP']
X['td'] = X['TEMP'] * X['DEWP']
X['ptd'] = X['PRES'] * X['TEMP'] * X['DEWP']
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

print('Testiraće se modeli sa sledećim parametrima: sa i bez normalizacije, sa Lasso, Ridge i bez regularizacije, za različite veličine polinoma fitovanja, sa i bez interakcije fičera i različite grupe fičera, sa MSE i gradient descent metodom sa različitim faktorom učenja. ')
print('Ukupno ima 2*3*10=60 kombinacija ne računajući izbor fičera.')
print('Pretpostavka: treba prvo polinomijalne fičere napraviti pa onda skalirati.')


def test_model(normalization=False, regularization=None, degree=1, features=['TEMP', 'PRES', 'DEWP', 'season', 'HUMI', 'cbwdx'], method='MSE', alpha=None):
    # https://datascience.stackexchange.com/questions/20525/should-i-standardize-first-or-generate-polynomials-first
    print_yellow(f"Model: norm={normalization}, reg={regularization}, deg={degree}, method={method}")
    poly = PolynomialFeatures(degree, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)

    scaler_poly = StandardScaler()

    if normalization:
        X_train_scaled_poly = scaler_poly.fit_transform(X_train_poly)
    else:
        X_train_scaled_poly = X_train_poly

    scalers.append(scaler_poly)

    linear_model = LinearRegression(fit_intercept=True)
    linear_model.fit(X_train_scaled_poly, y_train)
    models.append(linear_model)

    yhat = linear_model.predict(X_train_scaled_poly)
    print(
        f"training MSE (using sklearn function): {mean_squared_error(y_train, yhat) / 2}"
    )
    train_mses.append(mean_squared_error(y_train, yhat) / 2)

    X_val_poly = poly.fit_transform(X_val)
    if normalization:
        X_val_scaled_poly = scaler_poly.transform(X_val_poly)
    else:
        X_val_scaled_poly = X_val_poly
    yhat_val = linear_model.predict(X_val_scaled_poly)
    print(f"Cross validation MSE: {mean_squared_error(yhat_val, y_val) / 2}")
    val_mses.append(mean_squared_error(yhat_val, y_val) / 2)

    print("-" * 100)

test_model(normalization=True)


for degree in range(1, 11):
    poly = PolynomialFeatures(degree, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)

    scaler_poly = StandardScaler()
    X_train_scaled_poly = scaler_poly.fit_transform(X_train_poly)

    scalers.append(scaler_poly)

    linear_model = LinearRegression(fit_intercept=True)
    linear_model.fit(X_train_scaled_poly, y_train)
    models.append(linear_model)

    yhat = linear_model.predict(X_train_scaled_poly)
    print(
        f"training MSE (using sklearn function): {mean_squared_error(y_train, yhat) / 2}"
    )
    train_mses.append(mean_squared_error(y_train, yhat) / 2)

    X_val_poly = poly.fit_transform(X_val)
    X_val_scaled_poly = scaler_poly.transform(X_val_poly)
    yhat_val = linear_model.predict(X_val_scaled_poly)
    print(f"Cross validation MSE: {mean_squared_error(yhat_val, y_val) / 2}")
    val_mses.append(mean_squared_error(yhat_val, y_val) / 2)

    print("-" * 100)

    # TODO : SGDRegress

degrees = range(1, 11)
utils.plot_train_cv_mses(degrees,
                         train_mses,
                         val_mses,
                         title="degree of polynomial vs. train and CV MSEs")

# X_test_poly = poly.fit_transform(X_test)
# X_test_scaled_poly = scaler_linear.transform(X_test_poly)
# yhat_test = linear_model.predict(X_test_scaled_poly)
# print(f"Cross validation MSE: {mean_squared_error(yhat_test, y_test) / 2}")


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


model_evaluation(y_test, yhat_test, X_train_scaled.shape[0],
                 X_train_scaled.shape[1])
plt.show()
