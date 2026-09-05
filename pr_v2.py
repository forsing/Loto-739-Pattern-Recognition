#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LOTO 7/39 — PATTERN RECOGNITION V2

Sistem kombinuje:
- Matrix Profile;
- detekciju promena režima;
- Hidden Markov model;
- gradient boosting regresor;
- grafovski model povezanosti brojeva;
- strogu hronološku validaciju.

Isti postupak se zasebno primenjuje na Loto i Loto Plus.
Prvi red CSV fajla predstavlja najstarije, a poslednji najnovije izvlačenje.
"""


from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import ruptures as rpt
import stumpy
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp
from scipy.stats import energy_distance, wasserstein_distance


# =============================================================================
# PODEŠAVANJA
# =============================================================================

SEED = 39

LOTO_CSV = Path(
    "/data/loto7_4680_k71_loto_2962.csv"
)

LOTO_PLUS_CSV = Path(
    "/data/loto7_4680_k71_loto_plus_1718.csv"
)

BROJ_KUGLICA = 39
BROJ_IZVUCENIH = 7

TEORIJSKA_STOPA = BROJ_IZVUCENIH / BROJ_KUGLICA
SLUCAJNO_OCEKIVANJE = BROJ_IZVUCENIH**2 / BROJ_KUGLICA
UKUPNO_KOMBINACIJA = math.comb(BROJ_KUGLICA, BROJ_IZVUCENIH)

MIN_ISTORIJA = 250
HORIZONT_METE = 5
DISKONT_METE = 0.65

HOLDOUT_UDEO = 0.15
MIN_HOLDOUT = 150
MAX_HOLDOUT = 400

BROJ_FOLDOVA = 4
PURGE = HORIZONT_METE + 7

HMM_REZIMI = 3
MATRIX_PROFILE_PROZOR = 20
PELT_PENAL = 8.0

PROZOR_TRANZICIJE = 300
PROZOR_GRAFA = 250
MAKSIMALNI_GAP = 100

BROJ_MONTE_KARLO_EPIZODA = 10_000
BROJ_BOOTSTRAP_PONAVLJANJA = 2_000
BROJ_PERMUTACIJA = 2_000
BOOTSTRAP_BLOK = 12

EPS = 1e-12

warnings.filterwarnings("ignore")


PARAMETRI = [
    {
        "num_leaves": 7,
        "max_depth": 3,
        "learning_rate": 0.025,
        "n_estimators": 350,
        "min_child_samples": 150,
        "reg_alpha": 2.0,
        "reg_lambda": 8.0,
    },
    {
        "num_leaves": 11,
        "max_depth": 4,
        "learning_rate": 0.020,
        "n_estimators": 450,
        "min_child_samples": 120,
        "reg_alpha": 3.0,
        "reg_lambda": 10.0,
    },
    {
        "num_leaves": 15,
        "max_depth": 5,
        "learning_rate": 0.015,
        "n_estimators": 550,
        "min_child_samples": 100,
        "reg_alpha": 4.0,
        "reg_lambda": 12.0,
    },
]


# =============================================================================
# REZULTAT
# =============================================================================

@dataclass
class Rezultat:
    naziv: str
    broj_redova: int
    next_kombinacija: np.ndarray
    nested_pogoci: np.ndarray
    holdout_pogoci: np.ndarray
    holdout_predikcije: np.ndarray
    bootstrap: np.ndarray
    monte_karlo: np.ndarray
    permutacije: np.ndarray
    ci_donji: float
    ci_gornji: float
    p_monte_karlo: float
    p_permutacija: float
    wasserstein: float
    energy: float
    stabilnost: list[float]
    parametri: dict[str, Any]
    vaznosti: list[tuple[str, float]]


# =============================================================================
# PODACI
# =============================================================================

def ucitaj_csv(putanja: Path) -> np.ndarray:
    if not putanja.exists():
        raise FileNotFoundError(f"CSV ne postoji: {putanja}")

    okvir = pd.read_csv(putanja, header=None)

    if okvir.shape[1] != BROJ_IZVUCENIH:
        okvir = pd.read_csv(putanja)

    if okvir.shape[1] != BROJ_IZVUCENIH:
        raise ValueError("CSV mora imati tačno sedam kolona.")

    okvir = okvir.apply(pd.to_numeric, errors="coerce")

    if okvir.isna().any().any():
        raise ValueError("CSV sadrži neispravne vrednosti.")

    izvlacenja = np.sort(
        okvir.to_numpy(dtype=np.int16),
        axis=1,
    )

    if np.any((izvlacenja < 1) | (izvlacenja > 39)):
        raise ValueError("Brojevi moraju biti između 1 i 39.")

    if np.any(np.diff(izvlacenja, axis=1) == 0):
        raise ValueError("Izvlačenje sadrži ponovljen broj.")

    return izvlacenja


def binarna_matrica(izvlacenja: np.ndarray) -> np.ndarray:
    b = np.zeros((len(izvlacenja), 39), dtype=np.float64)
    redovi = np.arange(len(izvlacenja))[:, None]
    b[redovi, izvlacenja - 1] = 1.0
    return b


def opis_izvlacenja(
    izvlacenja: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    n = len(izvlacenja)
    opis = np.zeros((n, 7), dtype=np.float64)

    for i, kombinacija in enumerate(izvlacenja):
        opis[i, 0] = np.mean(kombinacija) / 39
        opis[i, 1] = np.std(kombinacija) / 39
        opis[i, 2] = np.sum(kombinacija % 2) / 7
        opis[i, 3] = np.sum(kombinacija <= 19) / 7
        opis[i, 4] = np.sum(np.diff(kombinacija) == 1) / 6
        opis[i, 5] = (
            np.sum(b[i] * b[i - 1]) / 7
            if i > 0 else 0.0
        )
        opis[i, 6] = (
            np.sum(b[i] * b[i - 2]) / 7
            if i > 1 else 0.0
        )

    return opis


def standardizuj(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    sd = float(np.std(x))

    if sd < EPS or not np.isfinite(sd):
        return np.zeros_like(x)

    return (x - np.mean(x)) / sd


# =============================================================================
# MATRIX PROFILE
# =============================================================================

def matrix_profile_osobine(
    opis: np.ndarray,
) -> np.ndarray:
    n, k = opis.shape
    rezultat = np.ones((n, k), dtype=np.float64)
    m = MATRIX_PROFILE_PROZOR

    for kolona in range(k):
        serija = np.asarray(opis[:, kolona], dtype=np.float64)

        if len(serija) < 2 * m + 1:
            continue

        profil = stumpy.stump(
            serija,
            m=m,
            ignore_trivial=True,
        )

        for pocetak in range(len(profil)):
            levi = int(profil[pocetak, 2])
            kraj = pocetak + m - 1

            if levi < 0:
                continue

            a = standardizuj(
                serija[pocetak:pocetak + m]
            )
            prethodni = standardizuj(
                serija[levi:levi + m]
            )

            rezultat[kraj, kolona] = np.sqrt(
                np.mean((a - prethodni) ** 2)
            )

    return np.clip(rezultat, 0.0, 10.0)


# =============================================================================
# PELT I HMM
# =============================================================================

def poslednja_pelt_promena(opis: np.ndarray) -> int:
    if len(opis) < 80:
        return 0

    sredina = opis.mean(axis=0)
    sd = opis.std(axis=0)
    sd[sd < EPS] = 1.0
    z = (opis - sredina) / sd

    model = rpt.Pelt(
        model="rbf",
        min_size=30,
        jump=5,
    ).fit(z)

    tacke = model.predict(pen=PELT_PENAL)
    tacke = [t for t in tacke if t < len(opis)]

    return tacke[-1] if tacke else 0


def fit_hmm(
    opis: np.ndarray,
) -> tuple[GaussianHMM, np.ndarray, np.ndarray]:
    sredina = opis.mean(axis=0)
    sd = opis.std(axis=0)
    sd[sd < EPS] = 1.0

    z = (opis - sredina) / sd

    model = GaussianHMM(
        n_components=HMM_REZIMI,
        covariance_type="diag",
        n_iter=200,
        tol=1e-4,
        min_covar=1e-4,
        random_state=SEED,
    )

    model.fit(z)

    return model, sredina, sd


def hmm_forward(
    model: GaussianHMM,
    opis: np.ndarray,
    sredina: np.ndarray,
    sd: np.ndarray,
) -> np.ndarray:
    z = (opis - sredina) / sd
    emisije = model._compute_log_likelihood(z)

    rezultat = np.zeros(
        (len(z), HMM_REZIMI),
        dtype=np.float64,
    )

    alpha = (
        np.log(np.maximum(model.startprob_, EPS))
        + emisije[0]
    )
    alpha -= logsumexp(alpha)
    rezultat[0] = np.exp(alpha)

    log_tranzicija = np.log(
        np.maximum(model.transmat_, EPS)
    )

    for t in range(1, len(z)):
        novi = np.empty(HMM_REZIMI)

        for stanje in range(HMM_REZIMI):
            novi[stanje] = (
                emisije[t, stanje]
                + logsumexp(
                    alpha + log_tranzicija[:, stanje]
                )
            )

        novi -= logsumexp(novi)
        alpha = novi
        rezultat[t] = np.exp(alpha)

    return rezultat


# =============================================================================
# DISTRIBUCIJSKE I GRAFOVSKE OSOBINE
# =============================================================================

def frekvencijski_odnos(
    b: np.ndarray,
    t: int,
    prozor: int | None,
) -> np.ndarray:
    pocetak = 0 if prozor is None else max(0, t - prozor)
    uzorak = b[pocetak:t]
    n = len(uzorak)

    brojanja = uzorak.sum(axis=0)
    ocekivanje = max(n * TEORIJSKA_STOPA, EPS)

    return (
        brojanja + ocekivanje
    ) / (
        2.0 * ocekivanje
    )


def vremenska_stopa(
    b: np.ndarray,
    t: int,
    poluzivot: float = 60.0,
) -> np.ndarray:
    istorija = b[:t]

    starost = np.arange(
        len(istorija) - 1,
        -1,
        -1,
    )

    w = np.exp(
        -math.log(2.0) * starost / poluzivot
    )

    return (
        istorija * w[:, None]
    ).sum(axis=0) / np.sum(w)


def gap_hazard(
    b: np.ndarray,
    t: int,
) -> tuple[np.ndarray, np.ndarray]:
    gap = np.zeros(39)
    hazard = np.full(39, TEORIJSKA_STOPA)

    for broj in range(39):
        pozicije = np.flatnonzero(b[:t, broj])

        if len(pozicije) == 0:
            gap[broj] = 1.0
            continue

        trenutni = min(
            t - 1 - int(pozicije[-1]),
            MAKSIMALNI_GAP,
        )

        gap[broj] = trenutni / MAKSIMALNI_GAP

        if len(pozicije) >= 2:
            istorijski = np.minimum(
                np.diff(pozicije) - 1,
                MAKSIMALNI_GAP,
            )
            rizik = np.sum(istorijski >= trenutni)
            dogadjaj = np.sum(istorijski == trenutni)
        else:
            rizik = 0
            dogadjaj = 0

        prior = 20.0

        hazard[broj] = (
            dogadjaj + prior * TEORIJSKA_STOPA
        ) / (
            rizik + prior
        )

    return gap, hazard


def tranzicija(
    b: np.ndarray,
    t: int,
) -> np.ndarray:
    if t < 2:
        return np.full(39, TEORIJSKA_STOPA)

    pocetak = max(1, t - PROZOR_TRANZICIJE)
    prethodni = b[pocetak - 1:t - 1]
    naredni = b[pocetak:t]
    poslednji = b[t - 1]

    slicnost = prethodni @ poslednji
    w = 1.0 + slicnost
    prior = 20.0

    return (
        (naredni * w[:, None]).sum(axis=0)
        + prior * TEORIJSKA_STOPA
    ) / (
        np.sum(w) + prior
    )


def graf(
    b: np.ndarray,
    t: int,
) -> tuple[np.ndarray, np.ndarray]:
    pocetak = max(0, t - PROZOR_GRAFA)
    uzorak = b[pocetak:t]

    parovi = uzorak.T @ uzorak
    np.fill_diagonal(parovi, 0.0)

    ocekivanje = max(
        len(uzorak) * 7 * 6 / (39 * 38),
        EPS,
    )

    odstupanje = (
        parovi - ocekivanje
    ) / math.sqrt(ocekivanje + EPS)

    poslednji = np.flatnonzero(b[t - 1])

    prema_poslednjem = (
        odstupanje[:, poslednji].mean(axis=1)
        if len(poslednji)
        else np.zeros(39)
    )

    centralnost = odstupanje.mean(axis=1)

    return (
        standardizuj(prema_poslednjem),
        standardizuj(centralnost),
    )


def change_score(
    opis: np.ndarray,
    t: int,
) -> float:
    if t < 80:
        return 0.0

    skorovi = []

    for prozor in (20, 40):
        stariji = opis[t - 2 * prozor:t - prozor]
        noviji = opis[t - prozor:t]

        sd = opis[:t].std(axis=0)
        sd[sd < EPS] = 1.0

        skorovi.append(
            np.mean(
                np.abs(
                    noviji.mean(axis=0)
                    - stariji.mean(axis=0)
                ) / sd
            )
        )

    return float(np.mean(skorovi))


NAZIVI_OSOBINA = [
    "broj_norm",
    "broj_sin",
    "broj_cos",
    "frekvencija_20",
    "frekvencija_50",
    "frekvencija_100",
    "frekvencija_200",
    "frekvencija_sve",
    "promena_20_100",
    "promena_50_200",
    "vremenska_stopa",
    "gap",
    "hazard",
    "tranzicija",
    "graf_poslednji",
    "graf_centralnost",
    "hmm_1",
    "hmm_2",
    "hmm_3",
    "pelt_starost",
    "change_score",
    "matrix_profile_1",
    "matrix_profile_2",
    "matrix_profile_3",
    "matrix_profile_4",
    "matrix_profile_5",
    "matrix_profile_6",
    "matrix_profile_7",
]


def osobine(
    b: np.ndarray,
    opis: np.ndarray,
    mp: np.ndarray,
    hmm_p: np.ndarray,
    pelt_promena: int,
    t: int,
) -> np.ndarray:
    f20 = frekvencijski_odnos(b, t, 20)
    f50 = frekvencijski_odnos(b, t, 50)
    f100 = frekvencijski_odnos(b, t, 100)
    f200 = frekvencijski_odnos(b, t, 200)
    fsve = frekvencijski_odnos(b, t, None)

    vreme = vremenska_stopa(b, t)
    gap, hazard = gap_hazard(b, t)
    trans = tranzicija(b, t)
    graf_poslednji, graf_centralnost = graf(b, t)

    brojevi = np.arange(1, 40)
    ugao = 2.0 * np.pi * brojevi / 39

    kontekst_indeks = min(t - 1, len(hmm_p) - 1)
    hmm_t = hmm_p[kontekst_indeks]
    mp_t = mp[min(t - 1, len(mp) - 1)]

    pelt_starost = (
        max(0, t - pelt_promena) / max(t, 1)
    )

    x = np.column_stack(
        [
            brojevi / 39,
            np.sin(ugao),
            np.cos(ugao),
            f20,
            f50,
            f100,
            f200,
            fsve,
            f20 - f100,
            f50 - f200,
            vreme,
            gap,
            hazard,
            trans,
            graf_poslednji,
            graf_centralnost,
            np.full(39, hmm_t[0]),
            np.full(39, hmm_t[1]),
            np.full(39, hmm_t[2]),
            np.full(39, pelt_starost),
            np.full(39, change_score(opis, t)),
            *[
                np.full(39, vrednost)
                for vrednost in mp_t
            ],
        ]
    )

    return x.astype(np.float32)


# =============================================================================
# KONTINUIRANA DISTRIBUCIJSKA META
# =============================================================================

def kontinuirana_meta(
    b: np.ndarray,
    t: int,
) -> np.ndarray:
    tezine = np.asarray(
        [
            DISKONT_METE**h
            for h in range(HORIZONT_METE)
        ],
        dtype=np.float64,
    )

    dostupno = min(
        HORIZONT_METE,
        len(b) - t,
    )

    if dostupno <= 0:
        raise ValueError("Meta nema buduća izvlačenja.")

    tezine = tezine[:dostupno]
    buducnost = b[t:t + dostupno]

    meta = (
        buducnost * tezine[:, None]
    ).sum(axis=0) / np.sum(tezine)

    return meta.astype(np.float32)


def dataset(
    b: np.ndarray,
    opis: np.ndarray,
    mp: np.ndarray,
    hmm_p: np.ndarray,
    pelt_promena: int,
    pocetak: int,
    kraj: int,
    meta: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    x_delovi = []
    y_delovi = []

    stvarni_kraj = (
        min(kraj, len(b) - HORIZONT_METE + 1)
        if meta
        else kraj
    )

    for t in range(pocetak, stvarni_kraj):
        x_delovi.append(
            osobine(
                b,
                opis,
                mp,
                hmm_p,
                pelt_promena,
                t,
            )
        )

        if meta:
            y_delovi.append(kontinuirana_meta(b, t))

    if not x_delovi:
        raise RuntimeError("Nema dovoljno podataka za dataset.")

    x = np.vstack(x_delovi)
    y = np.concatenate(y_delovi) if meta else None

    return x, y


# =============================================================================
# GRADIENT BOOSTING REGRESOR
# =============================================================================

def napravi_model(
    parametri: dict[str, Any],
) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="regression_l1",
        random_state=SEED,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
        n_jobs=-1,
        subsample=1.0,
        colsample_bytree=0.85,
        **parametri,
    )


def top_sedam(
    skorovi: np.ndarray,
) -> np.ndarray:
    brojevi = np.arange(1, 40)
    redosled = np.lexsort((brojevi, -skorovi))
    return np.sort(redosled[:7] + 1)


def predikcije(
    model: lgb.LGBMRegressor,
    x: np.ndarray,
) -> np.ndarray:
    broj_grupa = len(x) // 39
    skorovi = model.predict(x)

    return np.asarray(
        [
            top_sedam(
                skorovi[i * 39:(i + 1) * 39]
            )
            for i in range(broj_grupa)
        ],
        dtype=np.int16,
    )


def pogoci(
    pred: np.ndarray,
    stvarno: np.ndarray,
) -> np.ndarray:
    return np.asarray(
        [
            np.intersect1d(p, s).size
            for p, s in zip(pred, stvarno)
        ],
        dtype=np.int8,
    )


def fit_test(
    izvlacenja: np.ndarray,
    b: np.ndarray,
    opis: np.ndarray,
    mp: np.ndarray,
    train_kraj: int,
    test_pocetak: int,
    test_kraj: int,
    parametri: dict[str, Any],
) -> tuple[lgb.LGBMRegressor, np.ndarray, np.ndarray]:
    hmm, sredina, sd = fit_hmm(opis[:train_kraj])

    hmm_p = hmm_forward(
        hmm,
        opis[:test_kraj],
        sredina,
        sd,
    )

    pelt_promena = poslednja_pelt_promena(
        opis[:train_kraj]
    )

    x_train, y_train = dataset(
        b,
        opis,
        mp,
        hmm_p,
        pelt_promena,
        MIN_ISTORIJA,
        train_kraj,
        meta=True,
    )

    x_test, _ = dataset(
        b,
        opis,
        mp,
        hmm_p,
        pelt_promena,
        test_pocetak,
        test_kraj,
        meta=False,
    )

    model = napravi_model(parametri)
    model.fit(
        x_train,
        y_train,
        feature_name=NAZIVI_OSOBINA,
    )

    pred = predikcije(model, x_test)
    hit = pogoci(
        pred,
        izvlacenja[test_pocetak:test_kraj],
    )

    return model, pred, hit


# =============================================================================
# NESTED WALK-FORWARD
# =============================================================================

def nested_walk_forward(
    izvlacenja: np.ndarray,
    b: np.ndarray,
    opis: np.ndarray,
    mp: np.ndarray,
    razvoj_kraj: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    prvi_test = max(
        MIN_ISTORIJA + 350,
        int(razvoj_kraj * 0.55),
    )

    granice = np.linspace(
        prvi_test,
        razvoj_kraj,
        BROJ_FOLDOVA + 1,
        dtype=int,
    )

    spoljasnji_rezultati = []
    zbir_skorova = np.zeros(len(PARAMETRI))

    for fold in range(BROJ_FOLDOVA):
        test_pocetak = int(granice[fold])
        test_kraj = int(granice[fold + 1])
        train_kraj = test_pocetak - PURGE

        unutrasnji_test_duzina = max(
            100,
            int((train_kraj - MIN_ISTORIJA) * 0.20),
        )

        unutrasnji_test_pocetak = (
            train_kraj - unutrasnji_test_duzina
        )

        unutrasnji_train_kraj = (
            unutrasnji_test_pocetak - PURGE
        )

        print(f"  Nested fold {fold + 1}/{BROJ_FOLDOVA}")

        skorovi = []

        for i, parametri in enumerate(PARAMETRI):
            _, _, hit = fit_test(
                izvlacenja,
                b,
                opis,
                mp,
                unutrasnji_train_kraj,
                unutrasnji_test_pocetak,
                train_kraj,
                parametri,
            )

            skor = float(np.mean(hit))
            skorovi.append(skor)
            zbir_skorova[i] += skor

        najbolji = int(np.argmax(skorovi))

        _, _, spoljasnji_hit = fit_test(
            izvlacenja,
            b,
            opis,
            mp,
            train_kraj,
            test_pocetak,
            test_kraj,
            PARAMETRI[najbolji],
        )

        spoljasnji_rezultati.append(spoljasnji_hit)

    finalni = int(np.argmax(zbir_skorova))

    return (
        np.concatenate(spoljasnji_rezultati),
        PARAMETRI[finalni],
    )


# =============================================================================
# STATISTIČKA PROVERA
# =============================================================================

def blok_bootstrap(
    hit: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    n = len(hit)
    rezultat = np.empty(BROJ_BOOTSTRAP_PONAVLJANJA)

    for i in range(BROJ_BOOTSTRAP_PONAVLJANJA):
        uzorak = []

        while len(uzorak) < n:
            pocetak = int(rng.integers(0, n))
            indeksi = (
                pocetak + np.arange(BOOTSTRAP_BLOK)
            ) % n
            uzorak.extend(hit[indeksi].tolist())

        rezultat[i] = np.mean(uzorak[:n])

    return rezultat


def monte_karlo(
    n: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    epizode = rng.hypergeometric(
        ngood=7,
        nbad=32,
        nsample=7,
        size=(BROJ_MONTE_KARLO_EPIZODA, n),
    )

    return epizode, epizode.mean(axis=1)


def permutacioni_test(
    pred: np.ndarray,
    stvarno: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    rezultat = np.empty(BROJ_PERMUTACIJA)

    for i in range(BROJ_PERMUTACIJA):
        permutovano = stvarno[
            rng.permutation(len(stvarno))
        ]

        rezultat[i] = np.mean(
            pogoci(pred, permutovano)
        )

    return rezultat


def p_vrednost(
    raspodela: np.ndarray,
    posmatrano: float,
) -> float:
    return float(
        (
            np.sum(raspodela >= posmatrano) + 1
        ) / (
            len(raspodela) + 1
        )
    )


# =============================================================================
# OBRADA IGRE
# =============================================================================

def obradi(
    naziv: str,
    putanja: Path,
    seed_pomeraj: int,
) -> Rezultat:
    print()
    print("=" * 78)
    print(f"Obrada: {naziv}")
    print("=" * 78)

    izvlacenja = ucitaj_csv(putanja)
    b = binarna_matrica(izvlacenja)
    opis = opis_izvlacenja(izvlacenja, b)

    n = len(izvlacenja)

    holdout = min(
        MAX_HOLDOUT,
        max(MIN_HOLDOUT, round(n * HOLDOUT_UDEO)),
    )

    holdout_pocetak = n - holdout

    print(f"CSV: {putanja}")
    print(f"Broj redova: {n}")
    print(f"Zamrznuti holdout: {holdout}")

    print("Causal Matrix Profile...")
    mp = matrix_profile_osobine(opis)

    print("Nested walk-forward...")
    nested_hit, najbolji_parametri = nested_walk_forward(
        izvlacenja,
        b,
        opis,
        mp,
        holdout_pocetak,
    )

    print("Zamrznuta holdout provera...")
    model, holdout_pred, holdout_hit = fit_test(
        izvlacenja,
        b,
        opis,
        mp,
        holdout_pocetak - PURGE,
        holdout_pocetak,
        n,
        najbolji_parametri,
    )

    rng = np.random.default_rng(SEED + seed_pomeraj)

    mc_epizode, mc_proseci = monte_karlo(
        len(holdout_hit),
        rng,
    )

    bootstrap = blok_bootstrap(holdout_hit, rng)

    permutacije = permutacioni_test(
        holdout_pred,
        izvlacenja[holdout_pocetak:],
        rng,
    )

    prosek = float(np.mean(holdout_hit))
    ci_donji, ci_gornji = np.quantile(
        bootstrap,
        [0.025, 0.975],
    )

    nulti_hit = mc_epizode.ravel()

    # NEXT model nad svim dostupnim redovima.
    hmm, sredina, sd = fit_hmm(opis)
    hmm_p = hmm_forward(hmm, opis, sredina, sd)
    pelt_promena = poslednja_pelt_promena(opis)

    x_sve, y_sve = dataset(
        b,
        opis,
        mp,
        hmm_p,
        pelt_promena,
        MIN_ISTORIJA,
        n,
        meta=True,
    )

    next_model = napravi_model(najbolji_parametri)

    next_model.fit(
        x_sve,
        y_sve,
        feature_name=NAZIVI_OSOBINA,
    )

    x_next = osobine(
        b,
        opis,
        mp,
        hmm_p,
        pelt_promena,
        n,
    )

    next_kombinacija = top_sedam(
        next_model.predict(x_next)
    )

    vaznosti = next_model.booster_.feature_importance(
        importance_type="gain"
    )

    if np.sum(vaznosti) > 0:
        vaznosti = vaznosti / np.sum(vaznosti) * 100

    sortirane_vaznosti = sorted(
        zip(NAZIVI_OSOBINA, vaznosti.tolist()),
        key=lambda par: par[1],
        reverse=True,
    )

    return Rezultat(
        naziv=naziv,
        broj_redova=n,
        next_kombinacija=next_kombinacija,
        nested_pogoci=nested_hit,
        holdout_pogoci=holdout_hit,
        holdout_predikcije=holdout_pred,
        bootstrap=bootstrap,
        monte_karlo=mc_proseci,
        permutacije=permutacije,
        ci_donji=float(ci_donji),
        ci_gornji=float(ci_gornji),
        p_monte_karlo=p_vrednost(mc_proseci, prosek),
        p_permutacija=p_vrednost(permutacije, prosek),
        wasserstein=float(
            wasserstein_distance(
                holdout_hit.astype(float),
                nulti_hit.astype(float),
            )
        ),
        energy=float(
            energy_distance(
                holdout_hit.astype(float),
                nulti_hit.astype(float),
            )
        ),
        stabilnost=[
            float(np.mean(segment))
            for segment in np.array_split(holdout_hit, 4)
        ],
        parametri=najbolji_parametri,
        vaznosti=sortirane_vaznosti,
    )


# =============================================================================
# ISPIS
# =============================================================================

def formatiraj(kombinacija: np.ndarray) -> str:
    return ", ".join(
        f"{int(broj):02d}"
        for broj in kombinacija
    )


def zakljucak(rezultat: Rezultat) -> str:
    prosek = float(np.mean(rezultat.holdout_pogoci))

    znacajno = (
        rezultat.ci_donji > SLUCAJNO_OCEKIVANJE
        and rezultat.p_monte_karlo < 0.05
        and rezultat.p_permutacija < 0.05
        and sum(
            x > SLUCAJNO_OCEKIVANJE
            for x in rezultat.stabilnost
        ) >= 3
    )

    if prosek > SLUCAJNO_OCEKIVANJE and znacajno:
        return (
            "DA — pronađena prednost je statistički značajna "
            "i stabilna na zamrznutom holdoutu."
        )

    return (
        "NE — nije dokazana statistički pouzdana i vremenski "
        "stabilna prednost nad slučajnim izborom."
    )


def ispisi(rezultat: Rezultat) -> None:
    holdout_prosek = float(
        np.mean(rezultat.holdout_pogoci)
    )

    nested_prosek = float(
        np.mean(rezultat.nested_pogoci)
    )

    mc_donji, mc_gornji = np.quantile(
        rezultat.monte_karlo,
        [0.025, 0.975],
    )

    print()
    print("=" * 78)
    print(rezultat.naziv)
    print("=" * 78)
    print(f"NEXT: {formatiraj(rezultat.next_kombinacija)}")
    print(f"CSV redova: {rezultat.broj_redova}")

    print()
    print("HRONOLOŠKA PROVERA")
    print("-" * 78)
    print(f"Nested walk-forward:              {nested_prosek:.6f}")
    print(f"Zamrznuti holdout:                {holdout_prosek:.6f}")
    print(f"Slučajno očekivanje:              {SLUCAJNO_OCEKIVANJE:.6f}")
    print(
        f"Razlika prema slučajnom:          "
        f"{holdout_prosek - SLUCAJNO_OCEKIVANJE:+.6f}"
    )

    print()
    print("STATISTIČKA PROVERA")
    print("-" * 78)
    print(
        f"Blok-bootstrap 95%:               "
        f"[{rezultat.ci_donji:.6f}, {rezultat.ci_gornji:.6f}]"
    )
    print(
        f"Monte Karlo 95%:                  "
        f"[{mc_donji:.6f}, {mc_gornji:.6f}]"
    )
    print(f"Monte Karlo p-vrednost:           {rezultat.p_monte_karlo:.6f}")
    print(f"Permutaciona p-vrednost:          {rezultat.p_permutacija:.6f}")
    print(f"Wasserstein distanca:             {rezultat.wasserstein:.6f}")
    print(f"Energy distanca:                  {rezultat.energy:.6f}")

    print()
    print("STABILNOST HOLDOUTA")
    print("-" * 78)

    for i, vrednost in enumerate(rezultat.stabilnost, 1):
        print(f"Segment {i}:                       {vrednost:.6f}")

    print()
    print("NAJVAŽNIJE OSOBINE")
    print("-" * 78)

    for naziv, vrednost in rezultat.vaznosti[:10]:
        print(f"{naziv:<38} {vrednost:>8.3f}%")

    print()
    print("ODGOVOR NA GLAVNO PITANJE")
    print("-" * 78)
    print(zakljucak(rezultat))


# =============================================================================
# GLAVNI PROGRAM
# =============================================================================

def main() -> None:
    np.random.seed(SEED)

    print("=" * 78)
    print("LOTO 7/39 — PATTERN RECOGNITION V2")
    print("=" * 78)
    print(f"Seed: {SEED}")
    print(f"Teorijska stopa broja: {TEORIJSKA_STOPA:.9f}")
    print(f"Teorijsko očekivanje: {SLUCAJNO_OCEKIVANJE:.9f}")
    print(f"Ukupno retirement combinations: {UKUPNO_KOMBINACIJA:,}")

    loto = obradi("Loto", LOTO_CSV, 0)
    loto_plus = obradi("Loto Plus", LOTO_PLUS_CSV, 1)

    print()
    print("#" * 78)
    print("KONAČNE NEXT PREDIKCIJE")
    print("#" * 78)

    ispisi(loto)
    ispisi(loto_plus)


if __name__ == "__main__":
    main()



"""
==============================================================================
LOTO 7/39 — PATTERN RECOGNITION V2
==============================================================================
Seed: 39
Teorijska stopa broja: 0.179487179
Teorijsko očekivanje: 1.256410256
Ukupno retirement combinations: 15,380,937

==============================================================================
Obrada: Loto
==============================================================================
CSV: /data/loto7_4680_k71_loto_2962.csv
Broj redova: 2962
Zamrznuti holdout: 400
Causal Matrix Profile...
Nested walk-forward...
  Nested fold 1/4
  Nested fold 2/4
  Nested fold 3/4
  Nested fold 4/4
Zamrznuta holdout provera...

==============================================================================
Obrada: Loto Plus
==============================================================================
CSV: /data/loto7_4680_k71_loto_plus_1718.csv
Broj redova: 1718
Zamrznuti holdout: 258
Causal Matrix Profile...
Nested walk-forward...
  Nested fold 1/4
  Nested fold 2/4
  Nested fold 3/4
  Nested fold 4/4
Zamrznuta holdout provera...

##############################################################################
KONAČNE NEXT PREDIKCIJE
##############################################################################

==============================================================================
Loto
==============================================================================
NEXT: 09, x, 22, y, 26, z, 33
CSV redova: 2962

HRONOLOŠKA PROVERA
------------------------------------------------------------------------------
Nested walk-forward:              1.250650
Zamrznuti holdout:                1.230000
Slučajno očekivanje:              1.256410
Razlika prema slučajnom:          -0.026410

STATISTIČKA PROVERA
------------------------------------------------------------------------------
Blok-bootstrap 95%:               [1.130000, 1.335000]
Monte Karlo 95%:                  [1.165000, 1.350000]
Monte Karlo p-vrednost:           0.722728
Permutaciona p-vrednost:          0.169415
Wasserstein distanca:             0.033955
Energy distanca:                  0.027107

STABILNOST HOLDOUTA
------------------------------------------------------------------------------
Segment 1:                       1.120000
Segment 2:                       1.360000
Segment 3:                       1.170000
Segment 4:                       1.270000

NAJVAŽNIJE OSOBINE
------------------------------------------------------------------------------
frekvencija_sve                          11.197%
graf_centralnost                          8.693%
promena_50_200                            6.517%
vremenska_stopa                           6.404%
frekvencija_200                           5.666%
frekvencija_100                           5.086%
promena_20_100                            4.920%
broj_norm                                 4.889%
broj_sin                                  4.457%
matrix_profile_7                          3.870%

ODGOVOR NA GLAVNO PITANJE
------------------------------------------------------------------------------
NE — nije dokazana statistički pouzdana i vremenski stabilna prednost nad slučajnim izborom.

==============================================================================
Loto Plus
==============================================================================
NEXT: 02, x, 09, y, 26, z, 38
CSV redova: 1718

HRONOLOŠKA PROVERA
------------------------------------------------------------------------------
Nested walk-forward:              1.267884
Zamrznuti holdout:                1.201550
Slučajno očekivanje:              1.256410
Razlika prema slučajnom:          -0.054860

STATISTIČKA PROVERA
------------------------------------------------------------------------------
Blok-bootstrap 95%:               [1.093023, 1.317829]
Monte Karlo 95%:                  [1.143411, 1.372093]
Monte Karlo p-vrednost:           0.831517
Permutaciona p-vrednost:          0.394303
Wasserstein distanca:             0.068632
Energy distanca:                  0.056450

STABILNOST HOLDOUTA
------------------------------------------------------------------------------
Segment 1:                       1.276923
Segment 2:                       0.953846
Segment 3:                       1.296875
Segment 4:                       1.281250

NAJVAŽNIJE OSOBINE
------------------------------------------------------------------------------
frekvencija_sve                          10.125%
graf_centralnost                          8.411%
frekvencija_200                           7.392%
vremenska_stopa                           5.720%
promena_20_100                            5.670%
change_score                              5.642%
broj_sin                                  5.013%
promena_50_200                            4.813%
broj_norm                                 4.794%
broj_cos                                  4.427%

ODGOVOR NA GLAVNO PITANJE
------------------------------------------------------------------------------
NE — nije dokazana statistički pouzdana i vremenski stabilna prednost nad slučajnim izborom.
"""



"""
matrix profile + change-point detection + Hidden Markov režimi + gradient boosting + grafovski model + stroga hronološka validacija


V2 koristi isti pattern-recognition sistem, ali umesto LambdaRank-a koristi LightGBM gradient-boosting regresor sa stvarno kontinuiranom distribucijskom metom:
Time naredno izvlačenje ima najveću težinu, ali meta nije binarna.
"""
