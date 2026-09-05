#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LOTO 7/39 — FINALNI PATTERN RECOGNITION SISTEM

Jedna konačna arhitektura:

1. Causal Matrix Profile motivi i anomalije
2. PELT change-point detection
3. Hidden Markov režimi
4. Gap / survival / hazard distribucije
5. Uslovne tranzicije
6. Grafovske osobine parova
7. LightGBM LambdaRank
8. Nested walk-forward izbor parametara
9. Potpuno zamrznuti završni holdout
10. Blok-bootstrap
11. Permutacioni test
12. Monte Karlo poštenog Loto 7/39 procesa

Svako izvlačenje predstavlja grupu od 39 kandidata. LambdaRank direktno
rangira svih 39 brojeva, a prvih sedam čini NEXT kombinaciju.

Prvi CSV red je najstariji, poslednji je najnoviji.
Loto i Loto Plus obrađuju se potpuno zasebno.
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
    "/Users/4c/Desktop/GHQ/data/loto7_4680_k71_loto_2962.csv"
)

LOTO_PLUS_CSV = Path(
    "/Users/4c/Desktop/GHQ/data/loto7_4680_k71_loto_plus_1718.csv"
)

BROJ_KUGLICA = 39
BROJ_IZVUCENIH = 7

TEORIJSKA_STOPA = BROJ_IZVUCENIH / BROJ_KUGLICA
SLUCAJNO_OCEKIVANJE = BROJ_IZVUCENIH**2 / BROJ_KUGLICA
UKUPNO_KOMBINACIJA = math.comb(BROJ_KUGLICA, BROJ_IZVUCENIH)

MIN_ISTORIJA = 250

HOLDOUT_UDEO = 0.15
MIN_HOLDOUT = 150
MAX_HOLDOUT = 400

BROJ_SPOLJNIH_FOLDOVA = 4
UNUTRASNJI_VALIDACIONI_UDEO = 0.20
PURGE_KORACI = 7

HMM_REZIMI = 3
PELT_PENAL = 8.0
MATRIX_PROFILE_PROZOR = 20

PROZORI_FREKVENCIJE = (20, 50, 100, 200)
PROZOR_TRANZICIJE = 300
PROZOR_GRAFA = 250
MAKSIMALNI_GAP = 100

BROJ_MONTE_KARLO_EPIZODA = 10_000
BROJ_BOOTSTRAP_PONAVLJANJA = 2_000
BROJ_PERMUTACIJA = 2_000
BOOTSTRAP_BLOK = 12

EPS = 1e-12

warnings.filterwarnings("ignore")


PARAMETRI_RANKERA = [
    {
        "num_leaves": 7,
        "max_depth": 3,
        "learning_rate": 0.025,
        "n_estimators": 300,
        "min_child_samples": 120,
        "reg_alpha": 1.0,
        "reg_lambda": 5.0,
    },
    {
        "num_leaves": 11,
        "max_depth": 4,
        "learning_rate": 0.020,
        "n_estimators": 400,
        "min_child_samples": 100,
        "reg_alpha": 2.0,
        "reg_lambda": 8.0,
    },
    {
        "num_leaves": 15,
        "max_depth": 5,
        "learning_rate": 0.015,
        "n_estimators": 500,
        "min_child_samples": 90,
        "reg_alpha": 3.0,
        "reg_lambda": 10.0,
    },
]


# =============================================================================
# REZULTAT
# =============================================================================

@dataclass
class RezultatIgre:
    naziv: str
    broj_redova: int
    next_kombinacija: np.ndarray
    izabrani_parametri: dict[str, Any]
    nested_pogoci: np.ndarray
    holdout_pogoci: np.ndarray
    holdout_predikcije: np.ndarray
    holdout_stvarni: np.ndarray
    bootstrap_proseci: np.ndarray
    monte_karlo_proseci: np.ndarray
    permutacioni_proseci: np.ndarray
    ci_donji: float
    ci_gornji: float
    p_monte_karlo: float
    p_permutacija: float
    wasserstein: float
    energy: float
    stabilnost: list[float]
    vaznost_osobina: list[tuple[str, float]]


# =============================================================================
# UČITAVANJE PODATAKA
# =============================================================================

def ucitaj_csv(putanja: Path) -> np.ndarray:
    if not putanja.exists():
        raise FileNotFoundError(f"CSV fajl ne postoji: {putanja}")

    okvir = pd.read_csv(putanja, header=None)

    if okvir.shape[1] != BROJ_IZVUCENIH:
        okvir = pd.read_csv(putanja)

    if okvir.shape[1] != BROJ_IZVUCENIH:
        raise ValueError(
            f"{putanja} mora imati tačno sedam kolona."
        )

    okvir = okvir.apply(pd.to_numeric, errors="coerce")

    if okvir.isna().any().any():
        raise ValueError(
            f"{putanja} sadrži prazne ili nenumeričke vrednosti."
        )

    izvlacenja = np.sort(
        okvir.to_numpy(dtype=np.int16),
        axis=1,
    )

    if np.any((izvlacenja < 1) | (izvlacenja > BROJ_KUGLICA)):
        raise ValueError("Brojevi moraju biti između 1 i 39.")

    if np.any(np.diff(izvlacenja, axis=1) == 0):
        raise ValueError(
            "Jedno ili više izvlačenja sadrži ponovljen broj."
        )

    if len(izvlacenja) < MIN_ISTORIJA + MIN_HOLDOUT:
        raise ValueError("Nema dovoljno istorijskih izvlačenja.")

    return izvlacenja


def napravi_binarnu_matricu(
    izvlacenja: np.ndarray,
) -> np.ndarray:
    matrica = np.zeros(
        (len(izvlacenja), BROJ_KUGLICA),
        dtype=np.float64,
    )

    redovi = np.arange(len(izvlacenja))[:, None]
    matrica[redovi, izvlacenja - 1] = 1.0

    return matrica


# =============================================================================
# OPIS IZVLAČENJA
# =============================================================================

def opis_izvlacenja(
    izvlacenja: np.ndarray,
    binarno: np.ndarray,
) -> np.ndarray:
    n = len(izvlacenja)
    opis = np.zeros((n, 7), dtype=np.float64)

    for i, kombinacija in enumerate(izvlacenja):
        opis[i, 0] = np.mean(kombinacija) / BROJ_KUGLICA
        opis[i, 1] = np.std(kombinacija) / BROJ_KUGLICA
        opis[i, 2] = np.sum(kombinacija % 2 == 1) / BROJ_IZVUCENIH
        opis[i, 3] = np.sum(kombinacija <= 19) / BROJ_IZVUCENIH
        opis[i, 4] = np.sum(np.diff(kombinacija) == 1) / 6.0
        opis[i, 5] = (
            np.sum(binarno[i] * binarno[i - 1]) / BROJ_IZVUCENIH
            if i > 0
            else 0.0
        )
        opis[i, 6] = (
            np.sum(binarno[i] * binarno[i - 2]) / BROJ_IZVUCENIH
            if i > 1
            else 0.0
        )

    return opis


def standardizuj_vektor(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    sd = float(np.std(x))

    if not np.isfinite(sd) or sd < EPS:
        return np.zeros_like(x)

    return (x - float(np.mean(x))) / sd


# =============================================================================
# CAUSAL MATRIX PROFILE
# =============================================================================

def znormalizovana_distanca(
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    a = standardizuj_vektor(a)
    b = standardizuj_vektor(b)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def causal_matrix_profile(
    opis: np.ndarray,
    prozor: int = MATRIX_PROFILE_PROZOR,
) -> np.ndarray:
    """
    STUMPY pronalazi levi, isključivo raniji sused svake podsekvence.
    Tako osobina za trenutak t ne koristi buduća izvlačenja.
    """

    n, broj_opisa = opis.shape
    rezultat = np.ones((n, broj_opisa), dtype=np.float64)

    for kolona in range(broj_opisa):
        serija = np.asarray(opis[:, kolona], dtype=np.float64)

        if len(serija) < 2 * prozor + 1:
            continue

        profil = stumpy.stump(
            serija,
            m=prozor,
            ignore_trivial=True,
        )

        for pocetak in range(len(profil)):
            levi_indeks = int(profil[pocetak, 2])
            kraj_podsekvence = pocetak + prozor

            if levi_indeks < 0:
                rezultat[kraj_podsekvence - 1, kolona] = 1.0
                continue

            trenutna = serija[pocetak:pocetak + prozor]
            prethodna = serija[
                levi_indeks:levi_indeks + prozor
            ]

            rezultat[kraj_podsekvence - 1, kolona] = (
                znormalizovana_distanca(trenutna, prethodna)
            )

    # Samo prethodne vrednosti popunjavaju početak.
    for kolona in range(broj_opisa):
        poslednja = 1.0

        for i in range(n):
            vrednost = rezultat[i, kolona]

            if np.isfinite(vrednost):
                poslednja = vrednost
            else:
                rezultat[i, kolona] = poslednja

    return np.clip(rezultat, 0.0, 10.0)


# =============================================================================
# CHANGE-POINT I HMM REŽIMI
# =============================================================================

def poslednja_pelt_promena(
    opis_obuke: np.ndarray,
) -> int:
    if len(opis_obuke) < 80:
        return 0

    skalirano = opis_obuke.copy()
    sredina = skalirano.mean(axis=0)
    sd = skalirano.std(axis=0)
    sd[sd < EPS] = 1.0
    skalirano = (skalirano - sredina) / sd

    model = rpt.Pelt(
        model="rbf",
        min_size=30,
        jump=5,
    ).fit(skalirano)

    tacke = model.predict(pen=PELT_PENAL)
    validne = [t for t in tacke if t < len(opis_obuke)]

    return validne[-1] if validne else 0


def fit_hmm(
    opis_obuke: np.ndarray,
) -> tuple[GaussianHMM, np.ndarray, np.ndarray]:
    sredina = opis_obuke.mean(axis=0)
    sd = opis_obuke.std(axis=0)
    sd[sd < EPS] = 1.0

    skalirano = (opis_obuke - sredina) / sd

    hmm = GaussianHMM(
        n_components=HMM_REZIMI,
        covariance_type="diag",
        n_iter=200,
        tol=1e-4,
        random_state=SEED,
        min_covar=1e-4,
    )

    hmm.fit(skalirano)

    return hmm, sredina, sd


def hmm_forward_verovatnoce(
    hmm: GaussianHMM,
    opis: np.ndarray,
    sredina: np.ndarray,
    sd: np.ndarray,
) -> np.ndarray:
    """
    Forward filtriranje bez backward smoothing-a. Verovatnoća režima
    u trenutku t koristi samo podatke do trenutka t.
    """

    x = (opis - sredina) / sd
    log_emisije = hmm._compute_log_likelihood(x)

    n = len(x)
    k = hmm.n_components

    rezultat = np.zeros((n, k), dtype=np.float64)

    log_alpha = (
        np.log(np.maximum(hmm.startprob_, EPS))
        + log_emisije[0]
    )
    log_alpha -= logsumexp(log_alpha)
    rezultat[0] = np.exp(log_alpha)

    log_tranzicija = np.log(
        np.maximum(hmm.transmat_, EPS)
    )

    for t in range(1, n):
        novi = np.empty(k, dtype=np.float64)

        for stanje in range(k):
            novi[stanje] = (
                log_emisije[t, stanje]
                + logsumexp(
                    log_alpha + log_tranzicija[:, stanje]
                )
            )

        novi -= logsumexp(novi)
        log_alpha = novi
        rezultat[t] = np.exp(log_alpha)

    return rezultat


# =============================================================================
# DISTRIBUCIJSKE OSOBINE BROJEVA
# =============================================================================

def frekvencijski_odnos(
    binarno: np.ndarray,
    t: int,
    prozor: int | None,
) -> np.ndarray:
    pocetak = 0 if prozor is None else max(0, t - prozor)
    uzorak = binarno[pocetak:t]
    n = len(uzorak)

    if n == 0:
        return np.ones(BROJ_KUGLICA)

    brojanja = uzorak.sum(axis=0)
    ocekivano = n * TEORIJSKA_STOPA

    # Jedna puna pseudo-istorija usmerena prema uniformnoj raspodeli.
    return (brojanja + ocekivano) / (2.0 * ocekivano)


def vremenski_ponderisana_stopa(
    binarno: np.ndarray,
    t: int,
    poluzivot: float = 60.0,
) -> np.ndarray:
    istorija = binarno[:t]

    starost = np.arange(
        len(istorija) - 1,
        -1,
        -1,
        dtype=np.float64,
    )

    tezine = np.exp(
        -math.log(2.0) * starost / poluzivot
    )

    return (
        (istorija * tezine[:, None]).sum(axis=0)
        / np.sum(tezine)
    )


def gap_i_hazard(
    binarno: np.ndarray,
    t: int,
) -> tuple[np.ndarray, np.ndarray]:
    gap = np.zeros(BROJ_KUGLICA)
    hazard = np.full(BROJ_KUGLICA, TEORIJSKA_STOPA)

    for broj in range(BROJ_KUGLICA):
        pozicije = np.flatnonzero(binarno[:t, broj] > 0.5)

        if len(pozicije) == 0:
            gap[broj] = 1.0
            continue

        trenutni_gap = t - 1 - int(pozicije[-1])
        trenutni_gap = min(trenutni_gap, MAKSIMALNI_GAP)
        gap[broj] = trenutni_gap / MAKSIMALNI_GAP

        if len(pozicije) >= 2:
            zavrseni = np.minimum(
                np.diff(pozicije) - 1,
                MAKSIMALNI_GAP,
            )
            pod_rizikom = np.sum(zavrseni >= trenutni_gap)
            dogadjaji = np.sum(zavrseni == trenutni_gap)
        else:
            pod_rizikom = 0
            dogadjaji = 0

        prior = 20.0

        hazard[broj] = (
            dogadjaji + prior * TEORIJSKA_STOPA
        ) / (
            pod_rizikom + prior
        )

    return gap, hazard


def tranzicijski_skor(
    binarno: np.ndarray,
    t: int,
) -> np.ndarray:
    if t < 2:
        return np.full(BROJ_KUGLICA, TEORIJSKA_STOPA)

    pocetak = max(1, t - PROZOR_TRANZICIJE)

    prethodni = binarno[pocetak - 1:t - 1]
    naredni = binarno[pocetak:t]
    poslednje = binarno[t - 1]

    slicnost = prethodni @ poslednje
    tezine = 1.0 + slicnost

    prior = 20.0

    return (
        (naredni * tezine[:, None]).sum(axis=0)
        + prior * TEORIJSKA_STOPA
    ) / (
        np.sum(tezine) + prior
    )


def grafovski_skorovi(
    binarno: np.ndarray,
    t: int,
) -> tuple[np.ndarray, np.ndarray]:
    pocetak = max(0, t - PROZOR_GRAFA)
    uzorak = binarno[pocetak:t]

    matrica_parova = uzorak.T @ uzorak
    np.fill_diagonal(matrica_parova, 0.0)

    n = len(uzorak)

    ocekivanje_para = max(
        n
        * BROJ_IZVUCENIH
        * (BROJ_IZVUCENIH - 1)
        / (BROJ_KUGLICA * (BROJ_KUGLICA - 1)),
        EPS,
    )

    odstupanje = (
        matrica_parova - ocekivanje_para
    ) / math.sqrt(ocekivanje_para + EPS)

    poslednji = np.flatnonzero(binarno[t - 1] > 0.5)

    povezanost_sa_poslednjim = (
        odstupanje[:, poslednji].mean(axis=1)
        if len(poslednji)
        else np.zeros(BROJ_KUGLICA)
    )

    centralnost = odstupanje.mean(axis=1)

    return (
        standardizuj_vektor(povezanost_sa_poslednjim),
        standardizuj_vektor(centralnost),
    )


# =============================================================================
# MATRICA OSOBINA
# =============================================================================

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
    "vremenski_ponderisana_stopa",
    "gap",
    "hazard",
    "tranzicija",
    "graf_poslednje_izvlacenje",
    "graf_centralnost",
    "hmm_rezim_1",
    "hmm_rezim_2",
    "hmm_rezim_3",
    "pelt_starost_rezima",
    "change_score",
    "matrix_profile_1",
    "matrix_profile_2",
    "matrix_profile_3",
    "matrix_profile_4",
    "matrix_profile_5",
    "matrix_profile_6",
    "matrix_profile_7",
]


def change_score(
    opis: np.ndarray,
    t: int,
) -> float:
    if t < 80:
        return 0.0

    skorovi = []

    for prozor in (20, 40):
        noviji = opis[t - prozor:t]
        stariji = opis[t - 2 * prozor:t - prozor]

        razlika = np.abs(
            noviji.mean(axis=0) - stariji.mean(axis=0)
        )

        sd = opis[:t].std(axis=0)
        sd[sd < EPS] = 1.0

        skorovi.append(float(np.mean(razlika / sd)))

    return float(np.mean(skorovi))


def osobine_jednog_trenutka(
    binarno: np.ndarray,
    opis: np.ndarray,
    matrix_profile: np.ndarray,
    hmm_verovatnoce: np.ndarray,
    pelt_promena: int,
    t: int,
) -> np.ndarray:
    f20 = frekvencijski_odnos(binarno, t, 20)
    f50 = frekvencijski_odnos(binarno, t, 50)
    f100 = frekvencijski_odnos(binarno, t, 100)
    f200 = frekvencijski_odnos(binarno, t, 200)
    fsve = frekvencijski_odnos(binarno, t, None)

    vremenska = vremenski_ponderisana_stopa(binarno, t)
    gap, hazard = gap_i_hazard(binarno, t)
    tranzicija = tranzicijski_skor(binarno, t)
    graf_poslednji, graf_centralnost = grafovski_skorovi(
        binarno,
        t,
    )

    brojevi = np.arange(1, BROJ_KUGLICA + 1)
    ugao = 2.0 * np.pi * brojevi / BROJ_KUGLICA

    indeks_konteksta = max(0, t - 1)
    hmm_t = hmm_verovatnoce[indeks_konteksta]
    mp_t = matrix_profile[indeks_konteksta]

    starost_rezima = max(0, t - pelt_promena) / max(t, 1)
    promena_skor = change_score(opis, t)

    redovi = np.column_stack(
        [
            brojevi / BROJ_KUGLICA,
            np.sin(ugao),
            np.cos(ugao),
            f20,
            f50,
            f100,
            f200,
            fsve,
            f20 - f100,
            f50 - f200,
            vremenska,
            gap,
            hazard,
            tranzicija,
            graf_poslednji,
            graf_centralnost,
            np.full(BROJ_KUGLICA, hmm_t[0]),
            np.full(BROJ_KUGLICA, hmm_t[1]),
            np.full(BROJ_KUGLICA, hmm_t[2]),
            np.full(BROJ_KUGLICA, starost_rezima),
            np.full(BROJ_KUGLICA, promena_skor),
            *[
                np.full(BROJ_KUGLICA, vrednost)
                for vrednost in mp_t
            ],
        ]
    )

    return redovi.astype(np.float32)


def napravi_dataset(
    binarno: np.ndarray,
    opis: np.ndarray,
    matrix_profile: np.ndarray,
    hmm_verovatnoce: np.ndarray,
    pelt_promena: int,
    pocetak: int,
    kraj: int,
    sa_metom: bool = True,
) -> tuple[np.ndarray, np.ndarray | None, list[int]]:
    x_delovi = []
    y_delovi = []

    for t in range(pocetak, kraj):
        x_delovi.append(
            osobine_jednog_trenutka(
                binarno=binarno,
                opis=opis,
                matrix_profile=matrix_profile,
                hmm_verovatnoce=hmm_verovatnoce,
                pelt_promena=pelt_promena,
                t=t,
            )
        )

        if sa_metom:
            y_delovi.append(binarno[t].astype(np.int8))

    x = np.vstack(x_delovi)
    y = np.concatenate(y_delovi) if sa_metom else None
    grupe = [BROJ_KUGLICA] * (kraj - pocetak)

    return x, y, grupe


# =============================================================================
# LIGHTGBM LAMBDARANK
# =============================================================================

def napravi_ranker(
    parametri: dict[str, Any],
) -> lgb.LGBMRanker:
    return lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        eval_at=[7],
        label_gain=[0, 1],
        lambdarank_truncation_level=10,
        random_state=SEED,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
        n_jobs=-1,
        subsample=1.0,
        colsample_bytree=0.85,
        **parametri,
    )


def predikcije_po_grupama(
    model: lgb.LGBMRanker,
    x: np.ndarray,
    broj_grupa: int,
) -> np.ndarray:
    skorovi = model.predict(x)
    rezultat = np.empty(
        (broj_grupa, BROJ_IZVUCENIH),
        dtype=np.int16,
    )

    for grupa in range(broj_grupa):
        pocetak = grupa * BROJ_KUGLICA
        kraj = pocetak + BROJ_KUGLICA

        skor = skorovi[pocetak:kraj]
        brojevi = np.arange(1, BROJ_KUGLICA + 1)

        redosled = np.lexsort((brojevi, -skor))
        rezultat[grupa] = np.sort(
            redosled[:BROJ_IZVUCENIH] + 1
        )

    return rezultat


def izracunaj_pogotke(
    predikcije: np.ndarray,
    stvarna_izvlacenja: np.ndarray,
) -> np.ndarray:
    return np.asarray(
        [
            np.intersect1d(pred, stvarno).size
            for pred, stvarno in zip(
                predikcije,
                stvarna_izvlacenja,
            )
        ],
        dtype=np.int8,
    )


# =============================================================================
# JEDAN HRONOLOŠKI FIT I TEST
# =============================================================================

def fit_i_test(
    izvlacenja: np.ndarray,
    binarno: np.ndarray,
    opis: np.ndarray,
    matrix_profile: np.ndarray,
    train_kraj: int,
    test_pocetak: int,
    test_kraj: int,
    parametri: dict[str, Any],
) -> tuple[
    lgb.LGBMRanker,
    np.ndarray,
    np.ndarray,
]:
    hmm, sredina, sd = fit_hmm(opis[:train_kraj])

    hmm_verovatnoce = hmm_forward_verovatnoce(
        hmm,
        opis[:test_kraj],
        sredina,
        sd,
    )

    pelt_promena = poslednja_pelt_promena(
        opis[:train_kraj]
    )

    x_train, y_train, grupe_train = napravi_dataset(
        binarno=binarno,
        opis=opis,
        matrix_profile=matrix_profile,
        hmm_verovatnoce=hmm_verovatnoce,
        pelt_promena=pelt_promena,
        pocetak=MIN_ISTORIJA,
        kraj=train_kraj,
    )

    x_test, _, _ = napravi_dataset(
        binarno=binarno,
        opis=opis,
        matrix_profile=matrix_profile,
        hmm_verovatnoce=hmm_verovatnoce,
        pelt_promena=pelt_promena,
        pocetak=test_pocetak,
        kraj=test_kraj,
        sa_metom=False,
    )

    model = napravi_ranker(parametri)

    model.fit(
        x_train,
        y_train,
        group=grupe_train,
        feature_name=NAZIVI_OSOBINA,
    )

    predikcije = predikcije_po_grupama(
        model,
        x_test,
        test_kraj - test_pocetak,
    )

    pogoci = izracunaj_pogotke(
        predikcije,
        izvlacenja[test_pocetak:test_kraj],
    )

    return model, predikcije, pogoci


# =============================================================================
# NESTED WALK-FORWARD
# =============================================================================

def nested_walk_forward(
    izvlacenja: np.ndarray,
    binarno: np.ndarray,
    opis: np.ndarray,
    matrix_profile: np.ndarray,
    razvoj_kraj: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    prvi_spoljni_test = max(
        MIN_ISTORIJA + 300,
        int(razvoj_kraj * 0.55),
    )

    granice = np.linspace(
        prvi_spoljni_test,
        razvoj_kraj,
        BROJ_SPOLJNIH_FOLDOVA + 1,
        dtype=int,
    )

    svi_spoljni_pogoci = []
    zbir_unutrasnjih_skorova = np.zeros(
        len(PARAMETRI_RANKERA),
        dtype=np.float64,
    )

    for fold in range(BROJ_SPOLJNIH_FOLDOVA):
        spoljni_test_pocetak = int(granice[fold])
        spoljni_test_kraj = int(granice[fold + 1])

        spoljni_train_kraj = (
            spoljni_test_pocetak - PURGE_KORACI
        )

        raspolozivo = spoljni_train_kraj - MIN_ISTORIJA

        unutrasnja_duzina = max(
            80,
            int(raspolozivo * UNUTRASNJI_VALIDACIONI_UDEO),
        )

        unutrasnji_test_pocetak = (
            spoljni_train_kraj - unutrasnja_duzina
        )

        unutrasnji_train_kraj = (
            unutrasnji_test_pocetak - PURGE_KORACI
        )

        print(
            f"  Spoljni fold {fold + 1}/"
            f"{BROJ_SPOLJNIH_FOLDOVA}"
        )

        unutrasnji_skorovi = []

        for indeks, parametri in enumerate(PARAMETRI_RANKERA):
            _, _, pogoci = fit_i_test(
                izvlacenja=izvlacenja,
                binarno=binarno,
                opis=opis,
                matrix_profile=matrix_profile,
                train_kraj=unutrasnji_train_kraj,
                test_pocetak=unutrasnji_test_pocetak,
                test_kraj=spoljni_train_kraj,
                parametri=parametri,
            )

            prosek = float(np.mean(pogoci))
            unutrasnji_skorovi.append(prosek)
            zbir_unutrasnjih_skorova[indeks] += prosek

        najbolji_indeks = int(np.argmax(unutrasnji_skorovi))
        najbolji_parametri = PARAMETRI_RANKERA[najbolji_indeks]

        _, _, spoljni_pogoci = fit_i_test(
            izvlacenja=izvlacenja,
            binarno=binarno,
            opis=opis,
            matrix_profile=matrix_profile,
            train_kraj=spoljni_train_kraj,
            test_pocetak=spoljni_test_pocetak,
            test_kraj=spoljni_test_kraj,
            parametri=najbolji_parametri,
        )

        svi_spoljni_pogoci.append(spoljni_pogoci)

    finalni_indeks = int(
        np.argmax(zbir_unutrasnjih_skorova)
    )

    return (
        np.concatenate(svi_spoljni_pogoci),
        PARAMETRI_RANKERA[finalni_indeks],
    )


# =============================================================================
# STATISTIČKA PROVERA
# =============================================================================

def blok_bootstrap(
    pogoci: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    n = len(pogoci)
    blok = min(BOOTSTRAP_BLOK, n)

    rezultati = np.empty(
        BROJ_BOOTSTRAP_PONAVLJANJA,
        dtype=np.float64,
    )

    for ponavljanje in range(BROJ_BOOTSTRAP_PONAVLJANJA):
        uzorak = []

        while len(uzorak) < n:
            pocetak = int(rng.integers(0, n))
            indeksi = (
                pocetak + np.arange(blok)
            ) % n
            uzorak.extend(pogoci[indeksi].tolist())

        rezultati[ponavljanje] = np.mean(uzorak[:n])

    return rezultati


def monte_karlo(
    broj_koraka: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    epizode = rng.hypergeometric(
        ngood=BROJ_IZVUCENIH,
        nbad=BROJ_KUGLICA - BROJ_IZVUCENIH,
        nsample=BROJ_IZVUCENIH,
        size=(
            BROJ_MONTE_KARLO_EPIZODA,
            broj_koraka,
        ),
    )

    return epizode, epizode.mean(axis=1)


def permutacioni_test(
    predikcije: np.ndarray,
    stvarna: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    n = len(stvarna)
    rezultati = np.empty(BROJ_PERMUTACIJA)

    for p in range(BROJ_PERMUTACIJA):
        permutacija = rng.permutation(n)
        permutovana = stvarna[permutacija]

        pogoci = izracunaj_pogotke(
            predikcije,
            permutovana,
        )

        rezultati[p] = np.mean(pogoci)

    return rezultati


def desna_p_vrednost(
    nulta_raspodela: np.ndarray,
    posmatrano: float,
) -> float:
    return float(
        (
            np.sum(nulta_raspodela >= posmatrano) + 1
        )
        / (
            len(nulta_raspodela) + 1
        )
    )


def stabilnost_segmenta(
    pogoci: np.ndarray,
) -> list[float]:
    return [
        float(np.mean(segment))
        for segment in np.array_split(pogoci, 4)
        if len(segment)
    ]


# =============================================================================
# OBRADA JEDNE IGRE
# =============================================================================

def obradi_igru(
    naziv: str,
    csv_putanja: Path,
    seed_pomeraj: int,
) -> RezultatIgre:
    print()
    print("=" * 78)
    print(f"Obrada: {naziv}")
    print("=" * 78)

    izvlacenja = ucitaj_csv(csv_putanja)
    binarno = napravi_binarnu_matricu(izvlacenja)
    opis = opis_izvlacenja(izvlacenja, binarno)

    n = len(izvlacenja)

    holdout_duzina = int(round(n * HOLDOUT_UDEO))
    holdout_duzina = max(MIN_HOLDOUT, holdout_duzina)
    holdout_duzina = min(MAX_HOLDOUT, holdout_duzina)

    holdout_pocetak = n - holdout_duzina

    print(f"CSV: {csv_putanja}")
    print(f"Broj redova: {n}")
    print("Prvi red je najstariji.")
    print("Poslednji red je najnoviji.")
    print(f"Razvojni period: {holdout_pocetak}")
    print(f"Zamrznuti holdout: {holdout_duzina}")

    print("Causal Matrix Profile...")
    mp = causal_matrix_profile(opis)

    print("Nested walk-forward...")
    nested_pogoci, najbolji_parametri = nested_walk_forward(
        izvlacenja=izvlacenja,
        binarno=binarno,
        opis=opis,
        matrix_profile=mp,
        razvoj_kraj=holdout_pocetak,
    )

    print("Zamrznuta holdout provera...")

    finalni_train_kraj = holdout_pocetak - PURGE_KORACI

    holdout_model, holdout_predikcije, holdout_pogoci = (
        fit_i_test(
            izvlacenja=izvlacenja,
            binarno=binarno,
            opis=opis,
            matrix_profile=mp,
            train_kraj=finalni_train_kraj,
            test_pocetak=holdout_pocetak,
            test_kraj=n,
            parametri=najbolji_parametri,
        )
    )

    rng = np.random.default_rng(SEED + seed_pomeraj)

    print(
        f"Monte Karlo: "
        f"{BROJ_MONTE_KARLO_EPIZODA:,} epizoda..."
    )

    mc_epizode, mc_proseci = monte_karlo(
        holdout_duzina,
        rng,
    )

    print(
        f"Blok-bootstrap: "
        f"{BROJ_BOOTSTRAP_PONAVLJANJA:,} ponavljanja..."
    )

    bootstrap_proseci = blok_bootstrap(
        holdout_pogoci,
        rng,
    )

    print(
        f"Permutacioni test: "
        f"{BROJ_PERMUTACIJA:,} permutacija..."
    )

    permutacioni_proseci = permutacioni_test(
        holdout_predikcije,
        izvlacenja[holdout_pocetak:n],
        rng,
    )

    posmatrani_prosek = float(np.mean(holdout_pogoci))

    ci_donji, ci_gornji = np.quantile(
        bootstrap_proseci,
        [0.025, 0.975],
    )

    p_monte_karlo = desna_p_vrednost(
        mc_proseci,
        posmatrani_prosek,
    )

    p_permutacija = desna_p_vrednost(
        permutacioni_proseci,
        posmatrani_prosek,
    )

    nulti_pogoci = mc_epizode.ravel()

    wasserstein = float(
        wasserstein_distance(
            holdout_pogoci.astype(float),
            nulti_pogoci.astype(float),
        )
    )

    energy = float(
        energy_distance(
            holdout_pogoci.astype(float),
            nulti_pogoci.astype(float),
        )
    )

    # NEXT model se obučava nad svim sada dostupnim podacima.
    print("Završna NEXT obuka nad kompletnom istorijom...")

    hmm_sve, sredina_sve, sd_sve = fit_hmm(opis)

    hmm_verovatnoce_sve = hmm_forward_verovatnoce(
        hmm_sve,
        opis,
        sredina_sve,
        sd_sve,
    )

    pelt_sve = poslednja_pelt_promena(opis)

    x_sve, y_sve, grupe_sve = napravi_dataset(
        binarno=binarno,
        opis=opis,
        matrix_profile=mp,
        hmm_verovatnoce=hmm_verovatnoce_sve,
        pelt_promena=pelt_sve,
        pocetak=MIN_ISTORIJA,
        kraj=n,
    )

    next_model = napravi_ranker(najbolji_parametri)

    next_model.fit(
        x_sve,
        y_sve,
        group=grupe_sve,
        feature_name=NAZIVI_OSOBINA,
    )

    # Za NEXT red HMM koristi poslednju forward verovatnoću.
    x_next = osobine_jednog_trenutka(
        binarno=binarno,
        opis=opis,
        matrix_profile=mp,
        hmm_verovatnoce=hmm_verovatnoce_sve,
        pelt_promena=pelt_sve,
        t=n,
    )

    next_kombinacija = predikcije_po_grupama(
        next_model,
        x_next,
        broj_grupa=1,
    )[0]

    vaznosti = next_model.booster_.feature_importance(
        importance_type="gain"
    )

    zbir_vaznosti = float(np.sum(vaznosti))

    if zbir_vaznosti > 0:
        vaznosti = vaznosti / zbir_vaznosti * 100.0

    vaznost_osobina = sorted(
        zip(NAZIVI_OSOBINA, vaznosti.tolist()),
        key=lambda par: par[1],
        reverse=True,
    )

    return RezultatIgre(
        naziv=naziv,
        broj_redova=n,
        next_kombinacija=next_kombinacija,
        izabrani_parametri=najbolji_parametri,
        nested_pogoci=nested_pogoci,
        holdout_pogoci=holdout_pogoci,
        holdout_predikcije=holdout_predikcije,
        holdout_stvarni=izvlacenja[holdout_pocetak:n],
        bootstrap_proseci=bootstrap_proseci,
        monte_karlo_proseci=mc_proseci,
        permutacioni_proseci=permutacioni_proseci,
        ci_donji=float(ci_donji),
        ci_gornji=float(ci_gornji),
        p_monte_karlo=p_monte_karlo,
        p_permutacija=p_permutacija,
        wasserstein=wasserstein,
        energy=energy,
        stabilnost=stabilnost_segmenta(holdout_pogoci),
        vaznost_osobina=vaznost_osobina,
    )


# =============================================================================
# ZAKLJUČAK I ISPIS
# =============================================================================

def formatiraj_kombinaciju(
    kombinacija: np.ndarray,
) -> str:
    return ", ".join(
        f"{int(broj):02d}"
        for broj in kombinacija
    )


def statisticki_zakljucak(
    rezultat: RezultatIgre,
) -> str:
    prosek = float(np.mean(rezultat.holdout_pogoci))

    interval_iznad = (
        rezultat.ci_donji > SLUCAJNO_OCEKIVANJE
    )

    testovi_prolaze = (
        rezultat.p_monte_karlo < 0.05
        and rezultat.p_permutacija < 0.05
    )

    broj_dobrih_segmenata = sum(
        segment > SLUCAJNO_OCEKIVANJE
        for segment in rezultat.stabilnost
    )

    stabilno = broj_dobrih_segmenata >= 3

    if (
        prosek > SLUCAJNO_OCEKIVANJE
        and interval_iznad
        and testovi_prolaze
        and stabilno
    ):
        return (
            "DA — pronađena prednost je iznad slučajnog očekivanja, "
            "statistički značajna i stabilna kroz vreme."
        )

    return (
        "NE — model nije dokazao statistički pouzdanu i vremenski "
        "stabilnu prednost nad poštenim slučajnim izborom."
    )


def ispisi_rezultat(
    rezultat: RezultatIgre,
) -> None:
    holdout_prosek = float(
        np.mean(rezultat.holdout_pogoci)
    )

    nested_prosek = float(
        np.mean(rezultat.nested_pogoci)
    )

    mc_donji, mc_gornji = np.quantile(
        rezultat.monte_karlo_proseci,
        [0.025, 0.975],
    )

    print()
    print("=" * 78)
    print(rezultat.naziv)
    print("=" * 78)
    print(
        f"NEXT: "
        f"{formatiraj_kombinaciju(rezultat.next_kombinacija)}"
    )
    print(f"CSV redova: {rezultat.broj_redova}")

    print()
    print("HRONOLOŠKA PROVERA")
    print("-" * 78)
    print(
        f"Nested walk-forward prosek:       "
        f"{nested_prosek:.6f}"
    )
    print(
        f"Zamrznuti holdout prosek:         "
        f"{holdout_prosek:.6f}"
    )
    print(
        f"Slučajno očekivanje:              "
        f"{SLUCAJNO_OCEKIVANJE:.6f}"
    )
    print(
        f"Razlika prema slučajnom:          "
        f"{holdout_prosek - SLUCAJNO_OCEKIVANJE:+.6f}"
    )

    print()
    print("STATISTIČKA PROVERA")
    print("-" * 78)
    print(
        f"Blok-bootstrap interval 95%:      "
        f"[{rezultat.ci_donji:.6f}, "
        f"{rezultat.ci_gornji:.6f}]"
    )
    print(
        f"Monte Karlo interval 95%:         "
        f"[{mc_donji:.6f}, {mc_gornji:.6f}]"
    )
    print(
        f"Monte Karlo p-vrednost:           "
        f"{rezultat.p_monte_karlo:.6f}"
    )
    print(
        f"Permutaciona p-vrednost:          "
        f"{rezultat.p_permutacija:.6f}"
    )
    print(
        f"Wasserstein distanca:             "
        f"{rezultat.wasserstein:.6f}"
    )
    print(
        f"Energy distanca:                  "
        f"{rezultat.energy:.6f}"
    )

    print()
    print("STABILNOST HOLDOUTA")
    print("-" * 78)

    for i, segment in enumerate(
        rezultat.stabilnost,
        start=1,
    ):
        print(
            f"Hronološki segment {i}:            "
            f"{segment:.6f}"
        )

    print()
    print("NAJVAŽNIJE OSOBINE")
    print("-" * 78)

    for naziv, vaznost in rezultat.vaznost_osobina[:10]:
        print(f"{naziv:<38} {vaznost:>8.3f}%")

    print()
    print("IZABRANI PARAMETRI")
    print("-" * 78)

    for naziv, vrednost in rezultat.izabrani_parametri.items():
        print(f"{naziv:<38} {vrednost}")

    print()
    print("ODGOVOR NA GLAVNO PITANJE")
    print("-" * 78)
    print(statisticki_zakljucak(rezultat))


# =============================================================================
# GLAVNI PROGRAM
# =============================================================================

def main() -> None:
    np.random.seed(SEED)

    print("=" * 78)
    print("LOTO 7/39 — FINALNI PATTERN RECOGNITION SISTEM")
    print("=" * 78)
    print(f"Seed: {SEED}")
    print(f"Teorijska stopa broja: {TEORIJSKA_STOPA:.9f}")
    print(
        f"Teorijsko očekivanje pogodaka: "
        f"{SLUCAJNO_OCEKIVANJE:.9f}"
    )
    print(
        f"Ukupno mogućih kombinacija: "
        f"{UKUPNO_KOMBINACIJA:,}"
    )

    loto = obradi_igru(
        naziv="Loto",
        csv_putanja=LOTO_CSV,
        seed_pomeraj=0,
    )

    loto_plus = obradi_igru(
        naziv="Loto Plus",
        csv_putanja=LOTO_PLUS_CSV,
        seed_pomeraj=1,
    )

    print()
    print()
    print("#" * 78)
    print("KONAČNE NEXT PREDIKCIJE")
    print("#" * 78)

    ispisi_rezultat(loto)
    ispisi_rezultat(loto_plus)


if __name__ == "__main__":
    main()



"""
==============================================================================
LOTO 7/39 — FINALNI PATTERN RECOGNITION SISTEM
==============================================================================
Seed: 39
Teorijska stopa broja: 0.179487179
Teorijsko očekivanje pogodaka: 1.256410256
Ukupno mogućih kombinacija: 15,380,937

==============================================================================
Obrada: Loto
==============================================================================
CSV: /data/loto7_4680_k71_loto_2962.csv
Broj redova: 2962
Prvi red je najstariji.
Poslednji red je najnoviji.
Razvojni period: 2562
Zamrznuti holdout: 400
Causal Matrix Profile...
Nested walk-forward...
  Spoljni fold 1/4
  Spoljni fold 2/4
  Spoljni fold 3/4
  Spoljni fold 4/4
Zamrznuta holdout provera...
Monte Karlo: 10,000 epizoda...
Blok-bootstrap: 2,000 ponavljanja...
Permutacioni test: 2,000 permutacija...
Završna NEXT obuka nad kompletnom istorijom...

==============================================================================
Obrada: Loto Plus
==============================================================================
CSV: /data/loto7_4680_k71_loto_plus_1718.csv
Broj redova: 1718
Prvi red je najstariji.
Poslednji red je najnoviji.
Razvojni period: 1460
Zamrznuti holdout: 258
Causal Matrix Profile...
Nested walk-forward...
  Spoljni fold 1/4
  Spoljni fold 2/4
  Spoljni fold 3/4
  Spoljni fold 4/4
Zamrznuta holdout provera...
Monte Karlo: 10,000 epizoda...
Blok-bootstrap: 2,000 ponavljanja...
Permutacioni test: 2,000 permutacija...
Završna NEXT obuka nad kompletnom istorijom...


##############################################################################
KONAČNE NEXT PREDIKCIJE
##############################################################################

==============================================================================
Loto
==============================================================================
NEXT: 11, x, 21, y, 26, z, 33
CSV redova: 2962

HRONOLOŠKA PROVERA
------------------------------------------------------------------------------
Nested walk-forward prosek:       1.281006
Zamrznuti holdout prosek:         1.220000
Slučajno očekivanje:              1.256410
Razlika prema slučajnom:          -0.036410

STATISTIČKA PROVERA
------------------------------------------------------------------------------
Blok-bootstrap interval 95%:      [1.139937, 1.310000]
Monte Karlo interval 95%:         [1.165000, 1.350000]
Monte Karlo p-vrednost:           0.789521
Permutaciona p-vrednost:          0.244878
Wasserstein distanca:             0.036433
Energy distanca:                  0.026123

STABILNOST HOLDOUTA
------------------------------------------------------------------------------
Hronološki segment 1:            1.230000
Hronološki segment 2:            1.190000
Hronološki segment 3:            1.280000
Hronološki segment 4:            1.180000

NAJVAŽNIJE OSOBINE
------------------------------------------------------------------------------
broj_norm                                10.799%
tranzicija                               10.498%
frekvencija_sve                           9.428%
hazard                                    8.866%
vremenski_ponderisana_stopa               8.125%
graf_poslednje_izvlacenje                 7.416%
graf_centralnost                          7.335%
broj_sin                                  7.213%
promena_50_200                            6.262%
gap                                       5.620%

IZABRANI PARAMETRI
------------------------------------------------------------------------------
num_leaves                             11
max_depth                              4
learning_rate                          0.02
n_estimators                           400
min_child_samples                      100
reg_alpha                              2.0
reg_lambda                             8.0

ODGOVOR NA GLAVNO PITANJE
------------------------------------------------------------------------------
NE — model nije dokazao statistički pouzdanu i vremenski stabilnu prednost nad poštenim slučajnim izborom.

==============================================================================
Loto Plus
==============================================================================
NEXT: 01, x, 08, y, 26, z, 34
CSV redova: 1718

HRONOLOŠKA PROVERA
------------------------------------------------------------------------------
Nested walk-forward prosek:       1.240487
Zamrznuti holdout prosek:         1.244186
Slučajno očekivanje:              1.256410
Razlika prema slučajnom:          -0.012224

STATISTIČKA PROVERA
------------------------------------------------------------------------------
Blok-bootstrap interval 95%:      [1.155039, 1.333333]
Monte Karlo interval 95%:         [1.143411, 1.372093]
Monte Karlo p-vrednost:           0.592041
Permutaciona p-vrednost:          0.101949
Wasserstein distanca:             0.084628
Energy distanca:                  0.075062

STABILNOST HOLDOUTA
------------------------------------------------------------------------------
Hronološki segment 1:            1.246154
Hronološki segment 2:            1.138462
Hronološki segment 3:            1.296875
Hronološki segment 4:            1.296875

NAJVAŽNIJE OSOBINE
------------------------------------------------------------------------------
hazard                                   11.727%
tranzicija                               11.320%
frekvencija_sve                          10.704%
graf_centralnost                         10.006%
graf_poslednje_izvlacenje                 8.821%
vremenski_ponderisana_stopa               8.362%
promena_20_100                            6.338%
broj_norm                                 6.080%
broj_sin                                  5.713%
gap                                       5.106%

IZABRANI PARAMETRI
------------------------------------------------------------------------------
num_leaves                             15
max_depth                              5
learning_rate                          0.015
n_estimators                           500
min_child_samples                      90
reg_alpha                              3.0
reg_lambda                             10.0

ODGOVOR NA GLAVNO PITANJE
------------------------------------------------------------------------------
NE — model nije dokazao statistički pouzdanu i vremenski stabilnu prednost nad poštenim slučajnim izborom.
"""








"""
Pattern recognition je prepoznavanje obrazaca u podacima 
— pravilnosti, ponavljanja, odnosa, trendova ili anomalija koje nisu odmah očigledne.

Može koristiti:
- statistiku;
- mašinsko učenje;
- neuronske mreže;
- vremenske serije;
- grupisanje i klasifikaciju;
- prepoznavanje slika, zvuka, teksta ili numeričkih nizova.

Ključna razlika je između obrasca koji stvarno nosi informaciju i slučajnog privida obrasca. 
Zato se pronađeni obrazac proverava na podacima koje model nije koristio tokom traženja.
"""



"""
Za pattern recognition koristim sledeće metode, tehnike, algoritme i modele:

1. Statističko prepoznavanje obrazaca
   - autokorelacija i parcijalna autokorelacija;
   - cross-correlation;
   - testovi stacionarnosti;
   - analiza distribucije;
   - promena režima i change-point detection;
   - spektralna/Fourier analiza;
   - wavelet analiza;
   - entropija i međusobna informacija.

2. Otkrivanje vremenskih obrazaca
   - lag osobine;
   - rolling prozori;
   - eksponencijalno vremensko ponderisanje;
   - trend, ciklus i sezonalnost;
   - ARIMA/SARIMA;
   - state-space i Kalman filter;
   - Hidden Markov Model;
   - Bayesian change-point modeli.

3. Klasično mašinsko učenje
   - Random Forest;
   - Extra Trees;
   - Gradient Boosting;
   - XGBoost, LightGBM i CatBoost;
   - Support Vector Machine/Regression;
   - k-nearest neighbours;
   - Elastic Net;
   - Gaussian Process.

4. Neuronski modeli
   - MLP;
   - 1D CNN za lokalne sekvence;
   - LSTM i GRU;
   - Temporal Convolutional Network;
   - Transformer za vremenske serije;
   - autoencoder za latentne obrasce;
   - variational autoencoder;
   - neuronski survival model.

5. Grupisanje i prepoznavanje režima
   - K-means;
   - DBSCAN/HDBSCAN;
   - Gaussian Mixture Model;
   - hijerarhijsko grupisanje;
   - spectral clustering;
   - self-organizing maps.

6. Grafovski obrasci
   - matrica veza;
   - korelacioni i transition graf;
   - centralnost čvorova;
   - community detection;
   - node embeddings;
   - Graph Neural Network.

7. Otkrivanje anomalija
   - Isolation Forest;
   - Local Outlier Factor;
   - One-Class SVM;
   - robust z-score;
   - autoencoder reconstruction error;
   - Bayesian anomaly detection.

8. Prepoznavanje podsekvenci
   - Dynamic Time Warping;
   - motif discovery;
   - matrix profile;
   - shapelets;
   - edit distance;
   - nearest-neighbour poređenje istorijskih sekvenci.

9. Ensemble pristup
   - kombinovanje više nezavisnih modela;
   - ponderisanje prema rezultatima van uzorka;
   - stacking;
   - bagging;
   - boosting;
   - dinamički izbor modela prema trenutnom režimu.

10. Obavezna provera da obrazac nije slučajan
    - hronološki walk-forward;
    - nested walk-forward;
    - potpuno odvojeni holdout;
    - blok-bootstrap;
    - permutacioni test;
    - Monte Karlo nulta hipoteza;
    - korekcija za višestruko testiranje;
    - poređenje sa jednostavnom slučajnom osnovom;
    - provera stabilnosti kroz različite periode.

Za ozbiljan pattern-recognition sistem ne bih koristio samo jedan algoritam, već kombinaciju: 
matrix profile + change-point detection + Hidden Markov režimi + gradient boosting + grafovski model + stroga hronološka validacija.
"""



"""
Konačna arhitektura: 

LightGBM LambdaRank kao glavni model, jer zadatak nije klasična regresija nego rangiranje 39 brojeva i uzimanje prvih sedam. 
Svako istorijsko izvlačenje predstavlja jednu grupu od 39 kandidata, sa sedam izvučenih brojeva kao relevantnim rezultatima. 
LambdaRank je upravo namenjen optimizovanju vrha rang-liste. LightGBM dokumentacija

Ulazne osobine svakog broja obuhvataće:
- empirijsku gap/survival/hazard distribuciju;
- uslovne tranzicije nakon prethodnih izvlačenja;
- graf parova i trojki, korigovan prema slučajnom očekivanju;
- vremenski ponderisane distribucijske odnose;
- promenu tih odnosa kroz više prozora;
- verovatnoću trenutno aktivnog HMM režima;
- udaljenost od najbližih istorijskih motiva;
- položaj prema poslednjoj potvrđenoj promeni režima;
- odstupanje svih osobina od teorijske stope \(7/39\).

Pattern-recognition komponente imaju ove uloge:
- Matrix Profile traži motive i anomalije u multivarijantnim distribucijskim opisima izvlačenja, a ne u sirovom redosledu sedam sortiranih brojeva. Matrix Profile je namenjen pronalaženju motiva, diskordanata i segmenata vremenskih serija. Matrix Profile rad
- PELT change-point detection pronalazi promene raspodele uz penalizaciju lažnih promena. PELT rad
- HMM daje verovatnoće skrivenih režima, ali se ne koristi samostalno za izbor brojeva.
- Grafovski skor opisuje nelinearne veze brojeva.
- LambdaRank objedinjuje sve osobine i direktno rangira 39 brojeva.

Zaštita od lažnih obrazaca
- potpuno odvojena obrada Loto i Loto Plus CSV-a;
- prvi red je najstariji, poslednji najnoviji;
- nested walk-forward izbor osobina i parametara;
- zamrznuti završni holdout koji se ne koristi ni za jednu odluku;
- purging između obuke i provere;
- blok-bootstrap interval od 95%;
- permutacioni test;
- Monte Karlo poštenog Loto 7/39 procesa;
- korekcija za višestruko isprobavanje obrazaca;
- poređenje sa slučajnim izborom i jednostavnim distribucijskim osnovama;
- osobina se zadržava samo ako poboljšanje ponovi u većini hronoloških foldova.

Izlaz je jedna NEXT kombinacija za Loto i jedna za Loto Plus, 
uz jasan odgovor da li je prednost na zamrznutom holdoutu statistički pouzdana.

GNN, Transformer ili duboka neuronska mreža: 
sa približno 3.000 i 1.700 izvlačenja povećali bi složenost i rizik od preprilagođavanja više nego pouzdanu prediktivnu snagu.






Finalno rešenje:
Matrix Profile + PELT change-point detection + HMM režimi + grafovske osobine + gap/survival/hazard i transition distribucije + LightGBM LambdaRank + stroga nested walk-forward i zamrznuta holdout provera.
Jedan model se zasebno obučava na Loto CSV-u i Loto Plus CSV-u i daje po jednu NEXT kombinaciju. 
"""
