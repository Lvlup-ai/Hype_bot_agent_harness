"""A forty-row user table with nine labelled anomalies. No randomness anywhere.

The labels are the ground truth a rule is measured against. Two rows carry an
out-of-range score that is *not* labelled: noise, there to show that a rule
on the wrong column flags rows without finding anomalies.
"""

from __future__ import annotations

import hashlib
import json

COLUMNS = ("id", "age", "email", "signup_date", "country", "score")

TODAY = "2026-01-01"   # fixed, so "a date in the future" is a deterministic question

# (id, age, email, signup_date, country, score, anomaly)
_ROWS: tuple[tuple[int, int, str, str, str, int, bool], ...] = (
    (1, 34, "ana@example.org", "2021-03-12", "FR", 72, False),
    (2, 27, "ben@example.org", "2022-07-01", "DE", 40, False),
    (3, 45, "cleo@example.org", "2021-11-23", "ES", 250, False),      # noisy score, not an anomaly
    (4, 150, "dan@example.org", "2023-01-09", "IT", 66, True),        # impossible age
    (5, 52, "eve@example.org", "2020-05-30", "PT", 88, False),
    (6, 38, "finn@example.org", "2024-02-14", "FR", 51, False),
    (7, 29, "mia.example.org", "2022-09-08", "DE", 63, True),         # email without @
    (8, 61, "gus@example.org", "2019-12-01", "ES", 19, False),
    (9, -3, "hana@example.org", "2023-06-21", "IT", 77, True),        # negative age
    (10, 41, "ivan@example.org", "2021-08-17", "PT", 35, False),
    (11, 23, "jade@example.org", "2025-01-05", "FR", 90, False),
    (12, 36, "kai@example.org", "2031-06-15", "DE", 58, True),        # signup in the future
    (13, 48, "lena@example.org", "2020-10-10", "ES", 44, False),
    (14, 55, "milo@example.org", "2022-03-03", "IT", 81, False),
    (15, 207, "nora@example.org", "2021-04-28", "PT", 27, True),      # impossible age
    (16, 31, "omar@example.org", "2023-11-11", "FR", 69, False),
    (17, 26, "pia@example.org", "2024-08-19", "DE", 33, False),
    (18, 39, "", "2022-01-25", "ES", 74, True),                       # empty email
    (19, 44, "quin@example.org", "2020-02-02", "IT", 52, False),
    (20, 58, "rosa@example.org", "2021-09-30", "PT", 61, False),
    (21, 33, "sam@example.org", "2023-04-04", "FR", 47, False),
    (22, 0, "tess@example.org", "2022-12-12", "DE", 83, True),        # age zero in a user table
    (23, 47, "uma@example.org", "2024-05-05", "ES", 29, False),
    (24, 62, "vic@example.org", "2019-07-07", "IT", 95, False),
    (25, 28, "wren@example.org", "2025-03-15", "PT", 38, False),
    (26, 50, "xavi@example.org", "2021-06-06", "FR", 56, False),
    (27, 37, "yara@example.org", "2030-01-01", "DE", 42, True),       # signup in the future
    (28, 24, "zed@example.org", "2023-08-08", "ES", 71, False),
    (29, 59, "ada@example.org", "2020-04-16", "IT", 64, False),
    (30, 42, "bo@example.org", "2022-10-20", "PT", 22, False),
    (31, 35, "noah_at_example.org", "2021-01-31", "FR", 79, True),    # email without @
    (32, 30, "cy@example.org", "2024-11-02", "DE", 48, False),
    (33, 46, "dee@example.org", "2022-05-27", "ES", 86, False),
    (34, 53, "eli@example.org", "2023-02-18", "IT", 31, False),
    (35, 40, "fay@example.org", "2021-07-09", "PT", 300, False),      # noisy score, not an anomaly
    (36, 32, "gil@example.org", "2025-06-24", "FR", 55, False),
    (37, 49, "hal@example.org", "2020-09-13", "DE", 67, False),
    (38, 25, "ida@example.org", "2023-12-29", "ES", 93, False),
    (39, 57, "jon@example.org", "2022-08-22", "IT", 26, False),
    (40, 43, "kim@example.org", "2024-04-01", "PT", 60, False),
)


def rows() -> list[dict]:
    """The table as dictionaries, the ``anomaly`` label included."""
    return [dict(zip((*COLUMNS, "anomaly"), r)) for r in _ROWS]


def dataset_id() -> str:
    """A stable fingerprint of the table: the subject the budget and ledger are keyed on."""
    payload = json.dumps(_ROWS, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()[:12]


N_ANOMALIES = sum(1 for r in _ROWS if r[-1])
