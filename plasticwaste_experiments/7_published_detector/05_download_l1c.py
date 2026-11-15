"""
Download the two Conakry Level-1C scenes from the Copernicus Data Space, for Acolite.

MADOS was built from Acolite Rayleigh-corrected reflectance, which starts from L1C. These
are the same two dates as the L2A scenes (tile T28PFR). About 620 MB each, saved on E:.

The login is read from C:\\Users\\ASUS\\.cdse\\credentials.json, which stays on this
computer and is not part of the experiments folder.
"""
import json
import sys
import zipfile
from pathlib import Path

import requests

CRED = Path.home() / ".cdse" / "credentials.json"
DEST = Path(r"E:\SDG AI Lab\Guinea\marinext_test\l1c")
PRODUCTS = {"2024-01-11": ("4619844d-d7d5-4b85-af2c-39a6324f4d12",
                           "S2B_MSIL1C_20240111T111329_N0510_R137_T28PFR_20240115T143406.SAFE"),
            "2024-01-26": ("2de3cf71-4b9b-4395-8c87-1f78e1c3d618",
                           "S2A_MSIL1C_20240126T111331_N0510_R137_T28PFR_20240126T145153.SAFE")}
TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"


def token():
    c = json.loads(CRED.read_text(encoding="utf-8"))
    r = requests.post(TOKEN_URL, data={"client_id": "cdse-public", "grant_type": "password",
                                       "username": c["username"], "password": c["password"]}, timeout=60)
    if r.status_code != 200:
        raise SystemExit(f"login failed: {r.status_code} {r.json().get('error_description', '')}")
    return r.json()["access_token"]


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    for date, (pid, name) in PRODUCTS.items():
        safe = DEST / name
        if safe.exists():
            print("have", name)
            continue
        part = DEST / f"{name}.zip.part"
        url = f"https://download.dataspace.copernicus.eu/odata/v1/Products({pid})/$value"
        s = requests.Session()
        s.headers["Authorization"] = f"Bearer {token()}"
        r = s.get(url, allow_redirects=False, stream=True, timeout=60)
        while r.status_code in (301, 302, 303, 307):  # keep the token across the redirect
            r = s.get(r.headers["Location"], allow_redirects=False, stream=True, timeout=60)
        if r.status_code != 200:
            raise SystemExit(f"download of {date} failed: {r.status_code}")
        got = 0
        with open(part, "wb") as f:
            for chunk in r.iter_content(1 << 22):
                f.write(chunk)
                got += len(chunk)
        print(date, round(got / 1e6), "MB", flush=True)
        with zipfile.ZipFile(part) as z:
            z.extractall(DEST)
        part.unlink()
        if not safe.exists():
            raise SystemExit(f"{name} not found after unzipping")
        print("unzipped", name, flush=True)


if __name__ == "__main__":
    sys.exit(main())
