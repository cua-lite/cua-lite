"""Container HEALTHCHECK / `install.sh health` probe: exit 0 iff all three sites serve.

Run inside the container. Each site opens its port only after its index has
loaded (the Shop also after its search-engine warm-up), so HTTP 200 means ready.
"""
import urllib.request

for url in (
    "http://127.0.0.1:5000/",
    "http://127.0.0.1:5001/",
    "http://127.0.0.1:5002/abc",
):
    with urllib.request.urlopen(url, timeout=4) as response:
        if response.status != 200:
            raise SystemExit(f"{url} -> HTTP {response.status}")
print("True")
