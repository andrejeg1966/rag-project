from urllib.parse import urlencode
import urllib.request

basis = "[de.wikipedia.org](https://de.wikipedia.org/w/api.php)"
parameter = urlencode({
    "action": "query",
    "list": "search",
    "srsearch": "Retrieval Augmented Generation",
    "format": "json",
})
adresse = basis + "?" + parameter

print("Adresse:", adresse)
try:
    with urllib.request.urlopen(adresse, timeout=20) as antwort:
        print("Status:", antwort.status)
        print(antwort.read(200).decode("utf-8", "replace"))
except Exception as fehler:
    print("Fehler:", type(fehler).__name__, fehler)

