def match_dataset_crop(name: str, vocab: list[str], default_crop: str) -> str:
    """Map a recommendation label (e.g. 'rice') to a crop string seen in yield training data."""
    if not vocab:
        return default_crop
    n = (name or "").strip().lower()
    if not n:
        return default_crop

    stripped_vocab = [(str(v).strip(), str(v).strip().lower()) for v in vocab]

    for orig, low in stripped_vocab:
        if low == n:
            return orig

    for orig, low in stripped_vocab:
        if n in low or low in n:
            return orig

    # Common synonyms between recommendation labels and production dataset naming
    synonyms = {
        "rice": "rice",
        "maize": "maize",
        "chickpea": "gram",
        "kidneybeans": "other kharif pulses",
        "pigeonpeas": "arhar",
        "mothbeans": "moong",
        "mungbean": "moong",
        "blackgram": "urad",
        "lentil": "masoor",
        "pomegranate": "pomegranate",
        "banana": "banana",
        "mango": "mango",
        "grapes": "grapes",
        "watermelon": "watermelon",
        "muskmelon": "muskmelon",
        "apple": "apple",
        "orange": "orange",
        "papaya": "papaya",
        "coconut": "coconut",
        "cotton": "cotton",
        "jute": "jute",
        "coffee": "coffee",
    }
    key = synonyms.get(n)
    if key:
        for orig, low in stripped_vocab:
            if key in low:
                return orig

    return default_crop
