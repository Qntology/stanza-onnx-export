# 다국어 주소 키워드 및 패턴 사전
ADDRESS_DICTIONARY = {
    "ko": {
        "admin": ["서울특별시", "특별시", "광역시", "특별자치시", "도", "시", "군", "구", "읍", "면", "동", "리", "가", "나"],
        "street": ["대로", "로", "길", "번길", "단지", "마을"],
        "zipcode_regex": r"\b\d{5}\b",
        "order": "large_to_small"
    },
    "en": {
        "admin": ["State", "City", "County", "Province", "Region", "District", "Territory", "Prefecture"],
        "street": ["Street", "St", "Road", "Rd", "Avenue", "Ave", "Lane", "Ln", "Drive", "Dr", "Boulevard", "Blvd", "Court", "Ct", "Parkway", "Pkwy", "Square", "Sq", "Terrace", "Ter", "Way"],
        "zipcode_regex": r"\b\d{5}(-\d{4})?\b",
        "order": "small_to_large"
    },
    "de": {
        "admin": ["Bundesland", "Kreis", "Stadt", "Bezirk", "Gemeinde", "Landkreis"],
        "street": ["straße", "str.", "weg", "platz", "gasse", "allee", "ring", "damm", "ufer", "zeile"],
        "zipcode_regex": r"\b\d{5}\b",
        "order": "small_to_large"
    },
    "es": {
        "admin": ["Provincia", "Municipio", "Barrio", "Localidad", "Comunidad", "Ciudad", "Región"],
        "street": ["Calle", "C/", "Avenida", "Av.", "Paseo", "Plaza", "Camino", "Ronda", "Estrada", "Travesía", "Via", "Vía"],
        "zipcode_regex": r"\b\d{5}\b",
        "order": "small_to_large"
    },
    "fr": {
        "admin": ["Région", "Département", "Arrondissement", "Commune", "Quartier", "Ville", "Cité"],
        "street": ["Rue", "Avenue", "Av.", "Boulevard", "Bd", "Place", "Chemin", "Route", "Quai", "Allée", "Passage", "Impasse"],
        "zipcode_regex": r"\b\d{5}\b",
        "order": "small_to_large"
    },
    "ja": {
        "admin": ["東京都", "道", "府", "県", "市", "区", "郡", "村"],
        "street": ["丁目", "番지", "番", "号", "町", "条", "筋"],
        "zipcode_regex": r"\b\d{3}-\d{4}\b",
        "order": "large_to_small"
    },
    "zh-hans": {
        "admin": ["省", "直辖市", "市", "区", "县", "镇", "乡", "街道"],
        "street": ["路", "街", "大道", "巷", "弄", "号", "单元", "室"],
        "zipcode_regex": r"\b\d{6}\b",
        "order": "large_to_small"
    },
    "pt": {
        "admin": ["Distrito", "Concelho", "Estado", "Província", "UF", "Região"],
        "street": ["Rua", "R.", "Avenida", "Av.", "Praça", "Pr.", "Largo", "Lg.", "Travessa", "Tv.", "Estrada"],
        "zipcode_regex": r"\b\d{4}-\d{3}\b|\b\d{5}-\d{3}\b",
        "order": "small_to_large"
    },
    "ar": {
        "admin": ["منطقة", "محافظة", "مدينة", "بلدية"],
        "street": ["شارع", "طريق", "ميدان", "نهج"],
        "zipcode_regex": r"\b\d{5}(-\d{4})?\b",
        "order": "large_to_small"
    },
    "cs": {
        "admin": ["Kraj", "Okres", "Obec", "Město", "Čtvrť"],
        "street": ["ulice", "ul.", "náměstí", "nám.", "třída", "tř.", "nábřeží"],
        "zipcode_regex": r"\b\d{3}\s?\d{2}\b",
        "order": "small_to_large"
    },
    "it": {
        "admin": ["Regione", "Provincia", "Comune", "Località", "Quartiere"],
        "street": ["Via", "Viale", "Corso", "Piazza", "Largo", "Vico", "Vicolo", "Strada"],
        "zipcode_regex": r"\b\d{5}\b",
        "order": "small_to_large"
    },
    "nl": {
        "admin": ["Provincie", "Gemeente", "Stad", "Wijk"],
        "street": ["Straat", "Laan", "Weg", "Gracht", "Singel", "Plein", "Dijk"],
        "zipcode_regex": r"\b[1-9]\d{3}\s?[A-Z]{2}\b",
        "order": "small_to_large"
    }
}
