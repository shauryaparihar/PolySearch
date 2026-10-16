from app.evaluation.corpus import Corpus, CorpusDoc

DOCS = [
    CorpusDoc("en:Volcano", "en", "Volcano", "A volcano erupts lava and ash."),
    CorpusDoc("es:Volcán", "es", "Volcán", "Un volcán expulsa lava y ceniza."),
    CorpusDoc("fr:Volcan", "fr", "Volcan", "Un volcan projette de la lave et des cendres."),
    CorpusDoc("en:Glacier", "en", "Glacier", "A glacier is a mass of ice."),
    CorpusDoc("es:Glaciar", "es", "Glaciar", "Un glaciar es una masa de hielo."),
    CorpusDoc("fr:Glacier", "fr", "Glacier", "Un glacier est une masse de glace."),
    CorpusDoc("en:Jazz", "en", "Jazz", "Jazz is a music genre from New Orleans."),
    CorpusDoc("es:Jazz", "es", "Jazz", "El jazz es un género musical de Nueva Orleans."),
    CorpusDoc("fr:Jazz", "fr", "Jazz", "Le jazz est un genre musical né à La Nouvelle-Orléans."),
]


def corpus() -> Corpus:
    return Corpus(list(DOCS), "c" * 64)
