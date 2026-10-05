from rag_project.app.main_cleaning import build_parser


def test_cleaning_parser_accepts_wikipedia_arguments():
    args = build_parser().parse_args([
        "--wikipedia",
        "Artificial intelligence",
        "--wiki-lang",
        "en",
        "--wiki-max-docs",
        "2",
        "--allow-remote",
    ])

    assert args.wikipedia == ["Artificial intelligence"]
    assert args.wiki_lang == "en"
    assert args.wiki_max_docs == 2
    assert args.allow_remote is True
