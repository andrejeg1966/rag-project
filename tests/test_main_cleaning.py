import pytest

from rag_project.app.main_cleaning import build_parser


def test_cleaning_parser_accepts_url_arguments():
    args = build_parser().parse_args([
        "--url",
        "https://example.com/page",
        "--allow-remote",
    ])

    assert args.url == ["https://example.com/page"]
    assert args.allow_remote is True


@pytest.mark.parametrize("option", ["--wikipedia", "--csv-url", "--wiki-lang"])
def test_cleaning_parser_rejects_moved_sources(option):
    with pytest.raises(SystemExit):
        build_parser().parse_args([option, "x"])
