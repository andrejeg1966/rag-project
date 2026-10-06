from rag_project import main


def test_main_prints_message(capsys):
    main()

    captured = capsys.readouterr()
    assert captured.out == "Hello from rag-project!\n"
