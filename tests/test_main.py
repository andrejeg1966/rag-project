from rag_project import main
from rag_project.math_utils import add, mult


def test_add_returns_sum():
    assert add(2, 3) == 5
    assert add(10, 5) == 15
    assert add(-1, 1) == 0


def test_mult_returns_product():
    assert mult(2, 3) == 6
    assert mult(4, 5) == 20
    assert mult(-2, 3) == -6


def test_main_prints_message(capsys):
    main()

    captured = capsys.readouterr()
    assert captured.out == "Hello from rag-project!\n"
