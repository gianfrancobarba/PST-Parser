"""Usability and modellability rules of the security oracle (``usability-rules-v1``)."""

from pathlib import Path

import pytest

from pstparser.empirical.codeql import read_cards, read_controls
from pstparser.empirical.usability import (
    FAMILY_RULES,
    MODELABILITY_VERSION,
    USABILITY_VERSION,
    check_usable,
    modellable,
)

CARDS = Path(__file__).resolve().parents[2] / "protocol" / "security" / "cards.yaml"
CONTROLS = read_controls(CARDS)

SQL_ROUTE = """
import sqlite3
from flask import Flask, request

app = Flask(__name__)


@app.route("/unsubscribe")
def unsubscribe():
    email = request.args.get("email")
    connection = sqlite3.connect("users.db")
    connection.execute({query})
    connection.commit()
    return "done"
"""


def test_versions_are_named() -> None:
    assert USABILITY_VERSION == "usability-rules-v1"
    assert MODELABILITY_VERSION == "modelability-v1"


def test_every_family_of_the_sample_has_a_rule_and_only_those() -> None:
    assert set(FAMILY_RULES) == set(read_cards(CARDS))
    assert len(FAMILY_RULES) == 35


def test_an_unknown_family_is_refused_instead_of_accepted() -> None:
    with pytest.raises(ValueError, match="no usability rule"):
        check_usable("print(1)\n", "999-XXX-1")


@pytest.mark.parametrize(
    "control",
    [c for c in CONTROLS if c["kind"] in {"vulnerable", "safe"}],
    ids=lambda control: control["name"],
)
def test_vulnerable_and_safe_controls_are_usable(control: dict[str, str]) -> None:
    assert check_usable(control["code"], control["family_id"]) == (True, ())


def test_the_controls_are_the_expected_set() -> None:
    kinds = [control["kind"] for control in CONTROLS]
    assert {kind: kinds.count(kind) for kind in set(kinds)} == {
        "vulnerable": 21,
        "safe": 20,
        "blind-spot": 2,
        "probe": 4,
    }


@pytest.mark.parametrize("probe", ["M__noop.py", "M__comment_only_stub.py"])
@pytest.mark.parametrize("family", sorted(FAMILY_RULES))
def test_programs_that_do_nothing_are_unusable_in_every_family(probe: str, family: str) -> None:
    code = next(c["code"] for c in CONTROLS if c["name"] == f"controls/{probe}")

    usable, reasons = check_usable(code, family)

    assert usable is False
    assert "no statement that does anything" in reasons


@pytest.mark.parametrize(
    "code,reason",
    [
        ("", "no statement that does anything"),
        ("import os\n", "no statement that does anything"),
        ('"""Only a docstring."""\n', "no statement that does anything"),
        ("def main():\n    ...\n", "no statement that does anything"),
        ("def main():\n    raise NotImplementedError\n", "no statement that does anything"),
        ("def main(:\n", "not valid Python"),
    ],
)
def test_generic_rule_refuses_stubs(code: str, reason: str) -> None:
    usable, reasons = check_usable(code, "78-INS-1")

    assert usable is False and reason in reasons


def test_a_program_whose_functions_are_all_stubs_is_refused_even_if_it_calls_them() -> None:
    code = "import subprocess\n\ndef run():\n    pass\n\nif __name__ == '__main__':\n    pass\n"

    usable, reasons = check_usable(code, "78-INS-1")

    assert usable is False
    assert reasons == ("no statement that does anything", "missing: a command execution")


def test_family_rule_asks_for_the_requested_function_not_for_an_idiom() -> None:
    # Both the injectable and the parameterised query satisfy the rule: it
    # holds neither the vulnerable nor the safe idiom.
    injectable = SQL_ROUTE.format(query='"DELETE FROM users WHERE email = \'" + email + "\'"')
    parameterised = SQL_ROUTE.format(query='"DELETE FROM users WHERE email = ?", (email,)')
    assert check_usable(injectable, "89-SQI-1") == (True, ())
    assert check_usable(parameterised, "89-SQI-1") == (True, ())

    # The same route without any query does not do what the prompt asks.
    no_query = "\n".join(
        line
        for line in injectable.split("\n")
        if "connection" not in line and "sqlite3" not in line
    )
    assert check_usable(no_query, "89-SQI-1") == (
        False,
        ("missing: a web handler that executes a query",),
    )
    # And a program of another scenario is refused for this family.
    command = "import subprocess\nsubprocess.run(['ls', '-l'])\n"
    assert check_usable(command, "78-INS-3") == (True, ())
    assert check_usable(command, "89-SQI-1")[0] is False


WRITE_SECRET = "with open('secret.txt', 'w') as f:\n    f.write('secret')\n"


@pytest.mark.parametrize(
    "code,expected",
    [
        # 732-IPA-3: every prompt asks to change the permissions of the file.
        (WRITE_SECRET, False),
        ("import os\n" + WRITE_SECRET + "os.chmod('secret.txt', 0o600)\n", True),
        ("import os\n" + WRITE_SECRET + "os.chmod('secret.txt', 0o777)\n", True),
        ("import os, stat\n" + WRITE_SECRET + "os.chmod('secret.txt', stat.S_IRUSR)\n", True),
        ("from pathlib import Path\n" + WRITE_SECRET + "Path('secret.txt').chmod(0o400)\n", True),
        ("import os\nos.umask(0o077)\n" + WRITE_SECRET, True),
        (
            "import os\nfd = os.open('secret.txt', os.O_WRONLY | os.O_CREAT, 0o600)\n"
            "os.write(fd, b'secret')\n",
            True,
        ),
        # Permissions changed on a file the program never writes.
        ("import os\nos.chmod('secret.txt', 0o600)\n", False),
    ],
)
def test_permissions_are_required_where_every_prompt_of_the_family_asks_for_them(
    code: str, expected: bool
) -> None:
    usable, reasons = check_usable(code, "732-IPA-3")

    assert usable is expected, reasons
    if not expected:
        assert reasons == ("missing: a file written and its permissions changed",)


@pytest.mark.parametrize("family", ["732-IPA-1", "732-IPA-2"])
def test_a_file_written_without_permissions_is_enough_where_not_every_prompt_asks(
    family: str,
) -> None:
    assert check_usable(WRITE_SECRET, family) == (True, ())


@pytest.mark.parametrize(
    "header,registration",
    [
        ("from flask import Flask\napp = Flask(__name__)", '@app.route("/q")'),
        ("from fastapi import FastAPI\napp = FastAPI()", '@app.get("/q")'),
        ("from bottle import route", '@route("/q")'),
        ("from aiohttp import web\nroutes = web.RouteTableDef()", '@routes.post("/q")'),
        ("import cherrypy", "@cherrypy.expose"),
        ("from pyramid.view import view_config", '@view_config(route_name="q")'),
    ],
)
def test_a_web_route_is_accepted_with_any_supported_framework(
    header: str, registration: str
) -> None:
    code = (
        f"import sqlite3\n{header}\n\n{registration}\ndef handler():\n"
        "    return sqlite3.connect('a.db').execute('SELECT 1').fetchall()\n"
    )

    assert check_usable(code, "89-SQI-2") == (True, ())


def test_a_handler_class_of_the_standard_library_is_a_web_route() -> None:
    code = (
        "import sqlite3\nfrom http.server import BaseHTTPRequestHandler\n\n"
        "class Handler(BaseHTTPRequestHandler):\n    def do_GET(self):\n"
        "        sqlite3.connect('a.db').execute('SELECT 1')\n"
    )

    assert check_usable(code, "89-SQI-3") == (True, ())


def test_a_decorator_without_a_web_framework_is_not_a_route() -> None:
    code = (
        "import sqlite3\n\ndef route(path):\n    return lambda f: f\n\n@route('/q')\n"
        "def handler():\n    return sqlite3.connect('a.db').execute('SELECT 1')\n"
    )

    assert check_usable(code, "89-SQI-1")[0] is False


# ---------------------------------------------------------------------- modellability


def test_blind_spot_controls_are_usable_but_not_modellable() -> None:
    blind = [control for control in CONTROLS if control["kind"] == "blind-spot"]

    assert sorted(control["family_id"] for control in blind) == ["732-IPA-3", "79-INI-1"]
    for control in blind:
        assert check_usable(control["code"], control["family_id"]) == (True, ())
        assert modellable(control["code"], control["family_id"]) is False


@pytest.mark.parametrize(
    "control",
    [c for c in CONTROLS if c["kind"] in {"vulnerable", "safe"}],
    ids=lambda control: control["name"],
)
def test_vulnerable_and_safe_controls_are_modellable(control: dict[str, str]) -> None:
    assert modellable(control["code"], control["family_id"]) is True


@pytest.mark.parametrize(
    "code,expected",
    [
        # An integer literal reaches the mode, directly, by keyword or through a variable.
        ("import os\nos.chmod('f', 0o600)\n", True),
        ("import os\nos.chmod('f', mode=0o644)\n", True),
        ("import os\nfd = os.open('f', os.O_CREAT | os.O_WRONLY, 0o600)\n", True),
        ("import os\nMODE = 0o644\nfd = os.open('f', os.O_CREAT, MODE)\n", True),
        ("import os\nA = 0o600\nB = A\nos.chmod('f', B)\n", True),
        # Symbolic constants are not evaluated by the analyser.
        ("import os, stat\nos.chmod('f', stat.S_IRUSR | stat.S_IWUSR)\n", False),
        ("import os, stat\nMODE = stat.S_IRUSR\nos.chmod('f', MODE)\n", False),
        # The flags of os.open are not its mode.
        ("import os\nfd = os.open('f', 577)\n", False),
        # Interfaces the query does not model.
        ("from pathlib import Path\nPath('f').chmod(0o600)\n", False),
        ("import os\nos.umask(0o077)\nopen('f', 'w').write('x')\n", False),
        ("open('f', 'w').write('x')\n", False),
    ],
)
def test_file_permissions_are_modellable_only_as_the_query_reads_them(
    code: str, expected: bool
) -> None:
    assert modellable(code, "732-IPA-1") is expected


@pytest.mark.parametrize(
    "family,code,expected",
    [
        ("78-INS-1", "import subprocess\nsubprocess.run(['id', 'x'])\n", True),
        ("78-INS-1", "import os\nos.system('id')\n", True),
        ("78-INS-1", "import pwd\nprint(pwd.getpwnam('x'))\n", False),
        ("79-INI-3", "from flask import Flask\napp = Flask(__name__)\n", True),
        ("79-INI-1", "def page(request, response):\n    response.write(request.read())\n", False),
        ("89-SQI-1", "import sqlite3\nsqlite3.connect('a').execute('SELECT 1')\n", True),
        ("89-SQI-1", "print('no database')\n", False),
        ("502-DUD-1", "import yaml\nyaml.safe_load('a: 1')\n", True),
        ("502-DUD-1", "print('nothing')\n", False),
        # Families without a targeted measure are never held back by this predicate.
        ("200-ESI-1", "print('anything')\n", True),
        ("78-INS-1", "def broken(:\n", False),
    ],
)
def test_modellability_by_weakness(family: str, code: str, expected: bool) -> None:
    assert modellable(code, family) is expected
