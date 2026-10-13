"""ScaleCUA generated-request router tests."""

from __future__ import annotations

import base64
import contextlib
import inspect
import json
import threading

import pytest

from lite.gym.envs.lite.scalecua.src.osworld import judges


def test_scalecua_request_encoding_preserves_auth_tuple_and_bytes():
    encoded = judges._encode_request_value(
        {
            "auth": ("", "vlc-password"),
            "data": b"payload",
            "nested": [("a", "b")],
        }
    )

    assert encoded["auth"] == {"__tuple__": ["", "vlc-password"]}
    assert encoded["data"] == {"__bytes_b64__": "cGF5bG9hZA=="}
    assert encoded["nested"] == [{"__tuple__": ["a", "b"]}]
    json.dumps(encoded)


def test_scalecua_request_in_container_script_compiles():
    payload = base64.b64encode(
        json.dumps(
            {
                "method": "GET",
                "url": "http://localhost:8080/requests/status.xml",
                "kwargs": judges._encode_request_value(
                    {"auth": ("", "vlc"), "data": b"payload"}
                ),
            }
        ).encode("utf-8")
    ).decode("ascii")

    script = judges._build_request_script(payload)

    compile(script, "<scalecua_request_in_container>", "exec")
    assert '"status_code": resp.status_code' in script


@contextlib.contextmanager
def _shared_requests_router_with_passthrough_originals():
    """Install the shared-module router with observable passthrough sentinels."""
    import requests

    saved_installed = judges._REQUESTS_PATCH_INSTALLED
    saved_request = requests.request
    saved_verbs = {
        name: getattr(requests, name)
        for name in ("get", "options", "head", "post", "put", "patch", "delete")
    }
    saved_session_request = requests.sessions.Session.request
    passthrough_calls = []

    def passthrough_request(method, url, **kwargs):
        passthrough_calls.append((method, url, kwargs))
        return f"passthrough:{method}:{url}"

    def passthrough_session_request(session, method, url, **kwargs):
        return f"session-passthrough:{method}:{url}"

    requests.request = passthrough_request
    requests.sessions.Session.request = passthrough_session_request
    judges._REQUESTS_PATCH_INSTALLED = False
    judges._install_shared_requests_router()
    requests._cua_lite_test_passthrough_calls = passthrough_calls
    try:
        yield requests
    finally:
        judges._REQUESTS_PATCH_INSTALLED = saved_installed
        requests.request = saved_request
        for name, value in saved_verbs.items():
            setattr(requests, name, value)
        requests.sessions.Session.request = saved_session_request
        with contextlib.suppress(AttributeError):
            delattr(requests, "_cua_lite_test_passthrough_calls")
        with contextlib.suppress(AttributeError):
            delattr(judges._REQUESTS_ROUTER_LOCAL, "eval_env")


class _FakeRoutingEnv:
    vm_ip = "lite-scalecua-vm"
    server_port = 5000
    chromium_port = 1337
    vlc_port = 8080

    def __init__(self, name: str):
        self.name = name
        self.calls = []

    def request_in_container(self, method: str, url: str, **kwargs):
        self.calls.append((method, url, kwargs))
        return f"{self.name}:{method}:{url}"


def test_scalecua_requests_router_is_thread_local():
    # The shared-module router patches ``requests`` itself, so a module-global
    # import routes without rebinding a generated getter's globals.
    with _shared_requests_router_with_passthrough_originals():
        module_globals = {"requests": __import__("requests")}
        exec(
            """
def generated_getter():
    return (
        requests.get("http://localhost:5000/accessibility"),
        requests.Session().get("http://127.0.0.1:1337/json"),
        requests.get("https://example.com/outside"),
    )
""",
            module_globals,
        )

        def wrapper(metric):
            def wrapped():
                return metric()
            return wrapped

        wrapped_getter = wrapper(module_globals["generated_getter"])

        with judges._routed_requests_for_function(_FakeRoutingEnv("env-main"), wrapped_getter):
            assert wrapped_getter() == (
                "env-main:GET:http://localhost:5000/accessibility",
                "env-main:GET:http://127.0.0.1:1337/json",
                "passthrough:GET:https://example.com/outside",
            )

        outputs: dict[str, tuple[str, str, str]] = {}
        barrier = threading.Barrier(3)

        def worker(name: str):
            env = _FakeRoutingEnv(name)
            with judges._routed_requests_for_function(env, wrapped_getter):
                barrier.wait()
                outputs[name] = wrapped_getter()
                barrier.wait()

        threads = [
            threading.Thread(target=worker, args=("env-a",)),
            threading.Thread(target=worker, args=("env-b",)),
        ]
        for thread in threads:
            thread.start()
        barrier.wait()
        barrier.wait()
        for thread in threads:
            thread.join()

        assert outputs == {
            "env-a": (
                "env-a:GET:http://localhost:5000/accessibility",
                "env-a:GET:http://127.0.0.1:1337/json",
                "passthrough:GET:https://example.com/outside",
            ),
            "env-b": (
                "env-b:GET:http://localhost:5000/accessibility",
                "env-b:GET:http://127.0.0.1:1337/json",
                "passthrough:GET:https://example.com/outside",
            ),
        }


def test_scalecua_router_survives_function_local_import_and_spares_external():
    with _shared_requests_router_with_passthrough_originals():
        module_globals: dict = {}
        exec(
            "def get_local():\n"
            "    import requests\n"
            "    return (\n"
            "        requests.get('http://lite-scalecua-vm:5000/execute'),\n"
            "        requests.get('http://localhost:9222/json'),\n"
            "        requests.post('https://example.com/outside'),\n"
            "    )\n",
            module_globals,
        )
        get_local = module_globals["get_local"]

        with judges._routed_requests_for_function(_FakeRoutingEnv("env-x"), get_local):
            vm, cdp, ext = get_local()
        assert vm == "env-x:GET:http://lite-scalecua-vm:5000/execute"
        assert cdp == "env-x:GET:http://localhost:9222/json"
        assert ext == "passthrough:POST:https://example.com/outside"

        assert get_local() == (
            "passthrough:GET:http://lite-scalecua-vm:5000/execute",
            "passthrough:GET:http://localhost:9222/json",
            "passthrough:POST:https://example.com/outside",
        )


def test_scalecua_router_preserves_requests_api_signatures():
    import requests
    from requests import api

    with _shared_requests_router_with_passthrough_originals():
        for name in ("get", "options", "head", "post", "put", "patch", "delete"):
            assert inspect.signature(getattr(requests, name)) == inspect.signature(
                getattr(api, name)
            )


def test_scalecua_router_preserves_positional_params_and_head_default():
    with _shared_requests_router_with_passthrough_originals() as requests:
        env = _FakeRoutingEnv("env-y")
        with judges._routed_requests_for_function(env, lambda: None):
            assert requests.get(
                "http://localhost:5000/accessibility",
                {"q": "term"},
                timeout=5,
            ) == "env-y:GET:http://localhost:5000/accessibility"
            assert requests.head(
                "http://localhost:5000/accessibility",
                timeout=5,
            ) == "env-y:HEAD:http://localhost:5000/accessibility"
            assert requests.head("https://example.com/outside") == (
                "passthrough:HEAD:https://example.com/outside"
            )

        assert env.calls[0] == (
            "GET",
            "http://localhost:5000/accessibility",
            {"params": {"q": "term"}, "timeout": 5},
        )
        assert env.calls[1] == (
            "HEAD",
            "http://localhost:5000/accessibility",
            {"timeout": 5, "allow_redirects": False},
        )
        assert requests._cua_lite_test_passthrough_calls[-1] == (
            "HEAD",
            "https://example.com/outside",
            {"allow_redirects": False},
        )


def test_scalecua_no_getter_relies_on_unrouted_local_requests_import():
    import ast
    import glob
    import os

    assert hasattr(judges, "_install_shared_requests_router")

    roots = [
        os.path.join(
            ".cache", "lite.scalecua_tasks", "judge_functions", split, "verigen_getters"
        )
        for split in ("train", "rl")
    ]
    present = [r for r in roots if os.path.isdir(r)]
    if not present:
        pytest.skip("live verigen_getters overlays not present")

    offenders = []
    for root in present:
        for path in glob.glob(os.path.join(root, "*.py")):
            with open(path) as fh:
                tree = ast.parse(fh.read())
            for node in ast.walk(tree):
                if isinstance(
                    node, (ast.FunctionDef, ast.AsyncFunctionDef)
                ) and node.name.startswith("get_"):
                    if any(
                        (
                            isinstance(x, ast.Import)
                            and any(a.name == "requests" for a in x.names)
                        )
                        or (isinstance(x, ast.ImportFrom) and x.module == "requests")
                        for x in ast.walk(node)
                    ):
                        offenders.append(node.name)
    assert offenders, "expected the known function-local-import getter class to exist"
