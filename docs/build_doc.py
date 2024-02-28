#! python
#
# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

from http.server import HTTPServer, SimpleHTTPRequestHandler
from ghp_import import ghp_import
from typing import List, Union
import subprocess
import argparse
import shutil
import os


def define_and_parse_args():
    parser = argparse.ArgumentParser(description="Build documentations")
    parser.add_argument(
        "action",
        type=str,
        choices=["build", "serve", "gh-deploy"],
        help="Name of the model to export",
    )

    return parser.parse_args()


doc_root = os.path.dirname(os.path.abspath(__file__))
root = os.path.abspath(os.path.join(doc_root, ".."))
doxygen_available = shutil.which("doxygen") is not None


def clean_dirs(dir_names: Union[List[str], str]) -> None:
    if isinstance(dir_names, str):
        dir_names = [dir_names]
    for name in dir_names:
        xml_dir = os.path.join(doc_root, name)
        if os.path.exists(xml_dir) and os.path.isdir(xml_dir):
            shutil.rmtree(xml_dir)


def build_doc():

    clean_dirs(["build", "xml"])

    if not doxygen_available:
        raise RuntimeError("Doxygen is needed to build documentation")

    yield subprocess.check_output(
        ["doxygen", "Doxyfile"], cwd=doc_root, stderr=subprocess.STDOUT
    ).decode()
    yield subprocess.check_output(
        ["sphinx-apidoc", "-o", "source/python", "../intel_npu_acceleration_library"],
        cwd=doc_root,
        stderr=subprocess.STDOUT,
    ).decode()
    yield subprocess.check_output(
        ["sphinx-build", "-b", "html", "source", "build"],
        cwd=doc_root,
        stderr=subprocess.STDOUT,
    ).decode()

    clean_dirs("xml")


def build():
    for out in build_doc():
        print(out)


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory="build", **kwargs)


def serve(hostname="localhost", port=8000):
    build()
    server_address = (hostname, port)
    httpd = HTTPServer(server_address, Handler)
    print(f"Serving at address {hostname}:{port}")
    httpd.serve_forever()


def get_git_sha() -> str:
    return (
        subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
        )
        .decode()
        .strip()
    )


def deploy():
    build()

    message = f"Deployed with sha {get_git_sha()}"

    try:
        ghp_import(
            os.path.join(doc_root, "build"),
            mesg=message,
            remote="origin",
            branch="gh-pages",
            push=True,
            force=True,
            use_shell=False,
            no_history=False,
            nojekyll=True,
        )
    except ghp_import.GhpError as e:
        raise RuntimeError(f"Failed to deploy to GitHub. Error: \n{e.message}")


if __name__ == "__main__":
    args = define_and_parse_args()

    if args.action == "build":
        build()
    elif args.action == "serve":
        serve()
    elif args.action == "gh-deploy":
        deploy()
    else:
        raise RuntimeError(f"Unsuported action: {args.action}")
