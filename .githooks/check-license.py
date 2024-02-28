#! python
#
# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

import datetime
import sys
import os

LICENSE_TYPE = "Apache 2.0"
LICENSE_STR = f"SPDX-License-Identifier: {LICENSE_TYPE}"

COPYRIGHT = f"Copyright © {datetime.datetime.now().year} Intel Corporation"

if __name__ == "__main__":
    ret = 0
    for filename in sys.argv:
        _, file_extension = os.path.splitext(filename)
        if "CMakeLists.txt" in filename or file_extension in [
            ".h",
            ".hpp",
            ".cpp",
            ".c",
            ".py",
            ".js",
            ".sh",
        ]:
            with open(filename, encoding="utf-8") as fp:
                text = fp.read()
                if LICENSE_STR not in text:
                    print(f"[pre-commit] {filename} does not have a valid license!")
                    ret = 1
                if COPYRIGHT not in text:
                    print(f"[pre-commit] {filename} does not have a valid copyright!")
                    ret = 1

    sys.exit(ret)
