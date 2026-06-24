# -*- coding: utf-8 -*-
# Copyright 2007-2026 The HyperSpy developers
#
# This file is part of HyperSpy.
#
# HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with HyperSpy. If not, see <https://www.gnu.org/licenses/#GPL>.
"""
Tests for the hyperspy-ml-algorithms package.
"""

import subprocess
import sys

import pytest

# Exclude extracted test files — they still import hyperspy and will be
# refactored in Tasks 10-11 to use the new estimator API.
collect_ignore = [
    "test_incremental_svd.py",
    "test_mlpca.py",
    "test_ornmf.py",
    "test_rpca.py",
    "test_svd_pca.py",
]


def pytest_configure(config):
    """Verify hyperspy is NOT a pip-installed package."""
    result = subprocess.run(
        [sys.executable, "-m", "pip", "list", "--format=columns"],
        capture_output=True,
        text=True,
        check=False,
    )
    for line in result.stdout.splitlines():
        parts = line.strip().split()
        if parts and parts[0].lower() == "hyperspy":
            pytest.exit(
                "hyperspy is installed (pip list shows it). "
                "hyperspy-ml-algorithms must be tested without hyperspy.",
                returncode=1,
            )
