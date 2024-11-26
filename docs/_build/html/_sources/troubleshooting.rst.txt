Troubleshooting
===============

This section provides guidance on common issues, logging configurations, and reporting bugs.

Supported Platforms
-------------------
- **Operating Systems**: The package has been developed and tested on Linux (Ubuntu 22.04 and later).

- **Python Versions**: This package is compatible with Python 3.10 and above.

Installation Issues
-------------------
- **Problem**: "Package not found" or "No matching distribution found"

  - **Possible cause**: This usually occurs if the required Python version or dependencies are not satisfied.
  - **Solution**:

    1. Ensure you're using Python 3.10 or newer:

    .. code-block:: text

        python --version

    2. Check the package name is correct:


    .. code-block:: text

        pip install pydruglogics

- **Problem**: "Permission denied" or "Could not install package"

  - **Possible cause**: Administrative privileges are required or there is a conflict in the environment.
  - **Solution**: Use a virtual environment:

    .. code-block:: text

          python -m venv venv
          source venv/bin/activate  # On Linux
          pip install pydruglogics


Setting Logging Level to DEBUG
------------------------------
To gather more information about program execution, you can set the logger level to `DEBUG`.
This enables detailed logging that can help diagnose issues.

Here’s how to set the logging level:

.. code-block:: python

    from pydruglogics.utils.Logger import Logger
    Logger.set_logger(level=logging.DEBUG)

Reporting Issues
----------------
If you encounter a bug or wish to request a feature, please report it on our GitHub issue tracker:

`GitHub Issues Page <https://github.com/druglogics/pydruglogics/issues>`_

When reporting an issue, include:

- Your operating system and version (e.g, Ubuntu 22.04)
- Python version (e.g. Python 3.8.10)
- The error message and traceback (if applicable)
- Steps to reproduce the issue

Thank you for your help!

