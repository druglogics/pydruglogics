# Contributing to PyDrugLogics

Thank you for considering contributing to PyDrugLogics! This guide will help you understand how to contribute and ensure that your contributions are efficient and impactful.

## Code of Conduct

By participating in this project, you agree to uphold our [Code of Conduct](CODE_OF_CONDUCT.md). Please read it before contributing.


## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [I Have a Question](#i-have-a-question)
  - [I Want To Contribute](#i-want-to-contribute)
  - [Reporting Bugs](#reporting-bugs)
  - [Suggesting Enhancements](#suggesting-enhancements)
  - [Your First Code Contribution](#your-first-code-contribution)
  - [Improving The Documentation](#improving-the-documentation)
- [Styleguides](#styleguides)
  - [Commit Messages](#commit-messages)

## Code of Conduct

This project and everyone participating in it is governed by the [PyDrugLogics Code of Conduct](https://github.com/druglogics/pydruglogics/blob/CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to [szekereslaura98@gmail.com](mailto:szekereslaura98@gmail.com).

## I Have a Question

> If you want to ask a question, we assume that you have read the available [Documentation](https://druglogics.github.io/pydruglogics/).

Before you ask a question, it is best to search for existing [Issues](https://github.com/druglogics/pydruglogics/issues) that might help you. If you find a suitable issue and still need clarification, you can write your question in that issue. It is also advisable to search the internet for answers first.

If you still need to ask a question, we recommend the following:

- Open an [Issue](https://github.com/druglogics/pydruglogics/issues/new).
- Provide as much context as you can about what you're running into.
- Provide project and platform versions (e.g., Python version, OS, dependencies).

We will address your issue as soon as possible.

## I Want To Contribute

> ### Legal Notice 
> When contributing to this project, you must agree that you have authored 100% of the content, that you have the necessary rights to the content, and that the content you contribute may be provided under the project license.

### Reporting Bugs

#### Before Submitting a Bug Report

A good bug report should be clear and detailed. Before submitting a report, please:

- Ensure you are using the latest version.
- Check the [documentation](https://druglogics.github.io/pydruglogics/) to confirm your issue isn’t a misconfiguration.
- Search the [issues tracker](https://github.com/druglogics/pydruglogics/issues) to see if the bug has already been reported.
- Gather information such as:
  - Stack trace (Traceback)
  - OS, Platform, and Version (Windows, Linux, macOS, x86, ARM)
  - Python version and dependencies
  - Steps to reliably reproduce the issue

#### How Do I Submit a Good Bug Report?

> Security issues must not be reported publicly. Instead, send details to [szekereslaura98@gmail.com](mailto:szekereslaura98@gmail.com).

To submit a bug report:

- Open an [Issue](https://github.com/druglogics/pydruglogics/issues/new).
- Clearly describe the expected vs. actual behavior.
- Provide reproduction steps with minimal code examples.
- Include relevant system details and logs.

Once filed, the project team will:
- Label the issue appropriately.
- Attempt to reproduce the bug.
- If reproducible, mark it as `needs-fix` and prioritize it.

### Suggesting Enhancements

Before submitting an enhancement:

- Ensure you are using the latest version.
- Review the [documentation](https://druglogics.github.io/pydruglogics/) to check if the feature already exists.
- Search the [issues tracker](https://github.com/druglogics/pydruglogics/issues) for similar suggestions.
- Ensure the feature aligns with the project’s goals.

To submit an enhancement suggestion:

- Open an [Issue](https://github.com/druglogics/pydruglogics/issues/new) with a clear and descriptive title.
- Provide a step-by-step description of the enhancement.
- Describe the current vs. expected behavior and why the change is beneficial.
- Include screenshots or references to similar implementations, if applicable.

### Your First Code Contribution

To contribute code:

1. Fork the repository and clone it locally:
   ```bash
   git clone https://github.com/your-username/pydruglogics.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Install Pytest and run tests to ensure the setup works:
   ```bash
   pip install pytest pytest-cov
   pytest --cov=pydruglogics tests
   ```
4. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
5. Develop, test, and commit your changes:
   ```bash
   git add .
   git commit -m "feat: add feature description"
   ```
6. Push your branch and create a pull request.

### Improving The Documentation

Good documentation is essential for usability and maintainability. If you have suggestions, corrections, or improvements, we encourage you to get in touch.

#### How To Get Involved

If you would like to contribute to improving the documentation, please reach out to the project maintainers via email at [szekereslaura98@gmail.com](mailto:szekereslaura98@gmail.com) or open an issue on [GitHub](https://github.com/druglogics/pydruglogics/issues). We appreciate your feedback and look forward to collaborating with you!

## Styleguides

### Commit Messages

Commit messages should follow this structure:

```
[Short description]

[Optional body with details about the change.]

[References to issues, e.g., Closes #123.]
```

## Closing Notes

We value your contributions and look forward to collaborating with you to make PyDrugLogics better. If you have any questions or need clarification, feel free to reach out through issues or email [szekereslaura98@gmail.com](mailto:szekereslaura98@gmail.com). Thank you for your support!
