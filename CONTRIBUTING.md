# Contributing

Thanks for wanting to make this better. Contributions of all kinds are welcome.

---

## What's Most Useful

**Bug fixes** — If a test is wrong, an explanation is misleading, or something doesn't run, please open an issue or send a PR. These are the highest priority.

**Clearer explanations** — If you read a docstring and thought "this is confusing," the right answer is probably to fix it. Simpler language beats precise language for a tutorial.

**Test additions** — More edge cases in the test suite are always welcome.

**New architectures or domains** — If you want to add a Tutorial 05 for something like a GRU or a CNN applied to a new domain, open an issue first to discuss.

---

## What to Avoid

**Introducing dependencies** — The whole point is pure Python. Please don't add NumPy, PyTorch, or anything that requires `pip install`. Even `numpy` for "just one thing" breaks the promise.

**Making the code more "Pythonic" at the cost of clarity** — List comprehensions that do three things in one line are hard to teach from. Explicit is better than clever here.

**Changing the file numbering** — The `01_`, `02_` prefixes are intentional and readers navigate by them.

---

## How to Contribute

1. Fork the repo and clone your fork
2. Make your changes
3. Run the tests: `python3 -m pytest tutorials/ -v`
4. Open a PR with a short description of what you changed and why

For anything substantial (new tutorial, significant restructuring), open an issue first so we can discuss the approach before you invest a lot of time.

---

## Code Style

- Pure Python only (stdlib is fine: `math`, `random`, `os`, `sys`, `json`)
- Docstrings in NumPy style with `Parameters`, `Returns`, `Examples` sections
- Comments explain the *why*, not the *what*
- Line length: up to 100 characters
- No type annotations required (they can obscure simple ideas for beginners)

---

## A Note on Tone

The tutorial is written in a friendly, slightly self-deprecating first person. If you're adding explanatory text, match that tone. "This tripped me up for hours" is more useful to a reader than "Note that this implementation detail is non-trivial."
