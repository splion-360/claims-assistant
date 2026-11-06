# Project coding conventions

- Begin every Python file with a one-line module docstring describing its purpose.
  Describe what the module does; keep task history, instructional logs, and lists
  of absent features out of the module description.
- Document every function and method, including helpers and test functions, with
  Google-style docstrings. Use a concise summary and add `Args`, `Returns`,
  `Yields`, and `Raises` sections where applicable. Document meaningful contracts
  and side effects rather than restating the implementation.
- Follow YAGNI: implement current requirements; defer speculative functionality.
- Follow KISS: prefer the simplest design that satisfies the requirements.
- Follow DRY: maintain a single source of truth for shared knowledge and behavior;
  avoid premature abstractions for superficially similar code.
- Keep changes scoped to the authorized increment and preserve working behavior.
