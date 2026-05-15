# Tenderbot

Fetches UK government tenders from the Find a Tender service and uses Claude to identify those matching configured interests.

## Development

**TDD is mandatory.** Write tests before writing implementation code.

Run tests with:
```
uv run pytest
```

## Stack

- Python, managed with `uv`
- `anthropic` SDK for LLM inference
- `requests` for the Find a Tender API
- `pydantic` for structured outputs
- `pytest` for tests
