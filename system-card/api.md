---
depth: 2
project: Crucible
scan_hash: sha256:f2bf6ccd182d6086
scanned_at: '2026-08-01T12:22:10.381775+00:00'
section: api
---

## API Surface

Based on the architectural analysis, Crucible is a CLI-driven, verifier-grounded multi-agent search engine and orchestrator rather than a traditional web service. It operates via local execution environments and sandboxed workers rather than exposing an HTTP, REST, GraphQL, or WebSocket API surface.

### Routes

No HTTP routes or web-facing endpoints are exposed by the Crucible application. 

The primary entry points for the system are programmatic and command-line interfaces rather than network routes:
*   **CLI Entry Point**: Executed via `crucible/cli.py`, which handles user inputs to initiate task evaluations.
*   **SDK/Orchestration**: Programmatic execution is managed through `crucible/orchestrator.py` and `crucible/task.py`.

### Request/Response Contracts

As there is no web server or API routing layer, there are no standard HTTP request/response JSON contracts. Interactions with the system are managed via SDK artifacts and CLI commands.

*Note: No internal data models or schemas for the orchestrator-worker payloads were exposed in the provided source analysis.*

### Authentication & Authorization

Crucible does not implement inbound authentication or authorization mechanisms (e.g., JWT, OAuth, or Session cookies), as it does not serve external client requests over a network. 

However, it heavily utilizes **outbound authentication** via API keys to communicate with third-party LLM generation providers. The following authentication configurations are supported via environment variables:

| Provider | Environment Variable(s) | Description |
| :--- | :--- | :--- |
| **Anthropic** | `ANTHROPIC_API_KEY` | Configures API access for Anthropic Claude models. [.env.example:4-5] |
| **Google** | `GOOGLE_API_KEY` or `GEMINI_API_KEY` | Authenticates requests to Google Gemini models. [.env.example:7-10] |
| **OpenAI / Local** | `OPENAI_API_KEY`, `OPENAI_BASE_URL` | Connects to OpenAI or OpenAI-compatible local/alternative endpoints. [.env.example:12-14] |

### Error Handling

Crucible does not return standard HTTP status codes (e.g., `400 Bad Request`, `500 Internal Server Error`) or JSON error envelopes. 

Errors within the system are handled at the process level during the optimization loop (orchestrator-worker execution) or surfaced as CLI exceptions. External verification failures are treated as deterministic feedback within the multi-agent search engine's iterative refinement process rather than as structural API faults.

## Sources
- .env.example