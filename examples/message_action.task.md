You are compiling a workflow program for a typed CPS LLM runtime.

Input:
An array of message events:

```json
[
  {
    "event_id": "string",
    "text": "string"
  }
]
```

Output:
An array of action drafts:

```json
[
  {
    "event_id": "string",
    "kind": "ignore | create_task | create_calendar_event | draft_reply",
    "title": "string",
    "datetime_hint": "string | null"
  }
]
```

Compile a Program IR v2 that:

1. Defines `main(messages)`.
2. Uses `Map` to process each message with a `process_message(message)` function.
3. In `process_message`, performs a weak `ModelTask` to classify intent.
4. Performs a second weak `ModelTask` to extract the final action draft using `{ "message": message, "intent": intent }`.
5. Uses `AcceptancePolicy` on weak effects so low confidence or schema-invalid output captures to `Think`.
6. Returns the mapped array of drafts.

Do not encode business keywords in the Program.
Do not compile a weak-to-strong router.
Use `Perform(ModelTask { strength: Weak })` for semantic work.
Let the runtime capture continuations and schedule `Think` only when a typed effect is unresolved.
Models may request nested effects only by returning typed `HandlerDecision::RequestEffect`.
