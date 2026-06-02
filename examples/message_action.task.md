You are compiling a workflow program.

Input:
A single message event:

```json
{
  "event_id": "string",
  "text": "string"
}
```

Output:
A single action draft:

```json
{
  "event_id": "string",
  "kind": "ignore | create_task | create_calendar_event | draft_reply",
  "title": "string",
  "datetime_hint": "string | null",
  "source": "weak_model | strong_think"
}
```

Compile a Program IR that:

1. Calls the weak model with a `WeakCall` instruction for semantic classification and extraction.
2. Validates the weak model output against the action draft schema.
3. Lets the runtime capture a continuation if the weak output is low confidence or invalid.
4. Finishes with the action draft value.

Do not encode business keywords in the Program.
Use `WeakCall` instructions for semantic decisions.
Do not write a weak-to-strong router. Strong Think only handles unresolved continuation frames.
