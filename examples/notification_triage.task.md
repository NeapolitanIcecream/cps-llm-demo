You are compiling a Program IR for a high-frequency notification triage stream.

Input event:
{
  "event_id": string,
  "event_type": "notification",
  "payload": {
    "app": string,
    "title": string,
    "body": string,
    "received_at": string
  }
}

Output action draft:
{
  "event_id": string,
  "kind": "ignore" | "create_task" | "create_calendar_event" | "draft_reply",
  "title": string,
  "datetime_hint": string | null
}

Compile a Program that:
1. First tries any installed deterministic fast paths using the generic fast_path_apply local tool if fast paths exist.
2. If no fast path hits, use weak model tasks for cheap semantic classification/extraction.
3. If weak output is low confidence or schema-invalid, capture the continuation to Think.
4. Think may return the missing value, request a weak probe, or propose a ProgramPatch that installs a new generic fast path.

Do not encode notification-specific logic in Rust. Any fast path must appear as Program IR data or FastPathSpec data.
