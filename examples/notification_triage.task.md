Input:
One notification event JSON object.

Output:
One action draft JSON object:
{
  "event_id": string,
  "kind": "ignore" | "create_task" | "create_calendar_event" | "draft_reply",
  "title": string,
  "datetime_hint": string | null
}

Compile a Program IR that first tries installed deterministic fast paths, otherwise uses weak model effects for cheap semantic classification or extraction, and captures low-confidence or schema-invalid weak effects to StrongThink.

Do not assume notification-specific Rust code exists.
