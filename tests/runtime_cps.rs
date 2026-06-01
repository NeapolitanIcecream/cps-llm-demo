use std::sync::{Arc, Mutex};

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use cps_llm_demo::domain::{
    DecisionSource, IntentKind, MessageEvent, ResolvedIntent, WeakIntentGuess,
};
use cps_llm_demo::effects::{Continuation, EffectFrame, ThinkDecision};
use cps_llm_demo::models::{StrongModel, WeakModel};
use cps_llm_demo::runtime::{CapturePolicy, Runtime, deterministic_prefilter};
use cps_llm_demo::trace::TraceCollector;

struct StaticWeak {
    result: std::result::Result<WeakIntentGuess, String>,
}

#[async_trait]
impl WeakModel for StaticWeak {
    async fn classify_message(&self, _event: &MessageEvent) -> Result<WeakIntentGuess> {
        self.result.clone().map_err(|message| anyhow!(message))
    }
}

#[derive(Clone)]
struct CountingWeak {
    result: WeakIntentGuess,
    calls: Arc<Mutex<Vec<MessageEvent>>>,
}

impl CountingWeak {
    fn new(result: WeakIntentGuess) -> Self {
        Self {
            result,
            calls: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn call_count(&self) -> usize {
        self.calls.lock().unwrap().len()
    }
}

#[async_trait]
impl WeakModel for CountingWeak {
    async fn classify_message(&self, event: &MessageEvent) -> Result<WeakIntentGuess> {
        self.calls.lock().unwrap().push(event.clone());
        Ok(self.result.clone())
    }
}

#[derive(Clone)]
struct RecordingStrong {
    decision: std::result::Result<ThinkDecision, String>,
    calls: Arc<Mutex<Vec<EffectFrame>>>,
}

#[async_trait]
impl StrongModel for RecordingStrong {
    async fn think(&self, frame: &EffectFrame) -> Result<ThinkDecision> {
        self.calls.lock().unwrap().push(frame.clone());
        self.decision.clone().map_err(|message| anyhow!(message))
    }
}

#[test]
fn continuation_is_serializable_data() {
    let continuation = Continuation::AfterClassifyMessage {
        event: MessageEvent {
            event_id: "m1".to_owned(),
            text: "明天 10 点前把新版 proposal 发我一下".to_owned(),
        },
        weak_guess: None,
        weak_error: Some("temporary model error".to_owned()),
    };

    let value = serde_json::to_value(&continuation).unwrap();
    assert_eq!(value["tag"], "after_classify_message");

    let roundtrip: Continuation = serde_json::from_value(value).unwrap();
    assert_eq!(roundtrip, continuation);
}

#[test]
fn empty_message_is_deterministic_ignore() {
    let event = MessageEvent {
        event_id: "m0".to_owned(),
        text: "   ".to_owned(),
    };

    let draft = deterministic_prefilter(&event).unwrap();
    assert_eq!(draft.source, DecisionSource::DeterministicCode);
    assert_eq!(draft.kind, IntentKind::Ignore);
}

#[test]
fn semantic_messages_are_not_deterministically_prefiltered() {
    for text in [
        "验证码 839201，五分钟内有效",
        "Your OTP is 839201",
        "Use OTP839201 to sign in",
        "Your verification code is 839201",
        "verification code UX review",
        "Schedule unsubscribe flow review Friday 3pm",
        "hotpot dinner",
        "周五下午 3 点开产品评审会",
        "明天 10 点前把新版 proposal 发我一下",
        "你看看这个方向是不是可以继续推进？",
    ] {
        let event = MessageEvent {
            event_id: "case".to_owned(),
            text: text.to_owned(),
        };

        assert!(
            deterministic_prefilter(&event).is_none(),
            "message should reach weak model, but was prefiltered: {text}"
        );
    }
}

#[tokio::test]
async fn always_after_weak_policy_captures_without_reading_message_text() {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let trace = TraceCollector::default();
    let runtime = Runtime::new(
        CountingWeak::new(WeakIntentGuess {
            kind: IntentKind::CreateCalendarEvent,
            title: Some("some event".to_owned()),
            datetime_hint: Some("tomorrow".to_owned()),
            confidence: 0.99,
            rationale: "high confidence".to_owned(),
        }),
        RecordingStrong {
            decision: Ok(ThinkDecision::Value(ResolvedIntent {
                kind: IntentKind::CreateCalendarEvent,
                title: "confirmed by strong".to_owned(),
                datetime_hint: Some("tomorrow".to_owned()),
                confidence: 0.99,
                source: DecisionSource::StrongThink,
            })),
            calls: calls.clone(),
        },
        0.75,
        CapturePolicy::AlwaysAfterWeak,
        trace.clone(),
    );

    let draft = runtime
        .run_one(MessageEvent {
            event_id: "m1".to_owned(),
            text: "arbitrary text with no special marker".to_owned(),
        })
        .await
        .unwrap();

    assert_eq!(draft.source, DecisionSource::StrongThink);
    assert_eq!(draft.kind, IntentKind::CreateCalendarEvent);
    assert_eq!(calls.lock().unwrap().len(), 1);

    let events: Vec<_> = trace.events().into_iter().map(|item| item.event).collect();
    assert!(events.contains(&"capture_continuation".to_owned()));
    assert!(events.contains(&"strong_think".to_owned()));
    assert!(events.contains(&"resume_continuation".to_owned()));
}

#[tokio::test]
async fn weak_only_path_does_not_call_strong() {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let runtime = Runtime::new(
        StaticWeak {
            result: Ok(WeakIntentGuess {
                kind: IntentKind::CreateCalendarEvent,
                title: Some("产品评审会".to_owned()),
                datetime_hint: Some("周五下午 3 点".to_owned()),
                confidence: 0.82,
                rationale: "obvious meeting".to_owned(),
            }),
        },
        RecordingStrong {
            decision: Err("should not be called".to_owned()),
            calls: calls.clone(),
        },
        0.75,
        CapturePolicy::ConfidenceOnly,
        TraceCollector::default(),
    );

    let draft = runtime
        .run_one(MessageEvent {
            event_id: "m2".to_owned(),
            text: "周五下午 3 点开产品评审会".to_owned(),
        })
        .await
        .unwrap();

    assert_eq!(draft.source, DecisionSource::WeakModel);
    assert_eq!(draft.kind, IntentKind::CreateCalendarEvent);
    assert!(calls.lock().unwrap().is_empty());
}

#[tokio::test]
async fn every_non_empty_message_reaches_weak_model_first() {
    let weak = CountingWeak::new(WeakIntentGuess {
        kind: IntentKind::Ignore,
        title: Some("model decided ignore".to_owned()),
        datetime_hint: None,
        confidence: 0.90,
        rationale: "semantic classification by weak model".to_owned(),
    });
    let calls = Arc::new(Mutex::new(Vec::new()));
    let runtime = Runtime::new(
        weak.clone(),
        RecordingStrong {
            decision: Err("should not be called".to_owned()),
            calls,
        },
        0.75,
        CapturePolicy::ConfidenceOnly,
        TraceCollector::default(),
    );

    let messages = [
        "验证码 839201，五分钟内有效",
        "verification code UX review",
        "unsubscribe flow review",
        "周五下午 3 点开产品评审会",
    ];

    for (index, text) in messages.iter().enumerate() {
        let _ = runtime
            .run_one(MessageEvent {
                event_id: format!("m{index}"),
                text: text.to_string(),
            })
            .await
            .unwrap();
    }

    assert_eq!(weak.call_count(), messages.len());
}

#[tokio::test]
async fn out_of_range_weak_confidence_captures_continuation() {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let runtime = Runtime::new(
        StaticWeak {
            result: Ok(WeakIntentGuess {
                kind: IntentKind::CreateTask,
                title: Some("Send proposal".to_owned()),
                datetime_hint: Some("tomorrow".to_owned()),
                confidence: 75.0,
                rationale: "bad percent value".to_owned(),
            }),
        },
        RecordingStrong {
            decision: Ok(ThinkDecision::Value(ResolvedIntent {
                kind: IntentKind::CreateTask,
                title: "Send proposal".to_owned(),
                datetime_hint: Some("tomorrow".to_owned()),
                confidence: 0.9,
                source: DecisionSource::StrongThink,
            })),
            calls: calls.clone(),
        },
        0.75,
        CapturePolicy::ConfidenceOnly,
        TraceCollector::default(),
    );

    let draft = runtime
        .run_one(MessageEvent {
            event_id: "m8".to_owned(),
            text: "Please send the proposal tomorrow".to_owned(),
        })
        .await
        .unwrap();

    assert_eq!(draft.source, DecisionSource::StrongThink);
    let frames = calls.lock().unwrap();
    assert_eq!(frames.len(), 1);
    assert_eq!(frames[0].frame.reason, "low_confidence");
}

#[tokio::test]
async fn weak_error_becomes_think_effect() {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let runtime = Runtime::new(
        StaticWeak {
            result: Err("network timeout".to_owned()),
        },
        RecordingStrong {
            decision: Ok(ThinkDecision::Value(ResolvedIntent {
                kind: IntentKind::DraftReply,
                title: "回复对方：可以继续推进".to_owned(),
                datetime_hint: None,
                confidence: 0.8,
                source: DecisionSource::StrongThink,
            })),
            calls: calls.clone(),
        },
        0.75,
        CapturePolicy::ConfidenceOnly,
        TraceCollector::default(),
    );

    let draft = runtime
        .run_one(MessageEvent {
            event_id: "m4".to_owned(),
            text: "你看看这个方向是不是可以继续推进？".to_owned(),
        })
        .await
        .unwrap();

    assert_eq!(draft.source, DecisionSource::StrongThink);
    let frames = calls.lock().unwrap();
    assert_eq!(frames.len(), 1);
    assert_eq!(
        frames[0].frame.weak_error.as_deref(),
        Some("network timeout")
    );
}

#[tokio::test]
async fn invalid_weak_contract_captures_continuation() {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let runtime = Runtime::new(
        StaticWeak {
            result: Ok(WeakIntentGuess {
                kind: IntentKind::CreateTask,
                title: Some("   ".to_owned()),
                datetime_hint: Some("tomorrow".to_owned()),
                confidence: 0.96,
                rationale: "looks like a request".to_owned(),
            }),
        },
        RecordingStrong {
            decision: Ok(ThinkDecision::Value(ResolvedIntent {
                kind: IntentKind::CreateTask,
                title: "Send proposal".to_owned(),
                datetime_hint: Some("tomorrow".to_owned()),
                confidence: 0.9,
                source: DecisionSource::StrongThink,
            })),
            calls: calls.clone(),
        },
        0.75,
        CapturePolicy::ConfidenceOnly,
        TraceCollector::default(),
    );

    let draft = runtime
        .run_one(MessageEvent {
            event_id: "m6".to_owned(),
            text: "Please send it tomorrow".to_owned(),
        })
        .await
        .unwrap();

    assert_eq!(draft.source, DecisionSource::StrongThink);
    let frames = calls.lock().unwrap();
    assert_eq!(frames.len(), 1);
    assert_eq!(frames[0].frame.reason, "invalid_weak_contract");
}

#[tokio::test]
async fn strong_error_returns_user_fallback() {
    let runtime = Runtime::new(
        StaticWeak {
            result: Ok(WeakIntentGuess {
                kind: IntentKind::NeedStrongThink,
                title: None,
                datetime_hint: None,
                confidence: 0.2,
                rationale: "uncertain".to_owned(),
            }),
        },
        RecordingStrong {
            decision: Err("service unavailable".to_owned()),
            calls: Arc::new(Mutex::new(Vec::new())),
        },
        0.75,
        CapturePolicy::ConfidenceOnly,
        TraceCollector::default(),
    );

    let draft = runtime
        .run_one(MessageEvent {
            event_id: "m5".to_owned(),
            text: "can you judge this?".to_owned(),
        })
        .await
        .unwrap();

    assert_eq!(draft.source, DecisionSource::UserFallback);
    assert_eq!(draft.kind, IntentKind::DraftReply);
    assert!(draft.title.contains("Manual review required"));
}
