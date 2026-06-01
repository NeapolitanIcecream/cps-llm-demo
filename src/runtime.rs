use anyhow::Result;
use serde_json::json;

use crate::domain::{ActionDraft, DecisionSource, IntentKind, MessageEvent, WeakIntentGuess};
use crate::effects::{
    Continuation, EffectFrame, EffectKind, ExpectedType, ThinkDecision, ThinkFrame,
};
use crate::models::{StrongModel, WeakModel};
use crate::trace::TraceCollector;

#[derive(Debug, Clone, PartialEq)]
pub enum Step {
    Done(ActionDraft),
    Effect(Box<EffectFrame>),
}

pub struct Runtime<W, S> {
    pub weak: W,
    pub strong: S,
    pub threshold: f32,
    pub trace: TraceCollector,
}

impl<W, S> Runtime<W, S>
where
    W: WeakModel,
    S: StrongModel,
{
    pub fn new(weak: W, strong: S, threshold: f32, trace: TraceCollector) -> Self {
        Self {
            weak,
            strong,
            threshold,
            trace,
        }
    }

    pub async fn run_one(&self, event: MessageEvent) -> Result<ActionDraft> {
        let mut step = self.start(event).await?;

        loop {
            match step {
                Step::Done(output) => {
                    self.trace
                        .emit("done", &output.event_id, json!({ "source": output.source }));
                    return Ok(output);
                }
                Step::Effect(frame) => {
                    let frame = *frame;
                    self.trace.emit(
                        "strong_think",
                        &frame.frame.event.event_id,
                        json!({ "effect_id": frame.effect_id }),
                    );

                    let decision = match self.strong.think(&frame).await {
                        Ok(decision) => decision,
                        Err(err) => {
                            self.trace.emit(
                                "strong_error",
                                &frame.frame.event.event_id,
                                json!({ "error": err.to_string() }),
                            );
                            ThinkDecision::Abort {
                                reason: format!("strong_model_error: {err}"),
                            }
                        }
                    };

                    step = self.resume(frame.continuation, decision)?;
                }
            }
        }
    }

    async fn start(&self, event: MessageEvent) -> Result<Step> {
        self.trace.emit("start", &event.event_id, json!({}));

        if let Some(output) = deterministic_prefilter(&event) {
            self.trace.emit(
                "deterministic_done",
                &event.event_id,
                json!({ "source": output.source, "kind": output.kind }),
            );
            return Ok(Step::Done(output));
        }

        match self.weak.classify_message(&event).await {
            Ok(weak_guess) => {
                self.trace.emit(
                    "weak_guess",
                    &event.event_id,
                    json!({ "kind": weak_guess.kind, "confidence": weak_guess.confidence }),
                );

                if let Some(reason) = self.capture_reason(&event, &weak_guess) {
                    self.trace.emit(
                        "capture_continuation",
                        &event.event_id,
                        json!({ "cont": "after_classify_message", "reason": reason }),
                    );
                    Ok(Step::Effect(Box::new(make_effect_frame(
                        event,
                        Some(weak_guess),
                        None,
                        reason,
                    ))))
                } else {
                    Ok(Step::Done(action_from_weak(event, weak_guess)))
                }
            }
            Err(err) => {
                self.trace.emit(
                    "weak_error",
                    &event.event_id,
                    json!({ "error": err.to_string() }),
                );
                self.trace.emit(
                    "capture_continuation",
                    &event.event_id,
                    json!({ "cont": "after_classify_message", "reason": "weak_model_error" }),
                );
                Ok(Step::Effect(Box::new(make_effect_frame(
                    event,
                    None,
                    Some(err.to_string()),
                    "weak_model_error",
                ))))
            }
        }
    }

    fn capture_reason(
        &self,
        event: &MessageEvent,
        guess: &WeakIntentGuess,
    ) -> Option<&'static str> {
        if !guess.confidence.is_finite() || guess.confidence < self.threshold {
            return Some("low_confidence");
        }

        if guess.kind == IntentKind::NeedStrongThink {
            return Some("need_strong_think");
        }

        if requires_title(&guess.kind)
            && guess
                .title
                .as_ref()
                .map(|title| title.trim().is_empty())
                .unwrap_or(true)
        {
            return Some("invalid_weak_contract");
        }

        let text = event.text.to_lowercase();
        let force_think_markers = ["proposal", "方向", "推进"];
        if force_think_markers
            .iter()
            .any(|marker| text.contains(marker))
        {
            return Some("runtime_guard");
        }

        None
    }

    fn resume(&self, cont: Continuation, decision: ThinkDecision) -> Result<Step> {
        match (cont, decision) {
            (Continuation::AfterClassifyMessage { event, .. }, ThinkDecision::Value(intent)) => {
                self.trace.emit(
                    "resume_continuation",
                    &event.event_id,
                    json!({ "cont": "after_classify_message" }),
                );
                Ok(Step::Done(ActionDraft {
                    event_id: event.event_id,
                    kind: intent.kind,
                    title: intent.title,
                    datetime_hint: intent.datetime_hint,
                    source: DecisionSource::StrongThink,
                }))
            }
            (Continuation::AfterClassifyMessage { event, .. }, ThinkDecision::Abort { reason }) => {
                self.trace.emit(
                    "resume_continuation",
                    &event.event_id,
                    json!({ "cont": "after_classify_message" }),
                );
                Ok(Step::Done(ActionDraft {
                    event_id: event.event_id,
                    kind: IntentKind::DraftReply,
                    title: format!("Manual review required: {reason}"),
                    datetime_hint: None,
                    source: DecisionSource::UserFallback,
                }))
            }
        }
    }
}

pub fn deterministic_prefilter(event: &MessageEvent) -> Option<ActionDraft> {
    let text = event.text.trim().to_lowercase();

    if text.is_empty() {
        return Some(ignore(event, "empty message"));
    }

    if text.contains("验证码") || has_otp_marker(&text) || text.contains("verification code") {
        return Some(ignore(event, "verification code"));
    }

    if text.contains("unsubscribe") {
        return Some(ignore(event, "unsubscribe notice"));
    }

    None
}

fn has_otp_marker(text: &str) -> bool {
    text.split(|ch: char| !ch.is_ascii_alphanumeric())
        .any(|token| token == "otp" || has_code_like_otp_token(token))
}

fn has_code_like_otp_token(token: &str) -> bool {
    token.strip_prefix("otp").is_some_and(is_code_like_digits)
        || token.strip_suffix("otp").is_some_and(is_code_like_digits)
}

fn is_code_like_digits(value: &str) -> bool {
    value.len() >= 4 && value.chars().all(|ch| ch.is_ascii_digit())
}

pub(crate) fn make_effect_frame(
    event: MessageEvent,
    weak_guess: Option<WeakIntentGuess>,
    weak_error: Option<String>,
    reason: impl Into<String>,
) -> EffectFrame {
    let reason = reason.into();
    EffectFrame {
        effect_id: uuid::Uuid::new_v4().to_string(),
        effect: EffectKind::Think,
        continuation: Continuation::AfterClassifyMessage {
            event: event.clone(),
            weak_guess: weak_guess.clone(),
            weak_error: weak_error.clone(),
        },
        frame: ThinkFrame {
            reason,
            expected: ExpectedType::ResolvedIntent,
            event,
            weak_guess,
            weak_error,
            allowed_decisions: vec![
                "ignore".to_owned(),
                "create_task".to_owned(),
                "create_calendar_event".to_owned(),
                "draft_reply".to_owned(),
                "abort".to_owned(),
            ],
        },
    }
}

fn action_from_weak(event: MessageEvent, guess: WeakIntentGuess) -> ActionDraft {
    let title = guess
        .title
        .unwrap_or_else(|| default_title(&guess.kind).to_owned());
    ActionDraft {
        event_id: event.event_id,
        kind: guess.kind,
        title,
        datetime_hint: guess.datetime_hint,
        source: DecisionSource::WeakModel,
    }
}

fn ignore(event: &MessageEvent, title: &str) -> ActionDraft {
    ActionDraft {
        event_id: event.event_id.clone(),
        kind: IntentKind::Ignore,
        title: title.to_owned(),
        datetime_hint: None,
        source: DecisionSource::DeterministicCode,
    }
}

fn requires_title(kind: &IntentKind) -> bool {
    matches!(
        kind,
        IntentKind::CreateTask | IntentKind::CreateCalendarEvent | IntentKind::DraftReply
    )
}

fn default_title(kind: &IntentKind) -> &'static str {
    match kind {
        IntentKind::Ignore => "Ignored message",
        IntentKind::CreateTask => "Create task",
        IntentKind::CreateCalendarEvent => "Create calendar event",
        IntentKind::DraftReply => "Draft reply",
        IntentKind::NeedStrongThink => "Needs deeper handling",
    }
}
