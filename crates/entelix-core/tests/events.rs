//! `EventBus` tests.

#![allow(clippy::unwrap_used)]

use entelix_core::events::{Event, EventBus};
use entelix_core::ir::ModelWarning;

#[tokio::test]
async fn publish_with_no_subscribers_returns_zero() {
    let bus = EventBus::new(8);
    let n = bus.publish(Event::ToolDispatched {
        name: "calc".into(),
    });
    assert_eq!(n, 0);
}

#[tokio::test]
async fn subscribers_receive_published_events() {
    let bus = EventBus::new(8);
    let mut rx1 = bus.subscribe();
    let mut rx2 = bus.subscribe();
    assert_eq!(bus.receiver_count(), 2);

    let n = bus.publish(Event::ToolDispatched {
        name: "calc".into(),
    });
    assert_eq!(n, 2);

    let received = rx1.recv().await.unwrap();
    assert_eq!(
        received,
        Event::ToolDispatched {
            name: "calc".into()
        }
    );
    let received = rx2.recv().await.unwrap();
    assert_eq!(
        received,
        Event::ToolDispatched {
            name: "calc".into()
        }
    );
}

#[tokio::test]
async fn warning_event_carries_lossy_encode() {
    let bus = EventBus::new(8);
    let mut rx = bus.subscribe();

    let warning = ModelWarning::LossyEncode {
        field: "temperature".into(),
        detail: "codec ignored".into(),
    };
    bus.publish(Event::Warning(warning.clone()));
    assert_eq!(rx.recv().await.unwrap(), Event::Warning(warning));
}

#[tokio::test]
async fn cloned_bus_shares_channel() {
    let bus_a = EventBus::new(8);
    let bus_b = bus_a.clone();
    let mut rx = bus_b.subscribe();

    bus_a.publish(Event::ToolCompleted {
        name: "calc".into(),
        error: false,
    });
    assert_eq!(
        rx.recv().await.unwrap(),
        Event::ToolCompleted {
            name: "calc".into(),
            error: false,
        }
    );
}

#[tokio::test]
async fn subscribers_only_see_events_after_subscribe() {
    let bus = EventBus::new(8);
    bus.publish(Event::RequestStarted {
        codec: "anthropic-messages".into(),
        transport: "direct".into(),
        model: "claude-opus-4-7".into(),
    });
    let mut late = bus.subscribe();
    bus.publish(Event::RequestCompleted {
        model: "claude-opus-4-7".into(),
        input_tokens: 10,
        output_tokens: 5,
    });
    let event = late.recv().await.unwrap();
    assert!(matches!(event, Event::RequestCompleted { .. }));
}

#[tokio::test]
async fn event_serde_roundtrip() {
    let event = Event::Warning(ModelWarning::UnknownStopReason {
        raw: "future_filter".into(),
    });
    let json = serde_json::to_string(&event).unwrap();
    let back: Event = serde_json::from_str(&json).unwrap();
    assert_eq!(event, back);
}
