from rtst.context import ConversationContext


def test_keeps_only_recent_segments():
    ctx = ConversationContext(max_segments=3)
    for i in range(5):
        ctx.add(f"line {i}")
    assert ctx.recent() == ["line 2", "line 3", "line 4"]
    assert len(ctx) == 3


def test_ignores_blank_input():
    ctx = ConversationContext(max_segments=3)
    ctx.add("   ")
    ctx.add("")
    assert len(ctx) == 0


def test_zero_segments_disables_context():
    ctx = ConversationContext(max_segments=0)
    ctx.add("anything")
    assert ctx.recent() == []


def test_age_based_trimming():
    ctx = ConversationContext(max_segments=10, max_age_seconds=10.0)
    ctx.add("old", now=0.0)
    ctx.add("newer", now=8.0)
    # At t=12 the first entry (t=0) is older than 10s and should be dropped.
    assert ctx.recent(now=12.0) == ["newer"]


def test_as_prompt_joins_recent():
    ctx = ConversationContext(max_segments=3)
    ctx.add("a", now=1.0)
    ctx.add("b", now=2.0)
    assert ctx.as_prompt(now=2.5) == "a b"


def test_clear():
    ctx = ConversationContext(max_segments=3)
    ctx.add("x")
    ctx.clear()
    assert len(ctx) == 0
