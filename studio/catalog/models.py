from dataclasses import dataclass

STATUS_ORDER = ["discovered","downloaded","stitched","detected","scened",
                "visioned","grouped","beated","scripted","voiced","planned"]

# STATUS_ORDER drives the PIPELINE (next_status walks it, so 'planned' must
# stay its last rung). Progress DISPLAY needs one more: render_segment sets
# chapter.status='rendered', which is above 'planned' but not a pipeline
# stage. Rank by this, advance by STATUS_ORDER.
STATUS_RANK = STATUS_ORDER + ["rendered"]

def next_status(s: str) -> str | None:
    i = STATUS_ORDER.index(s)
    return STATUS_ORDER[i+1] if i+1 < len(STATUS_ORDER) else None

def fail_status(stage: str) -> str:
    return f"{stage}_failed"

@dataclass
class Series:
    id: int | None
    source: str
    series_url: str
    slug: str
    title: str
    added_at: str
    last_checked: str | None = None
    poll_priority: int = 100
    niche_primary: str | None = None
    niche_secondary: str | None = None
    genres: str | None = None
    synopsis: str | None = None

@dataclass
class Chapter:
    id: int | None
    series_id: int
    number: float
    label: str
    url: str
    status: str = "discovered"
    ep_dir: str | None = None
    error: str | None = None
    updated_at: str = ""
