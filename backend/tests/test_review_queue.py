import pytest

from claimdesk.models import ReviewDecision
from claimdesk.review_queue import QueueError, ReviewQueue


@pytest.fixture
def queue():
    queue = ReviewQueue()
    yield queue
    queue.close()


def submit(queue, components, sample, name):
    text = sample(name)
    return queue.submit(text, components.assessor.assess(text))


def test_submitted_claims_wait_for_review(queue, components, sample):
    claim = submit(queue, components, sample, "physio_M-1001")
    assert claim.id == 1
    assert claim.review_status == "pending"
    assert claim.assessment.status == "ready_for_review"
    assert claim.reviewer is None
    assert queue.get(claim.id) == claim


def test_list_is_newest_first_and_filters_by_status(queue, components, sample):
    first = submit(queue, components, sample, "physio_M-1001")
    second = submit(queue, components, sample, "vision_late_M-1003")
    queue.review(first.id, ReviewDecision(decision="approved", reviewer="J. Okafor"))
    assert [c.id for c in queue.list()] == [second.id, first.id]
    assert [c.id for c in queue.list("pending")] == [second.id]
    assert queue.counts() == {"approved": 1, "pending": 1}


def test_review_records_the_examiner(queue, components, sample):
    claim = submit(queue, components, sample, "vision_late_M-1003")
    decision = ReviewDecision(decision="denied", reviewer="J. Okafor", note="Past deadline.")
    reviewed = queue.review(claim.id, decision)
    assert reviewed.review_status == "denied"
    assert reviewed.reviewer == "J. Okafor"
    assert reviewed.note == "Past deadline."
    assert reviewed.reviewed_at is not None


def test_a_claim_is_decided_once(queue, components, sample):
    claim = submit(queue, components, sample, "physio_M-1001")
    queue.review(claim.id, ReviewDecision(decision="approved", reviewer="J. Okafor"))
    with pytest.raises(QueueError, match="already reviewed by J. Okafor"):
        queue.review(claim.id, ReviewDecision(decision="denied", reviewer="Someone else"))


def test_unknown_claim(queue):
    assert queue.get(42) is None
    with pytest.raises(QueueError, match="not found"):
        queue.review(42, ReviewDecision(decision="approved", reviewer="J. Okafor"))


def test_queue_survives_a_restart(tmp_path, components, sample):
    path = str(tmp_path / "claims.db")
    first = ReviewQueue(path)
    claim = submit(first, components, sample, "physio_M-1001")
    first.close()
    second = ReviewQueue(path)
    assert second.get(claim.id) == claim
    second.close()
