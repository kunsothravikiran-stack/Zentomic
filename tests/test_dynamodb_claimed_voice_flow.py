"""Synthetic durable-claim/voice composition, not live DynamoDB or Twilio."""

import unittest
from itertools import product
from unittest.mock import Mock, patch

from tests.test_callback_claim_store import ACCOUNT, CALL, URL, callback
from tests.test_dynamodb_callback_claim_store import ClaimClient
from zentomic.callback_claim_store import CallbackReplayError
from zentomic.confirmation_event import resolve_claimed_confirmation_event
from zentomic.dial import DialDecision
from zentomic.dial_event import resolve_claimed_dial_result_event
from zentomic.dynamodb_callback_claim_store import DynamoDBCallbackClaimStore
from zentomic.gather import GatherDecision
from zentomic.gather_event import resolve_claimed_gather_event
from zentomic.speech import SpeechDecision
from zentomic.speech_event import resolve_claimed_speech_event


CASES = (
    (resolve_claimed_gather_event, "zentomic.gather_event._resolve_validated_gather",
     {"routes": {"1": "demo-support"}, "attempts": 0},
     {"Digits": "1"}, GatherDecision("route", "demo-support", 1)),
    (resolve_claimed_speech_event, "zentomic.speech_event._resolve_validated_speech",
     {"attempts": 0}, {"SpeechResult": "synthetic support request"},
     SpeechDecision("classify", "synthetic support request", None, 1)),
    (resolve_claimed_confirmation_event,
     "zentomic.confirmation_event._resolve_validated_intent_confirmation",
     {"pending_intent": "support", "routes": {"support": "demo-support"},
      "attempts": 0}, {"Digits": "1"}, GatherDecision("route", "demo-support", 1)),
    (resolve_claimed_dial_result_event,
     "zentomic.dial_event._resolve_validated_dial_result",
     {"expected_dial_call_sid": "synthetic-child"},
     {"DialCallSid": "synthetic-child", "DialCallStatus": "busy"},
     DialDecision("fallback", "demo-reception")),
)


class DynamoDBClaimedVoiceFlowTests(unittest.TestCase):
    def invoke(self, case, event, client, *, validator=None, admission=None):
        resolve, _, config, _, _ = case
        # Fresh adapters share only the modeled durable table, not local claims.
        return resolve(
            event, **config, public_url=URL,
            validator=validator if validator is not None else Mock(return_value=True),
            expected_account_sid=ACCOUNT, expected_call_sid=CALL,
            workspace_id="trusted-workspace", step_id=resolve.__name__,
            claimer=DynamoDBCallbackClaimStore(
                client=client, table_name="demo-claims", before_request=admission,
            ), fallback_target="demo-reception",
        )

    def variants(self):
        return product(CASES, ("1.0", "2.0"), (False, True))

    def event(self, case, version, encoded, **overrides):
        return callback(
            {**case[3], "workspace_id": "untrusted", "step_id": "untrusted",
             "fallback_target": "untrusted", **overrides},
            version=version, encoded=encoded,
        )

    def assert_one_trusted_row(self, case, client):
        self.assertEqual(set(client.rows), {
            ("trusted-workspace", f"{len(CALL.encode())}:{CALL}{case[0].__name__}"),
        })

    def test_fresh_adapters_deny_changed_replays_before_each_voice_policy(self):
        for case, version, encoded in self.variants():
            with self.subTest(helper=case[0].__name__, version=version, encoded=encoded):
                client = ClaimClient()
                self.assertEqual(
                    self.invoke(case, self.event(case, version, encoded), client), case[4],
                )
                # Even newly signed input cannot reopen a consumed trusted step.
                changed = self.event(case, version, encoded, Digits="2",
                                     SpeechResult="changed synthetic request",
                                     DialCallStatus="completed")
                with patch(case[1]) as policy, self.assertRaises(CallbackReplayError):
                    self.invoke(case, changed, client)
                policy.assert_not_called()
                self.assertEqual(len(client.requests), 2)
                self.assert_one_trusted_row(case, client)

    def test_lost_claim_acknowledgment_never_releases_voice_input_or_reopens_step(self):
        for case, version, encoded in self.variants():
            for failure in (TimeoutError("synthetic lost acknowledgment"), None, []):
                with self.subTest(helper=case[0].__name__, version=version,
                                  encoded=encoded, failure=type(failure).__name__):
                    client = ClaimClient()
                    put = client.put_item

                    def uncertain(**request):
                        put(**request)
                        if isinstance(failure, Exception):
                            raise failure
                        return failure

                    client.put_item = uncertain
                    event = self.event(case, version, encoded)
                    with patch(case[1]) as policy:
                        with self.assertRaises((TimeoutError, ValueError)):
                            self.invoke(case, event, client)
                        client.put_item = put
                        with self.assertRaises(CallbackReplayError):
                            self.invoke(case, event, client)
                        policy.assert_not_called()
                    self.assertEqual(len(client.requests), 2)
                    self.assert_one_trusted_row(case, client)

    def test_rejected_identity_or_signature_cannot_consume_voice_claims(self):
        for case, version, encoded in self.variants():
            rejected = [({}, False), ({"AccountSid": "wrong-account"}, True),
                        ({"CallSid": "wrong-call"}, True)]
            if "expected_dial_call_sid" in case[2]:
                rejected.append(({"DialCallSid": "stale-child"}, True))
            with self.subTest(helper=case[0].__name__, version=version, encoded=encoded):
                client, admission = ClaimClient(), Mock()
                with patch(case[1]) as policy:
                    for fields, valid in rejected:
                        with self.assertRaises(ValueError):
                            self.invoke(
                                case, self.event(case, version, encoded, **fields), client,
                                validator=Mock(return_value=valid), admission=admission,
                            )
                    policy.assert_not_called()
                admission.assert_not_called()
                self.assertEqual(client.requests, [])
                self.assertEqual(client.rows, {})
                self.assertEqual(
                    self.invoke(case, self.event(case, version, encoded), client), case[4],
                )
                self.assert_one_trusted_row(case, client)

    def test_failure_after_claim_does_not_allow_policy_execution_on_redelivery(self):
        for case, version, encoded in self.variants():
            with self.subTest(helper=case[0].__name__, version=version, encoded=encoded):
                client = ClaimClient()
                event = self.event(case, version, encoded)
                failure = RuntimeError("synthetic policy failure")
                with patch(case[1], side_effect=failure) as policy:
                    with self.assertRaises(RuntimeError) as caught:
                        self.invoke(case, event, client)
                    self.assertIs(caught.exception, failure)
                    with self.assertRaises(CallbackReplayError):
                        self.invoke(case, event, client)
                    policy.assert_called_once()
                self.assert_one_trusted_row(case, client)

    def test_denied_request_admission_leaves_voice_step_available(self):
        for case, version, encoded in self.variants():
            with self.subTest(helper=case[0].__name__, version=version, encoded=encoded):
                client = ClaimClient()
                event = self.event(case, version, encoded)
                failure = TimeoutError("synthetic exhausted operation budget")
                with patch(case[1]) as policy, self.assertRaises(TimeoutError) as caught:
                    self.invoke(case, event, client, admission=Mock(side_effect=failure))
                self.assertIs(caught.exception, failure)
                policy.assert_not_called()
                self.assertEqual(client.requests, [])
                self.assertEqual(client.rows, {})
                self.assertEqual(self.invoke(case, event, client), case[4])
                self.assert_one_trusted_row(case, client)


if __name__ == "__main__":
    unittest.main()
