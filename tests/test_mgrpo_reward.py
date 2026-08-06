import unittest

from mgrpo_reward import (
    compute_reward_with_components,
    extract_pred_emotions,
    reward_emotion_overlap,
    reward_format,
    reward_label_count,
    reward_no_reasoning,
)


class MGRPORewardTest(unittest.TestCase):
    def test_canonicalizes_common_synonyms(self):
        response = "The character's emotional state is joy, sadness."
        self.assertEqual(extract_pred_emotions(response), {"happy", "sad"})

    def test_exact_format_scores_one(self):
        response = "The character's emotional state is happy, relieved."
        self.assertEqual(reward_format(response), 1.0)

    def test_reasoning_is_penalized(self):
        response = (
            "The character's emotional state is sad. "
            "Based on the tone of voice, the speaker appears upset."
        )
        self.assertEqual(reward_no_reasoning(response), 0.0)

    def test_overlap_uses_f1(self):
        response = "The character's emotional state is happy, sad."
        self.assertAlmostEqual(reward_emotion_overlap(response, "joy"), 2 / 3)

    def test_label_count_rejects_empty_output(self):
        self.assertEqual(reward_label_count(""), 0.0)

    def test_component_total_is_finite(self):
        total, components = compute_reward_with_components(
            "The character's emotional state is worried.",
            "worry",
        )
        self.assertGreater(total, 0)
        self.assertEqual(components["emotion_f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
