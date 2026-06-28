import re


ROMAN_URDU_FIXES = {
    "balnce": "balance",
    "balans": "balance",
    "rechrge": "recharge",
    "rechage": "recharge",
    "pakage": "package",
    "pckg": "package",
    "netwrk": "network",
    "internat": "internet",
    "jazzcash": "jazzcash",
    "complnt": "complaint",
    "activte": "activate",
    "actvate": "activate",
}


def normalize_text(text):
  """
  Normalize user query or FAQ text.

  Steps:
  - lowercase
  - fix common Roman Urdu spelling mistakes
  - remove extra spaces
  """

  text = str(text).lower().strip()

  for wrong, correct in ROMAN_URDU_FIXES.items():
    text = text.replace(wrong, correct)

  text = re.sub(r"\s+", " ", text)

  return text