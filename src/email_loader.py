import email
import email.policy
from pathlib import Path

def load_email(filepath):

    with open(filepath, "rb") as f:

        return email.parser.BytesParser(
            policy=email.policy.default
        ).parse(f)

def load_dataset():
    spam_path = Path("data/raw")
    ham_dir = spam_path / "easy_ham"
    spam_dir = spam_path / "spam"
    ham_filenames = [
        f for f in sorted(ham_dir.iterdir())
        if len(f.name) > 20
    ]
    spam_filenames = [
        f for f in sorted(spam_dir.iterdir())
        if len(f.name) > 20
    ]
    ham_emails = [load_email(f) for f in ham_filenames]
    spam_emails = [load_email(f) for f in spam_filenames]

    return ham_emails, spam_emails