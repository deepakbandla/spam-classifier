import re
from html import unescape


def html_to_plain_text(html):

    text = re.sub('<head.*?>.*?</head>', '', html,
                  flags=re.M | re.S | re.I)

    text = re.sub('<a\\s.*?>', ' HYPERLINK ', text,
                  flags=re.M | re.S | re.I)

    text = re.sub('<.*?>', '', text,
                  flags=re.M | re.S)

    text = re.sub(r'(\\s*\\n)+', '\\n', text,
                  flags=re.M | re.S)

    return unescape(text)


def email_to_text(email):
    """Convert an email message to plain text.

    If the input is already a string (e.g., raw email body from Streamlit),
    return it unchanged.
    """

    if isinstance(email, bytes):
        try:
            return email.decode("utf-8", errors="ignore")
        except Exception:
            return str(email)

    if isinstance(email, str):
        return email

    html = None

    for part in email.walk():

        ctype = part.get_content_type()

        if ctype not in ("text/plain", "text/html"):
            continue

        try:
            content = part.get_content()

        except:
            content = str(part.get_payload())

        if ctype == "text/plain":
            return content

        else:
            html = content

    if html:
        return html_to_plain_text(html)