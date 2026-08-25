"""The check that a generated document fetches nothing when it is opened.

Every HTML file MolCascade writes -- the cascade builder, the graph editor, the
run report -- is meant to work from a laptop with the network cable out, and to
keep working when the person opening it has no idea what it is allowed to do.
The Content-Security-Policy is what enforces that; these assertions are the
belt to its braces, and exist to catch a resource reference at the moment it is
added rather than when someone eventually opens the file offline.

The scan deliberately distinguishes two regions of the document:

*Markup, CSS and script* may contain no absolute URL at all.  There is no such
thing as a harmless one there: an ``href``, a ``src``, a CSS ``url()`` or a
``fetch`` is a network request, and the rule "no URLs here" is worth more than
any rule that tries to judge which ones are safe.

*The embedded payload* is a JSON blob inside a ``<textarea>``, and it carries
the citation for every backend the builder offers -- references that are close
to useless without their DOI.  A URL there is a string that gets rendered as
text, and the reason it can only ever be text is asserted below rather than
assumed: the builder scripts contain no ``innerHTML`` or any other sink that
turns a string into markup, so nothing in the payload can become a link, an
image or a request.  Take that sink away and this scoping would be wrong, which
is exactly why the sink check lives in the same function.
"""

from __future__ import annotations

import re

#: Anything that turns a string into live markup.  ``srcdoc`` is here because it
#: is the one attribute that smuggles a whole document past a scan for URLs.
_HTML_SINKS = re.compile(r"\b(?:innerHTML|outerHTML|insertAdjacentHTML|srcdoc)\b|document\.write")

#: Inline handlers, which the CSP would refuse anyway; catching them here names
#: the problem instead of leaving a silently dead button.
_INLINE_HANDLER = re.compile(r"\son(?:click|change|input|load|error|submit|focus|blur)=")

_PAYLOAD = re.compile(r'(<textarea\b[^>]*>)(.*?)(</textarea>)', re.DOTALL)


def strip_embedded_payloads(document: str) -> str:
    """Return the document with ``<textarea>`` contents blanked out."""

    return _PAYLOAD.sub(lambda match: match.group(1) + match.group(3), document)


def assert_offline(document: str) -> None:
    """Fail unless opening ``document`` in a browser touches no network."""

    assert not _HTML_SINKS.search(document), (
        "a generated document gained an HTML sink; embedded payload text could "
        "now become markup, and the URL scoping below is no longer sound"
    )
    assert not _INLINE_HANDLER.search(document), "inline event handler in a CSP-pinned document"

    scannable = strip_embedded_payloads(document)
    for scheme in ("http://", "https://"):
        assert scheme not in scannable, (
            f"{scheme!r} appears in the markup, style or script of a document that "
            "must load nothing remotely"
        )


__all__ = ["assert_offline", "strip_embedded_payloads"]
