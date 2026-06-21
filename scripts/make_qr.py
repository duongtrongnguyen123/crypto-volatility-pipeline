"""Generate a QR code PNG for the live demo URL.

Auto-detects the current cloudflared quick-tunnel URL from
``data/live/cloudflared.log`` (the URL changes on every tunnel restart), or
takes an explicit URL as the first argument.

Usage:
    python scripts/make_qr.py                      # auto-detect tunnel URL
    python scripts/make_qr.py https://my.url       # explicit URL
-> writes reports/figures/demo_qr.png
"""
import os
import re
import sys

import qrcode

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = os.path.join(ROOT, "data/live/cloudflared.log")
OUT = os.path.join(ROOT, "reports/figures/demo_qr.png")


def detect_url() -> str | None:
    if not os.path.exists(LOG):
        return None
    urls = re.findall(r"https://[a-z0-9-]+\.trycloudflare\.com", open(LOG).read())
    return urls[-1] if urls else None  # last = most recent tunnel


def main() -> None:
    url = sys.argv[1] if len(sys.argv) > 1 else detect_url()
    if not url:
        sys.exit("No URL given and none found in cloudflared.log — pass one explicitly.")
    qr = qrcode.QRCode(box_size=12, border=2,
                       error_correction=qrcode.constants.ERROR_CORRECT_M)
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="#16a34a", back_color="white")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    img.save(OUT)
    print(f"QR for {url}\n-> {OUT}")


if __name__ == "__main__":
    main()
