"""Store AI Hub token locally without placing it in history, argv, or the repo."""
import configparser
import getpass
import os
from pathlib import Path


def main():
    path = Path.home() / ".qai_hub" / "client.ini"
    config = configparser.ConfigParser()
    if path.exists():
        config.read(path)
    if not config.has_section("api"):
        config.add_section("api")
    token = getpass.getpass("AI Hub API token (hidden): ").strip()
    if not token:
        raise SystemExit("No token supplied; nothing changed")
    config.set("api", "api_token", token)
    config.set("api", "api_url", "https://workbench.aihub.qualcomm.com")
    config.set("api", "web_url", "https://workbench.aihub.qualcomm.com")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    # Restrict permissions before writing the secret, including on existing files.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    if os.name != "nt":
        os.fchmod(fd, 0o600)
    with os.fdopen(fd, "w") as stream:
        config.write(stream)
    print("Saved local AI Hub configuration. Next: accelerator-bench hub devices")


if __name__ == "__main__":
    main()
